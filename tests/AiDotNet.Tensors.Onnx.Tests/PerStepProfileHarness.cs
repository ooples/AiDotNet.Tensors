using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Phase 1 evidence harness #2: per-step timing for a small stacked-block
/// ONNX model that mirrors one ResNet-50 residual block and one BERT-base
/// transformer block. Complements <see cref="OpLevelPerfHarness"/> by
/// showing the REAL step sequence after optimisation passes: if our
/// fusion passes collapse ops, the step list reflects that; if an
/// unexpected step (Reshape, Transpose, Slice, etc.) shows up as
/// dominant, the op-level harness might be missing it.
///
/// <para>Uses <c>CompiledInferencePlan{T}.ProfilePerStep</c> (internal
/// diagnostic method) to time each step individually across N replays
/// after M warmup runs. Output is sorted by total time descending so the
/// top-N lines ARE the bottleneck list.</para>
///
/// <para>Skippable — run manually to gather data:
/// <c>dotnet test --filter FullyQualifiedName~PerStepProfileHarness --logger "console;verbosity=detailed"</c>.</para>
/// </summary>
public class PerStepProfileHarness
{
    private readonly ITestOutputHelper _output;
    public PerStepProfileHarness(ITestOutputHelper output) { _output = output; }

    private const int FLOAT  = OnnxTestHelpers.FLOAT;
    private const int Warmup = 5;
    private const int Iters  = 20;

    [SkippableFact]
    public void ProfileResNet50_ResidualBlock()
    {
        SkipUnlessEnabled();
        var modelBytes = BuildResNetResidualBlock();
        ProfileAndEmit("ResNet-50 bottleneck block (1×256×56×56)", modelBytes, "X", new[] { 1, 256, 56, 56 });
    }

    [SkippableFact]
    public void ProfileBertBase_TransformerBlock()
    {
        SkipUnlessEnabled();
        var modelBytes = BuildBertTransformerBlock();
        ProfileAndEmit("BERT-base transformer block (1×256×768)", modelBytes, "X", new[] { 1, 256, 768 });
    }

    private static void SkipUnlessEnabled()
    {
        // Evidence harness, skipped by default. Also hits two pre-existing
        // importer bugs in synthetic-model construction (TensorSplit +
        // ConvBnFusionPass) that real ONNX exports don't trigger — gating
        // prevents the synthetic failure from masking real regressions.
        Skip.IfNot(
            Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");
    }

    // ─── profiling core ─────────────────────────────────────────────────────

    private void ProfileAndEmit(string label, byte[] modelBytes, string inputName, int[] inputShape)
    {
        var engine = new CpuEngine();
        using var stream = new MemoryStream(modelBytes);
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);

        int total = 1; foreach (var d in inputShape) total *= d;
        var X = RandArr(0x5E0, total);
        X.AsSpan().CopyTo(result.Inputs[inputName].AsWritableSpan());

        var plan = (CompiledInferencePlan<float>)result.Plan!;
        var profile = plan.ProfilePerStep(Warmup, Iters);

        // Sort descending by time
        Array.Sort(profile, (a, b) => b.AvgMs.CompareTo(a.AvgMs));

        _output.WriteLine("");
        _output.WriteLine($"=== Per-step profile: {label} ===");
        _output.WriteLine($"Step count: {plan.StepCount}, warmup={Warmup}, iters={Iters}");
        _output.WriteLine("rank\tms/exec\topname");
        double total_ms = 0;
        int rank = 0;
        foreach (var (op, ms) in profile)
        {
            total_ms += ms;
            if (rank < 20 || ms > 0.1)  // top 20 OR any step ≥ 0.1ms
                _output.WriteLine($"{rank:D3}\t{ms:F3}\t{op}");
            rank++;
        }
        _output.WriteLine($"TOTAL per-exec: {total_ms:F2} ms across {profile.Length} steps");

        // Also aggregate by op name so Conv / MatMul etc. roll up.
        var byOp = new Dictionary<string, (int count, double totalMs)>(StringComparer.Ordinal);
        foreach (var (op, ms) in profile)
        {
            if (!byOp.TryGetValue(op, out var agg)) agg = (0, 0);
            byOp[op] = (agg.count + 1, agg.totalMs + ms);
        }
        _output.WriteLine("");
        _output.WriteLine($"--- Aggregated by op name, sorted by total ms ---");
        _output.WriteLine("op\tcount\ttotal_ms\tavg_ms");
        foreach (var kv in byOp.OrderByDescending(k => k.Value.totalMs))
        {
            _output.WriteLine($"{kv.Key}\t{kv.Value.count}\t{kv.Value.totalMs:F3}\t{kv.Value.totalMs / kv.Value.count:F3}");
        }
    }

    // ─── model builders ─────────────────────────────────────────────────────

    /// <summary>
    /// ResNet-50 bottleneck residual block:
    ///   y = x + Conv1x1_out( Relu( BN( Conv3x3( Relu( BN( Conv1x1_in(x) ) ) ) ) ) )
    ///   ReLU(y)
    /// Stage-2 dims: in=out=256, middle=64 (bottleneck), 56×56 spatial.
    /// </summary>
    private static byte[] BuildResNetResidualBlock()
    {
        const int N = 1, C = 256, H = 56, W = 56, MID = 64;
        var g = new GraphProto { Name = "rn50_block" };
        g.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { N, C, H, W }, FLOAT));
        g.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { N, C, H, W }, FLOAT));

        // Weights as initializers (random).
        AddInit(g, "Wa", new[] { MID, C, 1, 1 }, 0xA01);    // 1×1 reduce
        AddInit(g, "Wb", new[] { MID, MID, 3, 3 }, 0xA02);  // 3×3
        AddInit(g, "Wc", new[] { C, MID, 1, 1 }, 0xA03);    // 1×1 expand

        // BN params: scale/bias/mean/var per channel
        AddInit(g, "BNa_scale", new[] { MID }, 0xB01); AddInit(g, "BNa_bias", new[] { MID }, 0xB02);
        AddInitPositive(g, "BNa_mean", new[] { MID }, 0xB03); AddInitPositive(g, "BNa_var", new[] { MID }, 0xB04);
        AddInit(g, "BNb_scale", new[] { MID }, 0xB05); AddInit(g, "BNb_bias", new[] { MID }, 0xB06);
        AddInitPositive(g, "BNb_mean", new[] { MID }, 0xB07); AddInitPositive(g, "BNb_var", new[] { MID }, 0xB08);
        AddInit(g, "BNc_scale", new[] { C }, 0xB09); AddInit(g, "BNc_bias", new[] { C }, 0xB0A);
        AddInitPositive(g, "BNc_mean", new[] { C }, 0xB0B); AddInitPositive(g, "BNc_var", new[] { C }, 0xB0C);

        // Conv1x1 reduce
        AddConv(g, "conv_a", "X", "Wa", "ca", new[] { 1, 1 }, new[] { 1, 1 }, new[] { 0, 0, 0, 0 });
        AddBn  (g, "bn_a",   "ca", "BNa_scale", "BNa_bias", "BNa_mean", "BNa_var", "bna");
        AddRelu(g, "relu_a", "bna", "ra");

        // Conv3x3
        AddConv(g, "conv_b", "ra", "Wb", "cb", new[] { 3, 3 }, new[] { 1, 1 }, new[] { 1, 1, 1, 1 });
        AddBn  (g, "bn_b",   "cb", "BNb_scale", "BNb_bias", "BNb_mean", "BNb_var", "bnb");
        AddRelu(g, "relu_b", "bnb", "rb");

        // Conv1x1 expand
        AddConv(g, "conv_c", "rb", "Wc", "cc", new[] { 1, 1 }, new[] { 1, 1 }, new[] { 0, 0, 0, 0 });
        AddBn  (g, "bn_c",   "cc", "BNc_scale", "BNc_bias", "BNc_mean", "BNc_var", "bnc");

        // Residual add + final ReLU
        AddAdd (g, "add_r", "bnc", "X", "sumx");
        AddRelu(g, "relu_y", "sumx", "Y");

        return OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(g));
    }

    /// <summary>
    /// BERT-base transformer block (simplified, matches tf2onnx export shape):
    ///   ln1 = LayerNorm(x)
    ///   q   = MatMul(ln1, Wq) + bq
    ///   k   = MatMul(ln1, Wk) + bk
    ///   v   = MatMul(ln1, Wv) + bv
    ///   a   = Softmax( (q × kᵀ) / sqrt(d) )  × v
    ///   out = MatMul(a, Wo) + bo
    ///   skip1 = x + out
    ///   ln2 = LayerNorm(skip1)
    ///   ffn = MatMul(Gelu(MatMul(ln2, W1) + b1), W2) + b2
    ///   y   = skip1 + ffn
    ///
    /// Uses seq=256, hidden=768, ffn=3072 — BERT-base defaults.
    /// Attention heads collapsed to the 2-matmul formulation
    /// (head-dim reshape is absorbed into the first MatMul stage's emitted
    /// plan via our importer's Reshape/Transpose handling — which is part
    /// of what we're profiling).
    /// </summary>
    private static byte[] BuildBertTransformerBlock()
    {
        const int B = 1, S = 256, H = 768, FFN = 3072;
        var g = new GraphProto { Name = "bert_block" };
        g.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { B, S, H }, FLOAT));
        g.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { B, S, H }, FLOAT));

        // LayerNorm params (pre-attn)
        AddInit(g, "ln1_s", new[] { H }, 0x1A1);
        AddInit(g, "ln1_b", new[] { H }, 0x1A2);
        // Attention linear layers (Wq/Wk/Wv fused into the same shape; simplified: no head split)
        AddInit(g, "Wqkv", new[] { H, 3 * H }, 0x1B1);
        AddInit(g, "bqkv", new[] { 3 * H }, 0x1B2);
        AddInit(g, "Wo", new[] { H, H }, 0x1C1);
        AddInit(g, "bo", new[] { H }, 0x1C2);
        // LayerNorm post-attn
        AddInit(g, "ln2_s", new[] { H }, 0x1D1);
        AddInit(g, "ln2_b", new[] { H }, 0x1D2);
        // FFN
        AddInit(g, "W1", new[] { H, FFN }, 0x1E1);
        AddInit(g, "b1", new[] { FFN }, 0x1E2);
        AddInit(g, "W2", new[] { FFN, H }, 0x1F1);
        AddInit(g, "b2", new[] { H }, 0x1F2);

        // ln1 = LayerNorm(x)
        AddLn(g, "ln1", "X", "ln1_s", "ln1_b", "h1");

        // qkv = ln1 @ Wqkv + bqkv   (shape [B, S, 3H])
        AddNode(g, "matmul_qkv", "MatMul", new[] { "h1", "Wqkv" }, new[] { "qkv_mm" });
        AddNode(g, "add_qkv_b",  "Add",    new[] { "qkv_mm", "bqkv" }, new[] { "qkv_ab" });
        // Split into q/k/v on last axis (3 equal parts of size H)
        var splitNode = MakeNode("split_qkv", "Split", new[] { "qkv_ab" }, new[] { "q", "k", "v" });
        splitNode.Attribute.Add(new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 2 });
        g.Node.Add(splitNode);

        // scores = Softmax(q @ kᵀ × scale)
        // Simplification: use MatMul directly on [B,S,H] × [B,S,H]ᵀ. The importer will handle the transpose via the Transpose op.
        var trNode = MakeNode("transpose_k", "Transpose", new[] { "k" }, new[] { "kT" });
        trNode.Attribute.Add(new AttributeProto { Name = "perm", Type = AttributeProto.Types.AttributeType.Ints });
        trNode.Attribute[0].Ints.Add(0); trNode.Attribute[0].Ints.Add(2); trNode.Attribute[0].Ints.Add(1);
        g.Node.Add(trNode);
        AddNode(g, "matmul_scores", "MatMul", new[] { "q", "kT" }, new[] { "scores" });
        // Softmax axis=-1
        var sm = MakeNode("softmax_a", "Softmax", new[] { "scores" }, new[] { "attn_p" });
        sm.Attribute.Add(new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1 });
        g.Node.Add(sm);
        AddNode(g, "matmul_ctx", "MatMul", new[] { "attn_p", "v" }, new[] { "ctx" });

        // out = ctx @ Wo + bo
        AddNode(g, "matmul_out", "MatMul", new[] { "ctx", "Wo" }, new[] { "out_mm" });
        AddNode(g, "add_out_b",  "Add",    new[] { "out_mm", "bo" }, new[] { "attn_out" });

        // skip1 = X + attn_out
        AddNode(g, "skip1", "Add", new[] { "X", "attn_out" }, new[] { "s1" });

        // ln2 = LayerNorm(skip1)
        AddLn(g, "ln2", "s1", "ln2_s", "ln2_b", "h2");

        // ffn = Gelu(h2 @ W1 + b1) @ W2 + b2
        AddNode(g, "matmul_ffn1", "MatMul", new[] { "h2", "W1" }, new[] { "f1_mm" });
        AddNode(g, "add_ffn1_b",  "Add",    new[] { "f1_mm", "b1" }, new[] { "f1" });
        AddNode(g, "gelu",        "Gelu",   new[] { "f1" }, new[] { "g1" });
        AddNode(g, "matmul_ffn2", "MatMul", new[] { "g1", "W2" }, new[] { "f2_mm" });
        AddNode(g, "add_ffn2_b",  "Add",    new[] { "f2_mm", "b2" }, new[] { "ffn" });

        // Y = skip1 + ffn
        AddNode(g, "skip2", "Add", new[] { "s1", "ffn" }, new[] { "Y" });

        var model = OnnxTestGraphBuilder.WrapModel(g, opsetVersion: 17);  // LayerNormalization needs 17
        return OnnxTestGraphBuilder.Serialize(model);
    }

    // ─── graph-construction helpers ─────────────────────────────────────────

    private static void AddInit(GraphProto g, string name, int[] shape, int seed)
    {
        g.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer(name, shape, RandArr(seed, Prod(shape))));
    }

    private static void AddInitPositive(GraphProto g, string name, int[] shape, int seed)
    {
        var data = new float[Prod(shape)];
        var rng = new Random(seed);
        for (int i = 0; i < data.Length; i++) data[i] = 0.1f + (float)rng.NextDouble();
        g.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer(name, shape, data));
    }

    private static void AddConv(GraphProto g, string name, string x, string w, string y,
                                int[] kernel, int[] strides, int[] pads)
    {
        var n = MakeNode(name, "Conv", new[] { x, w }, new[] { y });
        var a1 = new AttributeProto { Name = "kernel_shape", Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var k in kernel) a1.Ints.Add(k);
        var a2 = new AttributeProto { Name = "strides", Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var s in strides) a2.Ints.Add(s);
        var a3 = new AttributeProto { Name = "pads", Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var p in pads) a3.Ints.Add(p);
        n.Attribute.Add(a1); n.Attribute.Add(a2); n.Attribute.Add(a3);
        g.Node.Add(n);
    }

    private static void AddBn(GraphProto g, string name, string x, string s, string b, string mean, string var, string y)
    {
        var n = MakeNode(name, "BatchNormalization", new[] { x, s, b, mean, var }, new[] { y });
        n.Attribute.Add(new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f });
        g.Node.Add(n);
    }

    private static void AddRelu(GraphProto g, string name, string x, string y)
    {
        g.Node.Add(MakeNode(name, "Relu", new[] { x }, new[] { y }));
    }

    private static void AddAdd(GraphProto g, string name, string a, string b, string y)
    {
        g.Node.Add(MakeNode(name, "Add", new[] { a, b }, new[] { y }));
    }

    private static void AddLn(GraphProto g, string name, string x, string s, string b, string y)
    {
        var n = MakeNode(name, "LayerNormalization", new[] { x, s, b }, new[] { y });
        n.Attribute.Add(new AttributeProto { Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1 });
        n.Attribute.Add(new AttributeProto { Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f });
        g.Node.Add(n);
    }

    private static void AddNode(GraphProto g, string name, string opType, string[] inputs, string[] outputs)
    {
        g.Node.Add(MakeNode(name, opType, inputs, outputs));
    }

    private static NodeProto MakeNode(string name, string opType, string[] inputs, string[] outputs)
    {
        var n = new NodeProto { Name = name, OpType = opType };
        foreach (var i in inputs) n.Input.Add(i);
        foreach (var o in outputs) n.Output.Add(o);
        return n;
    }

    // ─── scalar helpers ─────────────────────────────────────────────────────

    private static float[] RandArr(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static int Prod(int[] s) { int p = 1; foreach (var d in s) p *= d; return p; }
}
