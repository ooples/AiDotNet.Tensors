using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Variance-robust perf harness for Path-A/B/C validation. Replaces the
/// OpLevelPerfHarness's "run N iters per case, sequential" pattern — which
/// took 1+ minute, triggering thermal drift that inflated per-case variance
/// to ±15-20% across runs — with an INTERLEAVED measurement where each
/// iteration runs EVERY case once. Thermal/JIT/cache transients then affect
/// all cases equally and wash out in per-case IQR reporting.
///
/// <para>Protocol:</para>
/// <list type="number">
///   <item>Build all plans + ORT sessions upfront.</item>
///   <item>Global warmup: 50 iters of each case to stabilise JIT + CPU state.</item>
///   <item>Measurement: 200 interleaved iters — inner loop sweeps all cases,
///   outer loop repeats. Per-iter per-case time captured via
///   Stopwatch.GetTimestamp.</item>
///   <item>Report median, min, IQR (Q3-Q1), and P95 per case.</item>
/// </list>
///
/// <para>Median is the authoritative signal — it's robust to GC pauses and
/// one-off thermal spikes. IQR measures the stable spread; if IQR shrinks
/// below 5% of median, we can detect perf changes of 8-10% confidently.</para>
///
/// <para>Gated behind <c>AIDOTNET_RUN_PERF_HARNESS=1</c>.</para>
/// </summary>
public class StableOpPerfHarness
{
    private readonly ITestOutputHelper _output;
    public StableOpPerfHarness(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 30;
    private const int Iters = 100;
    private const int FLOAT = OnnxTestHelpers.FLOAT;

    [SkippableFact]
    public void MeasureAllOpsStable()
    {
        Skip.IfNot(
            System.Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable,
            "Onnx Runtime not loadable on this host.");

        var cases = BuildCases();
        _output.WriteLine($"Cases: {cases.Length}, Warmup: {Warmup}, Iters: {Iters}");
        _output.WriteLine("");

        // ── Global warmup (amortises JIT / thermal / cache cold-start) ──
        for (int w = 0; w < Warmup; w++)
        {
            for (int c = 0; c < cases.Length; c++) { cases[c].RunOurs(); cases[c].RunOrt(); }
        }

        // ── Measurement phase 1: OURS only, interleaved across cases ──
        // Separating ours from ORT in different phases prevents ORT's
        // GC / thread-pool transients from biasing our samples (previously
        // ORT showed IQR up to 50% of median, polluting ratio signals).
        // GC before each phase to start from a clean heap.
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        System.GC.Collect();

        var oursSamples = new double[cases.Length, Iters];
        double ticksToUs = 1_000_000.0 / Stopwatch.Frequency;

        for (int it = 0; it < Iters; it++)
        {
            for (int c = 0; c < cases.Length; c++)
            {
                long t0 = Stopwatch.GetTimestamp();
                cases[c].RunOurs();
                long t1 = Stopwatch.GetTimestamp();
                oursSamples[c, it] = (t1 - t0) * ticksToUs;
            }
        }

        // ── Measurement phase 2: ORT only ──
        // ORT needs re-warmup right before measurement. After ours runs
        // for ~20s of phase 1, ORT's thread pool threads have drifted off
        // cores and its internal caches are cold. Without re-warmup, the
        // first 20-30 ORT iters measure thread-wakeup latency instead of
        // steady-state compute.
        for (int w = 0; w < Warmup; w++)
            for (int c = 0; c < cases.Length; c++) cases[c].RunOrt();

        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        System.GC.Collect();

        var ortSamples = new double[cases.Length, Iters];
        for (int it = 0; it < Iters; it++)
        {
            for (int c = 0; c < cases.Length; c++)
            {
                long t0 = Stopwatch.GetTimestamp();
                cases[c].RunOrt();
                long t1 = Stopwatch.GetTimestamp();
                ortSamples[c, it] = (t1 - t0) * ticksToUs;
            }
        }

        // ── Report ────────────────────────────────────────────────────
        _output.WriteLine("=== Stable op-level perf (median µs, IQR µs, ratio) ===");
        _output.WriteLine("model\top\tshape\tours_med\tours_iqr\tort_med\tort_iqr\tratio\toccurs\tagg_ours_ms");
        double sumAggOurs = 0, sumAggOrt = 0;
        for (int c = 0; c < cases.Length; c++)
        {
            var oursStats = Summarise(oursSamples, c);
            var ortStats  = Summarise(ortSamples, c);
            double ratio = oursStats.Median / System.Math.Max(ortStats.Median, 0.001);
            double aggOurs = oursStats.Median * cases[c].Occurrences / 1000.0;
            double aggOrt  = ortStats.Median  * cases[c].Occurrences / 1000.0;
            sumAggOurs += aggOurs;
            sumAggOrt  += aggOrt;
            _output.WriteLine(
                $"{cases[c].Model}\t{cases[c].Op}\t{cases[c].Shape}\t" +
                $"{oursStats.Median:F1}\t{oursStats.IQR:F1}\t" +
                $"{ortStats.Median:F1}\t{ortStats.IQR:F1}\t" +
                $"{ratio:F2}\t{cases[c].Occurrences}\t{aggOurs:F1}");
        }
        _output.WriteLine($"TOTAL\t\t\t\t\t\t\t\t\t{sumAggOurs:F1}");
        _output.WriteLine($"Aggregate ours/ort = {sumAggOurs / System.Math.Max(sumAggOrt, 0.001):F2}x");
        _output.WriteLine($"LayerNormFloatInto calls: {AiDotNet.Tensors.Engines.CpuEngine._lnFloatIntoCalls}");
    }

    private static (double Median, double Min, double P95, double IQR) Summarise(double[,] samples, int caseIdx)
    {
        int n = samples.GetLength(1);
        var arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = samples[caseIdx, i];
        System.Array.Sort(arr);
        double median = arr[n / 2];
        double min = arr[0];
        double p95 = arr[(int)(n * 0.95)];
        double iqr = arr[n * 3 / 4] - arr[n / 4];
        return (median, min, p95, iqr);
    }

    // ─── cases ──────────────────────────────────────────────────────────────

    private static Case[] BuildCases()
    {
        var list = new System.Collections.Generic.List<Case>();

        // BERT-base hot shapes, seq=256, heads=12, head_dim=64, ffn=3072
        list.Add(MatMul("BERT", "MatMul", "[1,256,768]x[768,768] (QKV proj)",
                        a:new[]{1,256,768}, b:new[]{768,768}, occurs:48));
        list.Add(MatMul("BERT", "MatMul", "[1,256,768]x[768,3072] (FFN up)",
                        a:new[]{1,256,768}, b:new[]{768,3072}, occurs:12));
        list.Add(MatMul("BERT", "MatMul", "[1,256,3072]x[3072,768] (FFN down)",
                        a:new[]{1,256,3072}, b:new[]{3072,768}, occurs:12));
        list.Add(MatMul3D("BERT", "MatMul", "[12,256,64]x[12,64,256] (attn scores)",
                          b:12, m:256, k:64, n:256, occurs:12));
        list.Add(MatMul3D("BERT", "MatMul", "[12,256,256]x[12,256,64] (attn×V)",
                          b:12, m:256, k:256, n:64, occurs:12));
        list.Add(LayerNorm("BERT", "LayerNorm", "[1,256,768]", b:1, s:256, h:768, occurs:24));
        list.Add(SoftmaxOp("BERT", "Softmax", "[1,12,256,256] axis=-1",
                           dims:new[]{1,12,256,256}, axis:-1, occurs:12));
        list.Add(AddBin("BERT", "Add", "[1,256,768] + [1,256,768]",
                         dims:new[]{1,256,768}, occurs:24));

        // MatMul + Add(bias) + Activation chains — the Path B target.
        // These mirror BERT FFN's (linear1 → GELU) and (linear2 → residual-add)
        // but at smaller shapes so the harness runs in ~30s. We use the same
        // shape as PerformanceComparison.MatMulAddRelu_Chain_BeatsOrEqualsOrt
        // for continuity.
        list.Add(MatMulAddReluChain("BERT-like", "MatMul+Add+Relu",
                                    "M=64,K=256,N=256 (small FFN pattern)",
                                    m:64, k:256, n:256, occurs:12));
        list.Add(MatMulAddReluChain("BERT-like", "MatMul+Add+Relu",
                                    "M=256,K=768,N=3072 (FFN up pattern)",
                                    m:256, k:768, n:3072, occurs:12));

        // BERT actually uses GELU not ReLU — same FFN up shape as above
        // but with the GELU activation that hits the new SIMD GELU pass.
        list.Add(MatMulAddActivationChain("BERT-like", "MatMul+Add+GELU",
                                          "M=256,K=768,N=3072 (FFN up GELU)",
                                          m:256, k:768, n:3072, activation:"Gelu", occurs:12));

        return list.ToArray();
    }

    private static Case MatMulAddActivationChain(string model, string op, string shape,
        int m, int k, int n, string activation, int occurs)
    {
        var A = RandArr(0xC01, m * k);
        var W = RandArr(0xC02, k * n);
        var bias = RandArr(0xC03, n);

        var graph = new GraphProto { Name = "matmul_add_act" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("A", new[] { m, k }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { k, n }, W));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { n }, bias));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { m, n }, FLOAT));

        var mm = new NodeProto { OpType = "MatMul" };
        mm.Input.Add("A"); mm.Input.Add("W"); mm.Output.Add("MM");
        var add = new NodeProto { OpType = "Add" };
        add.Input.Add("MM"); add.Input.Add("B"); add.Output.Add("MMB");
        var act = new NodeProto { OpType = activation };
        act.Input.Add("MMB"); act.Output.Add("Y");
        // For Gelu, set approximate='tanh' — the tanh form routes through
        // engine.GELU's SIMD path. Without this attribute (approximate='none'),
        // the importer decomposes Gelu into 8+ erf-based scalar ops which is
        // orders of magnitude slower. Most BERT/GPT exporters emit
        // approximate='tanh' anyway for the same reason.
        if (activation == "Gelu")
            act.Attribute.Add(new AttributeProto {
                Name = "approximate",
                Type = AttributeProto.Types.AttributeType.String,
                S = Google.Protobuf.ByteString.CopyFromUtf8("tanh"),
            });
        graph.Node.Add(mm); graph.Node.Add(add); graph.Node.Add(act);

        var modelProto = OnnxTestGraphBuilder.WrapModel(graph);
        // Gelu is opset-20 — bump our opset so ORT recognises it.
        modelProto.OpsetImport.Clear();
        modelProto.OpsetImport.Add(new OperatorSetIdProto { Version = 20 });
        byte[] bytes = OnnxTestGraphBuilder.Serialize(modelProto);
        return BuildCase(model, op, shape, occurs, bytes,
            new[] { ("A", new[] { m, k }, A) });
    }

    // ─── Path B target: fused MatMul + Add(bias) + Activation chain ────────
    private static Case MatMulAddReluChain(string model, string op, string shape,
        int m, int k, int n, int occurs)
    {
        var A = RandArr(0xB01, m * k);
        var W = RandArr(0xB02, k * n);
        var bias = RandArr(0xB03, n);

        var graph = new GraphProto { Name = "matmul_add_relu" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("A", new[] { m, k }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { k, n }, W));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { n }, bias));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { m, n }, FLOAT));

        var mm = new NodeProto { OpType = "MatMul" };
        mm.Input.Add("A"); mm.Input.Add("W"); mm.Output.Add("MM");
        var add = new NodeProto { OpType = "Add" };
        add.Input.Add("MM"); add.Input.Add("B"); add.Output.Add("MMB");
        var relu = new NodeProto { OpType = "Relu" };
        relu.Input.Add("MMB"); relu.Output.Add("Y");
        graph.Node.Add(mm); graph.Node.Add(add); graph.Node.Add(relu);

        byte[] bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        return BuildCase(model, op, shape, occurs, bytes,
            new[] { ("A", new[] { m, k }, A) });
    }

    // ─── case factories ────────────────────────────────────────────────────

    private static Case MatMul(string model, string op, string shape, int[] a, int[] b, int occurs)
    {
        var A = RandArr(0xA01, a.Aggregate(1, (p, x) => p * x));
        var B = RandArr(0xA02, b.Aggregate(1, (p, x) => p * x));
        int[] outShape = a.Take(a.Length - 1).Append(b[b.Length - 1]).ToArray();
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "MatMul",
            inputs: new[] {
                (name: "A", shape: a, elemType: FLOAT),
                (name: "B", shape: b, elemType: FLOAT),
            },
            output: (name: "Y", shape: outShape, elemType: FLOAT),
            attributes: null,
            initializers: null);
        return BuildCase(model, op, shape, occurs, OnnxTestGraphBuilder.Serialize(model_),
            new[] { ("A", a, A), ("B", b, B) });
    }

    private static Case MatMul3D(string model, string op, string shape, int b, int m, int k, int n, int occurs)
    {
        var A = RandArr(0xA03, b * m * k);
        var B = RandArr(0xA04, b * k * n);
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "MatMul",
            inputs: new[] {
                (name: "A", shape: new[] { b, m, k }, elemType: FLOAT),
                (name: "B", shape: new[] { b, k, n }, elemType: FLOAT),
            },
            output: (name: "Y", shape: new[] { b, m, n }, elemType: FLOAT),
            attributes: null,
            initializers: null);
        return BuildCase(model, op, shape, occurs, OnnxTestGraphBuilder.Serialize(model_),
            new[] { ("A", new[] { b, m, k }, A), ("B", new[] { b, k, n }, B) });
    }

    private static Case LayerNorm(string model, string op, string shape, int b, int s, int h, int occurs)
    {
        var X = RandArr(0xE01, b * s * h);
        var scale = RandArr(0xE02, h);
        var bias = RandArr(0xE03, h);
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "LayerNormalization",
            inputs: new[] { (name: "X", shape: new[] { b, s, h }, elemType: FLOAT) },
            output: (name: "Y", shape: new[] { b, s, h }, elemType: FLOAT),
            attributes: new[] { IntAttr("axis", -1), FloatAttr("epsilon", 1e-5f) },
            initializers: new[] {
                (name: "scale", shape: new[] { h }, data: scale),
                (name: "bias",  shape: new[] { h }, data: bias),
            });
        model_.OpsetImport.Clear();
        model_.OpsetImport.Add(new OperatorSetIdProto { Version = 17 });
        return BuildCase(model, op, shape, occurs, OnnxTestGraphBuilder.Serialize(model_),
            new[] { ("X", new[] { b, s, h }, X) });
    }

    private static Case SoftmaxOp(string model, string op, string shape, int[] dims, int axis, int occurs)
    {
        var X = RandArr(0xF01, dims.Aggregate(1, (p, x) => p * x));
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "Softmax",
            inputs: new[] { (name: "X", shape: dims, elemType: FLOAT) },
            output: (name: "Y", shape: dims, elemType: FLOAT),
            attributes: new[] { IntAttr("axis", axis) },
            initializers: null);
        return BuildCase(model, op, shape, occurs, OnnxTestGraphBuilder.Serialize(model_),
            new[] { ("X", dims, X) });
    }

    private static Case AddBin(string model, string op, string shape, int[] dims, int occurs)
    {
        var A = RandArr(0xA01, dims.Aggregate(1, (p, x) => p * x));
        var B = RandArr(0xA02, dims.Aggregate(1, (p, x) => p * x));
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "Add",
            inputs: new[] {
                (name: "A", shape: dims, elemType: FLOAT),
                (name: "B", shape: dims, elemType: FLOAT),
            },
            output: (name: "Y", shape: dims, elemType: FLOAT),
            attributes: null,
            initializers: null);
        return BuildCase(model, op, shape, occurs, OnnxTestGraphBuilder.Serialize(model_),
            new[] { ("A", dims, A), ("B", dims, B) });
    }

    private static Case BuildCase(string model, string op, string shape, int occurs, byte[] modelBytes,
        (string Name, int[] Shape, float[] Data)[] inputs)
    {
        // Build ours
        var engine = new CpuEngine();
        using var importStream = new System.IO.MemoryStream(modelBytes);
        var result = AiDotNet.Tensors.Onnx.OnnxImporter.Import<float>(importStream, engine);
        if (result.UnsupportedOperators.Count > 0 || result.Plan is null)
            throw new System.InvalidOperationException($"Import failed: {string.Join(",", result.UnsupportedOperators)}");
        foreach (var input in inputs)
            input.Data.AsSpan().CopyTo(result.Inputs[input.Name].AsWritableSpan());

        // Build ORT
        var session = new InferenceSession(modelBytes);
        var feeds = new System.Collections.Generic.List<NamedOnnxValue>(inputs.Length);
        foreach (var input in inputs)
        {
            var t = new DenseTensor<float>(input.Data, input.Shape);
            feeds.Add(NamedOnnxValue.CreateFromTensor(input.Name, t));
        }

        return new Case
        {
            Model = model,
            Op = op,
            Shape = shape,
            Occurrences = occurs,
            RunOurs = () => result.Plan.Execute(),
            RunOrt = () => { using var _ = session.Run(feeds); },
        };
    }

    private static float[] RandArr(int seed, int n)
    {
        var rng = new System.Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    private static AttributeProto IntAttr(string name, int value) =>
        new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value };

    private static AttributeProto FloatAttr(string name, float value) =>
        new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Float, F = value };

    private sealed class Case
    {
        public string Model = "";
        public string Op = "";
        public string Shape = "";
        public int Occurrences;
        public System.Action RunOurs = () => { };
        public System.Action RunOrt = () => { };
    }
}
