using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// Phase 1 (systematic-debugging) evidence harness for Issue #169 acceptance
/// bullet 3 ("matches or beats ONNX Runtime on CPU latency"). Does NOT
/// assert wins — it records measurements. The output feeds Phase 2 triage:
/// which ops are disproportionately slow vs ORT, so we can target custom
/// kernels with scientific-method A/B testing instead of guessing.
///
/// <para>The table is the hot-shape set from ResNet-50 (conv-heavy) and
/// BERT-base (matmul-heavy), seq=256 batch=1. For each (op, shape) we
/// build a single-node ONNX model, run it through our importer + compiled
/// plan, and through ONNX Runtime's InferenceSession — then time N replays
/// after M warmup runs. The compile / session-build cost is excluded from
/// the timing loop because it's amortised in real inference.</para>
///
/// <para>Per-op wall-clock is multiplied by the op's expected occurrence
/// count in the full model (ResNet-50 or BERT-base), so the ratio table
/// sorts by total aggregate cost, not by per-call cost. That's the right
/// signal for "where do we bleed time".</para>
///
/// <para>Emits tab-separated output so the caller can paste into a
/// spreadsheet. Skippable when ORT is unavailable.</para>
/// </summary>
public class OpLevelPerfHarness
{
    private readonly ITestOutputHelper _output;
    public OpLevelPerfHarness(ITestOutputHelper output) { _output = output; }

    private const int Warmup = 5;
    private const int Iters  = 20;
    private const int FLOAT  = OnnxTestHelpers.FLOAT;

    [SkippableFact]
    public void EmitHotShapeRatioTable()
    {
        // Evidence harness for Phase 1 / 2 perf analysis. Gated behind
        // AIDOTNET_RUN_PERF_HARNESS=1 so CI skips it by default (~7s per run
        // post-fix, was 70s pre-fix). Run manually to refresh the ratio
        // table.
        Skip.IfNot(
            Environment.GetEnvironmentVariable("AIDOTNET_RUN_PERF_HARNESS") == "1",
            "Set AIDOTNET_RUN_PERF_HARNESS=1 to run this evidence harness.");
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable,
            "Onnx Runtime not loadable on this host; install Microsoft.ML.OnnxRuntime to gather ratios.");

        var cases = BuildCases();
        var rows = new List<Row>();
        foreach (var c in cases)
        {
            try
            {
                var r = Measure(c);
                rows.Add(r);
            }
            catch (Exception ex)
            {
                _output.WriteLine($"SKIP {c.Label}: {ex.GetType().Name}: {ex.Message}");
            }
        }

        // Sort by aggregate cost (ours µs × occurrences) descending — the bottleneck list.
        rows.Sort((a, b) => (b.OursUs * b.Occurrences).CompareTo(a.OursUs * a.Occurrences));

        _output.WriteLine("");
        _output.WriteLine("=== Op-level hot-shape ratio table (phase 1 evidence) ===");
        _output.WriteLine("model\top\tshape\tours_us\tort_us\tratio\toccurs\tagg_ours_ms\tagg_ort_ms");
        double sumOursMs = 0, sumOrtMs = 0;
        foreach (var r in rows)
        {
            double aggOurs = r.OursUs * r.Occurrences / 1000.0;
            double aggOrt  = r.OrtUs  * r.Occurrences / 1000.0;
            sumOursMs += aggOurs;
            sumOrtMs  += aggOrt;
            _output.WriteLine(
                $"{r.Model}\t{r.Op}\t{r.Shape}\t{r.OursUs:F1}\t{r.OrtUs:F1}\t{r.Ratio:F2}\t{r.Occurrences}\t{aggOurs:F1}\t{aggOrt:F1}");
        }
        _output.WriteLine($"TOTAL\t\t\t\t\t\t\t{sumOursMs:F1}\t{sumOrtMs:F1}");
        _output.WriteLine($"Aggregate us/ORT = {sumOursMs / Math.Max(sumOrtMs, 0.001):F1}x");
    }

    // ─── cases ──────────────────────────────────────────────────────────────

    private static List<Case> BuildCases()
    {
        var list = new List<Case>();

        // ResNet-50 hot conv + surrounding ops (counts are approximate per-inference occurrences).
        list.Add(Conv("RN", "Conv", "1x64x56x56; w 3x3x64->64",   n:1, c:64,  h:56,  w:56,  co:64,  kH:3, kW:3, stride:1, pad:1, occurs: 6));
        list.Add(Conv("RN", "Conv", "1x64x56x56; w 1x1x64->64",   n:1, c:64,  h:56,  w:56,  co:64,  kH:1, kW:1, stride:1, pad:0, occurs: 3));
        list.Add(Conv("RN", "Conv", "1x128x28x28; w 3x3x128->128",n:1, c:128, h:28,  w:28,  co:128, kH:3, kW:3, stride:1, pad:1, occurs: 6));
        list.Add(Conv("RN", "Conv", "1x256x14x14; w 3x3x256->256",n:1, c:256, h:14,  w:14,  co:256, kH:3, kW:3, stride:1, pad:1, occurs:12));
        list.Add(Conv("RN", "Conv", "1x512x7x7;   w 3x3x512->512",n:1, c:512, h: 7,  w: 7,  co:512, kH:3, kW:3, stride:1, pad:1, occurs: 6));
        list.Add(BatchNorm("RN", "BatchNorm", "1x256x14x14", n:1, c:256, h:14, w:14, occurs:53));
        list.Add(Relu     ("RN", "Relu",     "1x256x14x14", n:1, c:256, h:14, w:14, occurs:48));
        list.Add(MaxPool  ("RN", "MaxPool",  "1x64x112x112; 3x3 s2", n:1, c:64, h:112, w:112, kH:3, kW:3, stride:2, pad:1, occurs:1));

        // BERT-base hot ops (12 layers, seq=256, hidden=768, ffn=3072, heads=12, head_dim=64).
        list.Add(MatMul3D("BERT", "MatMul", "[1,256,768]x[768,768]  (QKV proj)",
                          b:1, m:256, k:768, n:768, occurs:48));
        list.Add(MatMul3D("BERT", "MatMul", "[12,256,64]x[12,64,256] (attn scores)",
                          b:12, m:256, k:64, n:256, occurs:12));
        list.Add(MatMul3D("BERT", "MatMul", "[12,256,256]x[12,256,64] (attn×V)",
                          b:12, m:256, k:256, n:64, occurs:12));
        list.Add(MatMul3D("BERT", "MatMul", "[1,256,768]x[768,3072] (FFN up)",
                          b:1, m:256, k:768, n:3072, occurs:12));
        list.Add(MatMul3D("BERT", "MatMul", "[1,256,3072]x[3072,768] (FFN down)",
                          b:1, m:256, k:3072, n:768, occurs:12));
        list.Add(LayerNorm("BERT", "LayerNorm", "[1,256,768]", b:1, s:256, h:768, occurs:24));
        list.Add(Softmax ("BERT", "Softmax", "[1,12,256,256] axis=-1", dims:new[]{1,12,256,256}, axis:-1, occurs:12));
        list.Add(AddBin  ("BERT", "Add", "[1,256,768] + [1,256,768]", dims:new[]{1,256,768}, occurs:24));

        return list;
    }

    // ─── measurement ────────────────────────────────────────────────────────

    private Row Measure(Case c)
    {
        // OURS: import → compile once, then time N replays of Execute.
        var engine = new CpuEngine();
        using var importStream = new MemoryStream(c.ModelBytes);
        var result = OnnxImporter.Import<float>(importStream, engine);
        if (result.UnsupportedOperators.Count > 0 || result.Plan is null)
            throw new InvalidOperationException($"Import failed: unsupported={{{string.Join(",", result.UnsupportedOperators)}}}, plan={(result.Plan is null ? "null" : "ok")}");

        foreach (var input in c.Inputs)
            input.Data.AsSpan().CopyTo(result.Inputs[input.Name].AsWritableSpan());

        // Warmup
        for (int i = 0; i < Warmup; i++) result.Plan.Execute();
        // Time
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) result.Plan.Execute();
        sw.Stop();
        double oursUs = sw.Elapsed.TotalMilliseconds * 1000.0 / Iters;

        // ORT: construct session, feed inputs, time N Run() calls.
        using var session = new InferenceSession(c.ModelBytes);
        var feeds = new List<NamedOnnxValue>(c.Inputs.Length);
        foreach (var input in c.Inputs)
        {
            var dims = Array.ConvertAll(input.Shape, i => (long)i);
            var t = new DenseTensor<float>(input.Data, input.Shape);
            feeds.Add(NamedOnnxValue.CreateFromTensor(input.Name, t));
        }
        for (int i = 0; i < Warmup; i++) { using var _ = session.Run(feeds); }
        var sw2 = Stopwatch.StartNew();
        for (int i = 0; i < Iters; i++) { using var _ = session.Run(feeds); }
        sw2.Stop();
        double ortUs = sw2.Elapsed.TotalMilliseconds * 1000.0 / Iters;

        return new Row
        {
            Model = c.Model,
            Op = c.Op,
            Shape = c.ShapeLabel,
            Label = c.Label,
            OursUs = oursUs,
            OrtUs = ortUs,
            Ratio = oursUs / Math.Max(ortUs, 0.001),
            Occurrences = c.Occurrences,
        };
    }

    // ─── case builders ──────────────────────────────────────────────────────

    private static Case Conv(string model, string op, string shape, int n, int c, int h, int w, int co, int kH, int kW, int stride, int pad, int occurs)
    {
        var X = RandArr(0xC01, n * c * h * w);
        var W = RandArr(0xC02, co * c * kH * kW);
        var oh = (h + 2 * pad - kH) / stride + 1;
        var ow = (w + 2 * pad - kW) / stride + 1;
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "Conv",
            inputs: new[] { (name: "X", shape: new[] { n, c, h, w }, elemType: FLOAT) },
            output: (name: "Y", shape: new[] { n, co, oh, ow }, elemType: FLOAT),
            attributes: new[]
            {
                IntsAttr("kernel_shape", kH, kW),
                IntsAttr("strides",      stride, stride),
                IntsAttr("pads",         pad, pad, pad, pad),
                IntsAttr("dilations",    1, 1),
            },
            initializers: new[] { (name: "W", shape: new[] { co, c, kH, kW }, data: W) });
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("X", new[] { n, c, h, w }, X) },
            Occurrences = occurs,
        };
    }

    private static Case BatchNorm(string model, string op, string shape, int n, int c, int h, int w, int occurs)
    {
        var X = RandArr(0xB01, n * c * h * w);
        var scale = RandArr(0xB02, c);
        var bias = RandArr(0xB03, c);
        var mean = RandArr(0xB04, c);
        var var  = PosArr (0xB05, c);
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "BatchNormalization",
            inputs: new[] { (name: "X", shape: new[] { n, c, h, w }, elemType: FLOAT) },
            output: (name: "Y", shape: new[] { n, c, h, w }, elemType: FLOAT),
            attributes: new[] { FloatAttr("epsilon", 1e-5f) },
            initializers: new[]
            {
                (name: "scale", shape: new[] { c }, data: scale),
                (name: "bias",  shape: new[] { c }, data: bias),
                (name: "mean",  shape: new[] { c }, data: mean),
                (name: "var",   shape: new[] { c }, data: var),
            });
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("X", new[] { n, c, h, w }, X) },
            Occurrences = occurs,
        };
    }

    private static Case Relu(string model, string op, string shape, int n, int c, int h, int w, int occurs)
    {
        var X = RandArr(0xA01, n * c * h * w);
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "Relu",
            inputs: new[] { (name: "X", shape: new[] { n, c, h, w }, elemType: FLOAT) },
            output: (name: "Y", shape: new[] { n, c, h, w }, elemType: FLOAT));
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("X", new[] { n, c, h, w }, X) },
            Occurrences = occurs,
        };
    }

    private static Case MaxPool(string model, string op, string shape, int n, int c, int h, int w, int kH, int kW, int stride, int pad, int occurs)
    {
        var X = RandArr(0xD01, n * c * h * w);
        int oh = (h + 2 * pad - kH) / stride + 1;
        int ow = (w + 2 * pad - kW) / stride + 1;
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "MaxPool",
            inputs: new[] { (name: "X", shape: new[] { n, c, h, w }, elemType: FLOAT) },
            output: (name: "Y", shape: new[] { n, c, oh, ow }, elemType: FLOAT),
            attributes: new[]
            {
                IntsAttr("kernel_shape", kH, kW),
                IntsAttr("strides", stride, stride),
                IntsAttr("pads",   pad, pad, pad, pad),
            });
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("X", new[] { n, c, h, w }, X) },
            Occurrences = occurs,
        };
    }

    private static Case MatMul3D(string model, string op, string shape, int b, int m, int k, int n, int occurs)
    {
        // Input A is dynamic [b, m, k]; B is a weight initializer [k, n] (or [b, k, n] for batched K×V case).
        // The test shape label indicates which form. To keep this harness uniform, we emit B as an initializer
        // in both cases — its numeric values are random so the operation is a pure compute microbench.
        int[] aShape;
        int[] bShape;
        int[] cShape;
        if (shape.StartsWith("[12,"))
        {
            // batched 3D × 3D: A[b,m,k] × B[b,k,n]  →  C[b,m,n]
            aShape = new[] { b, m, k };
            bShape = new[] { b, k, n };
            cShape = new[] { b, m, n };
        }
        else
        {
            // 3D × 2D (broadcast): A[1,m,k] × B[k,n]  →  C[1,m,n]
            aShape = new[] { 1, m, k };
            bShape = new[] { k, n };
            cShape = new[] { 1, m, n };
        }
        var A = RandArr(0x3A01, Prod(aShape));
        var B = RandArr(0x3A02, Prod(bShape));
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "MatMul",
            inputs: new[] { (name: "A", shape: aShape, elemType: FLOAT) },
            output: (name: "C", shape: cShape, elemType: FLOAT),
            initializers: new[] { (name: "B", shape: bShape, data: B) });
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("A", aShape, A) },
            Occurrences = occurs,
        };
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
            initializers: new[]
            {
                (name: "scale", shape: new[] { h }, data: scale),
                (name: "bias",  shape: new[] { h }, data: bias),
            });
        // LayerNormalization as a native op is opset-17. Bump the opset here.
        model_.OpsetImport.Clear();
        model_.OpsetImport.Add(new OperatorSetIdProto { Version = 17 });
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("X", new[] { b, s, h }, X) },
            Occurrences = occurs,
        };
    }

    private static Case Softmax(string model, string op, string shape, int[] dims, int axis, int occurs)
    {
        var X = RandArr(0xF01, Prod(dims));
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "Softmax",
            inputs: new[] { (name: "X", shape: dims, elemType: FLOAT) },
            output: (name: "Y", shape: dims, elemType: FLOAT),
            attributes: new[] { IntAttr("axis", axis) });
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("X", dims, X) },
            Occurrences = occurs,
        };
    }

    private static Case AddBin(string model, string op, string shape, int[] dims, int occurs)
    {
        var A = RandArr(0xAA1, Prod(dims));
        var B = RandArr(0xAA2, Prod(dims));
        var model_ = OnnxTestGraphBuilder.SingleOp(
            opType: "Add",
            inputs: new[]
            {
                (name: "A", shape: dims, elemType: FLOAT),
                (name: "B", shape: dims, elemType: FLOAT),
            },
            output: (name: "C", shape: dims, elemType: FLOAT));
        return new Case
        {
            Model = model, Op = op, ShapeLabel = shape, Label = $"{model}.{op} {shape}",
            ModelBytes = OnnxTestGraphBuilder.Serialize(model_),
            Inputs = new[] { new InputT("A", dims, A), new InputT("B", dims, B) },
            Occurrences = occurs,
        };
    }

    // ─── attribute builders ─────────────────────────────────────────────────

    private static AttributeProto IntsAttr(string name, params int[] values)
    {
        var a = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var v in values) a.Ints.Add(v);
        return a;
    }

    private static AttributeProto IntAttr(string name, int value)
    {
        return new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value };
    }

    private static AttributeProto FloatAttr(string name, float value)
    {
        return new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Float, F = value };
    }

    // ─── utility ────────────────────────────────────────────────────────────

    private static float[] RandArr(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return a;
    }

    // Strictly positive values for BN variance (required: var > 0 to avoid div-by-zero).
    private static float[] PosArr(int seed, int n)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 0.1f + (float)rng.NextDouble();
        return a;
    }

    private static int Prod(int[] s) { int p = 1; foreach (var d in s) p *= d; return p; }

    // ─── types ──────────────────────────────────────────────────────────────

    private sealed class Case
    {
        public string Model = string.Empty;
        public string Op = string.Empty;
        public string ShapeLabel = string.Empty;
        public string Label = string.Empty;
        public byte[] ModelBytes = Array.Empty<byte>();
        public InputT[] Inputs = Array.Empty<InputT>();
        public int Occurrences;
    }

    private sealed class InputT
    {
        public readonly string Name;
        public readonly int[] Shape;
        public readonly float[] Data;
        public InputT(string name, int[] shape, float[] data) { Name = name; Shape = shape; Data = data; }
    }

    private struct Row
    {
        public string Model;
        public string Op;
        public string Shape;
        public string Label;
        public double OursUs;
        public double OrtUs;
        public double Ratio;
        public int Occurrences;
    }
}
