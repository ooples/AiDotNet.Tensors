using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests;

/// <summary>
/// <see cref="OnnxExporter"/> round trips: each ONNX model is imported, its compiled plan exported back to ONNX, and
/// the export both re-imported through <see cref="OnnxImporter"/> and run in ONNX Runtime. All three must agree with
/// the original import's outputs.
/// </summary>
public class OnnxExportRoundTripTests
{
    public static IEnumerable<object[]> CaseNames => RoundTripCases.All.Select(c => new object[] { c.Name });

    [Theory]
    [MemberData(nameof(CaseNames))]
    public void Export_ReimportsToTheSameOutputs(string name)
    {
        var c = RoundTripCases.Get(name);
        var (expected, exported) = RoundTrip.ImportAndExport(c);
        var reimported = RoundTrip.Run(exported, c.Feeds);
        AssertClose(expected, reimported);
    }

    [SkippableTheory]
    [MemberData(nameof(CaseNames))]
    public void Export_RunsInOnnxRuntimeToTheSameOutputs(string name)
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var c = RoundTripCases.Get(name);
        var (expected, exported) = RoundTrip.ImportAndExport(c);
        var ort = OnnxRuntimeReference.RunSingleOutput(
            exported, c.Feeds.Select(f => (f.Name, f.Shape, f.Data)).ToArray());
        AssertClose(expected, ort);
    }

    [Fact]
    public void Export_KeepsTheGraphInputAndOutputNames()
    {
        var c = RoundTripCases.Get("ResNetBasicBlock");
        var (_, exported) = RoundTrip.ImportAndExport(c);
        var model = ModelProto.Parser.ParseFrom(exported);
        Assert.Equal(new[] { "X" }, model.Graph.Input.Select(i => i.Name).ToArray());
        Assert.Equal(new[] { "Y" }, model.Graph.Output.Select(o => o.Name).ToArray());
        Assert.Contains(model.OpsetImport, o => o.Domain == string.Empty && o.Version == 20);
    }

    [Fact]
    public void Export_WeightsBecomeInitializers_InputsDoNot()
    {
        var c = RoundTripCases.Get("ResNetBasicBlock");
        var (_, exported) = RoundTrip.ImportAndExport(c);
        var model = ModelProto.Parser.ParseFrom(exported);
        var initializers = new HashSet<string>(model.Graph.Initializer.Select(i => i.Name));
        Assert.DoesNotContain("X", initializers);
        // Both conv kernels travel as float initializers of their original shape.
        Assert.Equal(2, model.Graph.Initializer.Count(i =>
            i.DataType == (int)TensorProto.Types.DataType.Float && i.Dims.SequenceEqual(new long[] { 8, 8, 3, 3 })));
    }

    [Fact]
    public void Export_PlanOverload_NamesInputsAndOutput()
    {
        var c = RoundTripCases.Get("MlpGemmReluSoftmax");
        using var stream = new MemoryStream(c.Bytes);
        var imported = OnnxImporter.Import<float>(stream, new CpuEngine());
        var expected = RoundTrip.Execute(imported, c.Feeds);
        Assert.NotNull(imported.Plan);
        var plan = imported.Plan ?? throw new InvalidOperationException();
        var model = OnnxExporter.Export(plan);
        Assert.Equal("input", Assert.Single(model.Graph.Input).Name);
        Assert.Equal("output", Assert.Single(model.Graph.Output).Name);
        var renamed = new[] { new RoundTripCases.Feed("input", c.Feeds[0].Shape, c.Feeds[0].Data) };
        AssertClose(expected, RoundTrip.Run(model.ToByteArray(), renamed));
    }

    [Fact]
    public void Export_BelowTheMinimumOpset_Throws()
    {
        var c = RoundTripCases.Get("Relu");
        using var stream = new MemoryStream(c.Bytes);
        var imported = OnnxImporter.Import<float>(stream, new CpuEngine());
        var options = new OnnxExportOptions { OpsetVersion = OnnxExportOptions.MinimumOpsetVersion - 1 };
        Assert.Throws<ArgumentOutOfRangeException>(() => OnnxExporter.Export(imported, options));
    }

    [Fact]
    public void Export_OpsetBelowTwenty_DecomposesGelu()
    {
        var c = RoundTripCases.Get("GeluTanh");
        using var stream = new MemoryStream(c.Bytes);
        var imported = OnnxImporter.Import<float>(stream, new CpuEngine());
        var expected = RoundTrip.Execute(imported, c.Feeds);
        var model = OnnxExporter.Export(imported, new OnnxExportOptions { OpsetVersion = 17 });
        Assert.DoesNotContain(model.Graph.Node, n => n.OpType == "Gelu");
        Assert.Contains(model.OpsetImport, o => o.Domain == string.Empty && o.Version == 17);
        AssertClose(expected, RoundTrip.Run(model.ToByteArray(), c.Feeds));
    }

    [Fact]
    public void Export_DoublePlan_RoundTrips()
    {
        var c = RoundTripCases.Get("MlpGemmReluSoftmax");
        using var stream = new MemoryStream(c.Bytes);
        var imported = OnnxImporter.Import<double>(stream, new CpuEngine());
        foreach (var f in c.Feeds)
        {
            var t = imported.Inputs[f.Name];
            var span = t.AsWritableSpan();
            for (int i = 0; i < f.Data.Length; i++) span[i] = f.Data[i];
            t.MarkModified();
        }
        var plan = imported.Plan ?? throw new InvalidOperationException("no plan");
        var expected = plan.Execute().ToArray();

        var model = OnnxExporter.Export(imported);
        Assert.Equal((int)TensorProto.Types.DataType.Double, model.Graph.Input[0].Type.TensorType.ElemType);
        using var exported = new MemoryStream(model.ToByteArray());
        var again = OnnxImporter.Import<double>(exported, new CpuEngine());
        foreach (var f in c.Feeds)
        {
            var t = again.Inputs[f.Name];
            var span = t.AsWritableSpan();
            for (int i = 0; i < f.Data.Length; i++) span[i] = f.Data[i];
            t.MarkModified();
        }
        var actual = (again.Plan ?? throw new InvalidOperationException("no plan")).Execute().ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++) Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-9 * Math.Max(1, Math.Abs(expected[i])));
    }

    [Fact]
    public void Export_EngineTracedPlan_RoundTrips()
    {
        // The Tensors-only path: a plan traced from engine calls, exported with the plan overload.
        var (plan, input) = EnginePlans.LinearRelu();
        var data = RandomArray(seed: 77, n: input.Length);
        data.AsSpan().CopyTo(input.AsWritableSpan());
        input.MarkModified();
        var expected = plan.Execute().ToArray();

        var model = OnnxExporter.Export(plan);
        var feeds = new[] { new RoundTripCases.Feed("input", input._shape, data) };
        AssertClose(expected, RoundTrip.Run(model.ToByteArray(), feeds));
    }

    [Fact]
    public void Export_UnsupportedOperation_NamesIt()
    {
        var plan = EnginePlans.CumSum();
        var e = Assert.Throws<NotSupportedException>(() => OnnxExporter.Export(plan));
        Assert.Contains("TensorCumSum", e.Message);
    }
}

/// <summary>Plans traced directly from engine calls, as a Tensors-only user builds them.</summary>
internal static class EnginePlans
{
    internal static (ICompiledPlan<float> Plan, Tensor<float> Input) LinearRelu()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 2, 4 });
        var w = new Tensor<float>(RandomArray(seed: 78, n: 12), new[] { 4, 3 });
        var b = new Tensor<float>(RandomArray(seed: 79, n: 3), new[] { 3 });
        var plan = new CompiledModelCache<float>().GetOrCompileInference(input, () =>
            engine.ReLU(engine.TensorBroadcastAdd(engine.TensorMatMul(input, w), b)));
        return (plan, input);
    }

    internal static ICompiledPlan<float> CumSum()
    {
        var engine = new CpuEngine();
        var input = new Tensor<float>(new[] { 2, 4 });
        return new CompiledModelCache<float>().GetOrCompileInference(input, () =>
            engine.ReLU(engine.TensorCumSum(input, 1)));
    }
}

/// <summary>Import, export and re-run helpers for <see cref="OnnxExportRoundTripTests"/>.</summary>
internal static class RoundTrip
{
    internal static (float[] Expected, byte[] Exported) ImportAndExport(RoundTripCases.Case c)
    {
        using var stream = new MemoryStream(c.Bytes);
        var imported = OnnxImporter.Import<float>(stream, new CpuEngine());
        Assert.Empty(imported.UnsupportedOperators);
        var expected = Execute(imported, c.Feeds);
        return (expected, OnnxExporter.Export(imported).ToByteArray());
    }

    internal static float[] Run(byte[] model, IReadOnlyList<RoundTripCases.Feed> feeds)
    {
        using var stream = new MemoryStream(model);
        var imported = OnnxImporter.Import<float>(stream, new CpuEngine());
        Assert.Empty(imported.UnsupportedOperators);
        return Execute(imported, feeds);
    }

    internal static float[] Execute(OnnxImportResult<float> imported, IReadOnlyList<RoundTripCases.Feed> feeds)
    {
        var plan = imported.Plan ?? throw new InvalidOperationException("The import produced no plan.");
        foreach (var f in feeds)
        {
            var t = imported.Inputs[f.Name];
            f.Data.AsSpan().CopyTo(t.AsWritableSpan());
            t.MarkModified();
        }
        return plan.Execute().ToArray();
    }
}

/// <summary>The ONNX models the round-trip tests export: one per importer operator family, plus whole blocks.</summary>
internal static class RoundTripCases
{
    internal sealed class Feed
    {
        internal Feed(string name, int[] shape, float[] data) { Name = name; Shape = shape; Data = data; }
        internal string Name { get; }
        internal int[] Shape { get; }
        internal float[] Data { get; }
    }

    internal sealed class Case
    {
        internal Case(string name, byte[] bytes, IReadOnlyList<Feed> feeds) { Name = name; Bytes = bytes; Feeds = feeds; }
        internal string Name { get; }
        internal byte[] Bytes { get; }
        internal IReadOnlyList<Feed> Feeds { get; }
    }

    internal static Case Get(string name) => All.First(c => c.Name == name);

    internal static readonly IReadOnlyList<Case> All = Build();

    private static List<Case> Build()
    {
        var cases = new List<Case>();
        void Unary(string op, float lo = -2f, float hi = 2f, int opset = 17, params AttributeProto[] attrs)
        {
            var n = new Net(op);
            var x = n.In("X", new[] { 2, 3, 4 }, lo, hi);
            cases.Add(n.Build(n.Op(op, new[] { x }, attrs), new[] { 2, 3, 4 }, opset));
        }

        foreach (var op in new[] { "Relu", "Sigmoid", "Tanh", "Exp", "Abs", "Neg", "Erf" }) Unary(op);
        Unary("Log", 0.1f, 3f);
        Unary("Sqrt", 0.1f, 3f);
        Unary("Reciprocal", 0.5f, 3f);
        Unary("LeakyRelu", attrs: new[] { Float("alpha", 0.2f) });
        Unary("Softmax", attrs: new[] { Int("axis", -1) });
        cases.Add(Named("GeluTanh", n =>
        {
            var x = n.In("X", new[] { 2, 3, 4 });
            return (n.Op("Gelu", new[] { x }, Str("approximate", "tanh")), new[] { 2, 3, 4 }, 20);
        }));
        cases.Add(Named("GeluExact", n =>
        {
            var x = n.In("X", new[] { 2, 3, 4 });
            return (n.Op("Gelu", new[] { x }), new[] { 2, 3, 4 }, 20);
        }));

        foreach (var op in new[] { "Add", "Sub", "Mul", "Div", "Max", "Min" })
        {
            cases.Add(Named(op + "Broadcast", n =>
            {
                var a = n.In("A", new[] { 2, 3, 4 });
                var b = n.W(new[] { 3, 1 }, 0.5f, 2f);
                return (n.Op(op, new[] { a, b }), new[] { 2, 3, 4 }, 17);
            }));
        }
        cases.Add(Named("PowScalar", n =>
        {
            var a = n.In("A", new[] { 2, 3 }, 0.1f, 2f);
            return (n.Op("Pow", new[] { a, n.W(new int[0], 2f, 2f) }), new[] { 2, 3 }, 17);
        }));
        cases.Add(Named("MatMulBatched", n =>
        {
            var a = n.In("A", new[] { 2, 3, 4 });
            return (n.Op("MatMul", new[] { a, n.W(new[] { 4, 5 }) }), new[] { 2, 3, 5 }, 17);
        }));
        cases.Add(Named("MlpGemmReluSoftmax", n =>
        {
            var x = n.In("X", new[] { 2, 8 });
            var h = n.Op("Gemm", new[] { x, n.W(new[] { 8, 6 }), n.W(new[] { 6 }) });
            var r = n.Op("Relu", new[] { h });
            var g = n.Op("Gemm", new[] { r, n.W(new[] { 4, 6 }), n.W(new[] { 4 }) }, Int("transB", 1));
            return (n.Op("Softmax", new[] { g }, Int("axis", 1)), new[] { 2, 4 }, 17);
        }));

        cases.Add(Named("ReshapeTranspose", n =>
        {
            var x = n.In("X", new[] { 2, 3, 4 });
            var r = n.Op("Reshape", new[] { x, n.I64(new long[] { 2, 12 }) });
            var t = n.Op("Transpose", new[] { r }, Ints("perm", 1, 0));
            return (n.Op("Relu", new[] { t }), new[] { 12, 2 }, 17);
        }));
        cases.Add(Named("Permute4D", n =>
        {
            var x = n.In("X", new[] { 2, 3, 4, 5 });
            return (n.Op("Transpose", new[] { x }, Ints("perm", 0, 2, 3, 1)), new[] { 2, 4, 5, 3 }, 17);
        }));
        cases.Add(Named("FlattenSqueezeUnsqueeze", n =>
        {
            var x = n.In("X", new[] { 2, 1, 3, 4 });
            var s = n.Op("Squeeze", new[] { x, n.I64(new long[] { 1 }) });
            var u = n.Op("Unsqueeze", new[] { s, n.I64(new long[] { 3 }) });
            return (n.Op("Flatten", new[] { u }, Int("axis", 1)), new[] { 2, 12 }, 17);
        }));
        cases.Add(Named("SliceConcat", n =>
        {
            var x = n.In("X", new[] { 2, 6, 4 });
            var s = n.Op("Slice", new[] { x, n.I64(new long[] { 1 }), n.I64(new long[] { 4 }), n.I64(new long[] { 1 }) });
            return (n.Op("Concat", new[] { s, s }, Int("axis", 1)), new[] { 2, 6, 4 }, 17);
        }));
        cases.Add(Named("SplitAdd", n =>
        {
            var x = n.In("X", new[] { 2, 6 });
            var parts = n.OpMulti("Split", new[] { x }, 2, Int("axis", 1));
            return (n.Op("Add", new[] { parts[0], parts[1] }), new[] { 2, 3 }, 17);
        }));
        cases.Add(Named("GatherRows", n =>
        {
            var emb = n.In("E", new[] { 5, 4 });
            return (n.Op("Gather", new[] { emb, n.I64(new long[] { 4, 0, 2 }) }, Int("axis", 0)), new[] { 3, 4 }, 17);
        }));
        cases.Add(Named("ExpandRow", n =>
        {
            var x = n.In("X", new[] { 1, 4 });
            return (n.Op("Expand", new[] { x, n.I64(new long[] { 3, 4 }) }), new[] { 3, 4 }, 17);
        }));
        cases.Add(Named("ReduceSumMean", n =>
        {
            var x = n.In("X", new[] { 2, 3, 4 });
            var s = n.Op("ReduceSum", new[] { x, n.I64(new long[] { 2 }) }, Int("keepdims", 1));
            return (n.Op("ReduceMean", new[] { s, n.I64(new long[] { 1 }) }, Int("keepdims", 0)), new[] { 2, 1 }, 18);
        }));

        cases.Add(Named("ConvStridedBias", n =>
        {
            var x = n.In("X", new[] { 1, 3, 9, 9 });
            return (n.Op("Conv", new[] { x, n.W(new[] { 4, 3, 3, 3 }), n.W(new[] { 4 }) },
                Ints("kernel_shape", 3, 3), Ints("strides", 2, 2), Ints("pads", 1, 1, 1, 1)), new[] { 1, 4, 5, 5 }, 17);
        }));
        cases.Add(Named("ConvAsymmetricPad", n =>
        {
            var x = n.In("X", new[] { 1, 2, 6, 6 });
            return (n.Op("Conv", new[] { x, n.W(new[] { 3, 2, 3, 3 }) },
                Ints("kernel_shape", 3, 3), Ints("pads", 0, 0, 1, 1)), new[] { 1, 3, 5, 5 }, 17);
        }));
        cases.Add(Named("ConvDepthwise", n =>
        {
            var x = n.In("X", new[] { 1, 4, 6, 6 });
            return (n.Op("Conv", new[] { x, n.W(new[] { 4, 1, 3, 3 }) },
                Ints("kernel_shape", 3, 3), Ints("pads", 1, 1, 1, 1), Int("group", 4)), new[] { 1, 4, 6, 6 }, 17);
        }));
        cases.Add(Named("ConvTransposeStrided", n =>
        {
            var x = n.In("X", new[] { 1, 4, 4, 4 });
            return (n.Op("ConvTranspose", new[] { x, n.W(new[] { 4, 2, 3, 3 }) },
                Ints("kernel_shape", 3, 3), Ints("strides", 2, 2), Ints("pads", 1, 1, 1, 1)), new[] { 1, 2, 7, 7 }, 17);
        }));
        cases.Add(Named("MaxPoolAvgPool", n =>
        {
            var x = n.In("X", new[] { 1, 2, 8, 8 });
            var m = n.Op("MaxPool", new[] { x }, Ints("kernel_shape", 2, 2), Ints("strides", 2, 2));
            return (n.Op("AveragePool", new[] { m }, Ints("kernel_shape", 3, 3), Ints("strides", 1, 1),
                Ints("pads", 1, 1, 1, 1), Int("count_include_pad", 1)), new[] { 1, 2, 4, 4 }, 17);
        }));
        cases.Add(Named("ResNetBasicBlock", n =>
        {
            // conv-bn-relu-conv-bn + identity, relu, global pool, flatten, classifier: the ResNet-18 basic block.
            var x = n.In("X", new[] { 1, 8, 6, 6 });
            string ConvBn(string input)
            {
                var c = n.Op("Conv", new[] { input, n.W(new[] { 8, 8, 3, 3 }, -0.3f, 0.3f) },
                    Ints("kernel_shape", 3, 3), Ints("pads", 1, 1, 1, 1));
                return n.Op("BatchNormalization", new[]
                {
                    c, n.W(new[] { 8 }, 0.5f, 1.5f), n.W(new[] { 8 }), n.W(new[] { 8 }), n.W(new[] { 8 }, 0.5f, 2f),
                }, Float("epsilon", 1e-5f));
            }
            var a = n.Op("Relu", new[] { ConvBn(x) });
            var sum = n.Op("Add", new[] { ConvBn(a), x });
            var r = n.Op("Relu", new[] { sum });
            var gap = n.Op("GlobalAveragePool", new[] { r });
            var flat = n.Op("Flatten", new[] { gap }, Int("axis", 1));
            return (n.Op("Gemm", new[] { flat, n.W(new[] { 3, 8 }), n.W(new[] { 3 }) }, Int("transB", 1)), new[] { 1, 3 }, 17);
        }));
        cases.Add(Named("BertEncoderLayer", n =>
        {
            // Single-head self-attention + feed-forward with residuals and post-LayerNorm, as in BERT.
            const int S = 4, D = 8, F = 16;
            var x = n.In("X", new[] { 1, S, D });
            string Linear(string input, int i, int o) => n.Op("Add", new[] { n.Op("MatMul", new[] { input, n.W(new[] { i, o }, -0.4f, 0.4f) }), n.W(new[] { o }) });
            string Norm(string input) => n.Op("LayerNormalization", new[] { input, n.W(new[] { D }, 0.5f, 1.5f), n.W(new[] { D }) },
                Int("axis", -1), Float("epsilon", 1e-12f));
            var q = Linear(x, D, D);
            var k = Linear(x, D, D);
            var v = Linear(x, D, D);
            var kt = n.Op("Transpose", new[] { k }, Ints("perm", 0, 2, 1));
            var scores = n.Op("Div", new[] { n.Op("MatMul", new[] { q, kt }), n.W(new int[0], (float)Math.Sqrt(D), (float)Math.Sqrt(D)) });
            var probs = n.Op("Softmax", new[] { scores }, Int("axis", -1));
            var attn = Linear(n.Op("MatMul", new[] { probs, v }), D, D);
            var h = Norm(n.Op("Add", new[] { attn, x }));
            var ff = Linear(n.Op("Gelu", new[] { Linear(h, D, F) }), F, D);
            return (Norm(n.Op("Add", new[] { ff, h })), new[] { 1, S, D }, 20);
        }));
        cases.Add(Named("GruSequence", n =>
        {
            var x = n.In("X", new[] { 3, 1, 2 });
            var outs = n.OpMulti("GRU", new[] { x, n.W(new[] { 1, 9, 2 }), n.W(new[] { 1, 9, 3 }) }, 1, Int("hidden_size", 3));
            return (outs[0], new[] { 3, 1, 1, 3 }, 17);
        }));
        cases.Add(Named("WhereEqual", n =>
        {
            var a = n.In("A", new[] { 2, 3 });
            var b = n.In("B", new[] { 2, 3 });
            var eq = n.Op("Equal", new[] { a, a });
            return (n.Op("Where", new[] { eq, b, a }), new[] { 2, 3 }, 17);
        }));
        cases.Add(Named("LstmSequence", n =>
        {
            var x = n.In("X", new[] { 3, 1, 2 });
            var outs = n.OpMulti("LSTM", new[] { x, n.W(new[] { 1, 12, 2 }), n.W(new[] { 1, 12, 3 }) }, 1, Int("hidden_size", 3));
            return (outs[0], new[] { 3, 1, 1, 3 }, 17);
        }));
        return cases;
    }

    private static Case Named(string name, Func<Net, (string Output, int[] Shape, int Opset)> body)
    {
        var n = new Net(name);
        var (output, shape, opset) = body(n);
        return n.Build(output, shape, opset);
    }

    private static AttributeProto Int(string name, long value) => new() { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value };

    private static AttributeProto Float(string name, float value) => new() { Name = name, Type = AttributeProto.Types.AttributeType.Float, F = value };

    private static AttributeProto Str(string name, string value) => new() { Name = name, Type = AttributeProto.Types.AttributeType.String, S = ByteString.CopyFromUtf8(value) };

    private static AttributeProto Ints(string name, params long[] values)
    {
        var a = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        a.Ints.AddRange(values);
        return a;
    }

    /// <summary>A small graph under construction: inputs get seeded random feeds, weights seeded random initializers.</summary>
    private sealed class Net
    {
        private readonly string _name;
        private readonly GraphProto _graph;
        private readonly List<Feed> _feeds = new();
        private int _seed;
        private int _next;

        internal Net(string name)
        {
            _name = name;
            _graph = new GraphProto { Name = name };
            _seed = name.Aggregate(17, (h, ch) => unchecked(h * 31 + ch)) & 0x7fff;
        }

        internal string In(string name, int[] shape, float lo = -1f, float hi = 1f)
        {
            _graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo(name, shape, FLOAT));
            _feeds.Add(new Feed(name, shape, RandomArray(_seed++, Count(shape), lo, hi)));
            return name;
        }

        internal string W(int[] shape, float lo = -1f, float hi = 1f)
        {
            var name = "w" + _next++;
            _graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer(name, shape, RandomArray(_seed++, Count(shape), lo, hi)));
            return name;
        }

        internal string I64(long[] values)
        {
            var name = "i" + _next++;
            _graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer(name, new[] { values.Length }, values));
            return name;
        }

        internal string Op(string type, string[] inputs, params AttributeProto[] attributes)
            => OpMulti(type, inputs, 1, attributes)[0];

        internal string[] OpMulti(string type, string[] inputs, int outputs, params AttributeProto[] attributes)
        {
            var node = new NodeProto { OpType = type, Name = type + "_" + _next };
            node.Input.AddRange(inputs);
            var names = new string[outputs];
            for (int i = 0; i < outputs; i++) node.Output.Add(names[i] = "v" + _next++);
            node.Attribute.AddRange(attributes);
            _graph.Node.Add(node);
            return names;
        }

        internal Case Build(string output, int[] shape, int opset)
        {
            var last = _graph.Node[_graph.Node.Count - 1];
            if (last.Output[0] != output) throw new InvalidOperationException("A case's result must be its last node's first output.");
            last.Output[0] = "Y";
            _graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", shape, FLOAT));
            return new Case(_name, OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(_graph, opset)), _feeds);
        }

        private static int Count(int[] shape) => shape.Aggregate(1, (a, d) => a * d);
    }
}
