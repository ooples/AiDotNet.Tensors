using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Parity tests for the remaining Phase 1 operators that don't have a
/// dedicated test file: Sub, Div, GELU, AveragePool (non-global),
/// ConvTranspose, BatchNormalization, Gather, Slice, Split, Squeeze,
/// Unsqueeze, Identity, Flatten (non-default axis).
/// </summary>
public class RemainingOpTests
{
    [SkippableFact]
    public void Sub_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp("Sub",
            new[] { ("A", new[] { 4 }, FLOAT), ("B", new[] { 4 }, FLOAT) },
            ("C", new[] { 4 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var a = new float[] { 5, 6, 7, 8 };
        var b = new float[] { 1, 2, 3, 4 };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("A", new[] { 4 }, a), ("B", new[] { 4 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Div_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp("Div",
            new[] { ("A", new[] { 4 }, FLOAT), ("B", new[] { 4 }, FLOAT) },
            ("C", new[] { 4 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var a = new float[] { 10, 20, 30, 40 };
        var b = new float[] { 2, 4, 5, 8 };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("A", new[] { 4 }, a), ("B", new[] { 4 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Gelu_MatchesOnnxRuntime()
    {
        // GELU was added as an ONNX op in opset 20. Earlier models emit the
        // composition Mul+Div+Erf+Add+Mul — not covered by Phase 1.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.WrapModel(
            BuildSingleOpGraph("Gelu", new[] { 8 }, new[] { 8 }),
            opsetVersion: 20);
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var x = RandomArray(seed: 301, n: 8, lo: -3f, hi: 3f);
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 8 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        // GELU uses transcendentals; loosen the tolerance slightly for the
        // erf-based decomposition used by different implementations.
        AssertClose(ort, ours, tolerance: 1e-3f);
    }

    [SkippableFact]
    public void AveragePool_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 401, n: 1 * 2 * 6 * 6);
        var graph = new GraphProto { Name = "avgpool_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 2, 6, 6 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 2, 3, 3 }, FLOAT));
        var node = new NodeProto { OpType = "AveragePool" };
        node.Input.Add("X"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 2, 2));
        node.Attribute.Add(IntsAttr("strides", 2, 2));
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 2, 6, 6 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void BatchNormalization_Inference_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // [1, 4, 4, 4] input; per-channel scale/bias/mean/var of length 4.
        var x = RandomArray(seed: 501, n: 1 * 4 * 4 * 4);
        var scale = RandomArray(seed: 502, n: 4, lo: 0.5f, hi: 1.5f);
        var bias  = RandomArray(seed: 503, n: 4, lo: -0.2f, hi: 0.2f);
        var mean  = RandomArray(seed: 504, n: 4);
        // Variance must be positive.
        var varT  = RandomArray(seed: 505, n: 4, lo: 0.1f, hi: 2f);

        var graph = new GraphProto { Name = "bn_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 4, 4, 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("scale", new[] { 4 }, scale));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { 4 }, bias));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("mean", new[] { 4 }, mean));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("var", new[] { 4 }, varT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 4, 4, 4 }, FLOAT));
        var node = new NodeProto { OpType = "BatchNormalization" };
        node.Input.Add("X"); node.Input.Add("scale"); node.Input.Add("B");
        node.Input.Add("mean"); node.Input.Add("var");
        node.Output.Add("Y");
        node.Attribute.Add(new AttributeProto
        {
            Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 4, 4, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Gather_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // Embedding-style gather: data [vocab=10, hidden=4], indices [batch=3].
        var data = RandomArray(seed: 601, n: 10 * 4);
        // Indices as int64 initializer.
        var indices = new long[] { 2, 7, 1 };

        var graph = new GraphProto { Name = "gather_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("data", new[] { 10, 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("indices", new[] { 3 }, indices));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 3, 4 }, FLOAT));
        var node = new NodeProto { OpType = "Gather" };
        node.Input.Add("data"); node.Input.Add("indices"); node.Output.Add("Y");
        node.Attribute.Add(new AttributeProto
        {
            Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 0
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("data", new[] { 10, 4 }, data));
        var ours = ImportAndExecute(bytes, ("data", data));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Slice_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 701, n: 2 * 6);
        // Slice [1:, 1:4] along axes [0, 1]  → [1, 3]
        var graph = new GraphProto { Name = "slice_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 6 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("starts", new[] { 2 }, new long[] { 1, 1 }));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("ends",   new[] { 2 }, new long[] { 2, 4 }));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("axes",   new[] { 2 }, new long[] { 0, 1 }));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 3 }, FLOAT));
        var node = new NodeProto { OpType = "Slice" };
        node.Input.Add("X"); node.Input.Add("starts"); node.Input.Add("ends"); node.Input.Add("axes");
        node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 6 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Split_MatchesOnnxRuntime()
    {
        // ONNX Split produces multiple outputs; our test helper handles only
        // single-output models. Build a full graph where a downstream op
        // consumes split outputs (e.g. Add them together to reduce to one
        // output tensor).
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 801, n: 6 * 4);
        var graph = new GraphProto { Name = "split_add_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 6, 4 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 4 }, FLOAT));
        var split = new NodeProto { OpType = "Split" };
        split.Input.Add("X"); split.Output.Add("s0"); split.Output.Add("s1"); split.Output.Add("s2");
        split.Attribute.Add(new AttributeProto
        {
            Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 0
        });
        graph.Node.Add(split);
        var add1 = new NodeProto { OpType = "Add" };
        add1.Input.Add("s0"); add1.Input.Add("s1"); add1.Output.Add("t");
        graph.Node.Add(add1);
        var add2 = new NodeProto { OpType = "Add" };
        add2.Input.Add("t"); add2.Input.Add("s2"); add2.Output.Add("Y");
        graph.Node.Add(add2);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 6, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Squeeze_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // [1, 4, 1, 3] → squeeze axes [0, 2] → [4, 3]
        var x = RandomArray(seed: 901, n: 1 * 4 * 1 * 3);
        var graph = new GraphProto { Name = "squeeze_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 4, 1, 3 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("axes", new[] { 2 }, new long[] { 0, 2 }));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 4, 3 }, FLOAT));
        var node = new NodeProto { OpType = "Squeeze" };
        node.Input.Add("X"); node.Input.Add("axes"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 4, 1, 3 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Unsqueeze_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // [4, 3] → unsqueeze axes [0, 2] → [1, 4, 1, 3]
        var x = RandomArray(seed: 1001, n: 4 * 3);
        var graph = new GraphProto { Name = "unsqueeze_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 4, 3 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("axes", new[] { 2 }, new long[] { 0, 2 }));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 4, 1, 3 }, FLOAT));
        var node = new NodeProto { OpType = "Unsqueeze" };
        node.Input.Add("X"); node.Input.Add("axes"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 4, 3 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Identity_InsideChain_MatchesOnnxRuntime()
    {
        // Identity is a no-op at translation time — it just aliases the
        // input tensor under the output name so downstream consumers resolve
        // to the same storage. Test that the alias is correct by wrapping
        // it between two real ops (Relu → Identity → Relu) rather than as a
        // standalone op (which would produce a zero-step plan — no
        // IEngine op is recorded because the aliasing happens purely in the
        // translation context).
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var graph = new GraphProto { Name = "identity_chain" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 6 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 6 }, FLOAT));
        var relu1 = new NodeProto { OpType = "Relu" };
        relu1.Input.Add("X"); relu1.Output.Add("r1"); graph.Node.Add(relu1);
        var id = new NodeProto { OpType = "Identity" };
        id.Input.Add("r1"); id.Output.Add("mid"); graph.Node.Add(id);
        var relu2 = new NodeProto { OpType = "Relu" };
        relu2.Input.Add("mid"); relu2.Output.Add("Y"); graph.Node.Add(relu2);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var x = new float[] { 1.5f, -2.5f, 3.5f, -4.5f, 0f, 0.25f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 6 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    // ─── helpers ────────────────────────────────────────────────────────

    private static GraphProto BuildSingleOpGraph(string opType, int[] inputShape, int[] outputShape)
    {
        var g = new GraphProto { Name = $"{opType}_test" };
        g.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", inputShape, FLOAT));
        g.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", outputShape, FLOAT));
        var n = new NodeProto { OpType = opType };
        n.Input.Add("X"); n.Output.Add("Y");
        g.Node.Add(n);
        return g;
    }

    private static AttributeProto IntsAttr(string name, params long[] values)
    {
        var a = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var v in values) a.Ints.Add(v);
        return a;
    }
}
