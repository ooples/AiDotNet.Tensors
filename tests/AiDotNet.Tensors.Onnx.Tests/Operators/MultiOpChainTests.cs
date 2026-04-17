using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// End-to-end tests that exercise a short chain of operators, proving:
/// (a) the topological sort produces a valid execution order for multi-node
/// graphs; (b) intermediate tensors flow correctly between translators.
///
/// <para>These are the simplest integration tests that stand in for the
/// Issue #169 "import a real model" acceptance criterion until a full
/// ResNet/BERT/ViT test fixture lands.</para>
/// </summary>
public class MultiOpChainTests
{
    [SkippableFact]
    public void Linear_Relu_Add_Chain_MatchesOnnxRuntime()
    {
        // Graph:  Y = Relu(X @ W + B) + C
        // Shapes: X=[2,8], W=[8,4], B=[4], C=[2,4]
        // Matches the first layer of an MLP classifier.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);

        var x = RandomArray(seed: 1001, n: 2 * 8);
        var w = RandomArray(seed: 1002, n: 8 * 4);
        var b = RandomArray(seed: 1003, n: 4);
        var c = RandomArray(seed: 1004, n: 2 * 4);

        var graph = new GraphProto { Name = "mlp_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 8 }, FLOAT));
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("C", new[] { 2, 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 8, 4 }, w));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { 4 }, b));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 4 }, FLOAT));

        var gemm = new NodeProto { OpType = "Gemm" };
        gemm.Input.Add("X"); gemm.Input.Add("W"); gemm.Input.Add("B");
        gemm.Output.Add("gemm_out");
        graph.Node.Add(gemm);

        var relu = new NodeProto { OpType = "Relu" };
        relu.Input.Add("gemm_out");
        relu.Output.Add("relu_out");
        graph.Node.Add(relu);

        var add = new NodeProto { OpType = "Add" };
        add.Input.Add("relu_out"); add.Input.Add("C");
        add.Output.Add("Y");
        graph.Node.Add(add);

        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes,
            ("X", new[] { 2, 8 }, x), ("C", new[] { 2, 4 }, c));
        var ours = ImportAndExecute(bytes, ("X", x), ("C", c));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void OutOfOrderGraph_TopoSortStillResolves()
    {
        // Construct a graph where nodes are listed in non-topological order
        // to exercise the topo-sort code path. Same math as above, but the
        // Add node is listed before the Relu node that produces its input.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);

        var x = RandomArray(seed: 2001, n: 2 * 4);
        var w = RandomArray(seed: 2002, n: 4 * 4);
        var c = RandomArray(seed: 2003, n: 2 * 4);

        var graph = new GraphProto { Name = "out_of_order_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 4 }, FLOAT));
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("C", new[] { 2, 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 4, 4 }, w));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 4 }, FLOAT));

        // Add LAST node that consumes "relu_out" FIRST in the Node list —
        // topo-sort must reorder these.
        var add = new NodeProto { OpType = "Add" };
        add.Input.Add("relu_out"); add.Input.Add("C");
        add.Output.Add("Y");
        graph.Node.Add(add);

        var relu = new NodeProto { OpType = "Relu" };
        relu.Input.Add("matmul_out");
        relu.Output.Add("relu_out");
        graph.Node.Add(relu);

        var matmul = new NodeProto { OpType = "MatMul" };
        matmul.Input.Add("X"); matmul.Input.Add("W");
        matmul.Output.Add("matmul_out");
        graph.Node.Add(matmul);

        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes,
            ("X", new[] { 2, 4 }, x), ("C", new[] { 2, 4 }, c));
        var ours = ImportAndExecute(bytes, ("X", x), ("C", c));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void UnsupportedOp_CollectsDiagnostics_NoPlan()
    {
        // Use an op we haven't registered — the importer should collect it
        // into UnsupportedOperators rather than throw (StrictMode is off by
        // default).
        var graph = new GraphProto { Name = "unsupported_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 4 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 4 }, FLOAT));
        var node = new NodeProto { OpType = "DeliberatelyMissingOp" };
        node.Input.Add("X"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        using var stream = new MemoryStream(bytes);
        var engine = new Engines.CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine);

        Assert.Null(result.Plan);
        Assert.Contains("DeliberatelyMissingOp", result.UnsupportedOperators);
    }
}
