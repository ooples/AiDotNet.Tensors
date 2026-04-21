using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Numerical-parity tests for the ONNX tensor-manipulation translators
/// (Reshape, Transpose, Concat, Slice, Flatten, Identity).
/// </summary>
public class TensorManipOperatorTests
{
    [SkippableFact]
    public void Reshape_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 501, n: 24);
        var graph = new GraphProto { Name = "reshape_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 3, 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("shape",
            new[] { 2 }, new long[] { 4, 6 }));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 4, 6 }, FLOAT));
        var node = new NodeProto { OpType = "Reshape" };
        node.Input.Add("X"); node.Input.Add("shape"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 3, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Transpose_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 601, n: 24);
        var graph = new GraphProto { Name = "transpose_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 3, 4 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 4, 3, 2 }, FLOAT));
        var node = new NodeProto { OpType = "Transpose" };
        node.Input.Add("X"); node.Output.Add("Y");
        var perm = new AttributeProto { Name = "perm", Type = AttributeProto.Types.AttributeType.Ints };
        perm.Ints.Add(2); perm.Ints.Add(1); perm.Ints.Add(0);
        node.Attribute.Add(perm);
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 3, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Concat_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var a = RandomArray(seed: 701, n: 6);
        var b = RandomArray(seed: 702, n: 6);
        var graph = new GraphProto { Name = "concat_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("A", new[] { 2, 3 }, FLOAT));
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("B", new[] { 2, 3 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("C", new[] { 4, 3 }, FLOAT));
        var node = new NodeProto { OpType = "Concat" };
        node.Input.Add("A"); node.Input.Add("B"); node.Output.Add("C");
        node.Attribute.Add(new AttributeProto
        {
            Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = 0
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes,
            ("A", new[] { 2, 3 }, a), ("B", new[] { 2, 3 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Flatten_DefaultAxis_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 801, n: 2 * 3 * 4);
        var graph = new GraphProto { Name = "flatten_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 3, 4 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 12 }, FLOAT));
        var node = new NodeProto { OpType = "Flatten" };
        node.Input.Add("X"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 3, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }
}
