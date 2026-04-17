using AiDotNet.Tensors.Onnx.Protos;
using Google.Protobuf.Collections;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Numerical-parity tests for Conv, MaxPool, AveragePool, GlobalAveragePool.
/// </summary>
public class ConvOperatorTests
{
    [SkippableFact]
    public void Conv2D_SymmetricPad_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // [1, 3, 8, 8] input, 3x3 kernel, stride 1, pad 1, 4 output channels.
        var x = RandomArray(seed: 101, n: 1 * 3 * 8 * 8);
        var w = RandomArray(seed: 102, n: 4 * 3 * 3 * 3);
        var graph = new GraphProto { Name = "conv_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 3, 8, 8 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 4, 3, 3, 3 }, w));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 4, 8, 8 }, FLOAT));
        var node = new NodeProto { OpType = "Conv" };
        node.Input.Add("X"); node.Input.Add("W"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 3, 3));
        node.Attribute.Add(IntsAttr("strides", 1, 1));
        node.Attribute.Add(IntsAttr("pads", 1, 1, 1, 1));
        node.Attribute.Add(IntsAttr("dilations", 1, 1));
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 3, 8, 8 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void MaxPool2D_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 201, n: 1 * 2 * 6 * 6);
        var graph = new GraphProto { Name = "maxpool_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 2, 6, 6 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 2, 3, 3 }, FLOAT));
        var node = new NodeProto { OpType = "MaxPool" };
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
    public void GlobalAveragePool_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 301, n: 1 * 4 * 7 * 7);
        var graph = new GraphProto { Name = "gap_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 4, 7, 7 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 4, 1, 1 }, FLOAT));
        var node = new NodeProto { OpType = "GlobalAveragePool" };
        node.Input.Add("X"); node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 4, 7, 7 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    private static AttributeProto IntsAttr(string name, params long[] values)
    {
        var a = new AttributeProto { Name = name, Type = AttributeProto.Types.AttributeType.Ints };
        foreach (var v in values) a.Ints.Add(v);
        return a;
    }
}
