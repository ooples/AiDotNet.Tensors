using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Parity tests for Phase 2 conv/pool features that landed in the Phase 1
/// PR as follow-ups: asymmetric padding, MaxPool/AveragePool ceil_mode,
/// and grouped / depthwise convolution.
/// </summary>
public class AsymmetricPadTests
{
    [SkippableFact]
    public void Conv_AsymmetricPad_MatchesOnnxRuntime()
    {
        // pads = [0, 0, 1, 1] — extra bottom + right. Common in models
        // exported from TF / Keras where SAME padding resolves unevenly.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 4101, n: 1 * 3 * 5 * 5);
        var w = RandomArray(seed: 4102, n: 4 * 3 * 3 * 3);
        var graph = new GraphProto { Name = "conv_asym_pad" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 3, 5, 5 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 4, 3, 3, 3 }, w));
        // Output: (5 + 0 + 1 - 3)/1 + 1 = 4 tall, same → [1, 4, 4, 4].
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 4, 4, 4 }, FLOAT));
        var node = new NodeProto { OpType = "Conv" };
        node.Input.Add("X"); node.Input.Add("W"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 3, 3));
        node.Attribute.Add(IntsAttr("strides", 1, 1));
        node.Attribute.Add(IntsAttr("pads", 0, 0, 1, 1));
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 3, 5, 5 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void MaxPool_CeilMode_MatchesOnnxRuntime()
    {
        // Input 7x7, kernel 3x3, stride 2, ceil_mode=1.
        // floor((7-3)/2)+1 = 3 → ceil would be ceil(4/2)+1 = 3 also, so for
        // 7x7 it's the same. Use 6x6 instead: floor((6-3)/2)+1 = 2,
        // ceil((6-3)/2)+1 = 3 — ceil adds one.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 4201, n: 1 * 2 * 6 * 6);
        var graph = new GraphProto { Name = "maxpool_ceil" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 2, 6, 6 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 2, 3, 3 }, FLOAT));
        var node = new NodeProto { OpType = "MaxPool" };
        node.Input.Add("X"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 3, 3));
        node.Attribute.Add(IntsAttr("strides", 2, 2));
        node.Attribute.Add(new AttributeProto
        {
            Name = "ceil_mode", Type = AttributeProto.Types.AttributeType.Int, I = 1
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 2, 6, 6 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void DepthwiseConv_MatchesOnnxRuntime()
    {
        // Depthwise = group == Cin. Input [1, 4, 8, 8], kernel [4, 1, 3, 3],
        // group=4 → per-channel conv, output [1, 4, 8, 8] with pad=1.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 4301, n: 1 * 4 * 8 * 8);
        var w = RandomArray(seed: 4302, n: 4 * 1 * 3 * 3);
        var graph = new GraphProto { Name = "depthwise_conv" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 4, 8, 8 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 4, 1, 3, 3 }, w));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 4, 8, 8 }, FLOAT));
        var node = new NodeProto { OpType = "Conv" };
        node.Input.Add("X"); node.Input.Add("W"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 3, 3));
        node.Attribute.Add(IntsAttr("strides", 1, 1));
        node.Attribute.Add(IntsAttr("pads", 1, 1, 1, 1));
        node.Attribute.Add(new AttributeProto
        {
            Name = "group", Type = AttributeProto.Types.AttributeType.Int, I = 4
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 4, 8, 8 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void GroupedConv_2Groups_MatchesOnnxRuntime()
    {
        // Classic grouped conv: group=2, Cin=4, Cout=6. Input [1, 4, 6, 6],
        // kernel [6, 2, 3, 3] (each output channel sees Cin/group = 2 input
        // channels), group=2 → output [1, 6, 6, 6] with pad=1.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 4401, n: 1 * 4 * 6 * 6);
        var w = RandomArray(seed: 4402, n: 6 * 2 * 3 * 3);
        var graph = new GraphProto { Name = "grouped_conv" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 4, 6, 6 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 6, 2, 3, 3 }, w));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 6, 6, 6 }, FLOAT));
        var node = new NodeProto { OpType = "Conv" };
        node.Input.Add("X"); node.Input.Add("W"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 3, 3));
        node.Attribute.Add(IntsAttr("strides", 1, 1));
        node.Attribute.Add(IntsAttr("pads", 1, 1, 1, 1));
        node.Attribute.Add(new AttributeProto
        {
            Name = "group", Type = AttributeProto.Types.AttributeType.Int, I = 2
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 4, 6, 6 }, x));
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
