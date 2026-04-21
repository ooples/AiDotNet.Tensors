using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Parity tests for ConvTranspose (transposed 2D convolution). Appears in
/// image-generation decoders (VAE, upsampling heads) and some segmentation
/// models.
/// </summary>
public class ConvTransposeTests
{
    [SkippableFact]
    public void ConvTranspose2D_Stride2_Pad0_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // Input  [1, 3, 4, 4], kernel [3, 5, 3, 3] (in=3, out=5), stride=2, pad=0.
        // Output: (4-1)*2 + 3 = 9 in each spatial dim → [1, 5, 9, 9].
        var x = RandomArray(seed: 2001, n: 1 * 3 * 4 * 4);
        var w = RandomArray(seed: 2002, n: 3 * 5 * 3 * 3);

        var graph = new GraphProto { Name = "convtranspose_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 3, 4, 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 3, 5, 3, 3 }, w));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 5, 9, 9 }, FLOAT));
        var node = new NodeProto { OpType = "ConvTranspose" };
        node.Input.Add("X"); node.Input.Add("W"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 3, 3));
        node.Attribute.Add(IntsAttr("strides", 2, 2));
        node.Attribute.Add(IntsAttr("pads", 0, 0, 0, 0));
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 3, 4, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Conv3D_MatchesOnnxRuntime()
    {
        // 3D conv: input [1, 2, 4, 5, 5] (NCDHW), kernel [3, 2, 2, 3, 3]
        // stride 1, pad 0. Output: [1, 3, 3, 3, 3].
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 5201, n: 1 * 2 * 4 * 5 * 5);
        var w = RandomArray(seed: 5202, n: 3 * 2 * 2 * 3 * 3);

        var graph = new GraphProto { Name = "conv3d_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 2, 4, 5, 5 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 3, 2, 2, 3, 3 }, w));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 3, 3, 3, 3 }, FLOAT));
        var node = new NodeProto { OpType = "Conv" };
        node.Input.Add("X"); node.Input.Add("W"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 2, 3, 3));
        node.Attribute.Add(IntsAttr("strides", 1, 1, 1));
        node.Attribute.Add(IntsAttr("pads", 0, 0, 0, 0, 0, 0));
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 2, 4, 5, 5 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void MaxPool3D_MatchesOnnxRuntime()
    {
        // 3D pool: input [1, 2, 4, 4, 4], kernel 2×2×2, stride 2. Output [1, 2, 2, 2, 2].
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 5301, n: 1 * 2 * 4 * 4 * 4);
        var graph = new GraphProto { Name = "maxpool3d_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 1, 2, 4, 4, 4 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 1, 2, 2, 2, 2 }, FLOAT));
        var node = new NodeProto { OpType = "MaxPool" };
        node.Input.Add("X"); node.Output.Add("Y");
        node.Attribute.Add(IntsAttr("kernel_shape", 2, 2, 2));
        node.Attribute.Add(IntsAttr("strides", 2, 2, 2));
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 1, 2, 4, 4, 4 }, x));
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
