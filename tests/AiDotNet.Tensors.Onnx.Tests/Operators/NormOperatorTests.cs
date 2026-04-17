using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Numerical-parity tests for LayerNormalization and BatchNormalization.
/// LayerNorm is the workhorse for transformer (BERT/ViT); BatchNorm inference
/// is the ResNet path.
/// </summary>
public class NormOperatorTests
{
    [SkippableFact]
    public void LayerNormalization_LastAxis_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // Transformer-shaped input [2, 4, 8] — batch=2, seq=4, hidden=8.
        var x = RandomArray(seed: 901, n: 2 * 4 * 8);
        var scale = RandomArray(seed: 902, n: 8, lo: 0.5f, hi: 1.5f);
        var bias = RandomArray(seed: 903, n: 8, lo: -0.2f, hi: 0.2f);

        var graph = new GraphProto { Name = "layernorm_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 4, 8 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("scale", new[] { 8 }, scale));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { 8 }, bias));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 4, 8 }, FLOAT));
        var node = new NodeProto { OpType = "LayerNormalization" };
        node.Input.Add("X"); node.Input.Add("scale"); node.Input.Add("B");
        node.Output.Add("Y");
        node.Attribute.Add(new AttributeProto
        {
            Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1
        });
        node.Attribute.Add(new AttributeProto
        {
            Name = "epsilon", Type = AttributeProto.Types.AttributeType.Float, F = 1e-5f
        });
        graph.Node.Add(node);
        // LayerNormalization was added in opset 17; ORT accepts it at opset 17+.
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph, opsetVersion: 17));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 4, 8 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }
}
