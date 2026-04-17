using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Parity tests for ONNX LSTM and GRU — Phase 2 follow-ups from Issue #169.
/// Unrolled forward-direction unidirectional variants without peepholes,
/// sequence_lens, or custom activations.
/// </summary>
public class RecurrentOpTests
{
    [SkippableFact]
    public void Lstm_Forward_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // seq=3, batch=2, input=4, hidden=5, direction=forward.
        const int seq = 3, batch = 2, input = 4, hidden = 5;
        var x = RandomArray(seed: 5001, n: seq * batch * input);
        // W is [1, 4H, input]. Gate order ONNX: [i, o, f, c].
        var w = RandomArray(seed: 5002, n: 1 * 4 * hidden * input);
        // R is [1, 4H, H].
        var r = RandomArray(seed: 5003, n: 1 * 4 * hidden * hidden);
        // B is [1, 8H] = [Wb_i, Wb_o, Wb_f, Wb_c, Rb_i, Rb_o, Rb_f, Rb_c].
        var b = RandomArray(seed: 5004, n: 1 * 8 * hidden);

        var graph = new GraphProto { Name = "lstm_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { seq, batch, input }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 1, 4 * hidden, input }, w));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("R", new[] { 1, 4 * hidden, hidden }, r));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { 1, 8 * hidden }, b));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { seq, 1, batch, hidden }, FLOAT));

        var node = new NodeProto { OpType = "LSTM" };
        node.Input.Add("X"); node.Input.Add("W"); node.Input.Add("R"); node.Input.Add("B");
        node.Output.Add("Y");
        node.Attribute.Add(new AttributeProto
        {
            Name = "hidden_size", Type = AttributeProto.Types.AttributeType.Int, I = hidden
        });
        node.Attribute.Add(new AttributeProto
        {
            Name = "direction", Type = AttributeProto.Types.AttributeType.String,
            S = Google.Protobuf.ByteString.CopyFromUtf8("forward"),
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { seq, batch, input }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        // LSTMs accumulate nonlinearities over timesteps; widen tolerance
        // slightly for the transcendental composition.
        AssertClose(ort, ours, tolerance: 1e-3f);
    }

    [SkippableFact]
    public void Gru_Forward_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        const int seq = 3, batch = 2, input = 4, hidden = 5;
        var x = RandomArray(seed: 5101, n: seq * batch * input);
        var w = RandomArray(seed: 5102, n: 1 * 3 * hidden * input);
        var r = RandomArray(seed: 5103, n: 1 * 3 * hidden * hidden);
        var b = RandomArray(seed: 5104, n: 1 * 6 * hidden);

        var graph = new GraphProto { Name = "gru_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { seq, batch, input }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 1, 3 * hidden, input }, w));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("R", new[] { 1, 3 * hidden, hidden }, r));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { 1, 6 * hidden }, b));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { seq, 1, batch, hidden }, FLOAT));

        var node = new NodeProto { OpType = "GRU" };
        node.Input.Add("X"); node.Input.Add("W"); node.Input.Add("R"); node.Input.Add("B");
        node.Output.Add("Y");
        node.Attribute.Add(new AttributeProto
        {
            Name = "hidden_size", Type = AttributeProto.Types.AttributeType.Int, I = hidden
        });
        node.Attribute.Add(new AttributeProto
        {
            Name = "direction", Type = AttributeProto.Types.AttributeType.String,
            S = Google.Protobuf.ByteString.CopyFromUtf8("forward"),
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { seq, batch, input }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours, tolerance: 1e-3f);
    }
}
