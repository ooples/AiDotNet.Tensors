using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Parity tests for the Phase 2 quantized operator translators. Our
/// importer represents quantized tensors as floats holding integer-valued
/// quantized values (clamped to [-128, 127]); ORT uses a native int8
/// element type, so we assert equality only after its dequantize pass
/// produces float outputs — which is what the end-to-end model flow does
/// anyway.
/// </summary>
public class QuantizedOpTests
{
    [SkippableFact]
    public void DequantizeLinear_Scalar_MatchesOnnxRuntime()
    {
        // x is an int8 tensor but serialized as float so our plan can consume
        // it without a separate int8 path. ORT reads it as int8 and
        // dequantizes to float. Values are identical after dequant.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        const int FLOAT = 1, INT8 = 3;
        var x = new sbyte[] { -20, -10, 0, 10, 20, 30, 40, 50 };
        float scale = 0.25f;
        sbyte zp = 5;
        // ORT needs an int8 input tensor; build it directly in the graph
        // as an initializer (no graph input — graph just dequantizes the
        // initializer and emits the float output).
        var graph = new GraphProto { Name = "dequant_test" };
        // int8 initializer for x
        var xInit = new TensorProto { Name = "x", DataType = INT8 };
        xInit.Dims.Add(8);
        xInit.RawData = Google.Protobuf.ByteString.CopyFrom(Array.ConvertAll(x, s => unchecked((byte)s)));
        graph.Initializer.Add(xInit);
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("scale", new[] { 1 }, new[] { scale }));
        // Zero-point is int8 too.
        var zpInit = new TensorProto { Name = "zp", DataType = INT8 };
        zpInit.Dims.Add(1);
        zpInit.RawData = Google.Protobuf.ByteString.CopyFrom(new byte[] { unchecked((byte)zp) });
        graph.Initializer.Add(zpInit);
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 8 }, FLOAT));
        var node = new NodeProto { OpType = "DequantizeLinear" };
        node.Input.Add("x"); node.Input.Add("scale"); node.Input.Add("zp");
        node.Output.Add("Y");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes);
        // Our import: no graph inputs to feed (everything is in initializers).
        using var stream = new MemoryStream(bytes);
        var engine = new Engines.CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        Assert.NotNull(result.Plan);
        var output = result.Plan!.Execute();
        var ours = new float[output.AsSpan().Length];
        output.AsSpan().CopyTo(ours);
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void QuantizeLinear_RoundTrip_MatchesOnnxRuntime()
    {
        // Build a float input, quantize, dequantize, compare. Tests both
        // QuantizeLinear and DequantizeLinear translators composed.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        const int FLOAT = 1, INT8 = 3;
        var x = new float[] { -2.1f, -1.0f, 0f, 0.5f, 1.0f, 2.5f, 4.0f, 7.0f };

        var graph = new GraphProto { Name = "qdq_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 8 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("scale", new[] { 1 }, new[] { 0.5f }));
        var zpInit = new TensorProto { Name = "zp", DataType = INT8 };
        zpInit.Dims.Add(1);
        zpInit.RawData = Google.Protobuf.ByteString.CopyFrom(new byte[] { unchecked((byte)0) });
        graph.Initializer.Add(zpInit);
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 8 }, FLOAT));

        var q = new NodeProto { OpType = "QuantizeLinear" };
        q.Input.Add("X"); q.Input.Add("scale"); q.Input.Add("zp");
        q.Output.Add("Xq");
        graph.Node.Add(q);

        var d = new NodeProto { OpType = "DequantizeLinear" };
        d.Input.Add("Xq"); d.Input.Add("scale"); d.Input.Add("zp");
        d.Output.Add("Y");
        graph.Node.Add(d);

        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 8 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }
}
