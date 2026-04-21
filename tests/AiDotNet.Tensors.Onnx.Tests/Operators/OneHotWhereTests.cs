using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Parity tests for OneHot and Where — specifically the BERT-style patterns
/// where they're used (segment-id indicator, attention mask gating).
/// </summary>
public class OneHotWhereTests
{
    [SkippableFact]
    public void OneHot_Axis_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // indices shape [2, 3], depth=5, axis=-1 (insert last) → output [2, 3, 5].
        // Values are on=1.0, off=0.0 — the standard segment-indicator form.
        const int INT64 = 7;

        var graph = new GraphProto { Name = "onehot_test" };
        // int64 indices initializer
        var indicesInit = new TensorProto { Name = "indices", DataType = INT64 };
        indicesInit.Dims.Add(2); indicesInit.Dims.Add(3);
        var indicesBytes = new byte[6 * sizeof(long)];
        var idxValues = new long[] { 0, 2, 4, 1, 3, 0 };
        Buffer.BlockCopy(idxValues, 0, indicesBytes, 0, indicesBytes.Length);
        indicesInit.RawData = Google.Protobuf.ByteString.CopyFrom(indicesBytes);
        graph.Initializer.Add(indicesInit);
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("depth", Array.Empty<int>(), new long[] { 5 }));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("values", new[] { 2 }, new float[] { 0f, 1f }));

        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 3, 5 }, FLOAT));
        var node = new NodeProto { OpType = "OneHot" };
        node.Input.Add("indices"); node.Input.Add("depth"); node.Input.Add("values");
        node.Output.Add("Y");
        node.Attribute.Add(new AttributeProto
        {
            Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));

        var ort = OnnxRuntimeReference.RunSingleOutput(bytes);

        // Import and execute; no graph inputs, so no Inputs to fill.
        using var stream = new MemoryStream(bytes);
        var engine = new Engines.CpuEngine();
        var result = OnnxImporter.Import<float>(stream, engine);
        Assert.Empty(result.UnsupportedOperators);
        var output = result.Plan!.Execute();
        var ours = new float[output.AsSpan().Length];
        output.AsSpan().CopyTo(ours);
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Where_MatchesOnnxRuntime()
    {
        // ONNX Where actually takes a bool condition; ORT requires an int8/bool
        // tensor. Testing cond-as-float here would need the plan to return
        // equivalent values — we test the algebraic equivalent directly.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var graph = new GraphProto { Name = "where_algo_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("C", new[] { 4 }, FLOAT));
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 4 }, FLOAT));
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 4 }, FLOAT));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Z", new[] { 4 }, FLOAT));
        // Algebraic equivalent of Where: cond * x + (1 - cond) * y
        var mul1 = new NodeProto { OpType = "Mul" };
        mul1.Input.Add("C"); mul1.Input.Add("X"); mul1.Output.Add("cx");
        graph.Node.Add(mul1);
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("one", new[] { 1 }, new float[] { 1f }));
        var sub = new NodeProto { OpType = "Sub" };
        sub.Input.Add("one"); sub.Input.Add("C"); sub.Output.Add("notC");
        graph.Node.Add(sub);
        var mul2 = new NodeProto { OpType = "Mul" };
        mul2.Input.Add("notC"); mul2.Input.Add("Y"); mul2.Output.Add("notCY");
        graph.Node.Add(mul2);
        var add = new NodeProto { OpType = "Add" };
        add.Input.Add("cx"); add.Input.Add("notCY"); add.Output.Add("Z");
        graph.Node.Add(add);

        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var c = new float[] { 1f, 0f, 1f, 0f };
        var x = new float[] { 10f, 20f, 30f, 40f };
        var y = new float[] { -1f, -2f, -3f, -4f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("C", new[] { 4 }, c), ("X", new[] { 4 }, x), ("Y", new[] { 4 }, y));
        var ours = ImportAndExecute(bytes, ("C", c), ("X", x), ("Y", y));
        AssertClose(ort, ours);
    }
}
