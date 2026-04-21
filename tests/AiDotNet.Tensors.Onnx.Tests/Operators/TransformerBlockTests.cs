using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Tiny transformer-block parity tests — the fan-in pattern BERT / ViT
/// use everywhere: input → LayerNorm → MatMul (QKV projection) → MatMul
/// → Add → LayerNorm → MatMul → Gemm → residual Add.
/// These exercise the fan-out (one LayerNorm feeding three MatMuls for
/// Q/K/V) and residual-add patterns that aren't covered by the
/// single-chain MultiOpChainTests.
/// </summary>
public class TransformerBlockTests
{
    [SkippableFact]
    public void LayerNorm_FanOut_ThreeMatMuls_MatchesOnnxRuntime()
    {
        // Models the QKV projection pattern: normalize input, then project
        // three times (Q, K, V) with separate weight matrices, then sum the
        // three projections. Catches bugs where a single op's output feeds
        // multiple downstream consumers — the LazyNode's LazySource must
        // survive DCE when any of its consumers are kept.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        const int batch = 1, seq = 4, hidden = 8;
        var x = RandomArray(seed: 3001, n: batch * seq * hidden);
        var wq = RandomArray(seed: 3002, n: hidden * hidden);
        var wk = RandomArray(seed: 3003, n: hidden * hidden);
        var wv = RandomArray(seed: 3004, n: hidden * hidden);
        var scale = RandomArray(seed: 3005, n: hidden, lo: 0.5f, hi: 1.5f);
        var bias = RandomArray(seed: 3006, n: hidden, lo: -0.2f, hi: 0.2f);

        var graph = new GraphProto { Name = "fanout_layernorm" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X",
            new[] { batch, seq, hidden }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("scale", new[] { hidden }, scale));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("bias", new[] { hidden }, bias));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("Wq", new[] { hidden, hidden }, wq));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("Wk", new[] { hidden, hidden }, wk));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("Wv", new[] { hidden, hidden }, wv));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y",
            new[] { batch, seq, hidden }, FLOAT));

        // LayerNorm (last axis)
        var ln = new NodeProto { OpType = "LayerNormalization" };
        ln.Input.Add("X"); ln.Input.Add("scale"); ln.Input.Add("bias");
        ln.Output.Add("normed");
        ln.Attribute.Add(new AttributeProto
        {
            Name = "axis", Type = AttributeProto.Types.AttributeType.Int, I = -1
        });
        graph.Node.Add(ln);

        // Q = normed @ Wq
        var matQ = new NodeProto { OpType = "MatMul" };
        matQ.Input.Add("normed"); matQ.Input.Add("Wq"); matQ.Output.Add("q");
        graph.Node.Add(matQ);

        // K = normed @ Wk (SAME normed feeding another matmul — fan-out)
        var matK = new NodeProto { OpType = "MatMul" };
        matK.Input.Add("normed"); matK.Input.Add("Wk"); matK.Output.Add("k");
        graph.Node.Add(matK);

        // V = normed @ Wv (third fan-out consumer)
        var matV = new NodeProto { OpType = "MatMul" };
        matV.Input.Add("normed"); matV.Input.Add("Wv"); matV.Output.Add("v");
        graph.Node.Add(matV);

        // sum_qk = q + k
        var addQK = new NodeProto { OpType = "Add" };
        addQK.Input.Add("q"); addQK.Input.Add("k"); addQK.Output.Add("qk");
        graph.Node.Add(addQK);

        // Y = qk + v
        var addFinal = new NodeProto { OpType = "Add" };
        addFinal.Input.Add("qk"); addFinal.Input.Add("v"); addFinal.Output.Add("Y");
        graph.Node.Add(addFinal);

        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph, opsetVersion: 17));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { batch, seq, hidden }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void ResidualAdd_AroundLinear_MatchesOnnxRuntime()
    {
        // Transformer residual: Y = X + Linear(X). Tests the fan-out pattern
        // where input X feeds both the MatMul branch AND the final add.
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 4001, n: 2 * 4);
        var w = RandomArray(seed: 4002, n: 4 * 4);
        var b = RandomArray(seed: 4003, n: 4);

        var graph = new GraphProto { Name = "residual_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 2, 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("W", new[] { 4, 4 }, w));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("B", new[] { 4 }, b));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 2, 4 }, FLOAT));

        var gemm = new NodeProto { OpType = "Gemm" };
        gemm.Input.Add("X"); gemm.Input.Add("W"); gemm.Input.Add("B");
        gemm.Output.Add("lin");
        graph.Node.Add(gemm);

        var add = new NodeProto { OpType = "Add" };
        add.Input.Add("X"); add.Input.Add("lin"); add.Output.Add("Y");
        graph.Node.Add(add);

        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }
}
