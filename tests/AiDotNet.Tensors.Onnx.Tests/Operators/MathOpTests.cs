using AiDotNet.Tensors.Onnx.Protos;
using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Parity tests for math translators: Sqrt, Pow, Abs, Neg, Exp, Log, Erf,
/// ReduceSum, ReduceMean, Min, Max.
/// </summary>
public class MathOpTests
{
    [SkippableFact]
    public void Sqrt_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp("Sqrt",
            new[] { ("X", new[] { 4 }, FLOAT) }, ("Y", new[] { 4 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var x = new float[] { 0.25f, 1f, 4f, 9f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Pow_ScalarExponent_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var graph = new GraphProto { Name = "pow_scalar" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 4 }, FLOAT));
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInitializer("Y", new[] { 1 }, new[] { 3f }));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Z", new[] { 4 }, FLOAT));
        var node = new NodeProto { OpType = "Pow" };
        node.Input.Add("X"); node.Input.Add("Y"); node.Output.Add("Z");
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var x = new float[] { 0.5f, 1f, 2f, 3f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Erf_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        // The A-S approximation has max error ~1.5e-7, well inside 1e-4.
        var model = OnnxTestGraphBuilder.SingleOp("Erf",
            new[] { ("X", new[] { 9 }, FLOAT) }, ("Y", new[] { 9 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var x = new float[] { -3f, -2f, -1f, -0.5f, 0f, 0.5f, 1f, 2f, 3f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 9 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Abs_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp("Abs",
            new[] { ("X", new[] { 5 }, FLOAT) }, ("Y", new[] { 5 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var x = new float[] { -2f, -0.5f, 0f, 0.5f, 2f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 5 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void ReduceSum_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var x = RandomArray(seed: 6001, n: 12);
        var graph = new GraphProto { Name = "reducesum_test" };
        graph.Input.Add(OnnxTestGraphBuilder.MakeValueInfo("X", new[] { 3, 4 }, FLOAT));
        // opset 13+ takes axes as initializer input.
        graph.Initializer.Add(OnnxTestGraphBuilder.MakeInt64Initializer("axes", new[] { 1 }, new long[] { 1 }));
        graph.Output.Add(OnnxTestGraphBuilder.MakeValueInfo("Y", new[] { 3, 1 }, FLOAT));
        var node = new NodeProto { OpType = "ReduceSum" };
        node.Input.Add("X"); node.Input.Add("axes"); node.Output.Add("Y");
        node.Attribute.Add(new AttributeProto
        {
            Name = "keepdims", Type = AttributeProto.Types.AttributeType.Int, I = 1
        });
        graph.Node.Add(node);
        var bytes = OnnxTestGraphBuilder.Serialize(OnnxTestGraphBuilder.WrapModel(graph));
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 3, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Max_BinaryElementwise_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp("Max",
            new[] { ("A", new[] { 4 }, FLOAT), ("B", new[] { 4 }, FLOAT) },
            ("C", new[] { 4 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var a = new float[] { -1f, 0f, 1f, 2f };
        var b = new float[] { 0f, 0f, 2f, 1f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("A", new[] { 4 }, a), ("B", new[] { 4 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Min_BinaryElementwise_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp("Min",
            new[] { ("A", new[] { 4 }, FLOAT), ("B", new[] { 4 }, FLOAT) },
            ("C", new[] { 4 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var a = new float[] { -1f, 0f, 1f, 2f };
        var b = new float[] { 0f, 0f, 2f, 1f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("A", new[] { 4 }, a), ("B", new[] { 4 }, b));
        var ours = ImportAndExecute(bytes, ("A", a), ("B", b));
        AssertClose(ort, ours);
    }
}
