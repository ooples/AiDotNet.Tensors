using Xunit;
using static AiDotNet.Tensors.Onnx.Tests.OnnxTestHelpers;

namespace AiDotNet.Tensors.Onnx.Tests.Operators;

/// <summary>
/// Numerical-parity tests for activation translators (Relu, Sigmoid, Tanh,
/// Softmax, LeakyRelu). GELU has a separate test because ONNX Runtime's
/// default Gelu approximation is exact (erf-based) and so is ours.
/// </summary>
public class ActivationOperatorTests
{
    [SkippableTheory]
    [InlineData("Relu")]
    [InlineData("Sigmoid")]
    [InlineData("Tanh")]
    public void UnaryActivations_MatchOnnxRuntime(string opType)
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: opType,
            inputs: new[] { ("X", new[] { 2, 8 }, FLOAT) },
            output: ("Y", new[] { 2, 8 }, FLOAT));
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var x = RandomArray(seed: 11, n: 16, lo: -3f, hi: 3f);
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 8 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void Softmax_LastAxis_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "Softmax",
            inputs: new[] { ("X", new[] { 2, 4 }, FLOAT) },
            output: ("Y", new[] { 2, 4 }, FLOAT),
            attributes: new[]
            {
                new Protos.AttributeProto
                {
                    Name = "axis",
                    Type = Protos.AttributeProto.Types.AttributeType.Int,
                    I = -1,
                }
            });
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var x = RandomArray(seed: 22, n: 8);
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 2, 4 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }

    [SkippableFact]
    public void LeakyRelu_MatchesOnnxRuntime()
    {
        Skip.IfNot(OnnxRuntimeReference.IsOrtAvailable);
        var model = OnnxTestGraphBuilder.SingleOp(
            opType: "LeakyRelu",
            inputs: new[] { ("X", new[] { 6 }, FLOAT) },
            output: ("Y", new[] { 6 }, FLOAT),
            attributes: new[]
            {
                new Protos.AttributeProto
                {
                    Name = "alpha",
                    Type = Protos.AttributeProto.Types.AttributeType.Float,
                    F = 0.2f,
                }
            });
        var bytes = OnnxTestGraphBuilder.Serialize(model);
        var x = new float[] { -2f, -1f, -0.5f, 0f, 1f, 2f };
        var ort = OnnxRuntimeReference.RunSingleOutput(bytes, ("X", new[] { 6 }, x));
        var ours = ImportAndExecute(bytes, ("X", x));
        AssertClose(ort, ours);
    }
}
