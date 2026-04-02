using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Verifies that fused Linear+Activation ops produce identical gradients
/// to the equivalent unfused (separate MatMul + Add + Activation) ops.
/// </summary>
public class FusedLinearGradientTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var tensor = new Tensor<float>(shape);
        var span = tensor.AsWritableSpan();
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return tensor;
    }

    private void VerifyFusedMatchesUnfused(
        Func<IEngine, Tensor<float>, Tensor<float>, Tensor<float>, Tensor<float>> fusedOp,
        Func<IEngine, Tensor<float>, Tensor<float>> activationOp,
        int batchSize = 4, int inFeatures = 8, int outFeatures = 6)
    {
        var input = CreateRandom([batchSize, inFeatures], 42);
        var weight = CreateRandom([inFeatures, outFeatures], 43);
        var bias = CreateRandom([outFeatures], 44);

        // Unfused: separate ops
        Tensor<float> unfusedOutput;
        Dictionary<Tensor<float>, Tensor<float>> unfusedGrads;
        using (var tape = new GradientTape<float>())
        {
            var linear = _engine.TensorMatMul(input, weight);
            var biased = _engine.TensorBroadcastAdd(linear, bias);
            unfusedOutput = activationOp(_engine, biased);
            var loss = _engine.ReduceSum(unfusedOutput, [0, 1], keepDims: false);
            unfusedGrads = tape.ComputeGradients(loss, [input, weight, bias]);
        }

        // Fused: single op
        Tensor<float> fusedOutput;
        Dictionary<Tensor<float>, Tensor<float>> fusedGrads;
        using (var tape = new GradientTape<float>())
        {
            fusedOutput = fusedOp(_engine, input, weight, bias);
            var loss = _engine.ReduceSum(fusedOutput, [0, 1], keepDims: false);
            fusedGrads = tape.ComputeGradients(loss, [input, weight, bias]);
        }

        // Verify forward output matches
        Assert.Equal(unfusedOutput.Length, fusedOutput.Length);
        for (int i = 0; i < unfusedOutput.Length; i++)
            Assert.Equal(unfusedOutput[i], fusedOutput[i], 5);

        // Verify gradients match for each parameter
        foreach (var param in new[] { input, weight, bias })
        {
            Assert.True(unfusedGrads.ContainsKey(param), $"Unfused gradient missing for parameter");
            Assert.True(fusedGrads.ContainsKey(param), $"Fused gradient missing for parameter");

            var unfusedGrad = unfusedGrads[param];
            var fusedGrad = fusedGrads[param];
            Assert.Equal(unfusedGrad.Length, fusedGrad.Length);

            for (int i = 0; i < unfusedGrad.Length; i++)
                Assert.Equal(unfusedGrad[i], fusedGrad[i], 4);
        }
    }

    [Fact]
    public void FusedLinearReLU_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearReLU(i, w, b),
            (e, x) => e.ReLU(x));
    }

    [Fact]
    public void FusedLinearSigmoid_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearSigmoid(i, w, b),
            (e, x) => e.TensorSigmoid(x));
    }

    [Fact]
    public void FusedLinearTanh_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearTanh(i, w, b),
            (e, x) => e.Tanh(x));
    }

    [Fact]
    public void FusedLinearGELU_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearGELU(i, w, b),
            (e, x) => e.GELU(x));
    }

    [Fact]
    public void FusedLinearSwish_ProducesIdenticalGradients()
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearSwish(i, w, b),
            (e, x) => e.Swish(x));
    }

    [Theory]
    [InlineData(1, 4, 3)]
    [InlineData(8, 16, 12)]
    [InlineData(16, 32, 8)]
    public void FusedLinearReLU_DifferentSizes_ProducesIdenticalGradients(int batch, int inF, int outF)
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearReLU(i, w, b),
            (e, x) => e.ReLU(x), batch, inF, outF);
    }

    [Theory]
    [InlineData(1, 4, 3)]
    [InlineData(8, 16, 12)]
    public void FusedLinearGELU_DifferentSizes_ProducesIdenticalGradients(int batch, int inF, int outF)
    {
        VerifyFusedMatchesUnfused(
            (e, i, w, b) => e.FusedLinearGELU(i, w, b),
            (e, x) => e.GELU(x), batch, inF, outF);
    }
}
