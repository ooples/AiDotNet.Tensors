using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Regression tests for the SIMD double activation kernels overflowing to NaN
/// on large-magnitude inputs. The tanh-via-exponential decomposition
/// tanh(z) = (e^{2z} − 1)/(e^{2z} + 1) drove FastExpDouble256's 2^n
/// reconstruction into the biased-exponent-2047 (Inf) pattern for z ≥ ~355,
/// making the division Inf/Inf = NaN. For GELU's tanh argument
/// z = √(2/π)·(x + 0.044715·x³) that threshold is reached at x ≈ 19.8 — well
/// inside the range a single optimizer step can push transformer
/// pre-activations (AiDotNet ACEStepTests: one AdamW step on the paper-scale
/// stack produced pre-activations of ~21 and every tape-mode forward after it
/// returned NaN). The scalar tails use Math.Tanh and were always safe, which
/// is why only SIMD-lane elements were poisoned.
///
/// The kernels now clamp the tanh argument to ±20 (tanh(±20) rounds to
/// exactly ±1.0 in double, so the clamp is bit-exact, not an approximation).
/// These tests pin the contract: large inputs through the TENSOR kernels
/// (which take the SIMD path for ≥4 elements) must stay finite and match the
/// scalar reference.
/// </summary>
public class ActivationLargeInputOverflowTests
{
    private readonly CpuEngine _engine = new();

    private static double GeluReference(double x)
    {
        double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
        return 0.5 * x * (1.0 + Math.Tanh(inner));
    }

    [Theory]
    [InlineData(19.0)]   // just below the historical overflow threshold
    [InlineData(21.66)]  // the exact pre-activation magnitude from the ACEStep repro
    [InlineData(50.0)]
    [InlineData(400.0)]
    [InlineData(-21.66)]
    [InlineData(-400.0)]
    public void GELU_Double_LargeInputs_AreFiniteAndMatchScalarReference(double x)
    {
        // 8 identical elements: enough to force the Vector256 SIMD path
        // (length >= 4) with no scalar-tail masking of the result.
        var input = new Tensor<double>(new[] { 8 });
        for (int i = 0; i < 8; i++) input[i] = x;

        var result = _engine.GELU(input);

        double expected = GeluReference(x);
        for (int i = 0; i < 8; i++)
        {
            Assert.False(double.IsNaN(result[i]) || double.IsInfinity(result[i]),
                $"GELU({x}) produced non-finite value {result[i]} at lane {i}.");
            Assert.Equal(expected, result[i], 10);
        }
    }

    [Theory]
    [InlineData(25.0)]
    [InlineData(400.0)]
    [InlineData(-400.0)]
    public void Tanh_Double_LargeInputs_AreFiniteAndSaturate(double x)
    {
        // 20 elements: covers both the 16-wide unrolled SIMD block and the
        // 4-wide block of the double Tanh kernel.
        var input = new Tensor<double>(new[] { 20 });
        for (int i = 0; i < 20; i++) input[i] = x;

        var result = _engine.Tanh(input);

        double expected = Math.Tanh(x); // ±1.0 at these magnitudes
        for (int i = 0; i < 20; i++)
        {
            Assert.False(double.IsNaN(result[i]) || double.IsInfinity(result[i]),
                $"Tanh({x}) produced non-finite value {result[i]} at lane {i}.");
            Assert.Equal(expected, result[i], 12);
        }
    }

    [Theory]
    [InlineData(25.0)]
    [InlineData(400.0)]
    [InlineData(800.0)]   // also overflows the softplus-internal exp(x) without the cap
    [InlineData(-400.0)]
    public void Mish_Double_LargeInputs_AreFiniteAndMatchScalarReference(double x)
    {
        var input = new Tensor<double>(new[] { 8 });
        for (int i = 0; i < 8; i++) input[i] = x;

        var result = _engine.Mish(input);

        double softplus = x > 20.0 ? x : Math.Log(1.0 + Math.Exp(x));
        double expected = x * Math.Tanh(softplus);
        for (int i = 0; i < 8; i++)
        {
            Assert.False(double.IsNaN(result[i]) || double.IsInfinity(result[i]),
                $"Mish({x}) produced non-finite value {result[i]} at lane {i}.");
            Assert.Equal(expected, result[i], 8);
        }
    }

    [Fact]
    public void GELU_Double_MixedMagnitudes_OnlyLargeLanesWereAffected_AllNowCorrect()
    {
        // Mirrors the ACEStep failure shape: a vector with a mix of ordinary
        // and large pre-activations — historically the large lanes NaN'd while
        // their neighbors stayed correct, corrupting the whole forward.
        double[] values = { -2.5, 0.0, 1.3, 21.66, -21.66, 3.7, 19.9, 355.0 };
        var input = new Tensor<double>(new[] { values.Length });
        for (int i = 0; i < values.Length; i++) input[i] = values[i];

        var result = _engine.GELU(input);

        for (int i = 0; i < values.Length; i++)
        {
            Assert.False(double.IsNaN(result[i]) || double.IsInfinity(result[i]),
                $"GELU({values[i]}) produced non-finite value at lane {i}.");
            Assert.Equal(GeluReference(values[i]), result[i], 10);
        }
    }
}
