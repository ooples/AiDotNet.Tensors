using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Accuracy of <c>TensorLog</c> on <c>double</c>, which routes through the AVX2 kernel
/// <c>SimdKernels.LogUnsafe</c> -> <c>FastLogDouble256</c>.
/// </summary>
/// <remarks>
/// <para>
/// That kernel's own documentation states "Relative error ~1e-14 across the normal double range".
/// This test holds it to that claim.
/// </para>
/// <para>
/// Found via the differentiable-ops gradcheck sweep: TensorLog's BACKWARD is exactly 1/x, yet
/// finite differences through its FORWARD disagreed by ~2%. The backward was right and the forward
/// was wrong — the approximation error varies with x, so its slope corrupts any numerically
/// estimated derivative, and more importantly corrupts log-domain values directly (cross-entropy
/// and any other loss taking log of a probability).
/// </para>
/// <para>
/// Note the length dependence: the 16-wide block needs length >= 16 and the 4-wide needs
/// length - i >= 4, so a tensor of 6 elements takes the approximate path for elements 0-3 and the
/// accurate scalar tail for 4-5. Accuracy therefore varies WITHIN a single tensor, which is why
/// this reproduces at tiny sizes.
/// </para>
/// </remarks>
public class FastLogDoubleAccuracyTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public FastLogDoubleAccuracyTests(ITestOutputHelper output) => _out = output;

    [Theory]
    [InlineData(4)]    // exactly the 4-wide AVX2 path
    [InlineData(6)]    // 4 vectorized + 2 scalar tail
    [InlineData(64)]   // the 16-wide unrolled path
    public void TensorLog_Double_MatchesMathLog(int length)
    {
        var rng = new Random(20260730);
        var x = new Tensor<double>([length]);
        for (int i = 0; i < length; i++) x[i] = 0.05 + rng.NextDouble() * 20.0;

        var got = _engine.TensorLog(x);

        double worstRel = 0;
        int worstIdx = -1;
        for (int i = 0; i < length; i++)
        {
            double expected = Math.Log(x[i]);
            double rel = Math.Abs(got[i] - expected) / Math.Max(1e-300, Math.Abs(expected));
            if (rel > worstRel) { worstRel = rel; worstIdx = i; }
        }

        _out.WriteLine($"length={length} worst relative error {worstRel:E3} at index {worstIdx} " +
                       $"(x={x[worstIdx]:G17}, got={got[worstIdx]:G17}, Math.Log={Math.Log(x[worstIdx]):G17})");

        // Hold the kernel to its documented accuracy. 1e-12 is already three orders looser than
        // the "~1e-14" the implementation claims.
        Assert.True(worstRel < 1e-12,
            $"TensorLog<double> worst relative error {worstRel:E3} at index {worstIdx} exceeds 1e-12. " +
            $"FastLogDouble256 documents \"Relative error ~1e-14 across the normal double range\", so it is " +
            $"not meeting its own contract. x={x[worstIdx]:G17} got={got[worstIdx]:G17} " +
            $"expected={Math.Log(x[worstIdx]):G17}");
    }

    /// <summary>
    /// Accuracy must not depend on tensor length. Same value, different surrounding length, should
    /// give the same log — otherwise results silently change with batch size.
    /// </summary>
    [Fact]
    public void TensorLog_Double_IsLengthIndependent()
    {
        const double probe = 0.58944858761478613;
        var results = new double[3];
        var lengths = new[] { 2, 6, 32 };

        for (int k = 0; k < lengths.Length; k++)
        {
            var t = new Tensor<double>([lengths[k]]);
            for (int i = 0; i < t.Length; i++) t[i] = probe;
            results[k] = _engine.TensorLog(t)[0];
            _out.WriteLine($"length={lengths[k],3}: log({probe})={results[k]:G17}  " +
                           $"Math.Log={Math.Log(probe):G17}");
        }

        Assert.True(Math.Abs(results[0] - results[1]) < 1e-12 && Math.Abs(results[1] - results[2]) < 1e-12,
            $"TensorLog<double> of the same value differs by tensor length: " +
            $"len2={results[0]:G17} len6={results[1]:G17} len32={results[2]:G17}. The scalar tail is accurate " +
            $"while the vectorized body is not, so a value's log depends on how many neighbours it has.");
    }
}
