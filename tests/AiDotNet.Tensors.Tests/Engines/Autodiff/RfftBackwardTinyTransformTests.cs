using System;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression guards for #778 / AiDotNet#1856: the RFFT autodiff backward (which runs
/// <see cref="IEngine.IRFFT{T}"/>) threw <see cref="ArgumentOutOfRangeException"/> for a
/// length-1 transform axis. A length-1 forward RFFT pads to nFft = NextPowerOf2(1) = 1,
/// which is ODD, so numFreqs collapses to 1 and the even-only inverse formula
/// <c>(numFreqs-1)*2</c> produced nFft = 0 — an empty inverse whose output copy indexed
/// out of range. IRFFT now clamps nFft up to the caller-supplied outputLength, so the
/// single DC bin (which IS the signal for n = 1) round-trips exactly.
///
/// <para>For n = 1 every quantity is analytic, so these assert exact values rather than
/// merely "does not throw": RFFT([x]) = [x, 0] (DC bin only), IRFFT([re, im], 1) = [re]
/// (scale 1/nFft = 1), and d(Σ RFFT(x))/dx = 1 elementwise (the always-zero imaginary bin
/// contributes no gradient).</para>
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class RfftBackwardTinyTransformTests
{
    private readonly IEngine _engine = new CpuEngine();

    // IRFFT of a length-1 spectrum returns exactly its real part (the DC bin), scaled by
    // 1/nFft = 1. The imaginary bin is discarded — correct for a real inverse. Deterministic
    // inputs so the reconstructed value is checked against a closed form, per batch row.
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(3)]
    public void IRFft_LengthOneAxis_ReturnsRealPart(int batch)
    {
        // spectrum[b] = [re_b, im_b]; re/im chosen distinct and non-trivial per row.
        var spectrum = new Tensor<double>(new[] { batch, 2 });
        var sd = spectrum.GetDataArray();
        for (int b = 0; b < batch; b++)
        {
            sd[b * 2] = b + 1.5;          // re_b
            sd[b * 2 + 1] = -(b + 0.25);  // im_b (must be ignored by the inverse)
        }

        var y = _engine.IRFFT(spectrum, outputLength: 1);

        Assert.Equal(2, y.Rank);
        Assert.Equal(batch, y.Shape[0]);
        Assert.Equal(1, y.Shape[1]);
        for (int b = 0; b < batch; b++)
            Assert.Equal(b + 1.5, y[b], precision: 12); // == re_b, im_b dropped
    }

    // Round-trip: IRFFT(RFFT(x), 1) == x for a length-1 axis (RFFT([x]) = [x, 0], inverse
    // recovers x exactly with unit scale).
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    public void Rfft_IRfft_LengthOneAxis_RoundTripsExactly(int batch)
    {
        var x = new Tensor<double>(new[] { batch, 1 });
        var xd = x.GetDataArray();
        for (int b = 0; b < batch; b++) xd[b] = 0.75 - 0.5 * b;

        var spectrum = _engine.RFFT(x);
        Assert.Equal(2, spectrum.Shape[1]); // [re, im] DC only
        for (int b = 0; b < batch; b++)
        {
            Assert.Equal(xd[b], spectrum[b * 2 + 0], precision: 12); // re == x
            Assert.Equal(0.0, spectrum[b * 2 + 1], precision: 12);   // im == 0
        }

        var back = _engine.IRFFT(spectrum, outputLength: 1);
        for (int b = 0; b < batch; b++)
            Assert.Equal(xd[b], back[b], precision: 12);
    }

    // The full tape path #1856 hit: backprop Σ RFFT(x) over a length-1 axis. The gradient is
    // exactly 1 for every element (dRe/dx = 1, dIm/dx = 0), and must carry the input's shape.
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    public void RfftBackward_LengthOneAxis_GradientIsAllOnes(int batch)
    {
        var x = new Tensor<double>(new[] { batch, 1 });
        var xd = x.GetDataArray();
        for (int b = 0; b < batch; b++) xd[b] = 0.3 * (b + 1);

        using var tape = new GradientTape<double>();
        var spectrum = _engine.RFFT(x);
        var loss = _engine.ReduceSum(spectrum, null);
        var grads = tape.ComputeGradients(loss, new[] { x });

        Assert.True(grads.TryGetValue(x, out var gx));
        Assert.Equal(x.Rank, gx.Rank);
        Assert.Equal(x.Shape[0], gx.Shape[0]);
        Assert.Equal(x.Shape[1], gx.Shape[1]);
        for (int i = 0; i < gx.Length; i++)
            Assert.Equal(1.0, gx[i], precision: 12);
    }
}
