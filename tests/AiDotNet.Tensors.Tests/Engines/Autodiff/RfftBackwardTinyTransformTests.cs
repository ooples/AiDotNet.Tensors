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
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class RfftBackwardTinyTransformTests
{
    private readonly IEngine _engine = new CpuEngine();

    // Forward RFFT of a length-1 axis is the identity DC bin; its adjoint (IRFFT with
    // outputLength = 1) must reconstruct without throwing, for any leading batch shape.
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(3)]
    public void IRFft_LengthOneAxis_DoesNotThrow(int batch)
    {
        // RFFT([batch, 1]) -> [batch, 2] (interleaved DC re/im). Feed that straight back
        // into IRFFT(outputLength: 1) — the exact shape RFFT's backward produces.
        var spectrum = new Tensor<double>(new[] { batch, 2 });
        var sd = spectrum.GetDataArray();
        var rng = new Random(5);
        for (int i = 0; i < sd.Length; i++) sd[i] = rng.NextDouble() * 2 - 1;

        var ex = Record.Exception(() => _engine.IRFFT(spectrum, outputLength: 1));
        Assert.Null(ex);
    }

    // The full tape path #1856 hit: backprop a scalar loss through Engine.RFFT over a
    // length-1 axis. Must produce a finite gradient of the input's shape, not throw.
    [Theory]
    [InlineData(1)]
    [InlineData(8)]
    public void RfftBackward_LengthOneAxis_ProducesFiniteGradient(int batch)
    {
        var x = new Tensor<double>(new[] { batch, 1 });
        var xd = x.GetDataArray();
        var rng = new Random(11);
        for (int i = 0; i < xd.Length; i++) xd[i] = rng.NextDouble() * 2 - 1;

        using var tape = new GradientTape<double>();
        var spectrum = _engine.RFFT(x);
        var loss = _engine.ReduceSum(spectrum, null);
        var grads = tape.ComputeGradients(loss, new[] { x });

        Assert.True(grads.TryGetValue(x, out var gx));
        Assert.Equal(x.Length, gx.Length);
        for (int i = 0; i < gx.Length; i++)
            Assert.False(double.IsNaN(gx[i]) || double.IsInfinity(gx[i]),
                $"grad[{i}] not finite: {gx[i]}");
    }
}
