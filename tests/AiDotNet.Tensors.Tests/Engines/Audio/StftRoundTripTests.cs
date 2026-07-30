using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Audio;

/// <summary>
/// <c>ISTFT(STFT(x))</c> must reconstruct x. This isolates the transform pair from anything built on
/// top of it, so a silent <c>TimeStretch</c> can be attributed to the vocoder or to the transforms
/// rather than guessed at.
/// </summary>
/// <remarks>
/// Written because TimeStretch produces a silent output and the phase vocoder rewrite did not fix it.
/// If the round-trip is broken the vocoder is not the (only) culprit — notably ISTFT's explicit-length
/// handling was changed earlier in this branch, so it is a prime suspect and must be ruled in or out
/// before more vocoder work.
/// </remarks>
public class StftRoundTripTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();

    public StftRoundTripTests(ITestOutputHelper o) => _out = o;

    private static Tensor<double> Sine(int n, double hz, int sampleRate)
    {
        var t = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) t[i] = Math.Sin(2.0 * Math.PI * hz * i / sampleRate);
        return t;
    }

    private static Tensor<double> Hann(int n)
    {
        var w = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / (n - 1));
        return w;
    }

    private static double Rms(Tensor<double> t)
    {
        double s = 0;
        for (int i = 0; i < t.Length; i++) s += t[i] * t[i];
        return Math.Sqrt(s / Math.Max(1, t.Length));
    }

    /// <summary>
    /// A Hann window at 4x overlap satisfies COLA, so analysis-then-synthesis should return the input
    /// (interior samples at least). The assertion is on ENERGY: a silent output is the failure mode
    /// under investigation, so RMS is the discriminating statistic.
    /// </summary>
    [Theory]
    [InlineData(256, 64)]
    [InlineData(256, 128)]
    public void IstftOfStft_ReconstructsEnergy(int nFft, int hop)
    {
        const int n = 4096;
        var x = Sine(n, 500.0, 8000);
        var w = Hann(nFft);

        _cpu.STFT(x, nFft, hop, w, center: true, out var mag, out var phase);
        _out.WriteLine($"nFft={nFft} hop={hop} mag.Shape=[{string.Join(",", mag.Shape.ToArray())}] magRms={Rms(mag):F6}");

        var y = _cpu.ISTFT(mag, phase, nFft, hop, w, center: true, length: n);
        _out.WriteLine($"x.Rms={Rms(x):F6}  y.Rms={Rms(y):F6}  y.Length={y.Length}");

        Assert.Equal(n, y.Length);
        Assert.True(Rms(y) > 0.1 * Rms(x),
            $"ISTFT(STFT(x)) lost essentially all energy: x RMS {Rms(x):F6} -> y RMS {Rms(y):F6}. " +
            $"The transform pair itself does not round-trip, so anything layered on it (TimeStretch, " +
            $"PitchShift, GriffinLim) cannot work either.");
    }

    /// <summary>
    /// The STFT magnitude must carry energy in the first place — separates "STFT produced nothing"
    /// from "ISTFT threw it away".
    /// </summary>
    [Fact]
    public void Stft_ProducesNonZeroMagnitude()
    {
        var x = Sine(4096, 500.0, 8000);
        var w = Hann(256);
        _cpu.STFT(x, 256, 64, w, center: true, out var mag, out _);
        _out.WriteLine($"magRms={Rms(mag):F6} shape=[{string.Join(",", mag.Shape.ToArray())}]");
        Assert.True(Rms(mag) > 1e-6, $"STFT magnitude is all but zero (RMS {Rms(mag):E3}).");
    }
}
