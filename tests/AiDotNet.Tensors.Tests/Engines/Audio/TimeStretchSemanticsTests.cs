using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Audio;

/// <summary>
/// <c>TimeStretch</c> must actually time-stretch: a phase vocoder changes DURATION while preserving
/// pitch, so a pure sine in must come out a pure sine at the SAME frequency.
/// </summary>
/// <remarks>
/// <para>
/// This exists because the GPU/CPU parity failure for PitchShift is a VALUE difference (8.382e-2 on
/// shape [257]) with identical compositions on both sides, which pointed at the phase vocoder itself
/// rather than at the plumbing. Output length is not evidence either way — CpuEngine.TimeStretch passes
/// an explicit <c>length: Round(L / rate)</c> to ISTFT, so the length is correct by construction no
/// matter what the vocoder does to the values.
/// </para>
/// <para>
/// Reading the CPU implementation: <c>STFT</c> returns <c>[.., numFreqs, numFrames]</c>, but
/// <c>TimeStretch</c> reads <c>nFrames = mag._shape[^2]</c> (= numFreqs) and
/// <c>nFreq = mag._shape[^1]</c> (= numFrames). Its outer loop therefore interpolates along the
/// FREQUENCY axis and its phase accumulator sums differences between adjacent frequency BINS, where a
/// phase vocoder must interpolate along time and accumulate across time FRAMES.
/// </para>
/// <para>
/// A spectral test is the decisive check: index algebra can be argued about, but a transposed vocoder
/// cannot preserve a sine's frequency.
/// </para>
/// </remarks>
public class TimeStretchSemanticsTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();

    public TimeStretchSemanticsTests(ITestOutputHelper o) => _out = o;

    private const int SampleRate = 8000;
    private const int Length = 4096;
    private const double ToneHz = 500.0;

    private static Tensor<double> Sine(int n, double hz, int sampleRate)
    {
        var t = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) t[i] = Math.Sin(2.0 * Math.PI * hz * i / sampleRate);
        return t;
    }

    /// <summary>Dominant frequency (Hz) via a plain magnitude-DFT peak over the positive bins.</summary>
    private static double DominantHz(Tensor<double> signal, int sampleRate)
    {
        int n = signal.Length;
        double best = 0;
        int bestK = 0;
        // Skip DC and the immediate neighbours so a residual offset cannot win.
        for (int k = 2; k < n / 2; k++)
        {
            double re = 0, im = 0;
            for (int i = 0; i < n; i++)
            {
                double ang = -2.0 * Math.PI * k * i / n;
                re += signal[i] * Math.Cos(ang);
                im += signal[i] * Math.Sin(ang);
            }
            double mag = re * re + im * im;
            if (mag > best) { best = mag; bestK = k; }
        }
        return (double)bestK * sampleRate / n;
    }

    /// <summary>
    /// Stretching by 2x halves the duration and must leave the pitch alone. A tolerance of one DFT bin
    /// of the OUTPUT is generous; a transposed vocoder misses by far more than that.
    /// </summary>
    [Theory]
    [InlineData(2.0)]
    [InlineData(0.5)]
    public void TimeStretch_PreservesDominantFrequency(double rate)
    {
        var x = Sine(Length, ToneHz, SampleRate);
        var y = _cpu.TimeStretch(x, rate, nFft: 256, hopLength: 64);

        int expectedLen = (int)Math.Round(Length / rate);
        Assert.Equal(expectedLen, y.Shape[^1]);

        double inHz = DominantHz(x, SampleRate);
        double outHz = DominantHz(y, SampleRate);
        double binHz = (double)SampleRate / y.Length;
        _out.WriteLine($"rate={rate}  inDominant={inHz:F1} Hz  outDominant={outHz:F1} Hz  outBin={binHz:F1} Hz");

        Assert.True(Math.Abs(outHz - inHz) <= 2 * binHz,
            $"TimeStretch(rate={rate}) moved the dominant frequency from {inHz:F1} Hz to {outHz:F1} Hz " +
            $"(tolerance {2 * binHz:F1} Hz = 2 output DFT bins). A phase vocoder changes duration, not " +
            $"pitch; interpolating along the frequency axis instead of time does exactly this.");
    }

    /// <summary>
    /// Sanity anchor: the measurement method itself is sound. Plain decimation (taking every other
    /// sample) DOES shift pitch, so this confirms DominantHz can detect a pitch change at all — without
    /// it, a passing test above could just mean the detector is blind.
    /// </summary>
    [Fact]
    public void DominantHzDetector_SeesAPitchChangeWhenOneReallyHappens()
    {
        var x = Sine(Length, ToneHz, SampleRate);
        var decimated = new Tensor<double>([Length / 2]);
        for (int i = 0; i < Length / 2; i++) decimated[i] = x[i * 2];

        double inHz = DominantHz(x, SampleRate);
        double outHz = DominantHz(decimated, SampleRate);
        _out.WriteLine($"decimation: {inHz:F1} Hz -> {outHz:F1} Hz (expected ~{2 * ToneHz:F0} Hz)");

        // Dropping every other sample at the same nominal rate doubles the apparent frequency.
        Assert.True(outHz > inHz * 1.5,
            $"detector failed its own sanity check: decimation should roughly double the apparent " +
            $"frequency but reported {inHz:F1} -> {outHz:F1} Hz");
    }
}
