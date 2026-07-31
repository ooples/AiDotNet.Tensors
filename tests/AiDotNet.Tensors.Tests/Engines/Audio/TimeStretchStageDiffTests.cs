using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Audio;

/// <summary>
/// Localises the CPU/GPU TimeStretch divergence STAGE BY STAGE for the exact arguments
/// GpuCpuAutoDifferentialTests uses: L=257, nFft=512, hop=128, rate=0.5.
/// </summary>
/// <remarks>
/// <para>
/// Those arguments are VALID — the harness supplies 0.5 for any double and the DEFAULT for an int with
/// one (GpuCpuAutoDifferentialTests lines 173/175), so nFft=512 and hop=128 are the declared defaults
/// and nFft is a power of two. So this is a real defect, not a fuzzer artifact.
/// </para>
/// <para>
/// Five structural hypotheses were already eliminated by reading both implementations (reflect-pad
/// index convention, ISTFT centre placement, numFrames, vocoder axes, STFT mag/phase), so this
/// measures instead of reading: compare each intermediate stage and let the first mismatching one
/// name the culprit.
/// </para>
/// <para>
/// Note rate=0.5 means outFrames = floor(3/0.5) = 6 from only 3 analysis frames, i.e. a time
/// EXPANSION that extrapolates past the available frames — the regime the dedicated tests
/// (rate=1.5, small nFft) never exercise.
/// </para>
/// </remarks>
[Collection("DirectGpuSerial")]
public class TimeStretchStageDiffTests : IDisposable
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _cpu = new();
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;

    public TimeStretchStageDiffTests(ITestOutputHelper o)
    {
        _out = o;
        try { _gpu = new DirectGpuTensorEngine(); _available = _gpu.IsGpuAvailable; }
        catch { _available = false; }
    }

    public void Dispose() { _gpu?.Dispose(); GC.SuppressFinalize(this); }

    private const int L = 257;
    private const int NFft = 512;
    private const int Hop = 128;
    private const double Rate = 0.5;

    private static Tensor<float> Signal()
    {
        var rng = new Random(7);
        var t = new Tensor<float>([L]);
        for (int i = 0; i < L; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static Tensor<float> Hann(int n)
    {
        var w = new Tensor<float>([n]);
        for (int i = 0; i < n; i++) w[i] = (float)(0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / (n - 1)));
        return w;
    }

    private double MaxAbsDiff(Tensor<float> a, Tensor<float> b, string label)
    {
        if (a.Length != b.Length)
        {
            _out.WriteLine($"{label}: LENGTH differs cpu={a.Length} gpu={b.Length}");
            return double.NaN;
        }
        double worst = 0;
        int at = -1;
        for (int i = 0; i < a.Length; i++)
        {
            double d = Math.Abs((double)a[i] - b[i]);
            if (d > worst) { worst = d; at = i; }
        }
        _out.WriteLine($"{label}: maxAbsDiff={worst:E3} at [{at}]  (len={a.Length})");
        return worst;
    }

    /// <summary>Stage 1 — does the STFT itself agree on these arguments?</summary>
    [SkippableFact]
    public void Stage1_Stft_CpuMatchesGpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Signal();
        var w = Hann(NFft);

        _cpu.STFT(x, NFft, Hop, w, center: true, out var cMag, out var cPhase);
        ((AiDotNet.Tensors.Engines.IEngine)_gpu!).STFT(x, NFft, Hop, w, center: true, out var gMag, out var gPhase);

        _out.WriteLine($"cpu mag shape=[{string.Join(",", cMag.Shape.ToArray())}] gpu mag shape=[{string.Join(",", gMag.Shape.ToArray())}]");
        double dm = MaxAbsDiff(cMag, gMag, "STFT magnitude");

        // Phase is an ANGLE, so it must be compared modulo 2*pi. A raw comparison reported 6.283
        // (= 2*pi) at index 769 = k*numFrames + frame = 256*3 + 1, i.e. the NYQUIST bin, where the
        // imaginary part is ~0 and atan2 legitimately returns +pi on one engine and -pi on the other.
        // That is the same angle, and the vocoder accumulates WRAPPED DIFFERENCES so a constant
        // offset cancels anyway — comparing it raw was a defect in this test, not in either engine.
        double dp = 0;
        int dpAt = -1;
        for (int i = 0; i < cPhase.Length; i++)
        {
            double d = Math.Abs((double)cPhase[i] - gPhase[i]);
            d -= 2.0 * Math.PI * Math.Round(d / (2.0 * Math.PI));   // wrap into [-pi, pi]
            d = Math.Abs(d);
            if (d > dp) { dp = d; dpAt = i; }
        }
        _out.WriteLine($"STFT phase (mod 2pi): maxAbsDiff={dp:E3} at [{dpAt}]");

        Assert.True(dm < 1e-3, $"STFT magnitude diverges: {dm:E3}");
        Assert.True(dp < 1e-3, $"STFT phase diverges (mod 2pi): {dp:E3}");
    }

    /// <summary>Stage 2 — the whole op, for reference against the harness's reported error.</summary>
    [SkippableFact]
    public void Stage2_TimeStretch_CpuMatchesGpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Signal();

        var c = _cpu.TimeStretch(x, Rate, NFft, Hop);
        var g = _gpu!.TimeStretch(x, Rate, NFft, Hop);

        _out.WriteLine($"outFrames would be floor(3/{Rate}) = {(int)Math.Floor(3 / Rate)}");
        double d = MaxAbsDiff(c, g, "TimeStretch output");
        Assert.True(d < 1e-2, $"TimeStretch diverges: {d:E3}");
    }

    /// <summary>
    /// Stage 3 — ISTFT alone on a vocoder-shaped input, to separate the vocoder from the synthesis.
    /// Uses outFrames=6 frames of 257 bins, matching what the rate=0.5 path produces.
    /// </summary>
    [SkippableFact]
    public void Stage3_Istft_CpuMatchesGpu()
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Signal();
        var w = Hann(NFft);
        _cpu.STFT(x, NFft, Hop, w, center: true, out var mag, out var phase);

        int targetLen = (int)Math.Round(L / Rate);
        var c = _cpu.ISTFT(mag, phase, NFft, Hop, w, center: true, length: targetLen);
        var g = ((AiDotNet.Tensors.Engines.IEngine)_gpu!).ISTFT(mag, phase, NFft, Hop, w, center: true, length: targetLen);

        double d = MaxAbsDiff(c, g, "ISTFT output");
        Assert.True(d < 1e-2, $"ISTFT diverges: {d:E3}");
    }
}
