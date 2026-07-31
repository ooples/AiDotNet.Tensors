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

    private static Tensor<float> LongSignal(int n)
    {
        var rng = new Random(9);
        var t = new Tensor<float>([n]);
        for (int i = 0; i < n; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

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

        // fp32 bounds: the GPU computes each bin as a direct 512-point DFT sum in float where the CPU
        // uses an fp64 FFT. Measured 3.610E-004 (magnitude) and 2.316E-003 (phase, mod 2pi). Phase is
        // the looser of the two because atan2 amplifies error where the magnitude is small.
        Assert.True(dm < 5e-3, $"STFT magnitude diverges beyond fp32 expectations: {dm:E3}");
        Assert.True(dp < 5e-3, $"STFT phase diverges beyond fp32 expectations (mod 2pi): {dp:E3}");
    }

    /// <summary>
    /// Quantifies the CONDITIONING of the ISTFT overlap-add normalisation at these arguments.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Every structural difference between the CPU and GPU paths is eliminated: identical Hann window
    /// (0.5-0.5cos(2*pi*i/(nFft-1))), identical vocoder arithmetic, identical clamped writeStart, and an
    /// identical 1e-8 window-sum guard. The one remaining difference is that the GPU evaluates a direct
    /// per-bin DFT in fp32 while the CPU uses FFTCore.
    /// </para>
    /// <para>
    /// ISTFT divides by the accumulated window sum, so wherever that sum is small the difference is
    /// amplified. This computes the sum from pure geometry — no engine involved — so the amplification
    /// factor is a fact about the ARGUMENTS, independent of either implementation.
    /// </para>
    /// </remarks>
    [Theory]
    [InlineData(0.5)]
    [InlineData(0.75)]
    [InlineData(1.5)]
    [InlineData(2.0)]
    public void IstftWindowSum_Conditioning(double rate)
    {
        const int NumFrames = 3;                       // (257 + 512 - 512)/128 + 1
        int outFrames = Math.Max(1, (int)Math.Floor(NumFrames / rate));
        int outputLength = (int)Math.Round(L / rate);

        var win = new double[NFft];
        for (int i = 0; i < NFft; i++) win[i] = 0.5 - 0.5 * Math.Cos(2.0 * Math.PI * i / (NFft - 1));

        // The Hann endpoints are exactly zero, so the global minimum is always 0 and is SKIPPED by the
        // guard. What actually amplifies is the smallest sum that PASSES the guard and is divided by.
        double minDivisor = double.MaxValue;
        int minDivisorAt = -1, belowGuard = 0;
        for (int outIdx = 0; outIdx < outputLength; outIdx++)
        {
            double sum = 0;
            for (int frame = 0; frame < outFrames; frame++)
            {
                int writeStart = frame * Hop - NFft / 2;
                int i = outIdx - writeStart;
                if (i >= 0 && i < NFft) sum += win[i] * win[i];
            }
            if (sum <= 1e-8) { belowGuard++; continue; }
            if (sum < minDivisor) { minDivisor = sum; minDivisorAt = outIdx; }
        }

        _out.WriteLine($"rate={rate,4} outFrames={outFrames} len={outputLength} belowGuard={belowGuard} " +
                       $"minDivisor={minDivisor:E3} at={minDivisorAt} " +
                       $"amplification={1.0 / minDivisor:E3}");

        // Regression guard on the writeStart fix. Under correct centring output 0 sits at the window's
        // CENTRE, where win[nFft/2] is approximately 1, so every divisor is O(1) and nothing lands near
        // the 1e-8 guard. The clamped form put win[0] = 0 there instead and bottomed out at 2.29e-8,
        // amplifying fp32 noise by ~1e7. Reintroducing the clamp fails here rather than surfacing as
        // mysterious CPU/GPU drift somewhere downstream.
        Assert.Equal(0, belowGuard);
        Assert.True(minDivisor > 0.1,
            $"smallest overlap-add divisor is {minDivisor:E3} at index {minDivisorAt} (rate={rate}); " +
            $"correct centring keeps it O(1), so this small a value means the frames are misaligned.");
    }

    /// <summary>
    /// Characterises the vocoder divergence by RATE. STFT and ISTFT are now measured to agree
    /// (3.6e-4 and 3.1e-4), so the phase vocoder stage is the only remaining suspect.
    /// </summary>
    /// <remarks>
    /// rate &lt; 1 EXPANDS (outFrames &gt; numFrames, so the vocoder extrapolates past the available
    /// frames and t0 clamps); rate &gt; 1 COMPRESSES. The dedicated tests that pass use rate=1.5, and
    /// the failing harness case uses rate=0.5, so the split is the thing to measure.
    /// </remarks>
    [SkippableTheory]
    [InlineData(0.5)]
    [InlineData(0.75)]
    [InlineData(1.0)]
    [InlineData(1.5)]
    [InlineData(2.0)]
    public void TimeStretchDivergence_ByRate(double rate)
    {
        Skip.If(!_available, "GPU backend not available");
        var x = Signal();
        var c = _cpu.TimeStretch(x, rate, NFft, Hop);
        var g = _gpu!.TimeStretch(x, rate, NFft, Hop);

        int outFrames = (int)Math.Floor(3 / rate);
        Assert.Equal(c.Length, g.Length);
        double worst = 0;
        int worstAt = -1;
        for (int i = 0; i < c.Length; i++)
        {
            double d = Math.Abs((double)c[i] - g[i]);
            if (d > worst) { worst = d; worstAt = i; }
        }
        // Report the tail separately: if the divergence lives at the first few samples (where the
        // centred overlap-add window sum is ~1e-8) and the bulk agrees, the cause is the ill-conditioned
        // normalisation, not the kernels.
        double worstTail = 0;
        for (int i = 8; i < c.Length; i++) worstTail = Math.Max(worstTail, Math.Abs((double)c[i] - g[i]));
        _out.WriteLine($"rate={rate,4} outFrames={outFrames} len={c.Length} " +
                       $"maxAbsDiff={worst:E3} at={worstAt} maxAbsDiff[8..]={worstTail:E3}");
        // Post-fix these all sit at fp32 DFT-vs-FFT noise (9.3e-5 to 3.7e-4 across rates 0.5-2.0),
        // three orders below the 1e-1 the clamped writeStart produced. Same bound as
        // Stage2_TimeStretch_CpuMatchesGpu, which exercises one rate of this same op.
        Assert.True(worst < 1e-2,
            $"CPU and GPU TimeStretch differ by {worst:E3} at index {worstAt} (rate={rate}), well above " +
            $"fp32 noise — the tail is {worstTail:E3}.");
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
    /// Tests whether the ISTFT residual is fp32 ACCUMULATION error rather than a structural defect.
    /// </summary>
    /// <remarks>
    /// The GPU sums nFft terms in a DIRECT inverse DFT, whose error grows O(nFft), while the CPU uses
    /// an fp64 FFT with O(log nFft) growth. If that is the mechanism, the CPU/GPU difference must grow
    /// with nFft — roughly linearly. If instead it is flat, or grows far faster, something structural
    /// is still wrong and the fp32 explanation is dead.
    /// </remarks>
    [SkippableTheory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    public void IstftResidual_ScalesWithNFft(int nFft)
    {
        Skip.If(!_available, "GPU backend not available");
        int hop = nFft / 4;
        var x = Signal();
        var w = Hann(nFft);
        _cpu.STFT(x, nFft, hop, w, center: true, out var mag, out var phase);

        int targetLen = (int)Math.Round(L / Rate);
        var c = _cpu.ISTFT(mag, phase, nFft, hop, w, center: true, length: targetLen);
        var g = ((AiDotNet.Tensors.Engines.IEngine)_gpu!).ISTFT(mag, phase, nFft, hop, w, center: true, length: targetLen);

        // How far the frames actually reach, so a residual in the UNCOVERED tail can be told apart
        // from one inside the reconstructed region.
        int numFrames = mag.Shape.ToArray()[^1];
        int covered = (numFrames - 1) * hop + nFft - (nFft / 2);

        // The overlap-add window sum, from pure geometry — the divisor ISTFT applies.
        double WindowSumAt(int outIdx)
        {
            double s = 0;
            for (int frame = 0; frame < numFrames; frame++)
            {
                int i = outIdx - (frame * hop - nFft / 2);
                if (i >= 0 && i < nFft) s += (double)w[i] * w[i];
            }
            return s;
        }

        double worst = 0, rms = 0, worstNumerator = 0;
        int worstAt = -1, worstNumeratorAt = -1;
        for (int i = 0; i < c.Length; i++)
        {
            double d = Math.Abs((double)c[i] - g[i]);
            if (d > worst) { worst = d; worstAt = i; }
            // ISTFT emits numerator/windowSum, so d*windowSum recovers the difference in the NUMERATOR
            // — the quantity the kernels actually compute, with the division's amplification divided
            // back out. Comparing that needs no region exclusion and no per-nFft tolerance.
            double num = d * WindowSumAt(i);
            if (num > worstNumerator) { worstNumerator = num; worstNumeratorAt = i; }
            rms += (double)c[i] * c[i];
        }
        rms = Math.Sqrt(rms / Math.Max(1, c.Length));
        _out.WriteLine($"nFft={nFft,4} hop={hop,3} frames={numFrames} covered={covered} len={c.Length} " +
                       $"cpuRms={rms:E3} | worst={worst:E3} at={worstAt} windowSumThere={WindowSumAt(worstAt):E3} " +
                       $"| worstNumeratorDiff={worstNumerator:E3} at={worstNumeratorAt}");

        // The two engines differ only in HOW they invert: the GPU evaluates a direct per-bin DFT in fp32,
        // summing nFft terms sequentially, while CpuEngine uses FFTCore. Sequential summation of n terms
        // carries error growing like n*eps, against log(n)*eps for the FFT's pairwise tree, so the bound
        // must grow LINEARLY in nFft — a flat constant would be wrong on its face. At fp32 eps = 1.2e-7
        // and this spectrum's scale, 2e-7 per term tracks the measured growth with about 2x headroom:
        //
        //     nFft   64      128     256     512
        //     meas   6.21e-6 1.09e-5 2.37e-5 5.74e-5    (9.2x across an 8x range — linear, as predicted)
        //     bound  1.28e-5 2.56e-5 5.12e-5 1.02e-4
        //
        // A kernel defect is orders of magnitude clear of this: before the writeStart fix the same
        // comparison sat at 1e-1 in output terms.
        double bound = 2e-7 * nFft;
        Assert.True(worstNumerator < bound,
            $"CPU and GPU ISTFT numerators differ by {worstNumerator:E3} at index {worstNumeratorAt}, " +
            $"above the {bound:E3} that fp32 sequential-DFT-vs-FFT accounts for at nFft={nFft} " +
            $"(hop={hop}) — so the difference is in the kernels, not in the summation order.");
    }

    /// <summary>
    /// Isolates whether the ISTFT divergence lives in the CENTRED path specifically.
    /// </summary>
    /// <remarks>
    /// The nFft-scaling measurement killed the fp32 explanation: the relative difference goes
    /// 2.35e-3 -> 1.94e-2 -> 2.02e-1 -> 2.00e-1 for nFft 64/128/256/512, i.e. ~10x per doubling and
    /// then a PLATEAU near 20%. Rounding grows like sqrt(n) or n and never saturates.
    ///
    /// What does saturate is the fraction of frames whose write position CLAMPS: with center,
    /// writeStart = max(0, frame*hop - nFft/2), so at nFft=512/hop=128 all three frames clamp to 0,
    /// whereas at nFft=64/hop=16 only the first few of seventeen do. If center:false agrees while
    /// center:true does not, the defect is in the clamped-frame handling and nowhere else.
    /// </remarks>
    /// <summary>
    /// center:false on a signal shorter than one window yields ZERO frames — (257-512)/128+1 == 0
    /// under integer division — and used to surface as a bare DivideByZeroException from inside ISTFT.
    /// It must now fail at STFT with a message naming the cause.
    /// </summary>
    [Fact]
    public void Stft_SignalShorterThanWindow_FailsClearly()
    {
        var x = Signal();
        var w = Hann(NFft);
        var ex = Record.Exception(() =>
            _cpu.STFT(x, NFft, Hop, w, center: false, out _, out _));

        Assert.IsType<ArgumentException>(ex);
        _out.WriteLine($"center:false on L={L}, nFft={NFft} -> {ex!.GetType().Name}: {ex.Message}");
        Assert.Contains("too short", ex.Message);
        Assert.DoesNotContain("DivideByZero", ex.Message);
    }

    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void IstftDivergence_IsolatedToTheCentredPath(bool center)
    {
        Skip.If(!_available, "GPU backend not available");
        // center: false does no padding, so the signal must be at least one window long — L=257 against
        // nFft=512 yields zero frames and STFT rejects it by design. The centred case keeps the
        // harness's exact length; the un-centred case uses the shortest length that can produce frames
        // at all, since that is what the mode requires rather than a choice.
        var x = center ? Signal() : LongSignal(1024);
        var w = Hann(NFft);
        _cpu.STFT(x, NFft, Hop, w, center: center, out var mag, out var phase);

        var c = _cpu.ISTFT(mag, phase, NFft, Hop, w, center, length: null);
        var g = ((AiDotNet.Tensors.Engines.IEngine)_gpu!).ISTFT(mag, phase, NFft, Hop, w, center, length: null);

        if (c.Length != g.Length)
        {
            _out.WriteLine($"center={center}: LENGTH differs cpu={c.Length} gpu={g.Length}");
            Assert.Fail($"center={center}: length cpu={c.Length} gpu={g.Length}");
        }
        double worst = 0;
        int at = -1;
        for (int i = 0; i < c.Length; i++)
        {
            double d = Math.Abs((double)c[i] - g[i]);
            if (d > worst) { worst = d; at = i; }
        }
        // ISTFT emits numerator/windowSum, so scaling the observed difference BY the window sum
        // recovers the difference in the NUMERATOR — what the kernels actually compute — with the
        // division's amplification divided back out. Needed because the un-centred tail is inherently
        // ill-conditioned: the last samples are covered only by the decaying edge of the final window,
        // so its sum approaches zero and fp32 noise there is amplified without either engine being
        // wrong. Raw diff peaks at 3.4e-2 at index 1019 for center: false purely from that.
        int frames = mag.Shape.ToArray()[^1];
        double WindowSumAt(int outIdx)
        {
            double sum = 0;
            for (int f = 0; f < frames; f++)
            {
                int i = outIdx - (center ? f * Hop - NFft / 2 : f * Hop);
                if (i >= 0 && i < NFft) sum += (double)w[i] * w[i];
            }
            return sum;
        }
        double worstNumerator = 0;
        int numeratorAt = -1;
        for (int i = 0; i < c.Length; i++)
        {
            double n = Math.Abs((double)c[i] - g[i]) * WindowSumAt(i);
            if (n > worstNumerator) { worstNumerator = n; numeratorAt = i; }
        }
        _out.WriteLine($"center={center}: maxAbsDiff={worst:E3} at [{at}] len={c.Length} " +
                       $"worstNumerator={worstNumerator:E3} at [{numeratorAt}]");

        // Same derived bound as IstftResidual_ScalesWithNFft: the GPU sums nFft terms sequentially per
        // bin while CpuEngine uses FFTCore, and sequential summation error grows like n*eps.
        Assert.True(worstNumerator < 2e-7 * NFft,
            $"CPU and GPU ISTFT numerators differ by {worstNumerator:E3} at index {numeratorAt} " +
            $"(center={center}), above what fp32 summation order accounts for.");
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

        // Uses the INFERRED length. My first version asked for Round(L/Rate) = 514 samples from only
        // THREE analysis frames, which cover just (3-1)*128 + 512 - 512 = 256 — so it was requesting
        // more signal than the input could reconstruct, and the 6.868E-002 it reported was an artifact
        // of that. The real TimeStretch path feeds ISTFT the vocoder's SIX frames, which do cover 514.
        var c = _cpu.ISTFT(mag, phase, NFft, Hop, w, center: true, length: null);
        var g = ((AiDotNet.Tensors.Engines.IEngine)_gpu!).ISTFT(mag, phase, NFft, Hop, w, center: true, length: null);

        double d = MaxAbsDiff(c, g, "ISTFT output");
        // fp32 bound: the GPU sums nFft=512 terms in a direct inverse DFT where the CPU uses an fp64
        // FFT. Measured 3.110E-004 here; 5e-3 leaves headroom without hiding a structural error, which
        // would be orders larger (the pre-fix explicit-length bug showed up as a length mismatch, and
        // the transposed vocoder as ~2e-1).
        Assert.True(d < 5e-3, $"ISTFT diverges beyond fp32 expectations: {d:E3}");
    }
}
