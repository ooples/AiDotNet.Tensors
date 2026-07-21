using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

// NOT namespace ...LinearAlgebra.Fft — that shadows the public static class Fft the sibling suites call.
namespace AiDotNet.Tensors.Tests.LinearAlgebra.FftTests;

/// <summary>
/// Same-thermal-window A/B for the two CPU FFT cores: legacy FFTCore versus NativeFFTInPlace.
///
/// WHY THIS EXISTS RATHER THAN TWO SEPARATE RUNS: measuring the cores in separate processes gave numbers
/// that could not be trusted. Re-running IDENTICAL code minutes later moved individual shapes by up to 3.7x
/// (8192x256 RFFT: 18.221 ms then 68.047 ms), which is larger than the effect under test — this box has many
/// cores and throttles under sustained load, so any cross-run comparison mostly measures thermal state.
///
/// The fix is to interleave. Each repetition times legacy and modern back to back on the SAME input inside
/// ONE process, and the reported statistic is the median of the PER-REPETITION RATIO, not a ratio of
/// medians. A thermal excursion that slows one repetition slows both halves of that pair, so it largely
/// cancels in the ratio instead of landing entirely on whichever variant happened to run during it.
///
/// Allocation is reported separately and is deterministic — it reproduced to within 0.1% across runs while
/// timings swung 3x, so it is the stronger evidence of the two.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class RfftInterleavedAbBenchmark
{
    private readonly ITestOutputHelper _out;
    public RfftInterleavedAbBenchmark(ITestOutputHelper output) => _out = output;

    private static long AllocatedBytes()
    {
#if NET6_0_OR_GREATER
        return GC.GetTotalAllocatedBytes(precise: false);
#else
        return GC.GetTotalMemory(forceFullCollection: false);
#endif
    }

    private static Tensor<double> Signal(int batch, int n, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<double>([batch, n]);
        for (int i = 0; i < t.Length; i++) t[i] = rng.NextDouble() * 2 - 1;
        return t;
    }

    private static void ValidateRfftResult(Tensor<double> result, int batch, int n)
    {
        int nFft = 1;
        while (nFft < n) nFft <<= 1;
        int expectedWidth = (nFft / 2 + 1) * 2;
        Assert.Equal(2, result.Rank);
        Assert.Equal(batch, result.Shape[0]);
        Assert.Equal(expectedWidth, result.Shape[1]);
        Assert.Equal(batch * expectedWidth, result.Length);
        for (int i = 0; i < result.Length; i++)
            if (double.IsNaN(result[i]) || double.IsInfinity(result[i]))
                Assert.Fail($"RFFT produced a non-finite value at index {i}.");
    }

    /// <summary>
    /// Correctness first: the two cores must agree before any timing of them means anything. Both compute
    /// the same transform, so they should match to near machine precision — the summation order is identical,
    /// only twiddle provenance and numeric specialisation differ.
    /// </summary>
    [Fact]
    public void Legacy_and_modern_cores_agree()
    {
        var prior = AiDotNetEngine.Current;
        bool priorFftCore = CpuEngine.UseLegacyFftCore;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = AiDotNetEngine.Current;
            foreach (var (batch, n) in new[] { (8, 128), (16, 192), (4, 256), (2, 1024) })
            {
                using var x = Signal(batch, n, seed: 5);

                CpuEngine.UseLegacyFftCore = true;
                using var legacy = engine.RFFT(x);
                CpuEngine.UseLegacyFftCore = false;
                using var modern = engine.RFFT(x);

                ValidateRfftResult(legacy, batch, n);
                ValidateRfftResult(modern, batch, n);

                double maxRel = 0;
                for (int i = 0; i < legacy.Length; i++)
                    maxRel = Math.Max(maxRel, Math.Abs(legacy[i] - modern[i]) / Math.Max(1.0, Math.Abs(legacy[i])));

                _out.WriteLine($"batch={batch,4} n={n,5}  maxRel={maxRel:E3}");
                Assert.True(maxRel <= 1e-12, $"cores diverged at batch={batch} n={n}: maxRel={maxRel:E3}");
            }
        }
        finally { CpuEngine.UseLegacyFftCore = priorFftCore; AiDotNetEngine.Current = prior; }
    }

    [Fact(Timeout = 900000)]
    public async Task Interleaved_ab_legacy_vs_modern()
    {
        await Task.Yield();
        var prior = AiDotNetEngine.Current;
        bool priorFftCore = CpuEngine.UseLegacyFftCore;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var engine = AiDotNetEngine.Current;
            const int Warmup = 3, Reps = 15;

            _out.WriteLine("RFFT CPU cores — interleaved A/B in one process (same thermal window)");
            _out.WriteLine($"warmup={Warmup}, reps={Reps}, statistic = MEDIAN OF PER-REP RATIOS");
            _out.WriteLine("");
            _out.WriteLine("  batch      n   legacy ms   modern ms   median ratio   ratio p25..p75   alloc legacy KB   alloc modern KB   alloc x");
            _out.WriteLine("  " + new string('-', 128));

            foreach (var (batch, n) in new[] { (256, 128), (1024, 128), (4096, 192), (4096, 256), (8192, 256), (1024, 1024) })
            {
                using var x = Signal(batch, n, seed: 7);

                for (int i = 0; i < Warmup; i++)
                {
                    CpuEngine.UseLegacyFftCore = true;
                    using (var legacyWarmup = engine.RFFT(x)) ValidateRfftResult(legacyWarmup, batch, n);
                    CpuEngine.UseLegacyFftCore = false;
                    using (var modernWarmup = engine.RFFT(x)) ValidateRfftResult(modernWarmup, batch, n);
                }

                var ratios = new double[Reps];
                var legacyMs = new double[Reps];
                var modernMs = new double[Reps];
                long legacyBytes = 0, modernBytes = 0;

                for (int r = 0; r < Reps; r++)
                {
                    // Legacy half.
                    CpuEngine.UseLegacyFftCore = true;
                    long b0 = AllocatedBytes();
                    var swL = Stopwatch.StartNew();
                    using var legacy = engine.RFFT(x);
                    swL.Stop();
                    long b1 = AllocatedBytes();
                    ValidateRfftResult(legacy, batch, n);

                    // Modern half, immediately after — same thermal state.
                    CpuEngine.UseLegacyFftCore = false;
                    long modernBefore = AllocatedBytes();
                    var swM = Stopwatch.StartNew();
                    using var modern = engine.RFFT(x);
                    swM.Stop();
                    long b2 = AllocatedBytes();
                    ValidateRfftResult(modern, batch, n);

                    legacyMs[r] = swL.Elapsed.TotalMilliseconds;
                    modernMs[r] = swM.Elapsed.TotalMilliseconds;
                    ratios[r] = legacyMs[r] / Math.Max(modernMs[r], 1e-9);
                    legacyBytes += b1 - b0;
                    modernBytes += b2 - modernBefore;
                }

                Array.Sort(ratios);
                var lSorted = (double[])legacyMs.Clone(); Array.Sort(lSorted);
                var mSorted = (double[])modernMs.Clone(); Array.Sort(mSorted);

                double aL = legacyBytes / (double)Reps / 1024.0;
                double aM = modernBytes / (double)Reps / 1024.0;

                _out.WriteLine(
                    $"  {batch,5}  {n,5}  {lSorted[Reps / 2],9:F3}  {mSorted[Reps / 2],9:F3}  " +
                    $"{ratios[Reps / 2],12:F2}x  {ratios[Reps / 4],6:F2}..{ratios[3 * Reps / 4],-6:F2}  " +
                    $"{aL,15:F1}  {aM,15:F1}  {(aM > 0 ? aL / aM : 0),7:F2}x");
            }

            _out.WriteLine("");
            _out.WriteLine("Ratio > 1 means the modern core is faster. The p25..p75 spread is the honest error bar:");
            _out.WriteLine("if it straddles 1.00 the shape shows no reliable difference, however good the median looks.");
        }
        finally { CpuEngine.UseLegacyFftCore = priorFftCore; AiDotNetEngine.Current = prior; }
    }
}
