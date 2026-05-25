using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Comparative perf benchmark: BlasManaged.Gemm vs BlasProvider.TryGemmEx
/// (native OpenBLAS/MKL) on 12 representative shapes spanning the call-site
/// spectrum in the codebase.
///
/// <para>
/// Goal: identify shapes where BlasManaged is competitive with the native
/// path (so we can safely remove the third-party dependency) and shapes
/// where it lags (so we know what to optimize next).
/// </para>
///
/// <para>
/// This is a REPORTING test — no assertions other than "doesn't crash".
/// The output table is the deliverable. Run with:
///   dotnet test --filter "Compare_BlasManaged_vs_NativeBLAS_FP64" --logger "console;verbosity=normal"
/// </para>
/// </summary>
// Pure reporting benchmark (see summary above — "no assertions other than
// doesn't crash"). Ran ~3 min under CI coverage despite gating nothing; tag it
// so the CI filter (Category!=Benchmark&Category!=Performance) excludes it.
[Trait("Category", "Benchmark")]
[Collection("BlasManaged-Stats-Serial")]
public class ComparativeBenchmark
{
    private readonly ITestOutputHelper _output;

    public ComparativeBenchmark(ITestOutputHelper output) { _output = output; }

    private sealed record ShapeSpec(string Name, int M, int N, int K, bool TransA, bool TransB);

    private static readonly ShapeSpec[] Shapes =
    [
        new("Tiny_64sq",                   64,    64,   64,   false, false),
        new("Med_256sq",                   256,   256,  256,  false, false),
        new("Large_1024sq",                1024,  1024, 1024, false, false),
        new("L2_4096x16x512_TA",           4096,  16,   512,  true,  false),
        new("L2mirror_16x4096_TB",         16,    4096, 512,  false, true),
        new("WideFat_512x512x64",          512,   512,  64,   false, false),
        new("TallK_64x64x4096",            64,    64,   4096, false, false),
        new("BwdConvKern_256x256x128_TA",  256,   256,  128,  true,  false),
        new("FFN_128x768x768",             128,   768,  768,  false, false),
        new("Tiny_32sq",                   32,    32,   32,   false, false),
        new("Conv2D_2048x32x256",          2048,  32,   256,  false, false),
        new("Conv2Dbwd_32x2048x256",       32,    2048, 256,  false, false),
    ];

    [Fact]
    public void Compare_BlasManaged_vs_NativeBLAS_FP64()
    {
        bool blasAvailable = BlasProvider.IsAvailable;

        _output.WriteLine($"BlasManaged vs Native BLAS (FP64) — {DateTime.UtcNow:u}");
        _output.WriteLine($"Host: {HardwareFingerprint.Current}");
        _output.WriteLine($"Native BLAS available: {blasAvailable}");
        if (!blasAvailable)
            _output.WriteLine("  (NativeBLAS column will show N/A — OpenBLAS not loaded on this host)");
        _output.WriteLine("");
        _output.WriteLine($"{"Shape",-32} {"BlasManaged",12} {"NativeBLAS",12} {"Gap",10} {"Winner",-12} {"BM/Native",9}");
        _output.WriteLine(new string('-', 95));

        int bmWins = 0, tied = 0, nativeWins = 0;
        double worstRatio = 0;
        string? worstShape = null;

        // Track top-3 worst gaps for BlasManaged
        var top3 = new (string Name, double Ratio)[3];

        foreach (var s in Shapes)
        {
            double bmMedian = MeasureBlasManaged(s);
            (double blasMedian, bool blasOk) = MeasureNativeBLAS(s);

            if (!blasOk)
            {
                _output.WriteLine($"{s.Name,-32} {bmMedian,10:F3}ms {"N/A",12} {"—",10} {"BlasManaged",-12} {"—",9}");
                continue;
            }

            double ratio = bmMedian / blasMedian;
            string winner;
            if (ratio < 1.0)
            {
                winner = "BlasManaged";
                bmWins++;
            }
            else if (ratio < 1.2)
            {
                winner = "~tied";
                tied++;
            }
            else
            {
                winner = "NativeBLAS";
                nativeWins++;
                if (ratio > worstRatio)
                {
                    worstRatio = ratio;
                    worstShape = s.Name;
                }
            }

            double pctGap = (ratio - 1.0) * 100;
            string gapStr = pctGap >= 0 ? $"+{pctGap:F0}%" : $"{pctGap:F0}%";

            _output.WriteLine($"{s.Name,-32} {bmMedian,10:F3}ms {blasMedian,10:F3}ms {gapStr,10} {winner,-12} {ratio,8:F2}x");

            // Update top-3 worst (by ratio)
            UpdateTop3(top3, s.Name, ratio);
        }

        _output.WriteLine(new string('-', 95));
        _output.WriteLine("");
        _output.WriteLine("=== SUMMARY ===");
        _output.WriteLine($"  BlasManaged wins (ratio < 1.0):   {bmWins,2}");
        _output.WriteLine($"  Tied         (1.0 ≤ ratio < 1.2): {tied,2}");
        _output.WriteLine($"  NativeBLAS wins (ratio ≥ 1.2):    {nativeWins,2}");
        if (worstShape is not null)
            _output.WriteLine($"  Worst gap: {worstShape} — BlasManaged is {worstRatio:F2}x slower than native");
        _output.WriteLine("");
        _output.WriteLine("Top-3 shapes where BlasManaged most needs improvement:");
        Array.Sort(top3, (x, y) => y.Ratio.CompareTo(x.Ratio));
        int rank = 1;
        foreach (var (name, r) in top3)
        {
            if (name is null) continue;
            _output.WriteLine($"  {rank++}. {name}: {r:F2}x native BLAS");
        }

        _output.WriteLine("");
        _output.WriteLine("Legend:");
        _output.WriteLine("  ratio < 1.0   = BlasManaged faster than native BLAS");
        _output.WriteLine("  ratio 1.0-1.2 = tied (within 20% noise floor)");
        _output.WriteLine("  ratio > 1.2   = NativeBLAS faster; gap = how far BlasManaged needs to close");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Measurement helpers
    // ──────────────────────────────────────────────────────────────────────────

    private static double MeasureBlasManaged(ShapeSpec s)
    {
        int aRows = s.TransA ? s.K : s.M;
        int aCols = s.TransA ? s.M : s.K;
        int bRows = s.TransB ? s.N : s.K;
        int bCols = s.TransB ? s.K : s.N;
        var rng = new Random(42);
        double[] a = new double[aRows * aCols];
        double[] b = new double[bRows * bCols];
        double[] c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        // Scale iteration count to work size so large shapes don't run forever.
        long workEst = (long)s.M * s.N * s.K;
        int iters = workEst > 100_000_000L ? 10 : workEst > 10_000_000L ? 30 : 50;
        const int Warmup = 3;

        for (int i = 0; i < Warmup; i++)
            BlasManagedLib.Gemm<double>(
                a, lda: aCols, transA: s.TransA,
                b, ldb: bCols, transB: s.TransB,
                c, ldc: s.N,
                m: s.M, n: s.N, k: s.K);

        double[] times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<double>(
                a, lda: aCols, transA: s.TransA,
                b, ldb: bCols, transB: s.TransB,
                c, ldc: s.N,
                m: s.M, n: s.N, k: s.K);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        return times[iters / 2];   // median
    }

    private static (double Median, bool Ok) MeasureNativeBLAS(ShapeSpec s)
    {
        if (!BlasProvider.IsAvailable) return (0, false);

        int aRows = s.TransA ? s.K : s.M;
        int aCols = s.TransA ? s.M : s.K;
        int bRows = s.TransB ? s.N : s.K;
        int bCols = s.TransB ? s.K : s.N;
        var rng = new Random(42);
        double[] a = new double[aRows * aCols];
        double[] b = new double[bRows * bCols];
        double[] c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        long workEst = (long)s.M * s.N * s.K;
        int iters = workEst > 100_000_000L ? 10 : workEst > 10_000_000L ? 30 : 50;
        const int Warmup = 3;

        try
        {
            // TryGemmEx(m, n, k, a, aOffset, lda, transA, b, bOffset, ldb, transB, c, cOffset, ldc)
            for (int i = 0; i < Warmup; i++)
                BlasProvider.TryGemmEx(
                    m: s.M, n: s.N, k: s.K,
                    a: a, aOffset: 0, lda: aCols, transA: s.TransA,
                    b: b, bOffset: 0, ldb: bCols, transB: s.TransB,
                    c: c, cOffset: 0, ldc: s.N);

            double[] times = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                BlasProvider.TryGemmEx(
                    m: s.M, n: s.N, k: s.K,
                    a: a, aOffset: 0, lda: aCols, transA: s.TransA,
                    b: b, bOffset: 0, ldb: bCols, transB: s.TransB,
                    c: c, cOffset: 0, ldc: s.N);
                sw.Stop();
                times[i] = sw.Elapsed.TotalMilliseconds;
            }
            Array.Sort(times);
            return (times[iters / 2], true);
        }
        catch
        {
            // Should not happen — IsAvailable already passed — but be defensive.
            return (0, false);
        }
    }

    private static void UpdateTop3((string Name, double Ratio)[] top3, string name, double ratio)
    {
        // Keep the three entries with the highest ratio.
        int minIdx = 0;
        for (int i = 1; i < top3.Length; i++)
            if (top3[i].Ratio < top3[minIdx].Ratio) minIdx = i;
        if (ratio > top3[minIdx].Ratio)
            top3[minIdx] = (name, ratio);
    }
}
