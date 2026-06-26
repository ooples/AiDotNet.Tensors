using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #475 medium-axis routing A/B. The deterministic-mode default heuristic
/// (<see cref="AxisSelector"/>) skips the M-axis when <c>k &gt; 256</c> (intending finer
/// K/2D splits for large K), but in deterministic mode the K-axis is forbidden, so a
/// "medium" shape (m gives enough M-blocks, n only modestly large) falls through to the
/// N-axis — which under-subscribes the cores vs the M-axis. This measures each forced
/// axis vs the live default heuristic and vs native OpenBLAS, at full DOP, deterministic
/// mode, warmed min-of-N, through the real <see cref="BlasManaged.Gemm{T}"/> strategy path.
///
/// <para>
/// <c>PackingMode.DisableAutotune</c> is used so the heuristic (and thus the
/// <see cref="AxisSelector.ForceAxisForTest"/> override) runs on every call instead of the
/// cache returning a stored decision — by default autotune is heuristic-only, so this is
/// representative of the production Auto path.
/// </para>
///
/// <para>Run: <c>--ab-axis-routing</c>.</para>
/// </summary>
public static class AxisRoutingAbBench
{
    public static void Run()
    {
        Console.WriteLine("=== #475 medium-axis routing A/B (FP32, full DOP, deterministic) ===");
        Console.WriteLine($"Host: cores={Environment.ProcessorCount}  HasRawSgemm={BlasProvider.HasRawSgemm}  Deterministic={BlasProvider.IsDeterministicMode}");
        Console.WriteLine("Per shape: native OpenBLAS bar, the live heuristic's pick, and each forced axis (GF/s, min-of-N).");
        Console.WriteLine();

        int procs = Environment.ProcessorCount;
        const int mr = 6, nr = 16; // FP32 AVX2 6x16 panel tile

        // (label, m, n, k) — diffusion-relevant FP32 GEMM shapes.
        var shapes = new (string label, int m, int n, int k)[]
        {
            ("medium    384x1024x1024", 384, 1024, 1024),
            ("square   1024x1024x1024", 1024, 1024, 1024),
            ("attn-proj 256x1536x1536", 256, 1536, 1536),
            ("ffn-up    384x6144x1536", 384, 6144, 1536),
            ("ffn-big   384x4096x3456", 384, 4096, 3456),
        };

        foreach (var (label, m, n, k) in shapes)
        {
            var a = MakeRandom(m * k);
            var b = MakeRandom(k * n);
            var c = new float[m * n];
            double flops = 2.0 * m * n * k;
            var heuristic = AxisSelector.Select(m, n, k, mr, nr, procs, BlasProvider.IsDeterministicMode);

            double obGf  = BlasProvider.HasRawSgemm ? Gf(flops, TimeMinNative(a, b, c, m, n, k)) : double.NaN;
            double defGf = Gf(flops, TimeMinManaged(a, b, c, m, n, k, null));
            double mGf   = Gf(flops, TimeMinManaged(a, b, c, m, n, k, ParallelismAxis.M));
            double nGf   = Gf(flops, TimeMinManaged(a, b, c, m, n, k, ParallelismAxis.N));
            double gGf   = Gf(flops, TimeMinManaged(a, b, c, m, n, k, ParallelismAxis.MN_2D));

            double best = Math.Max(mGf, Math.Max(nGf, gGf));
            string bestAxis = best == mGf ? "M" : best == nGf ? "N" : "2D";

            Console.WriteLine($"  {label}   (heuristic picks {heuristic})");
            Console.WriteLine($"    OpenBLAS {obGf,6:F0} | default {defGf,6:F0} | M {mGf,6:F0}  N {nGf,6:F0}  2D {gGf,6:F0}  GF/s");
            Console.WriteLine($"    best-forced {bestAxis} {best,6:F0} -> default is {defGf / best * 100,4:F0}% of best (gap {best - defGf,5:F0} GF/s)");
            Console.WriteLine();
        }
    }

    /// <summary>
    /// Single-thread GEMM profiling repro — pins MaxDOP=1 and hammers one shape through the
    /// managed PackBoth path for a fixed wall-time, so a PMU profiler (PerfView CPU-counter
    /// sampling) can attribute the single-core stall (the ~72 vs OpenBLAS ~103 GF/s gap).
    /// Run: <c>--profile-gemm-st [seconds=25] [shape=ffn-up|ffn-big|square|attn|medium]</c>.
    /// </summary>
    public static void ProfileSingleThread(int seconds, string shape)
    {
        CpuParallelSettings.MaxDegreeOfParallelism = 1; // profile the single-core path
        (int m, int n, int k) = shape switch
        {
            "ffn-big" => (384, 4096, 3456),
            "square"  => (1024, 1024, 1024),
            "attn"    => (256, 1536, 1536),
            "medium"  => (384, 1024, 1024),
            _         => (384, 6144, 1536), // ffn-up
        };
        var a = MakeRandom(m * k);
        var b = MakeRandom(k * n);
        var c = new float[m * n];
        double flops = 2.0 * m * n * k;
        Console.WriteLine($"=== single-thread GEMM profile repro: {shape} {m}x{n}x{k}  MaxDOP={CpuParallelSettings.MaxDegreeOfParallelism}  {seconds}s ===");
        for (int i = 0; i < 5; i++) GemmOnce(a, b, c, m, n, k); // warm
        var total = Stopwatch.StartNew();
        var one = new Stopwatch();
        long iters = 0;
        double bestSec = double.MaxValue;
        while (total.Elapsed.TotalSeconds < seconds)
        {
            one.Restart();
            GemmOnce(a, b, c, m, n, k);
            one.Stop();
            bestSec = Math.Min(bestSec, one.Elapsed.TotalSeconds);
            iters++;
        }
        Console.WriteLine($"iters={iters}  best {flops / bestSec / 1e9:F1} GF/s  avg {flops / (total.Elapsed.TotalSeconds / iters) / 1e9:F1} GF/s  (sink {c[0]:E2})");
    }

    private static void GemmOnce(float[] a, float[] b, float[] c, int m, int n, int k)
    {
        var opts = new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune };
        BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k, in opts);
    }

    private static double Gf(double flops, double sec) => flops / sec / 1e9;

    private static unsafe double TimeMinNative(float[] a, float[] b, float[] c, int m, int n, int k)
        => TimeMin(() => { fixed (float* pa = a, pb = b, pc = c) BlasProvider.SgemmRaw(m, n, k, pa, k, pb, n, pc, n); }, m, n, k);

    private static double TimeMinManaged(float[] a, float[] b, float[] c, int m, int n, int k, ParallelismAxis? force)
    {
        return TimeMin(() =>
        {
            var opts = new BlasOptions<float> { PackingMode = PackingMode.DisableAutotune };
            AxisSelector.ForceAxisForTest = force;
            try { BlasManaged.Gemm<float>(a, k, false, b, n, false, c, n, m, n, k, in opts); }
            finally { AxisSelector.ForceAxisForTest = null; }
        }, m, n, k);
    }

    private static double TimeMin(Action op, int m, int n, int k)
    {
        double work = (double)m * n * k;
        int iters = work > 2e9 ? 2 : work > 5e8 ? 4 : 10;
        const int rounds = 12;
        for (int i = 0; i < 3 * iters; i++) op(); // warm
        var sw = new Stopwatch();
        double best = double.MaxValue;
        for (int r = 0; r < rounds; r++)
        {
            sw.Restart();
            for (int i = 0; i < iters; i++) op();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalSeconds / iters);
        }
        return best;
    }

    private static float[] MakeRandom(int len)
    {
        var rng = new Random(17);
        var x = new float[len];
        for (int i = 0; i < len; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        return x;
    }
}
