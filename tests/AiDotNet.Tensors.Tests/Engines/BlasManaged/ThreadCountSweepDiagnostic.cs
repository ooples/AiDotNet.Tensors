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
/// #409: clean thread-count vs work-size sweep for mid-size GEMMs. Tests the
/// hypothesis that the dispatch over-parallelizes mid-size shapes (≈1 GFLOP over
/// 32 threads = tiny per-thread work, so threading overhead dominates).
///
/// <para>
/// Methodology guards against the cold-start confound that produced earlier
/// false alarms: (1) a global warmup runs every shape at min/max threads to JIT
/// all paths + populate the autotune cache; (2) EVERY (shape, threadCount) timing
/// is preceded by its own warmup; (3) median of N timed runs. So all configs are
/// equally warm. Gated behind AIDOTNET_RUN_THREADSWEEP=1.
/// </para>
///
/// <para>
/// <b>Finding (corrected): there is NO reproducible thread-count lever.</b> An
/// initial run of THIS sweep showed dramatic spreads (Large 2t=11.3 vs 32t=19.8
/// ms, BERT 4–16t=44 ms) that looked like a 1.05–1.76× win from fewer threads.
/// That was MEASUREMENT VARIANCE — inter-config thermal/background drift across a
/// 36-config sequential sweep on a noisy 32-core box. A tighter re-measurement
/// (each of {2,4,8,32}t independently warmed, fewer configs, one run) showed all
/// thread counts within ~5% for every shape (e.g. Large: 2t=15.1, 4t=15.1,
/// 8t=15.4, 32t=15.1 ms). So thread count does NOT meaningfully affect these
/// shapes, and a thread-count autotune was prototyped and then REVERTED — it
/// captured a non-existent win and would only add hot-path complexity.
/// </para>
/// <para>
/// The real, thread-count-independent gap is fundamental managed-vs-OpenBLAS
/// efficiency: ~1.5× (Large), ~1.8× (BERT), ~2.3–3× (FFN) native — the kernel +
/// pack/strategy efficiency, not parallelism. Lesson recorded: mid-size GEMM
/// timings on a many-core box are high-variance; trust only tightly-controlled,
/// individually-warmed, single-run comparisons.
/// </para>
/// </summary>
[Trait("Category", "Benchmark")]
[Collection("BlasManaged-Perf-Serial")]
public class ThreadCountSweepDiagnostic
{
    private readonly ITestOutputHelper _output;
    public ThreadCountSweepDiagnostic(ITestOutputHelper output) { _output = output; }

    private sealed record Shape(string Name, int M, int N, int K);

    private static readonly Shape[] Shapes =
    [
        new("MLP_256x256x256",        256,  256,  256),   // 33 MFLOP
        new("FFN_128x768x768",        128,  768,  768),   // 151 MFLOP
        new("FFN_up_512x2048x512",    512,  2048, 512),   // 1.07 GFLOP
        new("FFN_down_512x512x2048",  512,  512,  2048),  // 1.07 GFLOP
        new("BERT_FFN_1024x3072x768", 1024, 3072, 768),   // 4.8 GFLOP
        new("Large_1024sq",           1024, 1024, 1024),  // 2.1 GFLOP
    ];

    private static readonly int[] ThreadCounts = { 1, 2, 4, 8, 16, 32 };

    [Fact]
    public void Sweep_FP64()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_THREADSWEEP") != "1") return;
        int procs = Environment.ProcessorCount;
        bool nativeOk = BlasProvider.IsAvailable;
        _output.WriteLine($"Thread-count sweep (FP64) — host {HardwareFingerprint.Current}, procs={procs}");

        // (1) Global warmup: JIT every path + populate autotune cache for each shape.
        foreach (var s in Shapes)
        {
            var (a, b, c) = Buf(s);
            for (int i = 0; i < 5; i++)
            {
                BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, new BlasOptions<double> { NumThreads = 1 });
                BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, default);
                if (nativeOk) BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N);
            }
        }

        _output.WriteLine($"{"Shape",-26} {"MFLOP",8} {"native",8} | per-thread-count ms (best marked *), best vs 32t");
        foreach (var s in Shapes)
        {
            double mflop = 2.0 * s.M * s.N * s.K / 1e6;
            double nat = nativeOk ? TimeProviderPath(s) : double.NaN;

            var ms = new double[ThreadCounts.Length];
            double best = double.MaxValue; int bestIdx = 0;
            for (int i = 0; i < ThreadCounts.Length; i++)
            {
                ms[i] = TimeManaged(s, ThreadCounts[i]);
                if (ms[i] < best) { best = ms[i]; bestIdx = i; }
            }
            int idx32 = Array.IndexOf(ThreadCounts, 32);
            double speedupVs32 = ms[idx32] / best;

            var cells = new string[ThreadCounts.Length];
            for (int i = 0; i < ThreadCounts.Length; i++)
                cells[i] = $"{ThreadCounts[i]}t={ms[i]:F2}{(i == bestIdx ? "*" : " ")}";

            string bestVsNat = nativeOk ? $"{best / nat:F1}x" : "n/a";
            string natCell = nativeOk ? $"{nat,7:F2}m" : "    n/a";
            _output.WriteLine($"{s.Name,-26} {mflop,8:F0} {natCell} | {string.Join("  ", cells)}  | best={ThreadCounts[bestIdx]}t {speedupVs32:F2}x-vs-32t  best/nat={bestVsNat}");
        }
    }

    private double TimeManaged(Shape s, int threads)
    {
        var (a, b, c) = Buf(s);
        var opts = new BlasOptions<double> { NumThreads = threads };
        // Per-config warmup (every config equally warm).
        for (int i = 0; i < 20; i++) BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, opts);
        int iters = IterFor(s);
        var t = new double[iters]; var sw = new Stopwatch();
        for (int i = 0; i < iters; i++) { sw.Restart(); BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, opts); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[iters / 2];
    }

    // Times the native provider path. TryGemmEx can route to managed or fail, so we
    // assert success — otherwise the number wouldn't be a native run.
    private double TimeProviderPath(Shape s)
    {
        var (a, b, c) = Buf(s);
        for (int i = 0; i < 20; i++)
            if (!BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N))
                throw new InvalidOperationException("TryGemmEx failed during warmup — result would not be a native run.");
        int iters = IterFor(s);
        var t = new double[iters]; var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            if (!BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N))
                throw new InvalidOperationException("TryGemmEx failed during timed run.");
            sw.Stop();
            t[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(t); return t[iters / 2];
    }

    private static int IterFor(Shape s)
    {
        long w = (long)s.M * s.N * s.K;
        return w > 2_000_000_000L ? 30 : w > 200_000_000L ? 60 : 120;
    }

    private static (double[] a, double[] b, double[] c) Buf(Shape s)
    {
        var r = new Random(42);
        var a = new double[s.M * s.K]; var b = new double[s.K * s.N]; var c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = r.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = r.NextDouble() * 2 - 1;
        return (a, b, c);
    }
}
