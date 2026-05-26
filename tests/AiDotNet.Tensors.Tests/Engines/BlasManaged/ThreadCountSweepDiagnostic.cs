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
/// <b>Findings (Ryzen 9 3950X, FP64).</b> The default (all 32 threads) is often
/// suboptimal, and near-native is reachable at the right thread count:
/// <list type="bullet">
///   <item>Large_1024sq: 32t=19.8 ms vs <b>2t=11.3 ms (1.76× faster, 1.1× native)</b></item>
///   <item>BERT_FFN: 4–16t pathologically slow (44 ms) but <b>2t=27.4 ms = 1.0× native</b></item>
///   <item>FFN_up: <b>wants 32t</b> (10 ms, 2.2× native) — opposite preference</item>
/// </list>
/// The optimum is shape- AND hardware-dependent and NON-monotonic (the 4–16t dip
/// is shared-packed-B bandwidth contention + this chip's 4-CCD/NUMA placement).
/// A static heuristic can't capture it without regressing one shape to fix
/// another (FFN_up vs Large are both ≈1 GFLOP with opposite optima). <b>The
/// robust fix is to autotune the thread count by measurement</b> — extend
/// <c>BlockSizeSweep</c> (the #407 measurement autotune) to also sweep thread
/// count and cache the per-shape winner. This diagnostic is the evidence the
/// lever is real and worth ~1.05–1.76× on these shapes.
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
            double nat = nativeOk ? TimeNative(s) : 0;

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

            _output.WriteLine($"{s.Name,-26} {mflop,8:F0} {nat,7:F2}m | {string.Join("  ", cells)}  | best={ThreadCounts[bestIdx]}t {speedupVs32:F2}x-vs-32t  best/nat={best / nat:F1}x");
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

    private double TimeNative(Shape s)
    {
        var (a, b, c) = Buf(s);
        for (int i = 0; i < 20; i++) BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N);
        int iters = IterFor(s);
        var t = new double[iters]; var sw = new Stopwatch();
        for (int i = 0; i < iters; i++) { sw.Restart(); BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
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
