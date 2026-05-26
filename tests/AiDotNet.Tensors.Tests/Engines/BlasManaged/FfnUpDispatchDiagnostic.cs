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
/// #409 follow-up: investigated the suspected FFN_up (512×2048×512) "dispatch
/// pathology" (the e2e bench showed ~96 ms / 19.5× native).
///
/// <para>
/// <b>Conclusion: there is NO pathology — it was a cold-start measurement
/// artifact.</b> FFN_up was the FIRST shape measured in the e2e bench, so it
/// absorbed the process's one-time costs (JIT of the generic PackBoth&lt;double&gt;
/// path, autotune-cache disk store on the first miss, thread-pool spin-up) — ~50 ms.
/// This diagnostic re-measures <c>default</c> both first (cold) and last (warm):
/// cold ≈ 62 ms, <b>warm ≈ 14 ms = ForcePackBoth ≈ 14 ms</b>. Whichever config is
/// measured first is the slow one, regardless of which it is. Warm steady-state
/// FFN_up is ~14 ms ≈ 3.4× native — in line with the other shapes (1.9–3.4×), not
/// a 20× outlier, and the default path is identical to ForcePackBoth when warm.
/// </para>
///
/// <para>
/// So the earlier "4× dispatch overhead vs #407's 23 ms" was an apples-to-oranges
/// cross-branch + cold-vs-warm comparison, not a real bug. The genuine remaining
/// gap is the general ~2–3× managed-vs-native on multi-threaded mid-size GEMMs
/// (parallelization/strategy efficiency), not an FFN_up-specific dispatch fault.
/// Gated behind AIDOTNET_RUN_FFNUP=1.
/// </para>
/// </summary>
[Trait("Category", "Benchmark")]
[Collection("BlasManaged-Perf-Serial")]
public class FfnUpDispatchDiagnostic
{
    private readonly ITestOutputHelper _output;
    public FfnUpDispatchDiagnostic(ITestOutputHelper output) { _output = output; }

    private const int M = 512, N = 2048, K = 512;

    [Fact]
    public void RootCause_FfnUp_FP64()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_FFNUP") != "1") return;
        bool nativeOk = BlasProvider.IsAvailable;

        _output.WriteLine($"FFN_up {M}x{N}x{K} FP64 — host {HardwareFingerprint.Current}, procs={Environment.ProcessorCount}");
        var heur = AutotuneDispatcher_TryHeuristic();
        _output.WriteLine($"heuristic blocking would be: (mc={heur.mc}, nc={heur.nc}, kc={heur.kc}), strategy={Dispatcher.SelectStrategy<double>(M, N, K, default)}");
        _output.WriteLine("");

        double nat = nativeOk ? TimeNative() : 0;
        double def = TimeManaged(default);
        double single = TimeManaged(new BlasOptions<double> { NumThreads = -1 });
        double t4 = TimeManaged(new BlasOptions<double> { NumThreads = 4 });
        double t8 = TimeManaged(new BlasOptions<double> { NumThreads = 8 });
        double pBoth = TimeManaged(new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth });
        double pAOnly = TimeManaged(new BlasOptions<double> { PackingMode = PackingMode.ForcePackAOnly });
        double strm = TimeManaged(new BlasOptions<double> { PackingMode = PackingMode.ForceStreaming });
        double defLast = TimeManaged(default); // re-measure default at the end (warm) to rule out order/warmup

        if (nativeOk) _output.WriteLine($"native               : {nat:F2} ms");
        _output.WriteLine($"default dispatch     : {def:F2} ms");
        _output.WriteLine($"default (warm, last) : {defLast:F2} ms");
        _output.WriteLine($"default, 1 thread    : {single:F2} ms");
        _output.WriteLine($"default, 4 threads   : {t4:F2} ms");
        _output.WriteLine($"default, 8 threads   : {t8:F2} ms");
        _output.WriteLine($"ForcePackBoth        : {pBoth:F2} ms");
        _output.WriteLine($"ForcePackAOnly       : {pAOnly:F2} ms");
        _output.WriteLine($"ForceStreaming       : {strm:F2} ms");
    }

    private (int mc, int nc, int kc) AutotuneDispatcher_TryHeuristic()
    {
        // Mirror FallbackToHeuristic's defaults for documentation.
        int mc = Math.Min(128, M), nc = Math.Min(512, N), kc = Math.Min(256, K);
        return (mc, nc, kc);
    }

    private double TimeManaged(BlasOptions<double> opts)
    {
        var (a, b, c) = Buf();
        const int iters = 15;
        for (int i = 0; i < 3; i++) BlasManagedLib.Gemm<double>(a, K, false, b, N, false, c, N, M, N, K, opts);
        var t = new double[iters]; var sw = new Stopwatch();
        for (int i = 0; i < iters; i++) { sw.Restart(); BlasManagedLib.Gemm<double>(a, K, false, b, N, false, c, N, M, N, K, opts); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[iters / 2];
    }

    private double TimeNative()
    {
        var (a, b, c) = Buf();
        const int iters = 15;
        for (int i = 0; i < 3; i++) BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N);
        var t = new double[iters]; var sw = new Stopwatch();
        for (int i = 0; i < iters; i++) { sw.Restart(); BlasProvider.TryGemmEx(M, N, K, a, 0, K, false, b, 0, N, false, c, 0, N); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[iters / 2];
    }

    private static (double[] a, double[] b, double[] c) Buf()
    {
        var r = new Random(42);
        var a = new double[M * K]; var b = new double[K * N]; var c = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = r.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = r.NextDouble() * 2 - 1;
        return (a, b, c);
    }
}
