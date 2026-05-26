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
/// #409: attribute the mid-size PackBoth managed-vs-OpenBLAS gap by phase. Forces
/// true serial via <c>CpuParallelSettings.MaxDegreeOfParallelism=1</c> (the
/// ic-loop uses ParallelForOrSerial, NOT options.NumThreads), with
/// <see cref="PackBothProfiler"/> on, splitting wall time into pack-B / pack-A /
/// microkernel / other (C-clear + dispatch + rent + epilogue).
/// Gated AIDOTNET_RUN_PACKPROFILE=1.
///
/// <para>
/// <b>Result (Ryzen 9 3950X, FP64, single-thread serial).</b> The microkernel
/// compute is <b>88–91% of PackBoth time</b>; packing is only ~10% (pack-A ~5%,
/// pack-B ~3–5%) and dispatch/C-clear/epilogue ~1–3%:
/// <list type="bullet">
///   <item>Large_1024sq: packB 3% / packA 5% / <b>kernel 91%</b> / other 1%</item>
///   <item>BERT_FFN:     packB 3% / packA 5% / <b>kernel 91%</b> / other 2%</item>
///   <item>FFN_up:       packB 5% / packA 5% / <b>kernel 88%</b> / other 3%</item>
/// </list>
/// <b>Conclusion: the managed-vs-OpenBLAS gap is the MICROKERNEL, not packing,
/// dispatch, or C-traffic.</b> Pack/dispatch optimization would be wasted effort
/// (~10% combined). The single-thread kernel runs ~38 GFLOPS (Large: 56.9 ms for
/// 2.15 GFLOP), matching the microkernel bench — i.e. 80% of the managed FMA
/// ceiling. The residual gap to OpenBLAS is fundamental managed-kernel-vs-asm
/// (~38 vs ~60 GFLOPS single-thread) plus parallel scaling — both hard, neither
/// addressable by the pack/dispatch path.
/// </para>
/// </summary>
[Trait("Category", "Benchmark")]
[Collection("BlasManaged-Perf-Serial")]
public class PackBothPhaseProfile
{
    private readonly ITestOutputHelper _output;
    public PackBothPhaseProfile(ITestOutputHelper output) { _output = output; }

    private sealed record Shape(string Name, int M, int N, int K);

    private static readonly Shape[] Shapes =
    [
        new("FFN_up_512x2048x512",    512,  2048, 512),
        new("FFN_down_512x512x2048",  512,  512,  2048),
        new("Large_1024sq",           1024, 1024, 1024),
        new("BERT_FFN_1024x3072x768", 1024, 3072, 768),
    ];

    [Fact]
    public void Profile_FP64()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_PACKPROFILE") != "1") return;
        bool nativeOk = BlasProvider.IsAvailable;
        _output.WriteLine($"PackBoth serial phase profile (FP64, 1 thread) — host {HardwareFingerprint.Current}");
        _output.WriteLine($"{"Shape",-26} {"wall",7} {"packB",13} {"packA",13} {"kernel",13} {"other",13} | native(MT) ser/nat");
        _output.WriteLine(new string('-', 110));

        var serialBoth = new BlasOptions<double> { PackingMode = PackingMode.ForcePackBoth, NumThreads = 1 };

        // PackBoth's ic-loop uses CpuParallelSettings.ParallelForOrSerial (NOT
        // options.NumThreads), so force true serial here for a clean per-phase
        // breakdown. Native (OpenBLAS) has its own threading and is unaffected.
        int savedMaxDop = CpuParallelSettings.MaxDegreeOfParallelism;
        CpuParallelSettings.MaxDegreeOfParallelism = 1;
        try
        {
        foreach (var s in Shapes)
        {
            var (a, b, c) = Buf(s);
            // Warm.
            for (int i = 0; i < 10; i++) BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, serialBoth);

            int iters = s.M * (long)s.N * s.K > 2_000_000_000L ? 20 : 40;
            PackBothProfiler.Enabled = true;
            PackBothProfiler.Reset();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++)
                BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, serialBoth);
            sw.Stop();
            PackBothProfiler.Enabled = false;

            double wall = sw.Elapsed.TotalMilliseconds / iters;
            double pb = PackBothProfiler.PackBMs / iters;
            double pa = PackBothProfiler.PackAMs / iters;
            double kn = PackBothProfiler.KernelMs / iters;
            double other = wall - pb - pa - kn;

            double nat = nativeOk ? TimeNative(s) : 0;

            _output.WriteLine(
                $"{s.Name,-26} {wall,6:F2}m {Pct(pb, wall),13} {Pct(pa, wall),13} {Pct(kn, wall),13} {Pct(other, wall),13} | {nat,7:F2}m {wall / nat,5:F1}x");
        }
        }
        finally { CpuParallelSettings.MaxDegreeOfParallelism = savedMaxDop; }
        _output.WriteLine("");
        _output.WriteLine("Note: 'wall' is single-thread serial; native is multi-threaded. The phase split shows WHERE");
        _output.WriteLine("the managed serial path spends time (kernel = the irreducible compute floor).");
    }

    private static string Pct(double ms, double wall) => $"{ms,6:F2}({100 * ms / wall,3:F0}%)";

    private double TimeNative(Shape s)
    {
        var (a, b, c) = Buf(s);
        for (int i = 0; i < 15; i++) BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N);
        const int iters = 30; var t = new double[iters]; var sw = new Stopwatch();
        for (int i = 0; i < iters; i++) { sw.Restart(); BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N); sw.Stop(); t[i] = sw.Elapsed.TotalMilliseconds; }
        Array.Sort(t); return t[iters / 2];
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
