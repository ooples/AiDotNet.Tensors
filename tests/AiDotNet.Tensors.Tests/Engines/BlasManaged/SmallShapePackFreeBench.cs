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
/// Sub-C (#371) diagnostic: current gap vs native OpenBLAS for the three small
/// shapes the issue targets, comparing the default dispatch (which routes K=64
/// shapes to PackAOnly) against a forced pack-free Streaming path. Decides
/// whether routing these shapes pack-free actually helps.
///
/// <para>Gated behind <c>AIDOTNET_RUN_371_BENCH=1</c>, native BLAS required.</para>
/// </summary>
[Trait("Category", "Benchmark")]
[Collection("BlasManaged-Perf-Serial")]
public class SmallShapePackFreeBench
{
    private readonly ITestOutputHelper _output;
    public SmallShapePackFreeBench(ITestOutputHelper output) { _output = output; }

    private sealed record ShapeSpec(string Name, int M, int N, int K);

    private static readonly ShapeSpec[] Shapes =
    [
        new("Tiny_32sq",   32,  32,  32),
        new("Tiny_64sq",   64,  64,  64),
        new("WideFat_512x512x64", 512, 512, 64),
    ];

    [Fact]
    public void Sub_C_SmallShape_Gap_And_PackFree_Comparison()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_371_BENCH") != "1") return;
        if (!BlasProvider.IsAvailable)
        {
            _output.WriteLine("Native BLAS not loaded — skipping.");
            return;
        }

        _output.WriteLine($"Sub-C (#371) small-shape gap (FP64) — {DateTime.UtcNow:u}, host {HardwareFingerprint.Current}");
        _output.WriteLine($"{"Shape",-22} {"native",9} {"auto",9} {"strm-par",10} {"strm-ser",10} {"auto/nat",9} {"par/nat",9} {"ser/nat",9} {"autoStrat",-16} {"packed?",-9}");
        _output.WriteLine(new string('-', 120));

        foreach (var s in Shapes)
        {
            var autoStrat = Dispatcher.SelectStrategy<double>(s.M, s.N, s.K, default);
            bool tinyPath = (long)s.M * s.N * s.K <= BlasManagedLib.TinyShapeWorkThreshold;

            double nativeMs = TimeNative(s);
            (double autoMs, bool packed) = TimeManaged(s, default);
            (double strmMs, _) = TimeManaged(s, new BlasOptions<double> { PackingMode = PackingMode.ForceStreaming });
            (double serialMs, _) = TimeManaged(s, new BlasOptions<double> { PackingMode = PackingMode.ForceStreaming, NumThreads = -1 });

            _output.WriteLine(
                $"{s.Name,-22} {nativeMs,8:F4}m {autoMs,8:F4}m {strmMs,9:F4}m {serialMs,9:F4}m " +
                $"{autoMs / nativeMs,8:F1}x {strmMs / nativeMs,8:F1}x {serialMs / nativeMs,8:F1}x " +
                $"{(tinyPath ? "tiny→streaming" : autoStrat.ToString()),-16} {(packed ? "yes" : "no"),-11}");
        }
        _output.WriteLine("");
        _output.WriteLine("auto = default dispatch; streaming = forced pack-free; ratios vs native OpenBLAS.");
    }

    private double TimeNative(ShapeSpec s)
    {
        var (a, b, c) = MakeBuffers(s);
        int iters = Iters(s);
        for (int i = 0; i < 5; i++)
            BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N);
        var times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, s.K, false, b, 0, s.N, false, c, 0, s.N);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        return times[iters / 2];
    }

    private (double Median, bool Packed) TimeManaged(ShapeSpec s, BlasOptions<double> opts)
    {
        var (a, b, c) = MakeBuffers(s);
        int iters = Iters(s);

        for (int i = 0; i < 5; i++)
            BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, opts);

        // Detect whether a pack happened: pack-cache miss count rises on an
        // actual pack/allocate, stays flat on the pack-free streaming path.
        BlasManagedStatsTracker.Reset();
        BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, opts);
        bool packed = BlasManagedStatsTracker.Snapshot().PackCacheMisses > 0;

        var times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K, opts);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(times);
        return (times[iters / 2], packed);
    }

    private static (double[] a, double[] b, double[] c) MakeBuffers(ShapeSpec s)
    {
        var rng = new Random(42);
        var a = new double[s.M * s.K];
        var b = new double[s.K * s.N];
        var c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;
        return (a, b, c);
    }

    private static int Iters(ShapeSpec s)
    {
        long work = (long)s.M * s.N * s.K;
        return work > 5_000_000L ? 200 : 2000;
    }
}
