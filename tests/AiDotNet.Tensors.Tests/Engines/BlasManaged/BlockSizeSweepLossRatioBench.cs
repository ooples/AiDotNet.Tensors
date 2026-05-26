using System;
using System.Collections.Generic;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-Q (#407) acceptance benchmark: does measurement-based block-size
/// autotune actually reduce the loss ratio vs native BLAS relative to the
/// heuristic block sizes?
///
/// <para>
/// For each compute-bound shape, times (1) native dgemm, (2) managed GEMM with
/// the heuristic block sizes (Mc,Nc,Kc)=(128,512,256)-clamped, and (3) managed
/// GEMM with the measured-autotune winner. Both managed variants are forced via
/// <see cref="AutotuneDispatcher.BlockOverride"/> so the only difference is the
/// blocking — an apples-to-apples isolation of the autotune's contribution.
/// Reports per-shape and median loss ratios (managed / native).
/// </para>
///
/// <para>
/// Reporting only (no assertions beyond "ran"); gated behind
/// <c>AIDOTNET_RUN_407_BENCH=1</c> and skipped if native BLAS isn't loaded.
/// </para>
/// </summary>
[Trait("Category", "Benchmark")]
[Collection("BlasManaged-Perf-Serial")]
public class BlockSizeSweepLossRatioBench
{
    private readonly ITestOutputHelper _output;
    public BlockSizeSweepLossRatioBench(ITestOutputHelper output) { _output = output; }

    private sealed record ShapeSpec(string Name, int M, int N, int K);

    // Compute-bound FFN / projection / square shapes where the block choice
    // dominates memory traffic (the shapes #407 calls out: FFN, QKV).
    private static readonly ShapeSpec[] Shapes =
    [
        new("FFN_128x768x768",        128,  768,  768),
        new("FFN_up_512x2048x512",    512,  2048, 512),
        new("FFN_down_512x512x2048",  512,  512,  2048),
        new("BERT_FFN_1024x3072x768", 1024, 3072, 768),
        new("GPT2_FFN_1024x3072x768", 1024, 3072, 768),
        new("ViT_QKV_768x2304x768",   768,  2304, 768),
        new("Large_1024sq",           1024, 1024, 1024),
    ];

    [Fact]
    public void Sub_Q_AutotuneVsHeuristic_LossRatioVsNative()
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_RUN_407_BENCH") != "1") return;
        if (!BlasProvider.IsAvailable)
        {
            _output.WriteLine("Native BLAS not loaded on this host — cannot compute loss ratio. Skipping.");
            return;
        }

        _output.WriteLine($"Sub-Q (#407) loss-ratio bench (FP64) — {DateTime.UtcNow:u}");
        _output.WriteLine($"Host: {HardwareFingerprint.Current}");
        _output.WriteLine("");
        _output.WriteLine($"{"Shape",-26} {"native",9} {"heuristic",10} {"autotune",10} {"lossH",7} {"lossA",7} {"winner(Mc,Nc,Kc)",-20}");
        _output.WriteLine(new string('-', 100));

        var lossH = new List<double>();
        var lossA = new List<double>();

        foreach (var s in Shapes)
        {
            double nativeMs = TimeNative(s);

            // Heuristic blocking: the BLIS default (clamped by Decide to the real mr/nr).
            double heurMs = TimeManaged(s, (128, 512, 256));

            // Measured-autotune winner.
            var winner = BlockSizeSweep.Measure<double>(
                s.M, s.N, s.K, transA: false, transB: false,
                mr: 4, nr: 8, procs: Environment.ProcessorCount, isDeterministic: false);
            double autoMs = TimeManaged(s, (winner.Mc, winner.Nc, winner.Kc));

            double rH = nativeMs > 0 ? heurMs / nativeMs : 0;
            double rA = nativeMs > 0 ? autoMs / nativeMs : 0;
            lossH.Add(rH);
            lossA.Add(rA);

            _output.WriteLine(
                $"{s.Name,-26} {nativeMs,8:F3}m {heurMs,9:F3}m {autoMs,9:F3}m {rH,6:F1}x {rA,6:F1}x  ({winner.Mc},{winner.Nc},{winner.Kc})");
        }

        _output.WriteLine("");
        _output.WriteLine($"Median loss ratio — heuristic: {Median(lossH):F1}x   autotune: {Median(lossA):F1}x");
        _output.WriteLine("(#407 acceptance: median loss ratio drops toward <= 8x vs the heuristic baseline.)");
    }

    private double TimeManaged(ShapeSpec s, (int Mc, int Nc, int Kc) block)
    {
        var (a, b, c) = MakeBuffers(s);
        int iters = Iters(s);

        AutotuneDispatcher.BlockOverride = block;
        try
        {
            for (int i = 0; i < 3; i++)
                BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K);

            var times = new double[iters];
            var sw = new Stopwatch();
            for (int i = 0; i < iters; i++)
            {
                sw.Restart();
                BlasManagedLib.Gemm<double>(a, s.K, false, b, s.N, false, c, s.N, s.M, s.N, s.K);
                sw.Stop();
                times[i] = sw.Elapsed.TotalMilliseconds;
            }
            Array.Sort(times);
            return times[iters / 2];
        }
        finally { AutotuneDispatcher.BlockOverride = null; }
    }

    private double TimeNative(ShapeSpec s)
    {
        var (a, b, c) = MakeBuffers(s);
        int iters = Iters(s);

        for (int i = 0; i < 3; i++)
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
        return work > 100_000_000L ? 10 : work > 10_000_000L ? 20 : 40;
    }

    private static double Median(List<double> xs)
    {
        if (xs.Count == 0) return 0;
        xs.Sort();
        int mid = xs.Count / 2;
        return (xs.Count & 1) == 1 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2.0;
    }
}
