using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// #375 G12/G13 guards. Perf-tagged (excluded from the CI correctness filter, run in the
/// perf pipeline) because they assert wall-clock behaviour.
/// </summary>
[Trait("Category", "Performance")]
[Collection("BlasManaged-Stats-Serial")]
public class HybridStrategyPerfTests
{
    /// <summary>
    /// G13: SelectStrategy must stay sub-µs. The hybrid added a learned-cache consult on the
    /// dispatch path; without the in-memory memo it did a File.Exists+ReadAllText+JSON parse
    /// per call (~77 µs measured). The memo makes the disk read at-most-once-per-shape. This
    /// guards against the disk-per-call regression returning (gate at 10 µs — generous, but
    /// 77 µs would fail it by 8×; the fixed path is ~0.8 µs).
    /// </summary>
    [Fact]
    public void SelectStrategy_HotPath_StaysFast()
    {
        var opts = default(BlasOptions<float>);
        for (int w = 0; w < 1000; w++)
            Dispatcher.SelectStrategy<float>(197, 197, 64, false, true, in opts);
        const int iters = 100_000;
        var times = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            Dispatcher.SelectStrategy<float>(197, 197, 64, false, true, in opts);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMicroseconds;
        }
        Array.Sort(times);
        double medianUs = times[iters / 2];
        Assert.True(medianUs < 10.0,
            $"SelectStrategy median {medianUs:F2} µs — disk-read-per-call regression (memo broken?).");
    }

    /// <summary>
    /// G12: the table's routed strategy must be no worse than the alternatives on the catalog
    /// (a hand-seeded table entry can't ship a regression). For each shape, measure the table's
    /// choice vs the other two strategies; assert table ≤ best × 1.25 (wall-clock tolerance).
    /// transB shapes (the ones the hybrid governs) on the current fingerprint.
    /// </summary>
    [Theory]
    [InlineData(96, 128, 64)]
    [InlineData(128, 128, 128)]
    [InlineData(512, 512, 64)]
    public void Table_Routing_NoWorse_Than_Alternatives_FP64(int M, int N, int K)
    {
        var rng = new Random(11);
        var a = new double[M * K];
        var b = new double[N * K]; // [N,K] transB
        var c = new double[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        double Time(PackingMode mode)
        {
            var opts = new BlasOptions<double> { PackingMode = mode };
            for (int w = 0; w < 5; w++) BlasManagedLib.Gemm<double>(a, K, false, b, K, true, c, N, M, N, K, opts);
            double best = double.MaxValue;
            for (int w = 0; w < 7; w++)
            {
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < 20; i++) BlasManagedLib.Gemm<double>(a, K, false, b, K, true, c, N, M, N, K, opts);
                sw.Stop();
                best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
            }
            return best;
        }

        var chosen = Dispatcher.SelectStrategy<double>(M, N, K, false, true, default);
        double chosenMs = Time(chosen);
        double bestAltMs = double.MaxValue;
        foreach (var mode in new[] { PackingMode.ForceStreaming, PackingMode.ForcePackBoth })
            bestAltMs = Math.Min(bestAltMs, Time(mode));

        Assert.True(chosenMs <= bestAltMs * 1.25,
            $"{M}x{N}x{K}: table chose {chosen} at {chosenMs:F3} ms but best alternative was {bestAltMs:F3} ms (>1.25×).");
    }
}
