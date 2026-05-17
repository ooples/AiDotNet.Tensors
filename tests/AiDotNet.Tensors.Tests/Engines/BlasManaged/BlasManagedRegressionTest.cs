using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

[Collection("BlasManaged-Perf-Serial")]
public class BlasManagedRegressionTest
{
    private readonly ITestOutputHelper _output;

    public BlasManagedRegressionTest(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Tolerance: a 50% slowdown vs the baseline median is considered a regression.
    /// This generous margin is intentional — the baseline values themselves already
    /// capture the 95th-percentile observed across multiple runs on a noisy dev host,
    /// so the effective detection threshold is ~3× the typical kernel median before
    /// a failure fires. The goal is to catch large regressions (rewritten kernels,
    /// algorithm changes) without flaking on scheduler jitter.
    /// </summary>
    private const double RegressionToleranceFactor = 1.50;

    /// <summary>
    /// Hard floor: if a shape's baseline median is below this threshold (very fast),
    /// don't enforce the relative check — measurement noise dominates at sub-ms resolution.
    /// </summary>
    private const double NoiseFloorMs = 1.0;

    [Fact]
    public void BlasManaged_RepresentativeShapes_NoRegressionFromBaseline()
    {
        // Locate the baseline JSON. It's deployed alongside the test assembly.
        string assemblyDir = Path.GetDirectoryName(typeof(BlasManagedRegressionTest).Assembly.Location)!;
        string baselinePath = Path.Combine(assemblyDir, "baselines", "blas-managed-baseline.json");

        if (!File.Exists(baselinePath))
        {
            _output.WriteLine($"Baseline file not found at {baselinePath} — skipping regression check.");
            return;
        }

        string json = File.ReadAllText(baselinePath);
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        var baselineShapes = root.GetProperty("shapes");

        var failures = new List<string>();

        _output.WriteLine($"  {"Shape",-30}  {"Baseline",10}  {"Current",10}  {"Threshold",10}  Verdict");
        _output.WriteLine(new string('-', 80));

        foreach (var baseline in baselineShapes.EnumerateArray())
        {
            string id = baseline.GetProperty("id").GetString()!;
            int m = baseline.GetProperty("m").GetInt32();
            int n = baseline.GetProperty("n").GetInt32();
            int k = baseline.GetProperty("k").GetInt32();
            bool transA = baseline.GetProperty("trans_a").GetBoolean();
            bool transB = baseline.GetProperty("trans_b").GetBoolean();
            double baselineMs = baseline.GetProperty("median_ms").GetDouble();

            double currentMs = MeasureMedianMs(m, n, k, transA, transB);

            // Below the noise floor: absolute numbers are dominated by timer resolution,
            // not kernel performance. Skip the relative check and just report.
            double thresholdMs = Math.Max(baselineMs * RegressionToleranceFactor, NoiseFloorMs);
            string verdict = currentMs <= thresholdMs ? "OK" : "REGRESSION";
            _output.WriteLine(
                $"  {id,-30}  {baselineMs,7:F3} ms  {currentMs,7:F3} ms  {thresholdMs,7:F3} ms  {verdict}");

            if (currentMs > thresholdMs)
            {
                failures.Add(
                    $"{id}: {currentMs:F3} ms > threshold {thresholdMs:F3} ms " +
                    $"(baseline {baselineMs:F3} ms × {RegressionToleranceFactor})");
            }
        }

        if (failures.Count > 0)
        {
            throw new Xunit.Sdk.XunitException(
                "BlasManaged regressed vs baseline on " + failures.Count + " shape(s):\n" +
                string.Join("\n", failures) + "\n\n" +
                "If this is a legitimate performance change (new hardware, intentional refactor),\n" +
                "re-capture the baseline by running the test, observing the 'Current' column,\n" +
                "and updating tests/AiDotNet.Tensors.Tests/Engines/BlasManaged/baselines/blas-managed-baseline.json.");
        }
    }

    private static double MeasureMedianMs(int m, int n, int k, bool transA, bool transB)
    {
        const int warmupIters = 3;
        const int measureIters = 30;

        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;

        var rng = new Random(42);
        double[] a = new double[aRows * aCols];
        double[] b = new double[bRows * bCols];
        double[] c = new double[m * n];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        int lda = aCols;
        int ldb = bCols;

        for (int i = 0; i < warmupIters; i++)
        {
            BlasManagedLib.Gemm<double>(a, lda, transA, b, ldb, transB, c, ldc: n, m, n, k);
        }

        double[] times = new double[measureIters];
        var sw = new Stopwatch();
        for (int i = 0; i < measureIters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<double>(a, lda, transA, b, ldb, transB, c, ldc: n, m, n, k);
            sw.Stop();
            times[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(times);
        return times[measureIters / 2];
    }
}
