using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Helpers.Autotune;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;

/// <summary>
/// One row of the perf bench: BlasManaged vs native OpenBLAS for a single shape.
/// </summary>
/// <param name="ShapeName">Stable name from <see cref="Shape.Name"/>.</param>
/// <param name="M">Rows of C.</param>
/// <param name="N">Cols of C.</param>
/// <param name="K">Inner dim.</param>
/// <param name="TransA">Whether A is transposed.</param>
/// <param name="TransB">Whether B is transposed.</param>
/// <param name="Dtype">"Single" or "Double".</param>
/// <param name="BlasManagedMedianMs">Median wall time for BlasManaged.Gemm.</param>
/// <param name="BlasManagedP95Ms">95th-percentile wall time for BlasManaged.Gemm.</param>
/// <param name="NativeMedianMs">Median for OpenBLAS, or -1 when native unavailable.</param>
/// <param name="NativeP95Ms">p95 for OpenBLAS, or -1 when native unavailable.</param>
/// <param name="NativeAvailable">Whether OpenBLAS was loaded and reachable.</param>
/// <param name="RatioBmOverNative">
/// BlasManaged median / Native median. &lt;1 = BlasManaged faster. 0 when
/// native unavailable (no comparison possible).
/// </param>
public record ShapeResult(
    string ShapeName,
    int M, int N, int K,
    bool TransA, bool TransB,
    string Dtype,
    double BlasManagedMedianMs,
    double BlasManagedP95Ms,
    double NativeMedianMs,
    double NativeP95Ms,
    bool NativeAvailable,
    double RatioBmOverNative);

/// <summary>
/// JSON envelope written by <see cref="PerfHarness.RunAll"/>. Captures hardware
/// + git context alongside the per-shape results.
/// </summary>
public record HarnessOutput(
    string GitSha,
    string HardwareFingerprint,
    string TimestampUtc,
    IReadOnlyList<ShapeResult> Shapes);

/// <summary>
/// Sub-issue A (#369) task A.5: per-shape median+p95 GEMM measurement runner.
///
/// <para>
/// <see cref="RunShape"/> measures one shape; <see cref="RunAll"/> sweeps a catalog
/// and writes a JSON envelope to disk. Iteration count scales with work size
/// (10 for large, 30 for medium, 50 for small) to keep total run time reasonable.
/// </para>
///
/// <para>
/// FP32 and FP64 measurements are intentionally split into two methods rather
/// than parameterized over T to avoid generic-dispatch overhead in the hot timing
/// loop — the measurement should reflect kernel cost, not the test infrastructure.
/// </para>
/// </summary>
public static class PerfHarness
{
    private const int Warmup = 3;

    /// <summary>
    /// Measures one shape on both backends. Returns a fully-populated
    /// <see cref="ShapeResult"/> even when native BLAS is unavailable (in which
    /// case <see cref="ShapeResult.NativeAvailable"/> is false and the native
    /// timing fields hold -1).
    /// </summary>
    public static ShapeResult RunShape(Shape s)
    {
        long workEst = (long)s.M * s.N * s.K;
        int iters = workEst > 100_000_000L ? 10 : workEst > 10_000_000L ? 30 : 50;

        return s.Dtype == DType.Single
            ? MeasureSingle(s, iters)
            : MeasureDouble(s, iters);
    }

    /// <summary>
    /// Sweeps the supplied shapes and writes a JSON file at <paramref name="outputPath"/>
    /// containing the <see cref="HarnessOutput"/> envelope. Creates parent directories.
    /// </summary>
    public static void RunAll(IEnumerable<Shape> shapes, string outputPath)
    {
        var results = shapes.Select(RunShape).ToList();
        var output = new HarnessOutput(
            GitSha: GetGitSha(),
            HardwareFingerprint: HardwareFingerprint.Current.ToString(),
            TimestampUtc: DateTime.UtcNow.ToString("u"),
            Shapes: results);

        var dir = Path.GetDirectoryName(outputPath);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        File.WriteAllText(outputPath, JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
    }

    private static ShapeResult MeasureSingle(Shape s, int iters)
    {
        int aRows = s.TransA ? s.K : s.M;
        int aCols = s.TransA ? s.M : s.K;
        int bRows = s.TransB ? s.N : s.K;
        int bCols = s.TransB ? s.K : s.N;
        var rng = new Random(42);
        var a = new float[aRows * aCols];
        var b = new float[bRows * bCols];
        var c = new float[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        for (int i = 0; i < Warmup; i++)
            BlasManagedLib.Gemm<float>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);

        var bmTimes = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<float>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);
            sw.Stop();
            bmTimes[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(bmTimes);
        double bmMedian = bmTimes[iters / 2];
        double bmP95 = bmTimes[Math.Min(iters - 1, (int)(iters * 0.95))];

        double nativeMedian = -1, nativeP95 = -1;
        bool nativeOk = BlasProvider.IsAvailable;
        if (nativeOk)
        {
            try
            {
                for (int i = 0; i < Warmup; i++)
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);

                var nTimes = new double[iters];
                for (int i = 0; i < iters; i++)
                {
                    sw.Restart();
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);
                    sw.Stop();
                    nTimes[i] = sw.Elapsed.TotalMilliseconds;
                }
                Array.Sort(nTimes);
                nativeMedian = nTimes[iters / 2];
                nativeP95 = nTimes[Math.Min(iters - 1, (int)(iters * 0.95))];
            }
            catch
            {
                nativeOk = false;
            }
        }

        double ratio = (nativeOk && nativeMedian > 0) ? bmMedian / nativeMedian : 0.0;
        return new ShapeResult(
            s.Name, s.M, s.N, s.K, s.TransA, s.TransB, "Single",
            bmMedian, bmP95, nativeMedian, nativeP95, nativeOk, ratio);
    }

    private static ShapeResult MeasureDouble(Shape s, int iters)
    {
        int aRows = s.TransA ? s.K : s.M;
        int aCols = s.TransA ? s.M : s.K;
        int bRows = s.TransB ? s.N : s.K;
        int bCols = s.TransB ? s.K : s.N;
        var rng = new Random(42);
        var a = new double[aRows * aCols];
        var b = new double[bRows * bCols];
        var c = new double[s.M * s.N];
        for (int i = 0; i < a.Length; i++) a[i] = rng.NextDouble() * 2 - 1;
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextDouble() * 2 - 1;

        for (int i = 0; i < Warmup; i++)
            BlasManagedLib.Gemm<double>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);

        var bmTimes = new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            BlasManagedLib.Gemm<double>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);
            sw.Stop();
            bmTimes[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(bmTimes);
        double bmMedian = bmTimes[iters / 2];
        double bmP95 = bmTimes[Math.Min(iters - 1, (int)(iters * 0.95))];

        double nativeMedian = -1, nativeP95 = -1;
        bool nativeOk = BlasProvider.IsAvailable;
        if (nativeOk)
        {
            try
            {
                for (int i = 0; i < Warmup; i++)
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);

                var nTimes = new double[iters];
                for (int i = 0; i < iters; i++)
                {
                    sw.Restart();
                    BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);
                    sw.Stop();
                    nTimes[i] = sw.Elapsed.TotalMilliseconds;
                }
                Array.Sort(nTimes);
                nativeMedian = nTimes[iters / 2];
                nativeP95 = nTimes[Math.Min(iters - 1, (int)(iters * 0.95))];
            }
            catch
            {
                nativeOk = false;
            }
        }

        double ratio = (nativeOk && nativeMedian > 0) ? bmMedian / nativeMedian : 0.0;
        return new ShapeResult(
            s.Name, s.M, s.N, s.K, s.TransA, s.TransB, "Double",
            bmMedian, bmP95, nativeMedian, nativeP95, nativeOk, ratio);
    }

    private static string GetGitSha()
    {
        try
        {
            var psi = new ProcessStartInfo("git", "rev-parse HEAD")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using var p = Process.Start(psi);
            if (p == null) return "unknown";
            string sha = p.StandardOutput.ReadToEnd().Trim();
            p.WaitForExit(2000);
            return string.IsNullOrEmpty(sha) ? "unknown" : sha;
        }
        catch
        {
            return "unknown";
        }
    }
}
