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
/// <param name="BlasManagedP99Ms">99th-percentile wall time (tail latency).</param>
/// <param name="BlasManagedGflops">Achieved GFLOPS at median (2*M*N*K / median_seconds / 1e9).</param>
/// <param name="BlasManagedAllocBytesPerCall">Mean managed bytes allocated per call (GC.GetAllocatedBytesForCurrentThread delta).</param>
/// <param name="NativeMedianMs">Median for OpenBLAS, or -1 when native unavailable.</param>
/// <param name="NativeP95Ms">p95 for OpenBLAS, or -1 when native unavailable.</param>
/// <param name="NativeP99Ms">p99 for OpenBLAS, or -1 when native unavailable.</param>
/// <param name="NativeGflops">Achieved GFLOPS for native, or 0 when unavailable.</param>
/// <param name="NativeAllocBytesPerCall">Mean managed bytes allocated per native call (P/Invoke trampoline only).</param>
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
    double BlasManagedP99Ms,
    double BlasManagedGflops,
    double BlasManagedAllocBytesPerCall,
    double NativeMedianMs,
    double NativeP95Ms,
    double NativeP99Ms,
    double NativeGflops,
    double NativeAllocBytesPerCall,
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
    /// Per-thread allocated bytes. <see cref="GC.GetAllocatedBytesForCurrentThread"/>
    /// is net5.0+ only (absent from this project's net471 profile), so on net471 the
    /// alloc metric is reported as 0 — the perf catalog is a net5.0+ artifact and its
    /// perf tests run there; net471 is build-only.
    /// </summary>
    private static long AllocatedBytes() =>
#if NET5_0_OR_GREATER
        GC.GetAllocatedBytesForCurrentThread();
#else
        0L;
#endif

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

        void Managed() => BlasManagedLib.Gemm<float>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);
        bool nativeOk = BlasProvider.IsAvailable;
        Action? native = null;
        if (nativeOk)
        {
            native = () => BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);
            try { native(); }
            catch { nativeOk = false; native = null; }
        }
        return Measure(s, "Single", iters, Managed, native);
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
        for (int i = 0; i < a.Length; i++) a[i] = (double)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (double)(rng.NextDouble() * 2 - 1);

        void Managed() => BlasManagedLib.Gemm<double>(a, aCols, s.TransA, b, bCols, s.TransB, c, s.N, s.M, s.N, s.K);
        bool nativeOk = BlasProvider.IsAvailable;
        Action? native = null;
        if (nativeOk)
        {
            native = () => BlasProvider.TryGemmEx(s.M, s.N, s.K, a, 0, aCols, s.TransA, b, 0, bCols, s.TransB, c, 0, s.N);
            try { native(); }
            catch { nativeOk = false; native = null; }
        }
        return Measure(s, "Double", iters, Managed, native);
    }

    /// <summary>
    /// Times BlasManaged and (when available) native OpenBLAS on one shape, alternating the two arms round by round.
    /// </summary>
    /// <remarks>
    /// The arms used to be timed one after the other: every BlasManaged call, then every native call. On a shared
    /// machine the load moves between the two windows, so the ratio measured the load as much as the kernels. The same
    /// shape (MobileNetV2_pw_196x96x96) took 1.84x and then 0.84x native's time in two runs a few minutes apart. Alternating puts
    /// both arms under the same load, and the ratio is the median of the per-round ratios. A sample batches enough calls
    /// to last at least <see cref="MinSampleMs"/>, so sub-microsecond shapes are not timer resolution. Allocation is
    /// counted in separate untimed loops.
    /// </remarks>
    internal static ShapeResult Measure(Shape s, string dtype, int iters, Action managed, Action? native)
    {
        for (int i = 0; i < Warmup; i++) { managed(); native?.Invoke(); }

        int calls = CallsPerSample(managed);
        if (native is not null) calls = Math.Max(calls, CallsPerSample(native));

        SettleHeap();
        var bmTimes = new double[iters];
        var nTimes = native is null ? null : new double[iters];
        var ratios = native is null ? null : new double[iters];
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            // Alternate which arm goes first, so neither always runs on the warmer cache.
            bool managedFirst = (i & 1) == 0;
            if (!managedFirst && native is not null) nTimes![i] = TimeCalls(native, calls, sw);
            bmTimes[i] = TimeCalls(managed, calls, sw);
            if (managedFirst && native is not null) nTimes![i] = TimeCalls(native, calls, sw);
            if (ratios is not null) ratios[i] = nTimes![i] > 0 ? bmTimes[i] / nTimes[i] : 0;
        }

        double bmAlloc = AllocPerCall(managed, iters);
        var (bmMedian, bmP95, bmP99) = Percentiles(bmTimes);
        double nativeMedian = -1, nativeP95 = -1, nativeP99 = -1, nativeGflops = 0, nativeAlloc = 0, ratio = 0;
        if (native is not null)
        {
            (nativeMedian, nativeP95, nativeP99) = Percentiles(nTimes!);
            nativeGflops = Gflops(s.M, s.N, s.K, nativeMedian);
            nativeAlloc = AllocPerCall(native, iters);
            Array.Sort(ratios!);
            ratio = ratios![iters / 2];
        }
        return new ShapeResult(
            s.Name, s.M, s.N, s.K, s.TransA, s.TransB, dtype,
            bmMedian, bmP95, bmP99, Gflops(s.M, s.N, s.K, bmMedian), bmAlloc,
            nativeMedian, nativeP95, nativeP99, nativeGflops, nativeAlloc,
            native is not null, ratio);
    }

    /// <summary>Shortest sample a timing round takes, in milliseconds.</summary>
    private const double MinSampleMs = 0.05;

    private static int CallsPerSample(Action run)
    {
        var sw = Stopwatch.StartNew();
        int calls = 0;
        do { run(); calls++; } while (sw.Elapsed.TotalMilliseconds < MinSampleMs && calls < 1_000_000);
        return calls;
    }

    // Milliseconds per call over one sample of `calls` calls.
    private static double TimeCalls(Action run, int calls, Stopwatch sw)
    {
        sw.Restart();
        for (int i = 0; i < calls; i++) run();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / calls;
    }

    private static double AllocPerCall(Action run, int calls)
    {
        SettleHeap();
        long start = AllocatedBytes();
        for (int i = 0; i < calls; i++) run();
        return (AllocatedBytes() - start) / (double)calls;
    }

    private static (double Median, double P95, double P99) Percentiles(double[] times)
    {
        var sorted = (double[])times.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return (sorted[n / 2], sorted[Math.Min(n - 1, (int)(n * 0.95))], sorted[Math.Min(n - 1, (int)(n * 0.99))]);
    }

    private static double Gflops(int m, int n, int k, double medianMs)
    {
        if (medianMs <= 0) return 0;
        double fmas = 2.0 * m * n * k;
        return fmas / (medianMs * 1e-3) / 1e9;
    }

    /// <summary>
    /// Drain pending finalizers and collect both generations before a timing
    /// window so a GC pause from a leaked tape (post-#441 finalizer queue) or
    /// from an earlier test's allocations doesn't fall inside the measurement.
    /// </summary>
    private static void SettleHeap()
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
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
