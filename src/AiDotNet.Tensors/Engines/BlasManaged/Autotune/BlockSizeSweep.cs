using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Helpers.Autotune;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-Q (#407): measurement-based (Mc, Nc, Kc) block-size autotune.
///
/// <para>
/// On an autotune cache miss for a large shape, <see cref="AutotuneDispatcher"/>
/// calls <see cref="Measure{T}"/>. This generates a candidate set of blocking
/// tuples (cross product of the BLIS-style Mc/Nc/Kc ranges, clamped + aligned to
/// the shape), times the managed <see cref="BlasManagedLib.Gemm{T}"/> on each,
/// and returns the fastest. The dispatcher caches the winner so the
/// measurement cost is paid once per (shape, host) and amortized across every
/// future call (and across processes via the on-disk autotune cache).
/// </para>
///
/// <para>
/// The probe times the real managed GEMM with the candidate blocking injected
/// via <see cref="AutotuneDispatcher.BlockOverride"/> (a ThreadStatic that also
/// guards against recursive re-measurement). It follows the
/// <see cref="PrefersManagedCache"/> probe pattern: ArrayPool buffers, a warmup
/// call, then a median of several timed runs to damp noise.
/// </para>
/// </summary>
internal static class BlockSizeSweep
{
    // Candidate ranges from #407. Cross-producted, then each tuple is clamped to
    // the shape and aligned to (mr, nr); duplicates collapse (e.g. on moderate
    // shapes many candidates clamp to the same values), so the effective count
    // is usually far smaller than 3 × 4 × 4.
    private static readonly int[] McCandidates = { 64, 128, 192 };
    private static readonly int[] KcCandidates = { 128, 256, 384, 512 };

    // Median of this many timed runs per candidate (after one warmup). Kept
    // small because each run is a full GEMM of the (large) shape — the winner is
    // stable at 3 with the min-of-runs tie-break below.
    private const int Runs = 3;

    /// <summary>
    /// Sub-Q (#407): true when offline pre-population was requested via
    /// <c>AIDOTNET_AUTOTUNE_OFFLINE=1</c>. Consumers that want zero first-call
    /// measurement overhead in production should pair this with a warmup call to
    /// <see cref="PrepopulateCommonShapes"/> (e.g. at app startup).
    /// </summary>
    public static bool OfflineRequested { get; } =
        Environment.GetEnvironmentVariable("AIDOTNET_AUTOTUNE_OFFLINE") == "1";

    /// <summary>
    /// Sub-Q (#407): warm the autotune cache for a curated set of common GEMM
    /// shapes (transformer FFN / attention-projection / MLP at typical dims),
    /// for both float and double, by running each through the real
    /// <see cref="BlasManagedLib.Gemm{T}"/> path with measurement forced on. The
    /// measured winners are stored via the normal dispatcher path, so every
    /// future call of those shapes is a cache hit with no first-call cost —
    /// including across processes (the autotune cache persists to disk).
    ///
    /// <para>
    /// Intended to be called once during application warmup (it is a no-op for
    /// shapes already cached). Safe to call regardless of the
    /// <c>AIDOTNET_AUTOTUNE_OFFLINE</c> / <c>AIDOTNET_BLAS_AUTOTUNE_MEASURE</c>
    /// env vars — it forces measurement for its own warmup calls only, on the
    /// calling thread.
    /// </para>
    /// </summary>
    public static void PrepopulateCommonShapes()
    {
        AutotuneDispatcher.ForceMeasureOnMiss = true;
        try
        {
            foreach (var (m, n, k) in CommonShapes)
            {
                WarmOne<float>(m, n, k);
                WarmOne<double>(m, n, k);
            }
        }
        finally
        {
            AutotuneDispatcher.ForceMeasureOnMiss = false;
        }
    }

    // Curated common shapes (M, N, K), trans-free, drawn from the transformer
    // FFN / QKV-projection / MLP-head families the AIsEval + #436 work exercises.
    private static readonly (int M, int N, int K)[] CommonShapes =
    {
        (512, 2048, 512),    // FFN up-projection (d=512)
        (512, 512, 2048),    // FFN down-projection (d=512)
        (1024, 3072, 768),   // BERT-base FFN up (d=768)
        (1024, 768, 3072),   // BERT-base FFN down
        (768, 2304, 768),    // fused QKV projection (d=768)
        (256, 1024, 256),    // small MLP hidden
        (128, 512, 784),     // MNIST-MLP first layer (bs=128)
    };

    private static void WarmOne<T>(int m, int n, int k) where T : unmanaged
    {
        int aLen = m * k, bLen = k * n, cLen = m * n;
        var a = ArrayPool<T>.Shared.Rent(aLen);
        var b = ArrayPool<T>.Shared.Rent(bLen);
        var c = ArrayPool<T>.Shared.Rent(cLen);
        try
        {
            FillRandom<T>(a.AsSpan(0, aLen));
            FillRandom<T>(b.AsSpan(0, bLen));
            // Real dispatch path → Decide → (miss + forced measure) → Store.
            BlasManagedLib.Gemm<T>(
                new ReadOnlySpan<T>(a, 0, aLen), k, false,
                new ReadOnlySpan<T>(b, 0, bLen), n, false,
                new Span<T>(c, 0, cLen), n,
                m, n, k);
        }
        catch
        {
            // Warmup is best-effort — a failure for one shape must not abort the rest.
        }
        finally
        {
            ArrayPool<T>.Shared.Return(a);
            ArrayPool<T>.Shared.Return(b);
            ArrayPool<T>.Shared.Return(c);
        }
    }

    /// <summary>
    /// Benchmark candidate blockings for the given shape and return the fastest.
    /// </summary>
    internal static (ParallelismAxis Axis, int Mc, int Nc, int Kc, int ThreadCount, double MeasuredMs)
        Measure<T>(
            int m, int n, int k,
            bool transA, bool transB,
            int mr, int nr,
            int procs,
            bool isDeterministic) where T : unmanaged
    {
        var axis = AxisSelector.Select(m, n, k, mr, nr, procs, isDeterministic);
        var candidates = GenerateCandidates(m, n, k, mr, nr);

        // Buffer dimensions mirror the caller's GEMM (row-major, contiguous).
        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;
        int aLen = aRows * aCols;
        int bLen = bRows * bCols;
        int cLen = m * n;
        int lda = aCols, ldb = bCols, ldc = n;

        var a = ArrayPool<T>.Shared.Rent(aLen);
        var b = ArrayPool<T>.Shared.Rent(bLen);
        var c = ArrayPool<T>.Shared.Rent(cLen);
        try
        {
            FillRandom<T>(a.AsSpan(0, aLen));
            FillRandom<T>(b.AsSpan(0, bLen));

            (int Mc, int Nc, int Kc) best = candidates[0];
            double bestMs = double.MaxValue;

            foreach (var cand in candidates)
            {
                double ms = TimeCandidate<T>(cand, a, aLen, lda, transA, b, bLen, ldb, transB, c, cLen, ldc, m, n, k);
                if (ms < bestMs)
                {
                    bestMs = ms;
                    best = cand;
                }
            }

            return (axis, best.Mc, best.Nc, best.Kc, procs, bestMs);
        }
        finally
        {
            ArrayPool<T>.Shared.Return(a);
            ArrayPool<T>.Shared.Return(b);
            ArrayPool<T>.Shared.Return(c);
        }
    }

    /// <summary>
    /// Time a single candidate: warm it up once, then take the median of
    /// <see cref="Runs"/> single-GEMM timings (median damps a one-off scheduler
    /// hiccup better than the mean for a handful of samples).
    /// </summary>
    private static double TimeCandidate<T>(
        (int Mc, int Nc, int Kc) cand,
        T[] a, int aLen, int lda, bool transA,
        T[] b, int bLen, int ldb, bool transB,
        T[] c, int cLen, int ldc,
        int m, int n, int k) where T : unmanaged
    {
        AutotuneDispatcher.BlockOverride = cand;
        try
        {
            // Warmup (also faults in pages / JITs the strategy for this blocking).
            BlasManagedLib.Gemm<T>(
                new ReadOnlySpan<T>(a, 0, aLen), lda, transA,
                new ReadOnlySpan<T>(b, 0, bLen), ldb, transB,
                new Span<T>(c, 0, cLen), ldc,
                m, n, k);

            Span<double> samples = stackalloc double[Runs];
            var sw = new Stopwatch();
            for (int r = 0; r < Runs; r++)
            {
                sw.Restart();
                BlasManagedLib.Gemm<T>(
                    new ReadOnlySpan<T>(a, 0, aLen), lda, transA,
                    new ReadOnlySpan<T>(b, 0, bLen), ldb, transB,
                    new Span<T>(c, 0, cLen), ldc,
                    m, n, k);
                sw.Stop();
                samples[r] = sw.Elapsed.TotalMilliseconds;
            }
            return Median(samples);
        }
        finally
        {
            AutotuneDispatcher.BlockOverride = null;
        }
    }

    private static double Median(Span<double> values)
    {
        // Small N — insertion sort in place, then pick the middle.
        for (int i = 1; i < values.Length; i++)
        {
            double key = values[i];
            int j = i - 1;
            while (j >= 0 && values[j] > key) { values[j + 1] = values[j]; j--; }
            values[j + 1] = key;
        }
        int mid = values.Length / 2;
        return (values.Length & 1) == 1
            ? values[mid]
            : (values[mid - 1] + values[mid]) / 2.0;
    }

    /// <summary>
    /// Cross-product the candidate ranges, clamp + align each to the shape, and
    /// dedupe. Always includes the heuristic default (128, 512, 256) so the
    /// measured winner is never worse than the heuristic would have picked.
    /// </summary>
    private static List<(int Mc, int Nc, int Kc)> GenerateCandidates(int m, int n, int k, int mr, int nr)
    {
        int[] ncCandidates = { 256, 512, 1024, Math.Min(n, 2048) };

        var seen = new HashSet<(int, int, int)>();
        var list = new List<(int Mc, int Nc, int Kc)>();

        void Add(int mc, int nc, int kc)
        {
            var clamped = AutotuneDispatcher.ClampBlocking(mc, nc, kc, m, n, k, mr, nr);
            if (clamped.Mc > 0 && clamped.Nc > 0 && clamped.Kc > 0 && seen.Add(clamped))
                list.Add(clamped);
        }

        foreach (int mc in McCandidates)
            foreach (int nc in ncCandidates)
                foreach (int kc in KcCandidates)
                    Add(mc, nc, kc);

        // Safety net: the BLIS-style heuristic default must always be in the set.
        Add(128, 512, 256);

        return list;
    }

    /// <summary>
    /// Fill a span of <typeparamref name="T"/> (float or double) with values in
    /// [-1, 1]. Values are irrelevant to timing; they only need to be finite to
    /// avoid denormal/NaN slow paths.
    /// </summary>
    private static void FillRandom<T>(Span<T> span) where T : unmanaged
    {
        var rng = new Random(42);
        if (typeof(T) == typeof(float))
        {
            var f = MemoryMarshal.Cast<T, float>(span);
            for (int i = 0; i < f.Length; i++) f[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        else if (typeof(T) == typeof(double))
        {
            var d = MemoryMarshal.Cast<T, double>(span);
            for (int i = 0; i < d.Length; i++) d[i] = rng.NextDouble() * 2 - 1;
        }
        else
        {
            throw new NotSupportedException($"BlockSizeSweep supports float/double only, not {typeof(T).Name}.");
        }
    }
}
