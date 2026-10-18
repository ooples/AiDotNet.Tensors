using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// CPU GEMM entry points and the process-wide deterministic-mode switch.
/// </summary>
/// <remarks>
/// Every GEMM runs the in-house managed BLAS (<see cref="Engines.BlasManaged.BlasManaged"/>,
/// AVX2/AVX-512 kernels gated by <c>IsSupported</c>, with a scalar fallback). The library ships
/// no native BLAS: OpenBLAS and Intel MKL were removed under issue #368 once the managed kernels
/// reached rough parity with OpenBLAS on the perf-bar catalog. Native BLAS dispatches SIMD code
/// .NET cannot gate on hardware features, so on an unsupported or virtualized CPU it could crash
/// the process with no managed stack trace; it also added about 66 MB of native packages.
/// </remarks>
internal static class BlasProvider
{
    // ─────────────────────────────────────────────────────────────────────
    // Shape-instrumentation hook (issue #403 Phase A.1).
    //
    // When subscribed, every Gemm/GemmEx call invokes the hook with
    // (m, n, k, transA, transB) so a higher-level profiler can build a
    // unique-shape catalog without modifying call sites. The hook is a single
    // volatile delegate read on the fast path — when unset, the per-call cost
    // is one null check.
    //
    // Subscribe by assigning ShapeLogHook = (m,n,k,tA,tB) => {...}.
    // Unsubscribe by assigning null. Concurrent subscribe/unsubscribe is not
    // supported (intended single-consumer profiler).
    // ─────────────────────────────────────────────────────────────────────
    internal static Action<int, int, int, bool, bool>? ShapeLogHook;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LogShape(int m, int n, int k, bool transA = false, bool transB = false)
    {
        var hook = ShapeLogHook;
        if (hook is not null) hook(m, n, k, transA, transB);
    }

    // Defaults to true (issue #164): deterministic-by-default. The managed GEMM is
    // bit-reproducible across thread counts in either mode; the flag stays load-bearing for:
    //   1. CpuParallelSettings.DeterministicReductions (set below), which serializes the
    //      order-dependent reduction kernels.
    //   2. CompiledModelCache.ComputeShapeKey, which mixes IsDeterministicMode into the plan
    //      key so plans compiled under the opposite setting are not reused.
    //   3. Public API contract: TensorCodecOptions.Deterministic and
    //      AiDotNetEngine.SetDeterministicMode are observable consumer surfaces.
    private static volatile bool _deterministicMode = true;

    /// <summary>
    /// Per-thread override of the process-wide <see cref="_deterministicMode"/>.
    /// <para>
    /// <c>null</c> (the default on every thread) means "inherit the process-wide setting."
    /// A non-null value wins over the process-wide default for the current thread only,
    /// letting one thread opt into or out of determinism without affecting any other
    /// thread. Set via <see cref="SetThreadLocalDeterministicMode"/> — typically driven
    /// by <c>TensorCodecOptions.SetCurrent</c>, which is itself thread-local.
    /// </para>
    /// <para>
    /// The override has no GEMM dispatch effect
    /// (every GEMM runs the managed kernel), but it threads
    /// through the cache-key invariant in CompiledModelCache so per-thread plans are
    /// segregated by the override. This guarantees forward-compatibility: when GPU or
    /// other backends re-introduce determinism-divergent kernels, the override is
    /// already wired end-to-end.
    /// </para>
    /// </summary>
    [ThreadStatic]
    private static bool? _threadLocalDeterministicOverride;

    /// <summary>
    /// Returns whether deterministic mode is currently enabled on the calling thread.
    /// Reads the thread-local override first, falling back to the process-wide default.
    /// </summary>
    public static bool IsDeterministicMode => _threadLocalDeterministicOverride ?? _deterministicMode;

    /// <summary>
    /// Installs a per-thread determinism override, or clears it with <c>null</c>. The
    /// override wins over the process-wide <see cref="SetDeterministicMode"/> value for
    /// this thread only. Typically driven by <c>TensorCodecOptions.SetCurrent</c>, which
    /// itself is thread-local.
    /// </summary>
    /// <param name="value">
    /// <c>true</c> to force deterministic mode on this thread; <c>false</c> to allow
    /// non-deterministic paths on this thread; <c>null</c> to clear the override and
    /// inherit the process-wide setting.
    /// </param>
    public static void SetThreadLocalDeterministicMode(bool? value) => _threadLocalDeterministicOverride = value;

    /// <summary>
    /// Reads the current thread-local determinism override without falling back to
    /// the process-wide setting. Returns <c>null</c> when no override is installed
    /// on the calling thread (the default — <see cref="IsDeterministicMode"/> is
    /// then driven entirely by the process-wide field).
    /// <para>Used by tests to capture-and-restore the override across cases that
    /// also touch the process-wide setting; without this, capture/restore can only
    /// observe the merged value from <see cref="IsDeterministicMode"/>, which
    /// loses the distinction between "no override" and "override = process-wide
    /// value" — the two are observationally identical at the merged read but
    /// behave differently when the process-wide value flips.</para>
    /// </summary>
    public static bool? GetThreadLocalDeterministicMode() => _threadLocalDeterministicOverride;

    public static void SetDeterministicMode(bool deterministic)
    {
        _deterministicMode = deterministic;
        // Unify the switch (plan item 1.4): deterministic mode must ALSO make the
        // order-dependent reduction/accumulation kernels bit-reproducible — their
        // multi-threaded partial-sum combine is non-associative. Previously callers
        // had to set this separately, so "deterministic mode" left those reductions
        // non-deterministic. GEMM stays parallel because its M/N-tile splits are
        // marked deterministicSafe (see CpuParallelSettings.ParallelForOrSerial), so
        // this serializes only the genuinely order-dependent reductions.
        CpuParallelSettings.DeterministicReductions = deterministic;
    }

    // ────────────────────────────────────────────────────────────────────
    // GEMM: C := A·B (row-major), optionally with transposed operands, always
    // served by the managed kernel. C is overwritten, never read.
    // ────────────────────────────────────────────────────────────────────

    internal static void Gemm(int m, int n, int k,
        float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb,
        float[] c, int cOffset, int ldc)
        => Gemm(m, n, k,
            new ReadOnlySpan<float>(a, aOffset, a.Length - aOffset), lda,
            new ReadOnlySpan<float>(b, bOffset, b.Length - bOffset), ldb,
            new Span<float>(c, cOffset, c.Length - cOffset), ldc);

    internal static void Gemm(int m, int n, int k,
        double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc)
        => Gemm(m, n, k,
            new ReadOnlySpan<double>(a, aOffset, a.Length - aOffset), lda,
            new ReadOnlySpan<double>(b, bOffset, b.Length - bOffset), ldb,
            new Span<double>(c, cOffset, c.Length - cOffset), ldc);

    internal static void Gemm(int m, int n, int k,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        Engines.BlasManaged.BlasManaged.Gemm<float>(a, lda, false, b, ldb, false, c, ldc, m, n, k);
        LogShape(m, n, k);
    }

    internal static void Gemm(int m, int n, int k,
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        Engines.BlasManaged.BlasManaged.Gemm<double>(a, lda, false, b, ldb, false, c, ldc, m, n, k);
        LogShape(m, n, k);
    }

    /// <summary>
    /// GEMM with optional transposed operands, writing C without reading it (beta = 0).
    /// </summary>
    internal static void GemmEx(int m, int n, int k,
        float[] a, int aOffset, int lda, bool transA,
        float[] b, int bOffset, int ldb, bool transB,
        float[] c, int cOffset, int ldc)
    {
        Engines.BlasManaged.BlasManaged.Gemm<float>(
            new ReadOnlySpan<float>(a, aOffset, a.Length - aOffset), lda, transA,
            new ReadOnlySpan<float>(b, bOffset, b.Length - bOffset), ldb, transB,
            new Span<float>(c, cOffset, c.Length - cOffset), ldc,
            m, n, k, new Engines.BlasManaged.BlasOptions<float> { BetaZero = true });
        LogShape(m, n, k, transA, transB);
    }

    /// <summary>
    /// Double-precision counterpart of <see cref="GemmEx(int, int, int, float[], int, int, bool, float[], int, int, bool, float[], int, int)"/>.
    /// </summary>
    internal static void GemmEx(int m, int n, int k,
        double[] a, int aOffset, int lda, bool transA,
        double[] b, int bOffset, int ldb, bool transB,
        double[] c, int cOffset, int ldc)
    {
        Engines.BlasManaged.BlasManaged.Gemm<double>(
            new ReadOnlySpan<double>(a, aOffset, a.Length - aOffset), lda, transA,
            new ReadOnlySpan<double>(b, bOffset, b.Length - bOffset), ldb, transB,
            new Span<double>(c, cOffset, c.Length - cOffset), ldc,
            m, n, k, new Engines.BlasManaged.BlasOptions<double> { BetaZero = true });
        LogShape(m, n, k, transA, transB);
    }
}
