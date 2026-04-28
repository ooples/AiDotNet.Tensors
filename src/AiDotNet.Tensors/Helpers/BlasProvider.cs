using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// External BLAS provider — opt-in via env var.
///
/// <para>
/// By default this class is a <b>pass-through stub</b>: every <c>Try*</c> gate
/// returns <c>false</c>, every <c>HasX</c> flag returns <c>false</c>, and the
/// engine routes through <see cref="Engines.Simd.SimdGemm"/>'s AVX2 blocked
/// kernel. Zero native dependencies, supply-chain-independent builds.
/// </para>
/// <para>
/// When the environment variable <c>AIDOTNET_USE_BLAS=1</c> is set at process
/// start, the provider dynamically loads <c>libopenblas</c> (ships via the
/// <c>AiDotNet.Native.OpenBLAS</c> NuGet) and routes <c>cblas_sgemm</c> /
/// <c>cblas_dgemm</c> through the native path. This closes the 10-25× matmul
/// gap vs MKL-backed PyTorch on transformer-FFN shapes (issue #242) without
/// changing the default build behaviour.
/// </para>
/// <para>
/// Missing native library at runtime falls back to the SimdGemm path — the
/// <c>HasX</c> flags read the actual dynamic-load success, so consumers that
/// gate on them (e.g. <c>HasRawSgemm</c>) still transparently fall through.
/// </para>
/// </summary>
internal static class BlasProvider
{
    // ─────────────────────────────────────────────────────────────────────
    // libopenblas P/Invoke. cblas_sgemm / cblas_dgemm use the standard CBLAS
    // enum layout: Layout = 101 (RowMajor), NoTrans = 111. lda/ldb/ldc are
    // leading dimensions in the row-major sense.
    // ─────────────────────────────────────────────────────────────────────
    private const int CblasRowMajor = 101;
    private const int CblasNoTrans = 111;

    [DllImport("libopenblas", EntryPoint = "cblas_sgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern void cblas_sgemm_native(
        int layout, int transA, int transB,
        int m, int n, int k,
        float alpha, float[] a, int lda,
        float[] b, int ldb,
        float beta, float[] c, int ldc);

    [DllImport("libopenblas", EntryPoint = "cblas_dgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern void cblas_dgemm_native(
        int layout, int transA, int transB,
        int m, int n, int k,
        double alpha, double[] a, int lda,
        double[] b, int ldb,
        double beta, double[] c, int ldc);

    // Array-slice variants. libopenblas takes pointer-offset natively, so we
    // bridge via span + MemoryMarshal.
    [DllImport("libopenblas", EntryPoint = "cblas_sgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void cblas_sgemm_ptr(
        int layout, int transA, int transB,
        int m, int n, int k,
        float alpha, float* a, int lda,
        float* b, int ldb,
        float beta, float* c, int ldc);

    [DllImport("libopenblas", EntryPoint = "cblas_dgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void cblas_dgemm_ptr(
        int layout, int transA, int transB,
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc);

    /// <summary>True iff <c>AIDOTNET_USE_BLAS=1</c>|<c>true</c>|<c>yes</c> at process start.</summary>
    private static readonly bool _blasOptIn = ReadOptIn();

    /// <summary>Resolved lazily on first use — probes libopenblas once.</summary>
    private static readonly Lazy<bool> _nativeAvailable = new(ProbeNativeLibrary);

    private static bool ReadOptIn()
    {
        var raw = Environment.GetEnvironmentVariable("AIDOTNET_USE_BLAS");
        if (string.IsNullOrWhiteSpace(raw)) return false;
        return string.Equals(raw, "1", StringComparison.OrdinalIgnoreCase)
            || string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase)
            || string.Equals(raw, "yes", StringComparison.OrdinalIgnoreCase);
    }

    private static bool ProbeNativeLibrary()
    {
        if (!_blasOptIn) return false;
        try
        {
            // Call into native with a trivially-small 1×1 gemm to verify the
            // symbol actually resolves. A DllNotFoundException / EntryPointNotFoundException
            // means the native lib isn't installed — fall back to SimdGemm.
            var a = new float[] { 2.0f };
            var b = new float[] { 3.0f };
            var c = new float[] { 0.0f };
            cblas_sgemm_native(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1.0f, a, 1, b, 1, 0.0f, c, 1);
            return c[0] == 6.0f;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
    }
    // Defaults to true (issue #164): deterministic-by-default. After the MKL.NET removal
    // in #131/#163, every BLAS dispatch in this stub returns false anyway and routes the
    // engine through SimdGemm — which is itself bit-exact across thread counts. The flag
    // is therefore informational at the BLAS layer today, but it remains load-bearing
    // for two consumers:
    //   1. CompiledModelCache.ComputeShapeKey mixes IsDeterministicMode into the plan
    //      key, so toggling the flag invalidates plans compiled under the opposite
    //      setting (forward-safe for any future re-introduction of divergent kernels,
    //      e.g. GPU paths that branch on determinism).
    //   2. Public API contract: TensorCodecOptions.Deterministic and
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
    /// In the post-MKL-removal build the override has no immediate dispatch effect
    /// (everything routes through SimdGemm regardless), but it correctly threads
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
    }

    /// <summary>True iff <c>AIDOTNET_USE_BLAS=1</c> is set AND libopenblas loaded successfully.</summary>
    internal static bool IsAvailable => _nativeAvailable.Value;

    internal static string BackendName => _nativeAvailable.Value
        ? "OpenBLAS (AIDOTNET_USE_BLAS=1)"
        : "SimdGemm (default; set AIDOTNET_USE_BLAS=1 to enable OpenBLAS)";

    internal static bool HasNativeSgemm => _nativeAvailable.Value;
    internal static bool HasNativeDgemm => _nativeAvailable.Value;
    internal static bool HasRawSgemm => _nativeAvailable.Value;
    internal static bool HasRawDgemm => _nativeAvailable.Value;
    /// <summary>Always false — post-MKL.NET-removal build has no MKL managed binding.</summary>
    internal static bool HasMklNet => false;
    /// <summary>
    /// False by default. When OpenBLAS is loaded opt-in, it's considered
    /// "verified" — OpenBLAS cblas_sgemm is strictly conforming, no special
    /// guard needed beyond the dynamic-load probe.
    /// </summary>
    internal static bool IsMklVerified => _nativeAvailable.Value;

    // ────────────────────────────────────────────────────────────────────
    // Try* entry points. When BLAS is disabled (default), every call
    // returns false and callers fall through to SimdGemm. When enabled
    // via AIDOTNET_USE_BLAS=1 and libopenblas loads, calls dispatch to
    // cblas_{s,d}gemm with standard row-major layout.
    // ────────────────────────────────────────────────────────────────────

    internal static bool TryGemm(int m, int n, int k,
        float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb,
        float[] c, int cOffset, int ldc)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            unsafe
            {
                fixed (float* pa = &a[aOffset])
                fixed (float* pb = &b[bOffset])
                fixed (float* pc = &c[cOffset])
                {
                    cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                }
            }
            return true;
        }
        catch { return false; }
    }

    internal static bool TryGemm(int m, int n, int k,
        double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            unsafe
            {
                fixed (double* pa = &a[aOffset])
                fixed (double* pb = &b[bOffset])
                fixed (double* pc = &c[cOffset])
                {
                    cblas_dgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                }
            }
            return true;
        }
        catch { return false; }
    }

    internal static bool TryGemm(int m, int n, int k,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            unsafe
            {
                fixed (float* pa = a)
                fixed (float* pb = b)
                fixed (float* pc = c)
                {
                    cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                }
            }
            return true;
        }
        catch { return false; }
    }

    internal static bool TryGemm(int m, int n, int k,
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            unsafe
            {
                fixed (double* pa = a)
                fixed (double* pb = b)
                fixed (double* pc = c)
                {
                    cblas_dgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                }
            }
            return true;
        }
        catch { return false; }
    }

    internal static bool TryGemmWithBeta(int m, int n, int k,
        float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb,
        float[] c, int cOffset, int ldc, float beta)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            unsafe
            {
                fixed (float* pa = &a[aOffset])
                fixed (float* pb = &b[bOffset])
                fixed (float* pc = &c[cOffset])
                {
                    cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0f, pa, lda, pb, ldb, beta, pc, ldc);
                }
            }
            return true;
        }
        catch { return false; }
    }

    internal static bool TryGemmWithBeta(int m, int n, int k,
        double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc, double beta)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            unsafe
            {
                fixed (double* pa = &a[aOffset])
                fixed (double* pb = &b[bOffset])
                fixed (double* pc = &c[cOffset])
                {
                    cblas_dgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        m, n, k, 1.0, pa, lda, pb, ldb, beta, pc, ldc);
                }
            }
            return true;
        }
        catch { return false; }
    }

    internal static bool TryGemmEx(int m, int n, int k,
        float[] a, int aOffset, int lda, bool transA,
        float[] b, int bOffset, int ldb, bool transB,
        float[] c, int cOffset, int ldc)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            int cblasTA = transA ? 112 : CblasNoTrans;  // 112 = CblasTrans
            int cblasTB = transB ? 112 : CblasNoTrans;
            unsafe
            {
                fixed (float* pa = &a[aOffset])
                fixed (float* pb = &b[bOffset])
                fixed (float* pc = &c[cOffset])
                {
                    cblas_sgemm_ptr(CblasRowMajor, cblasTA, cblasTB,
                        m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                }
            }
            return true;
        }
        catch { return false; }
    }

    // ────────────────────────────────────────────────────────────────────
    // Direct-dispatch hot paths. Historically these skipped the Try* gate
    // when the caller had already verified availability (via HasRawSgemm /
    // IsMklVerified). With external BLAS disabled, HasX always returns
    // false so these are never called — they throw to fail fast if a
    // future caller forgets to gate.
    // ────────────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgemmRaw(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc)
    {
        if (!_nativeAvailable.Value) ThrowDisabled();
        cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void DgemmRaw(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        if (!_nativeAvailable.Value) ThrowDisabled();
        cblas_dgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgemmDirect(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc)
        => SgemmRaw(m, n, k, a, lda, b, ldb, c, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void DgemmDirect(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
        => DgemmRaw(m, n, k, a, lda, b, ldb, c, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklSgemmZeroOffset(int m, int n, int k, float[] a, int lda, float[] b, int ldb, float[] c, int ldc)
    {
        if (!_nativeAvailable.Value) ThrowDisabled();
        cblas_sgemm_native(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklDgemmZeroOffset(int m, int n, int k, double[] a, int lda, double[] b, int ldb, double[] c, int ldc)
    {
        if (!_nativeAvailable.Value) ThrowDisabled();
        cblas_dgemm_native(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklSgemmDirect(int m, int n, int k, float[] a, int lda, float[] b, int ldb, float[] c, int ldc)
        => MklSgemmZeroOffset(m, n, k, a, lda, b, ldb, c, ldc);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklDgemmDirect(int m, int n, int k, double[] a, int lda, double[] b, int ldb, double[] c, int ldc)
        => MklDgemmZeroOffset(m, n, k, a, lda, b, ldb, c, ldc);

    private static void ThrowDisabled() =>
        throw new NotSupportedException(
            "BlasProvider native dispatch is disabled. Set AIDOTNET_USE_BLAS=1 at process start " +
            "to enable OpenBLAS. Gate on HasRawSgemm / IsMklVerified (both return false when the " +
            "native library isn't loaded) and fall through to SimdGemm.");
}
