using System;
using System.Runtime.CompilerServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// External BLAS provider — <b>disabled in the supply-chain-independence build</b>.
///
/// <para>
/// Historically this class dynamically loaded a native cblas-compatible library
/// (Intel MKL, OpenBLAS, Accelerate, etc.) via P/Invoke and exposed <c>TryGemm</c>
/// / <c>SgemmRaw</c> / <c>MklSgemmZeroOffset</c> hot paths that the rest of the
/// engine would preferentially use over the in-house <see cref="Engines.Simd.SimdGemm"/>.
/// After the <c>feat/finish-mkl-replacement</c> branch (Issue #131 completion),
/// the MKL.NET managed bindings were removed and the native P/Invoke loader is
/// also disabled — the whole engine now routes through SimdGemm's AVX2 blocked
/// kernel and the JIT micro-kernel in <see cref="CpuJit.CpuJitKernels"/>.
/// </para>
/// <para>
/// The class is retained as a compatibility shim: every <c>Try*</c> gate returns
/// <c>false</c> and every <c>HasX</c> flag returns <c>false</c>, so existing
/// consumers fall through to their SimdGemm fallback path without needing any
/// call-site changes. The raw-pointer dispatch methods (<c>SgemmRaw</c>,
/// <c>MklSgemmZeroOffset</c>, etc.) throw <see cref="NotSupportedException"/> if
/// called — they never are, because callers gate on <c>HasRawSgemm</c> /
/// <c>IsMklVerified</c> which always report <c>false</c>.
/// </para>
/// <para>
/// <b>This stub is a hard disable, not a runtime-opt-in mechanism</b>. All
/// <c>Try*</c> methods unconditionally return <c>false</c>; <c>HasX</c> flags
/// unconditionally return <c>false</c>; the <c>SgemmRaw</c>/<c>MklSgemmZeroOffset</c>
/// hot paths throw <see cref="NotSupportedException"/>. There is no env var or
/// build symbol that re-enables it. Per Issue #131 iter 18c benchmarks, SimdGemm
/// is at or faster than MKL on every tracked DiT-XL shape (Square 1152² 0.99×,
/// Attn A·V 0.995×, etc.), so the performance implications are negligible.
/// Users who want a system BLAS at their own risk must revert this file to a
/// prior revision (and re-add the <c>AiDotNet.Native.OneDNN</c> package if they
/// want oneDNN back as well); the default build has zero CPU native-math
/// dependencies.
/// </para>
/// </summary>
internal static class BlasProvider
{
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

    public static void SetDeterministicMode(bool deterministic)
    {
        _deterministicMode = deterministic;
    }

    /// <summary>Always false — external BLAS is disabled.</summary>
    internal static bool IsAvailable => false;

    /// <summary>
    /// Backend name for diagnostics. Historically could return "Intel MKL.NET" or
    /// "Native BLAS"; post-disable always returns a sentinel pointing at the
    /// in-house kernel.
    /// </summary>
    internal static string BackendName => "SimdGemm (external BLAS disabled)";

    /// <summary>Always false — external BLAS is disabled.</summary>
    internal static bool HasNativeSgemm => false;
    /// <summary>Always false — external BLAS is disabled.</summary>
    internal static bool HasNativeDgemm => false;
    /// <summary>Always false — external BLAS is disabled.</summary>
    internal static bool HasRawSgemm => false;
    /// <summary>Always false — external BLAS is disabled.</summary>
    internal static bool HasRawDgemm => false;
    /// <summary>Always false — post-MKL.NET-removal build has no MKL managed binding.</summary>
    internal static bool HasMklNet => false;
    /// <summary>Always false — external BLAS is disabled; the historical
    /// "MKL verified raw pointer dispatch" path is no longer available.</summary>
    internal static bool IsMklVerified => false;

    // ────────────────────────────────────────────────────────────────────
    // Try* entry points — all return false, forcing callers through their
    // SimdGemm fallback. Public signatures preserved for consumer compat.
    // ────────────────────────────────────────────────────────────────────

    internal static bool TryGemm(int m, int n, int k,
        float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb,
        float[] c, int cOffset, int ldc)
        => false;

    internal static bool TryGemm(int m, int n, int k,
        double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc)
        => false;

    internal static bool TryGemm(int m, int n, int k,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
        => false;

    internal static bool TryGemm(int m, int n, int k,
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
        => false;

    internal static bool TryGemmWithBeta(int m, int n, int k,
        float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb,
        float[] c, int cOffset, int ldc, float beta)
        => false;

    internal static bool TryGemmWithBeta(int m, int n, int k,
        double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc, double beta)
        => false;

    internal static bool TryGemmEx(int m, int n, int k,
        float[] a, int aOffset, int lda, bool transA,
        float[] b, int bOffset, int ldb, bool transB,
        float[] c, int cOffset, int ldc)
        => false;

    // ────────────────────────────────────────────────────────────────────
    // Direct-dispatch hot paths. Historically these skipped the Try* gate
    // when the caller had already verified availability (via HasRawSgemm /
    // IsMklVerified). With external BLAS disabled, HasX always returns
    // false so these are never called — they throw to fail fast if a
    // future caller forgets to gate.
    // ────────────────────────────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgemmRaw(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc)
        => ThrowDisabled();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void DgemmRaw(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
        => ThrowDisabled();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgemmDirect(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc)
        => ThrowDisabled();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void DgemmDirect(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
        => ThrowDisabled();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklSgemmZeroOffset(int m, int n, int k, float[] a, int lda, float[] b, int ldb, float[] c, int ldc)
        => ThrowDisabled();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklDgemmZeroOffset(int m, int n, int k, double[] a, int lda, double[] b, int ldb, double[] c, int ldc)
        => ThrowDisabled();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklSgemmDirect(int m, int n, int k, float[] a, int lda, float[] b, int ldb, float[] c, int ldc)
        => ThrowDisabled();

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void MklDgemmDirect(int m, int n, int k, double[] a, int lda, double[] b, int ldb, double[] c, int ldc)
        => ThrowDisabled();

    private static void ThrowDisabled() =>
        throw new NotSupportedException(
            "BlasProvider native dispatch is disabled. Gate on HasRawSgemm / IsMklVerified (both always false) " +
            "and fall through to SimdGemm. See feat/finish-mkl-replacement branch notes.");
}
