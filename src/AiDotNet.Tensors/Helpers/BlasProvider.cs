using System;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// External BLAS provider — opt-in via env var.
///
/// <para>
/// Default behaviour (industry-standard, matches PyTorch / NumPy / TF):
/// dynamically loads <c>libopenblas</c> from the
/// <c>AiDotNet.Native.OpenBLAS</c> NuGet (transitive dependency of every
/// AiDotNet install) and routes <c>cblas_sgemm</c> / <c>cblas_dgemm</c>
/// through the native path. Closes the 6-25× matmul gap vs MKL-backed
/// PyTorch on transformer-FFN shapes (issue #242) — and the
/// 80%-of-ViT-Base-train-time bottleneck that the
/// Issue319TransformerBlockPerfTests probe surfaces.
/// </para>
/// <para>
/// Opt-OUT: set <c>AIDOTNET_USE_BLAS=0</c> (or <c>false</c> / <c>no</c> /
/// <c>off</c>) at process start to disable the native path and route every
/// dispatch through <see cref="Engines.Simd.SimdGemm"/>'s AVX2 blocked
/// kernel. Use this for deterministic-bit-exact builds where BLAS's
/// non-associative reduction order is unacceptable, or for environments
/// where libopenblas refuses to load.
/// </para>
/// <para>
/// Missing native library at runtime falls back to the SimdGemm path
/// transparently — the <c>HasX</c> flags read the actual dynamic-load
/// success, so consumers that gate on them (e.g. <c>HasRawSgemm</c>) still
/// see <c>false</c> and route correctly.
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

    // ─────────────────────────────────────────────────────────────────────
    // Issue #338 Phase G.1 — Intel MKL via Microsoft.ML.Mkl.Redist.
    // MklImports.dll exports cblas_sgemm/cblas_dgemm directly (verified
    // via PE symbol probe). Same CBLAS enum layout as OpenBLAS — bit-
    // compatible call sites, different runtime. Opt-in via env var
    // AIDOTNET_BLAS_PROVIDER=mkl at process start. Default remains
    // OpenBLAS until MKL is verified faster on consumer's hardware.
    // ─────────────────────────────────────────────────────────────────────
    [DllImport("MklImports", EntryPoint = "cblas_sgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void mkl_cblas_sgemm_ptr(
        int layout, int transA, int transB,
        int m, int n, int k,
        float alpha, float* a, int lda,
        float* b, int ldb,
        float beta, float* c, int ldc);

    [DllImport("MklImports", EntryPoint = "cblas_dgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void mkl_cblas_dgemm_ptr(
        int layout, int transA, int transB,
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc);

    /// <summary>True iff <c>AIDOTNET_USE_BLAS=1</c>|<c>true</c>|<c>yes</c> at process start.</summary>
    private static readonly bool _blasOptIn = ReadOptIn();

    /// <summary>
    /// True iff <c>AIDOTNET_BLAS_PROVIDER=mkl</c> at process start.
    /// When false, MKL probes are skipped and OpenBLAS is the only candidate.
    /// </summary>
    private static readonly bool _preferMkl = string.Equals(
        Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER"),
        "mkl",
        StringComparison.OrdinalIgnoreCase);

    /// <summary>Resolved lazily on first use — probes libopenblas once.</summary>
    private static readonly Lazy<bool> _nativeAvailable = new(ProbeNativeLibrary);

    /// <summary>Resolved lazily on first use — probes MklImports once.</summary>
    private static readonly Lazy<bool> _mklAvailable = new(ProbeMklLibrary);

#if NET5_0_OR_GREATER
    static BlasProvider()
    {
        // Issue #338 Phase G.1: when AIDOTNET_BLAS_PROVIDER=mkl, redirect
        // every libopenblas P/Invoke to MklImports.dll in this assembly.
        // Single intercept covers all 17 cblas_*_ptr / cblas_*_native call
        // sites uniformly, so the rest of BlasProvider stays unchanged.
        // Process-wide: registering twice in the same process is a no-op,
        // and the resolver only fires for libopenblas DllImport names so
        // GPU/other native deps are unaffected.
        if (_preferMkl)
        {
            try
            {
                System.Runtime.InteropServices.NativeLibrary.SetDllImportResolver(
                    typeof(BlasProvider).Assembly,
                    (libraryName, assembly, searchPath) =>
                    {
                        if (string.Equals(libraryName, "libopenblas", StringComparison.OrdinalIgnoreCase))
                        {
                            if (System.Runtime.InteropServices.NativeLibrary.TryLoad(
                                "MklImports", assembly, searchPath, out var handle))
                                return handle;
                        }
                        return IntPtr.Zero;
                    });
            }
            catch (InvalidOperationException)
            {
                // Resolver already registered by another consumer — ignore;
                // first registration wins, our libopenblas calls will still
                // resolve correctly if that resolver also redirects them or
                // if libopenblas itself is loadable from the consumer's
                // runtime asset directory.
            }
        }
    }
#endif

    private static bool ReadOptIn()
    {
        // Industry-standard default: BLAS on when the native lib is available.
        // PyTorch / NumPy / TensorFlow all default to BLAS-mandatory; AiDotNet
        // ships AiDotNet.Native.OpenBLAS as a transitive NuGet dependency, so
        // libopenblas is loadable on every consumer install. The previous
        // opt-in default (`AIDOTNET_USE_BLAS=1` to enable) left double-precision
        // matmul running on the in-house SimdGemm.Dgemm AVX2 path at ~12 GFLOPS
        // when OpenBLAS Dgemm runs the same kernel at ~75 GFLOPS — a 6× perf
        // gap on every TensorMatMul call (which is ~80 % of ViT-Base CPU
        // train wall-clock per the Issue319TransformerBlockPerfTests probe).
        //
        // Opt-OUT is now `AIDOTNET_USE_BLAS=0|false|no` — for deterministic-
        // bit-exact builds where BLAS's non-associative reduction order is
        // unacceptable, or for environments where libopenblas refuses to load
        // and we want the SimdGemm fallback to engage without paying for the
        // ProbeNativeLibrary cost on every cold start.
        var raw = Environment.GetEnvironmentVariable("AIDOTNET_USE_BLAS");
        if (string.IsNullOrWhiteSpace(raw)) return true;
        // Explicit opt-out values disable BLAS; everything else (including
        // empty string after trim, unrecognized strings, and the legacy
        // 1/true/yes affirmations) leaves BLAS enabled.
        return !(string.Equals(raw, "0", StringComparison.OrdinalIgnoreCase)
              || string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase)
              || string.Equals(raw, "no", StringComparison.OrdinalIgnoreCase)
              || string.Equals(raw, "off", StringComparison.OrdinalIgnoreCase));
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

    /// <summary>
    /// Issue #338 Phase G.1: probe MklImports cblas_sgemm. Only runs when
    /// the user opts in via <c>AIDOTNET_BLAS_PROVIDER=mkl</c>. A successful
    /// probe means consumer-side MKL is loadable and we can route GEMM
    /// dispatches through MKL's threaded kernel for the d=128 transformer
    /// shapes where OpenBLAS leaves performance on the table.
    /// </summary>
    private static unsafe bool ProbeMklLibrary()
    {
        if (!_preferMkl) return false;
        try
        {
            float a = 2.0f, b = 3.0f, c = 0.0f;
            mkl_cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1.0f, &a, 1, &b, 1, 0.0f, &c, 1);
            return c == 6.0f;
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
        // Phase G.1: when AIDOTNET_BLAS_PROVIDER=mkl, the static-ctor
        // resolver redirects libopenblas → MklImports; the call below
        // then dispatches into MKL's cblas_sgemm transparently.
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

    /// <summary>
    /// Double-precision counterpart of <see cref="TryGemmEx(int, int, int, float[], int, int, bool, float[], int, int, bool, float[], int, int)"/>.
    /// Used by the compiled-plan double MatMul backward path (PR #319) to dispatch
    /// transposed Dgemm directly into pre-allocated output buffers without paying
    /// the engine's TensorTranspose + TensorMatMul allocation+copy cost.
    /// </summary>
    internal static bool TryGemmEx(int m, int n, int k,
        double[] a, int aOffset, int lda, bool transA,
        double[] b, int bOffset, int ldb, bool transB,
        double[] c, int cOffset, int ldc)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            int cblasTA = transA ? 112 : CblasNoTrans;  // 112 = CblasTrans
            int cblasTB = transB ? 112 : CblasNoTrans;
            unsafe
            {
                fixed (double* pa = &a[aOffset])
                fixed (double* pb = &b[bOffset])
                fixed (double* pc = &c[cOffset])
                {
                    cblas_dgemm_ptr(CblasRowMajor, cblasTA, cblasTB,
                        m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
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
