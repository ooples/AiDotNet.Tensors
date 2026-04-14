using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Native BLAS (cblas_sgemm / cblas_dgemm) provider. Dynamically loads a
/// cblas-compatible library (OpenBLAS, Accelerate, or Intel MKL if the user
/// supplies the native DLL out-of-band) at runtime via P/Invoke.
///
/// <para>
/// <b>Supply-chain note</b>: MKL.NET was removed from this project's dependencies
/// (was previously bundled as a 110 MB Intel binary). BLAS is now entirely
/// OPT-IN at runtime — install the optional <c>AiDotNet.Native.OpenBLAS</c>
/// package, set <c>AIDOTNET_BLAS_PATH</c> to a cblas-compatible library, or
/// place the library on the standard search path. Without one, all matmul
/// routes through <see cref="Engines.Simd.SimdGemm"/>'s AVX2 blocked kernel.
/// </para>
/// <para>
/// The public surface (<c>IsMklVerified</c>, <c>MklSgemmZeroOffset</c>, etc.)
/// is kept for consumer compatibility. After the MKL.NET removal they wrap
/// the native <c>cblas_sgemm</c> P/Invoke path rather than managed MKL.NET
/// bindings — the <c>Mkl</c> prefix is historical.
/// </para>
/// </summary>
internal static class BlasProvider
{
    private const int CblasRowMajor = 101;
    private const int CblasNoTrans = 111;
    private const int CblasTrans = 112;

    private static readonly object InitLock = new object();
    private static bool _initialized;
    private static bool _available;
    private static IntPtr _libraryHandle;
    private static CblasSgemm? _sgemm;
    private static CblasDgemm? _dgemm;
#if NET5_0_OR_GREATER
    // Raw function pointers for zero-overhead native BLAS calls.
    // These bypass delegate dispatch entirely — JIT emits a direct calli instruction.
    private static IntPtr _sgemmPtr;
    private static IntPtr _dgemmPtr;
#endif
    private static BlasSetNumThreads? _setNumThreads;
    private static MklSetDynamic? _setDynamic;
    private static int? ThreadCountOverride = ReadEnvInt("AIDOTNET_BLAS_THREADS");
    private static readonly bool TraceEnabled = ReadEnvBool("AIDOTNET_BLAS_TRACE");

    /// <summary>
    /// When true, BLAS GEMM paths are bypassed entirely — TryGemm and
    /// IsMklVerified return false so matmul falls through to the bit-exact
    /// blocked C# fallback. Guarantees bit-identical results across runs
    /// regardless of thread count, since the fallback's inner accumulation
    /// order is fixed. Volatile so lock-free reads in the hot TryGemm path
    /// see writes made under InitLock without stale caching.
    /// </summary>
    private static volatile bool _deterministicMode;

    /// <summary>
    /// Returns whether deterministic mode is currently enabled.
    /// </summary>
    public static bool IsDeterministicMode => _deterministicMode;

    public static void SetDeterministicMode(bool deterministic)
    {
        lock (InitLock)
        {
            if (_deterministicMode == deterministic) return;
            _deterministicMode = deterministic;
            // Deterministic mode routes all matmul through the bit-exact blocked C# fallback
            // by having TryGemm/IsMklVerified return false at the top. It does NOT call into
            // native thread-control functions — those paths are only triggered when
            // AIDOTNET_BLAS_THREADS is explicitly set via env var.
        }
    }

#if NETFRAMEWORK
    [DllImport("kernel32", SetLastError = true, CharSet = CharSet.Unicode)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern IntPtr GetProcAddress(IntPtr hModule, string procName);

    [DllImport("kernel32", SetLastError = true)]
    private static extern bool FreeLibrary(IntPtr hModule);
#endif

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void CblasSgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        float alpha,
        float* a, int lda,
        float* b, int ldb,
        float beta,
        float* c, int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private unsafe delegate void CblasDgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        double alpha,
        double* a, int lda,
        double* b, int ldb,
        double beta,
        double* c, int ldc);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void BlasSetNumThreads(int threads);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void MklSetDynamic(int enabled);

    /// <summary>
    /// Returns true if a native BLAS library has been successfully loaded.
    /// Useful for diagnostics to verify a backing library is available.
    /// </summary>
    internal static bool IsAvailable
    {
        get
        {
            EnsureInitialized();
            return _available && _sgemm != null;
        }
    }

    /// <summary>
    /// Returns the name of the BLAS backend being used. After the MKL.NET
    /// removal (feat/finish-mkl-replacement) only "Native BLAS" or "None"
    /// are possible.
    /// </summary>
    internal static string BackendName
    {
        get
        {
            EnsureInitialized();
            if (_sgemm != null) return "Native BLAS";
            return "None";
        }
    }

    internal static bool TryGemm(int m, int n, int k, float[] a, int aOffset, int lda, float[] b, int bOffset, int ldb, float[] c, int cOffset, int ldc)
    {
        // Deterministic mode: fall through to the bit-exact blocked C# fallback in
        // MatrixMultiplyHelper.MultiplyBlocked. Also avoids triggering native BLAS
        // initialization, which can be unsafe on some platforms.
        if (_deterministicMode) return false;

        if (!EnsureInitialized())
        {
            return false;
        }

        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
        {
            return false;
        }

        if (_sgemm == null)
        {
            return false;
        }

        unsafe
        {
            fixed (float* aPtrBase = a)
            fixed (float* bPtrBase = b)
            fixed (float* cPtrBase = c)
            {
                float* aPtr = aPtrBase + aOffset;
                float* bPtr = bPtrBase + bOffset;
                float* cPtr = cPtrBase + cOffset;
                _sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, aPtr, lda, bPtr, ldb, 0.0f, cPtr, ldc);
            }
        }

        return true;
    }

    /// <summary>
    /// Returns true if native BLAS is available for direct pointer calls.
    /// (After MKL.NET removal, this is the only BLAS path — kept for API
    /// continuity with the pre-removal consumers.)
    /// </summary>
    internal static bool HasNativeSgemm => _available && _sgemm != null;
    internal static bool HasNativeDgemm => _available && _dgemm != null;

#if NET5_0_OR_GREATER
    /// <summary>
    /// Returns true if raw function pointers are available for zero-overhead calli dispatch.
    /// This is the absolute fastest path — no delegate, no P/Invoke transition, just a
    /// direct native call instruction.
    /// </summary>
    internal static bool HasRawSgemm => _sgemmPtr != IntPtr.Zero;
    internal static bool HasRawDgemm => _dgemmPtr != IntPtr.Zero;

    /// <summary>
    /// Zero-overhead SGEMM via raw function pointer. The JIT emits a direct calli instruction
    /// to the native cblas_sgemm — no delegate allocation, no P/Invoke transition, no GC interaction.
    /// Caller must pin all arrays before calling.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgemmRaw(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc)
    {
        ((delegate* unmanaged[Cdecl]<int, int, int, int, int, int, float, float*, int, float*, int, float, float*, int, void>)_sgemmPtr)(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
    }

    /// <summary>
    /// Zero-overhead DGEMM via raw function pointer.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void DgemmRaw(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        ((delegate* unmanaged[Cdecl]<int, int, int, int, int, int, double, double*, int, double*, int, double, double*, int, void>)_dgemmPtr)(
            CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }
#endif

    /// <summary>
    /// Historically "MKL.NET available" — after the MKL.NET removal this always
    /// returns false. Kept for API continuity (some callers gate hot paths on
    /// this). Consumers should prefer <see cref="HasNativeSgemm"/> or
    /// <see cref="HasRawSgemm"/>.
    /// </summary>
    internal static bool HasMklNet => false;

    /// <summary>
    /// Historically "MKL verified" — after the MKL.NET removal this aliases the
    /// native-BLAS availability check. <c>IsMklVerified</c> == true means the
    /// native raw-pointer path is usable. Returns false in deterministic mode
    /// so direct-BLAS hot paths fall through to the blocked matmul fallback.
    /// </summary>
    internal static bool IsMklVerified
    {
        get
        {
            if (_deterministicMode) return false;
            EnsureInitialized();
#if NET5_0_OR_GREATER
            return _sgemmPtr != IntPtr.Zero;
#else
            return _sgemm != null;
#endif
        }
    }

    /// <summary>
    /// Zero-offset SGEMM fast path. Historically used MKL.NET Blas.gemm; now
    /// wraps the native cblas_sgemm raw pointer for equivalent zero-overhead
    /// dispatch. Only call when <see cref="IsMklVerified"/> is true and all
    /// arrays start at offset 0.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void MklSgemmZeroOffset(int m, int n, int k, float[] a, int lda, float[] b, int ldb, float[] c, int ldc)
    {
#if NET5_0_OR_GREATER
        fixed (float* ap = a) fixed (float* bp = b) fixed (float* cp = c)
            SgemmRaw(m, n, k, ap, lda, bp, ldb, cp, ldc);
#else
        fixed (float* ap = a) fixed (float* bp = b) fixed (float* cp = c)
            _sgemm!(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, ap, lda, bp, ldb, 0.0f, cp, ldc);
#endif
    }

    /// <summary>
    /// Zero-offset DGEMM fast path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void MklDgemmZeroOffset(int m, int n, int k, double[] a, int lda, double[] b, int ldb, double[] c, int ldc)
    {
#if NET5_0_OR_GREATER
        fixed (double* ap = a) fixed (double* bp = b) fixed (double* cp = c)
            DgemmRaw(m, n, k, ap, lda, bp, ldb, cp, ldc);
#else
        fixed (double* ap = a) fixed (double* bp = b) fixed (double* cp = c)
            _dgemm!(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, ap, lda, bp, ldb, 0.0, cp, ldc);
#endif
    }

    /// <summary>
    /// SGEMM with zero offsets — historical MKL.NET name, now wraps native.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void MklSgemmDirect(int m, int n, int k, float[] a, int lda, float[] b, int ldb, float[] c, int ldc)
        => MklSgemmZeroOffset(m, n, k, a, lda, b, ldb, c, ldc);

    /// <summary>
    /// DGEMM with zero offsets — historical MKL.NET name, now wraps native.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void MklDgemmDirect(int m, int n, int k, double[] a, int lda, double[] b, int ldb, double[] c, int ldc)
        => MklDgemmZeroOffset(m, n, k, a, lda, b, ldb, c, ldc);

    /// <summary>
    /// Raw SGEMM call with pre-pinned pointers. Skips ALL validation, initialization checks,
    /// and GC pinning. Caller must ensure pointers are valid and arrays are pinned.
    /// Only available when HasNativeSgemm is true.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void SgemmDirect(int m, int n, int k, float* a, int lda, float* b, int ldb, float* c, int ldc)
    {
        _sgemm!(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, lda, b, ldb, 0.0f, c, ldc);
    }

    /// <summary>
    /// Raw DGEMM call with pre-pinned pointers.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe void DgemmDirect(int m, int n, int k, double* a, int lda, double* b, int ldb, double* c, int ldc)
    {
        _dgemm!(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a, lda, b, ldb, 0.0, c, ldc);
    }

    /// <summary>
    /// GEMM with configurable beta: C = A*B + beta*C. When beta=1 and C is pre-filled with bias,
    /// this fuses bias addition into the GEMM — saving a full O(MN) memory pass.
    /// </summary>
    internal static bool TryGemmWithBeta(int m, int n, int k, float[] a, int aOffset, int lda, float[] b, int bOffset, int ldb, float[] c, int cOffset, int ldc, float beta)
    {
        if (_deterministicMode) return false;
        if (!EnsureInitialized()) return false;
        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
            return false;

        if (_sgemm == null) return false;

        unsafe
        {
            fixed (float* ap = a, bp = b, cp = c)
                _sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, ap + aOffset, lda, bp + bOffset, ldb, beta, cp + cOffset, ldc);
        }
        return true;
    }

    /// <summary>
    /// Double GEMM with configurable beta.
    /// </summary>
    internal static bool TryGemmWithBeta(int m, int n, int k, double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb, double[] c, int cOffset, int ldc, double beta)
    {
        if (_deterministicMode) return false;
        if (!EnsureInitialized()) return false;
        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
            return false;

        if (_dgemm == null) return false;

        unsafe
        {
            fixed (double* ap = a, bp = b, cp = c)
                _dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, ap + aOffset, lda, bp + bOffset, ldb, beta, cp + cOffset, ldc);
        }
        return true;
    }

    /// <summary>
    /// GEMM with transpose flags: C = alpha * op(A) * op(B) + beta * C
    /// where op(X) = X if transX=false, X^T if transX=true.
    /// Critical for backward pass performance (dA = dC @ B^T, dB = A^T @ dC).
    /// </summary>
    internal static bool TryGemmEx(int m, int n, int k,
        float[] a, int aOffset, int lda, bool transA,
        float[] b, int bOffset, int ldb, bool transB,
        float[] c, int cOffset, int ldc)
    {
        if (_deterministicMode) return false;
        if (!EnsureInitialized()) return false;

        int aRows = transA ? k : m;
        int aCols = transA ? m : k;
        int bRows = transB ? n : k;
        int bCols = transB ? k : n;
        if (!HasEnoughData(a.Length, aOffset, aRows, aCols, lda) ||
            !HasEnoughData(b.Length, bOffset, bRows, bCols, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
        {
            return false;
        }

        int transAFlag = transA ? CblasTrans : CblasNoTrans;
        int transBFlag = transB ? CblasTrans : CblasNoTrans;

        if (_sgemm == null) return false;

        unsafe
        {
            fixed (float* aPtrBase = a)
            fixed (float* bPtrBase = b)
            fixed (float* cPtrBase = c)
            {
                float* aPtr = aPtrBase + aOffset;
                float* bPtr = bPtrBase + bOffset;
                float* cPtr = cPtrBase + cOffset;
                _sgemm(CblasRowMajor, transAFlag, transBFlag, m, n, k, 1.0f, aPtr, lda, bPtr, ldb, 0.0f, cPtr, ldc);
            }
        }

        return true;
    }

    /// <summary>
    /// Span-based GEMM for zero-copy operations.
    /// </summary>
    internal static bool TryGemm(int m, int n, int k, ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        if (_deterministicMode) return false;
        if (!EnsureInitialized())
        {
            return false;
        }

        if (_sgemm == null)
        {
            return false;
        }

        unsafe
        {
            fixed (float* aPtr = a, bPtr = b, cPtr = c)
                _sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, aPtr, lda, bPtr, ldb, 0.0f, cPtr, ldc);
        }

        return true;
    }

    /// <summary>
    /// Span-based GEMM for zero-copy operations (double precision).
    /// </summary>
    internal static bool TryGemm(int m, int n, int k, ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (_deterministicMode) return false;
        if (!EnsureInitialized())
        {
            return false;
        }

        if (_dgemm == null)
        {
            return false;
        }

        unsafe
        {
            fixed (double* aPtr = a, bPtr = b, cPtr = c)
                _dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, aPtr, lda, bPtr, ldb, 0.0, cPtr, ldc);
        }

        return true;
    }

    internal static bool TryGemm(int m, int n, int k, double[] a, int aOffset, int lda, double[] b, int bOffset, int ldb, double[] c, int cOffset, int ldc)
    {
        if (_deterministicMode) return false;
        if (!EnsureInitialized()) return false;
        if (!HasEnoughData(a.Length, aOffset, m, k, lda) ||
            !HasEnoughData(b.Length, bOffset, k, n, ldb) ||
            !HasEnoughData(c.Length, cOffset, m, n, ldc))
        {
            return false;
        }

        if (_dgemm == null) return false;

        unsafe
        {
            fixed (double* aPtrBase = a)
            fixed (double* bPtrBase = b)
            fixed (double* cPtrBase = c)
            {
                double* aPtr = aPtrBase + aOffset;
                double* bPtr = bPtrBase + bOffset;
                double* cPtr = cPtrBase + cOffset;
                _dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, aPtr, lda, bPtr, ldb, 0.0, cPtr, ldc);
            }
        }

        return true;
    }

    private static bool EnsureInitialized()
    {
        if (_initialized)
        {
            return _available;
        }

        lock (InitLock)
        {
            if (_initialized)
            {
                return _available;
            }

            _available = TryLoadLibrary();
            _initialized = true;
            return _available;
        }
    }

    private static bool TryLoadLibrary()
    {
        Trace("[BLAS] TryLoadLibrary starting (native-only — MKL.NET removed)...");
        Trace($"[BLAS] AppContext.BaseDirectory: {AppContext.BaseDirectory}");

        var disable = Environment.GetEnvironmentVariable("AIDOTNET_DISABLE_BLAS");
        if (!string.IsNullOrWhiteSpace(disable) &&
            (string.Equals(disable, "1", StringComparison.OrdinalIgnoreCase) ||
             string.Equals(disable, "true", StringComparison.OrdinalIgnoreCase)))
        {
            Trace("[BLAS] Disabled via AIDOTNET_DISABLE_BLAS");
            return false;
        }

        string? explicitPath = Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PATH");
        if (!string.IsNullOrWhiteSpace(explicitPath))
        {
            Trace($"[BLAS] Explicit path from env: {explicitPath}");
            if (TryLoadNativeLibrary(explicitPath, out _libraryHandle))
            {
                Trace("[BLAS] Loaded from explicit path");
                if (TryLoadSymbols())
                {
                    Trace("[BLAS] Symbols loaded successfully from explicit path");
                    return true;
                }

                FreeNativeLibrary(_libraryHandle);
                _libraryHandle = IntPtr.Zero;
            }
            else
            {
                Trace("[BLAS] Failed to load from explicit path");
            }
            return false;
        }

        var candidates = GetCandidateLibraryNames();
        Trace($"[BLAS] Searching {candidates.Length} candidate paths...");

        foreach (var name in candidates)
        {
            if (!TryLoadNativeLibrary(name, out _libraryHandle))
            {
                continue;
            }

            Trace($"[BLAS] Loaded native library from: {name}");
            if (TryLoadSymbols())
            {
                Trace("[BLAS] BLAS symbols loaded successfully!");
                return true;
            }

            Trace($"[BLAS] Failed to load symbols from: {name}");
            FreeNativeLibrary(_libraryHandle);
            _libraryHandle = IntPtr.Zero;
        }

        Trace("[BLAS] No BLAS library found — matmul will use SimdGemm (AVX2 blocked)");
        return false;
    }

    private static void Trace(string message)
    {
        if (TraceEnabled)
        {
            Console.WriteLine(message);
        }
    }

    private static bool TryLoadSymbols()
    {
        _sgemm = TryLoadSymbol<CblasSgemm>("cblas_sgemm");
        _dgemm = TryLoadSymbol<CblasDgemm>("cblas_dgemm");
#if NET5_0_OR_GREATER
        // Also cache raw function pointers for zero-overhead calli dispatch
        if (_libraryHandle != IntPtr.Zero)
        {
            TryGetNativeExport(_libraryHandle, "cblas_sgemm", out _sgemmPtr);
            TryGetNativeExport(_libraryHandle, "cblas_dgemm", out _dgemmPtr);
        }
#endif
        Trace($"[BLAS] cblas_sgemm loaded: {_sgemm != null}");
        Trace($"[BLAS] cblas_dgemm loaded: {_dgemm != null}");
        TryLoadThreadControls();
        return _sgemm != null || _dgemm != null;
    }

    private static void TryLoadThreadControls()
    {
        _setNumThreads = TryLoadSymbol<BlasSetNumThreads>("openblas_set_num_threads")
            ?? TryLoadSymbol<BlasSetNumThreads>("mkl_set_num_threads");
        _setDynamic = TryLoadSymbol<MklSetDynamic>("mkl_set_dynamic");
        ApplyThreadSettings();
    }

    private static void ApplyThreadSettings()
    {
        if (_setNumThreads == null)
        {
            return;
        }

        // Only invoke native thread-control functions when the caller has explicitly
        // provided a ThreadCountOverride (set via the AIDOTNET_BLAS_THREADS env var).
        // The loaded symbols can be unsafe to call blindly on some platforms — on
        // net10.0 Windows, for example, invoking them without this explicit opt-in
        // has been seen to trigger access violations (see blas_native_init_crash.md).
        // Falling through leaves the native library at its own default thread count.
        if (!ThreadCountOverride.HasValue)
        {
            Trace("[BLAS] ApplyThreadSettings skipped — no explicit thread count requested");
            return;
        }

        try
        {
            int threadCount = ThreadCountOverride.Value;
            _setDynamic?.Invoke(0);
            _setNumThreads(threadCount);
            Trace($"[BLAS] Set thread count to {threadCount}");
        }
        catch (DllNotFoundException) { /* optional — don't fail */ }
        catch (EntryPointNotFoundException) { /* optional — don't fail */ }
        catch (BadImageFormatException) { /* optional — don't fail */ }
        catch (SEHException) { /* optional — don't fail */ }
    }

    private static T? TryLoadSymbol<T>(string name) where T : class
    {
        if (_libraryHandle == IntPtr.Zero)
        {
            return null;
        }

        if (!TryGetNativeExport(_libraryHandle, name, out var symbol) || symbol == IntPtr.Zero)
        {
            return null;
        }

        return Marshal.GetDelegateForFunctionPointer(symbol, typeof(T)) as T;
    }

    private static bool HasEnoughData(int length, int offset, int rows, int cols, int stride)
    {
        if (length <= 0 || rows <= 0 || cols <= 0)
        {
            return false;
        }

        if (offset < 0 || stride < cols)
        {
            return false;
        }

        long lastIndex = (long)offset + (long)(rows - 1) * stride + (cols - 1);
        return lastIndex < length;
    }

    private static string[] GetCandidateLibraryNames()
    {
        // After the MKL.NET removal, OpenBLAS is the preferred candidate library
        // (bundled via the optional AiDotNet.Native.OpenBLAS package). MKL native
        // DLLs are still loaded if the user supplies them — the mkl_rt symbols
        // follow the same cblas_sgemm ABI — but we no longer bundle or depend
        // on MKL in any form.
        string[] openblas =
        {
            "openblas",
            "libopenblas",
            "openblas.dll",
            "libopenblas.dll",
            "libopenblas.so",
            "libopenblas.so.0",
            "libopenblas.dylib"
        };

        string[] mkl =
        {
            "mkl_rt",
            "mkl_rt.dll",
            "libmkl_rt.so",
            "libmkl_rt.dylib"
        };

        // Prefer OpenBLAS by default; user can override via AIDOTNET_BLAS_PATH.
        var names = new List<string>(openblas.Length + mkl.Length);
        names.AddRange(openblas);
        names.AddRange(mkl);

        // Also try loading from various known directories
        var additionalPaths = new List<string>();
        var directories = new List<string?>
        {
            AppContext.BaseDirectory,
            Path.GetDirectoryName(typeof(BlasProvider).Assembly.Location),
            Environment.CurrentDirectory
        };

        // Also search in parent directories (helps with BenchmarkDotNet artifact subdirectories)
        var baseDir = AppContext.BaseDirectory;
        for (int i = 0; i < 5 && !string.IsNullOrEmpty(baseDir); i++)
        {
            baseDir = Path.GetDirectoryName(baseDir);
            if (!string.IsNullOrEmpty(baseDir))
            {
                directories.Add(baseDir);
            }
        }

        // Search in runtimes/win-x64/native subdirectory (common NuGet native layout)
        foreach (var dir in directories.ToList())
        {
            if (!string.IsNullOrEmpty(dir))
            {
                var nativePath = Path.Combine(dir!, "runtimes", "win-x64", "native");
                if (Directory.Exists(nativePath))
                {
                    directories.Add(nativePath);
                }
            }
        }

        foreach (var dir in directories.Where(d => !string.IsNullOrEmpty(d)))
        {
            foreach (var name in names.ToArray())
            {
                var fullPath = Path.Combine(dir!, name);
                if (!additionalPaths.Contains(fullPath))
                {
                    additionalPaths.Add(fullPath);
                }
            }
        }

        // Add directory-relative paths at the beginning for priority
        names.InsertRange(0, additionalPaths);

        return names.ToArray();
    }

    private static int? ReadEnvInt(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return null;
        }

        return int.TryParse(raw, out var value) && value > 0 ? value : null;
    }

    private static bool ReadEnvBool(string name)
    {
        var raw = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(raw))
        {
            return false;
        }

        return string.Equals(raw, "1", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "true", StringComparison.OrdinalIgnoreCase) ||
               string.Equals(raw, "yes", StringComparison.OrdinalIgnoreCase);
    }

#if NETFRAMEWORK
    private static bool TryLoadNativeLibrary(string path, out IntPtr handle)
    {
        handle = LoadLibrary(path);
        return handle != IntPtr.Zero;
    }

    private static bool TryGetNativeExport(IntPtr handle, string name, out IntPtr symbol)
    {
        symbol = GetProcAddress(handle, name);
        return symbol != IntPtr.Zero;
    }

    private static void FreeNativeLibrary(IntPtr handle)
    {
        if (handle != IntPtr.Zero)
        {
            FreeLibrary(handle);
        }
    }
#else
    private static bool TryLoadNativeLibrary(string path, out IntPtr handle)
        => NativeLibrary.TryLoad(path, out handle);

    private static bool TryGetNativeExport(IntPtr handle, string name, out IntPtr symbol)
        => NativeLibrary.TryGetExport(handle, name, out symbol);

    private static void FreeNativeLibrary(IntPtr handle)
        => NativeLibrary.Free(handle);
#endif
}
