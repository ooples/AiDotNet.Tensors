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
    // Raw P/Invoke. NEVER call these directly — go through the gated
    // cblas_sgemm_ptr / cblas_dgemm_ptr wrappers below, which serialize native
    // entry (OpenBLAS crashes on concurrent entry; see _nativeGemmGate).
    [DllImport("libopenblas", EntryPoint = "cblas_sgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void cblas_sgemm_native(
        int layout, int transA, int transB,
        int m, int n, int k,
        float alpha, float* a, int lda,
        float* b, int ldb,
        float beta, float* c, int ldc);

    [DllImport("libopenblas", EntryPoint = "cblas_dgemm", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void cblas_dgemm_native(
        int layout, int transA, int transB,
        int m, int n, int k,
        double alpha, double* a, int lda,
        double* b, int ldb,
        double beta, double* c, int ldc);

    /// <summary>
    /// OpenBLAS thread-count knob. Setting to 1 forces single-threaded GEMM
    /// kernel dispatch — required for bit-exact reproducibility because
    /// OpenBLAS multi-threaded GEMM partitions the K-dimension across threads
    /// and sums the partial products in (thread-completion-order, not
    /// fixed-order), so floating-point non-associativity makes the same
    /// inputs produce different outputs across runs.
    /// </summary>
    [DllImport("libopenblas", EntryPoint = "openblas_set_num_threads", CallingConvention = CallingConvention.Cdecl)]
    private static extern void openblas_set_num_threads_native(int num_threads);

    private static int _openblasThreadCount = -1; // -1 = unset

    /// <summary>
    /// Sets the OpenBLAS internal thread count. Pass 1 to force deterministic
    /// single-threaded GEMM. Caller must verify <see cref="HasNativeDgemm"/>
    /// is true before calling, otherwise the P/Invoke will throw.
    /// </summary>
    /// <remarks>
    /// Issue #411 CI history: an earlier libgomp probe variant also called
    /// <c>omp_set_num_threads(n)</c> via <c>libgomp.so.1</c> to cover
    /// OpenMP-threaded OpenBLAS builds. That extra call was removed because
    /// it set the OpenMP thread count PROCESS-WIDE, which caused
    /// cross-test interference on xUnit-parallel CI runs (other tests
    /// using OpenMP-backed libraries inherited the cap and starved their
    /// parallel work). Since <see cref="Engines.CpuEngine.FlashAttentionDouble"/>
    /// now bypasses BLAS entirely (using <see cref="Simd.SimdGemm.DgemmSequential"/>
    /// directly), the libgomp guard is no longer load-bearing for the
    /// originally-targeted #411 regression test.
    /// </remarks>
    internal static void TrySetOpenBlasThreads(int n)
    {
        if (!_nativeAvailable.Value) return;
        if (_openblasThreadCount == n) return;
        try { openblas_set_num_threads_native(n); _openblasThreadCount = n; }
        catch { /* libopenblas symbol missing — earlier OpenBLAS builds may lack it. Tolerate. */ }
    }

    /// <summary>
    /// Returns the cached OpenBLAS thread count last set by <see cref="TrySetOpenBlasThreads(int)"/>,
    /// or <c>-1</c> if no override has been applied this process. Used by callers that
    /// run an outer parallel-for around BLAS calls and need to scope OpenBLAS down to
    /// 1 thread (avoid oversubscription) then restore the prior value on exit.
    /// </summary>
    /// <remarks>
    /// PR #410 CodeRabbit fix: prefer <see cref="ScopeOpenBlasThreads(int)"/> over
    /// directly reading this value for save/restore patterns — the scope token uses a
    /// lock-protected depth counter so overlapping/nested calls don't restore stale
    /// values. This getter is left in place for backward compat with eager callers
    /// that already implement their own correctness guarantees.
    /// </remarks>
    internal static int GetOpenBlasThreadCount() => _openblasThreadCount;

    // PR #410 CodeRabbit fix: lock-protected scope for the OpenBLAS thread-count
    // set/restore contract. Without this, two concurrent callers each saving/setting/
    // restoring around their own BLAS-bound parallel-for could leave the global
    // thread count at a stale value (e.g., both see savedCount=8 → both set to 1 →
    // inner thread restores to 1 prematurely → outer thread restores to 1 instead
    // of 8). The depth counter ensures only the OUTERMOST scope restores.
    private static readonly object _openblasScopeLock = new object();
    private static int _openblasScopeDepth;
    private static int _openblasScopeRestoreTarget = -1;

    /// <summary>
    /// PR #410 CodeRabbit fix: thread-safe scoped override of the OpenBLAS thread
    /// count. Returns a disposable token; on disposal the outermost scope restores
    /// the prior thread count atomically. Nested/overlapping scopes within the same
    /// process are reference-counted — only the outer disposal touches OpenBLAS.
    ///
    /// <para>
    /// Use this for the "run an outer parallel-for around BLAS calls, force OpenBLAS
    /// to 1 thread to avoid oversubscription, restore on exit" pattern. Eager callers
    /// using <see cref="GetOpenBlasThreadCount"/> + <see cref="TrySetOpenBlasThreads"/>
    /// directly are NOT concurrency-safe across threads.
    /// </para>
    /// </summary>
    /// <param name="requestedThreads">Desired OpenBLAS thread count for the scope.</param>
    internal static OpenBlasThreadScope ScopeOpenBlasThreads(int requestedThreads)
    {
        if (!_nativeAvailable.Value) return default;
        return new OpenBlasThreadScope(requestedThreads);
    }

    /// <summary>
    /// Disposable scope token returned by <see cref="ScopeOpenBlasThreads(int)"/>.
    /// Restores the prior OpenBLAS thread count when the outermost scope disposes.
    /// </summary>
    internal readonly struct OpenBlasThreadScope : IDisposable
    {
        private readonly bool _active;

        internal OpenBlasThreadScope(int requestedThreads)
        {
            lock (_openblasScopeLock)
            {
                if (_openblasScopeDepth == 0)
                {
                    _openblasScopeRestoreTarget = _openblasThreadCount;
                }
                _openblasScopeDepth++;
                TrySetOpenBlasThreads(requestedThreads);
            }
            _active = true;
        }

        public void Dispose()
        {
            if (!_active) return;
            lock (_openblasScopeLock)
            {
                _openblasScopeDepth--;
                if (_openblasScopeDepth == 0)
                {
                    int restore = _openblasScopeRestoreTarget;
                    TrySetOpenBlasThreads(restore < 0 ? 0 : restore);
                }
            }
        }
    }


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

    // ─────────────────────────────────────────────────────────────────────
    // Issue #338 Phase G.5 — BF16 mixed-precision GEMM via full Intel MKL.
    // cblas_gemm_bf16bf16f32 is exported by mkl_rt.3.dll (from
    // intelmkl.redist.win-x64). NOT exported by Microsoft.ML.Mkl.Redist's
    // stripped MklImports.dll. Computes:
    //   C[m,n] = alpha * op(A_bf16) * op(B_bf16) + beta * C[m,n]
    // where A and B are BF16 (uint16 top-16-bits-of-FP32) and C is FP32.
    // On AVX-512-BF16 capable Intel CPUs (Cooper Lake, Sapphire Rapids,
    // etc.) this is ~2× faster than the FP32 sgemm via the dedicated
    // BF16 ALU path. On older CPUs MKL falls back to FP32 internally
    // (correctness preserved, no speedup).
    //
    // Opt-in via env var AIDOTNET_BLAS_PROVIDER=mkl-bf16. Distinct from
    // the `mkl` setting because BF16 requires the full Intel MKL redist
    // (intelmkl.redist.*, ~500MB on win-x64); the smaller Microsoft.ML.Mkl.Redist
    // path stays available under `AIDOTNET_BLAS_PROVIDER=mkl`.
    // ─────────────────────────────────────────────────────────────────────
    [DllImport("mkl_rt.3", EntryPoint = "cblas_gemm_bf16bf16f32", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void mkl_cblas_gemm_bf16bf16f32_ptr(
        int layout, int transA, int transB,
        int m, int n, int k,
        float alpha, ushort* a, int lda,
        ushort* b, int ldb,
        float beta, float* c, int ldc);

    // ─────────────────────────────────────────────────────────────────────
    // Issue #338 Phase G.10 — batched GEMM via MKL cblas_sgemm_batch.
    // Amortises per-call P/Invoke + MKL setup overhead across a group of
    // GEMMs with possibly different shapes. Used for the "compute dA and
    // dW backward GEMMs in one call" pattern in MatMul-backward
    // specializations.
    //
    // C signature:
    //   void cblas_sgemm_batch(
    //     CBLAS_LAYOUT Layout,
    //     const CBLAS_TRANSPOSE* TransA_array,
    //     const CBLAS_TRANSPOSE* TransB_array,
    //     const MKL_INT* M_array, const MKL_INT* N_array, const MKL_INT* K_array,
    //     const float* alpha_array,
    //     const float** A_array, const MKL_INT* lda_array,
    //     const float** B_array, const MKL_INT* ldb_array,
    //     const float* beta_array,
    //     float** C_array, const MKL_INT* ldc_array,
    //     MKL_INT group_count,
    //     const MKL_INT* group_size);
    //
    // For our use case (2 distinct-shape GEMMs in one call), we set
    // group_count=2, group_size=[1, 1], and pass per-group arrays.
    // ─────────────────────────────────────────────────────────────────────
    [DllImport("mkl_rt.3", EntryPoint = "cblas_sgemm_batch", CallingConvention = CallingConvention.Cdecl)]
    private static extern unsafe void mkl_cblas_sgemm_batch_ptr(
        int layout,
        int* transA_array, int* transB_array,
        int* m_array, int* n_array, int* k_array,
        float* alpha_array,
        float** a_array, int* lda_array,
        float** b_array, int* ldb_array,
        float* beta_array,
        float** c_array, int* ldc_array,
        int group_count,
        int* group_size);

    /// <summary>
    /// Issue #338 Phase G.10: probe MKL's cblas_sgemm_batch availability.
    /// Only fires when MKL is preferred AND cblas_sgemm_batch loads. Gates
    /// the batched-GEMM dispatch in MatMul-backward specs.
    /// </summary>
    private static readonly Lazy<bool> _mklBatchedAvailable = new(ProbeMklBatchedLibrary);
    internal static bool IsMklBatchedAvailable => _mklBatchedAvailable.Value;

    private static unsafe bool ProbeMklBatchedLibrary()
    {
        if (!_preferMkl) return false;
        if (!_nativeAvailable.Value) return false;
        try
        {
            // 1×1×1 GEMM with group_count=1, group_size=1
            float a = 2f, b = 3f, c = 0f;
            float* aP = &a, bP = &b, cP = &c;
            int transA = CblasNoTrans, transB = CblasNoTrans;
            int m = 1, n = 1, k = 1, lda = 1, ldb = 1, ldc = 1;
            float alpha = 1f, beta = 0f;
            int groupSize = 1;
            mkl_cblas_sgemm_batch_ptr(CblasRowMajor,
                &transA, &transB,
                &m, &n, &k,
                &alpha,
                &aP, &lda,
                &bP, &ldb,
                &beta,
                &cP, &ldc,
                1, &groupSize);
            return c == 6f;
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
    }

    /// <summary>
    /// Issue #338 Phase G.12: batch N same-shape GEMMs via cblas_sgemm_batch.
    /// Array-based variant — pins arrays internally so the caller doesn't
    /// pay per-call GCHandle overhead. Used for layer-level dW batching
    /// where 4 layers' weight gradients share (M, N, K).
    /// </summary>
    internal static unsafe bool TryGemmBatchSameShapeArrays(
        int m, int n, int k,
        bool transA, bool transB,
        float[][] aArrays, int[] aOffsets, int lda,
        float[][] bArrays, int[] bOffsets, int ldb,
        float[][] cArrays, int[] cOffsets, int ldc,
        int batchSize)
    {
        if (!_mklBatchedAvailable.Value) return false;
        if (batchSize <= 0) return true;
        try
        {
            float** aArr = stackalloc float*[batchSize];
            float** bArr = stackalloc float*[batchSize];
            float** cArr = stackalloc float*[batchSize];

            // Pin each array via fixed inside a recursive-style scope.
            // Since the count is dynamic (1..N batched specs), we can't
            // use multiple `fixed` keywords statically; use GCHandle here
            // BUT only for the pointer extraction — MKL grabs the
            // pointers and runs synchronously, so we can free immediately
            // after the call returns. This is faster than per-array
            // GCHandle.Pinned with try/finally cycles since we batch the
            // allocation in a single loop.
            var handles = stackalloc System.Runtime.InteropServices.GCHandle[batchSize * 3];
            for (int i = 0; i < batchSize; i++)
            {
                handles[i * 3 + 0] = System.Runtime.InteropServices.GCHandle.Alloc(aArrays[i], System.Runtime.InteropServices.GCHandleType.Pinned);
                handles[i * 3 + 1] = System.Runtime.InteropServices.GCHandle.Alloc(bArrays[i], System.Runtime.InteropServices.GCHandleType.Pinned);
                handles[i * 3 + 2] = System.Runtime.InteropServices.GCHandle.Alloc(cArrays[i], System.Runtime.InteropServices.GCHandleType.Pinned);
                aArr[i] = (float*)handles[i * 3 + 0].AddrOfPinnedObject() + aOffsets[i];
                bArr[i] = (float*)handles[i * 3 + 1].AddrOfPinnedObject() + bOffsets[i];
                cArr[i] = (float*)handles[i * 3 + 2].AddrOfPinnedObject() + cOffsets[i];
            }

            try
            {
                int tA = transA ? 112 : CblasNoTrans;
                int tB = transB ? 112 : CblasNoTrans;
                int mPlain = m, nPlain = n, kPlain = k;
                float alpha = 1f, beta = 0f;
                int ldaPlain = lda, ldbPlain = ldb, ldcPlain = ldc;
                int groupSize = batchSize;

                mkl_cblas_sgemm_batch_ptr(CblasRowMajor,
                    &tA, &tB,
                    &mPlain, &nPlain, &kPlain,
                    &alpha,
                    aArr, &ldaPlain,
                    bArr, &ldbPlain,
                    &beta,
                    cArr, &ldcPlain,
                    1, &groupSize);
            }
            finally
            {
                for (int i = 0; i < batchSize * 3; i++)
                    handles[i].Free();
            }
            return true;
        }
        catch { return false; }
    }

    /// <summary>
    /// Issue #338 Phase G.12: batch N same-shape GEMMs into a single
    /// MKL call via cblas_sgemm_batch with group_count=1. Used for the
    /// "compute dW across all layers' MatMul backwards" pattern — all
    /// share (M, N, K) and trans flags. Caller pre-pins arrays.
    /// </summary>
    internal static unsafe bool TryGemmBatchSameShape(
        int m, int n, int k,
        bool transA, bool transB,
        System.Runtime.InteropServices.GCHandle[] aHandles, int[] aOffsets, int lda,
        System.Runtime.InteropServices.GCHandle[] bHandles, int[] bOffsets, int ldb,
        System.Runtime.InteropServices.GCHandle[] cHandles, int[] cOffsets, int ldc,
        int batchSize)
    {
        if (!_mklBatchedAvailable.Value) return false;
        if (batchSize <= 0) return true;
        try
        {
            float** aArr = stackalloc float*[batchSize];
            float** bArr = stackalloc float*[batchSize];
            float** cArr = stackalloc float*[batchSize];
            for (int i = 0; i < batchSize; i++)
            {
                aArr[i] = (float*)aHandles[i].AddrOfPinnedObject() + aOffsets[i];
                bArr[i] = (float*)bHandles[i].AddrOfPinnedObject() + bOffsets[i];
                cArr[i] = (float*)cHandles[i].AddrOfPinnedObject() + cOffsets[i];
            }

            int tA = transA ? 112 : CblasNoTrans;
            int tB = transB ? 112 : CblasNoTrans;
            int mPlain = m, nPlain = n, kPlain = k;
            float alpha = 1f, beta = 0f;
            int ldaPlain = lda, ldbPlain = ldb, ldcPlain = ldc;
            int groupSize = batchSize;

            mkl_cblas_sgemm_batch_ptr(CblasRowMajor,
                &tA, &tB,
                &mPlain, &nPlain, &kPlain,
                &alpha,
                aArr, &ldaPlain,
                bArr, &ldbPlain,
                &beta,
                cArr, &ldcPlain,
                1, &groupSize);
            return true;
        }
        catch { return false; }
    }

    /// <summary>
    /// Phase G.10: batch two GEMMs (dA + dW backward pair) into one MKL call.
    /// Both GEMMs may have different M, N, K, lda, ldb, ldc and trans flags.
    /// Returns false if MKL batched API isn't available; caller falls back
    /// to 2 separate <see cref="TryGemmEx"/> calls.
    /// </summary>
    internal static unsafe bool TryGemmExBatch2(
        int m0, int n0, int k0,
        float[] a0, int aOff0, int lda0, bool transA0,
        float[] b0, int bOff0, int ldb0, bool transB0,
        float[] c0, int cOff0, int ldc0,
        int m1, int n1, int k1,
        float[] a1, int aOff1, int lda1, bool transA1,
        float[] b1, int bOff1, int ldb1, bool transB1,
        float[] c1, int cOff1, int ldc1)
    {
        if (!_mklBatchedAvailable.Value) return false;
        try
        {
            fixed (float* pA0 = &a0[aOff0])
            fixed (float* pB0 = &b0[bOff0])
            fixed (float* pC0 = &c0[cOff0])
            fixed (float* pA1 = &a1[aOff1])
            fixed (float* pB1 = &b1[bOff1])
            fixed (float* pC1 = &c1[cOff1])
            {
                int* transA = stackalloc int[2] { transA0 ? 112 : CblasNoTrans, transA1 ? 112 : CblasNoTrans };
                int* transB = stackalloc int[2] { transB0 ? 112 : CblasNoTrans, transB1 ? 112 : CblasNoTrans };
                int* M = stackalloc int[2] { m0, m1 };
                int* N = stackalloc int[2] { n0, n1 };
                int* K = stackalloc int[2] { k0, k1 };
                float* alpha = stackalloc float[2] { 1f, 1f };
                int* lda = stackalloc int[2] { lda0, lda1 };
                int* ldb = stackalloc int[2] { ldb0, ldb1 };
                float* beta = stackalloc float[2] { 0f, 0f };
                int* ldc = stackalloc int[2] { ldc0, ldc1 };
                float** aArr = stackalloc float*[2] { pA0, pA1 };
                float** bArr = stackalloc float*[2] { pB0, pB1 };
                float** cArr = stackalloc float*[2] { pC0, pC1 };
                int* groupSize = stackalloc int[2] { 1, 1 };

                mkl_cblas_sgemm_batch_ptr(CblasRowMajor,
                    transA, transB,
                    M, N, K,
                    alpha,
                    aArr, lda,
                    bArr, ldb,
                    beta,
                    cArr, ldc,
                    2, groupSize);
            }
            return true;
        }
        catch { return false; }
    }

    /// <summary>
    /// True iff <c>AIDOTNET_BLAS_PROVIDER=mkl-bf16</c> OR
    /// <c>mkl-bf16-force</c> at process start. The force variant also
    /// flips this on — without that, ProbeMklBf16Library's leading
    /// `if (!_preferMklBf16) return false;` guard short-circuits before
    /// the force bypass deeper in the probe can engage, and UseMklBf16
    /// stays false forever.
    /// </summary>
    private static readonly bool _preferMklBf16 =
        string.Equals(
            Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER"),
            "mkl-bf16",
            StringComparison.OrdinalIgnoreCase)
        || string.Equals(
            Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER"),
            "mkl-bf16-force",
            StringComparison.OrdinalIgnoreCase);

    /// <summary>Resolved lazily on first use — probes mkl_rt.3 BF16 GEMM once.</summary>
    private static readonly Lazy<bool> _mklBf16Available = new(ProbeMklBf16Library);

    internal static bool IsMklBf16Available => _mklBf16Available.Value;

    /// <summary>True iff <c>AIDOTNET_BLAS_PROVIDER=mkl-bf16</c> AND the probe succeeded.</summary>
    internal static bool UseMklBf16 => _preferMklBf16 && _mklBf16Available.Value;

    /// <summary>True iff <c>AIDOTNET_USE_BLAS=1</c>|<c>true</c>|<c>yes</c> at process start.</summary>
    private static readonly bool _blasOptIn = ReadOptIn();

    /// <summary>
    /// True iff <c>AIDOTNET_BLAS_PROVIDER=mkl</c> OR any value that starts
    /// with <c>mkl-</c> (e.g. <c>mkl-bf16</c> implies MKL routing).
    /// When false, MKL probes are skipped and OpenBLAS is the only candidate.
    /// </summary>
    private static readonly bool _preferMkl = PreferMklFromEnv();
    private static bool PreferMklFromEnv()
    {
        var raw = Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER");
        if (raw is null) return false;
        return string.Equals(raw, "mkl", StringComparison.OrdinalIgnoreCase)
            || raw.StartsWith("mkl-", StringComparison.OrdinalIgnoreCase);
    }

    /// <summary>Resolved lazily on first use — probes libopenblas once.</summary>
    private static readonly Lazy<bool> _nativeAvailable = new(ProbeNativeLibrary);

    /// <summary>Resolved lazily on first use — probes MklImports once.</summary>
    private static readonly Lazy<bool> _mklAvailable = new(ProbeMklLibrary);

    // ─────────────────────────────────────────────────────────────────────
    // Shape-instrumentation hook (issue #403 Phase A.1).
    //
    // When subscribed, every successful TryGemm/TryGemmWithBeta/TryGemmEx
    // call invokes the hook with (m, n, k, transA, transB) so a higher-level
    // profiler can build a unique-shape catalog without modifying call
    // sites. The hook is a single volatile delegate read on the fast path —
    // when unset, the per-call cost is one null check.
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
        if (!_blasOptIn)
        {
            _loadError = "native CPU BLAS disabled via AIDOTNET_USE_BLAS opt-out";
            return false;
        }
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
            if (c[0] == 6.0f) return true;
            _loadError = $"native CPU BLAS loaded but returned an incorrect result for the 1x1 probe GEMM (expected 6.0, got {c[0]}) — the library may be corrupt or ABI-incompatible";
            return false;
        }
        // Issue #444: capture WHY the native library failed to load so the
        // diagnostic surface (NativeLibraryDetector / PlatformDetector) can
        // distinguish "package not installed" from a transitive-dependency
        // load failure (e.g. a missing libgcc/libgfortran/pthread runtime that
        // makes Win32 LoadLibrary fail with ERROR_MOD_NOT_FOUND (126) even
        // though libopenblas.dll itself is present on disk).
        catch (DllNotFoundException ex)
        {
            _loadError = $"native CPU BLAS module not found (the package may not be installed, or a dependent runtime DLL is missing): {ex.Message}";
            return false;
        }
        catch (EntryPointNotFoundException ex)
        {
            _loadError = $"native CPU BLAS loaded but the cblas_sgemm entry point is missing (unexpected library or version): {ex.Message}";
            return false;
        }
        catch (BadImageFormatException ex)
        {
            _loadError = $"native CPU BLAS architecture mismatch (e.g. x86 DLL in an x64 process): {ex.Message}";
            return false;
        }
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

    /// <summary>
    /// Issue #338 Phase G.5: probe mkl_rt.3 cblas_gemm_bf16bf16f32.
    /// Only runs when AIDOTNET_BLAS_PROVIDER=mkl-bf16. Tests with:
    /// (a) 1×1×1 correctness GEMM (2.0 * 3.0 = 6.0 within BF16 tolerance)
    /// (b) 256×256×256 speed smoke test vs FP32 sgemm — BF16 must be
    ///     ≥1.3× faster to count as available. On CPUs without
    ///     AVX-512-BF16 instructions (Zen 2/3 AMD, pre-Cooper-Lake Intel)
    ///     MKL software-emulates BF16 with FP32 lowering, which is
    ///     SLOWER than direct FP32 sgemm — opt-in correctly refuses
    ///     so users don't accidentally pay a regression.
    /// Skip the speed smoke with AIDOTNET_BLAS_PROVIDER=mkl-bf16-force
    /// (for testing the kernel path on slow CPUs).
    /// <para>
    /// IMPORTANT: this probe LOADS mkl_rt.3.dll into the process. On
    /// systems where Microsoft.ML.Mkl.Redist's MklImports.dll is already
    /// loaded (via the `mkl` routing), the two MKL builds ship competing
    /// libiomp5md.dll OpenMP runtimes and the loader collision REGRESSES
    /// FP32 GEMM throughput substantially — even when BF16 itself is
    /// never called. To avoid this, the probe is only run when BF16 is
    /// explicitly opted into AND the user accepts the deployment caveat.
    /// Set <c>AIDOTNET_BLAS_PROVIDER=mkl</c> for the safer "MKL routing
    /// only, no BF16" path; use <c>mkl-bf16</c> only when targeting
    /// AVX-512-BF16 hardware and ready to manage the OpenMP loader
    /// hygiene (typically by removing MklImports.dll from the deploy
    /// directory or building a project that excludes
    /// Microsoft.ML.Mkl.Redist).
    /// </para>
    /// </summary>
    private static unsafe bool ProbeMklBf16Library()
    {
        if (!_preferMklBf16) return false;
        try
        {
            ushort a = Fp32ToBf16(2.0f);
            ushort b = Fp32ToBf16(3.0f);
            float c = 0.0f;
            mkl_cblas_gemm_bf16bf16f32_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                1, 1, 1, 1.0f, &a, 1, &b, 1, 0.0f, &c, 1);
            if (Math.Abs(c - 6.0f) > 0.01f) return false;

            // Force-on bypass for testing on slow CPUs. Also the
            // supported workaround for minimal-deploy scenarios that
            // exclude Microsoft.ML.Mkl.Redist (MklImports.dll) — the
            // FP32 speed gate below imports cblas_sgemm from MklImports,
            // so without that DLL the speed gate's `mkl_cblas_sgemm_ptr`
            // call throws DllNotFoundException. mkl-bf16-force bypasses
            // the speed comparison entirely and trusts mkl_rt.3 alone.
            if (string.Equals(
                Environment.GetEnvironmentVariable("AIDOTNET_BLAS_PROVIDER"),
                "mkl-bf16-force",
                StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            // Speed smoke test: 256×256×256 GEMM, FP32 vs BF16. The FP32
            // side imports from MklImports (Microsoft.ML.Mkl.Redist). When
            // that DLL is absent (minimal-deploy: only mkl_rt.3 / full
            // intelmkl.redist shipped), the speed gate can't run — but
            // BF16 itself is fully functional from mkl_rt.3. In that case
            // we still enable BF16: the user has explicitly opted in via
            // `mkl-bf16`, and the absent FP32 baseline is precisely the
            // configuration where BF16 is the only fast option anyway.
            try
            {
                const int K = 256;
                var aFp32 = new float[K * K];
                var bFp32 = new float[K * K];
                for (int i = 0; i < K * K; i++) { aFp32[i] = 0.001f * (i & 0xff); bFp32[i] = 0.002f * (i & 0xff); }
                var aBf = new ushort[K * K];
                var bBf = new ushort[K * K];
                fixed (float* aSrc = aFp32) fixed (ushort* aDst = aBf) Fp32ToBf16Bulk(aSrc, aDst, K * K);
                fixed (float* bSrc = bFp32) fixed (ushort* bDst = bBf) Fp32ToBf16Bulk(bSrc, bDst, K * K);
                var cOut = new float[K * K];

                // Warmup
                for (int w = 0; w < 3; w++)
                {
                    fixed (float* pa = aFp32) fixed (float* pb = bFp32) fixed (float* pc = cOut)
                        mkl_cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, K, K, 1f, pa, K, pb, K, 0f, pc, K);
                    fixed (ushort* pa = aBf) fixed (ushort* pb = bBf) fixed (float* pc = cOut)
                        mkl_cblas_gemm_bf16bf16f32_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, K, K, 1f, pa, K, pb, K, 0f, pc, K);
                }

                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < 20; i++)
                    fixed (float* pa = aFp32) fixed (float* pb = bFp32) fixed (float* pc = cOut)
                        mkl_cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, K, K, 1f, pa, K, pb, K, 0f, pc, K);
                sw.Stop();
                double fp32Ms = sw.Elapsed.TotalMilliseconds;

                sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < 20; i++)
                    fixed (ushort* pa = aBf) fixed (ushort* pb = bBf) fixed (float* pc = cOut)
                        mkl_cblas_gemm_bf16bf16f32_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans, K, K, K, 1f, pa, K, pb, K, 0f, pc, K);
                sw.Stop();
                double bf16Ms = sw.Elapsed.TotalMilliseconds;

                // Require BF16 to be ≥1.3× faster than FP32. On AVX-512-BF16
                // capable Intel CPUs this gate passes easily; on CPUs without
                // dedicated BF16 hardware it fails and we fall back to FP32.
                bool fastEnough = bf16Ms * 1.3 < fp32Ms;
                if (Environment.GetEnvironmentVariable("AIDOTNET_BF16_PROBE_DEBUG") == "1")
                    Console.WriteLine($"[BF16 Probe] FP32={fp32Ms:F2}ms, BF16={bf16Ms:F2}ms, 1.3× gate: {fastEnough} (need BF16 × 1.3 < FP32)");
                return fastEnough;
            }
            catch (DllNotFoundException)
            {
                // MklImports.dll absent — minimal-deploy scenario. BF16
                // itself (probed above via mkl_rt.3) is fine; honour the
                // user's explicit opt-in. Equivalent to mkl-bf16-force
                // without requiring the env var change.
                if (Environment.GetEnvironmentVariable("AIDOTNET_BF16_PROBE_DEBUG") == "1")
                    Console.WriteLine("[BF16 Probe] FP32 baseline unavailable (MklImports missing); enabling BF16 unconditionally.");
                return true;
            }
            catch (EntryPointNotFoundException)
            {
                if (Environment.GetEnvironmentVariable("AIDOTNET_BF16_PROBE_DEBUG") == "1")
                    Console.WriteLine("[BF16 Probe] FP32 entry point missing; enabling BF16 unconditionally.");
                return true;
            }
        }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
    }

    /// <summary>
    /// FP32 → BF16 conversion: take the top 16 bits of the float bit pattern.
    /// Uses round-to-nearest-even for better numerical accuracy than naive
    /// truncation. Matches the IEEE 754 BF16 (8-bit exp, 7-bit mantissa).
    /// NaN inputs are preserved as BF16 NaN (not silently rounded into ±∞)
    /// so numerical failure semantics carry through the BF16 path.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static unsafe ushort Fp32ToBf16(float x)
    {
        uint bits = *(uint*)&x;
        // NaN preservation: round-to-nearest-even on a bit pattern like
        // 0x7F800001 (smallest signalling NaN) carries into the exponent
        // and turns it into 0x7F80 — a BF16 infinity. Special-case NaNs
        // (abs(bits) > 0x7F800000) before rounding: keep the sign bit
        // and top-16 of the source, force a non-zero mantissa bit
        // (0x0040 — the BF16 quiet-NaN bit) so the result remains NaN.
        uint abs = bits & 0x7FFFFFFFu;
        if (abs > 0x7F800000u)
            return (ushort)((bits >> 16) | 0x0040u);

        // Round to nearest even: add half-ULP plus odd-bit detector.
        uint rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
        uint rounded = bits + rounding_bias;
        return (ushort)(rounded >> 16);
    }

    /// <summary>
    /// Bulk FP32 → BF16 conversion. Vectorized via AVX2 when available.
    /// At Issue #338 d=128 transformer shapes the scalar loop costs ~7ns
    /// per element; the AVX2 vector path drops this to ~0.5ns per element,
    /// which keeps the conversion overhead a fraction of the BF16 GEMM
    /// speedup instead of dominating it.
    /// </summary>
    internal static unsafe void Fp32ToBf16Bulk(float* src, ushort* dst, int count)
    {
#if NET5_0_OR_GREATER
        if (System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            int i = 0;
            // Process 8 floats per Vector256<float>, output 8 ushorts per
            // Vector128<ushort>. The conversion is "take top 16 bits with
            // round-to-nearest-even" — vectorized as:
            //   r = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
            // with a NaN-preservation overlay matching scalar Fp32ToBf16.
            var c7fff = System.Runtime.Intrinsics.Vector256.Create((int)0x7FFF);
            var c1 = System.Runtime.Intrinsics.Vector256.Create((int)1);
            var cAbsMask = System.Runtime.Intrinsics.Vector256.Create((int)0x7FFFFFFF);
            var cInfBits = System.Runtime.Intrinsics.Vector256.Create((int)0x7F800000);
            var cQuietNan = System.Runtime.Intrinsics.Vector256.Create((int)0x0040);
            for (; i + 8 <= count; i += 8)
            {
                var bits = System.Runtime.Intrinsics.X86.Avx.LoadVector256((int*)(src + i));
                var shifted = System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogical(bits, 16);
                var oddBit = System.Runtime.Intrinsics.X86.Avx2.And(shifted, c1);
                var bias = System.Runtime.Intrinsics.X86.Avx2.Add(c7fff, oddBit);
                var rounded = System.Runtime.Intrinsics.X86.Avx2.Add(bits, bias);
                var hi16 = System.Runtime.Intrinsics.X86.Avx2.ShiftRightLogical(rounded, 16);

                // NaN mask: abs(bits) > 0x7F800000. AVX2 CompareGreaterThan
                // is signed only — abs strips the sign so the comparison is
                // safe on non-negative results.
                var absBits = System.Runtime.Intrinsics.X86.Avx2.And(bits, cAbsMask);
                var nanMask = System.Runtime.Intrinsics.X86.Avx2.CompareGreaterThan(absBits, cInfBits);
                // BF16-NaN preserve: top-16 of source OR'd with the quiet
                // bit, narrowed below alongside hi16. Blend lane-wise.
                var nanHi16 = System.Runtime.Intrinsics.X86.Avx2.Or(shifted, cQuietNan);
                var merged = System.Runtime.Intrinsics.X86.Avx2.BlendVariable(hi16, nanHi16, nanMask);

                // Pack 8 ints (low 16 bits each) into 8 ushorts via shuffle.
                // PackUnsignedSaturate handles the narrowing.
                var lo = System.Runtime.Intrinsics.Vector256.GetLower(merged);
                var hi = System.Runtime.Intrinsics.Vector256.GetUpper(merged);
                var packed = System.Runtime.Intrinsics.X86.Sse41.PackUnsignedSaturate(lo, hi);
                System.Runtime.Intrinsics.X86.Sse2.Store(dst + i, packed);
            }
            for (; i < count; i++) dst[i] = Fp32ToBf16(src[i]);
            return;
        }
#endif
        for (int i = 0; i < count; i++)
            dst[i] = Fp32ToBf16(src[i]);
    }

    /// <summary>
    /// BF16 GEMM with FP32 accumulation: C[m,n] = alpha * A_bf16 @ B_bf16 + beta * C[m,n].
    /// Returns false when MKL BF16 is unavailable or env var not opted in;
    /// caller falls back to FP32 sgemm.
    /// </summary>
    internal static unsafe bool TryGemmBf16(int m, int n, int k,
        ushort[] a, int aOffset, int lda, bool transA,
        ushort[] b, int bOffset, int ldb, bool transB,
        float[] c, int cOffset, int ldc, float alpha = 1.0f, float beta = 0.0f)
    {
        if (!_preferMklBf16 || !_mklBf16Available.Value) return false;
        try
        {
            int cblasTA = transA ? 112 : CblasNoTrans;
            int cblasTB = transB ? 112 : CblasNoTrans;
            fixed (ushort* pa = &a[aOffset])
            fixed (ushort* pb = &b[bOffset])
            fixed (float* pc = &c[cOffset])
            {
                mkl_cblas_gemm_bf16bf16f32_ptr(CblasRowMajor, cblasTA, cblasTB,
                    m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
            }
            return true;
        }
        catch { return false; }
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
        // Force OpenBLAS to single-threaded mode so its multi-threaded GEMM
        // partial-sum reduction order doesn't reintroduce non-determinism
        // on top of the managed-side determinism the SimdGemm fallback
        // already guarantees. Restoring multi-threading on disable lets
        // perf go back up for callers that re-enter non-deterministic mode.
        TrySetOpenBlasThreads(deterministic ? 1 : 0);
        // Unify the switch (plan item 1.4): deterministic mode must ALSO make the
        // order-dependent reduction/accumulation kernels bit-reproducible — their
        // multi-threaded partial-sum combine is non-associative. Previously callers
        // had to set this separately, so "deterministic mode" left those reductions
        // non-deterministic. GEMM stays parallel because its M/N-tile splits are
        // marked deterministicSafe (see CpuParallelSettings.ParallelForOrSerial), so
        // this serializes only the genuinely order-dependent reductions.
        CpuParallelSettings.DeterministicReductions = deterministic;
    }

    /// <summary>True iff <c>AIDOTNET_USE_BLAS=1</c> is set AND libopenblas loaded successfully.</summary>
    internal static bool IsAvailable => _nativeAvailable.Value;

    /// <summary>
    /// Issue #444: the reason the native CPU BLAS probe failed, or <c>null</c> when it
    /// succeeded (or has not run yet). Forcing <see cref="IsAvailable"/> first guarantees
    /// the probe has run before this is read. Surfaced by
    /// <see cref="Engines.NativeLibraryDetector.OpenBlasLoadDiagnostic"/> so consumers can
    /// tell "package not installed" apart from a transitive-dependency load failure.
    /// </summary>
    private static volatile string? _loadError;

    /// <summary>
    /// Issue #444: human-readable reason the native CPU BLAS is unavailable, or <c>null</c>
    /// when it loaded successfully. Reading this forces the (lazy) load probe to run.
    /// </summary>
    internal static string? LoadError
    {
        get
        {
            _ = _nativeAvailable.Value; // ensure the probe (which sets _loadError) has run
            return _loadError;
        }
    }

    /// <summary>
    /// Issue #444: true when the active, successfully-loaded native CPU BLAS is OpenBLAS
    /// (the default provider — i.e. <c>AIDOTNET_BLAS_PROVIDER</c> is not set to an MKL
    /// variant). This is the signal the engine's GEMM dispatch actually gates on, so the
    /// diagnostic surface stays consistent with the path real matmul takes.
    /// </summary>
    internal static bool IsOpenBlasActive
    {
        get
        {
#if NET5_0_OR_GREATER
            return _nativeAvailable.Value && !_preferMkl;
#else
            // net471 has no DllImport resolver redirect, so the native library
            // bound by the libopenblas DllImports is always OpenBLAS.
            return _nativeAvailable.Value;
#endif
        }
    }

    /// <summary>
    /// Issue #444: true when the active, successfully-loaded native CPU BLAS is Intel MKL
    /// (<c>AIDOTNET_BLAS_PROVIDER=mkl*</c>, which redirects the libopenblas DllImports to
    /// MklImports via a resolver — NET5+ only).
    /// </summary>
    internal static bool IsMklActive
    {
        get
        {
#if NET5_0_OR_GREATER
            return _nativeAvailable.Value && _preferMkl;
#else
            return false;
#endif
        }
    }

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
    // Test-only shape instrumentation hook (Sub-issue A / #369).
    //
    // Set by the test infrastructure (ShapeInstrumenter) to record every
    // (m, n, k, transA, transB, dtype) tuple that flows through the public
    // Try* entry points. Always null in production builds — invocation is
    // a single null-check that the JIT inlines away. Hook fires BEFORE the
    // _nativeAvailable check so shapes are recorded even when libopenblas
    // is not loaded.
    // ────────────────────────────────────────────────────────────────────

    // Merge resolution (main #412 vs #402): kept main's 5-arg `Action<int,int,int,bool,bool>?
    // ShapeLogHook` declared at line 548 above + main's LogShape(m, n, k[, transA, transB])
    // wrapper that fires AFTER a successful BLAS call. Dropped #402's duplicate 6-arg field
    // that included Type — the existing ShapeInstrumenter consumer (Helpers/ShapeInstrumenter.cs)
    // uses the 5-arg shape and doesn't need the precision tag.

    // ────────────────────────────────────────────────────────────────────
    // Native-entry serialization gate.
    //
    // OpenBLAS (and any libopenblas-redirected MKL) corrupts its internal
    // per-thread working buffers when cblas_*gemm is ENTERED from more than one
    // managed thread at the same time — a hard native crash (access violation),
    // not merely wrong results. This is the native-path analog of the
    // StreamingWorkerPool concurrent-dispatch bug fixed in #492: there the shared
    // state was a managed worker pool; here it is OpenBLAS's own buffer table,
    // which we can't make reentrant from managed code. Reproduced by two
    // concurrent CpuEngine.TensorMatMul calls (the PaddedBuffer concurrency
    // stress guard) — single-threaded native is fine, 2+ simultaneous entries
    // segfault the process.
    //
    // Fix: at most one native GEMM is ever in flight, but HOW that's enforced is
    // mode-aware so concurrent inference isn't needlessly serialized:
    //
    //   • Deterministic mode (the default): results must be bit-reproducible, so every
    //     call uses the native kernel (the managed kernel differs bit-for-bit). The
    //     Try* entry points call the BLOCKING cblas_*gemm_ptr wrappers below; concurrent
    //     callers wait. When OpenBLAS runs multi-threaded this is also near-optimal — a
    //     single GEMM already fans across all cores, so serializing entry prevents
    //     oversubscription rather than adding it. (Deterministic mode forces OpenBLAS to
    //     1 thread, so concurrent GEMM there is genuinely serialized — an accepted
    //     correctness-over-speed tradeoff for that mode.)
    //
    //   • Non-deterministic mode: the Try* entry points take this gate via
    //     Monitor.TryEnter WITHOUT blocking; a caller that can't acquire it runs the
    //     concurrency-safe MANAGED kernel (BlasManaged, ~95% of OpenBLAS/core) instead
    //     of blocking — so N concurrent GEMMs stay parallel (1 native + N-1 managed).
    //     The native/managed kernel mix is acceptable precisely because this mode has
    //     opted out of bit-reproducibility. This mirrors the #492 StreamingWorkerPool
    //     TryEnter-then-fallback fix, on the native side.
    //
    // Either way, OpenBLAS's per-thread buffer table is never entered concurrently, so
    // the segfault can't occur. Managed-path callers (PreferManaged, native unavailable,
    // or a Try* returning false) never touch native state and are unaffected.
    //
    // NOTE: the α/β accumulate + raw paths (TryGemmWithBeta, TryGemmExBeta, SgemmRaw,
    // DgemmRaw) always use the BLOCKING wrappers regardless of mode — BlasManaged.Gemm
    // computes only C:=A·B (no α/β), so there's no one-line managed fallback for them.
    // They're conv/backward (training) paths, not the inference forward hot path.
    private static readonly object _nativeGemmGate = new();

    // Blocking-gated wrappers carrying the historical cblas_*gemm_ptr names. Used by the
    // deterministic-mode hot paths and the α/β + raw paths (which have no managed
    // fallback). The non-deterministic hot paths instead TryEnter _nativeGemmGate
    // themselves and call cblas_*gemm_native directly, so they never double-acquire.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static unsafe void cblas_sgemm_ptr(
        int order, int transA, int transB, int m, int n, int k,
        float alpha, float* a, int lda, float* b, int ldb, float beta, float* c, int ldc)
    {
        // Pointers are pinned by the caller's enclosing `fixed` block, which
        // encloses this call — the GC can't move the backing arrays for the
        // call's duration even though the lock is held inside this method.
        lock (_nativeGemmGate)
            cblas_sgemm_native(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static unsafe void cblas_dgemm_ptr(
        int order, int transA, int transB, int m, int n, int k,
        double alpha, double* a, int lda, double* b, int ldb, double beta, double* c, int ldc)
    {
        lock (_nativeGemmGate)
            cblas_dgemm_native(order, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    }

    // ────────────────────────────────────────────────────────────────────
    // Try* entry points. When BLAS is disabled (default), every call
    // returns false and callers fall through to SimdGemm. When enabled
    // via AIDOTNET_USE_BLAS=1 and libopenblas loads, calls dispatch to
    // cblas_{s,d}gemm with standard row-major layout.
    // ────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Single routing decision shared by every <c>TryGemm*</c> entry point:
    /// should this GEMM dispatch to the managed kernel (<see cref="Engines.BlasManaged.BlasManaged.Gemm{T}"/>)
    /// rather than native cblas? Three reasons, in priority order:
    /// <list type="number">
    ///   <item><see cref="Engines.BlasManaged.BlasManaged.PreferManaged"/> — supply-chain
    ///   force-managed (no native attack surface).</item>
    ///   <item><see cref="IsDeterministicMode"/> — the managed GEMM is parallel AND
    ///   bit-reproducible across thread counts (M/N/2D disjoint-write splits; K-axis
    ///   gated off — see <c>DeterministicParallelGemmContractTests</c>), whereas native
    ///   multi-threaded reduction order is not reproducible and native deterministic
    ///   mode runs single-threaded. Routing deterministic mode to managed gives
    ///   reproducibility WITHOUT serializing — a capability PyTorch's deterministic
    ///   mode lacks. Deterministic mode must NOT use the timing-based autotune below
    ///   (its winner varies with measurement noise → non-reproducible kernel choice).</item>
    ///   <item><see cref="Engines.BlasManaged.BlasManaged.AutotuneRouting"/> — in
    ///   non-deterministic mode, per-shape best-of-managed-vs-native (results need not
    ///   be bit-reproducible, so a measured kernel choice is fine).</item>
    /// </list>
    /// </summary>
    internal static bool ShouldRouteManaged(int m, int n, int k, bool transA, bool transB, Type dtype)
    {
        if (Engines.BlasManaged.BlasManaged.PreferManaged) return true;
        if (IsDeterministicMode) return true;
        return Engines.BlasManaged.BlasManaged.AutotuneRouting
            && !Engines.BlasManaged.PrefersManagedCache.BypassAutotune
            && Engines.BlasManaged.PrefersManagedCache.PrefersManaged(m, n, k, transA, transB, dtype);
    }

    internal static bool TryGemm(int m, int n, int k,
        float[] a, int aOffset, int lda,
        float[] b, int bOffset, int ldb,
        float[] c, int cOffset, int ldc)
    {
        if (ShouldRouteManaged(m, n, k, false, false, typeof(float)))
        {
            Engines.BlasManaged.BlasManaged.Gemm<float>(
                new ReadOnlySpan<float>(a, aOffset, a.Length - aOffset), lda, false,
                new ReadOnlySpan<float>(b, bOffset, b.Length - bOffset), ldb, false,
                new Span<float>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
        if (!_nativeAvailable.Value) return false;
        // Concurrent native entry crashes OpenBLAS (see _nativeGemmGate). Two modes:
        //   • Deterministic mode: results must be bit-reproducible, so EVERY call uses
        //     the native kernel (managed differs bit-for-bit). Serialize via the
        //     blocking cblas_*gemm_ptr wrapper — concurrent callers wait, never race.
        //   • Non-deterministic mode: never block. TryEnter; if another thread holds
        //     the gate, run the concurrency-safe managed kernel so concurrent inference
        //     stays parallel (the #492 fallback pattern). Kernel-mixing is acceptable
        //     here precisely because the caller opted out of bit-reproducibility.
        bool deterministic = IsDeterministicMode;
        if (!deterministic && !System.Threading.Monitor.TryEnter(_nativeGemmGate))
        {
            Engines.BlasManaged.BlasManaged.Gemm<float>(
                new ReadOnlySpan<float>(a, aOffset, a.Length - aOffset), lda, false,
                new ReadOnlySpan<float>(b, bOffset, b.Length - bOffset), ldb, false,
                new Span<float>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
        try
        {
            unsafe
            {
                fixed (float* pa = &a[aOffset])
                fixed (float* pb = &b[bOffset])
                fixed (float* pc = &c[cOffset])
                {
                    if (deterministic)
                        cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                    else
                        cblas_sgemm_native(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                }
            }
            LogShape(m, n, k);
            return true;
        }
        catch { return false; }
        finally { if (!deterministic) System.Threading.Monitor.Exit(_nativeGemmGate); }
    }

    internal static bool TryGemm(int m, int n, int k,
        double[] a, int aOffset, int lda,
        double[] b, int bOffset, int ldb,
        double[] c, int cOffset, int ldc)
    {
        if (ShouldRouteManaged(m, n, k, false, false, typeof(double)))
        {
            Engines.BlasManaged.BlasManaged.Gemm<double>(
                new ReadOnlySpan<double>(a, aOffset, a.Length - aOffset), lda, false,
                new ReadOnlySpan<double>(b, bOffset, b.Length - bOffset), ldb, false,
                new Span<double>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
        if (!_nativeAvailable.Value) return false;
        bool deterministic = IsDeterministicMode; // see TryGemm(float[]) note
        if (!deterministic && !System.Threading.Monitor.TryEnter(_nativeGemmGate))
        {
            Engines.BlasManaged.BlasManaged.Gemm<double>(
                new ReadOnlySpan<double>(a, aOffset, a.Length - aOffset), lda, false,
                new ReadOnlySpan<double>(b, bOffset, b.Length - bOffset), ldb, false,
                new Span<double>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
        try
        {
            unsafe
            {
                fixed (double* pa = &a[aOffset])
                fixed (double* pb = &b[bOffset])
                fixed (double* pc = &c[cOffset])
                {
                    if (deterministic)
                        cblas_dgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                    else
                        cblas_dgemm_native(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                }
            }
            LogShape(m, n, k);
            return true;
        }
        catch { return false; }
        finally { if (!deterministic) System.Threading.Monitor.Exit(_nativeGemmGate); }
    }

    internal static bool TryGemm(int m, int n, int k,
        ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc)
    {
        if (ShouldRouteManaged(m, n, k, false, false, typeof(float)))
        {
            Engines.BlasManaged.BlasManaged.Gemm<float>(a, lda, false, b, ldb, false, c, ldc, m, n, k);
            return true;
        }
        if (!_nativeAvailable.Value) return false;
        bool deterministic = IsDeterministicMode; // see TryGemm(float[]) note
        if (!deterministic && !System.Threading.Monitor.TryEnter(_nativeGemmGate))
        {
            Engines.BlasManaged.BlasManaged.Gemm<float>(a, lda, false, b, ldb, false, c, ldc, m, n, k);
            return true;
        }
        try
        {
            unsafe
            {
                fixed (float* pa = a)
                fixed (float* pb = b)
                fixed (float* pc = c)
                {
                    if (deterministic)
                        cblas_sgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                    else
                        cblas_sgemm_native(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                }
            }
            LogShape(m, n, k);
            return true;
        }
        catch { return false; }
        finally { if (!deterministic) System.Threading.Monitor.Exit(_nativeGemmGate); }
    }

    internal static bool TryGemm(int m, int n, int k,
        ReadOnlySpan<double> a, int lda, ReadOnlySpan<double> b, int ldb, Span<double> c, int ldc)
    {
        if (ShouldRouteManaged(m, n, k, false, false, typeof(double)))
        {
            Engines.BlasManaged.BlasManaged.Gemm<double>(a, lda, false, b, ldb, false, c, ldc, m, n, k);
            return true;
        }
        if (!_nativeAvailable.Value) return false;
        bool deterministic = IsDeterministicMode; // see TryGemm(float[]) note
        if (!deterministic && !System.Threading.Monitor.TryEnter(_nativeGemmGate))
        {
            Engines.BlasManaged.BlasManaged.Gemm<double>(a, lda, false, b, ldb, false, c, ldc, m, n, k);
            return true;
        }
        try
        {
            unsafe
            {
                fixed (double* pa = a)
                fixed (double* pb = b)
                fixed (double* pc = c)
                {
                    if (deterministic)
                        cblas_dgemm_ptr(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                    else
                        cblas_dgemm_native(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                }
            }
            LogShape(m, n, k);
            return true;
        }
        catch { return false; }
        finally { if (!deterministic) System.Threading.Monitor.Exit(_nativeGemmGate); }
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
            LogShape(m, n, k);
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
            LogShape(m, n, k);
            return true;
        }
        catch { return false; }
    }

    internal static bool TryGemmEx(int m, int n, int k,
        float[] a, int aOffset, int lda, bool transA,
        float[] b, int bOffset, int ldb, bool transB,
        float[] c, int cOffset, int ldc)
    {
        // PreferManaged / deterministic-mode / non-deterministic autotune best-of —
        // see ShouldRouteManaged. (BypassAutotune, flipped during the autotune probe,
        // is honored inside ShouldRouteManaged so the measurement still hits native.)
        if (ShouldRouteManaged(m, n, k, transA, transB, typeof(float)))
        {
            Engines.BlasManaged.BlasManaged.Gemm<float>(
                new ReadOnlySpan<float>(a, aOffset, a.Length - aOffset), lda, transA,
                new ReadOnlySpan<float>(b, bOffset, b.Length - bOffset), ldb, transB,
                new Span<float>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
        // Phase G.1: when AIDOTNET_BLAS_PROVIDER=mkl, the static-ctor
        // resolver redirects libopenblas → MklImports; the call below
        // then dispatches into MKL's cblas_sgemm transparently.
        if (!_nativeAvailable.Value) return false;
        bool deterministic = IsDeterministicMode; // see TryGemm(float[]) note
        if (!deterministic && !System.Threading.Monitor.TryEnter(_nativeGemmGate))
        {
            Engines.BlasManaged.BlasManaged.Gemm<float>(
                new ReadOnlySpan<float>(a, aOffset, a.Length - aOffset), lda, transA,
                new ReadOnlySpan<float>(b, bOffset, b.Length - bOffset), ldb, transB,
                new Span<float>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
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
                    if (deterministic)
                        cblas_sgemm_ptr(CblasRowMajor, cblasTA, cblasTB,
                            m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                    else
                        cblas_sgemm_native(CblasRowMajor, cblasTA, cblasTB,
                            m, n, k, 1.0f, pa, lda, pb, ldb, 0.0f, pc, ldc);
                }
            }
            LogShape(m, n, k, transA, transB);
            return true;
        }
        catch { return false; }
        finally { if (!deterministic) System.Threading.Monitor.Exit(_nativeGemmGate); }
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
        if (ShouldRouteManaged(m, n, k, transA, transB, typeof(double)))
        {
            Engines.BlasManaged.BlasManaged.Gemm<double>(
                new ReadOnlySpan<double>(a, aOffset, a.Length - aOffset), lda, transA,
                new ReadOnlySpan<double>(b, bOffset, b.Length - bOffset), ldb, transB,
                new Span<double>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
        if (!_nativeAvailable.Value) return false;
        bool deterministic = IsDeterministicMode; // see TryGemm(float[]) note
        if (!deterministic && !System.Threading.Monitor.TryEnter(_nativeGemmGate))
        {
            Engines.BlasManaged.BlasManaged.Gemm<double>(
                new ReadOnlySpan<double>(a, aOffset, a.Length - aOffset), lda, transA,
                new ReadOnlySpan<double>(b, bOffset, b.Length - bOffset), ldb, transB,
                new Span<double>(c, cOffset, c.Length - cOffset), ldc,
                m, n, k);
            return true;
        }
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
                    if (deterministic)
                        cblas_dgemm_ptr(CblasRowMajor, cblasTA, cblasTB,
                            m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                    else
                        cblas_dgemm_native(CblasRowMajor, cblasTA, cblasTB,
                            m, n, k, 1.0, pa, lda, pb, ldb, 0.0, pc, ldc);
                }
            }
            LogShape(m, n, k, transA, transB);
            return true;
        }
        catch { return false; }
        finally { if (!deterministic) System.Threading.Monitor.Exit(_nativeGemmGate); }
    }

    /// <summary>
    /// Double-precision GEMM with explicit α/β scalars: C = αAB + βC.
    /// Used by the 1×1 conv backward fast paths to write directly into the
    /// pre-allocated gradient destination with β=1 (accumulating mode)
    /// without an intermediate buffer + sum-loop.
    /// </summary>
    internal static bool TryGemmExBeta(int m, int n, int k,
        double[] a, int aOffset, int lda, bool transA,
        double[] b, int bOffset, int ldb, bool transB,
        double[] c, int cOffset, int ldc,
        double alpha, double beta)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            int cblasTA = transA ? 112 : CblasNoTrans;
            int cblasTB = transB ? 112 : CblasNoTrans;
            unsafe
            {
                fixed (double* pa = &a[aOffset])
                fixed (double* pb = &b[bOffset])
                fixed (double* pc = &c[cOffset])
                {
                    cblas_dgemm_ptr(CblasRowMajor, cblasTA, cblasTB,
                        m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
                }
            }
            LogShape(m, n, k, transA, transB);
            return true;
        }
        catch { return false; }
    }

    /// <summary>
    /// Float counterpart of <see cref="TryGemmExBeta(int, int, int, double[], int, int, bool, double[], int, int, bool, double[], int, int, double, double)"/>.
    /// </summary>
    internal static bool TryGemmExBeta(int m, int n, int k,
        float[] a, int aOffset, int lda, bool transA,
        float[] b, int bOffset, int ldb, bool transB,
        float[] c, int cOffset, int ldc,
        float alpha, float beta)
    {
        if (!_nativeAvailable.Value) return false;
        try
        {
            int cblasTA = transA ? 112 : CblasNoTrans;
            int cblasTB = transB ? 112 : CblasNoTrans;
            unsafe
            {
                fixed (float* pa = &a[aOffset])
                fixed (float* pb = &b[bOffset])
                fixed (float* pc = &c[cOffset])
                {
                    cblas_sgemm_ptr(CblasRowMajor, cblasTA, cblasTB,
                        m, n, k, alpha, pa, lda, pb, ldb, beta, pc, ldc);
                }
            }
            LogShape(m, n, k, transA, transB);
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
