using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// High-level wrapper over cuBLASLt's fused-epilogue matmul. Exposes
/// <c>D = alpha × A @ B + beta × C</c> with optional fused bias + ReLU
/// / GELU / Tanh-GELU in a single kernel launch.
///
/// <para><b>Why this matters:</b> transformer FFN blocks and attention
/// projections are (matmul + bias + activation) sequences — three
/// kernels as raw cuBLAS + elementwise ops. cuBLASLt fuses them into
/// one launch with one set of memory round-trips, which is frequently
/// a 1.5-2× end-to-end inference speedup on H100s.</para>
///
/// <para>Descriptor lifetime is per-call for simplicity; a future
/// optimization caches descriptors keyed by (dtype, layout, epilogue)
/// so a repeated call set reuses the handles.</para>
/// </summary>
public sealed class CuBlasLtMatmul : IDisposable
{
    private IntPtr _handle;
    private bool _disposed;

    /// <summary>True iff libcublasLt can be loaded at runtime.</summary>
    public static bool IsAvailable
    {
        get
        {
            try
            {
                CuBlasLtNative.cublasLtCreate(out var h);
                CuBlasLtNative.cublasLtDestroy(h);
                return true;
            }
            catch { return false; }
        }
    }

    public CuBlasLtMatmul()
    {
        var status = CuBlasLtNative.cublasLtCreate(out _handle);
        if (status != CublasStatus.Success)
            throw new InvalidOperationException($"cublasLtCreate failed: {status}.");
    }

    /// <summary>
    /// Run <c>D = alpha × (op(A) @ op(B)) + beta × C</c> with fused
    /// epilogue. All pointers are device pointers; caller owns
    /// allocation + stream.
    /// </summary>
    /// <param name="aDev">Device pointer to A; shape (m, k) row-major
    /// when <paramref name="transA"/>=false.</param>
    /// <param name="bDev">Device pointer to B; shape (k, n) row-major
    /// when <paramref name="transB"/>=false.</param>
    /// <param name="cDev">Device pointer to C (often null → beta must be 0).</param>
    /// <param name="dDev">Device pointer to D (output).</param>
    /// <param name="biasDev">Device pointer to rank-1 bias of length n;
    /// null disables bias fusion (must match epilogue choice).</param>
    /// <param name="epilogue">Fused post-matmul op.</param>
    public void MatmulFused(
        IntPtr aDev, int m, int k, bool transA,
        IntPtr bDev, int n, bool transB,
        IntPtr cDev, IntPtr dDev,
        IntPtr biasDev,
        CublasLtEpilogue epilogue,
        float alpha = 1f, float beta = 0f,
        IntPtr workspace = default, ulong workspaceSizeInBytes = 0,
        IntPtr stream = default,
        CublasDataType dtype = CublasDataType.Float32,
        CublasComputeType computeType = CublasComputeType.Float32,
        CublasDataType? outputDtype = null)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuBlasLtMatmul));

        IntPtr opDesc = IntPtr.Zero, aDesc = IntPtr.Zero, bDesc = IntPtr.Zero, cDesc = IntPtr.Zero, dDesc = IntPtr.Zero;
        try
        {
            Check(CuBlasLtNative.cublasLtMatmulDescCreate(out opDesc, computeType, CublasDataType.Float32), "DescCreate");

            // Set transpose flags + epilogue + bias pointer if applicable.
            int tA = transA ? 1 : 0;
            int tB = transB ? 1 : 0;
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransA, ref tA, sizeof(int));
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransB, ref tB, sizeof(int));
            int epi = (int)epilogue;
            SetAttr(opDesc, CublasLtMatmulDescAttributes.Epilogue, ref epi, sizeof(int));
            if (biasDev != IntPtr.Zero)
            {
                long biasPtrRaw = biasDev.ToInt64();
                SetAttr(opDesc, CublasLtMatmulDescAttributes.EpilogueBiasPointer, ref biasPtrRaw, sizeof(long));
            }

            // Layout descriptors — cuBLAS is column-major natively,
            // so for row-major C# data we pass ld = inner stride and
            // flip the transposes downstream. Callers should arrange
            // data accordingly; we document the convention and leave
            // the choice to them rather than silently reinterpreting.
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out aDesc, dtype, (ulong)m, (ulong)k, transA ? k : m), "A layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out bDesc, dtype, (ulong)k, (ulong)n, transB ? n : k), "B layout");
            CublasDataType resultType = outputDtype ?? dtype;
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out cDesc, resultType, (ulong)m, (ulong)n, m), "C layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(out dDesc, resultType, (ulong)m, (ulong)n, m), "D layout");

            Check(CuBlasLtNative.cublasLtMatmul(
                _handle, opDesc,
                ref alpha,
                aDev, aDesc, bDev, bDesc,
                ref beta,
                cDev == IntPtr.Zero ? dDev : cDev, cDesc,
                dDev, dDesc,
                IntPtr.Zero, // auto-select algo
                workspace, workspaceSizeInBytes,
                stream), "Matmul");
        }
        finally
        {
            if (dDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(dDesc);
            if (cDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(cDesc);
            if (bDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(bDesc);
            if (aDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(aDesc);
            if (opDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatmulDescDestroy(opDesc);
        }
    }

    /// <summary>
    /// Column-major <c>D[m,n] = alpha · op(A) · op(B) + beta · D</c> in FP32, with the operands described by their
    /// stored shapes (cuBLAS <c>cublasSgemm</c> semantics: A is stored <c>[m,k]</c>, or <c>[k,m]</c> when
    /// <paramref name="transA"/>; B is stored <c>[k,n]</c>, or <c>[n,k]</c> when <paramref name="transB"/>).
    /// The algorithm is cuBLASLt's heuristic choice within <paramref name="workspaceSizeInBytes"/> of the caller's
    /// workspace, so the call never allocates -- unlike cuBLAS's split-K path, which under CUDA graph capture takes
    /// its workspace from the CUBLAS_WORKSPACE_CONFIG pool (graph allocation nodes) even with a user workspace set.
    /// Split-K algorithms that reduce with atomics (<see cref="CublasLtReductionScheme.InPlace"/>) are excluded, so
    /// a given problem always runs the same algorithm and gives the same bits. The choice is cached per problem.
    /// Returns cuBLASLt's status; the caller falls back on anything but success.
    /// </summary>
    internal unsafe CublasStatus TryMatmulColumnMajor(
        bool transA, bool transB, int m, int n, int k,
        IntPtr a, int lda, IntPtr b, int ldb, IntPtr d, int ldd,
        float alpha, float beta, CublasComputeType computeType,
        IntPtr workspace, ulong workspaceSizeInBytes, IntPtr stream)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuBlasLtMatmul));
        IntPtr opDesc = IntPtr.Zero, aDesc = IntPtr.Zero, bDesc = IntPtr.Zero, dDesc = IntPtr.Zero;
        try
        {
            var status = CuBlasLtNative.cublasLtMatmulDescCreate(out opDesc, computeType, CublasDataType.Float32);
            if (status != CublasStatus.Success) return status;
            int tA = transA ? 1 : 0, tB = transB ? 1 : 0;
            if ((status = SetAttrStatus(opDesc, CublasLtMatmulDescAttributes.TransA, ref tA)) != CublasStatus.Success
                || (status = SetAttrStatus(opDesc, CublasLtMatmulDescAttributes.TransB, ref tB)) != CublasStatus.Success)
                return status;
            if ((status = CuBlasLtNative.cublasLtMatrixLayoutCreate(out aDesc, CublasDataType.Float32,
                    (ulong)(transA ? k : m), (ulong)(transA ? m : k), lda)) != CublasStatus.Success
                || (status = CuBlasLtNative.cublasLtMatrixLayoutCreate(out bDesc, CublasDataType.Float32,
                    (ulong)(transB ? n : k), (ulong)(transB ? k : n), ldb)) != CublasStatus.Success
                || (status = CuBlasLtNative.cublasLtMatrixLayoutCreate(out dDesc, CublasDataType.Float32,
                    (ulong)m, (ulong)n, ldd)) != CublasStatus.Success)
                return status;
            var key = new LtProblem(transA, transB, m, n, k, lda, ldb, ldd, computeType, workspaceSizeInBytes,
                GpuDeterminism.IsActive);
            // The lock covers the lookup and the insert only. Measuring a new problem times up to eight candidates on
            // the device, tens of milliseconds on a large GEMM, and holding the lock across it stalled every other
            // thread's GEMM -- including those whose problems were already cached. Two threads that meet the same new
            // problem may both measure it; the first insert wins and the other uses it, so each problem still has ONE
            // fixed choice per process.
            CublasLtMatmulHeuristicResult chosen;
            bool cached;
            lock (_algoCache) cached = _algoCache.TryGetValue(key, out chosen);
            if (!cached)
            {
                var problem = new MeasuredProblem(a, b, ldd, n, alpha, beta, workspace, stream);
                if ((status = ChooseAlgorithm(opDesc, aDesc, bDesc, dDesc, workspaceSizeInBytes, key, problem,
                        out CublasLtMatmulHeuristicResult measured)) != CublasStatus.Success)
                    return status;
                lock (_algoCache)
                {
                    if (!_algoCache.TryGetValue(key, out chosen))
                    {
                        _algoCache[key] = measured;
                        chosen = measured;
                    }
                }
            }
            return CuBlasLtNative.cublasLtMatmul(_handle, opDesc, ref alpha, a, aDesc, b, bDesc, ref beta,
                d, dDesc, d, dDesc, (IntPtr)chosen.Algo, workspace, workspaceSizeInBytes, stream);
        }
        finally
        {
            if (dDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(dDesc);
            if (bDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(bDesc);
            if (aDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(aDesc);
            if (opDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatmulDescDestroy(opDesc);
        }
    }

    private static unsafe CublasStatus SetAttrStatus(IntPtr desc, CublasLtMatmulDescAttributes attr, ref int value)
    {
        fixed (int* p = &value)
            return CuBlasLtNative.cublasLtMatmulDescSetAttribute(desc, attr, (IntPtr)p, sizeof(int));
    }

    // Deterministic is part of the key because the choice can differ by mode: see ChooseAlgorithm's fallback.
    private readonly record struct LtProblem(bool TransA, bool TransB, int M, int N, int K, int Lda, int Ldb, int Ldd,
        CublasComputeType ComputeType, ulong WorkspaceBytes, bool Deterministic);

    // The algorithm chosen for each problem: choosing is host (and, when measured, device) work, and one fixed choice
    // per problem is what keeps a process's results reproducible.
    private readonly Dictionary<LtProblem, CublasLtMatmulHeuristicResult> _algoCache = new();

    /// <summary>The operands of the call that first meets a problem, for timing the heuristic's candidates on it.</summary>
    private readonly record struct MeasuredProblem(IntPtr A, IntPtr B, int Ldd, int N, float Alpha, float Beta,
        IntPtr Workspace, IntPtr Stream);

    // How many of cuBLASLt's ranked candidates are timed. Its first choice is often a split-K algorithm whose separate
    // reduction kernel costs more than the split saves on small problems: on the parity MLP (batch 128) every step ran
    // eight splitKreduce kernels (~80 us of a ~460 us step) where cuBLAS's own heuristic picks single-kernel
    // "sliced" algorithms.
    private const int MeasuredCandidates = 8;

    /// <summary>
    /// The device this wrapper runs on (name and compute capability), set by the owning backend. With it, a problem's
    /// algorithm is chosen by timing the heuristic's candidates once and recording the winner's rank on disk
    /// (<see cref="Helpers.Autotune.AutotuneCache"/>); without it, the heuristic's first choice is taken.
    /// </summary>
    internal string? DeviceIdentity { get; set; }

    private static readonly bool s_autotune = Environment.GetEnvironmentVariable("AIDOTNET_CUBLASLT_AUTOTUNE") != "0";

    private static bool IsCapturing(IntPtr stream)
        => CudaNativeBindings.cuStreamIsCapturing(stream, out int capturing) != CudaResult.Success || capturing != 0;

    /// <summary>
    /// The autotune cache key of a problem. The winner is recorded as its RANK in the heuristic's list, not as the
    /// opaque algorithm, which is only meaningful to the cuBLASLt build that produced it; the key therefore carries
    /// the cuBLASLt version and the device, under which the heuristic's list is fixed. Recording the rank is what
    /// keeps GpuDeterminism: every later process (and every backend in this one) runs the same algorithm instead of
    /// re-timing and perhaps choosing differently.
    /// </summary>
    private (Helpers.Autotune.KernelId Id, Helpers.Autotune.ShapeProfile Shape) AutotuneKey(in LtProblem p)
    {
        ulong version = CuBlasLtNative.cublasLtGetVersion().ToUInt64();
        var id = new Helpers.Autotune.KernelId("cublaslt-sgemm", $"{DeviceIdentity}-lt{version}-{p.ComputeType}");
        var shape = new Helpers.Autotune.ShapeProfile(p.TransA ? 1 : 0, p.TransB ? 1 : 0, p.M, p.N, p.K, p.Lda, p.Ldb, p.Ldd,
            checked((int)(p.WorkspaceBytes >> 10)));
        return (id, shape);
    }

    private unsafe CublasStatus ChooseAlgorithm(IntPtr opDesc, IntPtr aDesc, IntPtr bDesc, IntPtr dDesc,
        ulong workspaceSizeInBytes, in LtProblem key, MeasuredProblem problem, out CublasLtMatmulHeuristicResult chosen)
    {
        chosen = default;
        var status = CuBlasLtNative.cublasLtMatmulPreferenceCreate(out var preference);
        if (status != CublasStatus.Success) return status;
        try
        {
            ulong maxWorkspace = workspaceSizeInBytes;
            uint schemes = (uint)(CublasLtReductionScheme.ComputeType | CublasLtReductionScheme.OutputType);
            if ((status = CuBlasLtNative.cublasLtMatmulPreferenceSetAttribute(preference,
                    CublasLtMatmulPreferenceAttributes.MaxWorkspaceBytes, (IntPtr)(&maxWorkspace), sizeof(ulong)))
                    != CublasStatus.Success
                || (status = CuBlasLtNative.cublasLtMatmulPreferenceSetAttribute(preference,
                    CublasLtMatmulPreferenceAttributes.ReductionSchemeMask, (IntPtr)(&schemes), sizeof(uint)))
                    != CublasStatus.Success)
                return status;
            bool tune = s_autotune && DeviceIdentity is not null;
            int requested = tune ? MeasuredCandidates : 1;
            var results = stackalloc CublasLtMatmulHeuristicResult[requested];
            status = CuBlasLtNative.cublasLtMatmulAlgoGetHeuristic(_handle, opDesc, aDesc, bDesc, dDesc, dDesc,
                preference, requested, results, out int found);
            if (status != CublasStatus.Success) return status;
            if (found < 1 || results[0].State != CublasStatus.Success) return CublasStatus.NotSupported;
            chosen = results[0];
            if (!tune || found < 2) return CublasStatus.Success;

            var (id, shape) = AutotuneKey(key);
            if (Helpers.Autotune.AutotuneCache.Lookup(id, shape) is { } recorded
                && recorded.Variant.StartsWith("rank:", StringComparison.Ordinal)
                && int.TryParse(recorded.Variant.Substring(5), System.Globalization.NumberStyles.None,
                    System.Globalization.CultureInfo.InvariantCulture, out int rank)
                && rank < found && results[rank].State == CublasStatus.Success)
            {
                chosen = results[rank];
                return CublasStatus.Success;
            }

            // A capture cannot time; the heuristic's first choice runs and nothing is recorded, so the next
            // uncaptured call of this problem in a later process still measures.
            if (IsCapturing(problem.Stream)) return CublasStatus.Success;
            int best = Fastest(opDesc, aDesc, bDesc, dDesc, results, found, problem, workspaceSizeInBytes, out float bestMs);
            chosen = results[best];
            bool stored = bestMs < float.MaxValue
                && Helpers.Autotune.AutotuneCache.TryStore(id, shape,
                    new Helpers.Autotune.KernelChoice { Variant = "rank:" + best.ToString(System.Globalization.CultureInfo.InvariantCulture), MeasuredTimeMs = bestMs });
            // Determinism holds because the measured rank is RECORDED and every later process runs it. A rank that
            // could not be recorded (an unwritable cache) would be re-measured -- and perhaps chosen differently --
            // by the next process, so under GpuDeterminism the heuristic's fixed first choice runs instead. Tuning is
            // not simply disabled under determinism: determinism is on by default, so that would disable it for all.
            if (!stored && key.Deterministic) chosen = results[0];
            return CublasStatus.Success;
        }
        finally
        {
            CuBlasLtNative.cublasLtMatmulPreferenceDestroy(preference);
        }
    }

    /// <summary>
    /// Index of the fastest of the <paramref name="count"/> candidates on this problem: each runs twice to warm, then
    /// three batches of four calls are timed with events and its best batch counts. The output goes to a scratch
    /// buffer, so the caller's D (which beta may accumulate into) is untouched. A candidate that fails is skipped; if
    /// timing itself is unavailable the heuristic's first choice stands and <paramref name="bestMs"/> stays MaxValue.
    /// </summary>
    private unsafe int Fastest(IntPtr opDesc, IntPtr aDesc, IntPtr bDesc, IntPtr dDesc,
        CublasLtMatmulHeuristicResult* candidates, int count, MeasuredProblem p, ulong workspaceSizeInBytes, out float bestMs)
    {
        const int Warmup = 2, Batches = 3, CallsPerBatch = 4;
        ulong bytes = (ulong)p.Ldd * (ulong)p.N * sizeof(float);
        IntPtr scratch = IntPtr.Zero, start = IntPtr.Zero, stop = IntPtr.Zero;
        bestMs = float.MaxValue;
        try
        {
            if (CudaNativeBindings.cuMemAlloc(out scratch, bytes) != CudaResult.Success) return 0;
            if (CudaNativeBindings.cuEventCreate(out start, 0) != CudaResult.Success
                || CudaNativeBindings.cuEventCreate(out stop, 0) != CudaResult.Success)
                return 0;
            float alpha = p.Alpha, beta = p.Beta;
            int best = 0;
            for (int c = 0; c < count; c++)
            {
                if (candidates[c].State != CublasStatus.Success) continue;
                IntPtr algo = (IntPtr)candidates[c].Algo;
                bool ok = true;
                for (int w = 0; w < Warmup && ok; w++)
                    ok = CuBlasLtNative.cublasLtMatmul(_handle, opDesc, ref alpha, p.A, aDesc, p.B, bDesc, ref beta,
                        scratch, dDesc, scratch, dDesc, algo, p.Workspace, workspaceSizeInBytes, p.Stream) == CublasStatus.Success;
                if (!ok) continue;
                float candidateMs = float.MaxValue;
                for (int batch = 0; batch < Batches && ok; batch++)
                {
                    CudaNativeBindings.cuEventRecord(start, p.Stream);
                    for (int i = 0; i < CallsPerBatch && ok; i++)
                        ok = CuBlasLtNative.cublasLtMatmul(_handle, opDesc, ref alpha, p.A, aDesc, p.B, bDesc, ref beta,
                            scratch, dDesc, scratch, dDesc, algo, p.Workspace, workspaceSizeInBytes, p.Stream) == CublasStatus.Success;
                    CudaNativeBindings.cuEventRecord(stop, p.Stream);
                    if (!ok || CudaNativeBindings.cuEventSynchronize(stop) != CudaResult.Success
                        || CudaNativeBindings.cuEventElapsedTime(out float ms, start, stop) != CudaResult.Success)
                    {
                        ok = false;
                        break;
                    }
                    candidateMs = Math.Min(candidateMs, ms);
                }
                if (ok && candidateMs < bestMs)
                {
                    bestMs = candidateMs;
                    best = c;
                }
            }
            return best;
        }
        finally
        {
            // The timed calls must finish before their scratch output is released.
            if (stop != IntPtr.Zero) CudaNativeBindings.cuEventSynchronize(stop);
            if (start != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(start);
            if (stop != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(stop);
            if (scratch != IntPtr.Zero) CudaNativeBindings.cuMemFree(scratch);
        }
    }

    /// <summary>
    /// Resident signed-INT8 GEMM with exact INT32 accumulation/output. The
    /// caller supplies already-packed device operands; scaling and nonlinear
    /// epilogues are intentionally separate because cuBLASLt's INT32 compute
    /// contract does not accept the FP32 per-channel scale ABI used by W8A8.
    /// Operands use standard column-major packing. Before transposition, A is
    /// <c>[m,k]</c> (or <c>[k,m]</c> when transposed), B is <c>[k,n]</c> (or
    /// <c>[n,k]</c> when transposed), and C/D are <c>[m,n]</c>. Device pointers,
    /// the reduction dimension, and the leading dimensions must satisfy the
    /// four-byte INT8 dot-product alignment contract.
    /// </summary>
    public void MatmulInt8ToInt32(
        IntPtr aDev, int m, int k, bool transA,
        IntPtr bDev, int n, bool transB,
        IntPtr dDev,
        IntPtr stream = default)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CuBlasLtMatmul));
        ValidateInt8LayoutArguments(aDev, m, k, transA, bDev, n, transB, dDev);

        ulong aRows = checked((ulong)(transA ? k : m));
        ulong aColumns = checked((ulong)(transA ? m : k));
        long aLeadingDimension = checked((long)aRows);
        ulong bRows = checked((ulong)(transB ? n : k));
        ulong bColumns = checked((ulong)(transB ? k : n));
        long bLeadingDimension = checked((long)bRows);

        IntPtr opDesc = IntPtr.Zero, aDesc = IntPtr.Zero, bDesc = IntPtr.Zero;
        IntPtr cDesc = IntPtr.Zero, dDesc = IntPtr.Zero;
        try
        {
            Check(CuBlasLtNative.cublasLtMatmulDescCreate(
                out opDesc, CublasComputeType.Int32, CublasDataType.Int32), "INT8 DescCreate");
            int tA = transA ? 1 : 0;
            int tB = transB ? 1 : 0;
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransA, ref tA, sizeof(int));
            SetAttr(opDesc, CublasLtMatmulDescAttributes.TransB, ref tB, sizeof(int));
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out aDesc, CublasDataType.Int8, aRows, aColumns,
                aLeadingDimension), "INT8 A layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out bDesc, CublasDataType.Int8, bRows, bColumns,
                bLeadingDimension), "INT8 B layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out cDesc, CublasDataType.Int32, (ulong)m, (ulong)n, m),
                "INT32 C layout");
            Check(CuBlasLtNative.cublasLtMatrixLayoutCreate(
                out dDesc, CublasDataType.Int32, (ulong)m, (ulong)n, m),
                "INT32 D layout");
            int alpha = 1, beta = 0;
            CublasStatus status = CuBlasLtNative.cublasLtMatmulInt32(
                _handle, opDesc, ref alpha,
                aDev, aDesc, bDev, bDesc, ref beta,
                dDev, cDesc, dDev, dDesc,
                IntPtr.Zero, IntPtr.Zero, 0, stream);
            if (status == CublasStatus.NotSupported)
                throw new NotSupportedException(
                    "cuBLASLt does not support this standard-column-major INT8 matmul on the active device.");
            Check(status, "INT8 Matmul");
        }
        finally
        {
            if (dDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(dDesc);
            if (cDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(cDesc);
            if (bDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(bDesc);
            if (aDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatrixLayoutDestroy(aDesc);
            if (opDesc != IntPtr.Zero) CuBlasLtNative.cublasLtMatmulDescDestroy(opDesc);
        }
    }

    internal static void ValidateInt8LayoutArguments(
        IntPtr aDev, int m, int k, bool transA,
        IntPtr bDev, int n, bool transB,
        IntPtr dDev)
    {
        if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m), "Matrix dimensions must be positive.");
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "Matrix dimensions must be positive.");
        if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k), "Matrix dimensions must be positive.");
        if ((m & 3) != 0 || (k & 3) != 0)
            throw new ArgumentException(
                $"Standard-layout INT8 matmul requires m and k to be multiples of four (m={m}, k={k}).");

        ValidateAlignedDevicePointer(aDev, nameof(aDev));
        ValidateAlignedDevicePointer(bDev, nameof(bDev));
        ValidateAlignedDevicePointer(dDev, nameof(dDev));

        int aLeadingDimension = transA ? k : m;
        int bLeadingDimension = transB ? n : k;
        if ((aLeadingDimension & 3) != 0 || (bLeadingDimension & 3) != 0)
            throw new ArgumentException(
                "Standard-layout INT8 operand leading dimensions must be multiples of four. " +
                $"Computed lda={aLeadingDimension}, ldb={bLeadingDimension} for " +
                $"transA={transA}, transB={transB}.");
    }

    private static void ValidateAlignedDevicePointer(IntPtr pointer, string paramName)
    {
        if (pointer == IntPtr.Zero)
            throw new ArgumentException("The device pointer cannot be null.", paramName);
        if ((pointer.ToInt64() & 3L) != 0)
            throw new ArgumentException("The device pointer must be at least four-byte aligned.", paramName);
    }

    private static void SetAttr<TAttr>(IntPtr desc, CublasLtMatmulDescAttributes attr, ref TAttr value, int sizeInBytes)
        where TAttr : unmanaged
    {
        unsafe
        {
            fixed (TAttr* p = &value)
            {
                Check(CuBlasLtNative.cublasLtMatmulDescSetAttribute(
                    desc, attr, (IntPtr)p, (ulong)sizeInBytes),
                    $"SetAttr {attr}");
            }
        }
    }

    private static void Check(CublasStatus status, string op)
    {
        if (status != CublasStatus.Success)
            throw new InvalidOperationException($"cublasLt {op} failed: {status}.");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_handle != IntPtr.Zero)
        {
            try { CuBlasLtNative.cublasLtDestroy(_handle); } catch { }
            _handle = IntPtr.Zero;
        }
    }
}
