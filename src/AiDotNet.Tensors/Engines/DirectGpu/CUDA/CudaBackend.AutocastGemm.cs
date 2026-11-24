using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Gpu;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The route an autocast GEMM takes. Under CUBLAS_COMPUTE_32F_FAST_16F/_16BF, cuBLASLt's deterministic reduction
/// schemes (no in-place atomic split-K) leave it with few candidates for a long reduction over a small output -- the
/// language-model head's input gradient, 256 x 672 over K = 49152, ran a SIMT split-K sgemm in 1.48 ms -- while the
/// classic cublasGemmEx serves the same FP16 problem with a tensor-core kernel in 0.48 ms, and a fixed-order split-K
/// (a strided-batched GEMM into per-slice partials, then one FP32 GEMM against a ones vector that sums them in a
/// fixed order) serves BF16 in 0.67-0.71 ms where cuBLASLt took 1.76 ms and the classic call 1.67 ms. Every route is
/// deterministic. Which one serves a problem is measured once on the device and recorded on disk, like cuBLASLt's own
/// choice, so every later process runs the same route.
/// </summary>
public sealed partial class CudaBackend
{
    private enum AutocastGemmRoute { Lt = 0, Classic = 1, SplitK = 2 }

    // InputType is A's and B's element type: CUDA_R_32F (the fast compute type reads FP32 operands) or the 16-bit type
    // the operands were cast to (CastOperands, under CUBLAS_COMPUTE_32F).
    private readonly record struct AutocastGemmProblem(bool TransA, bool TransB, int M, int N, int K, int Lda, int Ldb,
        int Ldd, int Compute, int InputType = CuBlasNative.CUDA_R_32F);

    private readonly Dictionary<AutocastGemmProblem, (AutocastGemmRoute Route, int Splits)> _autocastGemmRoutes = new();
    private readonly object _autocastGemmLock = new();

    // Split-K slices tried, and the bounds on the partial buffer and on when a reduction counts as long.
    private static readonly int[] s_splitKSlices = { 8, 16, 32 };
    private const long SplitKMaxPartialFloats = 8L * 1024 * 1024;
    private const int SplitKOnesLength = 64;

    /// <summary>
    /// An autocast column-major GEMM (cublasSgemm semantics) on its measured route. False when no autocast scope is
    /// open or cuBLAS/cuBLASLt cannot serve the problem; the caller then runs its own path.
    /// </summary>
    private bool TryAutocastGemmColumnMajor(bool transA, bool transB, int m, int n, int k,
        IntPtr a, int lda, IntPtr b, int ldb, IntPtr d, int ldd, float alpha, float beta, int autocastState)
    {
        if (AutocastCastTypeOf(autocastState) is int castType and not 0
            && TryCastAutocastGemm(transA, transB, m, n, k, alpha, a, lda, 0, b, ldb, 0, beta, d, ldd, 0, 1, castType,
                "autocast GEMM"))
            return true;
        return RunAutocastGemmOnRoute(new AutocastGemmProblem(transA, transB, m, n, k, lda, ldb, ldd,
            AutocastComputeOf(autocastState)), a, b, d, alpha, beta);
    }

    private bool RunAutocastGemmOnRoute(in AutocastGemmProblem problem, IntPtr a, IntPtr b, IntPtr d, float alpha,
        float beta)
    {
        bool transA = problem.TransA, transB = problem.TransB;
        int m = problem.M, n = problem.N, k = problem.K, lda = problem.Lda, ldb = problem.Ldb, ldd = problem.Ldd;
        (AutocastGemmRoute Route, int Splits) route;
        bool known;
        lock (_autocastGemmLock) known = _autocastGemmRoutes.TryGetValue(problem, out route);
        if (!known)
        {
            // A route chosen during a capture is the unmeasured fallback: kept for this call only, so the first
            // uncaptured call measures.
            bool capturing = IsStreamCapturing();
            route = ChooseAutocastGemmRoute(problem, a, b);
            if (!capturing) lock (_autocastGemmLock) _autocastGemmRoutes[problem] = route;
        }
        switch (route.Route)
        {
            case AutocastGemmRoute.Classic:
                return RunAutocastClassic(problem, a, b, d, alpha, beta);
            case AutocastGemmRoute.SplitK when route.Splits > 1 && EnsureSplitKBuffers(problem, route.Splits):
                return RunAutocastSplitK(problem, route.Splits, a, b, d, alpha, beta);
            default:
                // cuBLASLt here reads FP32 operands; 16-bit ones (a split-K route that cannot get its buffers) take the
                // classic GemmEx.
                return problem.InputType == CuBlasNative.CUDA_R_32F
                    ? TryLtMatmulCore(transA, transB, m, n, k, a, lda, b, ldb, d, ldd, alpha, beta, problem.Compute)
                    : RunAutocastClassic(problem, a, b, d, alpha, beta);
        }
    }

    private (Helpers.Autotune.KernelId Id, Helpers.Autotune.ShapeProfile Shape) AutocastGemmKey(in AutocastGemmProblem p)
    {
        ulong version = CuBlasLtNative.cublasLtGetVersion().ToUInt64();
        string input = p.InputType == CuBlasNative.CUDA_R_32F ? string.Empty : "-i" + p.InputType;
        var id = new Helpers.Autotune.KernelId("cublas-autocast-gemm", $"{DeviceName}-sm{_ccMajor}{_ccMinor}-lt{version}-c{p.Compute}{input}");
        return (id, new Helpers.Autotune.ShapeProfile(p.TransA ? 1 : 0, p.TransB ? 1 : 0, p.M, p.N, p.K, p.Lda, p.Ldb, p.Ldd));
    }

    private static string RouteName((AutocastGemmRoute Route, int Splits) r)
        => r.Route switch { AutocastGemmRoute.Classic => "classic", AutocastGemmRoute.SplitK => "splitk:" + r.Splits, _ => "lt" };

    private static (AutocastGemmRoute, int)? ParseRoute(string variant)
    {
        if (variant == "lt") return (AutocastGemmRoute.Lt, 0);
        if (variant == "classic") return (AutocastGemmRoute.Classic, 0);
        if (variant.StartsWith("splitk:", StringComparison.Ordinal)
            && int.TryParse(variant.Substring(7), System.Globalization.NumberStyles.None,
                System.Globalization.CultureInfo.InvariantCulture, out int s) && s > 1)
            return (AutocastGemmRoute.SplitK, s);
        return null;
    }

    // Split-K serves a long reduction over an output that leaves most SMs idle, written contiguously (ldd == m).
    private static IEnumerable<int> SplitKCandidates(AutocastGemmProblem p)
    {
        if (p.Ldd != p.M || p.K < 4L * Math.Max(p.M, p.N)) yield break;
        foreach (int s in s_splitKSlices)
            if (p.K % s == 0 && p.K / s >= 512 && (long)s * p.M * p.N <= SplitKMaxPartialFloats)
                yield return s;
    }

    private (AutocastGemmRoute, int) ChooseAutocastGemmRoute(AutocastGemmProblem p, IntPtr a, IntPtr b)
    {
        // cuBLASLt here takes FP32 operands only, so 16-bit operands choose between the classic GemmEx and split-K.
        bool castOperands = p.InputType != CuBlasNative.CUDA_R_32F;
        var lt = castOperands ? (AutocastGemmRoute.Classic, 0) : (AutocastGemmRoute.Lt, 0);
        var (id, shape) = AutocastGemmKey(p);
        if (Helpers.Autotune.AutotuneCache.Lookup(id, shape) is { } recorded && ParseRoute(recorded.Variant) is { } parsed)
            return parsed;
        // A capture cannot time (or allocate): cuBLASLt runs and nothing is recorded, so a later uncaptured call measures.
        if (IsStreamCapturing()) return lt;

        var candidates = castOperands
            ? new List<(AutocastGemmRoute, int)> { lt }
            : new List<(AutocastGemmRoute, int)> { lt, (AutocastGemmRoute.Classic, 0) };
        foreach (int s in SplitKCandidates(p)) candidates.Add((AutocastGemmRoute.SplitK, s));
        if (candidates.Count == 1) return candidates[0];   // nothing to choose between

        ulong bytes = (ulong)p.Ldd * (ulong)p.N * sizeof(float);
        IntPtr scratch = IntPtr.Zero, start = IntPtr.Zero, stop = IntPtr.Zero;
        (AutocastGemmRoute, int) best = lt;
        float bestMs = float.MaxValue;
        try
        {
            if (CudaNativeBindings.cuMemAlloc(out scratch, bytes) != CudaResult.Success
                || CudaNativeBindings.cuEventCreate(out start, 0) != CudaResult.Success
                || CudaNativeBindings.cuEventCreate(out stop, 0) != CudaResult.Success)
                return lt;
            foreach (var candidate in candidates)
            {
                if (candidate.Item1 == AutocastGemmRoute.SplitK && !EnsureSplitKBuffers(p, candidate.Item2)) continue;
                bool Run() => candidate.Item1 switch
                {
                    AutocastGemmRoute.Classic => RunAutocastClassic(p, a, b, scratch, 1f, 0f),
                    AutocastGemmRoute.SplitK => RunAutocastSplitK(p, candidate.Item2, a, b, scratch, 1f, 0f),
                    _ => TryLtMatmulCore(p.TransA, p.TransB, p.M, p.N, p.K, a, p.Lda, b, p.Ldb, scratch, p.Ldd, 1f, 0f, p.Compute),
                };
                bool ok = Run() && Run();   // warm (cuBLASLt measures its own candidates on its first call)
                float candidateMs = float.MaxValue;
                for (int batch = 0; batch < 3 && ok; batch++)
                {
                    CudaNativeBindings.cuEventRecord(start, _stream);
                    for (int i = 0; i < 4 && ok; i++) ok = Run();
                    CudaNativeBindings.cuEventRecord(stop, _stream);
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
                    best = candidate;
                }
            }
        }
        finally
        {
            if (stop != IntPtr.Zero) CudaNativeBindings.cuEventSynchronize(stop);
            if (start != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(start);
            if (stop != IntPtr.Zero) CudaNativeBindings.cuEventDestroy(stop);
            if (scratch != IntPtr.Zero) CudaNativeBindings.cuMemFree(scratch);
        }
        bool stored = bestMs < float.MaxValue && Helpers.Autotune.AutotuneCache.TryStore(id, shape,
            new Helpers.Autotune.KernelChoice { Variant = RouteName(best), MeasuredTimeMs = bestMs });
        // As for cuBLASLt's own choice: an unrecorded measurement could differ in the next process, so under
        // GpuDeterminism the fixed route (cuBLASLt) runs instead.
        if (!stored && GpuDeterminism.IsActive) best = lt;
        return best;
    }

    private unsafe bool RunAutocastClassic(in AutocastGemmProblem p, IntPtr a, IntPtr b, IntPtr d, float alpha, float beta)
    {
        float alphaValue = alpha, betaValue = beta;
        return CuBlasNative.cublasGemmEx(_cublasHandle,
            p.TransA ? CublasOperation.Transpose : CublasOperation.None,
            p.TransB ? CublasOperation.Transpose : CublasOperation.None,
            p.M, p.N, p.K, (IntPtr)(&alphaValue), a, p.InputType, p.Lda, b, p.InputType, p.Ldb,
            (IntPtr)(&betaValue), d, CuBlasNative.CUDA_R_32F, p.Ldd, p.Compute, CuBlasNative.CUBLAS_GEMM_DEFAULT)
            == Engines.CublasStatus.Success;
    }

    // The K slices as a strided batch: slice s reads A's and B's columns (or rows, transposed) [s*kc, (s+1)*kc).
    private unsafe bool RunAutocastSplitK(in AutocastGemmProblem p, int splits, IntPtr a, IntPtr b, IntPtr d,
        float alpha, float beta)
    {
        int kc = p.K / splits;
        long strideA = p.TransA ? kc : (long)kc * p.Lda;   // A is k x m (transposed) or m x k
        long strideB = p.TransB ? (long)kc * p.Ldb : kc;   // B is n x k (transposed) or k x n
        long outputs = (long)p.M * p.N;
        float alphaValue = alpha, zero = 0f, one = 1f, betaValue = beta;
        IntPtr handle = _cublasHandle;
        var tc = _threadCublas!.Value!;
        if (CuBlasNative.cublasGemmStridedBatchedEx(handle,
                p.TransA ? CublasOperation.Transpose : CublasOperation.None,
                p.TransB ? CublasOperation.Transpose : CublasOperation.None,
                p.M, p.N, kc, (IntPtr)(&alphaValue), a, p.InputType, p.Lda, strideA,
                b, p.InputType, p.Ldb, strideB, (IntPtr)(&zero),
                tc.SplitKPartials!.Handle, CuBlasNative.CUDA_R_32F, p.M, outputs, splits, p.Compute,
                CuBlasNative.CUBLAS_GEMM_DEFAULT) != Engines.CublasStatus.Success)
            return false;
        // D = partials[outputs x splits] * ones[splits] + beta * D, in strict FP32 and a fixed order.
        return CuBlasNative.cublasGemmEx(handle, CublasOperation.None, CublasOperation.None,
            checked((int)outputs), 1, splits, (IntPtr)(&one), tc.SplitKPartials.Handle, CuBlasNative.CUDA_R_32F,
            checked((int)outputs), tc.SplitKOnes!.Handle, CuBlasNative.CUDA_R_32F, splits, (IntPtr)(&betaValue),
            d, CuBlasNative.CUDA_R_32F, checked((int)outputs), CuBlasNative.CUBLAS_COMPUTE_32F_PEDANTIC,
            CuBlasNative.CUBLAS_GEMM_DEFAULT) == Engines.CublasStatus.Success;
    }

    // The calling thread's partial and ones buffers, grown outside a capture only (a capture cannot allocate). Per
    // thread, beside the handle's workspace and in the same stream order: threads issuing split-K GEMMs at once (one
    // capturing, one eager, on different streams) never share partials. A buffer outgrown is retired, not freed: a
    // graph captured with it keeps its address and replays into it until the backend is disposed.
    private bool EnsureSplitKBuffers(in AutocastGemmProblem p, int splits)
    {
        long need = (long)splits * p.M * p.N;
        if (!IsCublasAvailable || GemmLaunchStream != IntPtr.Zero) return false;   // a fan-out slice: no workspace either
        _ = _cublasHandle;
        if (_threadCublas?.Value is not { } tc) return false;
        if (tc.SplitKOnes is null)
        {
            if (IsStreamCapturing()) return false;
            var ones = new float[SplitKOnesLength];
            for (int i = 0; i < ones.Length; i++) ones[i] = 1f;
            tc.SplitKOnes = AllocateBuffer(ones);
        }
        if (tc.SplitKPartials is null || tc.SplitKPartials.Size < need)
        {
            if (IsStreamCapturing()) return false;
            // Grown to a power of two (at most SplitKMaxPartialFloats) so a thread retires few buffers.
            long size = 1;
            while (size < need) size <<= 1;
            size = Math.Min(size, Math.Max(need, SplitKMaxPartialFloats));
            if (tc.SplitKPartials is not null) (tc.RetiredSplitKPartials ??= new List<IGpuBuffer>()).Add(tc.SplitKPartials);
            tc.SplitKPartials = AllocateBuffer(checked((int)size));
        }
        return true;
    }

    private long _autocastCastDeclines;
    private int _autocastCastDeclineWarned;

    /// <summary>Tests: this backend's autocast GEMMs that could not cast their operands and ran the fast compute type.</summary>
    internal long AutocastCastDeclinesForTest => System.Threading.Interlocked.Read(ref _autocastCastDeclines);

    // Elements a column-major operand of `rows` x `cols` with leading dimension `ld` spans, over `batch` matrices
    // `stride` apart (stride 0: one matrix every batch entry reads).
    private static long OperandSpan(int rows, int cols, int ld, long stride, int batch)
    {
        if (rows <= 0 || cols <= 0) return 0;
        long one = (long)(cols - 1) * ld + rows;
        return batch <= 1 || stride == 0 ? one : stride * (batch - 1) + one;
    }

    // Whole floats holding `count` 16-bit elements, as the backend's 16-bit buffers are sized.
    private static int HalfWords(long count, string what)
    {
        long words = (count + 1) / 2;
        if (words > int.MaxValue)
            throw new NotSupportedException(
                $"{what} (autocast, cast operands): an operand of {count} elements is beyond one 16-bit scratch buffer.");
        return (int)words;
    }

    /// <summary>
    /// The autocast GEMM with its operands cast (PyTorch's autocast): A and B rounded to <paramref name="castType"/>
    /// (round to nearest even) into stream-ordered scratch on the GEMM's stream, then multiplied by cuBLAS on 16-bit
    /// inputs under CUBLAS_COMPUTE_32F (FP32 accumulation) into the FP32 result, so the products are those of the rounded
    /// operands whatever kernel cuBLAS picks. A single GEMM takes its measured route (the classic GemmEx or a fixed-order
    /// split-K); a strided batch runs cublasGemmStridedBatchedEx; a GEMM fan-out slice runs the classic GemmEx on its
    /// own stream, with its scratch in that stream's order. The scratch comes from the backend's allocator (its OOM
    /// reclamation, memory accounting and budget); an allocation that still fails throws, as any other allocation does,
    /// rather than run the GEMM unrounded. False -- nothing launched -- for an empty GEMM (nothing to round) and on a
    /// device without the 16-bit conversion kernel (BF16 below compute capability 8.0): the caller then runs the fast
    /// compute type, counted and warned about once per backend.
    /// </summary>
    private unsafe bool TryCastAutocastGemm(bool transA, bool transB, int m, int n, int k, float alpha,
        IntPtr a, int lda, long strideA, IntPtr b, int ldb, long strideB, float beta, IntPtr c, int ldc, long strideC,
        int batch, int castType, string what)
    {
        if (m <= 0 || n <= 0 || k <= 0 || batch <= 0) return false;
        var storage = castType == CuBlasNative.CUDA_R_16BF ? FftElementType.BFloat16 : FftElementType.Float16;
        if (!TryGetCastKernel(storage, out IntPtr kernel, out string? why))
        {
            System.Threading.Interlocked.Increment(ref _autocastCastDeclines);
            if (System.Threading.Interlocked.Exchange(ref _autocastCastDeclineWarned, 1) == 0)
                System.Diagnostics.Trace.TraceWarning(
                    $"Matmul autocast cannot round GEMM operands to {storage} on {DeviceName} ({why}); its GEMMs run the "
                    + "fast compute type instead, which cuBLAS may serve in FP32 for some shapes.");
            return false;
        }
        long aCount = OperandSpan(transA ? k : m, transA ? m : k, lda, strideA, batch);
        long bCount = OperandSpan(transB ? n : k, transB ? k : n, ldb, strideB, batch);
        int aWords = HalfWords(aCount, what), bWords = HalfWords(bCount, what);

        IntPtr fanout = GemmLaunchStream;
        if (fanout == IntPtr.Zero)
        {
            // Released in the main stream's order once the GEMM queued on it has read them.
            using var a16 = AllocateStreamOrderedScratchBufferUninitialized(aWords);
            using var b16 = AllocateStreamOrderedScratchBufferUninitialized(bWords);
            LaunchCast(kernel, a, a16.Handle, aCount, _stream);
            LaunchCast(kernel, b, b16.Handle, bCount, _stream);
            RunCastGemm(transA, transB, m, n, k, alpha, a16.Handle, lda, strideA, b16.Handle, ldb, strideB, beta, c, ldc,
                strideC, batch, castType, what, routed: true);
            return true;
        }

        // A fan-out slice runs on a leased stream the main stream joins only at the fan-out's end: its scratch is
        // allocated, read and freed in the slice stream's own order.
        IntPtr a16p = IntPtr.Zero, b16p = IntPtr.Zero;
        try
        {
            a16p = AllocCastScratchOnStream(fanout, aWords, what);
            b16p = AllocCastScratchOnStream(fanout, bWords, what);
            LaunchCast(kernel, a, a16p, aCount, fanout);
            LaunchCast(kernel, b, b16p, bCount, fanout);
            RunCastGemm(transA, transB, m, n, k, alpha, a16p, lda, strideA, b16p, ldb, strideB, beta, c, ldc, strideC,
                batch, castType, what, routed: false);
            return true;
        }
        finally
        {
            FreeCastScratchOnStream(b16p, fanout);
            FreeCastScratchOnStream(a16p, fanout);
        }
    }

    private unsafe void RunCastGemm(bool transA, bool transB, int m, int n, int k, float alpha, IntPtr a16, int lda,
        long strideA, IntPtr b16, int ldb, long strideB, float beta, IntPtr c, int ldc, long strideC, int batch,
        int castType, string what, bool routed)
    {
        if (batch == 1)
        {
            var problem = new AutocastGemmProblem(transA, transB, m, n, k, lda, ldb, ldc, CuBlasNative.CUBLAS_COMPUTE_32F,
                castType);
            bool ok = routed ? RunAutocastGemmOnRoute(problem, a16, b16, c, alpha, beta)
                : RunAutocastClassic(problem, a16, b16, c, alpha, beta);
            // The operands are already rounded: an FP32 GEMM on the original operands would silently drop the cast.
            if (!ok) throw new InvalidOperationException(what + " (autocast, cast operands): cuBLAS declined the GEMM.");
            return;
        }
        float alphaValue = alpha, betaValue = beta;
        CuBlasNative.CheckCublasStatus(
            CuBlasNative.cublasGemmStridedBatchedEx(_cublasHandle,
                transA ? CublasOperation.Transpose : CublasOperation.None,
                transB ? CublasOperation.Transpose : CublasOperation.None,
                m, n, k, (IntPtr)(&alphaValue), a16, castType, lda, strideA, b16, castType, ldb, strideB,
                (IntPtr)(&betaValue), c, CuBlasNative.CUDA_R_32F, ldc, strideC, batch,
                CuBlasNative.CUBLAS_COMPUTE_32F, CuBlasNative.CUBLAS_GEMM_DEFAULT), what + " (autocast, cast operands)");
    }

    private bool TryGetCastKernel(FftElementType storage, out IntPtr kernel, out string? why)
    {
        kernel = IntPtr.Zero;
        why = null;
        // The device check first: a device without the 16-bit type (BF16 below compute capability 8.0) declines without
        // building and catching an exception on every GEMM.
        if (!SupportsFftElementType(storage))
        {
            why = $"the device has no {storage} conversion";
            return false;
        }
        try
        {
            kernel = RequireGenericFftKernel("fftg_from_f32" + storage.KernelSuffix(), storage);
            if (kernel == IntPtr.Zero) why = "the conversion kernel did not load";
            return kernel != IntPtr.Zero;
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException)
        {
            why = ex.Message;
            return false;
        }
    }

    // 16-bit scratch in `stream`'s order (a GEMM fan-out slice's lease stream; fan-out does not run inside a capture).
    // Admitted under a GpuMemoryBudget like any stream-ordered request. On OOM the backend reclaims as its own allocator
    // does -- pooled and peer buffers released, unreachable results finalized, the context synchronized and the pool
    // trimmed -- and retries once; a second failure throws.
    private IntPtr AllocCastScratchOnStream(IntPtr stream, int words, string what)
    {
        ulong bytes = (ulong)words * sizeof(float);
        bool budgeted = _memoryAccount.LimitBytes is not null;
        if (budgeted) AdmitUnderBudget(PoolGrowthBound(bytes));
        CudaResult r;
        IntPtr pointer;
        try
        {
            r = CuBlasNative.cuMemAllocAsync(out pointer, bytes, stream);
            if (r == CudaResult.OutOfMemory && !IsStreamCapturing())
            {
                _bufferPool.DrainAll();
                DrainPeerPools();
                ReclaimUnreachableForRetry();
                CuBlasNative.CheckCudaResult(SynchronizeContextOutsideCapture(_cudaContext),
                    "cuCtxSynchronize (autocast operand scratch OOM reclamation)");
                if (_asyncMemPool != IntPtr.Zero) TrimPool(_asyncMemPool);
                r = CuBlasNative.cuMemAllocAsync(out pointer, bytes, stream);
            }
        }
        finally
        {
            if (budgeted) AiDotNet.Tensors.Engines.DevicePrimitives.DeviceMemoryAccount.SettleAdmission();
        }
        if (r == CudaResult.OutOfMemory)
            throw CudaAllocationError(r, $"{what} (autocast, cast operands): cuMemAllocAsync of {bytes} bytes failed.");
        CuBlasNative.CheckCudaResult(r, "cuMemAllocAsync (autocast operand scratch)");
        GpuMemoryTracker.OnAlloc(pointer, (long)bytes);
        return pointer;
    }

    private static void FreeCastScratchOnStream(IntPtr pointer, IntPtr stream)
    {
        if (pointer == IntPtr.Zero) return;
        GpuMemoryTracker.OnFree(pointer);
        CuBlasNative.cuMemFreeAsync(pointer, stream);
    }

    // The conversion kernel over `count` elements, in launches of at most 2^30 elements (its count is an int).
    private unsafe void LaunchCast(IntPtr kernel, IntPtr source, IntPtr destination, long count, IntPtr stream)
    {
        const long chunk = 1L << 30;
        void** args = stackalloc void*[3];
        for (long offset = 0; offset < count; offset += chunk)
        {
            IntPtr src = (IntPtr)((long)source + offset * sizeof(float));
            IntPtr dst = (IntPtr)((long)destination + offset * 2);
            int n = (int)Math.Min(chunk, count - offset);
            args[0] = &src;
            args[1] = &dst;
            args[2] = &n;
            LaunchKernelOnStream(kernel, GridFor(n), DefaultBlockSize, args, stream);
        }
    }

    /// <summary>Runs one route (0 cuBLASLt, 1 classic, 2 split-K over <paramref name="splits"/>) for tests.</summary>
    internal bool RunAutocastGemmRouteForTest(int route, int splits, bool transA, bool transB, int m, int n, int k,
        IGpuBuffer a, int lda, IGpuBuffer b, int ldb, IGpuBuffer d, int ldd, float alpha, float beta, PrecisionMode precision,
        bool castOperands = false)
    {
        int compute = precision == PrecisionMode.BFloat16 ? (int)CublasComputeType.Float32FastBF16 : (int)CublasComputeType.Float32Fast16F;
        using var _ = PushContext();
        if (castOperands)
        {
            // The operands rounded to 16 bits (as TryCastAutocastGemm does), then the route on 16-bit inputs.
            int castType = precision == PrecisionMode.BFloat16 ? CuBlasNative.CUDA_R_16BF : CuBlasNative.CUDA_R_16F;
            var storage = precision == PrecisionMode.BFloat16 ? FftElementType.BFloat16 : FftElementType.Float16;
            if (!TryGetCastKernel(storage, out IntPtr kernel, out string? _unused)) return false;
            long aCount = OperandSpan(transA ? k : m, transA ? m : k, lda, 0, 1);
            long bCount = OperandSpan(transB ? n : k, transB ? k : n, ldb, 0, 1);
            using var a16 = AllocateStreamOrderedScratchBufferUninitialized(HalfWords(aCount, "test"));
            using var b16 = AllocateStreamOrderedScratchBufferUninitialized(HalfWords(bCount, "test"));
            LaunchCast(kernel, a.Handle, a16.Handle, aCount, _stream);
            LaunchCast(kernel, b.Handle, b16.Handle, bCount, _stream);
            var cast = new AutocastGemmProblem(transA, transB, m, n, k, lda, ldb, ldd, CuBlasNative.CUBLAS_COMPUTE_32F, castType);
            bool castOk = route switch
            {
                1 => RunAutocastClassic(cast, a16.Handle, b16.Handle, d.Handle, alpha, beta),
                2 => EnsureSplitKBuffers(cast, splits) && RunAutocastSplitK(cast, splits, a16.Handle, b16.Handle, d.Handle, alpha, beta),
                _ => false,   // cuBLASLt reads FP32 operands here
            };
            Synchronize();
            return castOk;
        }
        var p = new AutocastGemmProblem(transA, transB, m, n, k, lda, ldb, ldd, compute);
        using var scope = ((IGemmAutocastBackend)this).EnterGemmAutocast(precision);
        bool ok = route switch
        {
            1 => RunAutocastClassic(p, a.Handle, b.Handle, d.Handle, alpha, beta),
            2 => EnsureSplitKBuffers(p, splits) && RunAutocastSplitK(p, splits, a.Handle, b.Handle, d.Handle, alpha, beta),
            _ => TryLtMatmulCore(transA, transB, m, n, k, a.Handle, lda, b.Handle, ldb, d.Handle, ldd, alpha, beta),
        };
        Synchronize();
        return ok;
    }

    // The thread's split-K buffers, with the cuBLAS handle they serve (the backend's Dispose).
    private static void DisposeSplitKBuffers(ThreadCublas tc)
    {
        tc.SplitKPartials?.Dispose();
        tc.SplitKPartials = null;
        tc.SplitKOnes?.Dispose();
        tc.SplitKOnes = null;
        if (tc.RetiredSplitKPartials is { } retired)
            foreach (var buffer in retired) buffer.Dispose();
        tc.RetiredSplitKPartials = null;
    }
}
