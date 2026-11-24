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

    private readonly record struct AutocastGemmProblem(bool TransA, bool TransB, int M, int N, int K, int Lda, int Ldb,
        int Ldd, int Compute);

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
        IntPtr a, int lda, IntPtr b, int ldb, IntPtr d, int ldd, float alpha, float beta, int compute)
    {
        var problem = new AutocastGemmProblem(transA, transB, m, n, k, lda, ldb, ldd, compute);
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
                return TryLtMatmulCore(transA, transB, m, n, k, a, lda, b, ldb, d, ldd, alpha, beta);
        }
    }

    private (Helpers.Autotune.KernelId Id, Helpers.Autotune.ShapeProfile Shape) AutocastGemmKey(in AutocastGemmProblem p)
    {
        ulong version = CuBlasLtNative.cublasLtGetVersion().ToUInt64();
        var id = new Helpers.Autotune.KernelId("cublas-autocast-gemm", $"{DeviceName}-sm{_ccMajor}{_ccMinor}-lt{version}-c{p.Compute}");
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
        var lt = (AutocastGemmRoute.Lt, 0);
        var (id, shape) = AutocastGemmKey(p);
        if (Helpers.Autotune.AutotuneCache.Lookup(id, shape) is { } recorded && ParseRoute(recorded.Variant) is { } parsed)
            return parsed;
        // A capture cannot time (or allocate): cuBLASLt runs and nothing is recorded, so a later uncaptured call measures.
        if (IsStreamCapturing()) return lt;

        var candidates = new List<(AutocastGemmRoute, int)> { lt, (AutocastGemmRoute.Classic, 0) };
        foreach (int s in SplitKCandidates(p)) candidates.Add((AutocastGemmRoute.SplitK, s));

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
                    _ => TryLtMatmulCore(p.TransA, p.TransB, p.M, p.N, p.K, a, p.Lda, b, p.Ldb, scratch, p.Ldd, 1f, 0f),
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
            p.M, p.N, p.K, (IntPtr)(&alphaValue), a, CuBlasNative.CUDA_R_32F, p.Lda, b, CuBlasNative.CUDA_R_32F, p.Ldb,
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
                p.M, p.N, kc, (IntPtr)(&alphaValue), a, CuBlasNative.CUDA_R_32F, p.Lda, strideA,
                b, CuBlasNative.CUDA_R_32F, p.Ldb, strideB, (IntPtr)(&zero),
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

    /// <summary>Runs one route (0 cuBLASLt, 1 classic, 2 split-K over <paramref name="splits"/>) for tests.</summary>
    internal bool RunAutocastGemmRouteForTest(int route, int splits, bool transA, bool transB, int m, int n, int k,
        IGpuBuffer a, int lda, IGpuBuffer b, int ldb, IGpuBuffer d, int ldd, float alpha, float beta, PrecisionMode precision)
    {
        int compute = precision == PrecisionMode.BFloat16 ? (int)CublasComputeType.Float32FastBF16 : (int)CublasComputeType.Float32Fast16F;
        var p = new AutocastGemmProblem(transA, transB, m, n, k, lda, ldb, ldd, compute);
        using var _ = PushContext();
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
