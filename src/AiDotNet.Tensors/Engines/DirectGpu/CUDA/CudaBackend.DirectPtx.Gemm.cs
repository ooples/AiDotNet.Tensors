using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the plain register-blocked tiled GEMM
/// (<see cref="PtxGemmKernel"/>, issue #836): <c>C[M,N] = A[M,K] @ B[K,N]</c>, the
/// alpha=1/beta=0 no-bias path behind Gemm / MatMul. Fails closed to cuBLAS for
/// unsupported shapes/architectures or until a shape is GPU-validated and promoted,
/// so production behavior is unchanged.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxGemmKey, PtxGemmKernel>
        _directPtxGemmKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxGemmDispatchCount;

    internal long DirectPtxGemmDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxGemmDispatchCount);

    private readonly record struct DirectPtxGemmKey(int M, int K, int N);

    /// <summary>
    /// Attempts the plain tiled GEMM <c>C[M,N] = A[M,K] @ B[K,N]</c>. Returns false
    /// (leaving the caller on the established cuBLAS path) for unsupported shapes, a
    /// physical-extent mismatch, or until the shape is promoted.
    /// </summary>
    internal bool TryDirectPtxGemm(
        IGpuBuffer a, IGpuBuffer b, IGpuBuffer c, int m, int k, int n)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxGemmKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "gemm-shape-not-implemented";
            return false;
        }
        if (!PtxGemmKernel.IsPromotedShape(m, k, n) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "gemm-performance-gate-not-met";
            return false;
        }
        long aBytes = checked((long)m * k * sizeof(float));
        long bBytes = checked((long)k * n * sizeof(float));
        long cBytes = checked((long)m * n * sizeof(float));
        if (a.SizeInBytes != aBytes || b.SizeInBytes != bBytes || c.SizeInBytes != cBytes)
        {
            DirectPtxLastError = "gemm-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxGemmKey(m, k, n);
            lock (_directPtxLock)
            {
                if (!_directPtxGemmKernels.TryGetValue(key, out PtxGemmKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX GEMM must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheGemmKernelSlow(key);
                }
                if (capturing && !_directPtxGemmKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX GEMM module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(a, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(b, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(c, kernel.Blueprint.Tensors[2]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxGemmDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxGemmKernel CreateAndCacheGemmKernelSlow(DirectPtxGemmKey key) =>
        _directPtxGemmKernels.GetOrAdd(key, () =>
            new PtxGemmKernel(_directPtxRuntime!, key.M, key.K, key.N));

    internal bool PrewarmDirectPtxGemm(int m, int k, int n)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxGemmKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "gemm-shape-not-implemented";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX GEMM prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxGemmKey(m, k, n);
                if (!_directPtxGemmKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheGemmKernelSlow(key);
            }
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool TryGetDirectPtxGemmAudit(int m, int k, int n, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxGemmKey(m, k, n);
            if (_directPtxGemmKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }
}
