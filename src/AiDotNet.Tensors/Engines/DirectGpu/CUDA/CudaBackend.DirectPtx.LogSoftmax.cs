using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the row-wise log-softmax direct-PTX kernel
/// (<see cref="PtxLogSoftmaxKernel"/>, issue #840): <c>x - logsumexp</c> over the last axis.
/// Fails closed to the established NVRTC path for unsupported shapes/architectures or until
/// a shape is GPU-validated and promoted.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxLogSoftmaxKey, PtxLogSoftmaxKernel>
        _directPtxLogSoftmaxKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxLogSoftmaxDispatchCount;

    internal long DirectPtxLogSoftmaxDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxLogSoftmaxDispatchCount);

    private readonly record struct DirectPtxLogSoftmaxKey(int M, int N);

    internal bool TryDirectPtxLogSoftmax(IGpuBuffer input, IGpuBuffer output, int m, int n)
    {
        if (!IsDirectPtxSoftmaxEnabled) return false;
        if (!PtxLogSoftmaxKernel.IsSupportedShape(m, n))
        {
            DirectPtxLastError = "log-softmax-shape-not-implemented";
            return false;
        }
        if (!PtxLogSoftmaxKernel.IsPromotedShape(m, n) &&
            !DirectPtxFeatureGate.SoftmaxExperimentOverride)
        {
            DirectPtxLastError = "log-softmax-performance-gate-not-met";
            return false;
        }
        long bytes = checked((long)m * n * sizeof(float));
        if (input.SizeInBytes != bytes || output.SizeInBytes != bytes)
        {
            DirectPtxLastError = "log-softmax-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxLogSoftmaxKey(m, n);
            lock (_directPtxLock)
            {
                if (!_directPtxLogSoftmaxKernels.TryGetValue(key, out PtxLogSoftmaxKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError = "Direct PTX log-softmax must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheLogSoftmaxKernelSlow(key);
                }
                if (capturing && !_directPtxLogSoftmaxKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX log-softmax module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxLogSoftmaxDispatchCount);
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
    private PtxLogSoftmaxKernel CreateAndCacheLogSoftmaxKernelSlow(DirectPtxLogSoftmaxKey key) =>
        _directPtxLogSoftmaxKernels.GetOrAdd(key, () => new PtxLogSoftmaxKernel(_directPtxRuntime!, key.M, key.N));

    internal bool PrewarmDirectPtxLogSoftmax(int m, int n)
    {
        if (!IsDirectPtxSoftmaxEnabled) return false;
        if (!PtxLogSoftmaxKernel.IsSupportedShape(m, n))
        {
            DirectPtxLastError = "log-softmax-shape-not-implemented";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX log-softmax prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxLogSoftmaxKey(m, n);
                if (!_directPtxLogSoftmaxKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheLogSoftmaxKernelSlow(key);
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

    internal bool TryGetDirectPtxLogSoftmaxAudit(int m, int n, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxLogSoftmaxKey(m, n);
            if (_directPtxLogSoftmaxKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }
}
