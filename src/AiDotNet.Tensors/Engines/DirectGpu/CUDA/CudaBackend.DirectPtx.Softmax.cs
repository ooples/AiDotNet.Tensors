using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the row-wise softmax direct-PTX kernel
/// (<see cref="PtxSoftmaxKernel"/>, issue #840): stable <c>exp(x-rowMax)/rowSumExp</c> over
/// the last axis. Fails closed to the established NVRTC softmax for unsupported
/// shapes/architectures or until a shape is GPU-validated and promoted, so production
/// behavior is unchanged.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxSoftmaxKey, PtxSoftmaxKernel>
        _directPtxSoftmaxKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxSoftmaxDispatchCount;

    internal long DirectPtxSoftmaxDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxSoftmaxDispatchCount);

    internal bool IsDirectPtxSoftmaxEnabled =>
        DirectPtxFeatureGate.IsSoftmaxEnabled && IsAvailable &&
        DirectPtxArchitecture.HasValidatedSoftmax(_ccMajor, _ccMinor);

    private readonly record struct DirectPtxSoftmaxKey(int M, int N);

    /// <summary>
    /// Attempts the row-wise softmax over the last axis (M rows of N logits). Returns false
    /// (leaving the caller on the NVRTC path) for unsupported shapes, an extent mismatch,
    /// during graph capture, or until the shape is promoted.
    /// </summary>
    internal bool TryDirectPtxSoftmax(IGpuBuffer input, IGpuBuffer output, int m, int n)
    {
        if (!IsDirectPtxSoftmaxEnabled) return false;
        if (!PtxSoftmaxKernel.IsSupportedShape(m, n))
        {
            DirectPtxLastError = "softmax-shape-not-implemented";
            return false;
        }
        if (!PtxSoftmaxKernel.IsPromotedShape(m, n) &&
            !DirectPtxFeatureGate.SoftmaxExperimentOverride)
        {
            DirectPtxLastError = "softmax-performance-gate-not-met";
            return false;
        }
        long bytes = checked((long)m * n * sizeof(float));
        if (input.SizeInBytes != bytes || output.SizeInBytes != bytes)
        {
            DirectPtxLastError = "softmax-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxSoftmaxKey(m, n);
            lock (_directPtxLock)
            {
                if (!_directPtxSoftmaxKernels.TryGetValue(key, out PtxSoftmaxKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError = "Direct PTX softmax must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheSoftmaxKernelSlow(key);
                }
                if (capturing && !_directPtxSoftmaxKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX softmax module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxSoftmaxDispatchCount);
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
    private PtxSoftmaxKernel CreateAndCacheSoftmaxKernelSlow(DirectPtxSoftmaxKey key) =>
        _directPtxSoftmaxKernels.GetOrAdd(key, () => new PtxSoftmaxKernel(_directPtxRuntime!, key.M, key.N));

    internal bool PrewarmDirectPtxSoftmax(int m, int n)
    {
        if (!IsDirectPtxSoftmaxEnabled) return false;
        if (!PtxSoftmaxKernel.IsSupportedShape(m, n))
        {
            DirectPtxLastError = "softmax-shape-not-implemented";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX softmax prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxSoftmaxKey(m, n);
                if (!_directPtxSoftmaxKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheSoftmaxKernelSlow(key);
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

    internal bool TryGetDirectPtxSoftmaxAudit(int m, int n, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxSoftmaxKey(m, n);
            if (_directPtxSoftmaxKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }
}
