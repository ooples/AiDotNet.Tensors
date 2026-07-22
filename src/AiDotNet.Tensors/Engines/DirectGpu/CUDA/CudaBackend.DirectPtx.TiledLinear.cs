#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the general-M register-blocked tiled fused-linear GEMM
/// (<see cref="PtxFusedLinearTiledKernel"/>, issue #836). Shares the fused-linear
/// feature gate and SM86 admission with the M=1 decode kernel; one cache key adds
/// M and the activation so a single tile serves the GemmBias*/FusedLinear* family.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxFusedLinearTiledKey, PtxFusedLinearTiledKernel>
        _directPtxFusedLinearTiledKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxFusedLinearTiledDispatchCount;

    internal long DirectPtxFusedLinearTiledDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxFusedLinearTiledDispatchCount);

    private readonly record struct DirectPtxFusedLinearTiledKey(int M, int K, int N, int Activation);

    /// <summary>
    /// Attempts the general-M tiled fused-linear kernel:
    /// <c>output[M,N] = activation(input[M,K] @ transpose(weights[N,K]) + bias[N])</c>.
    /// Fails closed to the established backend for unsupported shapes/architectures,
    /// with an exact reason. Weights are canonical output-major [N,K].
    /// </summary>
    internal bool TryDirectPtxFusedLinearTiled(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer bias,
        IGpuBuffer output,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "fused-linear-tiled-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLinearTiledKernel.IsPromotedShape(m, k, n) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "fused-linear-tiled-performance-gate-not-met";
            return false;
        }
        long inputBytes = checked((long)m * k * sizeof(float));
        long weightBytes = checked((long)n * k * sizeof(float));
        long biasBytes = checked((long)n * sizeof(float));
        long outputBytes = checked((long)m * n * sizeof(float));
        if (input.SizeInBytes != inputBytes || weights.SizeInBytes != weightBytes ||
            bias.SizeInBytes != biasBytes || output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "fused-linear-tiled-physical-extent-mismatch";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFusedLinearTiledKey(m, k, n, (int)activation);
            lock (_directPtxLock)
            {
                if (!_directPtxFusedLinearTiledKernels.TryGetValue(
                    key, out PtxFusedLinearTiledKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX tiled fused linear must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheFusedLinearTiledKernelSlow(key);
                }
                if (capturing && !_directPtxFusedLinearTiledKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX tiled fused-linear module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[3]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxFusedLinearTiledDispatchCount);
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
    private PtxFusedLinearTiledKernel CreateAndCacheFusedLinearTiledKernelSlow(
        DirectPtxFusedLinearTiledKey key) =>
        _directPtxFusedLinearTiledKernels.GetOrAdd(key, () =>
            new PtxFusedLinearTiledKernel(
                _directPtxRuntime!, key.M, key.K, key.N, (DirectPtxLinearActivation)key.Activation));

    internal bool PrewarmDirectPtxFusedLinearTiled(
        int m, int k, int n, DirectPtxLinearActivation activation)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "fused-linear-tiled-shape-not-implemented";
            return false;
        }
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX tiled fused linear prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxFusedLinearTiledKey(m, k, n, (int)activation);
                if (!_directPtxFusedLinearTiledKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheFusedLinearTiledKernelSlow(key);
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

    internal bool TryGetDirectPtxFusedLinearTiledAudit(
        int m, int k, int n, DirectPtxLinearActivation activation,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFusedLinearTiledKey(m, k, n, (int)activation);
            if (_directPtxFusedLinearTiledKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }
}
#endif
