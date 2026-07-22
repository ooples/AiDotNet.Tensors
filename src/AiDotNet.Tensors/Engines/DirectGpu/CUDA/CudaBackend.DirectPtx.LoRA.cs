using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the standard-layout fused LoRA forward
/// (<see cref="PtxFusedLoRAForwardStandardKernel"/>, issue #836):
/// <c>output = base + scale * (X[M,K] @ loraA[K,R]) @ loraB[R,N]</c> with r = 64,
/// matching the exact factor layout of <see cref="CudaBackend.FusedLoRAForward"/>. Fails
/// closed to the established NVRTC kernel for other ranks/shapes or until GPU-promoted.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxLoRAKey, PtxFusedLoRAForwardStandardKernel>
        _directPtxLoRAKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxLoRADispatchCount;

    internal long DirectPtxLoRADispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxLoRADispatchCount);

    private readonly record struct DirectPtxLoRAKey(int M, int K, int N, int ScaleBits);

    /// <summary>
    /// Attempts the standard-layout fused LoRA forward. Returns false (leaving the caller
    /// on the NVRTC path) for rank != 64, unsupported shapes, an extent mismatch, during
    /// graph capture, or unless the experiment override is set.
    /// </summary>
    internal bool TryDirectPtxFusedLoRAForward(
        IGpuBuffer input,
        IGpuBuffer loraA,
        IGpuBuffer loraB,
        IGpuBuffer baseOutput,
        IGpuBuffer output,
        int m,
        int k,
        int n,
        float scale)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxFusedLoRAForwardStandardKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "fused-lora-shape-not-implemented";
            return false;
        }
        if (!PtxFusedLoRAForwardStandardKernel.IsPromotedShape(m, k, n) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "fused-lora-performance-gate-not-met";
            return false;
        }
        int r = PtxFusedLoRAForwardStandardKernel.Rank;
        long xBytes = checked((long)m * k * sizeof(float));
        long aBytes = checked((long)k * r * sizeof(float));
        long bBytes = checked((long)r * n * sizeof(float));
        long mnBytes = checked((long)m * n * sizeof(float));
        if (input.SizeInBytes != xBytes || loraA.SizeInBytes != aBytes ||
            loraB.SizeInBytes != bBytes || baseOutput.SizeInBytes != mnBytes ||
            output.SizeInBytes != mnBytes)
        {
            DirectPtxLastError = "fused-lora-physical-extent-mismatch";
            return false;
        }

        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX fused LoRA is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            var key = new DirectPtxLoRAKey(m, k, n, BitConverter.ToInt32(BitConverter.GetBytes(scale), 0));
            lock (_directPtxLock)
            {
                if (!_directPtxLoRAKernels.TryGetValue(key, out PtxFusedLoRAForwardStandardKernel? kernel))
                {
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheLoRAKernelSlow(key, scale);
                }
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(loraA, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(loraB, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(baseOutput, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxLoRADispatchCount);
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
    private PtxFusedLoRAForwardStandardKernel CreateAndCacheLoRAKernelSlow(
        DirectPtxLoRAKey key, float scale) =>
        _directPtxLoRAKernels.GetOrAdd(key, () =>
            new PtxFusedLoRAForwardStandardKernel(_directPtxRuntime!, key.M, key.K, key.N, scale));

    internal bool TryGetDirectPtxLoRAAudit(int m, int k, int n, float scale, out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxLoRAKey(m, k, n, BitConverter.ToInt32(BitConverter.GetBytes(scale), 0));
            if (_directPtxLoRAKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }
}
