using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxLoRAKey, PtxFusedLoRAKernel>
        _directPtxLoRAKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxLoRADispatchCount;

    internal long DirectPtxLoRADispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxLoRADispatchCount);

    private readonly record struct DirectPtxLoRAKey(
        int Batch, int InputFeatures, int Rank, int OutputFeatures, int ScalingBits);

    internal bool TryDirectPtxFusedLoRA(
        IGpuBuffer input,
        IGpuBuffer baseOutput,
        IGpuBuffer loraA,
        IGpuBuffer loraB,
        IGpuBuffer output,
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        // LoRA remains experimental until a concrete specialization passes
        // the correctness, tail-latency, and strongest-competitor gates.
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxLoRASemantics(
            batch, inputFeatures, rank, outputFeatures, scaling))
            return false;
        if (DirectPtxBufferIsInvalid(input) || DirectPtxBufferIsInvalid(baseOutput) ||
            DirectPtxBufferIsInvalid(loraA) || DirectPtxBufferIsInvalid(loraB) ||
            DirectPtxBufferIsInvalid(output))
        {
            DirectPtxLastError = "fused-lora-null-or-invalid-buffer";
            return false;
        }

        long inputBytes = checked((long)batch * inputFeatures * sizeof(float));
        long outputBytes = checked((long)batch * outputFeatures * sizeof(float));
        long aBytes = checked((long)inputFeatures * rank * sizeof(float));
        long bBytes = checked((long)rank * outputFeatures * sizeof(float));
        if (input.SizeInBytes != inputBytes || baseOutput.SizeInBytes != outputBytes ||
            loraA.SizeInBytes != aBytes || loraB.SizeInBytes != bBytes ||
            output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "fused-lora-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, input) ||
            DirectPtxBuffersOverlap(output, loraA) ||
            DirectPtxBuffersOverlap(output, loraB) ||
            (DirectPtxBuffersOverlap(output, baseOutput) &&
             output.Handle != baseOutput.Handle))
        {
            DirectPtxLastError = "fused-lora-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxLoRAKey(
                batch, inputFeatures, rank, outputFeatures,
                PtxCompat.SingleToInt32Bits(scaling));
            lock (_directPtxLock)
            {
                if (!_directPtxLoRAKernels.TryGetValue(key, out PtxFusedLoRAKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX fused LoRA must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheDirectPtxLoRAKernelSlow(key);
                }
                if (capturing && !_directPtxLoRAKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX fused-LoRA module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(baseOutput, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(loraA, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(loraB, kernel.Blueprint.Tensors[3]),
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

    internal bool PrewarmDirectPtxFusedLoRA(
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxLoRASemantics(
            batch, inputFeatures, rank, outputFeatures, scaling))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX fused-LoRA prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxLoRAKey(
                    batch, inputFeatures, rank, outputFeatures,
                    PtxCompat.SingleToInt32Bits(scaling));
                if (!_directPtxLoRAKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheDirectPtxLoRAKernelSlow(key);
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

    internal bool TryGetDirectPtxFusedLoRAAudit(
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxLoRAKey(
                batch, inputFeatures, rank, outputFeatures,
                PtxCompat.SingleToInt32Bits(scaling));
            if (_directPtxLoRAKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    // Compatibility surface for the independently-authored standard-layout
    // #836 tests. Both names resolve to the same stricter validated backend.
    internal bool TryDirectPtxFusedLoRAForward(
        IGpuBuffer input,
        IGpuBuffer loraA,
        IGpuBuffer loraB,
        IGpuBuffer baseOutput,
        IGpuBuffer output,
        int m,
        int k,
        int n,
        int rank,
        float scale) =>
        TryDirectPtxFusedLoRA(
            input, baseOutput, loraA, loraB, output,
            m, k, rank, n, scale);

    internal bool TryGetDirectPtxLoRAAudit(
        int m,
        int k,
        int n,
        int rank,
        float scale,
        out DirectPtxKernelAudit audit) =>
        TryGetDirectPtxFusedLoRAAudit(m, k, rank, n, scale, out audit);

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedLoRAKernel CreateAndCacheDirectPtxLoRAKernelSlow(
        DirectPtxLoRAKey key) =>
        _directPtxLoRAKernels.GetOrAdd(
            key, () => new PtxFusedLoRAKernel(
                _directPtxRuntime!, key.Batch, key.InputFeatures, key.Rank,
                key.OutputFeatures, PtxCompat.Int32BitsToSingle(key.ScalingBits)));

    private bool ValidateDirectPtxLoRASemantics(
        int batch,
        int inputFeatures,
        int rank,
        int outputFeatures,
        float scaling)
    {
        if (!PtxFusedLoRAKernel.IsSupportedShape(
            batch, inputFeatures, rank, outputFeatures))
        {
            DirectPtxLastError = "fused-lora-shape-not-implemented";
            return false;
        }
        if (!PtxCompat.IsFinite(scaling))
        {
            DirectPtxLastError = "fused-lora-scaling-not-finite";
            return false;
        }
        return true;
    }
}
