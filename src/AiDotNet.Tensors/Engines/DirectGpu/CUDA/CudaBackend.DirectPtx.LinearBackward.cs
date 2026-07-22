using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxLinearBackwardKey, PtxFusedLinearBackwardKernel>
        _directPtxLinearBackwardKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private readonly record struct DirectPtxLinearBackwardKey(
        int M, int K, int N, int Activation);

    internal bool TryDirectPtxFusedLinearBackward(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer saved,
        IGpuBuffer gradInput,
        IGpuBuffer gradWeight,
        IGpuBuffer gradBias,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxLinearBackwardSemantics(m, k, n, activation))
            return false;
        if (DirectPtxBufferIsInvalid(gradOutput) || DirectPtxBufferIsInvalid(input) ||
            DirectPtxBufferIsInvalid(weights) || DirectPtxBufferIsInvalid(saved) ||
            DirectPtxBufferIsInvalid(gradInput) || DirectPtxBufferIsInvalid(gradWeight) ||
            DirectPtxBufferIsInvalid(gradBias))
        {
            DirectPtxLastError = "fused-linear-backward-null-or-invalid-buffer";
            return false;
        }

        long inputBytes = checked((long)m * k * sizeof(float));
        long outputBytes = checked((long)m * n * sizeof(float));
        long weightBytes = checked((long)k * n * sizeof(float));
        long biasBytes = checked((long)n * sizeof(float));
        if (gradOutput.SizeInBytes != outputBytes || input.SizeInBytes != inputBytes ||
            weights.SizeInBytes != weightBytes || saved.SizeInBytes != outputBytes ||
            gradInput.SizeInBytes != inputBytes || gradWeight.SizeInBytes != weightBytes ||
            gradBias.SizeInBytes != biasBytes)
        {
            DirectPtxLastError = "fused-linear-backward-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBackwardOutputsOverlap(
            gradOutput, input, weights, saved, gradInput, gradWeight, gradBias))
        {
            DirectPtxLastError = "fused-linear-backward-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxLinearBackwardKey(m, k, n, (int)activation);
            lock (_directPtxLock)
            {
                if (!_directPtxLinearBackwardKernels.TryGetValue(
                    key, out PtxFusedLinearBackwardKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX fused-linear backward must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheDirectPtxLinearBackwardKernelSlow(key);
                }
                if (capturing && !_directPtxLinearBackwardKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX fused-linear-backward module for capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(gradOutput, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(saved, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(gradInput, kernel.Blueprint.Tensors[4]),
                        DirectPtxTensorView.Create(gradWeight, kernel.Blueprint.Tensors[5]),
                        DirectPtxTensorView.Create(gradBias, kernel.Blueprint.Tensors[6]));
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

    internal bool PrewarmDirectPtxFusedLinearBackward(
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxLinearBackwardSemantics(m, k, n, activation))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX fused-linear-backward prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxLinearBackwardKey(m, k, n, (int)activation);
                if (!_directPtxLinearBackwardKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheDirectPtxLinearBackwardKernelSlow(key);
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

    internal bool TryGetDirectPtxFusedLinearBackwardAudits(
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation,
        out IReadOnlyList<DirectPtxKernelAudit> audits)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxLinearBackwardKey(m, k, n, (int)activation);
            if (_directPtxLinearBackwardKernels.TryGetValue(key, out var kernel))
            {
                audits = kernel.Audits;
                return true;
            }
        }
        audits = Array.Empty<DirectPtxKernelAudit>();
        return false;
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedLinearBackwardKernel CreateAndCacheDirectPtxLinearBackwardKernelSlow(
        DirectPtxLinearBackwardKey key) =>
        _directPtxLinearBackwardKernels.GetOrAdd(
            key, () => new PtxFusedLinearBackwardKernel(
                _directPtxRuntime!, key.M, key.K, key.N,
                (DirectPtxLinearActivation)key.Activation));

    private bool ValidateDirectPtxLinearBackwardSemantics(
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        if (!PtxFusedLinearBackwardKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "fused-linear-backward-shape-not-implemented";
            return false;
        }
        if (activation != DirectPtxLinearActivation.Relu &&
            activation != DirectPtxLinearActivation.GeluTanh &&
            activation != DirectPtxLinearActivation.Sigmoid &&
            activation != DirectPtxLinearActivation.Tanh &&
            activation != DirectPtxLinearActivation.Swish)
        {
            DirectPtxLastError = "fused-linear-backward-activation-not-implemented";
            return false;
        }
        return true;
    }

    private static bool DirectPtxBackwardOutputsOverlap(
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer saved,
        IGpuBuffer gradInput,
        IGpuBuffer gradWeight,
        IGpuBuffer gradBias)
    {
        return OverlapsInputs(gradInput) || OverlapsInputs(gradWeight) ||
            OverlapsInputs(gradBias) || DirectPtxBuffersOverlap(gradInput, gradWeight) ||
            DirectPtxBuffersOverlap(gradInput, gradBias) ||
            DirectPtxBuffersOverlap(gradWeight, gradBias);

        bool OverlapsInputs(IGpuBuffer output) =>
            DirectPtxBuffersOverlap(output, gradOutput) ||
            DirectPtxBuffersOverlap(output, input) ||
            DirectPtxBuffersOverlap(output, weights) ||
            DirectPtxBuffersOverlap(output, saved);
    }
}
