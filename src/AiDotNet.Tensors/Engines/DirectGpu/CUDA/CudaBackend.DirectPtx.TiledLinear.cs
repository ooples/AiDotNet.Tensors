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

    private readonly record struct DirectPtxFusedLinearTiledKey(
        int M, int K, int N, int Activation, int WeightLayout,
        bool HasBias, int BatchCount);

    /// <summary>
    /// Attempts the general-M tiled fused-linear kernel:
    /// <c>output[M,N] = activation(input[M,K] @ weights[K,N] + bias[N])</c>
    /// for input-major weights, or the equivalent product with prepacked
    /// output-major <c>weights[N,K]</c>.
    /// Fails closed to the established backend for unsupported shapes/architectures,
    /// with an exact reason. The selected physical layout is baked into the module.
    /// </summary>
    internal bool TryDirectPtxFusedLinearTiled(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer bias,
        IGpuBuffer output,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation,
        DirectPtxLinearWeightLayout weightLayout = DirectPtxLinearWeightLayout.OutputMajor)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "fused-linear-tiled-shape-not-implemented";
            return false;
        }
        if (!ValidateDirectPtxLinearSemantics(activation, weightLayout))
            return false;
        if (DirectPtxBufferIsInvalid(input) || DirectPtxBufferIsInvalid(weights) ||
            DirectPtxBufferIsInvalid(bias) || DirectPtxBufferIsInvalid(output))
        {
            DirectPtxLastError = "fused-linear-tiled-null-or-invalid-buffer";
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
        if (DirectPtxBuffersOverlap(output, input) ||
            DirectPtxBuffersOverlap(output, weights) ||
            DirectPtxBuffersOverlap(output, bias))
        {
            DirectPtxLastError = "fused-linear-tiled-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFusedLinearTiledKey(
                m, k, n, (int)activation, (int)weightLayout, true, 1);
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
                _directPtxRuntime!, key.M, key.K, key.N,
                (DirectPtxLinearActivation)key.Activation,
                (DirectPtxLinearWeightLayout)key.WeightLayout,
                key.HasBias, key.BatchCount));

    internal bool PrewarmDirectPtxFusedLinearTiled(
        int m, int k, int n, DirectPtxLinearActivation activation,
        DirectPtxLinearWeightLayout weightLayout = DirectPtxLinearWeightLayout.OutputMajor)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n))
        {
            DirectPtxLastError = "fused-linear-tiled-shape-not-implemented";
            return false;
        }
        if (!ValidateDirectPtxLinearSemantics(activation, weightLayout))
            return false;
        if (!PtxFusedLinearTiledKernel.IsPromotedShape(m, k, n) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "fused-linear-tiled-performance-gate-not-met";
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
                var key = new DirectPtxFusedLinearTiledKey(
                    m, k, n, (int)activation, (int)weightLayout, true, 1);
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
        DirectPtxLinearWeightLayout weightLayout,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFusedLinearTiledKey(
                m, k, n, (int)activation, (int)weightLayout, true, 1);
            if (_directPtxFusedLinearTiledKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    internal bool PrewarmDirectPtxGemmTiled(
        int m,
        int k,
        int n,
        DirectPtxLinearWeightLayout weightLayout,
        int batchCount = 1)
    {
        if (!IsDirectPtxFusedLinearEnabled)
            return false;
        if (!PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n) ||
            batchCount <= 0 || batchCount > 64)
        {
            DirectPtxLastError = "gemm-tiled-shape-not-implemented";
            return false;
        }
        if (!DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "gemm-tiled-performance-gate-not-met";
            return false;
        }
        if (!ValidateDirectPtxLinearWeightLayout(weightLayout))
            return false;
        if (IsStreamCapturing())
        {
            DirectPtxLastError = "Direct PTX GEMM prewarm is not capture-safe.";
            return false;
        }
        try
        {
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxFusedLinearTiledKey(
                    m, k, n, (int)DirectPtxLinearActivation.None,
                    (int)weightLayout, false, batchCount);
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

    internal bool TryGetDirectPtxGemmTiledAudit(
        int m,
        int k,
        int n,
        DirectPtxLinearWeightLayout weightLayout,
        int batchCount,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxFusedLinearTiledKey(
                m, k, n, (int)DirectPtxLinearActivation.None,
                (int)weightLayout, false, batchCount);
            if (_directPtxFusedLinearTiledKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    /// <summary>
    /// Allocates an exact output only when the input-major direct-PTX cell is
    /// eligible. A rejected specialization performs no temporary allocation.
    /// </summary>
    private IGpuBuffer? TryAllocateDirectPtxFusedLinearTiled(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer bias,
        int m,
        int k,
        int n,
        DirectPtxLinearActivation activation)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n) ||
            (!PtxFusedLinearTiledKernel.IsPromotedShape(m, k, n) &&
             !DirectPtxFeatureGate.FusedLinearExperimentOverride))
            return null;

        long inputBytes = checked((long)m * k * sizeof(float));
        long weightBytes = checked((long)k * n * sizeof(float));
        long biasBytes = checked((long)n * sizeof(float));
        if (input is null || weights is null || bias is null ||
            input.SizeInBytes != inputBytes || weights.SizeInBytes != weightBytes ||
            bias.SizeInBytes != biasBytes)
        {
            DirectPtxLastError = "fused-linear-tiled-physical-extent-mismatch";
            return null;
        }

        IGpuBuffer output = AllocateBuffer(checked(m * n));
        if (TryDirectPtxFusedLinearTiled(
            input, weights, bias, output, m, k, n, activation,
            DirectPtxLinearWeightLayout.InputMajor))
            return output;

        output.Dispose();
        return null;
    }

    /// <summary>
    /// Exact FP32 GEMM/batched-GEMM route using the same register/shared tile
    /// as fused linear, but with a three-pointer ABI and no bias/activation.
    /// Alpha=1 and beta=0 are part of this versioned specialization contract.
    /// </summary>
    internal bool TryDirectPtxGemmTiled(
        IGpuBuffer input,
        IGpuBuffer weights,
        IGpuBuffer output,
        int m,
        int k,
        int n,
        DirectPtxLinearWeightLayout weightLayout,
        int batchCount = 1)
    {
        if (!IsDirectPtxFusedLinearEnabled) return false;
        if (!PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n) ||
            batchCount <= 0 || batchCount > 64)
        {
            DirectPtxLastError = "gemm-tiled-shape-not-implemented";
            return false;
        }
        if (!ValidateDirectPtxLinearWeightLayout(weightLayout))
            return false;
        if (DirectPtxBufferIsInvalid(input) || DirectPtxBufferIsInvalid(weights) ||
            DirectPtxBufferIsInvalid(output))
        {
            DirectPtxLastError = "gemm-tiled-null-or-invalid-buffer";
            return false;
        }
        if (!PtxFusedLinearTiledKernel.IsPromotedShape(m, k, n) &&
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
        {
            DirectPtxLastError = "gemm-tiled-performance-gate-not-met";
            return false;
        }

        long inputBytes = checked((long)batchCount * m * k * sizeof(float));
        long weightBytes = checked((long)batchCount * k * n * sizeof(float));
        long outputBytes = checked((long)batchCount * m * n * sizeof(float));
        if (input.SizeInBytes != inputBytes || weights.SizeInBytes != weightBytes ||
            output.SizeInBytes != outputBytes)
        {
            DirectPtxLastError = "gemm-tiled-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(output, input) ||
            DirectPtxBuffersOverlap(output, weights))
        {
            DirectPtxLastError = "gemm-tiled-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxFusedLinearTiledKey(
                m, k, n, (int)DirectPtxLinearActivation.None,
                (int)weightLayout, false, batchCount);
            lock (_directPtxLock)
            {
                if (!_directPtxFusedLinearTiledKernels.TryGetValue(
                    key, out PtxFusedLinearTiledKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX GEMM must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheFusedLinearTiledKernelSlow(key);
                }
                if (capturing && !_directPtxFusedLinearTiledKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX GEMM module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.LaunchGemm(
                        DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weights, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
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

    private bool ValidateDirectPtxLinearSemantics(
        DirectPtxLinearActivation activation,
        DirectPtxLinearWeightLayout weightLayout)
    {
        if ((int)activation < (int)DirectPtxLinearActivation.None ||
            (int)activation > (int)DirectPtxLinearActivation.LeakyRelu)
        {
            DirectPtxLastError = "fused-linear-activation-not-implemented";
            return false;
        }
        return ValidateDirectPtxLinearWeightLayout(weightLayout);
    }

    private bool ValidateDirectPtxLinearWeightLayout(
        DirectPtxLinearWeightLayout weightLayout)
    {
        if (weightLayout != DirectPtxLinearWeightLayout.InputMajor &&
            weightLayout != DirectPtxLinearWeightLayout.OutputMajor)
        {
            DirectPtxLastError = "linear-weight-layout-not-implemented";
            return false;
        }
        return true;
    }
}
