using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxCrossEntropyKey, PtxFusedLinearCrossEntropyKernel>
        _directPtxCrossEntropyKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private readonly record struct DirectPtxCrossEntropyKey(
        int TargetKind, int Rows, int HiddenDimension, int Vocabulary);

    internal bool TryDirectPtxFusedLinearCrossEntropy(
        DirectPtxCrossEntropyTarget targetKind,
        IGpuBuffer hidden,
        IGpuBuffer weight,
        IGpuBuffer bias,
        IGpuBuffer target,
        IGpuBuffer meanLoss,
        int rows,
        int hiddenDimension,
        int vocabulary)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxCrossEntropySemantics(
            targetKind, rows, hiddenDimension, vocabulary))
            return false;
        if (DirectPtxBufferIsInvalid(hidden) || DirectPtxBufferIsInvalid(weight) ||
            DirectPtxBufferIsInvalid(bias) || DirectPtxBufferIsInvalid(target) ||
            DirectPtxBufferIsInvalid(meanLoss))
        {
            DirectPtxLastError = "fused-linear-ce-null-or-invalid-buffer";
            return false;
        }
        bool exactIndex = PtxFusedLinearCrossEntropyKernel.IsExactIndexCell(
            targetKind, rows, hiddenDimension, vocabulary);
        if (GpuDeterminism.IsActive && !exactIndex)
        {
            DirectPtxLastError =
                "Direct PTX fused linear/CE uses an atomic row reduction and is disabled in deterministic mode.";
            return false;
        }

        long hiddenBytes = checked((long)rows * hiddenDimension * sizeof(float));
        long weightBytes = checked((long)hiddenDimension * vocabulary * sizeof(float));
        long biasBytes = checked((long)vocabulary * sizeof(float));
        long targetBytes = checked((long)(targetKind == DirectPtxCrossEntropyTarget.Index
            ? rows : checked(rows * vocabulary)) * sizeof(float));
        if (hidden.SizeInBytes != hiddenBytes || weight.SizeInBytes != weightBytes ||
            bias.SizeInBytes != biasBytes || target.SizeInBytes != targetBytes ||
            meanLoss.SizeInBytes != sizeof(float))
        {
            DirectPtxLastError = "fused-linear-ce-physical-extent-mismatch";
            return false;
        }
        if (DirectPtxBuffersOverlap(meanLoss, hidden) ||
            DirectPtxBuffersOverlap(meanLoss, weight) ||
            DirectPtxBuffersOverlap(meanLoss, bias) ||
            DirectPtxBuffersOverlap(meanLoss, target))
        {
            DirectPtxLastError = "fused-linear-ce-output-alias-not-supported";
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxCrossEntropyKey(
                (int)targetKind, rows, hiddenDimension, vocabulary);
            lock (_directPtxLock)
            {
                if (!_directPtxCrossEntropyKernels.TryGetValue(
                    key, out PtxFusedLinearCrossEntropyKernel? kernel))
                {
                    if (capturing)
                    {
                        DirectPtxLastError =
                            "Direct PTX fused linear/CE must be prewarmed before CUDA graph capture.";
                        return false;
                    }
                    _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                    kernel = CreateAndCacheDirectPtxCrossEntropyKernelSlow(key);
                }
                if (capturing && !_directPtxCrossEntropyKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX fused linear/CE module for CUDA graph capture.");

                // Generic row kernels atomically accumulate into one scalar.
                // The exact B4/K16/V32 index cell performs one deterministic
                // final store and therefore needs neither an atomic nor pre-clear.
                if (!exactIndex)
                    MemsetBuffer(meanLoss, 0, sizeof(float));
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(hidden, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(weight, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(bias, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(target, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(meanLoss, kernel.Blueprint.Tensors[4]));
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

    internal bool PrewarmDirectPtxFusedLinearCrossEntropy(
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary)
    {
        if (!IsDirectPtxFusedLinearEnabled ||
            !DirectPtxFeatureGate.FusedLinearExperimentOverride)
            return false;
        if (!ValidateDirectPtxCrossEntropySemantics(
            targetKind, rows, hiddenDimension, vocabulary))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX fused linear/CE prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                var key = new DirectPtxCrossEntropyKey(
                    (int)targetKind, rows, hiddenDimension, vocabulary);
                if (!_directPtxCrossEntropyKernels.TryGetValue(key, out _))
                    _ = CreateAndCacheDirectPtxCrossEntropyKernelSlow(key);
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

    internal bool TryGetDirectPtxFusedLinearCrossEntropyAudit(
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary,
        out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxCrossEntropyKey(
                (int)targetKind, rows, hiddenDimension, vocabulary);
            if (_directPtxCrossEntropyKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private bool ValidateDirectPtxCrossEntropySemantics(
        DirectPtxCrossEntropyTarget targetKind,
        int rows,
        int hiddenDimension,
        int vocabulary)
    {
        if (targetKind != DirectPtxCrossEntropyTarget.Index &&
            targetKind != DirectPtxCrossEntropyTarget.Dense)
        {
            DirectPtxLastError = "fused-linear-ce-target-not-implemented";
            return false;
        }
        if (rows <= 0 || rows > 65_535 ||
            hiddenDimension <= 0 || hiddenDimension > 65_536 ||
            vocabulary <= 0 || vocabulary > 65_536)
        {
            DirectPtxLastError = "fused-linear-ce-shape-not-implemented";
            return false;
        }
        try
        {
            _ = checked(rows * hiddenDimension);
            _ = checked(hiddenDimension * vocabulary);
            _ = checked(rows * vocabulary);
        }
        catch (OverflowException)
        {
            DirectPtxLastError = "fused-linear-ce-shape-overflow";
            return false;
        }
        return true;
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedLinearCrossEntropyKernel CreateAndCacheDirectPtxCrossEntropyKernelSlow(
        DirectPtxCrossEntropyKey key) =>
        _directPtxCrossEntropyKernels.GetOrAdd(
            key, () => new PtxFusedLinearCrossEntropyKernel(
                _directPtxRuntime!, (DirectPtxCrossEntropyTarget)key.TargetKind,
                key.Rows, key.HiddenDimension, key.Vocabulary));
}
