#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly bool _directPtxRecurrentStateOptedIn =
        DirectPtxFeatureGate.IsRecurrentStateEnabled;
    private readonly DirectPtxKernelCache<DirectPtxRgLruKey, PtxFusedRgLruScan128x256Kernel>
        _directPtxRgLruKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private long _directPtxRgLruDispatchCount;

    internal long DirectPtxRgLruDispatchCount =>
        System.Threading.Interlocked.Read(ref _directPtxRgLruDispatchCount);
    internal int DirectPtxRgLruKernelCapacity => _directPtxRgLruKernels.Capacity;
    internal int DirectPtxRgLruPinnedKernelCount
    {
        get { lock (_directPtxLock) return _directPtxRgLruKernels.PinnedCount; }
    }

    internal bool IsDirectPtxRgLruEnabled =>
        _directPtxRecurrentStateOptedIn && IsAvailable &&
        DirectPtxArchitecture.HasExperimentalRgLruScan(_ccMajor, _ccMinor);

    /// <summary>
    /// Attempts the exact FP32 [1,128,256] channel-persistent RG-LRU scan.
    /// Shape, extent, layout, alignment, and alias contracts are admitted once;
    /// the emitted PTX has no dynamic shape, layout, stride, or loop branch.
    /// </summary>
    internal bool TryDirectPtxRgLruScanForward(
        IGpuBuffer value,
        IGpuBuffer recurrenceGate,
        IGpuBuffer inputGate,
        IGpuBuffer decay,
        IGpuBuffer output,
        int batch,
        int sequenceLength,
        int recurrentDimension)
    {
        if (!ValidateDirectPtxRgLruEligibility(batch, sequenceLength, recurrentDimension))
            return false;
        if (value is null || recurrenceGate is null || inputGate is null ||
            decay is null || output is null)
        {
            DirectPtxLastError = "rglru-null-buffer";
            return false;
        }

        DirectPtxEligibilityResult bufferResult =
            DirectPtxRecurrentEligibility.EvaluateBuffers(
                new DirectPtxRgLruBufferRequest(
                    (nuint)value.Handle, value.SizeInBytes,
                    (nuint)recurrenceGate.Handle, recurrenceGate.SizeInBytes,
                    (nuint)inputGate.Handle, inputGate.SizeInBytes,
                    (nuint)decay.Handle, decay.SizeInBytes,
                    (nuint)output.Handle, output.SizeInBytes));
        if (!bufferResult.IsEligible)
        {
            DirectPtxLastError = bufferResult.Reason;
            return false;
        }

        try
        {
            bool capturing = IsStreamCapturing();
            EnsureContextCurrent();
            var key = new DirectPtxRgLruKey(batch, sequenceLength, recurrentDimension);
            lock (_directPtxLock)
            {
                if (capturing && !_directPtxRgLruKernels.TryGetValue(key, out _))
                {
                    DirectPtxLastError =
                        "Direct PTX RG-LRU must be prewarmed before CUDA graph capture.";
                    return false;
                }
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                PtxFusedRgLruScan128x256Kernel kernel = GetOrCreateRgLruKernel(key);
                if (capturing && !_directPtxRgLruKernels.Pin(key))
                    throw new InvalidOperationException(
                        "Could not pin the direct-PTX RG-LRU module for CUDA graph capture.");
                lock (GpuDispatchLock)
                    kernel.Launch(
                        DirectPtxTensorView.Create(value, kernel.Blueprint.Tensors[0]),
                        DirectPtxTensorView.Create(recurrenceGate, kernel.Blueprint.Tensors[1]),
                        DirectPtxTensorView.Create(inputGate, kernel.Blueprint.Tensors[2]),
                        DirectPtxTensorView.Create(decay, kernel.Blueprint.Tensors[3]),
                        DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[4]));
            }
            System.Threading.Interlocked.Increment(ref _directPtxRgLruDispatchCount);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }

    internal bool PrewarmDirectPtxRgLruScan(
        int batch = PtxFusedRgLruScan128x256Kernel.Batch,
        int sequenceLength = PtxFusedRgLruScan128x256Kernel.SequenceLength,
        int recurrentDimension = PtxFusedRgLruScan128x256Kernel.RecurrentDimension)
    {
        if (!ValidateDirectPtxRgLruEligibility(batch, sequenceLength, recurrentDimension))
            return false;
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX RG-LRU prewarm is not capture-safe.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                _ = GetOrCreateRgLruKernel(
                    new DirectPtxRgLruKey(batch, sequenceLength, recurrentDimension));
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

    internal bool TryGetDirectPtxRgLruAudit(out DirectPtxKernelAudit audit)
    {
        lock (_directPtxLock)
        {
            var key = new DirectPtxRgLruKey(
                PtxFusedRgLruScan128x256Kernel.Batch,
                PtxFusedRgLruScan128x256Kernel.SequenceLength,
                PtxFusedRgLruScan128x256Kernel.RecurrentDimension);
            if (_directPtxRgLruKernels.TryGetValue(key, out var kernel))
            {
                audit = kernel.Audit;
                return true;
            }
        }
        audit = null!;
        return false;
    }

    private bool ValidateDirectPtxRgLruEligibility(
        int batch,
        int sequenceLength,
        int recurrentDimension)
    {
        if (!_directPtxRecurrentStateOptedIn)
        {
            DirectPtxLastError = "rglru-feature-disabled";
            return false;
        }
        if (!IsAvailable)
        {
            DirectPtxLastError = "rglru-backend-unavailable";
            return false;
        }
        DirectPtxEligibilityResult result = DirectPtxRecurrentEligibility.Evaluate(
            new DirectPtxRgLruRequest(
                _ccMajor, _ccMinor,
                DirectPtxPhysicalType.Float32,
                DirectPtxPhysicalLayout.BatchSequenceFeature,
                batch, sequenceLength, recurrentDimension,
                IsTraining: false));
        if (!result.IsEligible)
        {
            DirectPtxLastError = result.Reason;
            return false;
        }
        return true;
    }

    private PtxFusedRgLruScan128x256Kernel GetOrCreateRgLruKernel(DirectPtxRgLruKey key)
    {
        if (_directPtxRgLruKernels.TryGetValue(
            key, out PtxFusedRgLruScan128x256Kernel? existing))
            return existing;
        return CreateAndCacheRgLruKernelSlow(key);
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private PtxFusedRgLruScan128x256Kernel CreateAndCacheRgLruKernelSlow(
        DirectPtxRgLruKey key) =>
        _directPtxRgLruKernels.GetOrAdd(key, () =>
            new PtxFusedRgLruScan128x256Kernel(_directPtxRuntime!));

    private readonly record struct DirectPtxRgLruKey(
        int Batch,
        int SequenceLength,
        int RecurrentDimension);
}
#endif
