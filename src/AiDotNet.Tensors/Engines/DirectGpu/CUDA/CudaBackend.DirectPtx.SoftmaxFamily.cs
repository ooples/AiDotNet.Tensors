using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Backend dispatch for the remaining softmax-family direct-PTX kernels (issue #840):
/// log-sum-exp and its backward, softmax backward, masked fill and its backward, Taylor
/// softmax, and sparsemax. Every entry point fails closed to the established NVRTC kernel
/// for unsupported shapes/architectures or until a shape is GPU-validated and promoted.
/// </summary>
public sealed partial class CudaBackend
{
    private readonly DirectPtxKernelCache<DirectPtxSoftmaxKey, PtxLogSumExpKernel>
        _directPtxLogSumExpKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxSoftmaxKey, PtxLogSumExpBackwardKernel>
        _directPtxLogSumExpBackwardKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxSoftmaxKey, PtxSoftmaxBackwardKernel>
        _directPtxSoftmaxBackwardKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxMaskedFillBackwardKey, PtxMaskedFillBackwardKernel>
        _directPtxMaskedFillBackwardKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxSoftmaxKey, PtxTaylorSoftmaxKernel>
        _directPtxTaylorSoftmaxKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxSoftmaxKey, PtxSparsemaxKernel>
        _directPtxSparsemaxKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));
    private readonly DirectPtxKernelCache<DirectPtxMaskedFillKey, PtxMaskedFillKernel>
        _directPtxMaskedFillKernels = new(Math.Max(4, DirectPtxFeatureGate.CacheCapacity / 2));

    private readonly record struct DirectPtxMaskedFillKey(int Count, int FillBits);
    private readonly record struct DirectPtxMaskedFillBackwardKey(int Count);

    internal long DirectPtxLogSumExpDispatchCount => System.Threading.Interlocked.Read(ref _directPtxLogSumExpDispatchCount);
    internal long DirectPtxLogSumExpBackwardDispatchCount => System.Threading.Interlocked.Read(ref _directPtxLogSumExpBackwardDispatchCount);
    internal long DirectPtxSoftmaxBackwardDispatchCount => System.Threading.Interlocked.Read(ref _directPtxSoftmaxBackwardDispatchCount);
    internal long DirectPtxTaylorSoftmaxDispatchCount => System.Threading.Interlocked.Read(ref _directPtxTaylorSoftmaxDispatchCount);
    internal long DirectPtxSparsemaxDispatchCount => System.Threading.Interlocked.Read(ref _directPtxSparsemaxDispatchCount);
    internal long DirectPtxMaskedFillDispatchCount => System.Threading.Interlocked.Read(ref _directPtxMaskedFillDispatchCount);
    internal long DirectPtxMaskedFillBackwardDispatchCount => System.Threading.Interlocked.Read(ref _directPtxMaskedFillBackwardDispatchCount);

    private bool SoftmaxFamilyGateOpen =>
        IsDirectPtxSoftmaxEnabled && DirectPtxFeatureGate.SoftmaxExperimentOverride;

    // ---- Log-sum-exp: input[M,N] -> output[M] ----
    internal bool TryDirectPtxLogSumExp(IGpuBuffer input, IGpuBuffer output, int m, int n)
    {
        if (!SoftmaxFamilyGateOpen || !PtxLogSumExpKernel.IsSupportedShape(m, n))
        { DirectPtxLastError = "logsumexp-not-eligible"; return false; }
        if (input.SizeInBytes != checked((long)m * n * sizeof(float)) ||
            output.SizeInBytes != checked((long)m * sizeof(float)))
        { DirectPtxLastError = "logsumexp-physical-extent-mismatch"; return false; }
        return Dispatch2(() =>
        {
            var key = new DirectPtxSoftmaxKey(m, n);
            var kernel = _directPtxLogSumExpKernels.GetOrAdd(key, () => new PtxLogSumExpKernel(_directPtxRuntime!, m, n));
            lock (GpuDispatchLock)
                kernel.Launch(
                    DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
        }, ref _directPtxLogSumExpDispatchCount);
    }
    private long _directPtxLogSumExpDispatchCount;

    // ---- Log-sum-exp backward: input[M,N], grad[M] -> output[M,N] ----
    internal bool TryDirectPtxLogSumExpBackward(IGpuBuffer input, IGpuBuffer grad, IGpuBuffer output, int m, int n)
    {
        if (!SoftmaxFamilyGateOpen || !PtxLogSumExpBackwardKernel.IsSupportedShape(m, n))
        { DirectPtxLastError = "logsumexp-backward-not-eligible"; return false; }
        if (input.SizeInBytes != checked((long)m * n * sizeof(float)) ||
            grad.SizeInBytes != checked((long)m * sizeof(float)) ||
            output.SizeInBytes != checked((long)m * n * sizeof(float)))
        { DirectPtxLastError = "logsumexp-backward-physical-extent-mismatch"; return false; }
        return Dispatch2(() =>
        {
            var key = new DirectPtxSoftmaxKey(m, n);
            var kernel = _directPtxLogSumExpBackwardKernels.GetOrAdd(key, () => new PtxLogSumExpBackwardKernel(_directPtxRuntime!, m, n));
            lock (GpuDispatchLock)
                kernel.Launch(
                    DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.Create(grad, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
        }, ref _directPtxLogSumExpBackwardDispatchCount);
    }
    private long _directPtxLogSumExpBackwardDispatchCount;

    // ---- Softmax backward: softmax[M,N], grad[M,N] -> output[M,N] ----
    internal bool TryDirectPtxSoftmaxBackward(IGpuBuffer softmax, IGpuBuffer grad, IGpuBuffer output, int m, int n)
    {
        if (!SoftmaxFamilyGateOpen || !PtxSoftmaxBackwardKernel.IsSupportedShape(m, n))
        { DirectPtxLastError = "softmax-backward-not-eligible"; return false; }
        long bytes = checked((long)m * n * sizeof(float));
        if (softmax.SizeInBytes != bytes || grad.SizeInBytes != bytes || output.SizeInBytes != bytes)
        { DirectPtxLastError = "softmax-backward-physical-extent-mismatch"; return false; }
        return Dispatch2(() =>
        {
            var key = new DirectPtxSoftmaxKey(m, n);
            var kernel = _directPtxSoftmaxBackwardKernels.GetOrAdd(key, () => new PtxSoftmaxBackwardKernel(_directPtxRuntime!, m, n));
            lock (GpuDispatchLock)
                kernel.Launch(
                    DirectPtxTensorView.Create(softmax, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.Create(grad, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
        }, ref _directPtxSoftmaxBackwardDispatchCount);
    }
    private long _directPtxSoftmaxBackwardDispatchCount;

    // ---- Taylor softmax: input[M,N] -> output[M,N] ----
    internal bool TryDirectPtxTaylorSoftmax(IGpuBuffer input, IGpuBuffer output, int m, int n)
    {
        if (!SoftmaxFamilyGateOpen || !PtxTaylorSoftmaxKernel.IsSupportedShape(m, n))
        { DirectPtxLastError = "taylor-softmax-not-eligible"; return false; }
        long bytes = checked((long)m * n * sizeof(float));
        if (input.SizeInBytes != bytes || output.SizeInBytes != bytes)
        { DirectPtxLastError = "taylor-softmax-physical-extent-mismatch"; return false; }
        return Dispatch2(() =>
        {
            var key = new DirectPtxSoftmaxKey(m, n);
            var kernel = _directPtxTaylorSoftmaxKernels.GetOrAdd(key, () => new PtxTaylorSoftmaxKernel(_directPtxRuntime!, m, n));
            lock (GpuDispatchLock)
                kernel.Launch(
                    DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
        }, ref _directPtxTaylorSoftmaxDispatchCount);
    }
    private long _directPtxTaylorSoftmaxDispatchCount;

    // ---- Sparsemax: input[M,N] -> output[M,N] ----
    internal bool TryDirectPtxSparsemax(IGpuBuffer input, IGpuBuffer output, int m, int n)
    {
        if (!SoftmaxFamilyGateOpen || !PtxSparsemaxKernel.IsSupportedShape(m, n))
        { DirectPtxLastError = "sparsemax-not-eligible"; return false; }
        long bytes = checked((long)m * n * sizeof(float));
        if (input.SizeInBytes != bytes || output.SizeInBytes != bytes)
        { DirectPtxLastError = "sparsemax-physical-extent-mismatch"; return false; }
        return Dispatch2(() =>
        {
            var key = new DirectPtxSoftmaxKey(m, n);
            var kernel = _directPtxSparsemaxKernels.GetOrAdd(key, () => new PtxSparsemaxKernel(_directPtxRuntime!, m, n));
            lock (GpuDispatchLock)
                kernel.Launch(
                    DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[1]));
        }, ref _directPtxSparsemaxDispatchCount);
    }
    private long _directPtxSparsemaxDispatchCount;

    // ---- Masked fill: input[count], mask[count], fill -> output[count] (flat elementwise) ----
    internal bool TryDirectPtxMaskedFill(IGpuBuffer input, IGpuBuffer mask, IGpuBuffer output, int count, float fill)
    {
        if (!SoftmaxFamilyGateOpen || !PtxMaskedFillKernel.IsSupportedCount(count))
        { DirectPtxLastError = "masked-fill-not-eligible"; return false; }
        long bytes = checked((long)count * sizeof(float));
        if (input.SizeInBytes != bytes || mask.SizeInBytes != bytes || output.SizeInBytes != bytes)
        { DirectPtxLastError = "masked-fill-physical-extent-mismatch"; return false; }
        return Dispatch2(() =>
        {
            var key = new DirectPtxMaskedFillKey(count, BitConverter.ToInt32(BitConverter.GetBytes(fill), 0));
            var kernel = _directPtxMaskedFillKernels.GetOrAdd(key, () => new PtxMaskedFillKernel(_directPtxRuntime!, count, fill));
            lock (GpuDispatchLock)
                kernel.Launch(
                    DirectPtxTensorView.Create(input, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.Create(mask, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
        }, ref _directPtxMaskedFillDispatchCount);
    }
    private long _directPtxMaskedFillDispatchCount;

    // ---- Masked fill backward: grad[count], mask[count] -> output[count] (flat elementwise) ----
    internal bool TryDirectPtxMaskedFillBackward(IGpuBuffer grad, IGpuBuffer mask, IGpuBuffer output, int count)
    {
        if (!SoftmaxFamilyGateOpen || !PtxMaskedFillBackwardKernel.IsSupportedCount(count))
        { DirectPtxLastError = "masked-fill-backward-not-eligible"; return false; }
        long bytes = checked((long)count * sizeof(float));
        if (grad.SizeInBytes != bytes || mask.SizeInBytes != bytes || output.SizeInBytes != bytes)
        { DirectPtxLastError = "masked-fill-backward-physical-extent-mismatch"; return false; }
        return Dispatch2(() =>
        {
            var key = new DirectPtxMaskedFillBackwardKey(count);
            var kernel = _directPtxMaskedFillBackwardKernels.GetOrAdd(key, () => new PtxMaskedFillBackwardKernel(_directPtxRuntime!, count));
            lock (GpuDispatchLock)
                kernel.Launch(
                    DirectPtxTensorView.Create(grad, kernel.Blueprint.Tensors[0]),
                    DirectPtxTensorView.Create(mask, kernel.Blueprint.Tensors[1]),
                    DirectPtxTensorView.Create(output, kernel.Blueprint.Tensors[2]));
        }, ref _directPtxMaskedFillBackwardDispatchCount);
    }
    private long _directPtxMaskedFillBackwardDispatchCount;

    // Shared eligibility/dispatch shell: capture-safe context, cache locking, error trapping.
    private bool Dispatch2(Action launch, ref long counter)
    {
        try
        {
            if (IsStreamCapturing())
            {
                DirectPtxLastError = "Direct PTX softmax-family kernels must be prewarmed before CUDA graph capture.";
                return false;
            }
            EnsureContextCurrent();
            lock (_directPtxLock)
            {
                _directPtxRuntime ??= new DirectPtxRuntime(_cudaContext, _stream);
                launch();
            }
            System.Threading.Interlocked.Increment(ref counter);
            DirectPtxLastError = null;
            return true;
        }
        catch (Exception ex)
        {
            DirectPtxLastError = $"{ex.GetType().Name}: {ex.Message}";
            return false;
        }
    }
}
