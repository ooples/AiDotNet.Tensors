// Copyright (c) AiDotNet. All rights reserved.
// GPU dispatch for the RoI family (Issue #217 tail). Falls through to
// CpuEngine via inherited base.* when the backend does not implement
// IRoiBackend, T != float, or the shape isn't on the GPU surface
// (4D NCHW input, [K, 5] boxes).

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> RoIAlign<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned)
    {
        if (typeof(T) == typeof(float)
            && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            if (TryGetBackend(out var backend) && backend is IRoiBackend roi)
            {
                int N = input._shape[0], C = input._shape[1];
                int H = input._shape[2], W = input._shape[3];
                int K = boxes._shape[0];
                if (K == 0) return new Tensor<T>(new[] { 0, C, outputHeight, outputWidth });
                int outLen = CheckedOutLen(K, C, outputHeight, outputWidth);
                using var inBuf = GetOrAllocateBuffer(backend, input);
                using var bBuf = GetOrAllocateBuffer(backend, boxes);
                var outBuf = AllocateOutputBuffer(backend, outLen);
                try
                {
                    roi.RoIAlign(inBuf.Buffer, bBuf.Buffer, outBuf.Buffer,
                        N, C, H, W, K, outputHeight, outputWidth,
                        spatialScale, samplingRatio, aligned);
                    var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                    return new Tensor<T>(arr, new[] { K, C, outputHeight, outputWidth });
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.RoIAlign(input, boxes, outputHeight, outputWidth, spatialScale, samplingRatio, aligned);
    }

    /// <summary>
    /// Compute an output-element count in checked long arithmetic so a
    /// pathologically large shape can't wrap around to a small positive
    /// int and allocate an undersized buffer.
    /// </summary>
    private static int CheckedOutLen(int a, int b, int c, int d)
    {
        long n = checked((long)a * b * c * d);
        if (n > int.MaxValue)
            throw new OverflowException(
                $"RoI output element count {n} exceeds Int32.MaxValue — " +
                "split the K dimension or reduce output size.");
        return (int)n;
    }

    private static int CheckedOutLen(int a, int b, int c, int d, int e)
    {
        long n = checked((long)a * b * c * d * e);
        if (n > int.MaxValue)
            throw new OverflowException(
                $"RoI output element count {n} exceeds Int32.MaxValue.");
        return (int)n;
    }

    /// <inheritdoc/>
    public override Tensor<T> PsRoIAlign<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, int outputChannels,
        float spatialScale, int samplingRatio)
    {
        if (typeof(T) == typeof(float) && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            if (TryGetBackend(out var backend) && backend is IRoiBackend roi)
            {
                int N = input._shape[0], C = input._shape[1];
                int H = input._shape[2], W = input._shape[3];
                int K = boxes._shape[0];
                if (K == 0) return new Tensor<T>(new[] { 0, outputChannels, outputHeight, outputWidth });
                int outLen = CheckedOutLen(K, outputChannels, outputHeight, outputWidth);
                using var inBuf = GetOrAllocateBuffer(backend, input);
                using var bBuf = GetOrAllocateBuffer(backend, boxes);
                var outBuf = AllocateOutputBuffer(backend, outLen);
                try
                {
                    roi.PsRoIAlign(inBuf.Buffer, bBuf.Buffer, outBuf.Buffer,
                        N, C, H, W, K, outputHeight, outputWidth, outputChannels,
                        spatialScale, samplingRatio);
                    var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                    return new Tensor<T>(arr, new[] { K, outputChannels, outputHeight, outputWidth });
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.PsRoIAlign(input, boxes, outputHeight, outputWidth, outputChannels, spatialScale, samplingRatio);
    }

    /// <inheritdoc/>
    public override Tensor<T> PsRoIPool<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, int outputChannels, float spatialScale)
    {
        if (typeof(T) == typeof(float) && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            if (TryGetBackend(out var backend) && backend is IRoiBackend roi)
            {
                int N = input._shape[0], C = input._shape[1];
                int H = input._shape[2], W = input._shape[3];
                int K = boxes._shape[0];
                if (K == 0) return new Tensor<T>(new[] { 0, outputChannels, outputHeight, outputWidth });
                int outLen = CheckedOutLen(K, outputChannels, outputHeight, outputWidth);
                using var inBuf = GetOrAllocateBuffer(backend, input);
                using var bBuf = GetOrAllocateBuffer(backend, boxes);
                var outBuf = AllocateOutputBuffer(backend, outLen);
                try
                {
                    roi.PsRoIPool(inBuf.Buffer, bBuf.Buffer, outBuf.Buffer,
                        N, C, H, W, K, outputHeight, outputWidth, outputChannels, spatialScale);
                    var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                    return new Tensor<T>(arr, new[] { K, outputChannels, outputHeight, outputWidth });
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.PsRoIPool(input, boxes, outputHeight, outputWidth, outputChannels, spatialScale);
    }

    /// <inheritdoc/>
    public override Tensor<T> RoIPool<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, float spatialScale)
    {
        if (typeof(T) == typeof(float)
            && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            if (TryGetBackend(out var backend) && backend is IRoiBackend roi)
            {
                int N = input._shape[0], C = input._shape[1];
                int H = input._shape[2], W = input._shape[3];
                int K = boxes._shape[0];
                if (K == 0) return new Tensor<T>(new[] { 0, C, outputHeight, outputWidth });
                int outLen = CheckedOutLen(K, C, outputHeight, outputWidth);
                using var inBuf = GetOrAllocateBuffer(backend, input);
                using var bBuf = GetOrAllocateBuffer(backend, boxes);
                var outBuf = AllocateOutputBuffer(backend, outLen);
                try
                {
                    roi.RoIPool(inBuf.Buffer, bBuf.Buffer, outBuf.Buffer,
                        N, C, H, W, K, outputHeight, outputWidth, spatialScale);
                    var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                    return new Tensor<T>(arr, new[] { K, C, outputHeight, outputWidth });
                }
                catch { outBuf.Dispose(); throw; }
            }
        }
        return base.RoIPool(input, boxes, outputHeight, outputWidth, spatialScale);
    }
}
