// Copyright (c) CrowdTrainer LLC. All rights reserved.
// GPU dispatch for the RoI family (Issue #217 tail). Falls through to
// CpuEngine via inherited base.* when the backend does not implement
// IRoiBackend, T != float, or the shape isn't on the GPU surface
// (4D NCHW input, [K, 5] boxes).

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // Binds the current tape to this engine, so the backward of a device result runs on the device. Under a NoGradScope
    // nothing is recorded, and binding would hand this engine to the backward of a later CPU op on the same tape.
    private void BindTapeIfRecording<T>()
    {
        if (Autodiff.DifferentiableOps.IsRecording<T>())
            Autodiff.GradientTape<T>.Current?.BindEngineIfUnset(this);
    }

    /// <inheritdoc/>
    public override Tensor<T> RoIAlign<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned)
    {
        // Under GraphMode the CpuEngine op records the compiled node; each replay re-enters here outside
        // graph mode and runs the GPU kernel.
        if (Compilation.GraphMode.IsActive) return base.RoIAlign(input, boxes, outputHeight, outputWidth, spatialScale, samplingRatio, aligned);
        var result = RoIAlignOnGpu(input, boxes, outputHeight, outputWidth, spatialScale, samplingRatio, aligned);
        // The CPU op's tape edge, on a result the kernel produced (#1680): without it every input got a zero
        // gradient on the GPU engine. A result the CPU fallback produced already carries the edge.
        if (result.GradFn is null)
        {
            // Binds the tape to this engine, so the backward of a device result runs on the device.
            BindTapeIfRecording<T>();
            Autodiff.DifferentiableOps.RecordBinary("RoIAlign", result, input, boxes, Autodiff.BackwardFunctions<T>.RoIAlignBackward, new object[] { outputHeight, outputWidth, spatialScale, samplingRatio, aligned });
        }
        return result;
    }

    private Tensor<T> RoIAlignOnGpu<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth,
        float spatialScale, int samplingRatio, bool aligned)
    {
        if (typeof(T) == typeof(float)
            && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            using var __gpuOp = EnterGpuOp();
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
        // Under GraphMode the CpuEngine op records the compiled node; each replay re-enters here outside
        // graph mode and runs the GPU kernel.
        if (Compilation.GraphMode.IsActive) return base.PsRoIAlign(input, boxes, outputHeight, outputWidth, outputChannels, spatialScale, samplingRatio);
        var result = PsRoIAlignOnGpu(input, boxes, outputHeight, outputWidth, outputChannels, spatialScale, samplingRatio);
        // The CPU op's tape edge, on a result the kernel produced (#1680): without it every input got a zero
        // gradient on the GPU engine. A result the CPU fallback produced already carries the edge.
        if (result.GradFn is null)
        {
            // Binds the tape to this engine, so the backward of a device result runs on the device.
            BindTapeIfRecording<T>();
            Autodiff.DifferentiableOps.RecordBinary("PsRoIAlign", result, input, boxes, Autodiff.BackwardFunctions<T>.PsRoIAlignBackward, new object[] { outputHeight, outputWidth, outputChannels, spatialScale, samplingRatio });
        }
        return result;
    }

    private Tensor<T> PsRoIAlignOnGpu<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, int outputChannels,
        float spatialScale, int samplingRatio)
    {
        if (typeof(T) == typeof(float) && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            using var __gpuOp = EnterGpuOp();
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
        // Under GraphMode the CpuEngine op records the compiled node; each replay re-enters here outside
        // graph mode and runs the GPU kernel.
        if (Compilation.GraphMode.IsActive) return base.PsRoIPool(input, boxes, outputHeight, outputWidth, outputChannels, spatialScale);
        var result = PsRoIPoolOnGpu(input, boxes, outputHeight, outputWidth, outputChannels, spatialScale);
        // The CPU op's tape edge, on a result the kernel produced (#1680): without it every input got a zero
        // gradient on the GPU engine. A result the CPU fallback produced already carries the edge.
        if (result.GradFn is null)
        {
            // Binds the tape to this engine, so the backward of a device result runs on the device.
            BindTapeIfRecording<T>();
            Autodiff.DifferentiableOps.RecordBinary("PsRoIPool", result, input, boxes, Autodiff.BackwardFunctions<T>.PsRoIPoolBackward, new object[] { outputHeight, outputWidth, outputChannels, spatialScale });
        }
        return result;
    }

    private Tensor<T> PsRoIPoolOnGpu<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, int outputChannels, float spatialScale)
    {
        if (typeof(T) == typeof(float) && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            using var __gpuOp = EnterGpuOp();
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
        // Under GraphMode the CpuEngine op records the compiled node; each replay re-enters here outside
        // graph mode and runs the GPU kernel.
        if (Compilation.GraphMode.IsActive) return base.RoIPool(input, boxes, outputHeight, outputWidth, spatialScale);
        var result = RoIPoolOnGpu(input, boxes, outputHeight, outputWidth, spatialScale);
        // The CPU op's tape edge, on a result the kernel produced (#1680): without it every input got a zero
        // gradient on the GPU engine. A result the CPU fallback produced already carries the edge.
        if (result.GradFn is null)
        {
            // Binds the tape to this engine, so the backward of a device result runs on the device.
            BindTapeIfRecording<T>();
            Autodiff.DifferentiableOps.RecordBinary("RoIPool", result, input, boxes, Autodiff.BackwardFunctions<T>.RoIPoolBackward, new object[] { outputHeight, outputWidth, spatialScale });
        }
        return result;
    }

    private Tensor<T> RoIPoolOnGpu<T>(Tensor<T> input, Tensor<T> boxes,
        int outputHeight, int outputWidth, float spatialScale)
    {
        if (typeof(T) == typeof(float)
            && input.Rank == 4
            && boxes.Rank == 2 && boxes._shape[1] == 5)
        {
            using var __gpuOp = EnterGpuOp();
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
