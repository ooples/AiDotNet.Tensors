using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    // Channels-last causal depthwise conv on the device (ICausalDepthwiseConvBackend, all six backends). Records the
    // same single tape node as the CPU op; its backward runs the device kernels too, so training stays resident.
    public override Tensor<T> CausalDepthwiseConv1D<T>(Tensor<T> input, Tensor<T> kernel)
    {
        using var __gpuOp = EnterGpuOp();
        if (Compilation.GraphMode.IsActive || typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || CausalDepthwiseConvResolver.Resolve(backend) is not { } conv
            || input is null || kernel is null || input.Rank < 2 || kernel.Rank != 2
            || kernel._shape[1] != input._shape[input.Rank - 1] || kernel._shape[0] < 1 || input.Length == 0)
            return base.CausalDepthwiseConv1D(input!, kernel!);

        int channels = input._shape[input.Rank - 1], seqLen = input._shape[input.Rank - 2];
        int rows = input.Length / (seqLen * channels), taps = kernel._shape[0];
        try
        {
            using var xB = GetOrAllocateBuffer(backend, input);
            using var wB = GetOrAllocateBuffer(backend, kernel);
            using var yB = AllocateFullyWrittenOutputBuffer(backend, input.Length);
            conv.CausalDepthwiseConv1DForward(xB.Buffer, wB.Buffer, yB.Buffer, rows, seqLen, channels, taps);
            var result = DeferTensorResult<T>(backend, yB.Buffer, input.Length, (int[])input._shape.Clone());
            yB.RelinquishOwnership();
            Autodiff.DifferentiableOps.RecordBinary("CausalDepthwiseConv1D", result, input, kernel,
                CpuEngine.CausalDepthwiseConv1DBackward<T>);
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("CausalDepthwiseConv1D", capabilityGap);
            return base.CausalDepthwiseConv1D(input, kernel);
        }
    }

    public override Tensor<T> CausalDepthwiseConv1DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel)
    {
        using var __gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || CausalDepthwiseConvResolver.Resolve(backend) is not { } conv
            || gradOutput is null || kernel is null || gradOutput.Rank < 2 || kernel.Rank != 2
            || kernel._shape[1] != gradOutput._shape[gradOutput.Rank - 1] || kernel._shape[0] < 1 || gradOutput.Length == 0)
            return base.CausalDepthwiseConv1DBackwardInput(gradOutput!, kernel!);

        int channels = gradOutput._shape[gradOutput.Rank - 1], seqLen = gradOutput._shape[gradOutput.Rank - 2];
        int rows = gradOutput.Length / (seqLen * channels), taps = kernel._shape[0];
        try
        {
            using var gB = GetOrAllocateBuffer(backend, gradOutput);
            using var wB = GetOrAllocateBuffer(backend, kernel);
            using var dxB = AllocateFullyWrittenOutputBuffer(backend, gradOutput.Length);
            conv.CausalDepthwiseConv1DBackwardInput(gB.Buffer, wB.Buffer, dxB.Buffer, rows, seqLen, channels, taps);
            var result = DeferTensorResult<T>(backend, dxB.Buffer, gradOutput.Length, (int[])gradOutput._shape.Clone());
            dxB.RelinquishOwnership();
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("CausalDepthwiseConv1DBackwardInput", capabilityGap);
            return base.CausalDepthwiseConv1DBackwardInput(gradOutput, kernel);
        }
    }

    public override Tensor<T> CausalDepthwiseConv1DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int taps)
    {
        using var __gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend)
            || CausalDepthwiseConvResolver.Resolve(backend) is not { } conv
            || gradOutput is null || input is null || input.Rank < 2 || taps < 1 || input.Length == 0
            || !System.Linq.Enumerable.SequenceEqual(gradOutput._shape, input._shape))
            return base.CausalDepthwiseConv1DBackwardKernel(gradOutput!, input!, taps);

        int channels = input._shape[input.Rank - 1], seqLen = input._shape[input.Rank - 2];
        int rows = input.Length / (seqLen * channels);
        try
        {
            using var gB = GetOrAllocateBuffer(backend, gradOutput);
            using var xB = GetOrAllocateBuffer(backend, input);
            using var dwB = AllocateFullyWrittenOutputBuffer(backend, taps * channels);
            conv.CausalDepthwiseConv1DBackwardKernel(gB.Buffer, xB.Buffer, dwB.Buffer, rows, seqLen, channels, taps);
            var result = DeferTensorResult<T>(backend, dwB.Buffer, taps * channels, new[] { taps, channels });
            dwB.RelinquishOwnership();
            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("CausalDepthwiseConv1DBackwardKernel", capabilityGap);
            return base.CausalDepthwiseConv1DBackwardKernel(gradOutput, input, taps);
        }
    }
}
