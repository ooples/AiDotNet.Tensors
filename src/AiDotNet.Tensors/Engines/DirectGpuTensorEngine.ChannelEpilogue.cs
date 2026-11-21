// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// A convolution output's per-channel epilogue in one launch (<see cref="AiDotNet.Tensors.Engines.DirectGpu.IChannelEpilogueBackend"/>):
    /// <c>act(((z + bias[c]) * scale[c]) + shift[c])</c> over an NCHW <paramref name="z"/>, each of bias, scale and shift
    /// optional (any shape holding the channel count's values: a rank-1 [C] bias, an inference BatchNorm's [1, C, 1, 1]
    /// scale and shift). <paramref name="activation"/> is a <see cref="AiDotNet.Tensors.Engines.DirectGpu.ChannelEpilogueActivation"/> code. Null --
    /// nothing launched -- when the backend or shape does not take it; the caller runs the separate ops. Records nothing:
    /// the compiled plan's fusion supplies the backward.
    /// </summary>
    internal Tensor<float>? TryChannelEpilogue(Tensor<float> z, Tensor<float>? bias, Tensor<float>? scale,
        Tensor<float>? shift, int activation, float slope)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not AiDotNet.Tensors.Engines.DirectGpu.IChannelEpilogueBackend epilogue
            || !TryChannelLayout(z, bias, scale, shift, out int batch, out int channels, out int spatial)
            || !AiDotNet.Tensors.Engines.DirectGpu.ChannelEpilogueActivation.Accepts(activation, slope))
            return null;
        using var bufZ = GetOrAllocateBuffer(backend, z);
        using var bufBias = bias is null ? default : GetOrAllocateBuffer(backend, bias);
        using var bufScale = scale is null ? default : GetOrAllocateBuffer(backend, scale);
        using var bufShift = shift is null ? default : GetOrAllocateBuffer(backend, shift);
        var bufOut = AllocateFullyWrittenOutputBuffer(backend, z.Length);
        if (!epilogue.TryChannelEpilogue(bufZ.Buffer, bias is null ? null : bufBias.Buffer,
                scale is null ? null : bufScale.Buffer, shift is null ? null : bufShift.Buffer, bufOut.Buffer,
                batch, channels, spatial, activation, slope))
        {
            bufOut.Dispose();
            return null;
        }
        return DeferTensorResult<float>(backend, bufOut.Buffer, z.Length, z.Shape.ToArray());
    }

    /// <summary>
    /// The epilogue's backward to z in one launch: <c>act'(output) * gradOutput * scale[c]</c>, the mask from the
    /// forward's <paramref name="output"/>. Null -- nothing launched -- as for the forward.
    /// </summary>
    internal Tensor<float>? TryChannelEpilogueBackward(Tensor<float> gradOutput, Tensor<float> output,
        Tensor<float>? scale, int activation, float slope)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not AiDotNet.Tensors.Engines.DirectGpu.IChannelEpilogueBackend epilogue
            || gradOutput.Length != output.Length
            || !TryChannelLayout(output, null, scale, null, out int batch, out int channels, out int spatial)
            || !AiDotNet.Tensors.Engines.DirectGpu.ChannelEpilogueActivation.Accepts(activation, slope))
            return null;
        using var bufDy = GetOrAllocateBuffer(backend, gradOutput);
        using var bufY = GetOrAllocateBuffer(backend, output);
        using var bufScale = scale is null ? default : GetOrAllocateBuffer(backend, scale);
        var bufDz = AllocateFullyWrittenOutputBuffer(backend, output.Length);
        if (!epilogue.TryChannelEpilogueBackward(bufDy.Buffer, bufY.Buffer, scale is null ? null : bufScale.Buffer,
                bufDz.Buffer, batch, channels, spatial, activation, slope))
        {
            bufDz.Dispose();
            return null;
        }
        return DeferTensorResult<float>(backend, bufDz.Buffer, output.Length, output.Shape.ToArray());
    }

    // A contiguous rank-4 NCHW activation and per-channel operands of exactly C values each.
    private static bool TryChannelLayout(Tensor<float> z, Tensor<float>? bias, Tensor<float>? scale, Tensor<float>? shift,
        out int batch, out int channels, out int spatial)
    {
        batch = channels = spatial = 0;
        if (z.Rank != 4 || !z.IsContiguous) return false;
        batch = z.Shape._dims[0];
        channels = z.Shape._dims[1];
        spatial = z.Shape._dims[2] * z.Shape._dims[3];
        if (batch <= 0 || channels <= 0 || spatial <= 0) return false;
        return (bias is null || bias.Length == channels && bias.IsContiguous)
               && (scale is null || scale.Length == channels && scale.IsContiguous)
               && (shift is null || shift.Length == channels && shift.IsContiguous);
    }
}
