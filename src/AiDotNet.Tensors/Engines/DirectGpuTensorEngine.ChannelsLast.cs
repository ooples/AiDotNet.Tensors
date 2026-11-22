// Copyright (c) CrowdTrainer LLC. All rights reserved.
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>
    /// Whether this engine's backend runs convolutions faster on channels-last activations
    /// (<see cref="IChannelsLastBackend.PrefersChannelsLast"/>): a compiled plan then keeps a convolution stack's
    /// activations channels-last between its layers. Under an FP16/BF16 autocast scope whose convolutions run
    /// channels-last (cuDNN's 16-bit tensor-core kernels read NHWC natively) it does too: a channels-last activation then
    /// needs only a cast at each convolution, not a cast and a transpose.
    /// </summary>
    internal bool PrefersChannelsLast
        => TryGetBackend(out var backend) && backend is IChannelsLastBackend channelsLast
           && (channelsLast.PrefersChannelsLast || AutocastRunsChannelsLast(backend));

    /// <summary>Whether the backend takes a channels-last convolution stack of these channel counts.</summary>
    internal bool AcceptsChannelsLast(int channels, int outChannels)
        => TryGetBackend(out var backend) && backend is IChannelsLastBackend channelsLast
           && channelsLast.AcceptsChannelsLast(channels, outChannels);

    /// <summary>
    /// One convolution pass with a layout per activation (<see cref="IChannelsLastBackend.TryConvolutionChannelsLast"/>):
    /// a new tensor of logical shape <paramref name="outputShape"/> holding the pass's output (its layout the pass's for
    /// that operand), or null -- nothing launched -- when the backend does not take it. The geometry is the convolution's
    /// (for a transposed convolution, the convolution it transposes).
    /// </summary>
    internal Tensor<float>? TryConvolutionChannelsLast(int pass, Tensor<float> a, Tensor<float> b, int[] outputShape,
        in ChannelsLastConvGeometry g, bool inputNhwc, bool outputNhwc)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IChannelsLastBackend channelsLast
            || !a.IsContiguous || !b.IsContiguous)
            return null;
        long length = 1;
        foreach (int d in outputShape) length *= d;
        if (length <= 0 || length > int.MaxValue) return null;
        using var bufA = GetOrAllocateBuffer(backend, a);
        using var bufB = GetOrAllocateBuffer(backend, b);
        var bufC = AllocateFullyWrittenOutputBuffer(backend, (int)length);
        // Under autocast the pass runs in 16 bits, its activations cast in their layout (no transpose for a
        // channels-last one); a pass the 16-bit route declines runs the FP32 channels-last pass below.
        if (TryAutocastConvolutionChannelsLast(backend, pass, bufA.Buffer, bufB.Buffer, bufC.Buffer, g, inputNhwc,
                outputNhwc))
            return DeferTensorResult<float>(backend, bufC.Buffer, (int)length, outputShape);
        if (!channelsLast.TryConvolutionChannelsLast(pass, bufA.Buffer, bufB.Buffer, bufC.Buffer, g.Batch, g.Channels,
                g.Height, g.Width, g.OutChannels, g.KernelH, g.KernelW, g.OutHeight, g.OutWidth, g.PadH, g.PadW, g.StrideH,
                g.StrideW, g.DilationH, g.DilationW, inputNhwc, outputNhwc))
        {
            bufC.Dispose();
            return null;
        }
        return DeferTensorResult<float>(backend, bufC.Buffer, (int)length, outputShape);
    }

    /// <summary>
    /// Training BatchNorm and its activation over a channels-last activation of logical shape [n, c, h, w]
    /// (<see cref="IChannelsLastBackend.TryBatchNormChannelsLast"/>): the output (channels-last) and the batch mean and
    /// biased variance, or null when the backend does not take it.
    /// </summary>
    internal Tensor<float>? TryBatchNormChannelsLast(Tensor<float> input, Tensor<float> gamma, Tensor<float> beta,
        double epsilon, int act, float slope, out Tensor<float>? mean, out Tensor<float>? variance)
    {
        mean = variance = null;
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IChannelsLastBackend channelsLast || input.Rank != 4
            || !input.IsContiguous || gamma.Length != input.Shape._dims[1] || beta.Length != gamma.Length)
            return null;
        int channels = input.Shape._dims[1], rows = input.Length / channels;
        using var bufX = GetOrAllocateBuffer(backend, input);
        using var bufG = GetOrAllocateBuffer(backend, gamma);
        using var bufB = GetOrAllocateBuffer(backend, beta);
        var bufY = AllocateFullyWrittenOutputBuffer(backend, input.Length);
        var bufM = AllocateFullyWrittenOutputBuffer(backend, channels);
        var bufV = AllocateFullyWrittenOutputBuffer(backend, channels);
        if (!channelsLast.TryBatchNormChannelsLast(bufX.Buffer, bufY.Buffer, bufG.Buffer, bufB.Buffer, bufM.Buffer,
                bufV.Buffer, rows, channels, (float)epsilon, act, slope))
        {
            bufY.Dispose(); bufM.Dispose(); bufV.Dispose();
            return null;
        }
        mean = DeferTensorResult<float>(backend, bufM.Buffer, channels, new[] { channels });
        variance = DeferTensorResult<float>(backend, bufV.Buffer, channels, new[] { channels });
        return DeferTensorResult<float>(backend, bufY.Buffer, input.Length, input.Shape.ToArray());
    }

    /// <summary>
    /// The channels-last BatchNorm's backward with the activation's mask from its output
    /// (<see cref="IChannelsLastBackend.TryBatchNormChannelsLastBackward"/>): the input gradient (channels-last) and the
    /// gamma and beta gradients, or null when the backend does not take it.
    /// </summary>
    internal Tensor<float>? TryBatchNormChannelsLastBackward(Tensor<float> gradOutput, Tensor<float>? activationOutput,
        Tensor<float> input, Tensor<float> gamma, Tensor<float> mean, Tensor<float> variance, double epsilon, int act,
        float slope, out Tensor<float>? gradGamma, out Tensor<float>? gradBeta)
    {
        gradGamma = gradBeta = null;
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IChannelsLastBackend channelsLast || input.Rank != 4
            || gradOutput.Length != input.Length || (act != 0 && (activationOutput is null || activationOutput.Length != input.Length)))
            return null;
        int channels = input.Shape._dims[1], rows = input.Length / channels;
        using var bufDy = GetOrAllocateBuffer(backend, gradOutput);
        using var bufY = act != 0 ? GetOrAllocateBuffer(backend, activationOutput!) : default;
        using var bufX = GetOrAllocateBuffer(backend, input);
        using var bufG = GetOrAllocateBuffer(backend, gamma);
        using var bufM = GetOrAllocateBuffer(backend, mean);
        using var bufV = GetOrAllocateBuffer(backend, variance);
        var bufDx = AllocateFullyWrittenOutputBuffer(backend, input.Length);
        var bufDg = AllocateFullyWrittenOutputBuffer(backend, channels);
        var bufDb = AllocateFullyWrittenOutputBuffer(backend, channels);
        if (!channelsLast.TryBatchNormChannelsLastBackward(bufDy.Buffer, act != 0 ? bufY.Buffer : null, null, bufX.Buffer,
                bufG.Buffer, bufM.Buffer, bufV.Buffer, bufDx.Buffer, bufDg.Buffer, bufDb.Buffer, rows, channels,
                (float)epsilon, act != 0 ? 1 : 0, act, slope, accumulate: false))
        {
            bufDx.Dispose(); bufDg.Dispose(); bufDb.Dispose();
            return null;
        }
        gradGamma = DeferTensorResult<float>(backend, bufDg.Buffer, channels, gamma.Shape.ToArray());
        gradBeta = DeferTensorResult<float>(backend, bufDb.Buffer, channels, gamma.Shape.ToArray());
        return DeferTensorResult<float>(backend, bufDx.Buffer, input.Length, input.Shape.ToArray());
    }

    /// <summary>
    /// A rank-4 activation's layout change, NCHW to channels-last (<paramref name="toChannelsLast"/>) or back
    /// (<see cref="IChannelsLastBackend.TryChangeLayout"/>): a new tensor of the same logical shape, or null.
    /// </summary>
    internal Tensor<float>? TryChangeLayout(Tensor<float> input, bool toChannelsLast)
    {
        using var __gpuOp = EnterGpuOp();
        if (!TryGetBackend(out var backend) || backend is not IChannelsLastBackend channelsLast || input.Rank != 4
            || !input.IsContiguous)
            return null;
        var d = input.Shape._dims;
        using var bufX = GetOrAllocateBuffer(backend, input);
        var bufY = AllocateFullyWrittenOutputBuffer(backend, input.Length);
        if (!channelsLast.TryChangeLayout(bufX.Buffer, bufY.Buffer, d[0], d[1], d[2] * d[3], toChannelsLast))
        {
            bufY.Dispose();
            return null;
        }
        return DeferTensorResult<float>(backend, bufY.Buffer, input.Length, input.Shape.ToArray());
    }
}

/// <summary>A convolution's geometry for <see cref="DirectGpuTensorEngine.TryConvolutionChannelsLast"/>: x [batch,
/// channels, height, width], w [outChannels, channels, kernelH, kernelW], y [batch, outChannels, outHeight, outWidth].</summary>
internal readonly struct ChannelsLastConvGeometry
{
    internal readonly int Batch, Channels, Height, Width, OutChannels, KernelH, KernelW, OutHeight, OutWidth;
    internal readonly int PadH, PadW, StrideH, StrideW, DilationH, DilationW;

    internal ChannelsLastConvGeometry(int batch, int channels, int height, int width, int outChannels, int kernelH,
        int kernelW, int outHeight, int outWidth, int padH, int padW, int strideH, int strideW, int dilationH, int dilationW)
    {
        Batch = batch; Channels = channels; Height = height; Width = width; OutChannels = outChannels;
        KernelH = kernelH; KernelW = kernelW; OutHeight = outHeight; OutWidth = outWidth;
        PadH = padH; PadW = padW; StrideH = strideH; StrideW = strideW; DilationH = dilationH; DilationW = dilationW;
    }

    internal int[] InputShape => new[] { Batch, Channels, Height, Width };
    internal int[] OutputShape => new[] { Batch, OutChannels, OutHeight, OutWidth };
    internal int[] FilterShape => new[] { OutChannels, Channels, KernelH, KernelW };
}
