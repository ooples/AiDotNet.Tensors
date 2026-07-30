// Copyright (c) AiDotNet. All rights reserved.
// Metal GPU backend - Convolution and Pooling operations.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend
{
    // Shared dispatch for the conv/pool MSL kernels (issue #646): bind the buffers (indices 0..n-1) and the int
    // params (as setBytes at indices n..), then launch a 1D grid over `threads` output elements.
    // NOTE: validated only on a Metal runner — the dev box has no Apple GPU.
    private bool TryDispatchConvPoolMetal(System.IntPtr library, string libName, string kernelName,
        int threads, IGpuBuffer[] buffers, int[] intParams)
    {
        if (threads <= 0) return true;
        if (library == System.IntPtr.Zero)
            throw new InvalidOperationException($"Metal {libName} kernels are unavailable.");
        foreach (var b in buffers)
        {
            if (b is not MetalGpuBuffer)
                throw new ArgumentException("Buffers must be MetalGpuBuffer.", nameof(buffers));
        }
        var pipeline = GetPipeline(libName, library, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(threads);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < intParams.Length; i++) encoder.SetBytes((uint)intParams[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
        return true;
    }

    private static void ValidateDeformableGroups(int inChannels, int groups, int deformGroups)
    {
        if (groups <= 0 || deformGroups <= 0 || inChannels % groups != 0 || inChannels % deformGroups != 0)
            throw new ArgumentException("Groups and deformable groups must be positive divisors of the input channels.");
    }

    #region Convolution Operations

    /// <summary>
    /// 2D Convolution forward pass.
    /// </summary>
    public void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv2d_direct",
                batch * outChannels * outHeight * outWidth, new[] { input, kernel, output },
                new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// 2D Convolution backward for input gradients.
    /// </summary>
    public void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv2d_backward_input",
                batch * inChannels * inHeight * inWidth, new[] { gradOutput, kernel, gradInput },
                new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// 2D Convolution backward for kernel gradients.
    /// </summary>
    public void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv2d_backward_weights",
                outChannels * inChannels * kernelH * kernelW, new[] { input, gradOutput, gradKernel },
                new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    public void Conv1D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
        => Conv2D(input, kernel, output, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);

    public void Conv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
        => Conv2DBackwardInput(gradOutput, kernel, gradInput, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);

    public void Conv1DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
        => Conv2DBackwardKernel(input, gradOutput, gradKernel, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);

    public void Unfold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        if (strideH <= 0 || strideW <= 0) throw new ArgumentException("Stride must be positive.");
        try
        {
            int oH = (height + 2 * padH - kernelH) / strideH + 1;
            int oW = (width + 2 * padW - kernelW) / strideW + 1;
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "unfold",
                batch * channels * kernelH * kernelW * oH * oW, new[] { input, output },
                new[] { batch, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    public void Fold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int outputH, int outputW,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        if (strideH <= 0 || strideW <= 0) throw new ArgumentException("Stride must be positive.");
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "fold",
                batch * channels * outputH * outputW, new[] { input, output },
                new[] { batch, channels, outputH, outputW, kernelH, kernelW, strideH, strideW, padH, padW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// 3D Convolution forward pass.
    /// </summary>
    public void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv3d_direct",
                batch * outChannels * outDepth * outHeight * outWidth, new[] { input, kernel, output },
                new[] { batch, inChannels, inDepth, inHeight, inWidth, outChannels, outDepth, outHeight, outWidth, kernelD, kernelH, kernelW, strideD, strideH, strideW, padD, padH, padW, dilationD, dilationH, dilationW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Depthwise 2D convolution.
    /// </summary>
    public void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "depthwise_conv2d",
                batch * channels * outHeight * outWidth, new[] { input, kernel, output },
                new[] { batch, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Transposed 2D convolution.
    /// </summary>
    public void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        ThrowIfDisposed();
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv_transpose2d",
            checked(batch * outChannels * outHeight * outWidth), new[] { input, kernel, output },
            new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW, padH, padW, outputPadH, outputPadW });
    }

    /// <summary>
    /// ConvTranspose2D backward for input gradients.
    /// </summary>
    public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv_transpose2d_backward_input",
                batch * inChannels * inHeight * inWidth, new[] { gradOutput, kernel, gradInput },
                new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// ConvTranspose2D backward for kernel gradients.
    /// </summary>
    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "conv_transpose2d_backward_weights",
                inChannels * outChannels * kernelH * kernelW, new[] { input, gradOutput, gradKernel },
                new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Locally connected 2D convolution.
    /// </summary>
    public void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        ThrowIfDisposed();
        using var zeroBias = bias is null ? AllocateBuffer(outChannels) : null;
        if (zeroBias is not null)
            Fill(zeroBias, 0f, outChannels);
        IGpuBuffer effectiveBias = bias ?? zeroBias!;
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "locally_connected_conv2d",
            checked(batch * outChannels * outHeight * outWidth), new[] { input, weights, effectiveBias, output },
            new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW });
    }

    /// <summary>
    /// LocallyConnectedConv2D backward for input.
    /// </summary>
    public void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "locally_connected_conv2d_backward_input",
                batch * inChannels * inHeight * inWidth, new[] { gradOutput, weights, gradInput },
                new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// LocallyConnectedConv2D backward for weights.
    /// </summary>
    public void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "locally_connected_conv2d_backward_weights",
                outHeight * outWidth * outChannels * inChannels * kernelH * kernelW, new[] { gradOutput, input, gradWeights },
                new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// LocallyConnectedConv2D backward for bias.
    /// </summary>
    public void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "locally_connected_conv2d_backward_bias",
                outChannels, new[] { gradOutput, gradBias }, new[] { batch, outChannels, outHeight, outWidth }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Deformable 2D convolution (DCNv1/v2).
    /// </summary>
    public void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        ValidateDeformableGroups(inChannels, groups, deformGroups);
        int maskSize = checked(batch * deformGroups * kernelH * kernelW * outHeight * outWidth);
        using var unitMask = mask is null ? AllocateBuffer(maskSize) : null;
        if (unitMask is not null)
            Fill(unitMask, 1f, maskSize);
        IGpuBuffer effectiveMask = mask ?? unitMask!;
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "deformable_conv2d",
            checked(batch * outChannels * outHeight * outWidth),
            new[] { input, weights, offsets, effectiveMask, output },
            new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW, padH, padW, dilationH, dilationW, groups, deformGroups });
    }

    /// <summary>
    /// Deformable Conv2D backward for input.
    /// </summary>
    public void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        ValidateDeformableGroups(inChannels, groups, deformGroups);
        int maskSize = checked(batch * deformGroups * kernelH * kernelW * outHeight * outWidth);
        using var unitMask = mask is null ? AllocateBuffer(maskSize) : null;
        if (unitMask is not null)
            Fill(unitMask, 1f, maskSize);
        IGpuBuffer effectiveMask = mask ?? unitMask!;
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "deformable_conv2d_backward_input",
            checked(batch * inChannels * inHeight * inWidth),
            new[] { gradOutput, weights, offsets, effectiveMask, gradInput },
            new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW, padH, padW, dilationH, dilationW, groups, deformGroups });
    }

    /// <summary>
    /// Deformable Conv2D backward for weights.
    /// </summary>
    public void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        ValidateDeformableGroups(inChannels, groups, deformGroups);
        int maskSize = checked(batch * deformGroups * kernelH * kernelW * outHeight * outWidth);
        using var unitMask = mask is null ? AllocateBuffer(maskSize) : null;
        if (unitMask is not null)
            Fill(unitMask, 1f, maskSize);
        IGpuBuffer effectiveMask = mask ?? unitMask!;
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "deformable_conv2d_backward_weights",
            checked(outChannels * (inChannels / groups) * kernelH * kernelW),
            new[] { gradOutput, input, offsets, effectiveMask, gradWeights },
            new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW, padH, padW, dilationH, dilationW, groups, deformGroups });
    }

    /// <summary>
    /// Deformable Conv2D backward for offsets.
    /// </summary>
    public void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        ValidateDeformableGroups(inChannels, groups, deformGroups);
        int maskSize = checked(batch * deformGroups * kernelH * kernelW * outHeight * outWidth);
        using var unitMask = mask is null ? AllocateBuffer(maskSize) : null;
        if (unitMask is not null)
            Fill(unitMask, 1f, maskSize);
        IGpuBuffer effectiveMask = mask ?? unitMask!;
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "deformable_conv2d_backward_offset",
            checked(batch * deformGroups * 2 * kernelH * kernelW * outHeight * outWidth),
            new[] { gradOutput, input, weights, offsets, effectiveMask, gradOffsets },
            new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW, padH, padW, dilationH, dilationW, groups, deformGroups });
    }

    /// <summary>
    /// Deformable Conv2D backward for mask (DCNv2).
    /// </summary>
    public void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        ThrowIfDisposed();
        try
        {
            int ksG = kernelH * kernelW;
            if (groups > 0 && deformGroups > 0
                && inChannels % groups == 0 && inChannels % deformGroups == 0
                && TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "deformable_conv2d_backward_mask",
                    batch * deformGroups * ksG * outHeight * outWidth, new[] { gradOutput, input, weights, offsets, gradMask },
                    new[] { batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW, groups, deformGroups }))
                return;
        }
        catch
        {
            throw;
        }
    }

    #endregion

    #region Pooling Operations

    /// <summary>
    /// 2D Max pooling.
    /// </summary>
    public void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        int outputSize = checked(batch * channels * outHeight * outWidth);
        using var unusedIndices = indices is null ? AllocateIntBuffer(outputSize) : null;
        IGpuBuffer effectiveIndices = indices ?? unusedIndices!;
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "maxpool2d", outputSize,
            new[] { input, output, effectiveIndices },
            new[] { batch, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW,
                strideH, strideW, padH, padW });
    }

    /// <summary>
    /// 2D Max pooling backward.
    /// </summary>
    public void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "maxpool2d_backward",
                batch * channels * inHeight * inWidth, new[] { gradOutput, indices, gradInput },
                new[] { batch, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// 2D Average pooling.
    /// </summary>
    public void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "avgpool2d",
                batch * channels * outHeight * outWidth, new[] { input, output },
                new[] { batch, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad ? 1 : 0 }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// 2D Average pooling backward.
    /// </summary>
    public void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "avgpool2d_backward",
                batch * channels * inHeight * inWidth, new[] { gradOutput, gradInput },
                new[] { batch, channels, inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad ? 1 : 0 }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Global average pooling.
    /// </summary>
    public void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "global_avgpool2d",
                batch * channels, new[] { input, output }, new[] { batch, channels, height, width }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Global max pooling.
    /// </summary>
    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "global_maxpool2d_noidx",
                batch * channels, new[] { input, output }, new[] { batch, channels, height, width }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Global max pooling with indices.
    /// </summary>
    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "global_maxpool2d",
                batch * channels, new[] { input, output, indices }, new[] { batch, channels, height, width }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Global average pooling backward.
    /// </summary>
    public void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "global_avgpool2d_backward",
                batch * channels * height * width, new[] { gradOutput, gradInput }, new[] { batch, channels, height, width }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Global max pooling backward.
    /// </summary>
    public void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "global_maxpool2d_backward",
                batch * channels * height * width, new[] { gradOutput, indices, gradInput }, new[] { batch, channels, height, width }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Adaptive average pooling.
    /// </summary>
    public void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "adaptive_avgpool2d",
                batch * channels * outHeight * outWidth, new[] { input, output },
                new[] { batch, channels, inHeight, inWidth, outHeight, outWidth }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// 3D Max pooling.
    /// </summary>
    public void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        ThrowIfDisposed();
        int outputSize = checked(batch * channels * outDepth * outHeight * outWidth);
        using var unusedIndices = indices is null ? AllocateIntBuffer(outputSize) : null;
        IGpuBuffer effectiveIndices = indices ?? unusedIndices!;
        TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "maxpool3d", outputSize,
            new[] { input, output, effectiveIndices },
            new[] { batch, channels, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth,
                kernelD, kernelH, kernelW, strideD, strideH, strideW });
    }

    /// <summary>
    /// 3D Max pooling backward.
    /// </summary>
    public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        ThrowIfDisposed();
        try
        {
            if (TryDispatchConvPoolMetal(_convolutionLibrary, "Convolution", "maxpool3d_backward",
                batch * channels * inDepth * inHeight * inWidth, new[] { gradOutput, indices, gradInput },
                new[] { batch, channels, inDepth, inHeight, inWidth, outDepth, outHeight, outWidth }))
                return;
        }
        catch
        {
            throw;
        }
    }

    /// <summary>
    /// Bilinear sample from NCHW buffer at fractional (h, w) position.
    /// </summary>
    private static float BilinearSample(float[] data, int b, int c, float h, float w,
        int height, int width, int channels)
    {
        int h0 = (int)MathF.Floor(h);
        int w0 = (int)MathF.Floor(w);
        int h1 = h0 + 1;
        int w1 = w0 + 1;
        float lh = h - h0;
        float lw = w - w0;

        float v00 = (h0 >= 0 && h0 < height && w0 >= 0 && w0 < width)
            ? data[((b * channels + c) * height + h0) * width + w0] : 0f;
        float v01 = (h0 >= 0 && h0 < height && w1 >= 0 && w1 < width)
            ? data[((b * channels + c) * height + h0) * width + w1] : 0f;
        float v10 = (h1 >= 0 && h1 < height && w0 >= 0 && w0 < width)
            ? data[((b * channels + c) * height + h1) * width + w0] : 0f;
        float v11 = (h1 >= 0 && h1 < height && w1 >= 0 && w1 < width)
            ? data[((b * channels + c) * height + h1) * width + w1] : 0f;

        return (1 - lh) * (1 - lw) * v00 + (1 - lh) * lw * v01 +
               lh * (1 - lw) * v10 + lh * lw * v11;
    }

    #endregion
}
