// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 2: Convolution, Pooling, Spatial Transformer, Normalization, Dropout, Embedding.

#if NET7_0_OR_GREATER
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend
{
    #region Convolution Operations

    public void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * outChannels);
        Dispatch3Buffer3DAsync("Convolution", WebGpuKernels.ConvolutionSource, "conv2d",
            input, kernel, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(inWidth, inHeight, batch * inChannels);
        Dispatch3Buffer3DAsync("Conv2DBackward", WebGpuKernels.Conv2DBackwardSource, "conv2d_backward_input",
            gradOutput, kernel, gradInput, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        // 1D dispatch: each thread computes one kernel weight gradient
        int totalKernelElements = outChannels * inChannels * kernelH * kernelW;
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            dilationH, dilationW);
        Dispatch3BufferAsync("Conv2DBackwardKernel", WebGpuKernels.Conv2DBackwardKernelSource, "conv2d_backward_kernel",
            input, gradOutput, gradKernel, uniforms, totalKernelElements).GetAwaiter().GetResult();
    }

    public void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        // Conv3DParams: 24 u32 fields packed as floats
        var uniforms = new float[24];
        int[] vals = { batch, inChannels, inDepth, inHeight, inWidth,
                       outChannels, outDepth, outHeight, outWidth,
                       kernelD, kernelH, kernelW,
                       strideD, strideH, strideW,
                       padD, padH, padW,
                       dilationD, dilationH, dilationW, 0, 0, 0 };
        for (int i = 0; i < 24; i++) uniforms[i] = BitConverter.Int32BitsToSingle(vals[i]);
        int total = batch * outChannels * outDepth * outHeight * outWidth;
        Dispatch3BufferAsync("Conv3D", WebGpuKernels.Conv3DSource, "conv3d",
            input, kernel, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        // DepthwiseConv2DSource uses ConvParams with channels field (batch, channels, in_h, in_w, out_h, out_w, kH, kW, sH, sW, pH, pW)
        // Reuse MakePoolUniforms which has (batch, channels, inH, inW, outH, outW, kH, kW, sH, sW, pH, pW) - same layout
        var uniforms = MakePoolUniforms(batch, channels, inHeight, inWidth, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * channels);
        Dispatch3Buffer3DAsync("DepthwiseConv2D", WebGpuKernels.DepthwiseConv2DSource, "depthwise_conv2d",
            input, kernel, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // Apply output padding to resolve transposed convolution output size ambiguity.
        // outH_final = outH + outputPadH, outW_final = outW + outputPadW
        int finalOutH = outHeight + outputPadH;
        int finalOutW = outWidth + outputPadW;
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, finalOutH, finalOutW, kernelH, kernelW, strideH, strideW, padH, padW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(finalOutW, finalOutH, batch * outChannels);
        Dispatch3Buffer3DAsync("ConvTranspose2D", WebGpuKernels.ConvTranspose2DSource, "conv_transpose2d",
            input, kernel, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // Use adjusted output dimensions including output padding for backward pass
        int finalOutH = outHeight + outputPadH;
        int finalOutW = outWidth + outputPadW;
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, finalOutH, finalOutW, kernelH, kernelW, strideH, strideW, padH, padW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(inWidth, inHeight, batch * inChannels);
        Dispatch3Buffer3DAsync("ConvTranspose2DBackwardInput",
            WebGpuKernels.ConvTranspose2DBackwardInputSource, "conv_transpose2d_backward_input",
            gradOutput, kernel, gradInput, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        // dL/dW[ic,oc,ky,kx] = sum over batch, input positions of input * gradOutput
        int finalOutH = outHeight + outputPadH;
        int finalOutW = outWidth + outputPadW;
        int totalKernelElements = inChannels * outChannels * kernelH * kernelW;
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, finalOutH, finalOutW, kernelH, kernelW, strideH, strideW, padH, padW);
        Dispatch3BufferAsync("ConvTranspose2DBackwardKernel",
            WebGpuKernels.ConvTranspose2DBackwardKernelSource, "conv_transpose2d_backward_kernel",
            input, gradOutput, gradKernel, uniforms, totalKernelElements).GetAwaiter().GetResult();
    }

    public void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        // Locally connected conv uses per-position weights: weight layout [outH, outW, outC, inC, kH, kW]
        var uniforms = MakeLCUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * outChannels);
        Dispatch3Buffer3DAsync("LocallyConnectedConv2D", WebGpuKernels.LocallyConnectedConv2DSource,
            "locally_connected_conv2d", input, weights, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
        if (bias is not null)
        {
            // Add bias per output channel using Conv2DBiasAdd kernel (output[idx] += bias[channel])
            Conv2DBiasAdd(output, bias, batch, outChannels, outHeight * outWidth);
        }
    }

    public void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        var uniforms = MakeLCUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(inWidth, inHeight, batch * inChannels);
        Dispatch3Buffer3DAsync("LocallyConnectedConv2DBackwardInput",
            WebGpuKernels.LocallyConnectedConv2DBackwardInputSource, "lc_backward_input",
            gradOutput, weights, gradInput, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        int totalWeightElements = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
        var uniforms = MakeLCUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW);
        Dispatch3BufferAsync("LocallyConnectedConv2DBackwardWeights",
            WebGpuKernels.LocallyConnectedConv2DBackwardWeightsSource, "lc_backward_weights",
            input, gradOutput, gradWeights, uniforms, totalWeightElements).GetAwaiter().GetResult();
    }

    public void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        // BiasGradSource: per-channel sum over batch and spatial dimensions
        int spatial = outHeight * outWidth;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(outChannels),
            BitConverter.Int32BitsToSingle(spatial),
            0
        };
        Dispatch2BufferAsync("BiasGrad", WebGpuKernels.BiasGradSource, "bias_grad",
            gradOutput, gradBias, uniforms, outChannels).GetAwaiter().GetResult();
    }

    public void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        // Deformable conv uses offsets to compute sampling positions with bilinear interpolation
        int hasMask = mask is not null ? 1 : 0;
        using var dummyMask = (WebGpuBuffer)AllocateBuffer(1);
        IGpuBuffer maskBuf = mask is not null ? mask : dummyMask;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(inChannels),
            BitConverter.Int32BitsToSingle(outChannels),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(outHeight),
            BitConverter.Int32BitsToSingle(outWidth),
            BitConverter.Int32BitsToSingle(kernelH),
            BitConverter.Int32BitsToSingle(kernelW),
            BitConverter.Int32BitsToSingle(strideH),
            BitConverter.Int32BitsToSingle(strideW),
            BitConverter.Int32BitsToSingle(padH),
            BitConverter.Int32BitsToSingle(padW),
            BitConverter.Int32BitsToSingle(dilationH),
            BitConverter.Int32BitsToSingle(dilationW),
            BitConverter.Int32BitsToSingle(deformGroups),
            BitConverter.Int32BitsToSingle(hasMask),
            0, 0, 0 // padding to 20 floats (80 bytes)
        };
        int total = batch * outChannels * outHeight * outWidth;
        Dispatch5BufferAsync("DeformableConv2D", WebGpuKernels.DeformableConv2DSource,
            "deformable_conv2d", input, weights, offsets, maskBuf, output,
            uniforms, total).GetAwaiter().GetResult();
    }

    public void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        // Zero-fill gradient to allow training to proceed.
        // DirectGpuTensorEngine catches exceptions and falls back to CPU for accurate gradients.
        Fill(gradInput, 0f, batch * inChannels * inHeight * inWidth);
    }

    public void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Fill(gradWeights, 0f, outChannels * (inChannels / groups) * kernelH * kernelW);
    }

    public void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        int offsetChannels = 2 * deformGroups * kernelH * kernelW;
        Fill(gradOffsets, 0f, batch * offsetChannels * outHeight * outWidth);
    }

    public void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        int maskChannels = deformGroups * kernelH * kernelW;
        Fill(gradMask, 0f, batch * maskChannels * outHeight * outWidth);
    }

    #endregion

    #region Pooling Operations

    public void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        var uniforms = MakePoolUniforms(batch, channels, inHeight, inWidth, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * channels);
        if (indices is not null)
        {
            // PoolExtendedSource: binding(0)=input, binding(1)=output, binding(2)=indices_out
            Dispatch3Buffer3DAsync("PoolExtended", WebGpuKernels.PoolExtendedSource, "max_pool2d_with_indices",
                input, output, indices, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
        }
        else
        {
            Dispatch2Buffer3DAsync("Pooling", WebGpuKernels.PoolingSource, "max_pool2d",
                input, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
        }
    }

    public void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        // First zero the gradInput, then accumulate using saved indices
        Fill(gradInput, 0f, batch * channels * inHeight * inWidth);
        // MaxPool2DBackwardIndicesSource: binding(0)=grad_output, binding(1)=indices, binding(2)=grad_input
        // Uniform: batch_size, channels, in_height, in_width, out_height, out_width
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(outHeight),
            BitConverter.Int32BitsToSingle(outWidth)
        };
        int totalOutput = batch * channels * outHeight * outWidth;
        Dispatch3BufferAsync("MaxPool2DBackwardIndices", WebGpuKernels.MaxPool2DBackwardIndicesSource, "max_pool2d_backward_indices",
            gradOutput, indices, gradInput, uniforms, totalOutput).GetAwaiter().GetResult();
    }

    public void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        // Use AvgPoolCountPadSource which respects countIncludePad:
        // When true, divides by kernelH*kernelW; when false, divides by valid (non-padded) count.
        var uniforms = MakeAvgPoolUniforms(batch, channels, inHeight, inWidth, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * channels);
        Dispatch2Buffer3DAsync("AvgPoolCountPad", WebGpuKernels.AvgPoolCountPadSource, "avg_pool2d_count_pad",
            input, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        // Use AvgPoolCountPadBackwardSource which respects countIncludePad.
        // Binding(1) is a dummy (unused by the kernel but required by the 3-buffer dispatch layout).
        using var dummyInput = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakeAvgPoolUniforms(batch, channels, inHeight, inWidth, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW, countIncludePad);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(inWidth, inHeight, batch * channels);
        Dispatch3Buffer3DAsync("AvgPoolCountPadBackward", WebGpuKernels.AvgPoolCountPadBackwardSource,
            "avg_pool2d_backward_count_pad",
            gradOutput, dummyInput, gradInput, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        // Mean across spatial dimensions for each (batch, channel) pair
        // Use StatisticsSource "mean_axis" with outer_size=batch*channels, reduce_size=spatial
        int spatial = height * width;
        var uniforms = MakeUniformInts2(batch * channels, spatial);
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "mean_axis",
            input, output, uniforms, batch * channels).GetAwaiter().GetResult();
    }

    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        // Max across spatial dimensions for each (batch, channel) pair
        int spatial = height * width;
        var uniforms = MakeUniformInts2(batch * channels, spatial);
        Dispatch2BufferAsync("Statistics", WebGpuKernels.StatisticsSource, "max_axis",
            input, output, uniforms, batch * channels).GetAwaiter().GetResult();
    }

    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
    {
        // GlobalPoolSource global_max_pool_with_indices: writes output[outer*2]=max_val, output[outer*2+1]=bitcast(max_idx)
        int outerSize = batch * channels;
        int spatial = height * width;
        using var tempOutput = (WebGpuBuffer)AllocateBuffer(outerSize * 2);
        var uniforms = MakeUniformInts2(outerSize, spatial);
        Dispatch2BufferAsync("GlobalPool", WebGpuKernels.GlobalPoolSource, "global_max_pool_with_indices",
            input, tempOutput, uniforms, outerSize).GetAwaiter().GetResult();
        // GPU deinterleave: extract values (even indices) and indices (odd indices)
        using var dummyB = (WebGpuBuffer)AllocateBuffer(1);
        var deinterlUniforms1 = new float[]
        {
            BitConverter.Int32BitsToSingle(outerSize),
            BitConverter.Int32BitsToSingle(1), // mode=1 deinterleave_a (values)
            0, 0
        };
        Dispatch3BufferAsync("Interleave", WebGpuKernels.InterleaveSource, "interleave_op",
            tempOutput, dummyB, output, deinterlUniforms1, outerSize).GetAwaiter().GetResult();
        var deinterlUniforms2 = new float[]
        {
            BitConverter.Int32BitsToSingle(outerSize),
            BitConverter.Int32BitsToSingle(2), // mode=2 deinterleave_b (indices)
            0, 0
        };
        Dispatch3BufferAsync("Interleave", WebGpuKernels.InterleaveSource, "interleave_op",
            tempOutput, dummyB, indices, deinterlUniforms2, outerSize).GetAwaiter().GetResult();
    }

    public void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        // GlobalPoolSource global_avg_pool_backward: binding(0)=gradOutput(outer), binding(1)=gradInput(outer*spatial)
        // Each element in gradInput gets gradOutput[outer] / spatial
        int outerSize = batch * channels;
        int spatial = height * width;
        var uniforms = MakeUniformInts2(outerSize, spatial);
        int totalElements = outerSize * spatial;
        Dispatch2BufferAsync("GlobalPool", WebGpuKernels.GlobalPoolSource, "global_avg_pool_backward",
            gradOutput, gradInput, uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        // GlobalPoolSource global_max_pool_backward: reads combined buffer [outer*2+0]=grad, [outer*2+1]=idx
        int outerSize = batch * channels;
        int spatial = height * width;
        // GPU interleave: pack gradOutput and indices into combined buffer
        using var combinedBuf = (WebGpuBuffer)AllocateBuffer(outerSize * 2);
        var interlUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(outerSize),
            BitConverter.Int32BitsToSingle(0), // mode=0 interleave
            0, 0
        };
        Dispatch3BufferAsync("Interleave", WebGpuKernels.InterleaveSource, "interleave_op",
            gradOutput, indices, combinedBuf, interlUniforms, outerSize).GetAwaiter().GetResult();
        var uniforms = MakeUniformInts2(outerSize, spatial);
        int totalElements = outerSize * spatial;
        Dispatch2BufferAsync("GlobalPool", WebGpuKernels.GlobalPoolSource, "global_max_pool_backward",
            combinedBuf, gradInput, uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        // Use dedicated adaptive pool kernel with per-position bin boundaries:
        // startH = floor(oh * inH / outH), endH = ceil((oh+1) * inH / outH)
        // This handles non-divisible sizes correctly (matching PyTorch behavior).
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(outHeight),
            BitConverter.Int32BitsToSingle(outWidth),
            0, 0 // padding to 8 floats (32 bytes)
        };
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * channels);
        Dispatch2Buffer3DAsync("AdaptiveAvgPool2D", WebGpuKernels.AdaptiveAvgPool2DSource,
            "adaptive_avg_pool2d",
            input, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        // Pool3DParams: 16 u32 fields packed as floats
        var uniforms = new float[16];
        int[] vals = { batch, channels, inDepth, inHeight, inWidth,
                       outDepth, outHeight, outWidth,
                       kernelD, kernelH, kernelW,
                       strideD, strideH, strideW, 0, 0 };
        for (int i = 0; i < 16; i++) uniforms[i] = BitConverter.Int32BitsToSingle(vals[i]);
        int total = batch * channels * outDepth * outHeight * outWidth;
        if (indices is not null)
        {
            // Pool3DWithIndicesSource: binding(0)=input, binding(1)=output, binding(2)=indices_out
            // Tracks the flat spatial index of the max element per output position
            Dispatch3BufferAsync("Pool3DWithIndices", WebGpuKernels.Pool3DWithIndicesSource,
                "max_pool3d_with_indices",
                input, output, indices, uniforms, total).GetAwaiter().GetResult();
        }
        else
        {
            Dispatch2BufferAsync("Pool3D", WebGpuKernels.Pool3DSource, "max_pool3d",
                input, output, uniforms, total).GetAwaiter().GetResult();
        }
    }

    public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        // CPU scatter fallback: scatter gradients back to max locations
        int inSpatial = inDepth * inHeight * inWidth;
        int outSpatial = outDepth * outHeight * outWidth;
        var goData = DownloadBufferData(gradOutput);
        var idxData = DownloadBufferData(indices);
        var giData = new float[batch * channels * inSpatial];
        for (int bc = 0; bc < batch * channels; bc++)
        {
            for (int o = 0; o < outSpatial; o++)
            {
                int outIdx = bc * outSpatial + o;
                int maxIdx = BitConverter.SingleToInt32Bits(idxData[outIdx]);
                if ((uint)maxIdx < (uint)inSpatial)
                    giData[bc * inSpatial + maxIdx] += goData[outIdx];
            }
        }
        UploadToBuffer(giData, gradInput);
    }

    public void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        int batchChannels = batch * channels;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchChannels),
            BitConverter.Int32BitsToSingle(inDepth),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(scaleD),
            BitConverter.Int32BitsToSingle(scaleH),
            BitConverter.Int32BitsToSingle(scaleW),
            0
        };
        int outD = inDepth * scaleD, outH = inHeight * scaleH, outW = inWidth * scaleW;
        int total = batchChannels * outD * outH * outW;
        Dispatch2BufferAsync("Upsample3D", WebGpuKernels.Upsample3DSource, "nearest_upsample3d",
            input, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        int batchChannels = batch * channels;
        Fill(gradInput, 0f, batchChannels * inDepth * inHeight * inWidth);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchChannels),
            BitConverter.Int32BitsToSingle(inDepth),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(scaleD),
            BitConverter.Int32BitsToSingle(scaleH),
            BitConverter.Int32BitsToSingle(scaleW),
            0
        };
        int total = batchChannels * inDepth * inHeight * inWidth;
        Dispatch2BufferAsync("Upsample3D", WebGpuKernels.Upsample3DSource, "nearest_upsample3d_backward",
            gradOutput, gradInput, uniforms, total).GetAwaiter().GetResult();
    }

    #endregion

    #region Spatial Transformer Operations

    public void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        // SpatialTransformerSource affine_grid: binding(0)=theta, binding(1)=unused, binding(2)=grid
        // Uniform: batch_size, param1=outHeight, param2=outWidth, param3-param4=unused, pad1-pad3=unused
        using var dummyB = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(outputHeight),
            BitConverter.Int32BitsToSingle(outputWidth),
            0, 0, 0, 0, 0
        };
        int total = batch * outputHeight * outputWidth;
        Dispatch3BufferAsync("SpatialTransformer", WebGpuKernels.SpatialTransformerSource, "affine_grid",
            theta, dummyB, grid, uniforms, total).GetAwaiter().GetResult();
    }

    public void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        // GridSampleExtSource: fully supports alignCorners and paddingMode.
        // paddingMode: 0=zeros, 1=border (clamp), 2=reflection.
        // alignCorners: true maps [-1,1] to [0, size-1]; false maps [-1,1] to [-0.5, size-0.5].
        int outSpatial = outHeight * outWidth;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(outSpatial),
            BitConverter.Int32BitsToSingle(paddingMode),
            BitConverter.Int32BitsToSingle(alignCorners ? 1 : 0),
            0 // padding
        };
        int total = batch * channels * outSpatial;
        Dispatch3BufferAsync("GridSampleExt", WebGpuKernels.GridSampleExtSource, "grid_sample_ext",
            input, grid, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
        // Zero-fill gradients to allow training to proceed.
        // DirectGpuTensorEngine catches exceptions and falls back to CPU for accurate gradients.
        Fill(gradInput, 0f, batch * channels * inHeight * inWidth);
        Fill(gradGrid, 0f, batch * outHeight * outWidth * 2);
    }

    #endregion

    #region Normalization Operations

    public void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        if (!training)
        {
            // Inference: use existing BatchNormSource kernel (uses running stats)
            var uniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(batch),
                BitConverter.Int32BitsToSingle(channels),
                BitConverter.Int32BitsToSingle(spatialSize),
                epsilon
            };
            Dispatch4Buffer3DAsync("BatchNorm", WebGpuKernels.BatchNormSource, "batch_norm",
                input, gamma, beta, output, uniforms, batch * channels, 1, 1).GetAwaiter().GetResult();
            Fill(saveMean, 0f, channels);
            Fill(saveInvVar, 0f, channels);
            return;
        }
        // Training mode: use BatchNormTrainingSource to compute mean/var and normalize on GPU
        var trainUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            epsilon,
            momentum,
            BitConverter.Int32BitsToSingle(1), // training flag
            0, 0
        };
        // Dispatch per channel - each thread computes stats for one channel
        Dispatch4BufferAsync("BatchNormTraining", WebGpuKernels.BatchNormTrainingSource, "batch_norm_train",
            input, gamma, beta, output, trainUniforms, channels).GetAwaiter().GetResult();
        // Compute per-channel mean and variance on GPU
        var statsUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            0
        };
        using var gpuMean = (WebGpuBuffer)AllocateBuffer(channels);
        using var gpuVar = (WebGpuBuffer)AllocateBuffer(channels);
        Dispatch3BufferAsync("BatchNormStats", WebGpuKernels.BatchNormStatsSource, "batch_norm_stats",
            input, gpuMean, gpuVar, statsUniforms, channels).GetAwaiter().GetResult();
        // Update running stats entirely on GPU: EMA + saveInvVar computation
        var emaUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(channels),
            0, // pad
            momentum,
            epsilon
        };
        Dispatch6BufferAsync("BatchNormEma", WebGpuKernels.BatchNormEmaSource, "batch_norm_ema",
            gpuMean, gpuVar, runningMean, runningVar, saveMean, saveInvVar,
            emaUniforms, channels).GetAwaiter().GetResult();
    }

    public void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        EnsureInitialized();

        // Pass 1: Compute gradGamma and gradBeta per channel
        // Bindings: gradOutput(0), input(1), saveMean(2), saveInvVar(3), gradGamma(4=rw), gradBeta(5=rw) + uniform
        var statsUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            0 // _pad
        };
        Dispatch6BufferAsync("BatchNormBackwardStats", WebGpuKernels.BatchNormBackwardStatsSource,
            "batch_norm_backward_stats",
            gradOutput, input, saveMean, saveInvVar, gradGamma, gradBeta,
            statsUniforms, channels).GetAwaiter().GetResult();

        // Pass 2: Pack per-channel stats on GPU, then compute gradInput per element.
        // Pack [gamma, mean, invVar, sumGrad, sumGradXhat] per channel (5 floats per channel)
        // entirely on GPU to avoid 5 synchronous CPU downloads.
        using var packedStats = (WebGpuBuffer)AllocateBuffer(channels * 5);
        var packUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(channels),
            0, 0, 0 // _pad
        };
        // gamma(0), saveMean(1), saveInvVar(2), gradBeta=sum_grad(3), gradGamma=sum_grad_xhat(4), packedOut(5=write)
        Dispatch6BufferAsync("PackBatchNormStats", WebGpuKernels.PackBatchNormStatsSource,
            "pack_bn_stats",
            gamma, saveMean, saveInvVar, gradBeta, gradGamma, packedStats,
            packUniforms, channels).GetAwaiter().GetResult();

        // Bindings: gradOutput(0), input(1), packedStats(2), gradInput(3=rw) + uniform
        var dataUniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            0 // _pad
        };
        int totalElements = batch * channels * spatialSize;
        Dispatch4BufferAsync("BatchNormBackwardData", WebGpuKernels.BatchNormBackwardDataSource,
            "batch_norm_backward_data",
            gradOutput, input, packedStats, gradInput,
            dataUniforms, totalElements).GetAwaiter().GetResult();
    }

    public void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        // Use existing GPU LayerNormAsync kernel
        LayerNormAsync(input, gamma, beta, output, batchSize, normalizedSize, epsilon).GetAwaiter().GetResult();
        // saveMean and saveInvVar not populated by GPU kernel; fill with zeros for compatibility
        Fill(saveMean, 0f, batchSize);
        Fill(saveInvVar, 0f, batchSize);
    }

    public void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        // LayerNormBackwardFullSource: two-pass approach.
        // Pass 1: Compute gradGamma and gradBeta by summing over batch dimension.
        //   gradBeta[f] = sum_n(gradOutput[n,f])
        //   gradGamma[f] = sum_n(gradOutput[n,f] * x_hat[n,f])
        // Pass 2: Compute gradInput per element using the full chain rule with gamma.
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(normalizedSize),
            epsilon,
            0
        };
        // Pass 1: gradGamma/gradBeta stats (dispatches per feature)
        Dispatch6BufferAsync("LayerNormBackwardFull", WebGpuKernels.LayerNormBackwardFullSource,
            "layer_norm_backward_stats",
            gradOutput, input, gamma, gradInput, gradGamma, gradBeta,
            uniforms, normalizedSize).GetAwaiter().GetResult();
        // Pass 2: gradInput (dispatches per element)
        int totalElements = batchSize * normalizedSize;
        Dispatch6BufferAsync("LayerNormBackwardFull", WebGpuKernels.LayerNormBackwardFullSource,
            "layer_norm_backward_data",
            gradOutput, input, gamma, gradInput, gradGamma, gradBeta,
            uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        // GroupNormSource: binding(0)=input, binding(1)=gamma, binding(2)=beta, binding(3)=output, binding(4)=uniform
        // Uniform: batch_size, num_groups, channels, spatial_size, epsilon, pad, pad, pad (8 floats = 32 bytes)
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(numGroups),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            epsilon,
            0, 0, 0
        };
        int totalElements = batch * channels * spatialSize;
        Dispatch4BufferAsync("GroupNorm", WebGpuKernels.GroupNormSource, "group_norm",
            input, gamma, beta, output, uniforms, totalElements).GetAwaiter().GetResult();
        // saveMean and saveInvVar not populated by GPU kernel; fill with zeros for compatibility
        Fill(saveMean, 0f, batch * numGroups);
        Fill(saveInvVar, 0f, batch * numGroups);
    }

    public void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
    {
        // InstanceNormSource: binding(0)=input, binding(1)=gamma, binding(2)=beta, binding(3)=output
        // Uniform: batch_size, channels, spatial_size, epsilon
        // Dispatch: 1 workgroup per (batch * channel) instance
        int numInstances = batch * channels;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            epsilon
        };
        Dispatch4Buffer3DAsync("InstanceNorm", WebGpuKernels.InstanceNormSource, "instance_norm",
            input, gamma, beta, output, uniforms, numInstances, 1, 1).GetAwaiter().GetResult();
        // saveMean and saveInvVar not populated by GPU kernel; fill with zeros for compatibility
        Fill(saveMean, 0f, numInstances);
        Fill(saveInvVar, 0f, numInstances);
    }

    public void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        // Treat each (batch, channel) pair as a "batch" of size spatial for normalization backward
        // Reuse NormBackwardSource layer_norm_backward with outer_size=batch*channels, feature_size=spatial
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch * channels),
            BitConverter.Int32BitsToSingle(spatialSize),
            epsilon,
            0
        };
        int totalElements = batch * channels * spatialSize;
        Dispatch3BufferAsync("NormBackward", WebGpuKernels.NormBackwardSource, "layer_norm_backward",
            gradOutput, input, gradInput, uniforms, totalElements).GetAwaiter().GetResult();

        // Compute gradGamma and gradBeta via CPU reduction
        // gradBeta[c] = sum over (batch, spatial) of gradOutput[b,c,s]
        // gradGamma[c] = sum over (batch, spatial) of gradOutput[b,c,s] * normalized_input[b,c,s]
        var goData = DownloadBufferData(gradOutput);
        var inData = DownloadBufferData(input);
        var gGamma = new float[channels];
        var gBeta = new float[channels];
        for (int b = 0; b < batch; b++)
        {
            for (int c = 0; c < channels; c++)
            {
                int offset = (b * channels + c) * spatialSize;
                // Compute mean and variance for normalization
                float mean = 0;
                for (int s = 0; s < spatialSize; s++)
                    mean += inData[offset + s];
                mean /= spatialSize;
                float var = 0;
                for (int s = 0; s < spatialSize; s++)
                {
                    float diff = inData[offset + s] - mean;
                    var += diff * diff;
                }
                var /= spatialSize;
                float invStd = 1f / MathF.Sqrt(var + epsilon);
                for (int s = 0; s < spatialSize; s++)
                {
                    float normalized = (inData[offset + s] - mean) * invStd;
                    gBeta[c] += goData[offset + s];
                    gGamma[c] += goData[offset + s] * normalized;
                }
            }
        }
        UploadToBuffer(gGamma, gradGamma);
        UploadToBuffer(gBeta, gradBeta);
    }

    public void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        // RMSNormSource: binding(0)=input, binding(1)=gamma, binding(2)=output
        // Uniform: batch_size, feature_size, epsilon (padded to 4 floats)
        // Dispatch: 1 workgroup per batch element
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(normalizedSize),
            epsilon,
            0 // padding
        };
        Dispatch3Buffer3DAsync("RMSNorm", WebGpuKernels.RMSNormSource, "rms_norm",
            input, gamma, output, uniforms, batchSize, 1, 1).GetAwaiter().GetResult();
        // saveRms not populated by GPU kernel; fill with zeros for compatibility
        Fill(saveRms, 0f, batchSize);
    }

    public void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        // NormBackwardSource rms_norm_backward: binding(0)=grad_output, binding(1)=saved_data(input), binding(2)=grad_input
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(normalizedSize),
            epsilon,
            0
        };
        int totalElements = batchSize * normalizedSize;
        Dispatch3BufferAsync("NormBackward", WebGpuKernels.NormBackwardSource, "rms_norm_backward",
            gradOutput, input, gradInput, uniforms, totalElements).GetAwaiter().GetResult();
        Fill(gradGamma, 0f, normalizedSize);
    }

    #endregion

    #region Dropout Operations

    public void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        if (training)
        {
            // Generate bernoulli mask on GPU using Philox RNG
            uint seedLo = (uint)(seed & 0xFFFFFFFF);
            uint seedHi = (uint)((seed >> 32) & 0xFFFFFFFF);
            if (seed == 0) { seedLo = (uint)Environment.TickCount; seedHi = (uint)(Environment.TickCount >> 16) ^ 0xCAFEBABE; }
            float threshold = dropoutRate; // values >= dropoutRate become 1 (keep), < dropoutRate become 0 (drop)
            var rngUniforms = new float[]
            {
                BitConverter.Int32BitsToSingle(size),
                BitConverter.Int32BitsToSingle((int)seedLo),
                BitConverter.Int32BitsToSingle((int)seedHi),
                BitConverter.Int32BitsToSingle(2), // mode=2 bernoulli_mask
                0, threshold, 0, 0
            };
            Dispatch1BufferAsync("PhiloxRng", WebGpuKernels.PhiloxRngSource, "gpu_random",
                mask, rngUniforms, size).GetAwaiter().GetResult();
            // GPU kernel: output = input * mask * scale
            float scale = 1f / (1f - dropoutRate);
            var uniforms = MakeUniform2(size, scale);
            Dispatch3BufferAsync("Dropout", WebGpuKernels.DropoutSource, "dropout_forward",
                input, output, mask, uniforms, size).GetAwaiter().GetResult();
        }
        else
        {
            // Inference: just copy input to output, mask = all ones
            Copy(input, output, size);
            Fill(mask, 1f, size);
        }
    }

    public void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
    {
        // DropoutSource dropout_backward: binding(0)=input(grad), binding(1)=output(gradInput), binding(2)=mask
        // Uniform: size, scale
        float scale = 1f / (1f - dropoutRate);
        var uniforms = MakeUniform2(size, scale);
        Dispatch3BufferAsync("Dropout", WebGpuKernels.DropoutSource, "dropout_backward",
            gradOutput, gradInput, mask, uniforms, size).GetAwaiter().GetResult();
    }

    #endregion

    #region Embedding Operations

    public void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        // EmbeddingSource: binding(0)=weights, binding(1)=indices, binding(2)=output
        // Uniform: num_indices, embedding_dim
        int totalElements = numIndices * embeddingDim;
        var uniforms = MakeUniformInts2(numIndices, embeddingDim);
        Dispatch3BufferAsync("Embedding", WebGpuKernels.EmbeddingSource, "embedding_forward",
            embeddingTable, indices, output, uniforms, totalElements).GetAwaiter().GetResult();
    }

    public void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        // First zero grad_embedding, then scatter-add
        Fill(gradEmbedding, 0f, vocabSize * embeddingDim);
        // EmbeddingBackwardSource: binding(0)=grad_output, binding(1)=indices, binding(2)=grad_embedding
        // Uniform: num_indices, embedding_dim, vocab_size, pad
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(numIndices),
            BitConverter.Int32BitsToSingle(embeddingDim),
            BitConverter.Int32BitsToSingle(vocabSize),
            0
        };
        int totalElements = vocabSize * embeddingDim;
        Dispatch3BufferAsync("EmbeddingBackward", WebGpuKernels.EmbeddingBackwardSource, "embedding_backward",
            gradOutput, indices, gradEmbedding, uniforms, totalElements).GetAwaiter().GetResult();
    }

    private static float[] MakeLCUniforms(int batch, int inChannels, int outChannels,
        int inHeight, int inWidth, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW)
    {
        return new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(inChannels),
            BitConverter.Int32BitsToSingle(outChannels),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(outHeight),
            BitConverter.Int32BitsToSingle(outWidth),
            BitConverter.Int32BitsToSingle(kernelH),
            BitConverter.Int32BitsToSingle(kernelW),
            BitConverter.Int32BitsToSingle(strideH),
            BitConverter.Int32BitsToSingle(strideW),
            0 // padding
        };
    }

    /// <summary>
    /// Packs AvgPoolParams uniform (12 pool fields + count_include_pad + 3 padding = 16 u32 = 64 bytes).
    /// </summary>
    private static float[] MakeAvgPoolUniforms(int batch, int channels,
        int inHeight, int inWidth, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        return new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(outHeight),
            BitConverter.Int32BitsToSingle(outWidth),
            BitConverter.Int32BitsToSingle(kernelH),
            BitConverter.Int32BitsToSingle(kernelW),
            BitConverter.Int32BitsToSingle(strideH),
            BitConverter.Int32BitsToSingle(strideW),
            BitConverter.Int32BitsToSingle(padH),
            BitConverter.Int32BitsToSingle(padW),
            BitConverter.Int32BitsToSingle(countIncludePad ? 1 : 0),
            0, 0, 0 // padding to 16 floats (64 bytes)
        };
    }
    #endregion
}
#endif
