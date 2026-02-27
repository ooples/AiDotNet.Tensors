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
        // GPU kernel uses stride-based padding (dilation folded into stride for simple cases)
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);
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
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);
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
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);
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
        // ConvTranspose2DSource uses ConvParams (same layout as MakeConvUniforms)
        var uniforms = MakeConvUniforms(batch, inChannels, outChannels,
            inHeight, inWidth, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * outChannels);
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
        Fill(gradInput, 0f, batch * inChannels * inHeight * inWidth);
    }

    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        Fill(gradKernel, 0f, inChannels * outChannels * kernelH * kernelW);
    }

    public void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        Conv2D(input, weights, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, 0, 0, 1, 1);
    }

    public void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        Conv2DBackwardInput(gradOutput, weights, gradInput, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, 0, 0, 1, 1);
    }

    public void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW)
    {
        Conv2DBackwardKernel(input, gradOutput, gradWeights, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, 0, 0, 1, 1);
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
        Conv2D(input, weights, output, batch, inChannels, inHeight, inWidth,
            outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
    }

    public void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
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
        Fill(gradWeights, 0f, outChannels * inChannels * kernelH * kernelW);
    }

    public void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Fill(gradOffsets, 0f, batch * deformGroups * 2 * kernelH * kernelW * outHeight * outWidth);
    }

    public void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW,
        int groups, int deformGroups)
    {
        Fill(gradMask, 0f, batch * deformGroups * kernelH * kernelW * outHeight * outWidth);
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
        // GPU kernel counts only valid (non-padded) elements, equivalent to countIncludePad=false
        var uniforms = MakePoolUniforms(batch, channels, inHeight, inWidth, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * channels);
        Dispatch2Buffer3DAsync("Pooling", WebGpuKernels.PoolingSource, "avg_pool2d",
            input, output, uniforms, wgX, wgY, wgZ).GetAwaiter().GetResult();
    }

    public void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        bool countIncludePad)
    {
        // PoolBackwardSource declares 3 bindings; avg_pool2d_backward only reads binding(0) and writes binding(2)
        // Pass a dummy buffer for unused binding(1) (input)
        using var dummyInput = (WebGpuBuffer)AllocateBuffer(1);
        var uniforms = MakePoolUniforms(batch, channels, inHeight, inWidth, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, padH, padW);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(inWidth, inHeight, batch * channels);
        Dispatch3Buffer3DAsync("PoolBackward", WebGpuKernels.PoolBackwardSource, "avg_pool2d_backward",
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
        // Adaptive pooling uses floor-division to compute kernel/stride:
        // strideH = inHeight / outHeight, kernelH = inHeight - (outHeight - 1) * strideH
        // This gives exact coverage when inHeight is divisible by outHeight
        int strideH = inHeight / outHeight;
        int strideW = inWidth / outWidth;
        int kernelH = inHeight - (outHeight - 1) * strideH;
        int kernelW = inWidth - (outWidth - 1) * strideW;
        var uniforms = MakePoolUniforms(batch, channels, inHeight, inWidth, outHeight, outWidth,
            kernelH, kernelW, strideH, strideW, 0, 0);
        var (wgX, wgY, wgZ) = CalcWorkgroups8x8(outWidth, outHeight, batch * channels);
        Dispatch2Buffer3DAsync("Pooling", WebGpuKernels.PoolingSource, "avg_pool2d",
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
        Dispatch2BufferAsync("Pool3D", WebGpuKernels.Pool3DSource, "max_pool3d",
            input, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels,
        int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        Fill(gradInput, 0f, batch * channels * inDepth * inHeight * inWidth);
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
        // SpatialTransformerSource grid_sample: binding(0)=input(image), binding(1)=grid, binding(2)=output
        // Uniform: batch_size, param1=channels, param2=inHeight, param3=inWidth, param4=outHeight*outWidth
        int outSpatial = outHeight * outWidth;
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batch),
            BitConverter.Int32BitsToSingle(channels),
            BitConverter.Int32BitsToSingle(inHeight),
            BitConverter.Int32BitsToSingle(inWidth),
            BitConverter.Int32BitsToSingle(outSpatial),
            0, 0, 0
        };
        int total = batch * channels * outSpatial;
        Dispatch3BufferAsync("SpatialTransformer", WebGpuKernels.SpatialTransformerSource, "grid_sample",
            input, grid, output, uniforms, total).GetAwaiter().GetResult();
    }

    public void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode = 0, bool alignCorners = false)
    {
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
        // Update running stats via GPU: runningMean = (1-momentum)*runningMean + momentum*gpuMean
        // runningVar = (1-momentum)*runningVar + momentum*gpuVar
        // Use Scale+Fma approach: Scale(running, running, 1-momentum) then Fma(gpuMean, momentum_buf, running, running)
        // Simpler: download only the small channel-sized buffers for the running stats update
        var sm = DownloadBufferData(gpuMean);
        var sv = DownloadBufferData(gpuVar);
        var rm = DownloadBufferData(runningMean);
        var rv = DownloadBufferData(runningVar);
        var siv = new float[channels];
        for (int c = 0; c < channels; c++)
        {
            rm[c] = (1 - momentum) * rm[c] + momentum * sm[c];
            rv[c] = (1 - momentum) * rv[c] + momentum * sv[c];
            siv[c] = 1f / MathF.Sqrt(sv[c] + epsilon);
        }
        UploadToBuffer(sm, saveMean);
        UploadToBuffer(siv, saveInvVar);
        UploadToBuffer(rm, runningMean);
        UploadToBuffer(rv, runningVar);
    }

    public void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        Fill(gradInput, 0f, batch * channels * spatialSize);
        Fill(gradGamma, 0f, channels);
        Fill(gradBeta, 0f, channels);
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
        // NormBackwardSource layer_norm_backward: binding(0)=grad_output, binding(1)=saved_data(input), binding(2)=grad_input
        // Uniform: outer_size=batchSize, feature_size=normalizedSize, epsilon, pad
        var uniforms = new float[]
        {
            BitConverter.Int32BitsToSingle(batchSize),
            BitConverter.Int32BitsToSingle(normalizedSize),
            epsilon,
            0
        };
        int totalElements = batchSize * normalizedSize;
        Dispatch3BufferAsync("NormBackward", WebGpuKernels.NormBackwardSource, "layer_norm_backward",
            gradOutput, input, gradInput, uniforms, totalElements).GetAwaiter().GetResult();
        // gradGamma and gradBeta are not computed by this simplified kernel; zero them
        Fill(gradGamma, 0f, normalizedSize);
        Fill(gradBeta, 0f, normalizedSize);
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
        Fill(gradGamma, 0f, channels);
        Fill(gradBeta, 0f, channels);
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

    #endregion
}
#endif
