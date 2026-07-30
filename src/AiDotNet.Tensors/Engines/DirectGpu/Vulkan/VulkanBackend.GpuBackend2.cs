// Copyright (c) AiDotNet. All rights reserved.
// IDirectGpuBackend implementation part 2: Convolution, Pooling, Normalization, Attention operations.

using System;
using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend
{
    // Shared dispatch for the conv/pool GLSL kernels (issue #646): compile-or-cache the pipeline, bind the SSBOs,
    // push the int params, and launch a 1D grid over `threads` output elements.
    private bool TryDispatchConvPoolGlsl(string glsl, uint[] pushInts, int threads, params IGpuBuffer[] buffers)
    {
        if (threads <= 0) return true;
        var pipeline = GetOrCreateGlslPipeline(glsl, buffers.Length, (uint)(pushInts.Length * sizeof(uint)));
        if (pipeline is null)
            throw new InvalidOperationException("Vulkan convolution/pooling pipeline creation failed.");
        var storages = new VulkanBuffer[buffers.Length];
        for (int i = 0; i < buffers.Length; i++) storages[i] = AsVulkan(buffers[i]).Storage;
        var threadRes = _device.AcquireThreadResources();
        lock (_computeLock)
        {
            pipeline.UpdateDescriptorSet(storages);
            RecordAndExecuteWithPushData(pipeline, threads, pushInts, (uint)(pushInts.Length * sizeof(uint)), threadRes);
        }
        return true;
    }

    #region Convolution Operations

    public void Conv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        EnsureInitialized();
        try
        {
            int total = batch * outChannels * outHeight * outWidth;
            var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
                (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
                (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv2D, pc, total, input, kernel, output)) return;
        }
        catch { throw; }
    }

    public void Conv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        EnsureInitialized();
        try
        {
            int total = batch * inChannels * inHeight * inWidth;
            var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
                (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
                (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv2DBackwardInput, pc, total, gradOutput, kernel, gradInput)) return;
        }
        catch { throw; }
    }

    public void Conv2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW)
    {
        EnsureInitialized();
        try
        {
            int total = outChannels * inChannels * kernelH * kernelW;
            var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
                (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
                (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv2DBackwardKernel, pc, total, input, gradOutput, gradKernel)) return;
        }
        catch { throw; }
    }

    public void Conv1D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2D(input, kernel, output, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public void Conv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2DBackwardInput(gradOutput, kernel, gradInput, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public void Conv1DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inLength,
        int outChannels, int outLength, int kernelLength,
        int stride, int padding, int dilation)
    {
        Conv2DBackwardKernel(input, gradOutput, gradKernel, batch, inChannels, 1, inLength,
            outChannels, 1, outLength, 1, kernelLength, 1, stride, 0, padding, 1, dilation);
    }

    public void Unfold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int height, int width,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        try
        {
            int oH = (height + 2 * padH - kernelH) / strideH + 1;
            int oW = (width + 2 * padW - kernelW) / strideW + 1;
            int total = batch * channels * kernelH * kernelW * oH * oW;
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)height, (uint)width,
                (uint)kernelH, (uint)kernelW, (uint)strideH, (uint)strideW, (uint)padH, (uint)padW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Unfold, pc, total, input, output)) return;
        }
        catch { throw; }
    }

    public void Fold(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int outputH, int outputW,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        try
        {
            int total = batch * channels * outputH * outputW;
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)outputH, (uint)outputW,
                (uint)kernelH, (uint)kernelW, (uint)strideH, (uint)strideW, (uint)padH, (uint)padW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Fold, pc, total, input, output)) return;
        }
        catch { throw; }
    }

    public void Conv3D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inDepth, int inHeight, int inWidth,
        int outChannels, int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW,
        int padD, int padH, int padW,
        int dilationD, int dilationH, int dilationW)
    {
        EnsureInitialized();
        try
        {
            int total = batch * outChannels * outDepth * outHeight * outWidth;
            var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inDepth, (uint)inHeight, (uint)inWidth,
                (uint)outChannels, (uint)outDepth, (uint)outHeight, (uint)outWidth,
                (uint)kernelD, (uint)kernelH, (uint)kernelW, (uint)strideD, (uint)strideH, (uint)strideW,
                (uint)padD, (uint)padH, (uint)padW, (uint)dilationD, (uint)dilationH, (uint)dilationW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.Conv3D, pc, total, input, kernel, output)) return;
        }
        catch { throw; }
    }

    public void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        try
        {
            int total = batch * channels * outHeight * outWidth;
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth,
                (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
                (uint)strideH, (uint)strideW, (uint)padH, (uint)padW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.DepthwiseConv2D, pc, total, input, kernel, output)) return;
        }
        catch { throw; }
    }

    // GLSL conv-transpose mirrors the verified OpenCL `conv_transpose2d` gather kernel: each thread owns one
    // output element and accumulates over the input positions that scatter into it. Output-pad rows/cols naturally
    // resolve to 0 (no valid input maps to them), so outputPadH/W need no special handling — matching OpenCL/CUDA.
    private const string ConvTranspose2DGlsl = @"#version 450
layout(local_size_x = 256) in;
layout(set=0, binding=0) buffer B0 { float inp[]; };
layout(set=0, binding=1) buffer B1 { float wgt[]; };
layout(set=0, binding=2) buffer B2 { float outp[]; };
layout(push_constant) uniform PC {
    int batch;
    int inChannels;
    int inHeight;
    int inWidth;
    int outChannels;
    int outHeight;
    int outWidth;
    int kernelH;
    int kernelW;
    int strideH;
    int strideW;
    int padH;
    int padW;
};
void main() {
    uint gid = gl_GlobalInvocationID.x;
    int total = batch * outChannels * outHeight * outWidth;
    if (gid >= uint(total)) return;
    int idx = int(gid);
    int ow = idx % outWidth;
    int t = idx / outWidth;
    int oh = t % outHeight;
    t = t / outHeight;
    int oc = t % outChannels;
    int b = t / outChannels;
    float sum = 0.0;
    for (int ic = 0; ic < inChannels; ic++) {
        for (int kh = 0; kh < kernelH; kh++) {
            for (int kw = 0; kw < kernelW; kw++) {
                int ihBase = oh + padH - kh;
                int iwBase = ow + padW - kw;
                if ((ihBase % strideH) == 0 && (iwBase % strideW) == 0) {
                    int ih = ihBase / strideH;
                    int iw = iwBase / strideW;
                    if (ih >= 0 && ih < inHeight && iw >= 0 && iw < inWidth) {
                        float inVal = inp[((b * inChannels + ic) * inHeight + ih) * inWidth + iw];
                        // weights layout [inChannels, outChannels, kH, kW]
                        float wVal = wgt[((ic * outChannels + oc) * kernelH + kh) * kernelW + kw];
                        sum += inVal * wVal;
                    }
                }
            }
        }
    }
    outp[((b * outChannels + oc) * outHeight + oh) * outWidth + ow] = sum;
}";

    public void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        EnsureInitialized();
        int total = checked(checked(batch * outChannels) * checked(outHeight * outWidth));
        var pc = new uint[]
        {
            (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
            (uint)outChannels, (uint)outHeight, (uint)outWidth,
            (uint)kernelH, (uint)kernelW, (uint)strideH, (uint)strideW, (uint)padH, (uint)padW
        };
        TryDispatchConvPoolGlsl(ConvTranspose2DGlsl, pc, total, input, kernel, output);
    }

    public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        Conv2D(gradOutput, kernel, gradInput, batch, outChannels, outHeight, outWidth,
            inChannels, inHeight, inWidth, kernelH, kernelW, strideH, strideW, padH, padW, 1, 1);
    }

    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int outputPadH, int outputPadW)
    {
        EnsureInitialized();
        try
        {
            int total = inChannels * outChannels * kernelH * kernelW;
            var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
                (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
                (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)outputPadH, (uint)outputPadW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.ConvTranspose2DBackwardWeights, pc, total, input, gradOutput, gradKernel)) return;
        }
        catch { throw; }
    }

    public void LocallyConnectedConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer? bias, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW)
    {
        EnsureInitialized();
        using var biasDummy = bias is null ? AllocateBuffer(Math.Max(1, outChannels)) : null;
        if (biasDummy is not null) Fill(biasDummy, 0f, outChannels);
        int total = checked(checked(batch * outChannels) * checked(outHeight * outWidth));
        var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
            (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
            (uint)strideH, (uint)strideW };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.LocallyConnectedConv2D, pc, total,
            input, weights, bias ?? biasDummy!, output);
    }

    public void LocallyConnectedConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW)
    {
        EnsureInitialized();
        try
        {
            int total = batch * inChannels * inHeight * inWidth;
            var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
                (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW, (uint)strideH, (uint)strideW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.LocallyConnectedConv2DBackwardInput, pc, total, gradOutput, weights, gradInput)) return;
        }
        catch { throw; }
    }

    public void LocallyConnectedConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW)
    {
        EnsureInitialized();
        try
        {
            int total = outHeight * outWidth * outChannels * inChannels * kernelH * kernelW;
            var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
                (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW, (uint)strideH, (uint)strideW };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.LocallyConnectedConv2DBackwardWeights, pc, total, gradOutput, input, gradWeights)) return;
        }
        catch { throw; }
    }

    public void LocallyConnectedConv2DBackwardBias(IGpuBuffer gradOutput, IGpuBuffer gradBias,
        int batch, int outChannels, int outHeight, int outWidth)
    {
        EnsureInitialized();
        try
        {
            var pc = new uint[] { (uint)batch, (uint)outChannels, (uint)outHeight, (uint)outWidth };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.LocallyConnectedConv2DBackwardBias, pc, outChannels, gradOutput, gradBias)) return;
        }
        catch { throw; }
    }

    /// <summary>
    /// Bilinearly samples input at fractional position (fy, fx) for given batch and channel.
    /// Returns 0 for out-of-bounds positions (zero-padding).
    /// </summary>
    private static float BilinearSample(float[] inp, int b, int c, float fy, float fx,
        int channels, int height, int width)
    {
        int iy0 = (int)MathF.Floor(fy);
        int ix0 = (int)MathF.Floor(fx);
        float dy = fy - iy0;
        float dx = fx - ix0;
        float val = 0f;
        for (int jy = 0; jy <= 1; jy++)
            for (int jx = 0; jx <= 1; jx++)
            {
                int py = iy0 + jy, px = ix0 + jx;
                if (py >= 0 && py < height && px >= 0 && px < width)
                {
                    float w = (jy == 0 ? 1f - dy : dy) * (jx == 0 ? 1f - dx : dx);
                    val += w * inp[((b * channels + c) * height + py) * width + px];
                }
            }
        return val;
    }

    private static void ValidateDeformableGroups(int inChannels, int outChannels, int groups, int deformGroups)
    {
        if (groups <= 0) throw new ArgumentOutOfRangeException(nameof(groups));
        if (deformGroups <= 0) throw new ArgumentOutOfRangeException(nameof(deformGroups));
        if (inChannels % groups != 0 || outChannels % groups != 0)
            throw new ArgumentException("Input and output channels must be divisible by groups.");
        if (inChannels % deformGroups != 0 || outChannels % deformGroups != 0)
            throw new ArgumentException("Input and output channels must be divisible by deformGroups.");
    }

    public void DeformableConv2D(IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int groups, int deformGroups)
    {
        ValidateDeformableGroups(inChannels, outChannels, groups, deformGroups);
        int maskSize = checked(checked(checked(batch * deformGroups) * kernelH * kernelW) * outHeight * outWidth);
        using var maskDummy = mask is null ? AllocateBuffer(Math.Max(1, maskSize)) : null;
        if (maskDummy is not null) Fill(maskDummy, 1f, maskSize);
        int total = checked(checked(batch * outChannels) * outHeight * outWidth);
        var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
            (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
            (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW,
            (uint)groups, (uint)deformGroups };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.DeformableConv2D, pc, total,
            input, weights, offsets, mask ?? maskDummy!, output);
    }

    public void DeformableConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int groups, int deformGroups)
    {
        ValidateDeformableGroups(inChannels, outChannels, groups, deformGroups);
        int maskSize = checked(checked(checked(batch * deformGroups) * kernelH * kernelW) * outHeight * outWidth);
        using var maskDummy = mask is null ? AllocateBuffer(Math.Max(1, maskSize)) : null;
        if (maskDummy is not null) Fill(maskDummy, 1f, maskSize);
        int total = checked(checked(batch * inChannels) * inHeight * inWidth);
        var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
            (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
            (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW,
            (uint)groups, (uint)deformGroups };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.DeformableConv2DBackwardInput, pc, total,
            gradOutput, weights, offsets, mask ?? maskDummy!, gradInput);
    }

    public void DeformableConv2DBackwardWeights(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradWeights,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int groups, int deformGroups)
    {
        ValidateDeformableGroups(inChannels, outChannels, groups, deformGroups);
        int maskSize = checked(checked(checked(batch * deformGroups) * kernelH * kernelW) * outHeight * outWidth);
        using var maskDummy = mask is null ? AllocateBuffer(Math.Max(1, maskSize)) : null;
        if (maskDummy is not null) Fill(maskDummy, 1f, maskSize);
        int total = checked(checked(outChannels * (inChannels / groups)) * kernelH * kernelW);
        var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
            (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
            (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW,
            (uint)groups, (uint)deformGroups };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.DeformableConv2DBackwardWeights, pc, total,
            gradOutput, input, offsets, mask ?? maskDummy!, gradWeights);
    }

    public void DeformableConv2DBackwardOffset(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer? mask, IGpuBuffer gradOffsets,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int groups, int deformGroups)
    {
        ValidateDeformableGroups(inChannels, outChannels, groups, deformGroups);
        int maskSize = checked(checked(checked(batch * deformGroups) * kernelH * kernelW) * outHeight * outWidth);
        using var maskDummy = mask is null ? AllocateBuffer(Math.Max(1, maskSize)) : null;
        if (maskDummy is not null) Fill(maskDummy, 1f, maskSize);
        int total = checked(checked(checked(batch * deformGroups) * 2 * kernelH * kernelW) * outHeight * outWidth);
        var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
            (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
            (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW,
            (uint)groups, (uint)deformGroups };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.DeformableConv2DBackwardOffset, pc, total,
            gradOutput, input, weights, offsets, mask ?? maskDummy!, gradOffsets);
    }

    public void DeformableConv2DBackwardMask(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer weights, IGpuBuffer offsets, IGpuBuffer gradMask,
        int batch, int inChannels, int inHeight, int inWidth,
        int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW,
        int dilationH, int dilationW, int groups, int deformGroups)
    {
        ValidateDeformableGroups(inChannels, outChannels, groups, deformGroups);
        int total = checked(checked(checked(batch * deformGroups) * kernelH * kernelW) * outHeight * outWidth);
        var pc = new uint[] { (uint)batch, (uint)inChannels, (uint)inHeight, (uint)inWidth,
            (uint)outChannels, (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
            (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)dilationH, (uint)dilationW,
            (uint)groups, (uint)deformGroups };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.DeformableConv2DBackwardMask, pc, total,
            gradOutput, input, weights, offsets, gradMask);
    }

    #endregion

    #region Pooling Operations

    public void MaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        int total = checked(checked(batch * channels) * outHeight * outWidth);
        using var indicesDummy = indices is null ? AllocateBuffer(Math.Max(1, total)) : null;
        var pc = new uint[] { (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth,
            (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
            (uint)strideH, (uint)strideW, (uint)padH, (uint)padW };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.MaxPool2D, pc, total,
            input, output, indices ?? indicesDummy!);
    }

    public void MaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW)
    {
        EnsureInitialized();
        int total = checked(checked(batch * channels) * inHeight * inWidth);
        GlslNaryOp(VulkanResidentKernels.MaxPool2DBackward,
            new[] { gradOutput, indices, gradInput }, total,
            new[] { (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth, (uint)outHeight, (uint)outWidth });
    }

    public void AvgPool2D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, bool countIncludePad)
    {
        EnsureInitialized();
        try
        {
            int total = batch * channels * outHeight * outWidth;
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth,
                (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
                (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)(countIncludePad ? 1 : 0) };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.AvgPool2D, pc, total, input, output)) return;
        }
        catch { throw; }
    }

    public void AvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth,
        int kernelH, int kernelW,
        int strideH, int strideW, int padH, int padW, bool countIncludePad)
    {
        EnsureInitialized();
        try
        {
            int total = batch * channels * inHeight * inWidth;
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth,
                (uint)outHeight, (uint)outWidth, (uint)kernelH, (uint)kernelW,
                (uint)strideH, (uint)strideW, (uint)padH, (uint)padW, (uint)(countIncludePad ? 1 : 0) };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.AvgPool2DBackward, pc, total, gradOutput, gradInput)) return;
        }
        catch { throw; }
    }

    public void GlobalAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        try
        {
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)height, (uint)width };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.GlobalAvgPool2D, pc, batch * channels, input, output)) return;
        }
        catch { throw; }
    }

    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        try
        {
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)height, (uint)width };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.GlobalMaxPool2DNoIndices, pc, batch * channels, input, output)) return;
        }
        catch { throw; }
    }

    public void GlobalMaxPool2D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer indices, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        try
        {
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)height, (uint)width };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.GlobalMaxPool2D, pc, batch * channels, input, output, indices)) return;
        }
        catch { throw; }
    }

    public void GlobalAvgPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        try
        {
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)height, (uint)width };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.GlobalAvgPool2DBackward, pc, batch * channels * height * width, gradOutput, gradInput)) return;
        }
        catch { throw; }
    }

    public void GlobalMaxPool2DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput, int batch, int channels, int height, int width)
    {
        EnsureInitialized();
        try
        {
            Fill(gradInput, 0f, batch * channels * height * width); // only argmax positions are written
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)height, (uint)width };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.GlobalMaxPool2DBackward, pc, batch * channels, gradOutput, indices, gradInput)) return;
        }
        catch { throw; }
    }

    public void AdaptiveAvgPool2D(IGpuBuffer input, IGpuBuffer output, int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth)
    {
        EnsureInitialized();
        try
        {
            var pc = new uint[] { (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth, (uint)outHeight, (uint)outWidth };
            if (TryDispatchConvPoolGlsl(VulkanConvPoolKernels.AdaptiveAvgPool2D, pc, batch * channels * outHeight * outWidth, input, output)) return;
        }
        catch { throw; }
    }

    public void MaxPool3D(IGpuBuffer input, IGpuBuffer output, IGpuBuffer? indices,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth,
        int kernelD, int kernelH, int kernelW,
        int strideD, int strideH, int strideW)
    {
        EnsureInitialized();
        int total = checked(checked(checked(batch * channels) * outDepth) * outHeight * outWidth);
        using var indicesDummy = indices is null ? AllocateBuffer(Math.Max(1, total)) : null;
        var pc = new uint[] { (uint)batch, (uint)channels, (uint)inDepth, (uint)inHeight, (uint)inWidth,
            (uint)outDepth, (uint)outHeight, (uint)outWidth,
            (uint)kernelD, (uint)kernelH, (uint)kernelW, (uint)strideD, (uint)strideH, (uint)strideW };
        TryDispatchConvPoolGlsl(VulkanConvPoolKernels.MaxPool3D, pc, total,
            input, output, indices ?? indicesDummy!);
    }

    public void MaxPool3DBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradInput,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int outDepth, int outHeight, int outWidth)
    {
        EnsureInitialized();
        int total = checked(checked(checked(batch * channels) * inDepth) * inHeight * inWidth);
        GlslNaryOp(VulkanResidentKernels.MaxPool3DBackward,
            new[] { gradOutput, indices, gradInput }, total,
            new[] { (uint)batch, (uint)channels, (uint)inDepth, (uint)inHeight, (uint)inWidth,
                (uint)outDepth, (uint)outHeight, (uint)outWidth });
    }

    public void NearestNeighborUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        EnsureInitialized();
        if (DispatchNearestUpsample3D(input, output, batch, channels, inDepth, inHeight, inWidth,
            scaleD, scaleH, scaleW, false)) return;
    }

    public void NearestNeighborUpsample3DBackward(IGpuBuffer gradOutput, IGpuBuffer gradInput,
        int batch, int channels, int inDepth, int inHeight, int inWidth,
        int scaleD, int scaleH, int scaleW)
    {
        EnsureInitialized();
        if (DispatchNearestUpsample3D(gradOutput, gradInput, batch, channels, inDepth, inHeight, inWidth,
            scaleD, scaleH, scaleW, true)) return;
    }

    #endregion

    #region Spatial Transformer Operations

    private bool DispatchNearestUpsample3D(IGpuBuffer input, IGpuBuffer output,
        int batch, int channels, int depth, int height, int width,
        int scaleD, int scaleH, int scaleW, bool backward)
    {
        int threads = backward
            ? batch * channels * depth * height * width
            : batch * channels * depth * scaleD * height * scaleH * width * scaleW;
        GlslUnaryOp(backward ? VulkanGlslKernels.NearestNeighborUpsample3DBackward : VulkanGlslKernels.NearestNeighborUpsample3D,
            input, output, threads,
            new uint[] { (uint)batch, (uint)channels, (uint)depth, (uint)height, (uint)width, (uint)scaleD, (uint)scaleH, (uint)scaleW },
            8 * sizeof(uint));
        return true;
    }

    private bool DispatchGridSampleBackwardResident(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid, int batch, int channels, int inHeight, int inWidth,
        int outHeight, int outWidth, int paddingMode, bool alignCorners)
    {
        var push = new uint[]
        {
            (uint)batch, (uint)channels, (uint)inHeight, (uint)inWidth, (uint)outHeight, (uint)outWidth,
            (uint)paddingMode, alignCorners ? 1u : 0u
        };
        GlslBinaryOp(VulkanGlslKernels.GridSampleBackwardInput, gradOutput, grid, gradInput,
            batch * channels * inHeight * inWidth, push, 8 * sizeof(uint));
        GlslQuadOp(VulkanGlslKernels.GridSampleBackwardGrid, gradOutput, input, grid, gradGrid,
            batch * outHeight * outWidth, push, 8 * sizeof(uint));
        return true;
    }

    public void AffineGrid(IGpuBuffer theta, IGpuBuffer grid, int batch, int outputHeight, int outputWidth)
    {
        EnsureInitialized();
        GlslUnaryOp(VulkanGlslKernels.AffineGrid2D, theta, grid, batch * outputHeight * outputWidth,
            new uint[] { (uint)batch, (uint)outputHeight, (uint)outputWidth }, 3 * sizeof(uint));
        if (grid.Size >= 0) return;
    }

    /// <summary>
    /// Applies padding mode to a pixel coordinate for grid sampling.
    /// paddingMode: 0=zeros (returns -1 for out-of-bounds), 1=border (clamp), 2=reflection.
    /// </summary>
    private static int ApplyGridPadding(int coord, int size, int paddingMode)
    {
        if (coord >= 0 && coord < size) return coord;
        if (paddingMode == 0) return -1; // zeros: signal out-of-bounds
        if (paddingMode == 1) return Math.Max(0, Math.Min(size - 1, coord)); // border: clamp
        // reflection: reflect at boundaries
        if (size <= 1) return 0;
        int period = 2 * (size - 1);
        int c = coord % period;
        if (c < 0) c += period;
        return c < size ? c : period - c;
    }

    public void GridSample(IGpuBuffer input, IGpuBuffer grid, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode, bool alignCorners)
    {
        EnsureInitialized();
        GridSample2D(input, grid, output, batch, inHeight, inWidth, channels, outHeight, outWidth,
            0, paddingMode, alignCorners);
        if (output.Size >= 0) return;
    }

    public void GridSampleBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer grid,
        IGpuBuffer gradInput, IGpuBuffer gradGrid,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int paddingMode, bool alignCorners)
    {
        EnsureInitialized();
        if (DispatchGridSampleBackwardResident(gradOutput, input, grid, gradInput, gradGrid,
            batch, channels, inHeight, inWidth, outHeight, outWidth, paddingMode, alignCorners)) return;
    }

    #endregion

    #region Normalization Operations

    private bool DispatchNormalizationForward(IGpuBuffer[] buffers, int threads, uint[] pushConstants)
    {
        GlslNaryOp(VulkanGlslKernels.NormalizationForward, buffers, threads, pushConstants);
        return true;
    }

    private bool DispatchNormalizationBackward(IGpuBuffer[] buffers, int threads, uint[] pushConstants)
    {
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward, buffers, threads, pushConstants);
        return true;
    }

    public void BatchNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training)
    {
        EnsureInitialized();
        if (DispatchNormalizationForward(
            new[] { input, output, gamma, beta, runningMean, runningVar, saveMean, saveInvVar }, channels,
            new[] { 0u, (uint)batch, (uint)channels, (uint)spatialSize, 1u, FloatBits(epsilon), FloatBits(momentum), training ? 1u : 0u }))
            return;
    }

    public bool TryFusedBatchNormActivation(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer runningMean, IGpuBuffer runningVar, IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        int batch, int channels, int spatialSize, float epsilon, float momentum, bool training,
        FusedActivationType activation) => false;

    public void BatchNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        EnsureInitialized();
        if (DispatchNormalizationBackward(
            new[] { gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta }, channels,
            new[] { 0u, 0u, (uint)batch, (uint)channels, (uint)spatialSize }))
            return;
    }

    public void LayerNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batchSize, int normalizedSize, float epsilon)
    {
        EnsureInitialized();
        using var layerRunningMeanDummy = AllocateBuffer(1);
        using var layerRunningVarianceDummy = AllocateBuffer(1);
        if (DispatchNormalizationForward(
            new[] { input, output, gamma, beta, layerRunningMeanDummy, layerRunningVarianceDummy, saveMean, saveInvVar }, batchSize,
            new[] { 1u, (uint)batchSize, (uint)normalizedSize, 1u, 1u, FloatBits(epsilon), 0u, 1u }))
            return;
    }

    public void LayerNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batchSize, int normalizedSize, float epsilon)
    {
        EnsureInitialized();
        var layerBuffers = new[] { gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta };
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward, layerBuffers, batchSize,
            new[] { 1u, 0u, (uint)batchSize, (uint)normalizedSize, 1u });
        if (DispatchNormalizationBackward(layerBuffers, normalizedSize,
            new[] { 1u, 1u, (uint)batchSize, (uint)normalizedSize, 1u }))
            return;
    }

    public void GroupNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int numGroups, int channels, int spatialSize, float epsilon)
    {
        EnsureInitialized();
        using var groupRunningMeanDummy = AllocateBuffer(1);
        using var groupRunningVarianceDummy = AllocateBuffer(1);
        if (DispatchNormalizationForward(
            new[] { input, output, gamma, beta, groupRunningMeanDummy, groupRunningVarianceDummy, saveMean, saveInvVar }, batch * numGroups,
            new[] { 2u, (uint)batch, (uint)channels, (uint)spatialSize, (uint)numGroups, FloatBits(epsilon), 0u, 1u }))
            return;
    }

    public void InstanceNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar, int batch, int channels, int spatialSize, float epsilon)
        => GroupNorm(input, output, gamma, beta, saveMean, saveInvVar, batch, channels, channels, spatialSize, epsilon);

    public void InstanceNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma,
        IGpuBuffer saveMean, IGpuBuffer saveInvVar,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatialSize, float epsilon)
    {
        EnsureInitialized();
        var instanceBuffers = new[] { gradOutput, input, gamma, saveMean, saveInvVar, gradInput, gradGamma, gradBeta };
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward, instanceBuffers, batch * channels,
            new[] { 2u, 0u, (uint)batch, (uint)channels, (uint)spatialSize });
        if (DispatchNormalizationBackward(instanceBuffers, channels,
            new[] { 2u, 1u, (uint)batch, (uint)channels, (uint)spatialSize }))
            return;
    }

    public void RmsNorm(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer saveRms,
        int batchSize, int normalizedSize, float epsilon)
    {
        EnsureInitialized();
        using var rmsBetaDummy = AllocateBuffer(1);
        using var rmsRunningMeanDummy = AllocateBuffer(1);
        using var rmsRunningVarianceDummy = AllocateBuffer(1);
        using var rmsSavedMeanDummy = AllocateBuffer(Math.Max(1, batchSize));
        if (DispatchNormalizationForward(
            new[] { input, output, gamma, rmsBetaDummy, rmsRunningMeanDummy, rmsRunningVarianceDummy, rmsSavedMeanDummy, saveRms }, batchSize,
            new[] { 3u, (uint)batchSize, (uint)normalizedSize, 1u, 1u, FloatBits(epsilon), 0u, 1u }))
            return;
    }

    public void RmsNormBackward(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveRms,
        IGpuBuffer gradInput, IGpuBuffer gradGamma, int batchSize, int normalizedSize, float epsilon)
    {
        EnsureInitialized();
        using var rmsSavedMeanDummy = AllocateBuffer(Math.Max(1, batchSize));
        using var rmsGradBetaDummy = AllocateBuffer(Math.Max(1, normalizedSize));
        var rmsBuffers = new[] { gradOutput, input, gamma, rmsSavedMeanDummy, saveRms, gradInput, gradGamma, rmsGradBetaDummy };
        GlslNaryOp(VulkanGlslKernels.NormalizationBackward, rmsBuffers, batchSize,
            new[] { 3u, 0u, (uint)batchSize, (uint)normalizedSize, 1u });
        if (DispatchNormalizationBackward(rmsBuffers, normalizedSize,
            new[] { 3u, 1u, (uint)batchSize, (uint)normalizedSize, 1u }))
            return;
    }

    #endregion

    #region Dropout

    private bool DispatchDropoutResident(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask,
        int size, float dropoutRate, ulong seed, bool training)
    {
        if (training)
        {
            if (dropoutRate < 0f || dropoutRate >= 1f)
                throw new ArgumentOutOfRangeException(nameof(dropoutRate), $"dropoutRate must be in [0, 1), got {dropoutRate}.");
            DropoutMask(mask, size, 1f - dropoutRate, seed);
            Multiply(input, mask, output, size);
        }
        else
        {
            Copy(input, output, size);
            Fill(mask, 1f, size);
        }
        return true;
    }

    public void Dropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer mask, int size, float dropoutRate, ulong seed, bool training)
    {
        EnsureInitialized();
        if (DispatchDropoutResident(input, output, mask, size, dropoutRate, seed, training)) return;
    }

    public void DropoutBackward(IGpuBuffer gradOutput, IGpuBuffer mask, IGpuBuffer gradInput, int size, float dropoutRate)
        => ResidentBinary(ResidentBinaryOp.Multiply, gradOutput, mask, gradInput, size);

    public bool TryFusedBiasDropout(IGpuBuffer input, IGpuBuffer output, IGpuBuffer bias, IGpuBuffer mask,
        int rows, int cols, float dropoutRate, float scale) => false;

    #endregion

    #region Embedding

    private bool DispatchEmbeddingResident(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output,
        int numIndices, int embeddingDim)
    {
        int vocabSize = embeddingTable.Size / embeddingDim;
        GlslBinaryOp(VulkanGlslKernels.GatherRows, embeddingTable, indices, output, numIndices * embeddingDim,
            new uint[] { (uint)numIndices, (uint)embeddingDim, (uint)vocabSize }, 3 * sizeof(uint));
        return true;
    }

    private bool DispatchEmbeddingBackwardResident(IGpuBuffer gradOutput, IGpuBuffer indices,
        IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        GlslBinaryOp(VulkanGlslKernels.EmbeddingBackward, gradOutput, indices, gradEmbedding,
            vocabSize * embeddingDim,
            new uint[] { (uint)numIndices, (uint)embeddingDim, (uint)vocabSize }, 3 * sizeof(uint));
        return true;
    }

    public void Embedding(IGpuBuffer indices, IGpuBuffer embeddingTable, IGpuBuffer output, int numIndices, int embeddingDim)
    {
        EnsureInitialized();
        if (DispatchEmbeddingResident(indices, embeddingTable, output, numIndices, embeddingDim)) return;
    }

    public void EmbeddingBackward(IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradEmbedding, int numIndices, int embeddingDim, int vocabSize)
    {
        EnsureInitialized();
        if (DispatchEmbeddingBackwardResident(gradOutput, indices, gradEmbedding, numIndices, embeddingDim, vocabSize)) return;
    }

    #endregion
}
