// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : ICausalDepthwiseConvBackend
{
    private void DispatchCausalConv(string glsl, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o,
        int rows, int seqLen, int channels, int taps, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(rows), "Causal convolution exceeds the Vulkan launch limit.");
        GlslNaryOp(glsl, new[] { a, b, o }, (int)workItems,
            new[] { (uint)rows, (uint)seqLen, (uint)channels, (uint)taps });
    }

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DForward(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv(VulkanRecurrenceKernels.CausalConvForward, input, kernel, output, rows, seqLen, channels, taps,
            (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv(VulkanRecurrenceKernels.CausalConvBackwardInput, gradOutput, kernel, gradInput, rows, seqLen, channels, taps,
            (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv(VulkanRecurrenceKernels.CausalConvBackwardKernel, gradOutput, input, gradKernel, rows, seqLen, channels, taps,
            (long)taps * channels);
}
