// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET7_0_OR_GREATER

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : ICausalDepthwiseConvBackend
{
    private void DispatchCausalConv(string key, string wgsl, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o,
        int rows, int seqLen, int channels, int taps, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(rows), "Causal convolution exceeds the WebGPU launch limit.");
        DispatchRecurrence(key, wgsl, (int)workItems, new[] { a, b, o }, new[] { rows, seqLen, channels, taps });
    }

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DForward(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv("causal_conv_fwd", WebGpuRecurrenceKernels.CausalConvForward, input, kernel, output,
            rows, seqLen, channels, taps, (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv("causal_conv_bwd_input", WebGpuRecurrenceKernels.CausalConvBackwardInput, gradOutput, kernel, gradInput,
            rows, seqLen, channels, taps, (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv("causal_conv_bwd_kernel", WebGpuRecurrenceKernels.CausalConvBackwardKernel, gradOutput, input, gradKernel,
            rows, seqLen, channels, taps, (long)taps * channels);
}
#endif
