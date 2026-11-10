// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : ICausalDepthwiseConvBackend
{
    private void DispatchCausalConv(string name, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o,
        int rows, int seqLen, int channels, int taps, long workItems)
    {
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(rows), "Causal convolution exceeds the Metal launch limit.");
        DispatchRecurrence(name, (int)workItems, new[] { M(a), M(b), M(o) }, new[] { rows, seqLen, channels, taps });
    }

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DForward(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv("causal_dwconv1d_forward", input, kernel, output, rows, seqLen, channels, taps,
            (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv("causal_dwconv1d_backward_input", gradOutput, kernel, gradInput, rows, seqLen, channels, taps,
            (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int rows, int seqLen, int channels, int taps)
        => DispatchCausalConv("causal_dwconv1d_backward_kernel", gradOutput, input, gradKernel, rows, seqLen, channels, taps,
            (long)taps * channels);
}
