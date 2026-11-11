// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : ICausalDepthwiseConvBackend
{
    private void LaunchCausalConv(string name, IGpuBuffer a, IGpuBuffer b, IGpuBuffer o,
        int rows, int seqLen, int channels, int taps, long workItems)
    {
        if (_context == null)
            throw new GpuCapabilityUnavailableException($"OpenCL context is not initialized. Cannot execute {name}.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException($"OpenCL kernel not found: {name}");
        if (workItems == 0) return;
        if (workItems > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(rows), "Causal convolution exceeds the OpenCL launch limit.");
        int total = (int)workItems;
        int localSize = CalculateOptimalWorkGroupSize1D(total);
        int globalSize = ((total + localSize - 1) / localSize) * localSize;
        uint arg = 0;
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)a).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)b).Buffer.Handle);
        kernel.SetArg(arg++, ((DirectOpenClGpuBuffer)o).Buffer.Handle);
        kernel.SetArg(arg++, rows);
        kernel.SetArg(arg++, seqLen);
        kernel.SetArg(arg++, channels);
        kernel.SetArg(arg, taps);
        kernel.Execute1D(globalSize, localSize);
    }

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DForward(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int rows, int seqLen, int channels, int taps)
        => LaunchCausalConv("causal_dwconv1d_forward", input, kernel, output, rows, seqLen, channels, taps,
            (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int rows, int seqLen, int channels, int taps)
        => LaunchCausalConv("causal_dwconv1d_backward_input", gradOutput, kernel, gradInput, rows, seqLen, channels, taps,
            (long)rows * seqLen * channels);

    void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int rows, int seqLen, int channels, int taps)
        => LaunchCausalConv("causal_dwconv1d_backward_kernel", gradOutput, input, gradKernel, rows, seqLen, channels, taps,
            (long)taps * channels);
}
