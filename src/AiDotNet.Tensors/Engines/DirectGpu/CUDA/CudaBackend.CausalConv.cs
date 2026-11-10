// Copyright (c) CrowdTrainer LLC. All rights reserved.

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : ICausalDepthwiseConvBackend
{
    private IntPtr CausalConvKernel(string name)
        => _kernelCache.TryGetValue(name, out var k) ? k
            : throw new GpuCapabilityUnavailableException($"CUDA kernel not found: {name}");

    private static uint CausalConvGrid(int rows, int seqLen, int channels)
    {
        long total = checked((long)rows * seqLen * channels);
        long grid = (total + DefaultBlockSize - 1) / DefaultBlockSize;
        if (grid > uint.MaxValue) throw new ArgumentOutOfRangeException(nameof(rows), "Causal convolution grid exceeds the launch limit.");
        return (uint)grid;
    }

    unsafe void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DForward(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int rows, int seqLen, int channels, int taps)
    {
        var fn = CausalConvKernel("causal_dwconv1d_forward");
        if (rows * (long)seqLen * channels == 0) return;
        using var _ = PushContext();
        IntPtr px = input.Handle, pw = kernel.Handle, py = output.Handle;
        void** a = stackalloc void*[7];
        a[0] = &px; a[1] = &pw; a[2] = &py; a[3] = &rows; a[4] = &seqLen; a[5] = &channels; a[6] = &taps;
        LaunchKernel(fn, CausalConvGrid(rows, seqLen, channels), (uint)DefaultBlockSize, a);
    }

    unsafe void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int rows, int seqLen, int channels, int taps)
    {
        var fn = CausalConvKernel("causal_dwconv1d_backward_input");
        if (rows * (long)seqLen * channels == 0) return;
        using var _ = PushContext();
        IntPtr pg = gradOutput.Handle, pw = kernel.Handle, pdx = gradInput.Handle;
        void** a = stackalloc void*[7];
        a[0] = &pg; a[1] = &pw; a[2] = &pdx; a[3] = &rows; a[4] = &seqLen; a[5] = &channels; a[6] = &taps;
        LaunchKernel(fn, CausalConvGrid(rows, seqLen, channels), (uint)DefaultBlockSize, a);
    }

    unsafe void ICausalDepthwiseConvBackend.CausalDepthwiseConv1DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int rows, int seqLen, int channels, int taps)
    {
        var fn = CausalConvKernel("causal_dwconv1d_backward_kernel");
        if (taps * (long)channels == 0) return;
        using var _ = PushContext();
        IntPtr pg = gradOutput.Handle, px = input.Handle, pdw = gradKernel.Handle;
        void** a = stackalloc void*[7];
        a[0] = &pg; a[1] = &px; a[2] = &pdw; a[3] = &rows; a[4] = &seqLen; a[5] = &channels; a[6] = &taps;
        LaunchKernel2D(fn, (uint)((channels + 31) / 32), (uint)taps, 1, 32, 8, a);
    }
}
