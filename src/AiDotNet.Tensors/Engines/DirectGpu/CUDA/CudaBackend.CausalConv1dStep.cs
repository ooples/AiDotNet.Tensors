// Copyright (c) CrowdTrainer LLC. All rights reserved.
// CudaBackend.ICausalConv1dStepBackend: a causal depthwise convolution's decode step in one launch.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : ICausalConv1dStepBackend
{
    /// <inheritdoc />
    unsafe void ICausalConv1dStepBackend.CausalConv1dStep(IGpuBuffer history, IGpuBuffer input, IGpuBuffer weight,
        IGpuBuffer? bias, IGpuBuffer output, int taps, int batch, int channels, bool silu)
    {
        if (taps < 0 || taps > ICausalConv1dStepBackend.Limits.MaxTaps) throw new ArgumentOutOfRangeException(nameof(taps), $"conv1d_step carries 0 to {ICausalConv1dStepBackend.Limits.MaxTaps} previous inputs.");
        if (batch <= 0 || channels <= 0) throw new ArgumentOutOfRangeException(nameof(batch), "batch and channels must be positive.");
        if (!IsAvailable) throw new GpuCapabilityUnavailableException("CUDA backend is not available.");
        if (!_kernelCache.TryGetValue("conv1d_step", out var kernel))
            throw new InvalidOperationException("CUDA kernel not found: conv1d_step");
        using var _ = PushContext();
        IntPtr h = history.Handle, x = input.Handle, w = weight.Handle, b = bias?.Handle ?? IntPtr.Zero, y = output.Handle;
        int t = taps, n = batch, c = channels, act = silu ? 1 : 0;
        void** args = stackalloc void*[9];
        args[0] = &h; args[1] = &x; args[2] = &w; args[3] = &b; args[4] = &y; args[5] = &t; args[6] = &n; args[7] = &c; args[8] = &act;
        const int Threads = 256;
        LaunchKernel(kernel, (uint)((batch * channels + Threads - 1) / Threads), Threads, args);
    }
}
