// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// A convolution's per-channel epilogue (bias, inference BatchNorm scale and shift, activation) as one pass each way:
// channel_epilogue_fwd / channel_epilogue_bwd of CudaActivationKernels.
public sealed partial class CudaBackend : IChannelEpilogueBackend
{
    unsafe bool IChannelEpilogueBackend.TryChannelEpilogue(IGpuBuffer input, IGpuBuffer? bias, IGpuBuffer? scale,
        IGpuBuffer? shift, IGpuBuffer output, int batch, int channels, int spatial, int activation, float slope)
    {
        long n = (long)batch * channels * spatial;
        if (!IsAvailable || !ChannelEpilogueActivation.Accepts(activation, slope) || batch <= 0 || channels <= 0
            || spatial <= 0 || (long)batch * channels > int.MaxValue || input.Size < n || output.Size < n
            || (bias is not null && bias.Size < channels) || (scale is not null && scale.Size < channels)
            || (shift is not null && shift.Size < channels)
            || !_kernelCache.TryGetValue("channel_epilogue_fwd", out var kernel))
            return false;
        using var _ = PushContext();
        int rows = batch * channels, c = channels, s = spatial, act = activation;
        float sl = activation == ChannelEpilogueActivation.LeakyRelu ? slope : 0f;
        IntPtr zPtr = input.Handle, yPtr = output.Handle;
        IntPtr bPtr = bias?.Handle ?? IntPtr.Zero, scPtr = scale?.Handle ?? IntPtr.Zero, shPtr = shift?.Handle ?? IntPtr.Zero;
        void** args = stackalloc void*[10];
        args[0] = &zPtr; args[1] = &bPtr; args[2] = &scPtr; args[3] = &shPtr; args[4] = &yPtr;
        args[5] = &rows; args[6] = &c; args[7] = &s; args[8] = &act; args[9] = &sl;
        LaunchKernel2D(kernel, (uint)((spatial + DefaultBlockSize - 1) / DefaultBlockSize),
            (uint)Math.Min(rows, MaxGridDimY), DefaultBlockSize, 1, args);
        return true;
    }

    unsafe bool IChannelEpilogueBackend.TryChannelEpilogueBackward(IGpuBuffer gradOutput, IGpuBuffer output,
        IGpuBuffer? scale, IGpuBuffer gradInput, int batch, int channels, int spatial, int activation, float slope)
    {
        long n = (long)batch * channels * spatial;
        if (!IsAvailable || !ChannelEpilogueActivation.Accepts(activation, slope) || batch <= 0 || channels <= 0
            || spatial <= 0 || (long)batch * channels > int.MaxValue || gradOutput.Size < n || output.Size < n
            || gradInput.Size < n || (scale is not null && scale.Size < channels)
            || !_kernelCache.TryGetValue("channel_epilogue_bwd", out var kernel))
            return false;
        using var _ = PushContext();
        int rows = batch * channels, c = channels, s = spatial, act = activation;
        float sl = activation == ChannelEpilogueActivation.LeakyRelu ? slope : 0f;
        IntPtr dyPtr = gradOutput.Handle, yPtr = output.Handle, dzPtr = gradInput.Handle;
        IntPtr scPtr = scale?.Handle ?? IntPtr.Zero;
        void** args = stackalloc void*[9];
        args[0] = &dyPtr; args[1] = &yPtr; args[2] = &scPtr; args[3] = &dzPtr;
        args[4] = &rows; args[5] = &c; args[6] = &s; args[7] = &act; args[8] = &sl;
        LaunchKernel2D(kernel, (uint)((spatial + DefaultBlockSize - 1) / DefaultBlockSize),
            (uint)Math.Min(rows, MaxGridDimY), DefaultBlockSize, 1, args);
        return true;
    }
}
