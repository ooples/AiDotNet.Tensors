// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

// A convolution's per-channel epilogue (bias, inference BatchNorm scale and shift, activation) as one pass each way, through
// PortableChannelEpilogue and ChannelEpilogueKernelTemplate's kernels, compiled on first need.
public sealed partial class HipBackend : IChannelEpilogueBackend
{
    private IntPtr _channelEpilogueModule;
    private bool? _channelEpilogueReady;

    // The module, built once; a compile the device rejects leaves the separate passes serving, and says why.
    private bool ChannelEpilogueReady
    {
        get
        {
            if (_channelEpilogueReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_channelEpilogueReady is null)
                {
                    var failure = CompileKernelModule(
                        ChannelEpilogueKernelTemplate.GetModuleSource(WinogradKernelDialect.Hip), "channel_epilogue",
                        ref _channelEpilogueModule, ChannelEpilogueKernelTemplate.GetKernelNames(), useFastMath: false);
                    if (failure is not null)
                        System.Diagnostics.Trace.TraceWarning($"HIP channel epilogue kernels unavailable: {failure.Message}");
                    bool ok = failure is null;
                    foreach (var name in ChannelEpilogueKernelTemplate.GetKernelNames()) ok &= HasTunedKernel(name);
                    _channelEpilogueReady = ok;
                }
                return _channelEpilogueReady.Value;
            }
        }
    }

    private unsafe void LaunchChannelEpilogue(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        int count = buffers.Length + ints.Length;
        IntPtr* handles = stackalloc IntPtr[buffers.Length];
        int* values = stackalloc int[ints.Length];
        void** args = stackalloc void*[count];
        for (int i = 0; i < buffers.Length; i++) { handles[i] = buffers[i].Handle; args[i] = &handles[i]; }
        for (int i = 0; i < ints.Length; i++) { values[i] = ints[i]; args[buffers.Length + i] = &values[i]; }
        LaunchKernel(TunedKernel(kernelName), (uint)groups, (uint)ChannelEpilogueKernelTemplate.GroupSize, args);
    }

    bool IChannelEpilogueBackend.TryChannelEpilogue(IGpuBuffer input, IGpuBuffer? bias, IGpuBuffer? scale,
        IGpuBuffer? shift, IGpuBuffer output, int batch, int channels, int spatial, int activation, float slope)
        => ChannelEpilogueReady && PortableChannelEpilogue.TryForward(this, LaunchChannelEpilogue, input, bias, scale, shift,
            output, batch, channels, spatial, activation, slope);

    bool IChannelEpilogueBackend.TryChannelEpilogueBackward(IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer? scale,
        IGpuBuffer gradInput, int batch, int channels, int spatial, int activation, float slope)
        => ChannelEpilogueReady && PortableChannelEpilogue.TryBackward(this, LaunchChannelEpilogue, gradOutput, output,
            scale, gradInput, batch, channels, spatial, activation, slope);
}
