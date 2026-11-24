#if NET7_0_OR_GREATER
// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// A convolution's per-channel epilogue (bias, inference BatchNorm scale and shift, activation) as one pass each way, through
// PortableChannelEpilogue and the WGSL restatement of ChannelEpilogueKernelTemplate (WebGpuChannelEpilogueKernels), built on
// first need.
public sealed partial class WebGpuBackend : IChannelEpilogueBackend
{
    private const string ChannelEpilogueModulePrefix = "ChannelEpilogue_";
    private bool? _channelEpilogueReady;

    // Both pipelines, built once; a module the implementation rejects leaves the separate passes serving, and says why.
    private bool ChannelEpilogueReady
    {
        get
        {
            if (_channelEpilogueReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_channelEpilogueReady is null)
                {
                    bool ok = IsAvailable;
                    if (ok)
                    {
                        try
                        {
                            foreach (var name in ChannelEpilogueKernelTemplate.GetKernelNames())
                                GetOrCreatePipelineAsync(ChannelEpilogueModulePrefix + name,
                                    WebGpuChannelEpilogueKernels.SourceFor(name), name).GetAwaiter().GetResult();
                        }
                        catch (Exception ex)
                        {
                            System.Diagnostics.Trace.TraceWarning($"WebGPU channel epilogue kernels unavailable: {ex.Message}");
                            ok = false;
                        }
                    }
                    _channelEpilogueReady = ok;
                }
                return _channelEpilogueReady.Value;
            }
        }
    }

    private void LaunchChannelEpilogue(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        // The uniform struct holds the ints (bit-cast) and its padding words.
        var uniforms = new float[WebGpuChannelEpilogueKernels.UniformWords];
        for (int i = 0; i < ints.Length; i++) uniforms[i] = BitConverter.Int32BitsToSingle(ints[i]);
        DispatchNBufferAsync(ChannelEpilogueModulePrefix + kernelName, WebGpuChannelEpilogueKernels.SourceFor(kernelName),
            kernelName, buffers, uniforms, groups * ChannelEpilogueKernelTemplate.GroupSize).GetAwaiter().GetResult();
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
#endif
