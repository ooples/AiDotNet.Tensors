// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// A convolution's per-channel epilogue (bias, inference BatchNorm scale and shift, activation) as one pass each way, through
// PortableChannelEpilogue and ChannelEpilogueKernelTemplate's kernels as GLSL compute shaders, built on first need.
public sealed unsafe partial class VulkanBackend : IChannelEpilogueBackend
{
    private static readonly Dictionary<string, string> s_channelEpilogueShaders = BuildChannelEpilogueShaders();

    private static Dictionary<string, string> BuildChannelEpilogueShaders()
    {
        var shaders = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var name in ChannelEpilogueKernelTemplate.GetKernelNames())
            shaders[name] = ChannelEpilogueKernelTemplate.GetGlslSource(name);
        return shaders;
    }

    private bool? _channelEpilogueReady;

    // Both pipelines, built once; a shader the driver rejects leaves the separate passes serving, and says why.
    private bool ChannelEpilogueReady
    {
        get
        {
            if (_channelEpilogueReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_channelEpilogueReady is null)
                {
                    bool ok = true;
                    try
                    {
                        foreach (var spec in ChannelEpilogueKernelTemplate.Kernels)
                            if (GetOrCreateGlslPipeline(s_channelEpilogueShaders[spec.Name], spec.Buffers.Length,
                                    (uint)(spec.Ints.Length * sizeof(uint))) is null)
                            {
                                System.Diagnostics.Trace.TraceWarning(
                                    $"Vulkan channel epilogue kernels unavailable: '{spec.Name}' did not build.");
                                ok = false;
                                break;
                            }
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan channel epilogue kernels unavailable: {ex.Message}");
                        ok = false;
                    }
                    _channelEpilogueReady = ok;
                }
                return _channelEpilogueReady.Value;
            }
        }
    }

    private void LaunchChannelEpilogue(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
    {
        var push = new uint[ints.Length];
        for (int i = 0; i < ints.Length; i++) push[i] = unchecked((uint)ints[i]);
        // The shaders' local size is the group size, so the thread count dispatches exactly `groups` workgroups.
        TryDispatchConvPoolGlsl(s_channelEpilogueShaders[kernelName], push,
            groups * ChannelEpilogueKernelTemplate.GroupSize, buffers);
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
