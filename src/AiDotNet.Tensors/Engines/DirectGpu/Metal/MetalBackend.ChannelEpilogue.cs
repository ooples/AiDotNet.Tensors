// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU. The template emits the same bodies as the
// OpenCL instance, whose ChannelEpiloguePortTests run on a real device.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// A convolution's per-channel epilogue (bias, inference BatchNorm scale and shift, activation) as one pass each way, through
// PortableChannelEpilogue and ChannelEpilogueKernelTemplate's kernels, compiled as their own library on first need.
public sealed partial class MetalBackend : IChannelEpilogueBackend
{
    private const string ChannelEpilogueLibraryName = "ChannelEpilogue";
    private IntPtr _channelEpilogueLibrary;
    private bool? _channelEpilogueReady;

    // The library, compiled once, and both kernels able to run 256 threads in one threadgroup; otherwise the separate
    // passes serve, and the trace says why.
    private bool ChannelEpilogueReady
    {
        get
        {
            if (_channelEpilogueReady is { } ready) return ready;
            lock (_tunedSlotsLock)
            {
                if (_channelEpilogueReady is null)
                {
                    bool ok;
                    try
                    {
                        _channelEpilogueLibrary = _shaderLibrary.CompileLibrary(ChannelEpilogueLibraryName,
                            ChannelEpilogueKernelTemplate.GetModuleSource(WinogradKernelDialect.Metal));
                        ok = _channelEpilogueLibrary != IntPtr.Zero;
                        if (ok)
                            foreach (var name in ChannelEpilogueKernelTemplate.GetKernelNames())
                                ok &= GetPipeline(ChannelEpilogueLibraryName, _channelEpilogueLibrary, name)
                                    .MaxTotalThreadsPerThreadgroup >= ChannelEpilogueKernelTemplate.GroupSize;
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal channel epilogue kernels unavailable: {ex.Message}");
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
        var pipeline = GetPipeline(ChannelEpilogueLibraryName, _channelEpilogueLibrary, kernelName);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        for (int i = 0; i < buffers.Length; i++) encoder.SetBuffer((MetalGpuBuffer)buffers[i], i);
        for (int i = 0; i < ints.Length; i++) encoder.SetBytes((uint)ints[i], (uint)(buffers.Length + i));
        encoder.DispatchThreadgroups(new MTLSize((ulong)groups, 1, 1),
            new MTLSize((ulong)ChannelEpilogueKernelTemplate.GroupSize, 1, 1));
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
