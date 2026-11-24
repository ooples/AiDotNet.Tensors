// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // A convolution's per-channel epilogue (bias, inference BatchNorm scale and shift, activation) as one pass each way,
    // through PortableChannelEpilogue and ChannelEpilogueKernelTemplate's kernels, compiled on first need.
    public sealed partial class OpenClBackend : IChannelEpilogueBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _channelEpilogueReady;

        private bool ChannelEpilogueReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _channelEpilogueReady);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_channelEpilogueReady == 0)
                        System.Threading.Volatile.Write(ref _channelEpilogueReady, CompileChannelEpilogueModule() ? 1 : 2);
                    return _channelEpilogueReady == 1;
                }
            }
        }

        // The module, built once, and both kernels able to run 256 work-items in one group; otherwise the separate
        // passes serve, and the trace says why.
        private bool CompileChannelEpilogueModule()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                var program = CompileOrLoadCached(ChannelEpilogueKernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "channel epilogue kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in ChannelEpilogueKernelTemplate.GetKernelNames())
                        built.Add((name, new DirectOpenClKernel(context, program, name)));
                }
                catch
                {
                    foreach (var (_, kernel) in built) kernel.Dispose();
                    program.Dispose();
                    throw;
                }
                _programs.Add(program);
                foreach (var (name, kernel) in built) _kernelCache[name] = kernel;
                foreach (var name in ChannelEpilogueKernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < ChannelEpilogueKernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            $"OpenCL channel epilogue kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL channel epilogue kernels unavailable: {ex.Message}");
                return false;
            }
        }

        private void LaunchChannelEpilogue(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * ChannelEpilogueKernelTemplate.GroupSize, ChannelEpilogueKernelTemplate.GroupSize);
        }

        bool IChannelEpilogueBackend.TryChannelEpilogue(IGpuBuffer input, IGpuBuffer? bias, IGpuBuffer? scale,
            IGpuBuffer? shift, IGpuBuffer output, int batch, int channels, int spatial, int activation, float slope)
            => ChannelEpilogueReady && PortableChannelEpilogue.TryForward(this, LaunchChannelEpilogue, input, bias, scale,
                shift, output, batch, channels, spatial, activation, slope);

        bool IChannelEpilogueBackend.TryChannelEpilogueBackward(IGpuBuffer gradOutput, IGpuBuffer output,
            IGpuBuffer? scale, IGpuBuffer gradInput, int batch, int channels, int spatial, int activation, float slope)
            => ChannelEpilogueReady && PortableChannelEpilogue.TryBackward(this, LaunchChannelEpilogue, gradOutput, output,
                scale, gradInput, batch, channels, spatial, activation, slope);
    }
}
