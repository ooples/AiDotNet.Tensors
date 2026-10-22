// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1481: the stride-2 input gradients (3 x 3 four-phase, 1 x 1, and the compiled plan's projection-shortcut pair) on
    // OpenCL, through PortableConvDgradS2 and ConvDgradS2KernelTemplate's kernels, compiled on first need.
    public sealed partial class OpenClBackend : IConvShortcutPairBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _dgradS2Ready;

        // The module, built once, and every shared-memory kernel able to run its 256 work-items in one group (a kernel
        // capped below that would fail the launch with CL_INVALID_WORK_GROUP_SIZE); otherwise the other input-gradient
        // routes serve these shapes, and the trace says why.
        private bool DgradS2Ready
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _dgradS2Ready);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_dgradS2Ready == 0) System.Threading.Volatile.Write(ref _dgradS2Ready, CompileDgradS2Module() ? 1 : 2);
                    return _dgradS2Ready == 1;
                }
            }
        }

        private bool CompileDgradS2Module()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                var program = CompileOrLoadCached(ConvDgradS2KernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "stride-2 input-gradient kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in ConvDgradS2KernelTemplate.GetKernelNames())
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
                foreach (var name in ConvDgradS2KernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < ConvDgradS2KernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            $"OpenCL stride-2 input-gradient kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL stride-2 input-gradient kernels unavailable: {ex.Message}");
                return false;
            }
        }

        private void LaunchDgradS2(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * ConvDgradS2KernelTemplate.GroupSize, ConvDgradS2KernelTemplate.GroupSize);
        }

        private int DgradS2GroupTarget => ((IWinogradFusedFilterGradient)this).FusedFilterGradientGroupTarget;

        // Per-route entry points, for the slot candidates and the tests. False: nothing launched.
        internal bool TryConv2DBackwardInputS2K3(in TunedConv2DBackwardArgs a, int splits = 0)
            => PortableConvDgradS2.AdmitsK3(a) && DgradS2Ready
               && PortableConvDgradS2.TryK3(this, LaunchDgradS2, a, DgradS2GroupTarget, splits);

        internal bool TryConv2DBackwardInputS2K1(in TunedConv2DBackwardArgs a, int splits = 0)
            => PortableConvDgradS2.AdmitsK1(a) && DgradS2Ready
               && PortableConvDgradS2.TryK1(this, LaunchDgradS2, a, DgradS2GroupTarget, splits);

        internal bool TryConv2DBackwardInputS2K3Pair(in TunedConv2DBackwardArgs a, IGpuBuffer dyShortcut,
            IGpuBuffer wShortcut, int shortcutChannels, int splits = 0)
            => PortableConvDgradS2.AdmitsK3(a) && DgradS2Ready
               && PortableConvDgradS2.TryPair(this, LaunchDgradS2, a, dyShortcut, wShortcut, shortcutChannels,
                   DgradS2GroupTarget, splits);

        bool IConvShortcutPairBackend.SupportsConv2DBackwardInputShortcutPair(int batch, int inChannels, int inHeight,
            int inWidth, int outChannels, int shortcutChannels, int outHeight, int outWidth)
            => PortableConvDgradS2.AdmitsPair(batch, inChannels, inHeight, inWidth, outChannels, shortcutChannels,
                outHeight, outWidth) && DgradS2Ready;

        bool IConvShortcutPairBackend.TryConv2DBackwardInputShortcutPair(IGpuBuffer gradOutput, IGpuBuffer kernel,
            IGpuBuffer gradShortcut, IGpuBuffer kernelShortcut, IGpuBuffer gradInput, int batch, int inChannels,
            int inHeight, int inWidth, int outChannels, int shortcutChannels, int outHeight, int outWidth)
            => TryConv2DBackwardInputS2K3Pair(new TunedConv2DBackwardArgs(gradOutput, kernel, gradInput, false, batch,
                inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, 3, 3, 2, 2, 1, 1, 1, 1), gradShortcut,
                kernelShortcut, shortcutChannels);
    }
}
