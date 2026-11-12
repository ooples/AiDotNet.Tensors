// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The fused MoE kernels (IMoeKernelBackend): MoeKernelTemplate's OpenCL C module, compiled on first need.
    public sealed partial class OpenClBackend : IMoeKernelBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _moeReady;

        bool IMoeKernelBackend.MoeKernelsReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _moeReady);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_moeReady == 0) System.Threading.Volatile.Write(ref _moeReady, CompileMoeModule() ? 1 : 2);
                    return _moeReady == 1;
                }
            }
        }

        private bool CompileMoeModule()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                var program = CompileOrLoadCached(MoeKernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "fused MoE kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in MoeKernelTemplate.GetKernelNames())
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
                foreach (var name in MoeKernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < MoeKernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL fused MoE kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL fused MoE kernels unavailable: {ex.Message}");
                return false;
            }
        }

        void IMoeKernelBackend.LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * MoeKernelTemplate.GroupSize, MoeKernelTemplate.GroupSize);
        }
    }
}
