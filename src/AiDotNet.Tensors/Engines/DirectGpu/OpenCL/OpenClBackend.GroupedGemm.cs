// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The jagged-rows grouped GEMM (IGroupedGemmBackend) on OpenCL, through PortableGroupedGemm and
    // GroupedGemmKernelTemplate's kernels, compiled on first need.
    public sealed partial class OpenClBackend : IGroupedGemmBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _groupedGemmReady;

        // The module, built once, and both kernels able to run their 256 work-items in one group; otherwise the host op
        // runs, and the trace says why.
        private bool GroupedGemmReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _groupedGemmReady);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_groupedGemmReady == 0) System.Threading.Volatile.Write(ref _groupedGemmReady, CompileGroupedGemmModule() ? 1 : 2);
                    return _groupedGemmReady == 1;
                }
            }
        }

        private bool CompileGroupedGemmModule()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                var program = CompileOrLoadCached(GroupedGemmKernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "grouped GEMM kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in GroupedGemmKernelTemplate.GetKernelNames())
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
                foreach (var name in GroupedGemmKernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < GroupedGemmKernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL grouped GEMM kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex) when (PortableGroupedGemm.IsBuildFailure(ex))
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL grouped GEMM kernels unavailable: {ex.Message}");
                return false;
            }
        }

        private void LaunchGroupedGemm(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * GroupedGemmKernelTemplate.GroupSize, GroupedGemmKernelTemplate.GroupSize);
        }

        bool IGroupedGemmBackend.TryGroupedRowsGemm(IGpuBuffer a, IGpuBuffer b, IGpuBuffer y, int[] ends, int m, int n, int k,
            bool transposeB, IGpuBuffer? bias, bool biasPerGroup)
            => GroupedGemmReady
               && PortableGroupedGemm.TryForward(this, LaunchGroupedGemm, a, b, y, ends, m, n, k, transposeB, bias, biasPerGroup);

        bool IGroupedGemmBackend.TryGroupedRowsWeightGrad(IGpuBuffer a, IGpuBuffer c, IGpuBuffer w, int[] ends, int m, int k, int n)
            => GroupedGemmReady && PortableGroupedGemm.TryWeightGrad(this, LaunchGroupedGemm, a, c, w, ends, m, k, n);
    }
}
