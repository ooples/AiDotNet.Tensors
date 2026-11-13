// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The sweep data transforms (IDataTransformBackend) on OpenCL, through PortableDataTransform and
    // DataTransformKernelTemplate's kernels, compiled on first need.
    public sealed partial class OpenClBackend : IDataTransformBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _dataTransformReady;

        // The module, built once, and every kernel able to run its 256 work-items in one group; otherwise the host ops
        // run, and the trace says why.
        private bool DataTransformReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _dataTransformReady);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_dataTransformReady == 0)
                        System.Threading.Volatile.Write(ref _dataTransformReady, CompileDataTransformModule() ? 1 : 2);
                    return _dataTransformReady == 1;
                }
            }
        }

        private bool CompileDataTransformModule()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                var program = CompileOrLoadCached(DataTransformKernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "data transform kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in DataTransformKernelTemplate.GetKernelNames())
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
                foreach (var name in DataTransformKernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < DataTransformKernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            $"OpenCL data transform kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex) when (PortableDataTransform.IsBuildFailure(ex))
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL data transform kernels unavailable: {ex.Message}");
                return false;
            }
        }

        private void LaunchDataTransform(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * DataTransformKernelTemplate.GroupSize, DataTransformKernelTemplate.GroupSize);
        }

        bool IDataTransformBackend.TryBoxMuller(IGpuBuffer uniforms, IGpuBuffer output, int count, BoxMullerBranch branch,
            float mean, float std, bool clamp, float low, float high)
            => DataTransformReady
               && PortableDataTransform.TryBoxMuller(LaunchDataTransform, uniforms, output, count, branch, mean, std, clamp, low, high);

        bool IDataTransformBackend.TryUniformThreshold(IGpuBuffer uniforms, IGpuBuffer? thresholds, IGpuBuffer output, int count,
            bool above, float threshold, float onSuccess, float otherwise)
            => DataTransformReady
               && PortableDataTransform.TryUniformThreshold(LaunchDataTransform, uniforms, thresholds, output, count, above, threshold,
                   onSuccess, otherwise);

        bool IDataTransformBackend.TryLinearFilter(IGpuBuffer x, IGpuBuffer b, IGpuBuffer a, IGpuBuffer? zi, IGpuBuffer y,
            IGpuBuffer workspace, int signals, int length, int nb, int na, bool guard, float maxAbs)
            => DataTransformReady
               && PortableDataTransform.TryLinearFilter(LaunchDataTransform, x, b, a, zi, y, workspace, signals, length, nb, na, guard,
                   maxAbs);

        bool IDataTransformBackend.TryGatherRows(IGpuBuffer table, IGpuBuffer ids, IGpuBuffer output, int rows, int rowSize)
            => DataTransformReady && PortableDataTransform.TryGatherRows(LaunchDataTransform, table, ids, output, rows, rowSize);

        bool IDataTransformBackend.TryScatterRows(IGpuBuffer source, IGpuBuffer ids, IGpuBuffer table, int rows, int rowSize)
            => DataTransformReady && PortableDataTransform.TryScatterRows(LaunchDataTransform, source, ids, table, rows, rowSize);

        bool IDataTransformBackend.TryRoundIntegral(IGpuBuffer input, IGpuBuffer output, int count, RoundingMode mode, float low,
            float high)
            => DataTransformReady && PortableDataTransform.TryRoundIntegral(LaunchDataTransform, input, output, count, mode, low, high);
    }
}
