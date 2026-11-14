// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // Raw bytes and numeric text into float32 (IDataDecodeBackend) on OpenCL, through PortableDataDecode and
    // DataDecodeKernelTemplate's kernels, compiled on first need.
    public sealed partial class OpenClBackend : IDataDecodeBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _dataDecodeReady;

        // The module, built once, and every kernel able to run its 256 work-items in one group; otherwise the host op
        // runs, and the trace says why.
        private bool DataDecodeReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _dataDecodeReady);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_dataDecodeReady == 0)
                        System.Threading.Volatile.Write(ref _dataDecodeReady, CompileDataDecodeModule() ? 1 : 2);
                    return _dataDecodeReady == 1;
                }
            }
        }

        private bool CompileDataDecodeModule()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                // Integer code only: no math flag changes a result.
                var program = CompileOrLoadCached(DataDecodeKernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.StrictMathFlags, "data decode kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in DataDecodeKernelTemplate.GetKernelNames())
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
                foreach (var name in DataDecodeKernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < DataDecodeKernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning($"OpenCL data decode kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex) when (PortableDataDecode.IsBuildFailure(ex))
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL data decode kernels unavailable: {ex.Message}");
                return false;
            }
        }

        private void LaunchDataDecode(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * DataDecodeKernelTemplate.GroupSize, DataDecodeKernelTemplate.GroupSize);
        }

        bool IDataDecodeBackend.TryDecodeBytes(IGpuBuffer words, IGpuBuffer output, int count, NumericStorageType sourceType,
            bool littleEndian)
            => DataDecodeReady && PortableDataDecode.TryDecode(LaunchDataDecode, words, output, count, sourceType, littleEndian);

        bool IDataDecodeBackend.TryCountSeparators(IGpuBuffer words, IGpuBuffer segmentCounts, int length, byte delimiter)
            => DataDecodeReady && PortableDataDecode.TryCountSeparators(LaunchDataDecode, words, segmentCounts, length, delimiter);

        bool IDataDecodeBackend.TryExclusiveScan(IGpuBuffer values, IGpuBuffer total, int count)
            => DataDecodeReady && PortableDataDecode.TryScan(LaunchDataDecode, values, total, count);

        bool IDataDecodeBackend.TryScatterFieldBounds(IGpuBuffer words, IGpuBuffer segmentOffsets, IGpuBuffer bounds,
            int length, byte delimiter, bool trailingNewline)
            => DataDecodeReady && PortableDataDecode.TryScatterBounds(LaunchDataDecode, words, segmentOffsets, bounds, length,
                delimiter, trailingNewline);

        bool IDataDecodeBackend.TryParseFields(IGpuBuffer words, IGpuBuffer bounds, IGpuBuffer output, IGpuBuffer status,
            int fields, int styleMask, float missingValue)
            => DataDecodeReady && PortableDataDecode.TryParse(LaunchDataDecode, words, bounds, output, status, fields, styleMask,
                missingValue);

        bool IDataDecodeBackend.TryEncodeBytes(IGpuBuffer input, IGpuBuffer words, int count, NumericStorageType targetType,
            bool littleEndian)
            => DataDecodeReady && PortableDataDecode.TryEncode(LaunchDataDecode, input, words, count, targetType, littleEndian);
    }
}
