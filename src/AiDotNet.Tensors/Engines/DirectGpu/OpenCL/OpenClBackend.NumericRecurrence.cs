// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // The batched numeric recurrences (INumericRecurrenceBackend) on OpenCL, through PortableNumericRecurrence and
    // NumericRecurrenceKernelTemplate's kernels, compiled on first need.
    public sealed partial class OpenClBackend : INumericRecurrenceBackend
    {
        // 0 not yet asked, 1 ready, 2 declined.
        private int _numericRecurrenceReady;

        // The module, built once, and every kernel able to run its 256 work-items in one group; otherwise the host ops
        // run, and the trace says why.
        private bool NumericRecurrenceReady
        {
            get
            {
                int cached = System.Threading.Volatile.Read(ref _numericRecurrenceReady);
                if (cached != 0) return cached == 1;
                lock (_tunedSlotsLock)
                {
                    if (_numericRecurrenceReady == 0)
                        System.Threading.Volatile.Write(ref _numericRecurrenceReady, CompileNumericRecurrenceModule() ? 1 : 2);
                    return _numericRecurrenceReady == 1;
                }
            }
        }

        private bool CompileNumericRecurrenceModule()
        {
            var context = _context;
            if (context is null) return false;
            try
            {
                var program = CompileOrLoadCached(NumericRecurrenceKernelTemplate.GetModuleSource(WinogradKernelDialect.OpenCl),
                    OpenClBuildOptions.SafeMathFlags, "numeric recurrence kernels");
                var built = new System.Collections.Generic.List<(string, DirectOpenClKernel)>();
                try
                {
                    foreach (var name in NumericRecurrenceKernelTemplate.GetKernelNames())
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
                foreach (var name in NumericRecurrenceKernelTemplate.GetKernelNames())
                {
                    int size = KernelWorkGroupSize(name);
                    // Zero: the driver does not answer for the kernel, so the device's limit decides.
                    if (size == 0) size = (int)Math.Min(_maxWorkGroupSize, (ulong)int.MaxValue);
                    if (size < NumericRecurrenceKernelTemplate.GroupSize)
                    {
                        System.Diagnostics.Trace.TraceWarning(
                            $"OpenCL numeric recurrence kernels unavailable: '{name}' runs at most {size} work-items.");
                        return false;
                    }
                }
                return true;
            }
            catch (Exception ex) when (PortableNumericRecurrence.IsBuildFailure(ex))
            {
                System.Diagnostics.Trace.TraceWarning($"OpenCL numeric recurrence kernels unavailable: {ex.Message}");
                return false;
            }
        }

        private void LaunchNumericRecurrence(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints)
        {
            var kernel = _kernelCache[kernelName];
            uint arg = 0;
            foreach (var b in buffers) kernel.SetArg(arg++, BufferHandle(b));
            foreach (int v in ints) kernel.SetArg(arg++, v);
            kernel.Execute1D(groups * NumericRecurrenceKernelTemplate.GroupSize, NumericRecurrenceKernelTemplate.GroupSize);
        }

        bool INumericRecurrenceBackend.TryTridiagonalSolve(IGpuBuffer lower, IGpuBuffer diagonal, IGpuBuffer upper, IGpuBuffer rhs,
            IGpuBuffer x, IGpuBuffer workspace, int batch, int n)
            => NumericRecurrenceReady
               && PortableNumericRecurrence.TryTridiagonalSolve(LaunchNumericRecurrence, lower, diagonal, upper, rhs, x, workspace, batch, n);

        bool INumericRecurrenceBackend.TryFritschCarlsonLimit(IGpuBuffer slopes, IGpuBuffer secants, IGpuBuffer limited, int batch, int n)
            => NumericRecurrenceReady
               && PortableNumericRecurrence.TryFritschCarlsonLimit(LaunchNumericRecurrence, slopes, secants, limited, batch, n);

        bool INumericRecurrenceBackend.TryCtcForwardBackward(IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer lengths,
            IGpuBuffer logAlpha, IGpuBuffer logBeta, IGpuBuffer logLikelihood, int maxTime, int batch, int classes, int maxStates, int blank)
            => NumericRecurrenceReady
               && PortableNumericRecurrence.TryCtcForwardBackward(LaunchNumericRecurrence, logProbs, targets, lengths, logAlpha, logBeta,
                   logLikelihood, maxTime, batch, classes, maxStates, blank);

        bool INumericRecurrenceBackend.TryCtcLabelPosteriors(IGpuBuffer logAlpha, IGpuBuffer logBeta, IGpuBuffer targets,
            IGpuBuffer lengths, IGpuBuffer logLikelihood, IGpuBuffer gamma, int maxTime, int batch, int classes, int maxStates, int blank)
            => NumericRecurrenceReady
               && PortableNumericRecurrence.TryCtcLabelPosteriors(LaunchNumericRecurrence, logAlpha, logBeta, targets, lengths,
                   logLikelihood, gamma, maxTime, batch, classes, maxStates, blank);

        bool INumericRecurrenceBackend.TryLinearSumAssignment(IGpuBuffer cost, IGpuBuffer assignment, IGpuBuffer workspace,
            int batch, int size)
            => NumericRecurrenceReady
               && PortableNumericRecurrence.TryLinearSumAssignment(LaunchNumericRecurrence, cost, assignment, workspace, batch, size);
    }
}
