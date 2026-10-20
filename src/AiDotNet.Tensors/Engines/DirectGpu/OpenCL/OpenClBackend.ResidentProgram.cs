using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1449: the resident program on OpenCL. OpenCL has no cooperative launch and no forward-progress guarantee across
    // work-groups, so a persistent kernel spinning on a grid barrier can deadlock when not every work-group is
    // resident, and no OpenCL query proves they are. A row-local chain is instead ONE kernel in which each work-group
    // takes a row through every stage, with only work-group barriers (resident_rows); any other chain is its stages as
    // ordinary commands on the in-order queue. Flushing every chain at once measured slower than leaving it to the
    // flush policy (0.41-0.95x on an RX 5500 XT), so neither path forces a flush. Separately, each stage is enqueued
    // and flushed on its own. The kernels share res_run_stage's per-element arithmetic, so all give bit-identical
    // results.
    public sealed partial class OpenClBackend
    {
        private readonly object _residentProgramLock = new();
        private bool _residentProgramTried, _residentProgramAvailable;
        private string? _residentProgramError;

        /// <summary>Whether this device can run a resident program (its stage kernel compiled).</summary>
        internal bool SupportsResidentProgram => IsAvailable && EnsureResidentProgramKernel();

        /// <summary>
        /// Runs <paramref name="stages"/> in order. A row-local chain (<see cref="ResidentProgramKernelTemplate.TryPlanRowProgram"/>)
        /// is ONE kernel, each work-group taking a row through every stage; any other chain is its stages enqueued as
        /// ordinary commands. Either way the flush policy decides when the queue is flushed, as for any op. Returns the
        /// number of kernels enqueued.
        /// </summary>
        internal int RunResidentProgram(ReadOnlySpan<ResidentStage> stages)
        {
            ResidentProgramKernelTemplate.RequireChain(stages);
            var kernel = ResidentStageKernel();
            if (ResidentProgramKernelTemplate.TryPlanRowProgram(stages, out var plan) && plan is not null)
            {
                var rows = _kernelCache[ResidentProgramKernelTemplate.RowProgramKernel];
                for (int i = 0; i < ResidentProgramKernelTemplate.MaxRowProgramBuffers; i++)
                    rows.SetArg((uint)i, BufferHandle(i < plan.Buffers.Length ? plan.Buffers[i] : plan.Buffers[0]));
                var p = plan.Parameters();
                for (int i = 0; i < p.Length; i++) rows.SetArg((uint)(ResidentProgramKernelTemplate.MaxRowProgramBuffers + i), p[i]);
                rows.Execute1D(plan.Groups * ResidentProgramKernelTemplate.GroupThreads, ResidentProgramKernelTemplate.GroupThreads);
                return 1;
            }
            foreach (var stage in stages) EnqueueResidentStage(kernel, stage);
            return stages.Length;
        }

        /// <summary>
        /// The same stages, each enqueued and flushed on its own (what a resident program is measured against). With
        /// <paramref name="flushEachStage"/> false the stages are only enqueued and the flush policy decides, which is
        /// what the backend does for ordinary ops (the benchmark's third column).
        /// </summary>
        internal void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages, bool flushEachStage = true)
        {
            ResidentProgramKernelTemplate.RequireChain(stages);
            var kernel = ResidentStageKernel();
            foreach (var stage in stages)
            {
                EnqueueResidentStage(kernel, stage);
                if (flushEachStage) FlushQueue();
            }
        }

        private static void EnqueueResidentStage(DirectOpenClKernel kernel, in ResidentStage s)
        {
            // An unread operand still needs a valid buffer: the input stands in.
            kernel.SetArg(0, BufferHandle(s.A));
            kernel.SetArg(1, BufferHandle(s.B ?? s.A));
            kernel.SetArg(2, BufferHandle(s.C ?? s.A));
            kernel.SetArg(3, BufferHandle(s.Output));
            kernel.SetArg(4, (int)s.Op);
            kernel.SetArg(5, s.Rows);
            kernel.SetArg(6, s.Cols);
            int groups = ResidentProgramKernelTemplate.GroupsFor(s);
            kernel.Execute1D(groups * ResidentProgramKernelTemplate.GroupThreads, ResidentProgramKernelTemplate.GroupThreads);
        }

        private void FlushQueue()
        {
            if (_context is null) throw new GpuCapabilityUnavailableException("Context not available");
            int err = OpenClNativeBindings.Flush(_context.CommandQueue);
            if (err != OpenClNativeBindings.CL_SUCCESS) throw new InvalidOperationException($"clFlush failed: {err}");
            DirectOpenClFlushPolicy.OnFlushed();
        }

        private DirectOpenClKernel ResidentStageKernel()
        {
            if (!EnsureResidentProgramKernel())
                throw new GpuCapabilityUnavailableException("OpenCL resident program kernel is unavailable"
                    + (_residentProgramError is null ? "." : ": " + _residentProgramError));
            return _kernelCache[ResidentProgramKernelTemplate.StageKernel];
        }

        private bool EnsureResidentProgramKernel()
        {
            if (_residentProgramTried) return _residentProgramAvailable;
            lock (_residentProgramLock)
            {
                if (_residentProgramTried) return _residentProgramAvailable;
                try
                {
                    if (_context is null) throw new GpuCapabilityUnavailableException("Context not available");
                    // The kernel requires a 256-wide work-group (a softmax row is reduced by one group of that size).
                    if (_context.MaxWorkGroupSize < (ulong)ResidentProgramKernelTemplate.GroupThreads)
                        throw new GpuCapabilityUnavailableException(
                            $"the device's largest work-group is {_context.MaxWorkGroupSize}; the stage kernel needs {ResidentProgramKernelTemplate.GroupThreads}.");
                    // Strict math: the softmax's exp and division must round like the other backends' instances.
                    var program = CompileOrLoadCached(ResidentProgramKernelTemplate.GetModuleSource(ResidentKernelDialect.OpenCl),
                        OpenClBuildOptions.SafeMathFlags, "resident program kernel");
                    _programs.Add(program);
                    foreach (var name in ResidentProgramKernelTemplate.GetKernelNames(ResidentKernelDialect.OpenCl))
                        _kernelCache[name] = new DirectOpenClKernel(_context, program, name);
                    _residentProgramAvailable = true;
                }
                catch (Exception ex) when (ex is not OutOfMemoryException)
                {
                    _residentProgramError = ex.Message;
                    System.Diagnostics.Trace.TraceWarning($"OpenCL resident program kernel unavailable: {ex.Message}");
                }
                _residentProgramTried = true;
                return _residentProgramAvailable;
            }
        }
    }
}
