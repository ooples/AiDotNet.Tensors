using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL
{
    // #1449: the resident program on OpenCL. OpenCL has no cooperative launch and no forward-progress guarantee across
    // work-groups, so a persistent kernel spinning on a grid barrier can deadlock when not every work-group is
    // resident, and no OpenCL query proves they are. The chain is instead every stage enqueued on the in-order queue
    // and handed to the device in ONE flush (the flush policy held meanwhile); the queue's order is the barrier.
    // Separately, each stage is enqueued and flushed on its own. Both use ResidentProgramKernelTemplate's stage kernel,
    // so the two give bit-identical results.
    public sealed partial class OpenClBackend
    {
        private readonly object _residentProgramLock = new();
        private bool _residentProgramTried, _residentProgramAvailable;
        private string? _residentProgramError;

        /// <summary>Whether this device can run a resident program (its stage kernel compiled).</summary>
        internal bool SupportsResidentProgram => IsAvailable && EnsureResidentProgramKernel();

        /// <summary>
        /// Runs <paramref name="stages"/> in order as one submission: every stage enqueued, then one flush. Returns
        /// the number of stage dispatches enqueued (one per stage).
        /// </summary>
        internal int RunResidentProgram(ReadOnlySpan<ResidentStage> stages)
        {
            ResidentProgramKernelTemplate.RequireChain(stages);
            var kernel = ResidentStageKernel();
            DirectOpenClFlushPolicy.Hold();
            try
            {
                foreach (var stage in stages) EnqueueResidentStage(kernel, stage);
            }
            finally
            {
                DirectOpenClFlushPolicy.Release();
            }
            FlushQueue();
            return stages.Length;
        }

        /// <summary>The same stages, each enqueued and flushed on its own (what a resident program is measured
        /// against).</summary>
        internal void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages)
        {
            ResidentProgramKernelTemplate.RequireChain(stages);
            var kernel = ResidentStageKernel();
            foreach (var stage in stages)
            {
                EnqueueResidentStage(kernel, stage);
                FlushQueue();
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
