using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1450: the resident program on Vulkan. Vulkan compute guarantees no forward progress across workgroups, so a
// persistent dispatch spinning on a grid barrier is unsafe in general. The chain is instead every stage recorded into
// ONE command buffer (the backend's batch mode) with a compute memory barrier between dispatches, submitted once: no
// host round trip between stages. Separately, each stage is its own submission. Both run
// ResidentProgramKernelTemplate's stage shader, so the two give bit-identical results.
public sealed unsafe partial class VulkanBackend
{
    private bool? _residentProgramReady;

    /// <summary>Whether this device can run a resident program (its stage shader built).</summary>
    internal bool SupportsResidentProgram
    {
        get
        {
            if (!IsAvailable) return false;
            if (_residentProgramReady is { } ready) return ready;
            lock (_computeLock)
            {
                if (_residentProgramReady is null)
                {
                    try
                    {
                        _residentProgramReady = GetOrCreateGlslPipeline(ResidentStageShader, 4, ResidentPushBytes) is not null;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Vulkan resident program shader unavailable: {ex.Message}");
                        _residentProgramReady = false;
                    }
                }
                return _residentProgramReady.Value;
            }
        }
    }

    private static readonly string ResidentStageShader = ResidentProgramKernelTemplate.GetModuleSource(ResidentKernelDialect.Glsl);
    private const uint ResidentPushBytes = 3 * sizeof(uint);

    /// <summary>
    /// Runs <paramref name="stages"/> in order as one command buffer and one submission, which this call waits for.
    /// Inside a batch the caller already opened, the stages join that batch instead. Returns the number of stage
    /// dispatches recorded.
    /// </summary>
    internal int RunResidentProgram(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        RequireResidentProgram();
        lock (_computeLock)
        {
            if (_batchMode)
            {
                foreach (var stage in stages) DispatchResidentStage(stage);
                return stages.Length;
            }
            BeginBatch();
            try
            {
                foreach (var stage in stages) DispatchResidentStage(stage);
            }
            finally
            {
                EndBatch();
            }
        }
        return stages.Length;
    }

    /// <summary>The same stages, each its own submission (what a resident program is measured against).</summary>
    internal void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        RequireResidentProgram();
        foreach (var stage in stages)
        {
            DispatchResidentStage(stage);
            // An unwaited dispatch would otherwise wait in the open command buffer for the next ones.
            _device.SubmitOpenCommands();
        }
    }

    private void RequireResidentProgram()
    {
        if (!SupportsResidentProgram)
            throw new GpuCapabilityUnavailableException("The Vulkan resident program shader is unavailable.");
    }

    private void DispatchResidentStage(in ResidentStage s)
    {
        var push = new uint[] { (uint)s.Op, (uint)s.Rows, (uint)s.Cols };
        int threads = ResidentProgramKernelTemplate.GroupsFor(s) * ResidentProgramKernelTemplate.GroupThreads;
        // An unread operand still needs a bound buffer: the input stands in.
        TryDispatchConvPoolGlsl(ResidentStageShader, push, threads, s.A, s.B ?? s.A, s.C ?? s.A, s.Output);
    }
}
