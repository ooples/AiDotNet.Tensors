using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1450: the resident program on Vulkan. Vulkan compute guarantees no forward progress across workgroups, so a
// persistent dispatch spinning on a grid barrier is unsafe in general. A row-local chain is instead ONE dispatch in
// which each workgroup takes a row through every stage (resident_rows); separately, each stage is its own submission.
// The shaders share the stage shader's per-element arithmetic, so all give bit-identical results.
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
    /// Runs <paramref name="stages"/> in order. A row-local chain (<see cref="ResidentProgramKernelTemplate.TryPlanRowProgram"/>)
    /// is ONE dispatch in which each workgroup takes a row through every stage, with only workgroup barriers; any other
    /// chain is its stages as ordinary dispatches. Both go through the backend's ordinary submission (coalesced into
    /// the open command buffer, or into a batch the caller opened): forcing a submission per chain measured slower
    /// (0.48-0.96x on an RX 5500 XT). Returns the number of dispatches recorded.
    /// </summary>
    internal int RunResidentProgram(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        RequireResidentProgram();
        if (ResidentProgramKernelTemplate.TryPlanRowProgram(stages, out var plan) && plan is not null)
        {
            string shader = ResidentProgramKernelTemplate.GetRowProgramShader(ResidentKernelDialect.Glsl, plan.Buffers.Length);
            if (GetOrCreateGlslPipeline(shader, plan.Buffers.Length, ResidentRowPushBytes) is not null)
            {
                var p = plan.Parameters();
                var push = new uint[p.Length];
                for (int i = 0; i < p.Length; i++) push[i] = (uint)p[i];
                TryDispatchConvPoolGlsl(shader, push, plan.Groups * ResidentProgramKernelTemplate.GroupThreads, plan.Buffers);
                return 1;
            }
        }
        foreach (var stage in stages) DispatchResidentStage(stage);
        return stages.Length;
    }

    private const uint ResidentRowPushBytes = (4 + ResidentProgramKernelTemplate.MaxStages) * sizeof(uint);

    /// <summary>
    /// The same stages, each its own submission (what a resident program is measured against). With
    /// <paramref name="submitEachStage"/> false the stages go through the ordinary dispatch path, which coalesces
    /// unwaited dispatches into the open command buffer (the benchmark's third column).
    /// </summary>
    internal void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages, bool submitEachStage = true)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        RequireResidentProgram();
        foreach (var stage in stages)
        {
            DispatchResidentStage(stage);
            // An unwaited dispatch would otherwise wait in the open command buffer for the next ones.
            if (submitEachStage) _device.SubmitOpenCommands();
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
