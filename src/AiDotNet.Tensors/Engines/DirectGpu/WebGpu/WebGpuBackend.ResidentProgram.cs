#if NET7_0_OR_GREATER
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1452: the resident program on WebGPU. WebGPU has no cooperative launch, no cross-workgroup barrier and no
// forward-progress guarantee. A row-local chain is ONE dispatch of resident_rows, each workgroup taking a row through
// every stage with workgroup barriers; any other chain is every stage dispatch in ONE compute pass of one command
// encoder, submitted once (dispatchComputeChainWithUniforms): WebGPU orders a pass's dispatches.
// Separately, each stage is its own pass and submission (the backend's usual dispatch). Both run
// ResidentProgramKernelTemplate's stage shader, so the two give bit-identical results. Like the backend's other
// synchronous capabilities, both wait for their submissions.
public sealed partial class WebGpuBackend
{
    private const string ResidentProgramModuleKey = "ResidentProgram:stage";
    private const string ResidentRowModuleKey = "ResidentProgram:rows:";
    private static readonly string ResidentStageShader = ResidentProgramKernelTemplate.GetModuleSource(ResidentKernelDialect.Wgsl);
    private bool? _residentProgramReady;
    private string? _residentProgramError;

    /// <summary>Why the stage shader did not build, when it did not.</summary>
    internal string? ResidentProgramError => _residentProgramError;
    private readonly object _residentProgramLock = new object();

    /// <summary>Whether this device can run a resident program (its stage shader built).</summary>
    internal bool SupportsResidentProgram
    {
        get
        {
            if (!IsAvailable) return false;
            if (_residentProgramReady is { } ready) return ready;
            lock (_residentProgramLock)
            {
                if (_residentProgramReady is null)
                {
                    try
                    {
                        ResidentStagePipelineAsync().GetAwaiter().GetResult();
                        _residentProgramReady = true;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        _residentProgramError = ex.Message;
                        System.Diagnostics.Trace.TraceWarning($"WebGPU resident program shader unavailable: {ex.Message}");
                        _residentProgramReady = false;
                    }
                }
                return _residentProgramReady.Value;
            }
        }
    }

    /// <summary>
    /// Runs <paramref name="stages"/> in order as one submission, which this call waits for: one dispatch for a
    /// row-local chain, one per stage in one compute pass otherwise. Returns the number of dispatches.
    /// </summary>
    internal int RunResidentProgram(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        RequireResidentProgram();
        var copy = stages.ToArray();
        RunResidentProgramAsync(copy).GetAwaiter().GetResult();
        return ResidentProgramKernelTemplate.TryPlanRowProgram(copy, out _) ? 1 : copy.Length;
    }

    /// <summary>The same stages, each its own pass and submission (what a resident program is measured against).</summary>
    internal void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        RequireResidentProgram();
        RunStagesSeparatelyAsync(stages.ToArray()).GetAwaiter().GetResult();
    }

    private async Task RunResidentProgramAsync(ResidentStage[] stages)
    {
        if (ResidentProgramKernelTemplate.TryPlanRowProgram(stages, out var plan) && plan is not null)
        {
            // Each distinct buffer is bound once, read-write (the shader indexes them by slot): no buffer appears in
            // two bindings, so WebGpuBindGroup takes no snapshot and nothing aliases a writable binding.
            int rowsPipe = await GetOrCreatePipelineAsync(ResidentRowModuleKey + plan.Buffers.Length + ":" + string.Join(",", plan.StageWords),
                ResidentProgramKernelTemplate.GetRowProgramShader(ResidentKernelDialect.Wgsl, plan), "main");
            var bound = new WebGpuBuffer[plan.Buffers.Length];
            for (int i = 0; i < bound.Length; i++) bound[i] = AsWgpu(plan.Buffers[i]);
            using var uniforms = new WebGpuBuffer(UniformInts(plan.Parameters()), WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
            using var bind = new WebGpuBindGroup(rowsPipe, bound);
            await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(rowsPipe, bind.BindGroupId, uniforms.BufferId, plan.Groups, 1, 1);
            await WebGpuNativeBindings.SubmitAndWaitAsync();
            return;
        }
        int pipe = await ResidentStagePipelineAsync();
        var held = new List<IDisposable>(2 * stages.Length);
        try
        {
            var pipelines = new int[stages.Length];
            var groups = new int[stages.Length];
            var uniforms = new int[stages.Length];
            var workgroups = new int[stages.Length];
            for (int i = 0; i < stages.Length; i++)
            {
                var (bind, u) = BindResidentStage(pipe, stages[i], stages.AsSpan(0, i));
                held.Add(bind); held.Add(u);
                pipelines[i] = pipe; groups[i] = bind.BindGroupId; uniforms[i] = u.BufferId;
                workgroups[i] = ResidentProgramKernelTemplate.GroupsFor(stages[i]);
            }
            await WebGpuNativeBindings.DispatchComputeChainWithUniformsAsync(pipelines, groups, uniforms, workgroups);
            await WebGpuNativeBindings.SubmitAndWaitAsync();
        }
        finally
        {
            foreach (var resource in held) resource.Dispose();
        }
    }

    private async Task RunStagesSeparatelyAsync(ResidentStage[] stages)
    {
        int pipe = await ResidentStagePipelineAsync();
        foreach (var stage in stages)
        {
            // Each group is created just before its own dispatch, after the stages before it ran: nothing is stale.
            var (bind, u) = BindResidentStage(pipe, stage, ReadOnlySpan<ResidentStage>.Empty);
            using (bind)
            using (u)
            {
                await WebGpuNativeBindings.DispatchComputeWithUniformsAsync(pipe, bind.BindGroupId, u.BufferId,
                    ResidentProgramKernelTemplate.GroupsFor(stage), 1, 1);
                await WebGpuNativeBindings.SubmitAndWaitAsync();
            }
        }
    }

    private Task<int> ResidentStagePipelineAsync()
        => GetOrCreatePipelineAsync(ResidentProgramModuleKey, ResidentStageShader, "main");

    private void RequireResidentProgram()
    {
        if (!SupportsResidentProgram)
            throw new GpuCapabilityUnavailableException("The WebGPU resident program shader is unavailable.");
    }

    // Stands in for an unread second operand (the softmax's); an unread bias binds SharedDummyBuffer. Distinct from
    // every operand: WebGpuBindGroup replaces a buffer bound twice with a snapshot taken when the group is created,
    // which in a chain is before the earlier stages have written it.
    private WebGpuBuffer? _residentUnusedOperand;

    private WebGpuBuffer ResidentUnusedOperand
    {
        get
        {
            lock (_residentProgramLock) return _residentUnusedOperand ??= new WebGpuBuffer(1);
        }
    }

    private (WebGpuBindGroup Bind, WebGpuBuffer Uniforms) BindResidentStage(int pipe, in ResidentStage s,
        ReadOnlySpan<ResidentStage> runBefore)
    {
        // A dispatch may not bind one buffer as both a read and a writable storage binding, so an output that is also
        // an input is refused. A read operand bound twice is replaced by a snapshot when the group is created (above);
        // in a one-pass chain every group exists before the first dispatch, so that is refused only when a stage in
        // runBefore writes the buffer (the snapshot would hold its value from before the chain).
        var a = AsWgpu(s.A);
        var b = s.B is null ? ResidentUnusedOperand : AsWgpu(s.B);
        var c = s.C is null ? AsWgpu(SharedDummyBuffer) : AsWgpu(s.C);
        var o = AsWgpu(s.Output);
        if (SameBinding(o, a) || SameBinding(o, b) || SameBinding(o, c))
            throw new ArgumentException($"{s.Op}: on WebGPU a stage's output must be a different buffer from its inputs.");
        foreach (var (x, y) in new[] { (a, b), (a, c), (b, c) })
        {
            if (!SameBinding(x, y)) continue;
            foreach (var earlier in runBefore)
                if (SameBinding(AsWgpu(earlier.Output), x))
                    throw new ArgumentException($"{s.Op}: on WebGPU a stage may not repeat an operand that an earlier stage of the chain writes.");
        }
        var uniforms = new WebGpuBuffer(UniformInts((int)s.Op, s.Rows, s.Cols, 0), WebGpuBufferUsage.Uniform | WebGpuBufferUsage.CopyDst);
        try
        {
            var bind = new WebGpuBindGroup(pipe, a, b, c, o);
            return (bind, uniforms);
        }
        catch
        {
            uniforms.Dispose();
            throw;
        }
    }

    private static bool SameBinding(WebGpuBuffer x, WebGpuBuffer y)
        => x.BufferId == y.BufferId && x.OffsetBytes == y.OffsetBytes && x.SizeBytes == y.SizeBytes;
}
#endif
