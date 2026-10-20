#if NET7_0_OR_GREATER
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

// #1452: the resident program on WebGPU. WebGPU has no cooperative launch, no cross-workgroup barrier and no
// forward-progress guarantee, so the chain is every stage dispatch in ONE compute pass of one command encoder,
// submitted once (dispatchComputeChainWithUniforms): WebGPU orders a pass's dispatches with implicit storage barriers.
// Separately, each stage is its own pass and submission (the backend's usual dispatch). Both run
// ResidentProgramKernelTemplate's stage shader, so the two give bit-identical results. Like the backend's other
// synchronous capabilities, both wait for their submissions.
public sealed partial class WebGpuBackend
{
    private const string ResidentProgramModuleKey = "ResidentProgram:stage";
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
    /// Runs <paramref name="stages"/> in order as one compute pass and one submission, which this call waits for.
    /// Returns the number of stage dispatches encoded.
    /// </summary>
    internal int RunResidentProgram(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        RequireResidentProgram();
        var copy = stages.ToArray();
        RunResidentProgramAsync(copy).GetAwaiter().GetResult();
        return copy.Length;
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
                var (bind, u) = BindResidentStage(pipe, stages[i]);
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
            var (bind, u) = BindResidentStage(pipe, stage);
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

    private (WebGpuBindGroup Bind, WebGpuBuffer Uniforms) BindResidentStage(int pipe, in ResidentStage s)
    {
        // Every bind group of a chain is created before its first dispatch runs, so a buffer bound twice would be
        // read through a stale snapshot (above); and a dispatch may not bind one buffer as both a read and a writable
        // storage binding. Both are refused.
        var a = AsWgpu(s.A);
        var b = s.B is null ? ResidentUnusedOperand : AsWgpu(s.B);
        var c = s.C is null ? AsWgpu(SharedDummyBuffer) : AsWgpu(s.C);
        var o = AsWgpu(s.Output);
        if (SameBinding(o, a) || SameBinding(o, b) || SameBinding(o, c))
            throw new ArgumentException($"{s.Op}: on WebGPU a stage's output must be a different buffer from its inputs.");
        if (SameBinding(a, b) || SameBinding(a, c) || SameBinding(b, c))
            throw new ArgumentException($"{s.Op}: on WebGPU a stage's operands must be different buffers.");
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
