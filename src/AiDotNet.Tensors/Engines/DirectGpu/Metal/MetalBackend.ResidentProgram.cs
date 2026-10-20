using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// #1451: the resident program on Metal. Metal has no cooperative launch and no cross-threadgroup barrier. A row-local
// chain is ONE dispatch of resident_rows, each threadgroup taking a row through every stage with threadgroup barriers;
// any other chain is every stage dispatch in ONE compute encoder, a buffer memory barrier between them, committed as
// one command buffer. Separately, each stage is its own command buffer (the backend's usual dispatch). Both run
// ResidentProgramKernelTemplate's stage kernel, so the two give bit-identical results.
// NOTE: validated only on a Metal runner (metal.yml) - the dev box has no Apple GPU.
public sealed partial class MetalBackend
{
    private const string ResidentProgramLibraryName = "ResidentProgram";
    private IntPtr _residentProgramLibrary;
    private bool? _residentProgramReady;
    private readonly object _residentProgramLock = new object();

    /// <summary>Whether this device can run a resident program (its stage kernel compiled, with a 256-thread
    /// threadgroup).</summary>
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
                        // Strict math: the softmax's exp and division must round like the other backends' instances.
                        _residentProgramLibrary = _shaderLibrary.CompileLibrary(ResidentProgramLibraryName,
                            ResidentProgramKernelTemplate.GetModuleSource(ResidentKernelDialect.Metal),
                            requireStrictFloatingPoint: true);
                        var pipeline = GetPipeline(ResidentProgramLibraryName, _residentProgramLibrary,
                            ResidentProgramKernelTemplate.StageKernel);
                        _residentProgramReady = pipeline.MaxTotalThreadsPerThreadgroup >= ResidentProgramKernelTemplate.GroupThreads;
                    }
                    catch (Exception ex) when (ex is not OutOfMemoryException)
                    {
                        System.Diagnostics.Trace.TraceWarning($"Metal resident program kernel unavailable: {ex.Message}");
                        _residentProgramLibrary = IntPtr.Zero;
                        _residentProgramReady = false;
                    }
                }
                return _residentProgramReady.Value;
            }
        }
    }

    /// <summary>
    /// Runs <paramref name="stages"/> in order as one command buffer, which this call commits and waits for: one
    /// dispatch for a row-local chain, one per stage otherwise. Returns the number of dispatches encoded.
    /// </summary>
    internal int RunResidentProgram(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        var pipeline = ResidentStagePipeline();
        if (ResidentProgramKernelTemplate.TryPlanRowProgram(stages, out var plan) && plan is not null)
        {
            var rows = GetPipeline(ResidentProgramLibraryName, _residentProgramLibrary, ResidentProgramKernelTemplate.RowProgramKernel);
            using var rowEncoder = _commandQueue.CreateScopedComputeEncoder();
            rowEncoder.SetPipelineState(rows.Handle);
            for (int i = 0; i < ResidentProgramKernelTemplate.MaxRowProgramBuffers; i++)
                rowEncoder.SetBuffer(AsResidentBuffer(i < plan.Buffers.Length ? plan.Buffers[i] : plan.Buffers[0]), i);
            var p = plan.Parameters();
            rowEncoder.SetBytes(new RowProgramParams
            {
                Rows = p[0], Cols = p[1], Count = p[2],
                W0 = p[4], W1 = p[5], W2 = p[6], W3 = p[7], W4 = p[8], W5 = p[9], W6 = p[10], W7 = p[11],
            }, (ulong)ResidentProgramKernelTemplate.MaxRowProgramBuffers);
            rowEncoder.DispatchThreadgroups(new MTLSize((ulong)plan.Groups, 1, 1),
                new MTLSize((ulong)ResidentProgramKernelTemplate.GroupThreads, 1, 1));
            return 1;
        }
        using (var encoder = _commandQueue.CreateScopedComputeEncoder())
        {
            for (int i = 0; i < stages.Length; i++)
            {
                if (i > 0) encoder.MemoryBarrier(MTLBarrierScope.Buffers);
                EncodeResidentStage(in encoder, pipeline, stages[i]);
            }
        }
        return stages.Length;
    }

    /// <summary>The same stages, each its own command buffer (what a resident program is measured against).</summary>
    internal void RunStagesSeparately(ReadOnlySpan<ResidentStage> stages)
    {
        ResidentProgramKernelTemplate.RequireChain(stages);
        var pipeline = ResidentStagePipeline();
        foreach (var stage in stages)
        {
            using var encoder = _commandQueue.CreateScopedComputeEncoder();
            EncodeResidentStage(in encoder, pipeline, stage);
        }
    }

    private MetalPipelineState ResidentStagePipeline()
    {
        ThrowIfDisposed();
        if (!SupportsResidentProgram)
            throw new GpuCapabilityUnavailableException("The Metal resident program kernel is unavailable.");
        return GetPipeline(ResidentProgramLibraryName, _residentProgramLibrary, ResidentProgramKernelTemplate.StageKernel);
    }

    private static void EncodeResidentStage(in ScopedComputeEncoder encoder, MetalPipelineState pipeline, in ResidentStage s)
    {
        encoder.SetPipelineState(pipeline.Handle);
        // An unread operand still needs a bound buffer: the input stands in.
        encoder.SetBuffer(AsResidentBuffer(s.A), 0);
        encoder.SetBuffer(AsResidentBuffer(s.B ?? s.A), 1);
        encoder.SetBuffer(AsResidentBuffer(s.C ?? s.A), 2);
        encoder.SetBuffer(AsResidentBuffer(s.Output), 3);
        encoder.SetBytes((int)s.Op, 4);
        encoder.SetBytes(s.Rows, 5);
        encoder.SetBytes(s.Cols, 6);
        encoder.DispatchThreadgroups(new MTLSize((ulong)ResidentProgramKernelTemplate.GroupsFor(s), 1, 1),
            new MTLSize((ulong)ResidentProgramKernelTemplate.GroupThreads, 1, 1));
    }

    // The row program's parameter block: resident_rows' ResRowParams (rows, cols, count, unused, eight stage words).
    [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct RowProgramParams
    {
        public int Rows, Cols, Count, Unused, W0, W1, W2, W3, W4, W5, W6, W7;
    }

    private static MetalGpuBuffer AsResidentBuffer(IGpuBuffer buffer)
        => buffer as MetalGpuBuffer ?? throw new ArgumentException("Buffers must be MetalGpuBuffer.", nameof(buffer));
}
