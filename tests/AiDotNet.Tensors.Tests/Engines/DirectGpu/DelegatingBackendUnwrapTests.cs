// Gated to non-Framework targets: the mock backend is a DispatchProxy (.NET Core 5+).
#if NETCOREAPP
using AiDotNet.Tensors.Engines.Gpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="DelegatingGpuBackend.InnerBackend"/> is how the engine's per-backend caches (the IRFFT tables) reach the
/// physical backend under a DeferredScope, whose backend is a <see cref="RecordingGpuBackend"/>. Keyed on the wrapper,
/// each scope added its own entries and engine disposal skipped the physical backend's queued release.
/// </summary>
public class DelegatingBackendUnwrapTests
{
    [Fact]
    public void InnerBackend_returns_the_wrapped_backend_through_nested_wrappers()
    {
        var physical = MockDirectGpuBackend.Create(new MockBackendState());
        var recording = new RecordingGpuBackend(physical);
        var outer = new DelegatingGpuBackend(recording);

        Assert.Same(physical, recording.InnerBackend);
        Assert.Same(recording, outer.InnerBackend);

        // The engine's unwrap loop: any depth of wrapping ends at the one physical backend.
        AiDotNet.Tensors.Engines.DirectGpu.IDirectGpuBackend backend = outer;
        while (backend is DelegatingGpuBackend wrapper) backend = wrapper.InnerBackend;
        Assert.Same(physical, backend);
    }

    private static MockGpuBuffer RecordAdd(RecordingGpuBackend recording, float left, float right)
    {
        var output = new MockGpuBuffer(new float[2]);
        recording.Add(new MockGpuBuffer(new[] { left, left }), new MockGpuBuffer(new[] { right, right }), output, 2);
        return output;
    }

    // What a recording-only recorder wrote holds nothing until its graph runs; the graph says so when it has.
    [Fact]
    public void A_recorded_output_is_unexecuted_until_its_graph_runs()
    {
        var physical = MockDirectGpuBackend.Create(new MockBackendState());
        var recording = new RecordingGpuBackend(physical) { ExecuteWhileRecording = false };
        using var builder = new AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder();
        recording.BeginRecording(builder);
        MockGpuBuffer output;
        try { output = RecordAdd(recording, 1f, 2f); }
        finally { recording.EndRecording(); }

        Assert.True(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(output));
        using var graph = builder.Build();
        Assert.True(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(output));
        graph.Execute(physical);
        Assert.False(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(output));
        Assert.Equal(new[] { 3f, 3f }, output.Data);
    }

    // Clearing the builder discards its nodes: their outputs never run, whatever is recorded and run afterwards.
    [Fact]
    public void An_output_recorded_before_a_clear_stays_unexecuted()
    {
        var physical = MockDirectGpuBackend.Create(new MockBackendState());
        var recording = new RecordingGpuBackend(physical) { ExecuteWhileRecording = false };
        using var builder = new AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder();
        recording.BeginRecording(builder);
        MockGpuBuffer discarded, kept;
        try
        {
            discarded = RecordAdd(recording, 1f, 2f);
            builder.Clear();
            kept = RecordAdd(recording, 5f, 6f);
        }
        finally { recording.EndRecording(); }

        using var graph = builder.Build();
        graph.Execute(physical);
        Assert.False(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(kept));
        Assert.True(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(discarded));
        Assert.Equal(new[] { 11f, 11f }, kept.Data);
        Assert.Equal(new[] { 0f, 0f }, discarded.Data);
    }

    // A recorder that runs each op as it records it has written its outputs already, before and after a clear.
    [Fact]
    public void Outputs_of_a_builder_that_runs_while_recording_are_never_unexecuted()
    {
        // The two flags DeferredScope sets for ExecuteEagerlyWhileRecording: the recorder runs each op on the wrapped
        // backend as it records it, and the builder is told its writes have run.
        var physical = MockDirectGpuBackend.Create(new MockBackendState());
        var recording = new RecordingGpuBackend(physical) { ExecuteWhileRecording = true };
        using var builder = new AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder();
        builder.MarkWritesRunWhileRecording();
        recording.BeginRecording(builder);
        try
        {
            var first = RecordAdd(recording, 1f, 2f);
            Assert.False(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(first));
            Assert.Equal(new[] { 3f, 3f }, first.Data);
            builder.Clear();
            var second = RecordAdd(recording, 5f, 6f);
            Assert.False(AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder.IsUnexecutedWrite(second));
            Assert.Equal(new[] { 11f, 11f }, second.Data);
        }
        finally { recording.EndRecording(); }
    }

    // Every wrapper implements the index conversions by forwarding, so a recording wrapper over a backend without them
    // used to record the conversion and fail only when the graph replayed, outside the engine's capability fallback.
    [Fact]
    public void Recording_an_index_conversion_the_physical_backend_lacks_fails_at_record_time()
    {
        var physical = MockDirectGpuBackend.Create(new MockBackendState());
        var recording = new RecordingGpuBackend(new DelegatingGpuBackend(physical)) { ExecuteWhileRecording = false };
        using var graph = new AiDotNet.Tensors.Engines.Gpu.Graph.ExecutionGraphBuilder();
        var raw = new MockGpuBuffer(new float[4]);
        var output = new MockGpuBuffer(new float[4]);
        recording.BeginRecording(graph);
        try
        {
            Assert.Throws<AiDotNet.Tensors.Engines.DirectGpu.GpuCapabilityUnavailableException>(
                () => recording.Int32ToFloat(raw, output, 4));
            Assert.Throws<AiDotNet.Tensors.Engines.DirectGpu.GpuCapabilityUnavailableException>(
                () => recording.ScatterPositions(output, 2, 2, 4));
            Assert.Equal(0, graph.NodeCount);
        }
        finally
        {
            recording.EndRecording();
        }
    }
}
#endif
