using System;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class BufferAllocationDiagnosticsTests
{
    [Fact]
    public void DelegatingAllocationCountsOneSuccessfulStorageRequest()
    {
        var recorder = new BufferAllocationDiagnostics.RequestRecorder();
        var storage = new object();
        var result = recorder.Observe("CUDA", 128, () => recorder.Observe("CUDA", 256, () => storage));
        Assert.Same(storage, result);
        Assert.Equal(1, recorder.Snapshot()["CUDA"].Count);
        Assert.Equal(128, recorder.Snapshot()["CUDA"].Bytes);
        Assert.False(BufferAllocationDiagnostics.RequestRecorder.IsInside);
    }

    [Fact]
    public void FailedAllocationDoesNotCountAndRestoresRequestScope()
    {
        var recorder = new BufferAllocationDiagnostics.RequestRecorder();
        Assert.Throws<OutOfMemoryException>(() => recorder.Observe<object>("CUDA", 128,
            () => throw new OutOfMemoryException()));
        Assert.Empty(recorder.Snapshot());
        Assert.False(BufferAllocationDiagnostics.RequestRecorder.IsInside);
        recorder.Observe("CUDA", 64, () => new object());
        Assert.Equal(64, recorder.Snapshot()["CUDA"].Bytes);
    }

    [Fact]
    public void ConcurrentIndependentRequestsHaveExactCountsAndBytes()
    {
        var recorder = new BufferAllocationDiagnostics.RequestRecorder();
        Parallel.For(0, 1000, _ => recorder.Observe("CUDA", 64, () => new object()));
        Assert.Equal(1000, recorder.Snapshot()["CUDA"].Count);
        Assert.Equal(64000, recorder.Snapshot()["CUDA"].Bytes);
    }

    [Fact]
    public void StateCallbackPreservesOuterRequestAndBackendTotals()
    {
        var recorder = new BufferAllocationDiagnostics.RequestRecorder();
        var storage = new object();
        var result = recorder.Observe("CUDA", 128, (recorder, storage), static state =>
            state.recorder.Observe("CUDA", 256, state.storage, static value => value));
        recorder.Observe("OpenCL", 64, storage, static value => value);
        Assert.Same(storage, result);
        Assert.Equal(128, recorder.Snapshot()["CUDA"].Bytes);
        Assert.Equal(1, recorder.Snapshot()["CUDA"].Count);
        Assert.Equal(64, recorder.Snapshot()["OpenCL"].Bytes);
        Assert.Equal(1, recorder.Snapshot()["OpenCL"].Count);
        Assert.False(BufferAllocationDiagnostics.RequestRecorder.IsInside);
    }

    [Fact]
    public void NonpositiveRequestsAndFailedStateCallbackDoNotInventStorage()
    {
        var recorder = new BufferAllocationDiagnostics.RequestRecorder();
        recorder.Observe("CUDA", 0, 0, static value => value);
        recorder.Observe("CUDA", -1, 0, static value => value);
        Assert.Throws<OutOfMemoryException>(() => recorder.Observe<int, object>("CUDA", 128, 0,
            static _ => throw new OutOfMemoryException()));
        Assert.Empty(recorder.Snapshot());
        Assert.False(BufferAllocationDiagnostics.RequestRecorder.IsInside);
    }
}
