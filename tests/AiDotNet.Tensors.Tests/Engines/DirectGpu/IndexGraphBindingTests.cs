using System;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class IndexGraphBindingTests
{
    private static IGpuBuffer Copy(DirectGpuTensorEngine engine, IDirectGpuBackend backend, Tensor<int> ids, int bound = 0)
        => (IGpuBuffer)typeof(DirectGpuTensorEngine).GetMethod("GetHostIndexDeviceCopy", BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(engine, new object[] { backend, ids, bound, false })!;
    private static int First(IGpuBuffer buffer)
        => BitConverter.ToInt32(BitConverter.GetBytes(((MockGpuBuffer)buffer).Data[0]), 0);

    // A destination table with a repeat is refused at first use (the pre-capture and capture runs) exactly as every later
    // replay refuses it, so no graph is built that only fails when replayed.
    [Fact]
    public void DuplicateDestinations_AreRejectedOnFirstUse_NotOnlyOnReplay()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var unique = (IGpuBuffer)typeof(DirectGpuTensorEngine).GetMethod("GetHostIndexDeviceCopy", BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(engine, new object[] { backend, new Tensor<int>(new[] { 2, 0, 1 }, new[] { 3 }), 3, true })!;
        Assert.NotNull(unique);
        var ex = Assert.Throws<TargetInvocationException>(() => typeof(DirectGpuTensorEngine)
            .GetMethod("GetHostIndexDeviceCopy", BindingFlags.NonPublic | BindingFlags.Instance)!
            .Invoke(engine, new object[] { backend, new Tensor<int>(new[] { 2, 0, 2 }, new[] { 3 }), 3, true }));
        Assert.IsType<InvalidOperationException>(ex.InnerException);
    }

    [Theory]
    [InlineData(new int[0], true)]
    [InlineData(new[] { 5 }, true)]
    [InlineData(new[] { 3, 1, 2, 0 }, true)]
    [InlineData(new[] { 3, 1, 3, 0 }, false)]
    [InlineData(new[] { -4, 7, -4 }, false)]
    [InlineData(new[] { int.MinValue, 0, int.MaxValue }, true)]
    [InlineData(new[] { int.MinValue, 0, int.MaxValue, 0 }, false)]
    [InlineData(new[] { 0, 1 << 30, 5 << 20 }, true)]
    public void DistinctDestinationCheck_MatchesASet(int[] ids, bool distinct)
    {
        Assert.Equal(distinct, DirectGpuTensorEngine.AllDistinct(ids));
        Assert.Equal(new System.Collections.Generic.HashSet<int>(ids).Count == ids.Length, DirectGpuTensorEngine.AllDistinct(ids));
    }

    [Fact]
    public void FailedInitialUpload_IsNotPublished_AndRetryUploadsVersionZero()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        var ids = new Tensor<int>(new[] { 7 }, new[] { 1 });
        state.ThrowOnCall["UploadIntBufferInPlace"] = new InvalidOperationException("upload failed");
        var ex = Assert.Throws<TargetInvocationException>(() => Copy(engine, backend, ids));
        Assert.IsType<InvalidOperationException>(ex.InnerException);
        Assert.Single(state.AllocatedBuffers);
        Assert.Equal(1, state.AllocatedBuffers[0].DisposeCount);
        state.ThrowOnCall.Clear();
        Assert.Equal(7, First(Copy(engine, backend, ids)));
        Assert.Equal(2, state.AllocatedBuffers.Count);
    }

    [Fact]
    public void PublicReplay_RefreshesOriginalNonContiguousView()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.CreateGraphCapture(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var scope = engine.EnterResidentCaptureScope();
        var storage = new Tensor<int>(new[] { 1, 2, 3, 4 }, new[] { 2, 2 });
        var ids = storage.Transpose(new[] { 1, 0 });
        Assert.False(ids.IsContiguous);
        var buffer = Copy(engine, backend, ids);
        IntPtr graph = engine.CaptureGpuGraph(() => Copy(engine, backend, ids));
        storage.SetFlat(0, 6);
        engine.LaunchGpuGraph(graph);
        var uploaded = new int[4];
        Buffer.BlockCopy(((MockGpuBuffer)buffer).Data, 0, uploaded, 0, uploaded.Length * sizeof(int));
        Assert.Equal(new[] { 6, 3, 2, 4 }, uploaded);
        Assert.Equal(1, state.GraphLaunches);
        engine.DestroyGpuGraph(graph);
    }

    [Fact]
    public void ReplayReadsReplacementStorageWithTheSameMutationVersion()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.CreateGraphCapture(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var scope = engine.EnterResidentCaptureScope();
        using var ids = new Tensor<int>(new[] { 1 }, new[] { 1 });
        using var replacement = new Tensor<int>(new[] { 3 }, new[] { 1 });
        ids.SetFlat(0, 2);
        int version = ids.GpuCacheVersion;
        var buffer = Copy(engine, backend, ids);
        IntPtr graph = engine.CaptureGpuGraph(() => Copy(engine, backend, ids));
        try
        {
            new CpuEngine().TensorSetInPlace(ids, replacement);
            Assert.Equal(version, ids.GpuCacheVersion);
            engine.LaunchGpuGraph(graph);
            Assert.Equal(3, First(buffer));
        }
        finally { engine.DestroyGpuGraph(graph); }
    }

    [Fact]
    public void EmbeddingLookup_ValidatesChangedHostIdsBeforeReplay()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.CreateGraphCapture(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var scope = engine.EnterResidentCaptureScope();
        var table = new Tensor<float>(new[] { 2, 1 });
        var ids = new Tensor<int>(new[] { 0 }, new[] { 1 });
        void Lookup() => ((IEngine)engine).TensorEmbeddingLookup(table, ids);
        Lookup();
        IntPtr graph = engine.CaptureGpuGraph(Lookup);
        try
        {
            ids.SetFlat(0, 1);
            engine.LaunchGpuGraph(graph);
            ids.SetFlat(0, 2);
            Assert.Throws<ArgumentOutOfRangeException>(() => engine.LaunchGpuGraph(graph));
            Assert.Equal(1, state.GraphLaunches);
        }
        finally { engine.DestroyGpuGraph(graph); }
    }

    [Fact]
    public void VocabularyBounds_BelongToEachGraph_AndRetiredInputsAreNotRefreshed()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.CreateGraphCapture(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var scope = engine.EnterResidentCaptureScope();
        var ids = new Tensor<int>(new[] { 1 }, new[] { 1 });
        var buffer = Copy(engine, backend, ids, 2);
        IntPtr first = engine.CaptureGpuGraph(() => Copy(engine, backend, ids, 2));
        engine.DestroyGpuGraph(first);
        ids.SetFlat(0, 3);
        Copy(engine, backend, ids, 4);
        IntPtr second = engine.CaptureGpuGraph(() => Copy(engine, backend, ids, 4));
        engine.LaunchGpuGraph(second);
        Assert.Equal(3, First(buffer));
        var other = new Tensor<int>(new[] { 0 }, new[] { 1 });
        Copy(engine, backend, other, 1);
        IntPtr third = engine.CaptureGpuGraph(() => Copy(engine, backend, other, 1));
        engine.DestroyGpuGraph(second);
        ids.SetFlat(0, 99);
        state.ThrowOnCall["UploadIntBufferInPlace"] = new InvalidOperationException("retired source was uploaded");
        engine.LaunchGpuGraph(third);
        Assert.Equal(2, state.GraphLaunches);
        engine.DestroyGpuGraph(third);
    }

    [Fact]
    public void ReboundCurrentResidentBuffer_IsRejectedBeforeLaunch()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.CreateGraphCapture(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var scope = engine.EnterResidentCaptureScope();
        using var original = backend.AllocateIntBuffer(new[] { 1 });
        var ids = Tensor<int>.FromGpuBuffer(backend, original, new[] { 1 },
            AiDotNet.Tensors.Engines.Gpu.GpuTensorRole.Intermediate, ownsBuffer: false, bufferContainsRawInt32: true);
        IntPtr graph = engine.CaptureGpuGraph(() => engine.PrepareResidentIndexForCompiledStep(backend, ids));
        using var replacement = backend.AllocateIntBuffer(new[] { 2 });
        engine.BindResidentBuffer(ids, replacement, backend);
        Assert.Throws<InvalidOperationException>(() => engine.LaunchGpuGraph(graph));
        Assert.Equal(0, state.GraphLaunches);
        engine.DestroyGpuGraph(graph);
    }

    [Fact]
    public void LabelWordInitializationFailure_ReleasesAllocation_AndAccessorsTakeLeases()
    {
        var state = new MockBackendState();
        var backend = MockDirectGpuBackend.Create(state);
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        engine.StrictGpuLeases = true;
        var method = typeof(DirectGpuTensorEngine).GetMethod("FusedCeLabelErrorWord", BindingFlags.NonPublic | BindingFlags.Instance)!;
        state.ThrowOnCall["Fill"] = new InvalidOperationException("fill failed");
        Assert.Throws<TargetInvocationException>(() => method.Invoke(engine, new object[] { backend }));
        Assert.Single(state.AllocatedBuffers);
        Assert.Equal(1, state.AllocatedBuffers[0].DisposeCount);
        state.ThrowOnCall.Clear();
        var word = (MockGpuBuffer)method.Invoke(engine, new object[] { backend })!;
        word.Data[0] = 1;
        Assert.True(engine.TakeFusedCeLabelError());
        engine.ClearFusedCeLabelError();
        IDeviceReadback? readback = null;
        try
        {
            engine.QueueFusedCeLabelErrorCheck(ref readback);
            Assert.False(engine.ReadFusedCeLabelErrorCheck(readback));
        }
        finally { readback?.Dispose(); }
    }
}
