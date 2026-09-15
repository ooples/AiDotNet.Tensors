using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("EngineCurrentGlobalState")]
public sealed class PersistentCacheOwnershipTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Invalidation_RetiresThePriorOwnerAndRecoversFromFailedUpload(bool failUpload)
    {
        var backend = new TrackingBackend();
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        float[] values = { 1, 2, 3 };
        using var tensor = new Tensor<float>(values, new[] { 3 });
        engine.RegisterResidentParamBuffer(tensor);
        MockGpuBuffer original = Assert.IsType<MockGpuBuffer>(tensor.TryGetGpuBuffer());
        values[0] = 10;
        backend.FailAllocation = failUpload;

        engine.InvalidatePersistentTensor(tensor);

        Assert.Equal(0, original.DisposeCount);
        Assert.Equal(failUpload ? 0 : 1, engine.CachedTensorCount);
        backend.FailAllocation = false;
        engine.RegisterResidentParamBuffer(tensor);
        MockGpuBuffer current = Assert.IsType<MockGpuBuffer>(tensor.TryGetGpuBuffer());
        Assert.NotSame(original, current);
        Assert.Equal(new float[] { 10, 2, 3 }, current.Data);
        Assert.Equal(1, original.DisposeCount);
        Assert.Equal(0, current.DisposeCount);

        engine.Dispose();
        Assert.All(backend.Allocations, buffer => Assert.Equal(1, buffer.DisposeCount));
    }

    [Fact]
    public async Task ConcurrentInvalidations_RetireEveryReplacedAllocationExactlyOnce()
    {
        var backend = new TrackingBackend();
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        using var tensor = new Tensor<float>(new float[] { 1, 2, 3 }, new[] { 3 });
        engine.RegisterResidentParamBuffer(tensor);

        Task completion = Task.WhenAll(Enumerable.Range(0, 8)
            .Select(_ => Task.Run(() => engine.InvalidatePersistentTensor(tensor))));
        Assert.Same(completion, await Task.WhenAny(completion, Task.Delay(TimeSpan.FromSeconds(30))));
        await completion;
        Assert.Equal(9, backend.Allocations.Count);
        Assert.Equal(1, engine.CachedTensorCount);
        engine.RegisterResidentParamBuffer(tensor);
        MockGpuBuffer current = Assert.IsType<MockGpuBuffer>(tensor.TryGetGpuBuffer());
        Assert.Equal(new float[] { 1, 2, 3 }, current.Data);
        Assert.Single(backend.Allocations, buffer => buffer.DisposeCount == 0);
        Assert.Equal(0, current.DisposeCount);

        engine.Dispose();
        Assert.All(backend.Allocations, buffer => Assert.Equal(1, buffer.DisposeCount));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Invalidation_RemovesStaleCacheWhenUploadSynchronizationAndCleanupFail(bool failDisposal)
    {
        var backend = new TrackingBackend();
        using var direct = new DirectGpuEngine(backend);
        using var engine = new DirectGpuTensorEngine(direct);
        float[] values = { 1, 2, 3 };
        using var tensor = new Tensor<float>(values, new[] { 3 });
        engine.RegisterResidentParamBuffer(tensor);
        MockGpuBuffer original = Assert.IsType<MockGpuBuffer>(tensor.TryGetGpuBuffer());
        values[0] = 10;
        backend.FailSynchronization = true;
        backend.FailBufferDisposal = failDisposal;

        if (failDisposal)
            Assert.Throws<IOException>(() => engine.InvalidatePersistentTensor(tensor));
        else
            engine.InvalidatePersistentTensor(tensor);

        Assert.Equal(0, engine.CachedTensorCount);
        Assert.Equal(0, original.DisposeCount);
        backend.FailSynchronization = false;
        backend.FailBufferDisposal = false;
        engine.RegisterResidentParamBuffer(tensor);
        MockGpuBuffer current = Assert.IsType<MockGpuBuffer>(tensor.TryGetGpuBuffer());
        Assert.NotSame(original, current);
        Assert.Equal(new float[] { 10, 2, 3 }, current.Data);
        engine.Dispose();
        Assert.All(backend.Allocations, buffer => Assert.Equal(1, buffer.DisposeCount));
    }

    private sealed class ThrowingDisposalBuffer : IGpuBuffer
    {
        private readonly MockGpuBuffer _inner;
        internal ThrowingDisposalBuffer(MockGpuBuffer inner) => _inner = inner;
        public int Size => _inner.Size;
        public long SizeInBytes => _inner.SizeInBytes;
        public IntPtr Handle => _inner.Handle;
        public void Dispose()
        {
            _inner.Dispose();
            throw new IOException("Injected device cleanup failure.");
        }
    }

    private sealed class TrackingBackend : DelegatingGpuBackend
    {
        internal ConcurrentBag<MockGpuBuffer> Allocations { get; } = new();
        internal bool FailAllocation { get; set; }
        internal bool FailSynchronization { get; set; }
        internal bool FailBufferDisposal { get; set; }

        internal TrackingBackend() : base(MockDirectGpuBackend.Create(new MockBackendState())) { }

        public override IGpuBuffer AllocateBuffer(float[] values)
        {
            if (FailAllocation) throw new OutOfMemoryException("Injected device allocation failure.");
            var buffer = new MockGpuBuffer((float[])values.Clone());
            Allocations.Add(buffer);
            return FailBufferDisposal ? new ThrowingDisposalBuffer(buffer) : buffer;
        }

        public override void Synchronize()
        {
            if (FailSynchronization) throw new InvalidOperationException("Injected device synchronization failure.");
        }
    }
}
