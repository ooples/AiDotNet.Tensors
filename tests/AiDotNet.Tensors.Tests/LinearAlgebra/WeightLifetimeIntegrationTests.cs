// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.IO;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// Issue #276 follow-up gaps: end-to-end Tensor.Lifetime + WeightRegistry
/// dispatch, integrating streaming pool / offload allocator into a real
/// model-author workflow.
///
/// <para>WeightRegistry is process-wide static state — Configure / Reset /
/// Register all share global slots. xUnit parallelizes test classes by
/// default; <c>[Collection("WeightRegistry")]</c> serializes this class
/// against any other class that touches the registry to prevent
/// interleaved-test corruption.</para>
/// </summary>
[Collection("WeightRegistry")]
public class WeightLifetimeIntegrationTests
{
    private sealed class TrackingOffloadAllocator : IGpuOffloadAllocator
    {
        private readonly System.Collections.Generic.HashSet<IntPtr> _live = new();
        private int _allocationCount;
        public int FailOnAllocation { get; set; } = int.MaxValue;
        public int FailAfterFreeAttempt { get; set; } = int.MaxValue;
        public int FreeAttempts { get; private set; }
        public int LiveCount => _live.Count;

        public bool IsAvailable => true;
        public int FreeCount { get; private set; }
        public int DisposeCount { get; private set; }

        public GpuOffloadHandle Allocate(long bytes, OffloadScheme scheme)
        {
            if (++_allocationCount == FailOnAllocation) throw new OutOfMemoryException("Injected allocation failure.");
            IntPtr pointer = Marshal.AllocHGlobal(checked((int)bytes));
            _live.Add(pointer);
            return new GpuOffloadHandle(pointer, pointer, bytes, scheme);
        }

        public void Free(GpuOffloadHandle handle)
        {
            FreeAttempts++;
            if (!_live.Remove(handle.HostPointer))
                throw new InvalidOperationException("Allocation was freed through the wrong allocator.");
            Marshal.FreeHGlobal(handle.HostPointer);
            FreeCount++;
            if (FreeAttempts == FailAfterFreeAttempt)
                throw new IOException("Injected failure after the allocator released its handle.");
        }

        public void Dispose()
        {
            foreach (IntPtr pointer in _live)
                Marshal.FreeHGlobal(pointer);
            _live.Clear();
            DisposeCount++;
        }
    }

    private sealed class ThrowingTraceListener : System.Diagnostics.TraceListener
    {
        public override void Write(string? message) => throw new IOException("Injected diagnostic failure.");
        public override void WriteLine(string? message) => throw new IOException("Injected diagnostic failure.");
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Reset_AttemptsEveryFreeAndClearsHandlesWhenOneFreeThrows(bool failLogging)
    {
        WeightRegistry.Reset();
        var allocator = new TrackingOffloadAllocator { FailAfterFreeAttempt = 1 };
        using var listener = new ThrowingTraceListener();
        try
        {
            var weights = new[] { new Tensor<float>(new[] { 2 }), new Tensor<float>(new[] { 3 }) };
            WeightRegistry.ConfigureAndRegisterBatch(new GpuOffloadOptions(), weights, allocator);
            if (failLogging) System.Diagnostics.Trace.Listeners.Add(listener);

            WeightRegistry.Reset();

            Assert.Equal(2, allocator.FreeAttempts);
            Assert.Equal(2, allocator.FreeCount);
            Assert.Equal(0, allocator.LiveCount);
            Assert.Equal(1, allocator.DisposeCount);
            WeightRegistry.Reset();
            Assert.Equal(2, allocator.FreeAttempts);
            GC.KeepAlive(weights);
        }
        finally
        {
            System.Diagnostics.Trace.Listeners.Remove(listener);
            WeightRegistry.Reset();
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Configure_PrunesEveryDeadOwnerEvenWhenOneFreeThrows(bool failLogging)
    {
        WeightRegistry.Reset();
        var allocator = new TrackingOffloadAllocator { FailAfterFreeAttempt = 1 };
        var replacement = new TrackingOffloadAllocator();
        using var listener = new ThrowingTraceListener();
        try
        {
            WeightRegistry.Configure(new GpuOffloadOptions(), allocator);
            WeakReference[] abandoned = RegisterAndAbandonOffloadWeights();
            for (int attempt = 0; attempt < 5 && Array.Exists(abandoned, weak => weak.IsAlive); attempt++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
            }
            Assert.All(abandoned, weak => Assert.False(weak.IsAlive));
            if (failLogging) System.Diagnostics.Trace.Listeners.Add(listener);

            WeightRegistry.Configure(new GpuOffloadOptions(), replacement);

            Assert.Equal(2, allocator.FreeAttempts);
            Assert.Equal(2, allocator.FreeCount);
            Assert.Equal(0, allocator.LiveCount);
            Assert.Equal(1, allocator.DisposeCount);
            Assert.Same(replacement, WeightRegistry.OffloadAllocator);
            WeightRegistry.Configure(new GpuOffloadOptions(), replacement);
            Assert.Equal(2, allocator.FreeAttempts);
        }
        finally
        {
            System.Diagnostics.Trace.Listeners.Remove(listener);
            WeightRegistry.Reset();
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static WeakReference[] RegisterAndAbandonOffloadWeights()
    {
        var first = new Tensor<float>(new[] { 2 });
        var second = new Tensor<float>(new[] { 3 });
        WeightRegistry.RegisterBatch(new[] { first, second }, WeightLifetime.GpuOffload);
        return new[] { new WeakReference(first), new WeakReference(second) };
    }

    [Fact]
    public void Tensor_DefaultLifetime_IsDefault()
    {
        var t = new Tensor<float>(new[] { 4 });
        Assert.Equal(WeightLifetime.Default, t.Lifetime);
        Assert.Equal(-1L, t.StreamingPoolHandle);
        Assert.Equal(IntPtr.Zero, t.OffloadDevicePointer);
    }

    [Fact]
    public void ConfigureBatch_FailedSecondAllocationPreservesPriorConfigurationAndWeights()
    {
        WeightRegistry.Reset();
        var priorAllocator = new TrackingOffloadAllocator();
        var candidateAllocator = new TrackingOffloadAllocator { FailOnAllocation = 2 };
        try
        {
            var priorOptions = new GpuOffloadOptions();
            WeightRegistry.Configure(priorOptions, priorAllocator);
            var first = new Tensor<float>(new float[] { 1, 2 }, new[] { 2 });
            var second = new Tensor<float>(new float[] { 3, 4 }, new[] { 2 });
            Assert.Throws<OutOfMemoryException>(() => WeightRegistry.ConfigureAndRegisterBatch(
                new GpuOffloadOptions(), new[] { first, second }, candidateAllocator));
            Assert.Same(priorOptions, WeightRegistry.CurrentOptions);
            Assert.Same(priorAllocator, WeightRegistry.OffloadAllocator);
            Assert.Equal(0, priorAllocator.DisposeCount);
            Assert.Equal(0, candidateAllocator.LiveCount);
            Assert.Equal(1, candidateAllocator.FreeCount);
            foreach (var weight in new[] { first, second })
            {
                Assert.Equal(WeightLifetime.Default, weight.Lifetime);
                Assert.Equal(-1, weight.OffloadRegistryHandle);
                Assert.Equal(IntPtr.Zero, weight.OffloadHostPointer);
            }
            Assert.Equal(new float[] { 1, 2 }, first.ToArray());
            Assert.Equal(new float[] { 3, 4 }, second.ToArray());
            candidateAllocator.FailOnAllocation = int.MaxValue;
            WeightRegistry.ConfigureAndRegisterBatch(new GpuOffloadOptions(), new[] { first, first, second }, candidateAllocator);
            Assert.Equal(2, candidateAllocator.LiveCount);
            WeightRegistry.RegisterBatch(new[] { first, second }, WeightLifetime.GpuOffload);
            Assert.Equal(2, candidateAllocator.LiveCount);
        }
        finally { WeightRegistry.Reset(); candidateAllocator.Dispose(); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ConfigureBatch_InvalidBackingPathLeavesPreviousGlobalsAndDataUnchanged(bool lazy)
    {
        WeightRegistry.Reset();
        string invalidDirectory = Path.GetTempFileName();
        try
        {
            var previous = new GpuOffloadOptions();
            WeightRegistry.Configure(previous);
            var weight = new Tensor<float>(new float[] { 1, 2 }, new[] { 2 });
            Assert.ThrowsAny<IOException>(() => WeightRegistry.ConfigureAndRegisterBatch(
                new GpuOffloadOptions { StreamingBackingStorePath = invalidDirectory }, lazy ? Array.Empty<Tensor<float>>() : new[] { weight }));
            Assert.Same(previous, WeightRegistry.CurrentOptions);
            Assert.Equal(WeightLifetime.Default, weight.Lifetime);
            Assert.Equal(-1, weight.StreamingPoolHandle);
            Assert.Equal(new float[] { 1, 2 }, weight.ToArray());
        }
        finally { WeightRegistry.Reset(); File.Delete(invalidDirectory); }
    }

    [Fact]
    public void RegisterBatch_FailedDeltaPreservesExistingOffloadAndUnregisteredWeights()
    {
        WeightRegistry.Reset();
        var allocator = new TrackingOffloadAllocator { FailOnAllocation = 3 };
        try
        {
            var existing = new Tensor<float>(new float[] { 7 }, new[] { 1 });
            WeightRegistry.ConfigureAndRegisterBatch(new GpuOffloadOptions(), new[] { existing }, allocator);
            long originalHandle = existing.OffloadRegistryHandle;
            var first = new Tensor<float>(new float[] { 1 }, new[] { 1 });
            var second = new Tensor<float>(new float[] { 2 }, new[] { 1 });
            Assert.Throws<OutOfMemoryException>(() => WeightRegistry.RegisterBatch(
                new[] { existing, first, second }, WeightLifetime.GpuOffload));
            Assert.Equal(1, allocator.LiveCount);
            Assert.Equal(originalHandle, existing.OffloadRegistryHandle);
            Assert.Equal(7, Marshal.PtrToStructure<float>(existing.OffloadHostPointer));
            Assert.Equal(WeightLifetime.Default, first.Lifetime);
            Assert.Equal(WeightLifetime.Default, second.Lifetime);
            Assert.Equal(-1, first.OffloadRegistryHandle);
            Assert.Equal(-1, second.OffloadRegistryHandle);
        }
        finally { WeightRegistry.Reset(); }
    }

    [Fact]
    public void RegisterWeight_Streaming_RoutesToStreamingPool()
    {
        var dir = Path.Combine(Path.GetTempPath(), $"aidotnet-life-{Guid.NewGuid():N}");
        try
        {
            WeightRegistry.Configure(new GpuOffloadOptions
            {
                StreamingBackingStorePath = dir,
                StreamingPoolMaxResidentBytes = 1024L * 1024,
            });

            var t = new Tensor<float>(new[] { 64 });
            for (int i = 0; i < 64; i++) t[i] = i;
            t.Lifetime = WeightLifetime.Streaming;
            WeightRegistry.RegisterWeight(t);

            Assert.True(t.StreamingPoolHandle >= 0);
            Assert.True(WeightRegistry.StreamingPool.ResidentBytes >= 64 * sizeof(float));

            WeightRegistry.UnregisterWeight(t);
            Assert.Equal(-1L, t.StreamingPoolHandle);
        }
        finally
        {
            WeightRegistry.Reset();
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void RegisterWeight_GpuOffload_NoBackend_FallsBackToDefault()
    {
        // No allocator configured → registration silently demotes to Default
        // so consumers can opt in without crashing on hosts that lack the
        // matching GPU runtime.
        WeightRegistry.Reset();
        var t = new Tensor<float>(new[] { 16 });
        t.Lifetime = WeightLifetime.GpuOffload;
        WeightRegistry.RegisterWeight(t);
        Assert.Equal(WeightLifetime.Default, t.Lifetime);
        Assert.Equal(IntPtr.Zero, t.OffloadDevicePointer);
    }

    [Fact]
    public void Configure_CannotReplaceAllocatorThatOwnsLiveOffloadHandles()
    {
        WeightRegistry.Reset();
        var allocatorA = new TrackingOffloadAllocator();
        var allocatorB = new TrackingOffloadAllocator();
        try
        {
            var options = new GpuOffloadOptions();
            WeightRegistry.Configure(options, allocatorA);
            var tensor = new Tensor<float>(new[] { 8 })
            {
                Lifetime = WeightLifetime.GpuOffload,
            };
            WeightRegistry.RegisterWeight(tensor);

            Assert.Throws<InvalidOperationException>(
                () => WeightRegistry.Configure(options, allocatorB));
            Assert.Equal(0, allocatorA.DisposeCount);
            Assert.Equal(0, allocatorB.DisposeCount);

            WeightRegistry.UnregisterWeight(tensor);
            Assert.Equal(1, allocatorA.FreeCount);

            WeightRegistry.Configure(options, allocatorB);
            Assert.Equal(1, allocatorA.DisposeCount);
            Assert.Same(allocatorB, WeightRegistry.OffloadAllocator);
        }
        finally
        {
            WeightRegistry.Reset();
        }
    }
}
