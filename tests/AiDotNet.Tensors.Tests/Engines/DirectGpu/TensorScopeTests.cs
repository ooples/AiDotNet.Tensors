// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A <see cref="TensorScope"/> frees the GPU results created inside it when it ends, except those kept, so an inference
/// call's dead intermediates return to the pool without a garbage collection (#1432).
/// </summary>
[Collection("VulkanGlobalState")]   // DeviceMemoryReclaim's counters are process-wide
public sealed class TensorScopeTests
{
    // A machine without a GPU skips, naming why; a lane that requires one (AIDOTNET_REQUIRE_GPU_TESTS=1) fails instead,
    // with the initialization error, so a broken driver cannot pass as a skip.
    private static DirectGpuTensorEngine? TryEngine(out string skipReason)
    {
        Exception? failure = null;
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) { skipReason = string.Empty; return gpu; }
            gpu.Dispose();
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            failure = ex;
        }
        skipReason = failure is null
            ? "No DirectGpu backend."
            : $"No DirectGpu backend: {failure.GetType().Name}: {failure.Message}";
        if (string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS"), "1", StringComparison.Ordinal))
            throw new InvalidOperationException("GPU tests were required (AIDOTNET_REQUIRE_GPU_TESTS=1) but " + skipReason, failure);
        return null;
    }

    private static Tensor<float> Host(int length, float start)
    {
        var host = new float[length];
        for (int i = 0; i < length; i++) host[i] = start + i;
        return new Tensor<float>(host, new[] { length });
    }

    private static Tensor<float> Resident(DirectGpuTensorEngine gpu, int length, float start)
    {
        var host = new float[length];
        for (int i = 0; i < length; i++) host[i] = start + i;
        // An op on a host tensor returns a resident result the engine owns.
        return gpu.TensorAddScalar(new Tensor<float>(host, new[] { length }), 0f);
    }

    [SkippableFact]
    public void EndingTheScope_FreesItsResults_ExceptTheKeptOne()
    {
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        Tensor<float> intermediate, kept;
        using (var scope = TensorScope.Begin())
        {
            intermediate = Resident(gpu, 64, 1f);
            kept = scope.Keep(gpu.TensorAddScalar(intermediate, 1f));
            Assert.True(scope.TrackedCount >= 1);
        }

        Assert.Equal(2f, kept.ToArray()[0]);
        Assert.Equal(65f, kept.ToArray()[63]);
        var ex = Assert.Throws<InvalidOperationException>(() => intermediate.ToArray());
        Assert.Contains("TensorScope", ex.Message);
        // A freed result is not an input either: the engine refuses it instead of reading a recycled buffer.
        Assert.Throws<InvalidOperationException>(() => gpu.TensorAddScalar(intermediate, 1f).ToArray());
    }

    [SkippableFact]
    public void ADestinationFromBeforeTheScope_KeepsWhatAnIntoOpWroteThere()
    {
        // The destination is the caller's: an *Into op inside the scope writes it on the device, and the scope's end must
        // not free that copy (reading it threw "released" before the scope tracked which storages it created).
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        var destination = new Tensor<float>(new[] { 3, 2 });
        using (TensorScope.Begin())
        {
            var source = gpu.TensorAddScalar(new Tensor<float>(new float[] { 0, 1, 2, 3, 4, 5 }, new[] { 2, 3 }), 0f);
            gpu.TensorPermuteInto(destination, source, new[] { 1, 0 });
        }

        Assert.Equal(new float[] { 0, 3, 1, 4, 2, 5 }, destination.ToArray());
    }

    [SkippableFact]
    public void WhatIsCachedWhileTheScopeIsSuspended_OutlivesTheScope()
    {
        // An engine cache built during a call (the device-folded kernel of a convolution with batch norm) is made with the
        // scope suspended; the scope's end must not free it, though the engine is enlisted with the scope by then.
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        Tensor<float> cached, intermediate;
        using (TensorScope.Begin())
        {
            // The scope's own activation entry: it enlists the engine, so the scope's end runs the activation release.
            intermediate = new Tensor<float>(new[] { 4, 4 });
            gpu.TensorPermuteInto(intermediate, gpu.TensorAddScalar(Host(16, 1f).Reshape(4, 4), 0f), new[] { 1, 0 });
            using (TensorScope.Suspend())
            {
                long before = gpu.ActivationCacheTimestampSnapshot();
                cached = new Tensor<float>(new[] { 4, 4 });
                gpu.TensorPermuteInto(cached, gpu.TensorAddScalar(Host(16, 100f).Reshape(4, 4), 0f), new[] { 1, 0 });
                // The case is about an activation-cache entry: an op whose result owns its buffer would pass vacuously.
                Assert.True(gpu.ActivationCacheTimestampSnapshot() > before, "the suspended op cached no activation.");
            }
        }

        Assert.Equal(100f, cached.ToArray()[0]);    // transposed 4 x 4 of 100..115
        Assert.Equal(104f, cached.ToArray()[1]);
        Assert.Equal(115f, cached.ToArray()[15]);
        // The release did run: the scope's own entry is freed.
        Assert.Throws<InvalidOperationException>(() => intermediate.ToArray());
    }
    [SkippableFact]
    public void InsideADeferredExecutionScope_TheReleaseWaitsForTheGraph()
    {
        // A recorded graph references the scope's buffers until it executes; HIP's buffers do not consult the deferral
        // gate themselves, so the scope queues its whole release behind it.
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        Tensor<float> intermediate;
        AiDotNet.Tensors.Engines.Gpu.GpuBufferReleaseDeferral.Begin();
        try
        {
            using (TensorScope.Begin())
                intermediate = Resident(gpu, 16, 3f);
            Assert.Equal(18f, intermediate.ToArray()[15]);   // still valid: the graph has not run yet
        }
        finally
        {
            AiDotNet.Tensors.Engines.Gpu.GpuBufferReleaseDeferral.EndAndRelease();
        }

        Assert.Throws<InvalidOperationException>(() => gpu.TensorAddScalar(intermediate, 1f).ToArray());
    }
    [SkippableFact]
    public void KeepingAView_KeepsTheResultItViews()
    {
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        Tensor<float> view;
        using (var scope = TensorScope.Begin())
        {
            var result = gpu.TensorAddScalar(Resident(gpu, 12, 0f), 1f);
            view = scope.Keep(gpu.Reshape(result, new[] { 3, 4 }));
        }

        Assert.Equal(new[] { 3, 4 }, view.Shape.ToArray());
        Assert.Equal(12f, view.ToArray()[11]);
    }

    [SkippableFact]
    public void KeepInANestedScope_HandsTheResultToTheEnclosingScope()
    {
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        Tensor<float> keptToOuter, keptToCaller, handedUp;
        using (var outer = TensorScope.Begin())
        {
            using (var inner = TensorScope.Begin())
            {
                handedUp = inner.Keep(Resident(gpu, 8, 10f));
                keptToCaller = inner.Keep(Resident(gpu, 8, 20f));
            }
            Assert.Equal(10f, handedUp.ToArray()[0]);   // alive while the enclosing scope is open
            keptToOuter = outer.Keep(keptToCaller);
        }

        Assert.Equal(20f, keptToOuter.ToArray()[0]);
        Assert.Throws<InvalidOperationException>(() => handedUp.ToArray());
    }

    // Results that live in the activation cache rather than owning their buffer (TensorPower's FinishGpuOp path) are
    // freed by the engine's cache when the scope ends, like a finished tape's intermediates, unless kept.
    [SkippableFact]
    public void ActivationCacheResults_AreFreedToo_ExceptTheKeptOne()
    {
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        var source = new Tensor<float>(new[] { 1f, 2f, 3f, 4f }, new[] { 4 });
        Tensor<float> intermediate, kept;
        using (var scope = TensorScope.Begin())
        {
            intermediate = gpu.TensorPower(source, 3f);
            kept = scope.Keep(gpu.TensorPower(source, 0.5f));
        }

        Assert.Equal(2f, kept.ToArray()[3], 4);
        Assert.Throws<InvalidOperationException>(() => intermediate.ToArray());
    }

    // A result built over a buffer it owns (Tensor.FromGpuBuffer, as the fused attention and FFT paths make them).
    [SkippableFact]
    public void OwnedBufferResults_AreFreedToo()
    {
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        // An engine that reports a GPU must expose its backend: a null here is a failure, not a missing device.
        var backend = Assert.IsAssignableFrom<IDirectGpuBackend>(gpu.GetBackend());
        Tensor<float> dropped, kept;
        using (var scope = TensorScope.Begin())
        {
            dropped = Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(new[] { 1f, 2f }), new[] { 2 });
            kept = scope.Keep(Tensor<float>.FromGpuBuffer(backend, backend.AllocateBuffer(new[] { 5f, 6f }), new[] { 2 }));
        }

        Assert.Equal(new[] { 5f, 6f }, kept.ToArray());
        Assert.Throws<InvalidOperationException>(() => dropped.ToArray());
    }

    [SkippableFact]
    public void DisposingOutOfOrder_Throws()
    {
        var outer = TensorScope.Begin();
        var inner = TensorScope.Begin();
        try
        {
            Assert.Throws<InvalidOperationException>(() => outer.Dispose());
        }
        finally
        {
            inner.Dispose();
            outer.Dispose();
        }
        Assert.Null(TensorScope.Current);
    }

    // The control arm makes the comparison mean something: the same loop without a scope reaches the collection
    // threshold, so a scoped loop that does not is freeing its results, not just allocating too little.
    [SkippableFact]
    public void ScopedForwards_NeverMakeACollectionDue_UnscopedOnesDo()
    {
        using var gpu = TryEngine(out string noGpu);
        Skip.If(gpu is null, noGpu);
        // DeviceMemoryReclaim's byte count is how OpenCL, Vulkan and HIP decide to collect. CUDA does not keep it: it
        // reclaims by free device memory (CudaBackend.ReclaimUnderPressure), so neither arm could reach the threshold.
        Skip.If(gpu.TestBackend is CudaBackend,
            "Not applicable: CUDA reclaims by free-memory pressure (ReclaimUnderPressure), not DeviceMemoryReclaim's byte count.");
        const int Length = 1 << 20;   // 4 MB per result
        // Two results per call; about twice the due threshold in all, and under CollectNowAfterDeviceBytes, past which a
        // collection runs on the spot and clears the due flag the control arm reads.
        int calls = (int)(2 * DeviceMemoryReclaim.CollectAfterDeviceBytes / (2 * 4L * Length)) + 1;
        var input = new Tensor<float>(new float[Length], new[] { Length });

        DeviceMemoryReclaim.ResetCountsForTests();
        for (int i = 0; i < calls; i++)
        {
            using var scope = TensorScope.Begin();
            var hidden = gpu.TensorAddScalar(input, 1f);
            scope.Keep(gpu.TensorMultiplyScalar(hidden, 2f)).Dispose();
        }
        bool scopedDue = DeviceMemoryReclaim.CollectionDue;

        DeviceMemoryReclaim.ResetCountsForTests();
        var held = new System.Collections.Generic.List<Tensor<float>>();
        for (int i = 0; i < calls; i++)
        {
            var hidden = gpu.TensorAddScalar(input, 1f);
            held.Add(hidden);   // dead in a real forward; held here so a collection cannot run between calls
            held.Add(gpu.TensorMultiplyScalar(hidden, 2f));
        }
        bool unscopedDue = DeviceMemoryReclaim.CollectionDue;
        DeviceMemoryReclaim.ResetCountsForTests();

        Assert.True(unscopedDue, "control: the unscoped loop never reached the collection threshold.");
        Assert.False(scopedDue, "the scoped loop's results were not returned to the pool when each scope ended.");
    }
}
