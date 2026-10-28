// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
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
    private static DirectGpuTensorEngine? TryEngine()
    {
        try
        {
            var gpu = new DirectGpuTensorEngine();
            if (gpu.IsGpuAvailable) return gpu;
            gpu.Dispose();
        }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException) { }
        return null;
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
        using var gpu = TryEngine();
        Skip.If(gpu is null, "No DirectGpu backend.");
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
    public void KeepingAView_KeepsTheResultItViews()
    {
        using var gpu = TryEngine();
        Skip.If(gpu is null, "No DirectGpu backend.");
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
        using var gpu = TryEngine();
        Skip.If(gpu is null, "No DirectGpu backend.");
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
        using var gpu = TryEngine();
        Skip.If(gpu is null, "No DirectGpu backend.");
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
        using var gpu = TryEngine();
        Skip.If(gpu is null, "No DirectGpu backend.");
        var backend = gpu.GetBackend();
        Skip.If(backend is null, "No DirectGpu backend.");
        Tensor<float> dropped, kept;
        using (var scope = TensorScope.Begin())
        {
            dropped = Tensor<float>.FromGpuBuffer(backend!, backend!.AllocateBuffer(new[] { 1f, 2f }), new[] { 2 });
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
        using var gpu = TryEngine();
        Skip.If(gpu is null, "No DirectGpu backend.");
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
