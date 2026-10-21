#if NET5_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// #1331: disposing a large result returns its pooled array to the thread cache at once, so the next result of that
/// size reuses it without waiting for a collection, unless something other than the disposed tensor can still read
/// the array.
/// </summary>
[Collection(ResultBufferTrackerCollection.Name)]
public sealed class DisposeRecyclesResultTests
{
    // The issue's MultiHeadAttentionForward output shape: [128, 32, 64] floats, 1 MB.
    private static readonly int[] Shape = { 128, 32, 64 };

    private static readonly CpuEngine Engine = new();

    private static Tensor<float> Input(float scale)
    {
        var t = new Tensor<float>(Shape);
        var span = t.AsWritableSpan();
        for (int i = 0; i < span.Length; i++) span[i] = (i % 101) * scale;
        return t;
    }

    private static float[] ArrayOf(Tensor<float> t) => t.GetDataArrayUnmarked();

    /// <summary>
    /// Every test here reasons about one array reaching the next result, which needs that result to come from the thread
    /// cache. Four pieces of process or thread state ahead of it can place the array somewhere else instead - a disabled
    /// pool, ForceFreshAllocations (#318), an active arena (allocation tier 0), and the cache's own limits of four
    /// buffers per size and 64 MB per thread. All four outlive the test that set them, which is why
    /// <see cref="DisposedResult_ArrayGoesToTheNextResult_WithoutACollection"/> passed alone and failed 15 minutes into
    /// the suite (#1477) on nothing but array identity, naming no cause. The cache is this class's to reset - it is per
    /// thread and per element type, and xUnit builds the class on the thread that runs the test - and the other three
    /// are asserted so a recurrence says which one took the allocation.
    /// </summary>
    public DisposeRecyclesResultTests()
    {
        Assert.True(TensorPool.Enabled, "the tensor pool is disabled, so no result is pooled");
        Assert.False(TensorPool.ForceFreshAllocations, "ForceFreshAllocations is set, so every result is a fresh array");
        Assert.Null(TensorArena.Current);
        ThreadLocalTensorCache<float>.Clear();
    }

    [Fact]
    public void DisposedResult_ArrayGoesToTheNextResult_WithoutACollection()
    {
        var a = Input(1f);
        var b = Input(2f);
        var first = Engine.TensorAdd(a, b);
        var firstArray = ArrayOf(first);
        int gen0 = GC.CollectionCount(0);

        first.Dispose();
        // Which half failed, if this ever fails again: the cache takes the disposed array, or the reuse below cannot
        // happen at all. Asserting it here separates "the cache declined" from "the array was not handed on".
        Assert.True(ThreadLocalTensorCache<float>.RetainedBytes >= (long)firstArray.Length * sizeof(float),
            $"the thread cache did not take the disposed result (retains {ThreadLocalTensorCache<float>.RetainedBytes} bytes)");
        var second = Engine.TensorAdd(b, b);

        Assert.Equal(gen0, GC.CollectionCount(0));
        Assert.Same(firstArray, ArrayOf(second));
        // The reused array holds the new result's values.
        var values = second.AsSpan();
        for (int i = 0; i < values.Length; i += 997) Assert.Equal((i % 101) * 4f, values[i]);
    }

    [Fact]
    public void DisposeInALoop_AllocatesNoResultArrays_AndInducesNoCollections()
    {
        var a = Input(1f);
        var b = Input(2f);
        // Warm the path, the thread cache and the tracker.
        for (int i = 0; i < 4; i++) Engine.TensorAdd(a, b).Dispose();

        long induced = ResultBufferTracker.InducedCollections;
        long before = GC.GetAllocatedBytesForCurrentThread();
        const int Calls = 200;
        for (int i = 0; i < Calls; i++)
        {
            using var y = Engine.TensorAdd(a, b);
        }
        long perCall = (GC.GetAllocatedBytesForCurrentThread() - before) / Calls;

        // A fresh result array is 1 MB; every other call took one before this fix (a result whose array came from the
        // thread cache was untracked, so disposing it returned nothing).
        Assert.True(perCall < 64 * 1024, $"allocated {perCall} bytes per call");
        Assert.Equal(induced, ResultBufferTracker.InducedCollections);
    }

    [Fact]
    public void MultiHeadAttentionForward_DisposedInALoop_ReusesItsOutput()
    {
        // The issue's case: AIsEval [128, 32, 64], whose 1 MB output was written through GetDataArray(), which marked
        // every call's output escaped, so none was ever reused.
        const int Heads = 4, DModel = 64;
        var input = Input(0.01f);
        Tensor<float> Weight(float scale)
        {
            var w = new Tensor<float>(new[] { DModel, DModel });
            var span = w.AsWritableSpan();
            for (int i = 0; i < span.Length; i++) span[i] = ((i * 7) % 13 - 6) * scale;
            return w;
        }
        var q = Weight(0.01f); var k = Weight(0.02f); var v = Weight(0.03f); var o = Weight(0.04f);
        for (int i = 0; i < 4; i++) Engine.MultiHeadAttentionForward(input, q, k, v, o, Heads).Dispose();

        long before = GC.GetAllocatedBytesForCurrentThread();
        const int Calls = 50;
        for (int i = 0; i < Calls; i++)
        {
            using var y = Engine.MultiHeadAttentionForward(input, q, k, v, o, Heads);
        }
        long perCall = (GC.GetAllocatedBytesForCurrentThread() - before) / Calls;

        Assert.True(perCall < 256 * 1024, $"allocated {perCall} bytes per call (the output alone is 1 MB)");
    }

    [Fact]
    public void EscapedResult_IsNotRecycledOnDispose()
    {
        var a = Input(1f);
        var first = Engine.TensorAdd(a, a);
        var raw = first.GetDataArray();
        float kept = raw[777];

        first.Dispose();
        var second = Engine.TensorAdd(Input(3f), a);

        Assert.NotSame(raw, ArrayOf(second));
        Assert.Equal(kept, raw[777]);
    }

    [Fact]
    public void LiveView_KeepsTheArray_UntilItIsDisposedToo()
    {
        var a = Input(1f);
        var first = Engine.TensorAdd(a, a);
        var array = ArrayOf(first);
        var view = first.Reshape(128 * 32, 64);

        first.Dispose();
        var second = Engine.TensorAdd(Input(3f), a);
        Assert.NotSame(array, ArrayOf(second));
        Assert.Equal((777 % 101) * 2f, view.AsSpan()[777]);

        view.Dispose();
        second.Dispose();
        var third = Engine.TensorAdd(a, a);
        Assert.True(ReferenceEquals(array, ArrayOf(third)) || ReferenceEquals(ArrayOf(third), ArrayOf(second)),
            "once its last view was disposed, the result's array was not reused");
    }

    [Fact]
    public void TensorOverTheSameVector_KeepsTheArray()
    {
        var a = Input(1f);
        var first = Engine.TensorAdd(a, a);
        var array = ArrayOf(first);
        // Its own storage over the result's vector: disposing the result releases only the result's storage.
        var sibling = new Tensor<float>(Shape, first.DataVector);

        first.Dispose();
        var second = Engine.TensorAdd(Input(3f), a);

        Assert.NotSame(array, ArrayOf(second));
        Assert.Equal((777 % 101) * 2f, sibling.AsSpan()[777]);
        GC.KeepAlive(sibling);
    }

    [Fact]
    public void ZeroCopyWrap_KeepsTheArray()
    {
        var a = Input(1f);
        var first = Engine.TensorAdd(a, a);
        var array = ArrayOf(first);
        var wrap = Tensor<float>.FromMemory(new Memory<float>(array), Shape);

        first.Dispose();
        var second = Engine.TensorAdd(Input(3f), a);

        Assert.NotSame(array, ArrayOf(second));
        Assert.Equal((777 % 101) * 2f, wrap.AsSpan()[777]);
        GC.KeepAlive(wrap);
    }
}
#endif
