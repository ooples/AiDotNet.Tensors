#if NET5_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// Large CPU results recycle deterministically: a <see cref="TensorScope"/> hands the arrays of the results created
/// inside it to the next results of their size when it ends, with no collection, and results nobody scoped or
/// disposed stay within a fixed bound however many the caller makes.
/// </summary>
[Collection(ResultBufferTrackerCollection.Name)]
public sealed class ScopedResultRecyclingTests
{
    // 1M floats: 4 MB results, well past the large-object threshold the tracker pools from.
    private const int N = 1 << 20;
    private static readonly CpuEngine Engine = new();

    public ScopedResultRecyclingTests()
    {
        Assert.True(TensorPool.Enabled, "the tensor pool is disabled, so no result is pooled");
        Assert.False(TensorPool.ForceFreshAllocations, "ForceFreshAllocations is set, so every result is a fresh array");
        Assert.Null(TensorArena.Current);
        ThreadLocalTensorCache<float>.Clear();
    }

    private static Tensor<float> Input(int seed)
    {
        var values = new float[N];
        for (int i = 0; i < N; i++) values[i] = ((i * 7919 + seed * 104729) % 2001 - 1000) * 1e-3f;
        return new Tensor<float>(values, new[] { N });
    }

    // Five large results per call, four of them dead intermediates.
    private static Tensor<float> Chain(Tensor<float> a, Tensor<float> b, Tensor<float> c, Tensor<float> d)
        => Engine.TensorAdd(Engine.TensorMultiply(Engine.TensorSubtract(Engine.TensorMultiply(Engine.TensorAdd(a, b), c), d), c), a);

    private static int Hash(Tensor<float> t)
    {
        int h = unchecked((int)2166136261);
        // Indexer reads: a span would hand out raw access, which stops the array from ever being recycled.
        for (int i = 0; i < t.Length; i += 61) h = unchecked((h ^ MathCompat.SingleToInt32Bits(t[i])) * 16777619);
        return h;
    }

    private static void Settle()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    [Fact]
    public void ScopedLoop_ReusesEveryResult_WithNoCollection_AndBoundedTrackedBytes()
    {
        Tensor<float> a = Input(1), b = Input(2), c = Input(3), d = Input(4);
        int expected = Hash(Chain(a, b, c, d));
        Settle();

        // Warm up: the first calls allocate the arrays the rest reuse.
        for (int i = 0; i < 4; i++) { using var warm = TensorScope.Begin(); Assert.Equal(expected, Hash(Chain(a, b, c, d))); }

        long trackedBefore = ResultBufferTracker.TrackedBytes;
        long inducedBefore = ResultBufferTracker.InducedCollections;
        int gen1 = GC.CollectionCount(1), gen2 = GC.CollectionCount(2);
        long allocated = GC.GetAllocatedBytesForCurrentThread();
        long maxTracked = 0;
        const int iterations = 2000;
        for (int i = 0; i < iterations; i++)
        {
            using var scope = TensorScope.Begin();
            var result = Chain(a, b, c, d);
            if (i % 250 == 0) Assert.Equal(expected, Hash(result));
            maxTracked = Math.Max(maxTracked, ResultBufferTracker.TrackedBytes);
        }
        long perIteration = (GC.GetAllocatedBytesForCurrentThread() - allocated) / iterations;

        Assert.Equal(0, ResultBufferTracker.InducedCollections - inducedBefore);
        Assert.Equal(0, GC.CollectionCount(2) - gen2);
        Assert.Equal(0, GC.CollectionCount(1) - gen1);
        // Only small bookkeeping (tensor objects, the scope) is left to gen 0: never a large array.
        Assert.True(perIteration < 85_000, $"a scoped iteration allocated {perIteration} B: a large result was fresh");
        Assert.True(maxTracked <= Math.Max(trackedBefore, 6L * N * sizeof(float)),
            $"tracked bytes reached {maxTracked >> 20} MB (from {trackedBefore >> 20} MB): scoped results were not reused");
    }

    [Fact]
    public void ReadingAResultAfterItsScope_Throws()
    {
        Tensor<float> a = Input(1), b = Input(2);
        Tensor<float> result;
        using (TensorScope.Begin()) result = Engine.TensorAdd(a, b);
        var error = Assert.Throws<InvalidOperationException>(() => result[0]);
        Assert.Contains("TensorScope", error.Message);
        Assert.Throws<InvalidOperationException>(() => result.AsSpan()[0]);
    }

    [Fact]
    public void KeptResult_SurvivesItsScope_AndIsNotReused()
    {
        Tensor<float> a = Input(1), b = Input(2);
        Tensor<float> kept;
        float[] expected;
        using (var scope = TensorScope.Begin())
        {
            kept = scope.Keep(Engine.TensorAdd(a, b));
            expected = kept.ToArray();
        }
        // Later scoped results of the same size must not take the kept array.
        for (int i = 0; i < 8; i++) { using var scope = TensorScope.Begin(); Engine.TensorMultiply(a, b); }
        Assert.Equal(expected, kept.ToArray());
    }

    [Fact]
    public void DetachedResult_SurvivesEveryOpenScope()
    {
        Tensor<float> a = Input(1), b = Input(2);
        Tensor<float> detached;
        float first;
        using (TensorScope.Begin())
        using (TensorScope.Begin())
        {
            detached = TensorScope.Detach(Engine.TensorAdd(a, b));
            first = detached[0];
        }
        for (int i = 0; i < 8; i++) { using var scope = TensorScope.Begin(); Engine.TensorMultiply(a, b); }
        Assert.Equal(first, detached[0]);
    }

    [Fact]
    public void ResultKeptInANestedScope_IsReleasedByTheEnclosingOne()
    {
        Tensor<float> a = Input(1), b = Input(2);
        Tensor<float> result;
        using (TensorScope.Begin())
        {
            using (var inner = TensorScope.Begin()) result = inner.Keep(Engine.TensorAdd(a, b));
            Assert.Equal(a[5] + b[5], result[5]);
        }
        Assert.Throws<InvalidOperationException>(() => result[5]);
    }

    [Fact]
    public void EscapedResult_IsNeitherInvalidatedNorReused()
    {
        Tensor<float> a = Input(1), b = Input(2);
        float[] raw;
        Tensor<float> result;
        using (TensorScope.Begin())
        {
            result = Engine.TensorAdd(a, b);
            raw = result.GetDataArray();   // raw access: whoever holds it may read it after the scope
        }
        var expected = (float[])raw.Clone();
        for (int i = 0; i < 8; i++) { using var scope = TensorScope.Begin(); Engine.TensorMultiply(a, b); }
        Assert.Equal(expected, raw);
        Assert.Equal(expected[3], result[3]);
    }

    [Fact]
    public void TensorThatExistedBeforeTheScope_IsUntouched_EvenWhenAnOpWritesIt()
    {
        Tensor<float> a = Input(1), b = Input(2);
        var destination = Engine.TensorAdd(a, b);
        using (TensorScope.Begin()) Engine.TensorMultiplyInto(destination, a, b);
        Assert.Equal(a[7] * b[7], destination[7]);
    }

    [Fact]
    public void DisposedResult_ThatTheThreadCacheDeclines_IsStillReusedWithoutACollection()
    {
        Tensor<float> a = Input(1), b = Input(2);
        // Fill this size's thread-cache bucket (four buffers), so the next disposed result is declined by it.
        var held = new Tensor<float>[5];
        for (int i = 0; i < held.Length; i++) held[i] = Engine.TensorAdd(a, b);
        for (int i = 0; i < 4; i++) held[i].Dispose();
        var declined = held[4];
        float[] array = declined.GetDataArrayUnmarked();
        declined.Dispose();

        long induced = ResultBufferTracker.InducedCollections;
        int gen0 = GC.CollectionCount(0);
        bool reused = false;
        var taken = new Tensor<float>[5];
        for (int i = 0; i < taken.Length; i++)
        {
            taken[i] = Engine.TensorAdd(a, b);
            reused |= ReferenceEquals(taken[i].GetDataArrayUnmarked(), array);
        }
        Assert.True(reused, "the declined array waited for a collection");
        Assert.Equal(0, ResultBufferTracker.InducedCollections - induced);
        Assert.Equal(0, GC.CollectionCount(0) - gen0);
        GC.KeepAlive(taken);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static int Unscoped(Tensor<float> a, Tensor<float> b, Tensor<float> c, Tensor<float> d) => Hash(Chain(a, b, c, d));

    [Fact]
    public void UnscopedLoop_WithNoOtherGarbage_StaysBounded_AndTheTrackerNeverRunsGen2()
    {
        // The collections the tracker induces run inside an op, while that op's inputs are alive: they are promoted,
        // and a gen-0 collection never freed them once they died. With nothing else allocating, tracked bytes climbed
        // to the 1 GB cap and a third of the results were fresh until a natural gen-2.
        Tensor<float> a = Input(1), b = Input(2), c = Input(3), d = Input(4);
        int expected = Unscoped(a, b, c, d);
        for (int i = 0; i < 40; i++) Assert.Equal(expected, Unscoped(a, b, c, d));

        long induced = ResultBufferTracker.InducedCollections;
        int gen2 = GC.CollectionCount(2);
        long fresh = GC.GetAllocatedBytesForCurrentThread();
        long maxTracked = 0;
        const int iterations = 600;
        for (int i = 0; i < iterations; i++)
        {
            Assert.Equal(expected, Unscoped(a, b, c, d));
            maxTracked = Math.Max(maxTracked, ResultBufferTracker.TrackedBytes);
        }
        long perIteration = (GC.GetAllocatedBytesForCurrentThread() - fresh) / iterations;
        long budget = ResultBufferTracker.CollectBudgetBytes;

        Assert.True(ResultBufferTracker.InducedCollections - induced > 0, "the unscoped loop never needed a collection");
        Assert.Equal(0, GC.CollectionCount(2) - gen2);
        // At most one collection budget of results in flight, plus the arrays a collection's survivors hold.
        Assert.True(maxTracked <= 4 * budget, $"tracked bytes reached {maxTracked >> 20} MB with a {budget >> 20} MB budget");
        // Five 4 MB results per iteration: most must be reused.
        Assert.True(perIteration < N * sizeof(float), $"an unscoped iteration allocated {perIteration >> 10} KB fresh");
    }
}
#else
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>
/// .NET Framework does not pool large results, so a <see cref="TensorScope"/> recycles none: a CPU result stays
/// readable after its scope.
/// </summary>
public sealed class ScopedResultRecyclingTests
{
    [Fact]
    public void CpuResult_StaysReadableAfterItsScope()
    {
        var engine = new CpuEngine();
        var a = new Tensor<float>(new float[1 << 20], new[] { 1 << 20 });
        a[9] = 2f;
        Tensor<float> result;
        using (TensorScope.Begin()) result = engine.TensorAdd(a, a);
        Assert.Equal(4f, result[9]);
    }
}
#endif
