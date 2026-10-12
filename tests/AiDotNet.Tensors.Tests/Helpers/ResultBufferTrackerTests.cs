#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Helpers;

/// <summary>The tracker is process-wide state; these must not run beside anything that allocates large results.</summary>
[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class ResultBufferTrackerCollection
{
    public const string Name = "ResultBufferTracker";
}

/// <summary>
/// Contract of <see cref="ResultBufferTracker"/>: a large result nobody disposed gives its array back once every wrapper
/// is gone, raw access other than a lease keeps it from ever being reused, a live zero-copy view keeps it (and its data),
/// and an explicit return never lets the same array reach two results.
/// </summary>
[Collection(ResultBufferTrackerCollection.Name)]
public sealed class ResultBufferTrackerTests
{
    // 500 x 500 doubles = 2 MB: well above the large-object threshold the tracker starts at.
    private const int N = 500;

    private static Matrix<double> Source()
    {
        var m = new Matrix<double>(N, N);
        var span = m.AsWritableSpanUnmarked();
        for (int i = 0; i < span.Length; i++) span[i] = i * 0.5;
        return m;
    }

    // Produces a result in a frame of its own so the caller holds no reference to it, then returns its array identity
    // through a weak reference only.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static WeakReference ProduceAndDrop(Matrix<double> source, Action<Matrix<double>>? touch = null)
    {
        var result = source.Multiply(2.0);
        touch?.Invoke(result);
        return new WeakReference(result.AsSpanUnmarkedArray());
    }

    private static void Collect()
    {
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
    }

    [Fact]
    public void DiscardedResult_ArrayIsReused_ByTheNextSameSizeResult()
    {
        var source = Source();
        var dropped = ProduceAndDrop(source);
        Collect();
        long reusedBefore = ResultBufferTracker.ReusedArrays;

        var next = source.Multiply(3.0);

        Assert.True(ResultBufferTracker.ReusedArrays > reusedBefore, "the next large result did not reuse a freed array");
        Assert.Same(dropped.Target, next.AsSpanUnmarkedArray());
        // Reused storage carries the new result's values, not the old ones.
        var values = next.AsSpanUnmarked();
        for (int i = 0; i < values.Length; i += 997) Assert.Equal(i * 0.5 * 3.0, values[i]);
    }

    [Fact]
    public void RawAccess_ReleasesTheTrackingEntryAtOnce()
    {
        // An escaped array can never be reused, so the tracker must not keep holding it: it used to stay (dead, held
        // strongly) until a rent of its size scanned it or a trim ran.
        var source = Source();
        var result = source.Multiply(2.0);
        Assert.NotNull(result.ResultOwner);
        long escapedBefore = ResultBufferTracker.EscapedArrays;
        int trackedBefore = ResultBufferTracker.TrackedCount;
        _ = result.AsSpan()[0];
        Assert.Equal(trackedBefore - 1, ResultBufferTracker.TrackedCount);
        Assert.True(ResultBufferTracker.EscapedArrays > escapedBefore);
        Assert.Null(ResultBufferTracker.Find(result.AsSpanUnmarkedArray()));
        GC.KeepAlive(result);
    }

    [Fact]
    public void RawAccess_KeepsTheArrayFromEverBeingReused()
    {
        var source = Source();
        long escapedBefore = ResultBufferTracker.EscapedArrays;
        var dropped = ProduceAndDrop(source, r => _ = r.AsSpan()[0]);
        Collect();

        var next = source.Multiply(3.0);

        Assert.NotSame(dropped.Target, next.AsSpanUnmarkedArray());
        Assert.True(ResultBufferTracker.EscapedArrays > escapedBefore, "the escaped array was not set aside");
    }

    [Fact]
    public void Lease_DoesNotEscape_SoTheArrayIsStillReused()
    {
        var source = Source();
        var dropped = ProduceAndDrop(source, r =>
        {
            using var lease = r.Lease();
            _ = lease.Span[0];
        });
        Collect();

        var next = source.Multiply(3.0);

        Assert.Same(dropped.Target, next.AsSpanUnmarkedArray());
    }

    [Fact]
    public void LiveZeroCopyView_KeepsTheArrayAndItsData()
    {
        var source = Source();
        Vector<double> view = MakeViewOfDroppedResult(source);
        Collect();

        // Another large result of the same size must not be handed the array the view still reads.
        var other = source.Multiply(7.0);
        Assert.NotSame(view.AsSpanUnmarkedArray(), other.AsSpanUnmarkedArray());
        var values = view.AsSpanUnmarked();
        for (int i = 0; i < values.Length; i += 997) Assert.Equal(i * 0.5 * 2.0, values[i]);
        GC.KeepAlive(view);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Vector<double> MakeViewOfDroppedResult(Matrix<double> source)
    {
        var result = source.Multiply(2.0);
        // Zero-copy: the vector wraps the matrix's array (the constructor attaches the tracked owner).
        return Vector<double>.FromMemory(new Memory<double>(result.AsSpanUnmarkedArray()));
    }

    [Fact]
    public void ExplicitReturn_UntracksTheArray_SoItIsNeverHandedOutTwice()
    {
        var source = Source();
        var first = (Matrix<double>)source.Multiply(2.0);
        var array = first.AsSpanUnmarkedArray();
        MatrixAllocator.Return(first);   // back to the thread cache, and untracked

        var a = source.Multiply(3.0);    // takes it from the cache
        Collect();                       // `first` is collected now; its old owner must not free the array again
        var b = source.Multiply(4.0);

        Assert.Same(array, a.AsSpanUnmarkedArray());
        Assert.NotSame(a.AsSpanUnmarkedArray(), b.AsSpanUnmarkedArray());
        GC.KeepAlive(a);
    }

    [Fact]
    public void ReusedArray_DoesNotInheritThePreviousOwnersHostSyncState()
    {
        var source = Source();
        // The dropped result had device-side state keyed by its array: a device binding and a retention mark.
        var dropped = ProduceAndDrop(source, r =>
        {
            var array = r.AsSpanUnmarkedArray();
            HostSync.ForArray(array).GetOrCreateDevice();
            HostSync.MarkRetained(array);
        });
        Collect();

        var next = source.Multiply(3.0);

        Assert.Same(dropped.Target, next.AsSpanUnmarkedArray());
        Assert.Null(HostSync.FindForArray(next.AsSpanUnmarkedArray()));
        GC.KeepAlive(next);
    }

    [Fact]
    public void Stress_PoisonsFreedArraysBeforeReuse()
    {
        bool saved = ResultBufferTracker.Stress;
        ResultBufferTracker.Stress = true;
        try
        {
            var source = Source();
            var dropped = ProduceAndDrop(source);
            Collect();
            var array = (double[]?)dropped.Target;
            Assert.NotNull(array);
            // The next large rent collects, takes the freed array and poisons it before the new result overwrites it;
            // observe the poison through a rent that does not write.
            var raw = MatrixAllocator.RentUninitialized<double>(N, N);
            Assert.Same(array, raw.AsSpanUnmarkedArray());
            Assert.True(double.IsNaN(raw.AsSpanUnmarked()[12345]), "a reclaimed array was not poisoned in stress mode");
        }
        finally
        {
            ResultBufferTracker.Stress = saved;
        }
    }
}

/// <summary>
/// Collection pacing of <see cref="ResultBufferTracker"/> in loops that discard their results: after warm-up a steady
/// loop runs on reuse, a size whose results are held does not slow reuse of other sizes, and trimming the free arrays
/// keeps the sizes still being asked for. Each test uses sizes no other test uses, since the backoff is kept per size.
/// </summary>
[Collection(ResultBufferTrackerCollection.Name)]
public sealed class ResultBufferTrackerPacingTests
{
    private static Matrix<double> Square(int n)
    {
        var m = new Matrix<double>(n, n);
        var span = m.AsWritableSpanUnmarked();
        for (int i = 0; i < span.Length; i++) span[i] = i;
        return m;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Discard(Matrix<double> source) => GC.KeepAlive(source.Multiply(2.0));

    // Fraction of `measured` discarded results that reused an array, after `warm` unmeasured ones.
    private static double ReusedFraction(Matrix<double> source, int warm, int measured)
    {
        for (int i = 0; i < warm; i++) Discard(source);
        long before = ResultBufferTracker.ReusedArrays;
        for (int i = 0; i < measured; i++) Discard(source);
        return (ResultBufferTracker.ReusedArrays - before) / (double)measured;
    }

    private static void WithSettings(long collectBudget, long freeCap, Action body)
    {
        long savedBudget = ResultBufferTracker.CollectBudgetBytes, savedCap = ResultBufferTracker.FreeCapBytes;
        bool savedStress = ResultBufferTracker.Stress;
        ResultBufferTracker.CollectBudgetBytes = collectBudget;
        ResultBufferTracker.FreeCapBytes = freeCap;
        ResultBufferTracker.Stress = false;   // stress collects on every miss, which is not the pacing under test
        try { body(); }
        finally
        {
            ResultBufferTracker.CollectBudgetBytes = savedBudget;
            ResultBufferTracker.FreeCapBytes = savedCap;
            ResultBufferTracker.Stress = savedStress;
        }
    }

    [Fact]
    public void SteadyLoop_RunsOnReuse_AfterWarmUp()
    {
        // Pacing on fresh bytes alone gave each cycle a full budget of fresh arrays once the freed ones ran out: about
        // half the results of a steady loop were fresh, however long it ran.
        WithSettings(8L << 20, 64L << 20, () =>
        {
            var source = Square(410);                       // 1.3 MB results
            double reused = ReusedFraction(source, warm: 40, measured: 200);
            Assert.True(reused > 0.9, $"a steady loop reused {reused:P0} of its results after warm-up");
        });
    }

    [Fact]
    public void HeldResultsOfOneSize_DoNotSlowReuseOfAnother()
    {
        // One process-wide backoff let a size whose results were held (every collection for it freed nothing) push the
        // collection budget to 32x for every size: a maxpool loop after a conv2d loop reused 3% of its results.
        WithSettings(4L << 20, 64L << 20, () =>
        {
            // 100 held results of 1.4 MB: the collections asked for this size free nothing at 4, 12, 28, 60 and 124 MB,
            // which takes its backoff to the 32x cap.
            var held = new List<Matrix<double>>();
            var heldSource = Square(420);
            for (int i = 0; i < 100; i++) held.Add(heldSource.Multiply(2.0));

            var source = Square(430);                       // 1.5 MB results, discarded
            double reused = ReusedFraction(source, warm: 10, measured: 200);
            GC.KeepAlive(held);
            Assert.True(reused > 0.9, $"after another size's results were held, this size reused {reused:P0} of its results");
        });
    }

    [Fact]
    public void Trim_KeepsTheSizeStillInUse_OverSizesNoLongerAskedFor()
    {
        // Trimming largest-first dropped the arrays of the biggest size still in use (a 1024x1024 transpose reused 19%)
        // while free arrays of sizes nobody asked for any more stayed.
        WithSettings(8L << 20, 12L << 20, () =>
        {
            var stale = Square(400);                        // 1.25 MB results, discarded once and never asked for again
            for (int i = 0; i < 12; i++) Discard(stale);

            var source = Square(700);                       // 3.9 MB results, the largest size, in steady use
            double reused = ReusedFraction(source, warm: 30, measured: 120);
            Assert.True(reused > 0.8, $"the largest size in use reused {reused:P0} of its results with stale sizes free");
        });
    }
}

/// <summary>
/// A thread-cache hit is in an explicit-return cycle, so its result is not tracked (the registration and unregistration
/// cost about 1-2 us per call in a 1M-element subtract + Return loop); a result dropped instead of returned is collected,
/// the next rent misses the cache, and that result is tracked again.
/// </summary>
[Collection(ResultBufferTrackerCollection.Name)]
public sealed class ResultBufferTrackerExplicitReturnTests
{
    private const int Length = 1 << 20;   // 4 MB of floats: well above the tracking threshold

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static bool RentIsTracked(bool returnIt)
    {
        var t = TensorAllocator.RentUninitialized<float>(new[] { Length });
        bool tracked = t.DataVector.ResultOwner is not null;
        if (returnIt) TensorPool.Return(t);
        return tracked;
    }

    [Fact]
    public void ThreadCacheHits_AreNotTracked_AndDroppedResultsAreTrackedAgain()
    {
        bool savedStress = ResultBufferTracker.Stress;
        ResultBufferTracker.Stress = false;
        try
        {
            RentIsTracked(returnIt: true);                              // seeds the thread cache with one array
            Assert.False(RentIsTracked(returnIt: true), "a cache hit in an explicit-return cycle was tracked");
            Assert.False(RentIsTracked(returnIt: false), "a cache hit was tracked");   // ...and this one is dropped
            GC.Collect();
            GC.WaitForPendingFinalizers();
            // The dropped array is gone and the cache is empty: this rent misses and its result is tracked.
            Assert.True(RentIsTracked(returnIt: true), "a result after a cache miss was not tracked");
        }
        finally
        {
            ResultBufferTracker.Stress = savedStress;
        }
    }
}

internal static class ResultBufferTrackerTestExtensions
{
    // The backing array of a matrix or vector, read without marking it escaped (test-only identity checks).
    internal static double[] AsSpanUnmarkedArray(this MatrixBase<double> m)
        => System.Runtime.InteropServices.MemoryMarshal.TryGetArray<double>(m.AsMemoryUnmarkedForTests(), out var seg) ? seg.Array! : throw new InvalidOperationException();

    internal static double[] AsSpanUnmarkedArray(this VectorBase<double> v)
        => System.Runtime.InteropServices.MemoryMarshal.TryGetArray<double>(v.AsMemoryUnmarkedForTests(), out var seg) ? seg.Array! : throw new InvalidOperationException();
}
#endif
