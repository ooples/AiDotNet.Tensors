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

internal static class ResultBufferTrackerTestExtensions
{
    // The backing array of a matrix or vector, read without marking it escaped (test-only identity checks).
    internal static double[] AsSpanUnmarkedArray(this MatrixBase<double> m)
        => System.Runtime.InteropServices.MemoryMarshal.TryGetArray<double>(m.AsMemoryUnmarkedForTests(), out var seg) ? seg.Array! : throw new InvalidOperationException();

    internal static double[] AsSpanUnmarkedArray(this VectorBase<double> v)
        => System.Runtime.InteropServices.MemoryMarshal.TryGetArray<double>(v.AsMemoryUnmarkedForTests(), out var seg) ? seg.Array! : throw new InvalidOperationException();
}
#endif
