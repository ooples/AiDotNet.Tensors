// Copyright (c) CrowdTrainer LLC. All rights reserved.

#if NETCOREAPP
using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The typed pool hands a buffer back only for the element type and count it holds, accounts idle memory in bytes of
/// that type, and never crosses backends.
/// </summary>
public sealed class GpuTypedBufferPoolTests
{
    private sealed class TypedBuffer : IGpuBuffer
    {
        public TypedBuffer(GpuElementType type, int size)
        {
            ElementType = type;
            Size = size;
            Handle = (IntPtr)1;
        }

        public GpuElementType ElementType { get; }
        public int Size { get; }
        public long SizeInBytes => (long)Size * GpuElementTypes.SizeInBytes(ElementType);
        public IntPtr Handle { get; private set; }
        public int Disposals { get; private set; }
        public void Release() => Handle = IntPtr.Zero;
        public void Dispose()
        {
            Disposals++;
            Handle = IntPtr.Zero;
        }
    }

    private static IDirectGpuBackend Backend() => MockDirectGpuBackend.Create(new MockBackendState());

    private static GpuTypedBufferPool Pool() => new(buffer => buffer.Handle != IntPtr.Zero);

    [Fact]
    public void A_buffer_comes_back_only_for_its_own_type_and_count()
    {
        var backend = Backend();
        var pool = Pool();
        var doubles = new TypedBuffer(GpuElementType.Float64, 8);
        Assert.True(pool.TryReturn(backend, doubles));
        Assert.Equal(64, pool.Bytes);
        Assert.Equal(1, pool.Count);

        // Same bytes, another type; same type, another count: neither is this buffer.
        Assert.False(pool.TryTake(backend, GpuElementType.Int64, 8, out _));
        Assert.False(pool.TryTake(backend, GpuElementType.Float32, 16, out _));
        Assert.False(pool.TryTake(backend, GpuElementType.Float64, 4, out _));
        Assert.Equal(64, pool.Bytes);

        Assert.True(pool.TryTake(backend, GpuElementType.Float64, 8, out var taken));
        Assert.Same(doubles, taken);
        Assert.Equal(0, pool.Bytes);
        Assert.Equal(0, pool.Count);
        Assert.Equal(0, doubles.Disposals);
    }

    [Fact]
    public void Idle_memory_is_counted_in_bytes_of_each_type()
    {
        var backend = Backend();
        var pool = Pool();
        Assert.True(pool.TryReturn(backend, new TypedBuffer(GpuElementType.Float64, 10)));   // 80
        Assert.True(pool.TryReturn(backend, new TypedBuffer(GpuElementType.Float16, 10)));   // 20
        Assert.True(pool.TryReturn(backend, new TypedBuffer(GpuElementType.Int8, 10)));      // 10
        Assert.True(pool.TryReturn(backend, new TypedBuffer(GpuElementType.Int8, 10)));      // 10
        Assert.Equal(120, pool.Bytes);
        Assert.Equal(4, pool.Count);
    }

    [Fact]
    public void A_buffer_that_is_released_or_untyped_or_of_another_backend_is_left_with_the_caller()
    {
        var backend = Backend();
        var pool = Pool();
        var released = new TypedBuffer(GpuElementType.Float64, 4);
        released.Release();
        Assert.False(pool.TryReturn(backend, released));
        Assert.False(pool.TryReturn(backend, new TypedBuffer(GpuElementType.Unspecified, 4)));

        Assert.True(pool.TryReturn(backend, new TypedBuffer(GpuElementType.Float64, 4)));
        var foreign = new TypedBuffer(GpuElementType.Float64, 4);
        Assert.False(pool.TryReturn(Backend(), foreign));
        Assert.Equal(0, foreign.Disposals);
        Assert.Equal(32, pool.Bytes);
    }

    [Fact]
    public void Taking_for_another_backend_or_a_released_buffer_disposes_it_instead()
    {
        var backend = Backend();
        var pool = Pool();
        var first = new TypedBuffer(GpuElementType.Float64, 4);
        var second = new TypedBuffer(GpuElementType.Float64, 4);
        Assert.True(pool.TryReturn(backend, first));
        Assert.True(pool.TryReturn(backend, second));
        second.Release();   // freed behind the pool's back (its context went away)

        Assert.True(pool.TryTake(backend, GpuElementType.Float64, 4, out var taken));
        Assert.Same(first, taken);
        Assert.Equal(1, second.Disposals);
        Assert.Equal(0, pool.Bytes);

        Assert.True(pool.TryReturn(backend, first));
        Assert.False(pool.TryTake(Backend(), GpuElementType.Float64, 4, out _));
        Assert.Equal(1, first.Disposals);
        Assert.Equal(0, pool.Bytes);
    }

    [Fact]
    public void Trim_frees_everything_but_the_size_about_to_be_asked_for_and_clear_releases_the_backend()
    {
        var backend = Backend();
        var pool = Pool();
        var kept = new TypedBuffer(GpuElementType.Float64, 4);
        var sameCountOtherType = new TypedBuffer(GpuElementType.Int64, 4);
        var otherCount = new TypedBuffer(GpuElementType.Float64, 5);
        Assert.True(pool.TryReturn(backend, kept));
        Assert.True(pool.TryReturn(backend, sameCountOtherType));
        Assert.True(pool.TryReturn(backend, otherCount));

        pool.Trim(GpuElementType.Float64, 4);
        Assert.Equal(0, kept.Disposals);
        Assert.Equal(1, sameCountOtherType.Disposals);
        Assert.Equal(1, otherCount.Disposals);
        Assert.Equal(32, pool.Bytes);

        pool.Clear();
        Assert.Equal(1, kept.Disposals);
        Assert.Equal(0, pool.Bytes);
        // Cleared, the pool serves whichever backend returns to it next.
        Assert.True(pool.TryReturn(Backend(), new TypedBuffer(GpuElementType.Float32, 2)));
        Assert.Equal(8, pool.Bytes);
    }

    [Fact]
    public void A_drained_size_leaves_no_entry_behind()
    {
        var backend = Backend();
        var pool = Pool();
        for (int length = 1; length <= 8; length++)
            Assert.True(pool.TryReturn(backend, new TypedBuffer(GpuElementType.Float64, length)));
        Assert.Equal(8, pool.Sizes);

        Assert.True(pool.TryTake(backend, GpuElementType.Float64, 1, out _));
        Assert.Equal(7, pool.Sizes);

        // A stack of dead buffers is dropped with them.
        var dead = new TypedBuffer(GpuElementType.Int8, 3);
        Assert.True(pool.TryReturn(backend, dead));
        dead.Release();
        Assert.False(pool.TryTake(backend, GpuElementType.Int8, 3, out _));
        Assert.Equal(7, pool.Sizes);

        pool.Trim(GpuElementType.Float64, 4);
        Assert.Equal(1, pool.Sizes);
        Assert.Equal(1, pool.Count);
        pool.Clear();
        Assert.Equal(0, pool.Sizes);
    }

    [Fact]
    public void A_checked_copy_refuses_a_range_outside_either_buffer()
    {
        var eight = new TypedBuffer(GpuElementType.Float64, 8);
        var four = new TypedBuffer(GpuElementType.Float64, 4);
        Assert.Equal(8, GpuCopyLayout.CheckedElementBytes(eight, 4, four, 0, 4, "Copy"));
        Assert.Equal(8, GpuCopyLayout.CheckedElementBytes(eight, 8, four, 4, 0, "Copy"));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuCopyLayout.CheckedElementBytes(eight, 0, four, 0, 5, "Copy"));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuCopyLayout.CheckedElementBytes(eight, 5, four, 0, 4, "Copy"));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuCopyLayout.CheckedElementBytes(eight, 0, four, 1, 4, "Copy"));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuCopyLayout.CheckedElementBytes(eight, -1, four, 0, 1, "Copy"));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuCopyLayout.CheckedElementBytes(eight, 0, four, -1, 1, "Copy"));
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuCopyLayout.CheckedElementBytes(eight, 0, four, 0, -1, "Copy"));
        // A count near int.MaxValue does not wrap past the check.
        Assert.Throws<ArgumentOutOfRangeException>(() => GpuCopyLayout.CheckedElementBytes(eight, int.MaxValue, four, 0, int.MaxValue, "Copy"));
    }

    [Fact]
    public void Arguments_are_checked()
    {
        var pool = Pool();
        Func<IGpuBuffer, bool>? noCheck = null;
        IGpuBuffer? noBuffer = null;
        IDirectGpuBackend? noBackend = null;
        Assert.Throws<ArgumentNullException>(() => new GpuTypedBufferPool(noCheck));
        Assert.Throws<ArgumentNullException>(() => pool.TryReturn(Backend(), noBuffer));
        Assert.Throws<ArgumentNullException>(() => pool.TryReturn(noBackend, new TypedBuffer(GpuElementType.Float32, 1)));
    }
}
#endif
