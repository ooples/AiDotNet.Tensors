// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.LinearAlgebra;

/// <summary>
/// The chunked stream the lossless encoder writes into (#2322): what goes in comes back out of
/// <see cref="PooledChunkWriteStream.ToArray"/> byte for byte, across chunk boundaries.
/// </summary>
public class PooledChunkWriteStreamTests
{
    [Fact]
    public void ToArray_ReturnsEveryByteWritten_AcrossChunkBoundaries()
    {
        // 2.5 MiB in odd-sized writes, so writes straddle the 1 MiB chunk boundaries.
        int total = (5 << 20) / 2 + 17;
        var expected = new byte[total];
        for (int i = 0; i < total; i++) expected[i] = (byte)(i * 31 + (i >> 9));

        using var stream = new PooledChunkWriteStream();
        stream.WriteByte(expected[0]);
        int at = 1;
        int size = 1;
        while (at < total)
        {
            int n = Math.Min(size, total - at);
            stream.Write(expected, at, n);
            at += n;
            size = size * 3 + 7;
        }

        Assert.Equal(total, stream.Length);
        Assert.Equal(expected, stream.ToArray());
    }

    [Fact]
    public void ToArray_OfAnEmptyStream_IsEmpty()
    {
        using var stream = new PooledChunkWriteStream();
        Assert.Empty(stream.ToArray());
    }

    [Fact]
    public void Disposed_StreamRejectsUse()
    {
        var stream = new PooledChunkWriteStream();
        stream.WriteByte(1);
        stream.Dispose();
        Assert.False(stream.CanWrite);
        Assert.Throws<ObjectDisposedException>(() => stream.WriteByte(2));
        Assert.Throws<ObjectDisposedException>(() => stream.ToArray());
        Assert.Throws<ObjectDisposedException>(() => stream.Flush());
    }
}
