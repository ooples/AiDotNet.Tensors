// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.IO;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// A write-only, append-only stream over chunks rented from <see cref="ArrayPool{T}.Shared"/>. It
/// collects output of unknown final size, such as a compressor's, without a pre-sized or growing
/// contiguous buffer: <see cref="ToArray"/> allocates the one exact-length result, and disposing
/// the stream returns the chunks to the pool.
/// </summary>
internal sealed class PooledChunkWriteStream : Stream
{
    // 1 MiB: the largest array the shared pool keeps on every target framework.
    private const int ChunkSize = 1 << 20;

    private readonly List<byte[]> _chunks = new List<byte[]>();
    private long _length;
    private bool _disposed;

    public override bool CanRead => false;

    public override bool CanSeek => false;

    public override bool CanWrite => !_disposed;

    public override long Length => _length;

    public override long Position
    {
        get => _length;
        set => throw new NotSupportedException();
    }

    public override void Write(byte[] buffer, int offset, int count)
    {
        if (buffer is null) throw new ArgumentNullException(nameof(buffer));
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (count < 0 || count > buffer.Length - offset) throw new ArgumentOutOfRangeException(nameof(count));
        Append(new ReadOnlySpan<byte>(buffer, offset, count));
    }

#if NETCOREAPP2_1_OR_GREATER || NETSTANDARD2_1_OR_GREATER
    public override void Write(ReadOnlySpan<byte> buffer) => Append(buffer);
#endif

    public override void WriteByte(byte value)
    {
        Span<byte> one = stackalloc byte[1];
        one[0] = value;
        Append(one);
    }

    /// <summary>Copies everything written so far into a new array of exactly <see cref="Length"/> bytes.</summary>
    public byte[] ToArray()
    {
        ThrowIfDisposed();
        if (_length > int.MaxValue)
            throw new InvalidOperationException($"Stream holds {_length} bytes, more than one array can.");
        var result = new byte[_length];
        long remaining = _length;
        int at = 0;
        foreach (var chunk in _chunks)
        {
            int n = (int)Math.Min(ChunkSize, remaining);
            Buffer.BlockCopy(chunk, 0, result, at, n);
            at += n;
            remaining -= n;
        }
        return result;
    }

    /// <summary>Copies everything written so far to the start of <paramref name="destination"/>, which must hold <see cref="Length"/> bytes.</summary>
    public void CopyTo(byte[] destination)
    {
        ThrowIfDisposed();
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (destination.Length < _length)
            throw new ArgumentException($"Destination holds {destination.Length} bytes; the stream has {_length}.", nameof(destination));
        long remaining = _length;
        int at = 0;
        foreach (var chunk in _chunks)
        {
            if (remaining <= 0) break;
            int n = (int)Math.Min(ChunkSize, remaining);
            Buffer.BlockCopy(chunk, 0, destination, at, n);
            at += n;
            remaining -= n;
        }
    }

    // Writes land in the chunks directly, so there is nothing to flush.
    public override void Flush() => ThrowIfDisposed();

    public override int Read(byte[] buffer, int offset, int count) => throw new NotSupportedException();

    public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

    public override void SetLength(long value) => throw new NotSupportedException();

    protected override void Dispose(bool disposing)
    {
        if (!_disposed)
        {
            _disposed = true;
            foreach (var chunk in _chunks) ArrayPool<byte>.Shared.Return(chunk);
            _chunks.Clear();
        }
        base.Dispose(disposing);
    }

    private void Append(ReadOnlySpan<byte> data)
    {
        ThrowIfDisposed();
        while (!data.IsEmpty)
        {
            int used = (int)(_length % ChunkSize);
            int index = (int)(_length / ChunkSize);
            if (index == _chunks.Count) _chunks.Add(ArrayPool<byte>.Shared.Rent(ChunkSize));
            int n = Math.Min(ChunkSize - used, data.Length);
            data.Slice(0, n).CopyTo(_chunks[index].AsSpan(used));
            _length += n;
            data = data.Slice(n);
        }
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(PooledChunkWriteStream));
    }
}
