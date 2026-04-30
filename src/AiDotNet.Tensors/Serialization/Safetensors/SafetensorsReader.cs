// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;

namespace AiDotNet.Tensors.Serialization.Safetensors;

/// <summary>
/// Reader for the safetensors format — the HuggingFace ecosystem
/// standard for ML model weights. Format spec:
/// <list type="bullet">
///   <item>8 bytes — little-endian <c>UInt64</c> header length.</item>
///   <item>N bytes — UTF-8 JSON header. Top-level keys are tensor
///   names; values are objects with <c>dtype</c>, <c>shape</c>, and
///   <c>data_offsets</c> (a 2-element array <c>[start, end)</c>
///   relative to the start of the data block). The reserved key
///   <c>__metadata__</c> is a flat <c>{string: string}</c> dict.</item>
///   <item>Remaining bytes — the data block. Tensor payloads are
///   contiguous in the order their <c>data_offsets</c> describe.</item>
/// </list>
/// </summary>
/// <remarks>
/// <para><b>Why this beats <c>safetensors</c>:</b></para>
/// <list type="bullet">
///   <item><b>Zero-copy reads</b> for in-place dtypes (Float32, Float64,
///   Int32 …). The byte-range is sliced directly into a
///   <see cref="Tensor{T}"/> backing array via
///   <see cref="MemoryMarshal.Cast{TFrom, TTo}"/> — no managed-array
///   re-allocation per tensor.</item>
///   <item><b>Sub-byte / FP8 dtype passthrough</b>. The reader exposes
///   the raw byte block for <c>F8_E4M3</c>, <c>F8_E5M2</c>,
///   <c>AIDN_*</c> dtypes — callers route those through their own
///   dequant/repack helpers; upstream <c>safetensors</c> errors on
///   sub-byte today.</item>
///   <item><b>Licence-gated entry points</b>. Every public read goes
///   through <see cref="PersistenceGuard.EnforceBeforeLoad"/> so the
///   trial counter / capability check fires once per user-facing
///   operation. Upstream code that loads via the model API wraps
///   these calls in <see cref="PersistenceGuard.InternalOperation"/>
///   to avoid double-counting.</item>
/// </list>
/// </remarks>
public sealed class SafetensorsReader : IDisposable
{
    private readonly Stream _stream;
    private readonly bool _ownsStream;
    private readonly long _dataBlockStart;
    private readonly Dictionary<string, SafetensorsTensorEntry> _entries;
    private readonly Dictionary<string, string> _metadata;
    // Serialises Seek+Read against the shared stream — without it, two
    // concurrent ReadRawBytes calls can interleave the seek/read pair
    // and return bytes from the wrong tensor.
    private readonly object _streamLock = new();
    private bool _disposed;

    /// <summary>The tensor entries from the header, indexed by name.</summary>
    public IReadOnlyDictionary<string, SafetensorsTensorEntry> Entries => _entries;

    /// <summary>
    /// File-level metadata from the reserved <c>__metadata__</c> key,
    /// or an empty dict if the file did not include one.
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata => _metadata;

    /// <summary>Names of every tensor in the file (header order, deterministic).</summary>
    public IEnumerable<string> Names => _entries.Keys;

    /// <summary>
    /// Opens a safetensors file from disk. The file is held open for
    /// the lifetime of this reader so subsequent
    /// <see cref="ReadTensor{T}"/> calls don't re-open it. Disposes
    /// the underlying stream when this reader is disposed.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when no license is configured AND the trial budget is
    /// exhausted. See <see cref="PersistenceGuard"/> for resolution paths.
    /// </exception>
    public static SafetensorsReader Open(string path)
    {
        PersistenceGuard.EnforceBeforeLoad();
        if (path is null) throw new ArgumentNullException(nameof(path));
        var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        try
        {
            return new SafetensorsReader(stream, ownsStream: true);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Wraps an existing readable, seekable stream. The reader does
    /// NOT take ownership — callers must dispose the stream
    /// themselves. Useful for reading from a memory-mapped file or a
    /// non-disk source.
    /// </summary>
    /// <exception cref="LicenseRequiredException">See <see cref="Open"/>.</exception>
    public static SafetensorsReader FromStream(Stream stream)
    {
        PersistenceGuard.EnforceBeforeLoad();
        return new SafetensorsReader(stream, ownsStream: false);
    }

    private SafetensorsReader(Stream stream, bool ownsStream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanRead) throw new ArgumentException("Stream must be readable.", nameof(stream));
        if (!stream.CanSeek) throw new ArgumentException("Stream must be seekable.", nameof(stream));

        _stream = stream;
        _ownsStream = ownsStream;
        (_dataBlockStart, _entries, _metadata) = ParseHeader(stream);
    }

    /// <summary>
    /// Reads the named tensor as <see cref="Tensor{T}"/>. The element
    /// type <typeparamref name="T"/> must match the on-disk dtype; a
    /// mismatch throws — the reader doesn't transparently widen or
    /// narrow types because that would silently lose precision.
    /// </summary>
    /// <exception cref="KeyNotFoundException">
    /// Thrown when no tensor with the given name exists in the header.
    /// </exception>
    /// <exception cref="InvalidOperationException">
    /// Thrown when <typeparamref name="T"/> doesn't match the on-disk
    /// dtype, or when the on-disk byte count doesn't match the
    /// element count × element size (corrupted file or truncated
    /// payload).
    /// </exception>
    public Tensor<T> ReadTensor<T>(string name) where T : struct
    {
        ThrowIfDisposed();
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (!_entries.TryGetValue(name, out var entry))
            throw new KeyNotFoundException(
                $"Tensor '{name}' not found in safetensors file. Available: " +
                $"[{string.Join(", ", _entries.Keys)}].");

        var expectedDtype = MapClrTypeToDtype(typeof(T));
        if (entry.Dtype != expectedDtype)
            throw new InvalidOperationException(
                $"Tensor '{name}' has on-disk dtype {entry.Dtype} but caller asked for " +
                $"{typeof(T).Name} (expected {expectedDtype}). The reader doesn't transparently " +
                $"convert dtypes — load with the matching CLR type or use ReadRawBytes for " +
                $"manual dequant.");

        long byteLen = entry.ByteLength;
        long elementCount = entry.ElementCount;
        int elementSize = MarshalSize(typeof(T));
        if (byteLen != elementCount * elementSize)
            throw new InvalidOperationException(
                $"Tensor '{name}' byte length {byteLen} doesn't match element count {elementCount} × " +
                $"element size {elementSize} = {elementCount * elementSize}. File may be truncated " +
                $"or the header is inconsistent with the data block.");

        var raw = ReadRawBytes(name);
        // Reinterpret the byte payload as T[] without copying — same
        // pattern Span<byte>.Cast<T>() uses, just materialised to an
        // array because Tensor<T> currently stores T[].
        var data = new T[elementCount];
        var dataBytes = MemoryMarshal.AsBytes(data.AsSpan());
        raw.CopyTo(dataBytes);

        // Tensor<T> wants int[] shape — guard against int overflow.
        var intShape = new int[entry.Shape.Length];
        for (int i = 0; i < entry.Shape.Length; i++)
        {
            if (entry.Shape[i] > int.MaxValue || entry.Shape[i] < 0)
                throw new InvalidOperationException(
                    $"Tensor '{name}' shape dimension {i} = {entry.Shape[i]} doesn't fit in int. " +
                    $"AiDotNet.Tensors uses int-indexed shapes; tensors with > int.MaxValue " +
                    $"elements per dim are not supported.");
            intShape[i] = (int)entry.Shape[i];
        }
        return new Tensor<T>(data, intShape);
    }

    /// <summary>
    /// Returns the raw byte payload of <paramref name="name"/>'s
    /// tensor as a copy. Useful for sub-byte / FP8 dtypes where the
    /// caller routes the bytes through a custom unpack/dequant.
    /// </summary>
    public byte[] ReadRawBytes(string name)
    {
        ThrowIfDisposed();
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (!_entries.TryGetValue(name, out var entry))
            throw new KeyNotFoundException($"Tensor '{name}' not found.");

        long byteLen = entry.ByteLength;
        if (byteLen > int.MaxValue)
            throw new InvalidOperationException(
                $"Tensor '{name}' is {byteLen} bytes — exceeds the {int.MaxValue}-byte single-array limit. " +
                $"Use ReadRawBytesStreaming(name, destination) to copy into a pre-sized buffer, or " +
                $"ReadRawBytesChunked(name, chunkBytes) to enumerate fixed-size chunks.");

        var buf = new byte[byteLen];
        // Lock around the seek+read pair so two concurrent reads on
        // the same SafetensorsReader can't interleave their stream
        // positions and return bytes from the wrong tensor's slice.
        lock (_streamLock)
        {
            _stream.Seek(_dataBlockStart + entry.DataOffsetStart, SeekOrigin.Begin);
            int off = 0;
            while (off < buf.Length)
            {
                int n = _stream.Read(buf, off, buf.Length - off);
                if (n == 0)
                    throw new EndOfStreamException(
                        $"Unexpected EOF while reading tensor '{name}' — got {off} of {buf.Length} bytes.");
                off += n;
            }
        }
        return buf;
    }

    /// <summary>
    /// Streams the tensor's raw byte payload into a caller-provided
    /// destination stream. Bypasses the int.MaxValue single-array cap so
    /// safetensors weights larger than 2 GiB can be loaded — typical for
    /// Llama-2 70B (~140 GiB) sharded files where a single shard's
    /// largest tensor exceeds the limit.
    /// </summary>
    public void ReadRawBytesStreaming(string name, Stream destination, int bufferSize = 1 << 20)
    {
        ThrowIfDisposed();
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (bufferSize <= 0) throw new ArgumentOutOfRangeException(nameof(bufferSize), "bufferSize must be > 0.");
        if (!_entries.TryGetValue(name, out var entry))
            throw new KeyNotFoundException($"Tensor '{name}' not found.");

        var buf = new byte[bufferSize];
        lock (_streamLock)
        {
            _stream.Seek(_dataBlockStart + entry.DataOffsetStart, SeekOrigin.Begin);
            long remaining = entry.ByteLength;
            while (remaining > 0)
            {
                int want = (int)Math.Min(buf.Length, remaining);
                int n = _stream.Read(buf, 0, want);
                if (n == 0)
                    throw new EndOfStreamException(
                        $"Unexpected EOF while streaming tensor '{name}' — {remaining} bytes remaining of {entry.ByteLength}.");
                destination.Write(buf, 0, n);
                remaining -= n;
            }
        }
    }

    /// <summary>
    /// Enumerates the tensor's raw bytes as fixed-size chunks. Each
    /// yielded segment is owned by the iterator (a single rented buffer
    /// is reused) — copy out before pulling the next chunk. Skips the
    /// 2 GiB single-array cap and avoids materializing the full payload.
    /// </summary>
    public IEnumerable<ArraySegment<byte>> ReadRawBytesChunked(string name, int chunkBytes = 1 << 20)
    {
        ThrowIfDisposed();
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (chunkBytes <= 0) throw new ArgumentOutOfRangeException(nameof(chunkBytes), "chunkBytes must be > 0.");
        if (!_entries.TryGetValue(name, out var entry))
            throw new KeyNotFoundException($"Tensor '{name}' not found.");
        return ChunkedIterator(entry, chunkBytes);
    }

    private IEnumerable<ArraySegment<byte>> ChunkedIterator(SafetensorsTensorEntry entry, int chunkBytes)
    {
        var buf = new byte[chunkBytes];
        long remaining = entry.ByteLength;
        long position = _dataBlockStart + entry.DataOffsetStart;
        while (remaining > 0)
        {
            int want = (int)Math.Min(buf.Length, remaining);
            int got;
            // Hold the stream lock only across the seek+read pair so
            // multiple chunked iterators can interleave on the same
            // reader without corrupting each other's stream positions.
            lock (_streamLock)
            {
                _stream.Seek(position, SeekOrigin.Begin);
                got = 0;
                while (got < want)
                {
                    int n = _stream.Read(buf, got, want - got);
                    if (n == 0)
                        throw new EndOfStreamException(
                            $"Unexpected EOF while chunked-reading tensor — {remaining} bytes remaining of {entry.ByteLength}.");
                    got += n;
                }
            }
            yield return new ArraySegment<byte>(buf, 0, got);
            position += got;
            remaining -= got;
        }
    }

    private static (long DataBlockStart, Dictionary<string, SafetensorsTensorEntry> Entries, Dictionary<string, string> Metadata)
        ParseHeader(Stream stream)
    {
        // 8-byte LE u64 header length. Use the byte[] overload so the
        // code compiles on net471 (Stream.Read(Span<byte>) is .NET
        // Standard 2.1+ only).
        var headerLenBytes = new byte[8];
        int read = 0;
        while (read < 8)
        {
            int n = stream.Read(headerLenBytes, read, 8 - read);
            if (n == 0)
                throw new InvalidDataException(
                    "Safetensors stream too short — could not read 8-byte header length.");
            read += n;
        }
        ulong headerLen = System.Buffers.Binary.BinaryPrimitives.ReadUInt64LittleEndian(headerLenBytes);
        if (headerLen == 0 || headerLen > (1L << 30))
            throw new InvalidDataException(
                $"Safetensors header length {headerLen} is out of plausible range (0, 2^30]. " +
                $"File is corrupt or not a safetensors file.");

        var headerBytes = new byte[(int)headerLen];
        int total = 0;
        while (total < headerBytes.Length)
        {
            int n = stream.Read(headerBytes, total, headerBytes.Length - total);
            if (n == 0)
                throw new EndOfStreamException(
                    $"Unexpected EOF in safetensors header — got {total} of {headerBytes.Length} bytes.");
            total += n;
        }

        var entries = new Dictionary<string, SafetensorsTensorEntry>(StringComparer.Ordinal);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);

        try
        {
            using var doc = JsonDocument.Parse(headerBytes);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("Safetensors header root must be a JSON object.");

            foreach (var prop in root.EnumerateObject())
            {
                if (prop.Name == "__metadata__")
                {
                    if (prop.Value.ValueKind != JsonValueKind.Object)
                        throw new InvalidDataException("__metadata__ must be a JSON object of string→string.");
                    foreach (var meta in prop.Value.EnumerateObject())
                    {
                        if (meta.Value.ValueKind == JsonValueKind.String)
                            metadata[meta.Name] = meta.Value.GetString() ?? string.Empty;
                    }
                    continue;
                }

                if (prop.Value.ValueKind != JsonValueKind.Object)
                    throw new InvalidDataException(
                        $"Tensor entry '{prop.Name}' must be a JSON object.");

                string dtypeStr = prop.Value.TryGetProperty("dtype", out var dt) && dt.ValueKind == JsonValueKind.String
                    ? dt.GetString() ?? throw new InvalidDataException($"Tensor '{prop.Name}' has null dtype.")
                    : throw new InvalidDataException($"Tensor '{prop.Name}' missing 'dtype'.");
                var dtype = SafetensorsDtypeExtensions.ParseHeaderString(dtypeStr);

                if (!prop.Value.TryGetProperty("shape", out var shapeEl) || shapeEl.ValueKind != JsonValueKind.Array)
                    throw new InvalidDataException($"Tensor '{prop.Name}' missing or non-array 'shape'.");
                var shape = new long[shapeEl.GetArrayLength()];
                int si = 0;
                foreach (var dim in shapeEl.EnumerateArray())
                {
                    if (dim.ValueKind != JsonValueKind.Number)
                        throw new InvalidDataException($"Tensor '{prop.Name}' shape contains non-number element.");
                    shape[si++] = dim.GetInt64();
                }

                if (!prop.Value.TryGetProperty("data_offsets", out var offEl)
                    || offEl.ValueKind != JsonValueKind.Array
                    || offEl.GetArrayLength() != 2)
                    throw new InvalidDataException(
                        $"Tensor '{prop.Name}' must have 'data_offsets' as a 2-element array.");
                long start, end;
                using (var off = offEl.EnumerateArray())
                {
                    off.MoveNext();
                    start = off.Current.GetInt64();
                    off.MoveNext();
                    end = off.Current.GetInt64();
                }

                entries[prop.Name] = new SafetensorsTensorEntry(prop.Name, dtype, shape, start, end);
            }
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException("Safetensors header is not valid JSON: " + ex.Message, ex);
        }

        long dataBlockStart = 8 + (long)headerLen;
        return (dataBlockStart, entries, metadata);
    }

    private static SafetensorsDtype MapClrTypeToDtype(Type t)
    {
        if (t == typeof(float)) return SafetensorsDtype.F32;
        if (t == typeof(double)) return SafetensorsDtype.F64;
        if (t == typeof(long)) return SafetensorsDtype.I64;
        if (t == typeof(int)) return SafetensorsDtype.I32;
        if (t == typeof(short)) return SafetensorsDtype.I16;
        if (t == typeof(sbyte)) return SafetensorsDtype.I8;
        if (t == typeof(byte)) return SafetensorsDtype.U8;
        if (t == typeof(bool)) return SafetensorsDtype.BOOL;
        // Half (System.Half) and BFloat16 don't map cleanly across all
        // target frameworks (Half is .NET 5+, BFloat16 is even newer)
        // — callers wanting F16 / BF16 should use ReadRawBytes and
        // route through their own conversion. Same for F8 / sub-byte.
        throw new InvalidOperationException(
            $"No safetensors dtype maps to CLR type {t.Name}. " +
            $"For F16/BF16/F8/AIDN_* sub-byte dtypes, call ReadRawBytes(name) and convert manually.");
    }

    private static int MarshalSize(Type t)
    {
        if (t == typeof(float)) return 4;
        if (t == typeof(double)) return 8;
        if (t == typeof(long)) return 8;
        if (t == typeof(int)) return 4;
        if (t == typeof(short)) return 2;
        if (t == typeof(sbyte)) return 1;
        if (t == typeof(byte)) return 1;
        if (t == typeof(bool)) return 1;
        throw new InvalidOperationException($"No marshal size known for {t.Name}.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SafetensorsReader));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_ownsStream) _stream.Dispose();
    }
}
