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
/// Writer for the safetensors format. Builds a header JSON with one
/// entry per added tensor, pads to 8-byte alignment, then emits a
/// contiguous data block with the tensor payloads in declaration
/// order.
/// </summary>
/// <remarks>
/// <para><b>Usage:</b></para>
/// <code>
/// using var w = SafetensorsWriter.Create("model.safetensors");
/// w.Add("embed.weight", embedTensor);
/// w.Add("ln.weight", lnTensor);
/// w.Metadata["framework"] = "AiDotNet.Tensors";
/// w.Save();
/// </code>
/// <para>The two-phase API (<c>Add</c> then <c>Save</c>) lets the
/// header-and-offset planning stay deterministic — every tensor's
/// byte range is fixed before any payload bytes are written, so the
/// reader's <c>data_offsets</c> always match what the writer
/// produced.</para>
/// </remarks>
public sealed class SafetensorsWriter : IDisposable
{
    private readonly Stream _stream;
    private readonly bool _ownsStream;
    private readonly List<PlannedTensor> _tensors = new();
    private readonly Dictionary<string, string> _metadata = new(StringComparer.Ordinal);
    private bool _saved;
    private bool _disposed;

    /// <summary>
    /// File-level metadata to emit under the reserved
    /// <c>__metadata__</c> JSON key. Mutate freely before
    /// <see cref="Save"/>.
    /// </summary>
    public IDictionary<string, string> Metadata => _metadata;

    /// <summary>
    /// Creates a writer that emits to <paramref name="path"/>. The
    /// file is opened with truncation — any previous content is
    /// overwritten on <see cref="Save"/>.
    /// </summary>
    /// <exception cref="LicenseRequiredException">
    /// Thrown when no license is configured AND the trial budget is
    /// exhausted, OR the configured license lacks
    /// <c>tensors:save</c>. See <see cref="PersistenceGuard"/>.
    /// </exception>
    public static SafetensorsWriter Create(string path)
    {
        PersistenceGuard.EnforceBeforeSave();
        if (path is null) throw new ArgumentNullException(nameof(path));
        var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        try
        {
            return new SafetensorsWriter(stream, ownsStream: true);
        }
        catch
        {
            stream.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Wraps an existing writable, seekable stream. The writer does
    /// NOT take ownership — callers must dispose the stream
    /// themselves.
    /// </summary>
    /// <exception cref="LicenseRequiredException">See <see cref="Create"/>.</exception>
    public static SafetensorsWriter ToStream(Stream stream)
    {
        PersistenceGuard.EnforceBeforeSave();
        return new SafetensorsWriter(stream, ownsStream: false);
    }

    private SafetensorsWriter(Stream stream, bool ownsStream)
    {
        if (stream is null) throw new ArgumentNullException(nameof(stream));
        if (!stream.CanWrite) throw new ArgumentException("Stream must be writable.", nameof(stream));
        if (!stream.CanSeek) throw new ArgumentException("Stream must be seekable.", nameof(stream));
        _stream = stream;
        _ownsStream = ownsStream;
    }

    /// <summary>
    /// Records <paramref name="tensor"/> under the given name. Order
    /// matters — the writer emits payloads in the order ops were
    /// added so debugging a corrupted file by hex-dumping always
    /// follows the header order. Throws if a name was already added.
    /// </summary>
    public void Add<T>(string name, Tensor<T> tensor) where T : struct
    {
        ThrowIfDisposed();
        if (_saved)
            throw new InvalidOperationException(
                "Cannot Add after Save — open a new writer if you need to extend.");
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Tensor name cannot be empty.", nameof(name));
        if (name == "__metadata__")
            throw new ArgumentException(
                "'__metadata__' is reserved for the metadata dict — use Metadata[\"...\"] = \"...\" instead.",
                nameof(name));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        for (int i = 0; i < _tensors.Count; i++)
            if (_tensors[i].Name == name)
                throw new ArgumentException($"Tensor name '{name}' already added.", nameof(name));

        var dtype = MapClrTypeToDtype(typeof(T));
        var span = tensor.AsSpan();
        var bytes = MemoryMarshal.AsBytes(span).ToArray();

        var shape = new long[tensor._shape.Length];
        for (int i = 0; i < shape.Length; i++) shape[i] = tensor._shape[i];

        _tensors.Add(new PlannedTensor(name, dtype, shape, bytes));
    }

    /// <summary>
    /// Adds a raw byte payload under <paramref name="name"/> with the
    /// supplied dtype and shape. Useful for sub-byte / FP8 dtypes
    /// the typed <see cref="Add{T}"/> overload doesn't cover.
    /// </summary>
    public void AddRaw(string name, SafetensorsDtype dtype, long[] shape, byte[] payload)
    {
        ThrowIfDisposed();
        if (_saved)
            throw new InvalidOperationException("Cannot Add after Save.");
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (name == "__metadata__")
            throw new ArgumentException(
                "'__metadata__' is reserved for the metadata dict.", nameof(name));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (payload is null) throw new ArgumentNullException(nameof(payload));
        for (int i = 0; i < _tensors.Count; i++)
            if (_tensors[i].Name == name)
                throw new ArgumentException($"Tensor name '{name}' already added.", nameof(name));

        // Validate shape entries and the payload length up front so a
        // malformed AddRaw produces a clear error AT THE CALL SITE
        // rather than corrupting the safetensors header that a future
        // reader trips over. Reject negative dims, overflow on the
        // element-count product, and any payload whose byte count
        // doesn't match the dtype's natural per-element size.
        long elemCount = 1;
        for (int i = 0; i < shape.Length; i++)
        {
            if (shape[i] < 0)
                throw new ArgumentException(
                    $"Shape dimension {i} = {shape[i]} is negative.", nameof(shape));
            try { elemCount = checked(elemCount * shape[i]); }
            catch (OverflowException ex)
            {
                throw new ArgumentException(
                    $"Shape product overflows long for tensor '{name}'.", nameof(shape), ex);
            }
        }
        long expectedBytes;
        try { expectedBytes = checked(elemCount * dtype.ElementByteSize()); }
        catch (OverflowException ex)
        {
            throw new ArgumentException(
                $"Element count × element size overflows long for tensor '{name}'.", nameof(shape), ex);
        }
        // Sub-byte dtypes pack multiple elements per byte; the header
        // declares element-byte-size = 1 for them, so the payload's
        // byte count is the packed-byte count which is shape-product
        // divided by lanes-per-byte. We can only assert the typed-
        // dtype invariant — sub-byte payloads are caller-responsibility.
        if (!IsSubBytePackedDtype(dtype) && payload.Length != expectedBytes)
            throw new ArgumentException(
                $"Payload length {payload.Length} does not match expected " +
                $"element count {elemCount} × element size {dtype.ElementByteSize()} = {expectedBytes} " +
                $"for tensor '{name}' dtype {dtype}.", nameof(payload));

        _tensors.Add(new PlannedTensor(name, dtype, (long[])shape.Clone(), (byte[])payload.Clone()));
    }

    private static bool IsSubBytePackedDtype(SafetensorsDtype dtype) => dtype switch
    {
        SafetensorsDtype.AIDN_NF4 => true,
        SafetensorsDtype.AIDN_FP4 => true,
        SafetensorsDtype.AIDN_INT4 => true,
        SafetensorsDtype.AIDN_INT3 => true,
        SafetensorsDtype.AIDN_INT2 => true,
        SafetensorsDtype.AIDN_INT1 => true,
        _ => false,
    };

    /// <summary>
    /// Finalises the file: builds the header JSON, computes byte
    /// offsets, writes the 8-byte length + header + payloads to the
    /// underlying stream. After <see cref="Save"/>, no more tensors
    /// can be added.
    /// </summary>
    public void Save()
    {
        ThrowIfDisposed();
        if (_saved)
            throw new InvalidOperationException("Save already called.");

        // Reset to byte 0 + truncate so a wrapped stream that already
        // had content (a reused MemoryStream / FileStream) doesn't
        // prepend garbage to our header or leave stale trailing bytes
        // beyond our last byte. Create()'s FileMode.Create already
        // handles new files; this protects ToStream() callers.
        _stream.Seek(0, SeekOrigin.Begin);
        _stream.SetLength(0);

        // Compute offsets first so the header can include them.
        long cursor = 0;
        foreach (var t in _tensors)
        {
            t.DataStart = cursor;
            cursor += t.Bytes.Length;
            t.DataEnd = cursor;
        }
        long totalDataLen = cursor;

        // Build the header JSON. We emit tensor entries in
        // insertion order — the reader iterates JSON properties in
        // their textual order so this gives a deterministic round
        // trip.
        using var ms = new MemoryStream();
        using (var jw = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false }))
        {
            jw.WriteStartObject();
            // __metadata__ first so a hex-dump shows it before the
            // tensor entries — matches HuggingFace's writer convention.
            if (_metadata.Count > 0)
            {
                jw.WriteStartObject("__metadata__");
                foreach (var kv in _metadata) jw.WriteString(kv.Key, kv.Value);
                jw.WriteEndObject();
            }
            foreach (var t in _tensors)
            {
                jw.WriteStartObject(t.Name);
                jw.WriteString("dtype", t.Dtype.ToHeaderString());
                jw.WriteStartArray("shape");
                foreach (var d in t.Shape) jw.WriteNumberValue(d);
                jw.WriteEndArray();
                jw.WriteStartArray("data_offsets");
                jw.WriteNumberValue(t.DataStart);
                jw.WriteNumberValue(t.DataEnd);
                jw.WriteEndArray();
                jw.WriteEndObject();
            }
            jw.WriteEndObject();
        }
        var headerJson = ms.ToArray();

        // Pad header up to 8-byte alignment so the data block starts
        // on an aligned boundary — improves mmap / SIMD load perf.
        int padTo8 = (int)(8 - (headerJson.Length % 8)) % 8;
        if (padTo8 > 0)
        {
            // Append spaces inside the JSON's trailing whitespace —
            // valid JSON since whitespace is allowed anywhere outside
            // a string. We pad just before the closing `}` would have
            // been; but since we already serialised, append spaces
            // after the final `}` which is also legal (the parser
            // stops at the last `}`).
            // Simplest: store a longer buffer and copy, ending with N spaces.
            var padded = new byte[headerJson.Length + padTo8];
            Buffer.BlockCopy(headerJson, 0, padded, 0, headerJson.Length);
            for (int i = 0; i < padTo8; i++) padded[headerJson.Length + i] = (byte)' ';
            headerJson = padded;
        }

        // Header length prefix (8-byte LE u64).
        var lenPrefix = new byte[8];
        System.Buffers.Binary.BinaryPrimitives.WriteUInt64LittleEndian(lenPrefix, (ulong)headerJson.Length);
        _stream.Write(lenPrefix, 0, lenPrefix.Length);

        // Header.
        _stream.Write(headerJson, 0, headerJson.Length);

        // Data block in order.
        foreach (var t in _tensors)
            _stream.Write(t.Bytes, 0, t.Bytes.Length);

        _stream.Flush();
        _saved = true;
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
        throw new InvalidOperationException(
            $"No safetensors dtype maps to CLR type {t.Name}. " +
            $"For F16/BF16/F8/AIDN_* sub-byte dtypes, use AddRaw with the explicit dtype.");
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(SafetensorsWriter));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        // Auto-save BEFORE flipping _disposed: Save() calls
        // ThrowIfDisposed() at its top, so the previous order
        // (`_disposed = true` first) made Save() immediately throw
        // ObjectDisposedException — the catch swallowed it and the
        // file was never written. Run Save() while the writer is
        // still considered "alive" so the planning + offset logic
        // actually executes.
        if (!_saved)
        {
            try { Save(); } catch { /* swallow on disposal — caller already dropped */ }
        }
        _disposed = true;
        if (_ownsStream) _stream.Dispose();
    }

    private sealed class PlannedTensor
    {
        public string Name { get; }
        public SafetensorsDtype Dtype { get; }
        public long[] Shape { get; }
        public byte[] Bytes { get; }
        public long DataStart { get; set; }
        public long DataEnd { get; set; }
        public PlannedTensor(string name, SafetensorsDtype dtype, long[] shape, byte[] bytes)
        {
            Name = name;
            Dtype = dtype;
            Shape = shape;
            Bytes = bytes;
        }
    }
}
