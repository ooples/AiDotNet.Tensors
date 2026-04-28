// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text.Json;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;

namespace AiDotNet.Tensors.Serialization.Safetensors;

/// <summary>
/// Writes a checkpoint as multiple safetensors shards plus a
/// <c>model.safetensors.index.json</c> file, matching the HuggingFace
/// convention used for any model whose total weights exceed a single
/// shard's byte budget.
/// </summary>
/// <remarks>
/// <para><b>Why automatic sharding:</b></para>
/// <para>
/// Llama-7B in F16 is 13 GB; in F32 it's 26 GB. A single safetensors
/// file at that size is awkward — slow to upload to HF Hub, blocked
/// by some filesystems' max-file-size limits, and harder for a
/// resumable downloader. The HF tooling caps individual shards at
/// ~5 GB by default and emits an index file pointing at each shard.
/// This writer follows that convention so the produced directory
/// loads via <see cref="ShardedSafetensorsReader.Open"/> AND through
/// any HuggingFace tool that reads sharded checkpoints.
/// </para>
/// <para><b>Greedy bin-packing:</b></para>
/// <para>
/// Tensors are routed into the current shard until the next tensor
/// would exceed the byte budget; then a new shard is opened. Single
/// tensors larger than the budget are placed in their own shard
/// (rather than refused) so a small budget doesn't break a model
/// with one large embedding. The resulting shard count is within 1
/// of the optimal bin-packing for any practical input.
/// </para>
/// </remarks>
public sealed class ShardedSafetensorsWriter
{
    /// <summary>Default per-shard byte budget — matches HF's 5 GB default.</summary>
    public const long DefaultShardSizeBytes = 5L * 1024 * 1024 * 1024;

    private readonly string _outputDir;
    private readonly string _filenamePrefix;
    private readonly long _shardSizeBytes;
    private readonly List<PendingTensor> _pending = new();
    private readonly Dictionary<string, string> _metadata = new(StringComparer.Ordinal);
    private bool _saved;

    /// <summary>
    /// File-level metadata. Written into the index's <c>metadata</c>
    /// object; not duplicated into individual shards.
    /// </summary>
    public IDictionary<string, string> Metadata => _metadata;

    /// <summary>
    /// Creates a sharded writer that emits to
    /// <paramref name="outputDir"/>. The directory is created if it
    /// doesn't exist.
    /// </summary>
    /// <param name="outputDir">Directory to write shard files + index into.</param>
    /// <param name="filenamePrefix">Stem used for shard files
    /// (default <c>model</c> → <c>model-00001-of-000N.safetensors</c>
    /// + <c>model.safetensors.index.json</c>). Match the HF tooling
    /// pattern by leaving this as the default.</param>
    /// <param name="shardSizeBytes">Target per-shard byte budget.
    /// Default <see cref="DefaultShardSizeBytes"/>. Tests use a small
    /// value to force multi-shard output deterministically.</param>
    public ShardedSafetensorsWriter(
        string outputDir,
        string filenamePrefix = "model",
        long shardSizeBytes = DefaultShardSizeBytes)
    {
        if (outputDir is null) throw new ArgumentNullException(nameof(outputDir));
        if (filenamePrefix is null) throw new ArgumentNullException(nameof(filenamePrefix));
        if (string.IsNullOrEmpty(filenamePrefix))
            throw new ArgumentException("Filename prefix cannot be empty.", nameof(filenamePrefix));
        // Reject any directory-separator or invalid filename character
        // — without this, a prefix like "../escape" or "subdir/file"
        // would be interpolated into the shard filename and cause
        // Path.Combine(outputDir, ...) to escape outputDir entirely.
        // Path.GetInvalidFileNameChars covers OS-specific cases ('/'
        // on POSIX, '\' on Windows, plus control characters etc.)
        char[] invalid = Path.GetInvalidFileNameChars();
        for (int i = 0; i < filenamePrefix.Length; i++)
        {
            char c = filenamePrefix[i];
            if (Array.IndexOf(invalid, c) >= 0
                || c == Path.DirectorySeparatorChar
                || c == Path.AltDirectorySeparatorChar)
                throw new ArgumentException(
                    $"Filename prefix '{filenamePrefix}' contains invalid character '{c}' " +
                    $"(directory separators / control chars / OS-reserved chars are rejected to " +
                    $"prevent path-traversal in the emitted shard filenames).",
                    nameof(filenamePrefix));
        }
        if (filenamePrefix == "." || filenamePrefix == "..")
            throw new ArgumentException(
                $"Filename prefix '{filenamePrefix}' is a path-relative segment.",
                nameof(filenamePrefix));
        if (shardSizeBytes <= 0)
            throw new ArgumentOutOfRangeException(nameof(shardSizeBytes), "Shard size must be > 0 bytes.");
        _outputDir = outputDir;
        _filenamePrefix = filenamePrefix;
        _shardSizeBytes = shardSizeBytes;
    }

    /// <summary>Adds a tensor under <paramref name="name"/>.</summary>
    /// <remarks>
    /// Tensors are buffered (their byte payloads materialised into
    /// the planning structure) and only routed to shards on
    /// <see cref="Save"/>. Order matters — the bin-packer is greedy
    /// and walks the buffer in declaration order so shard contents
    /// are deterministic for a given input order.
    /// </remarks>
    public void Add<T>(string name, Tensor<T> tensor) where T : struct
    {
        if (_saved) throw new InvalidOperationException("Cannot Add after Save.");
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        // Mirror SafetensorsWriter's name validation so a malformed
        // name is rejected up front rather than buffering OK and
        // throwing inside Save() after some shard files were already
        // written. SafetensorsWriter (which the per-shard emit goes
        // through) refuses both empty names and the reserved
        // "__metadata__" key, so the same checks have to fire here.
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Tensor name cannot be empty.", nameof(name));
        if (name == "__metadata__")
            throw new ArgumentException(
                "'__metadata__' is reserved for the metadata dict — use Metadata[\"...\"] = \"...\" instead.",
                nameof(name));
        for (int i = 0; i < _pending.Count; i++)
            if (_pending[i].Name == name)
                throw new ArgumentException($"Tensor name '{name}' already added.", nameof(name));

        // Materialise the byte payload now so the bin-packer has a
        // concrete size to dispatch against.
        var bytes = MemoryMarshal.AsBytes(tensor.AsSpan()).ToArray();
        var dtype = MapClrTypeToDtype(typeof(T));
        var shape = new long[tensor._shape.Length];
        for (int i = 0; i < shape.Length; i++) shape[i] = tensor._shape[i];

        _pending.Add(new PendingTensor(name, dtype, shape, bytes));
    }

    /// <summary>
    /// Adds a raw byte payload — for sub-byte / FP8 dtypes where
    /// the typed <see cref="Add{T}"/> overload doesn't apply.
    /// </summary>
    public void AddRaw(string name, SafetensorsDtype dtype, long[] shape, byte[] payload)
    {
        if (_saved) throw new InvalidOperationException("Cannot Add after Save.");
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (payload is null) throw new ArgumentNullException(nameof(payload));
        // Same name-validation contract as Add<T> — see comment there.
        if (string.IsNullOrEmpty(name))
            throw new ArgumentException("Tensor name cannot be empty.", nameof(name));
        if (name == "__metadata__")
            throw new ArgumentException(
                "'__metadata__' is reserved for the metadata dict — use Metadata[\"...\"] = \"...\" instead.",
                nameof(name));
        for (int i = 0; i < _pending.Count; i++)
            if (_pending[i].Name == name)
                throw new ArgumentException($"Tensor name '{name}' already added.", nameof(name));

        // Validate shape entries and the payload length up front so
        // a malformed AddRaw produces a clear error AT THE CALL SITE
        // rather than corrupting a downstream shard's header. Sub-byte
        // packed dtypes are exempt from the strict byte-count check
        // because the on-disk byte count is shape-product / lanes-per-byte
        // rather than shape-product × sizeof(dtype).
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
        bool isSubByte = dtype == SafetensorsDtype.AIDN_NF4
            || dtype == SafetensorsDtype.AIDN_FP4
            || dtype == SafetensorsDtype.AIDN_INT4
            || dtype == SafetensorsDtype.AIDN_INT3
            || dtype == SafetensorsDtype.AIDN_INT2
            || dtype == SafetensorsDtype.AIDN_INT1;
        if (!isSubByte)
        {
            long expectedBytes;
            try { expectedBytes = checked(elemCount * dtype.ElementByteSize()); }
            catch (OverflowException ex)
            {
                throw new ArgumentException(
                    $"Element count × element size overflows long for tensor '{name}'.", nameof(shape), ex);
            }
            if (payload.Length != expectedBytes)
                throw new ArgumentException(
                    $"Payload length {payload.Length} does not match expected " +
                    $"element count {elemCount} × element size {dtype.ElementByteSize()} = {expectedBytes} " +
                    $"for tensor '{name}' dtype {dtype}.", nameof(payload));
        }

        _pending.Add(new PendingTensor(name, dtype, (long[])shape.Clone(), (byte[])payload.Clone()));
    }

    /// <summary>
    /// Bin-packs the buffered tensors into shards, writes each
    /// shard via <see cref="SafetensorsWriter"/>, and emits the
    /// index file. Counts as ONE EnforceBeforeSave operation against
    /// the licence — the per-shard writes run inside an
    /// <see cref="PersistenceGuard.InternalOperation"/> scope.
    /// </summary>
    /// <returns>The number of shards produced.</returns>
    public int Save()
    {
        if (_saved) throw new InvalidOperationException("Save already called.");

        PersistenceGuard.EnforceBeforeSave();

        if (!Directory.Exists(_outputDir)) Directory.CreateDirectory(_outputDir);

        // Greedy bin-packing — see class docs for tradeoff.
        var shards = new List<List<PendingTensor>>();
        var current = new List<PendingTensor>();
        long currentBytes = 0;
        foreach (var t in _pending)
        {
            if (current.Count > 0 && currentBytes + t.Bytes.Length > _shardSizeBytes)
            {
                shards.Add(current);
                current = new List<PendingTensor>();
                currentBytes = 0;
            }
            current.Add(t);
            currentBytes += t.Bytes.Length;
        }
        if (current.Count > 0) shards.Add(current);
        if (shards.Count == 0)
        {
            // No tensors at all — emit just the index pointing at an
            // empty single shard. Mirrors HF's behaviour for a
            // metadata-only checkpoint.
            shards.Add(new List<PendingTensor>());
        }

        var weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
        long totalSize = 0;
        // Per-shard write happens under an InternalOperation scope so
        // each shard's SafetensorsWriter.Create call doesn't tick the
        // user's trial counter — Save() above already counted one
        // operation for the whole sharded checkpoint.
        using (PersistenceGuard.InternalOperation())
        {
            for (int s = 0; s < shards.Count; s++)
            {
                string shardName = $"{_filenamePrefix}-{s + 1:D5}-of-{shards.Count:D5}.safetensors";
                string shardPath = Path.Combine(_outputDir, shardName);
                using var w = SafetensorsWriter.Create(shardPath);
                foreach (var t in shards[s])
                {
                    w.AddRaw(t.Name, t.Dtype, t.Shape, t.Bytes);
                    weightMap[t.Name] = shardName;
                    totalSize += t.Bytes.Length;
                }
                w.Save();
            }
        }

        // Emit index file. Schema follows HF:
        //   { "metadata": { "total_size": N, ...userMetadata }, "weight_map": { name: shardFile } }
        string indexPath = Path.Combine(_outputDir, $"{_filenamePrefix}.safetensors.index.json");
        using (var fs = new FileStream(indexPath, FileMode.Create, FileAccess.Write, FileShare.None))
        using (var jw = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true }))
        {
            jw.WriteStartObject();
            jw.WriteStartObject("metadata");
            jw.WriteNumber("total_size", totalSize);
            foreach (var kv in _metadata) jw.WriteString(kv.Key, kv.Value);
            jw.WriteEndObject();
            jw.WriteStartObject("weight_map");
            foreach (var kv in weightMap) jw.WriteString(kv.Key, kv.Value);
            jw.WriteEndObject();
            jw.WriteEndObject();
        }

        _saved = true;
        return shards.Count;
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
            $"No safetensors dtype maps to CLR type {t.Name}. Use AddRaw for sub-byte / FP8 dtypes.");
    }

    private sealed class PendingTensor
    {
        public string Name { get; }
        public SafetensorsDtype Dtype { get; }
        public long[] Shape { get; }
        public byte[] Bytes { get; }
        public PendingTensor(string name, SafetensorsDtype dtype, long[] shape, byte[] bytes)
        {
            Name = name;
            Dtype = dtype;
            Shape = shape;
            Bytes = bytes;
        }
    }
}
