// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;

namespace AiDotNet.Tensors.Serialization.Safetensors;

/// <summary>
/// Reader for HuggingFace's sharded safetensors convention — a
/// directory containing <c>model.safetensors.index.json</c> and
/// multiple shard files (<c>model-00001-of-000N.safetensors</c>, …).
/// The index file is a JSON object with a <c>weight_map</c> sub-object
/// mapping each tensor name to the shard file that holds it.
/// </summary>
/// <remarks>
/// <para>This is the reader you reach for when loading larger HF
/// models (Llama-7B and up) that don't fit in one safetensors file.
/// HF caps individual shards at ~5 GB by default and writes the
/// index alongside.</para>
///
/// <para><b>Implementation:</b> opens every shard up front (lazy
/// per-shard initialisation could be added if a model has many shards
/// but the caller only needs a few; today HF models top out at ~30
/// shards even for 70B models, so eager open keeps the code simple).
/// Each <see cref="ReadTensor{T}"/> dispatches to the right shard via
/// the index's <c>weight_map</c>.</para>
/// </remarks>
public sealed class ShardedSafetensorsReader : IDisposable
{
    private readonly Dictionary<string, SafetensorsReader> _shards;
    private readonly Dictionary<string, string> _weightMap;     // tensor name → shard filename
    private readonly Dictionary<string, string> _metadata;
    private bool _disposed;

    /// <summary>The aggregated tensor entries from every shard's header.</summary>
    public IReadOnlyDictionary<string, SafetensorsTensorEntry> Entries { get; }

    /// <summary>
    /// File-level metadata from the index's <c>metadata</c> key, plus
    /// any <c>__metadata__</c> dicts merged from individual shards
    /// (shard metadata wins on key collision — last shard last).
    /// </summary>
    public IReadOnlyDictionary<string, string> Metadata => _metadata;

    /// <summary>Names of every tensor across every shard.</summary>
    public IEnumerable<string> Names => _weightMap.Keys;

    /// <summary>
    /// Opens a sharded safetensors directory by index-file path.
    /// </summary>
    /// <param name="indexPath">
    /// Path to <c>model.safetensors.index.json</c> (or whatever the
    /// index file is named — HF's writer also produces variants like
    /// <c>diffusion_pytorch_model.safetensors.index.json</c>).
    /// </param>
    public static ShardedSafetensorsReader Open(string indexPath)
    {
        PersistenceGuard.EnforceBeforeLoad();
        if (indexPath is null) throw new ArgumentNullException(nameof(indexPath));
        if (!File.Exists(indexPath))
            throw new FileNotFoundException($"Sharded safetensors index not found: {indexPath}");

        return new ShardedSafetensorsReader(indexPath);
    }

    private ShardedSafetensorsReader(string indexPath)
    {
        var indexDir = Path.GetDirectoryName(indexPath) ?? ".";
        var (weightMap, indexMetadata) = ParseIndex(indexPath);
        _weightMap = weightMap;
        _metadata = new Dictionary<string, string>(indexMetadata, StringComparer.Ordinal);

        // Open every distinct shard file — each open suppresses its
        // own EnforceBeforeLoad via InternalOperation since this
        // sharded reader's Open already counted one operation.
        var distinctShards = new HashSet<string>(weightMap.Values, StringComparer.Ordinal);
        _shards = new Dictionary<string, SafetensorsReader>(StringComparer.Ordinal);
        var aggregateEntries = new Dictionary<string, SafetensorsTensorEntry>(StringComparer.Ordinal);
        try
        {
            using (PersistenceGuard.InternalOperation())
            {
                foreach (var shardName in distinctShards)
                {
                    var shardPath = Path.Combine(indexDir, shardName);
                    var shard = SafetensorsReader.Open(shardPath);
                    _shards[shardName] = shard;
                    foreach (var entry in shard.Entries)
                    {
                        // The index promises uniqueness across shards;
                        // a duplicate is a corrupt-index symptom.
                        if (aggregateEntries.ContainsKey(entry.Key))
                            throw new InvalidDataException(
                                $"Tensor '{entry.Key}' appears in multiple shards — index is inconsistent.");
                        aggregateEntries[entry.Key] = entry.Value;
                    }
                    foreach (var meta in shard.Metadata)
                    {
                        // Shard-level metadata is rare but allowed; the
                        // index's metadata takes priority unless the
                        // shard provides a key the index didn't.
                        if (!_metadata.ContainsKey(meta.Key))
                            _metadata[meta.Key] = meta.Value;
                    }
                }
            }
        }
        catch
        {
            // On failure, dispose any already-opened shards.
            foreach (var s in _shards.Values) s.Dispose();
            _shards.Clear();
            throw;
        }
        Entries = aggregateEntries;
    }

    /// <summary>
    /// Reads <paramref name="name"/>'s tensor from the appropriate
    /// shard. See <see cref="SafetensorsReader.ReadTensor{T}"/> for
    /// dtype / shape contract.
    /// </summary>
    public Tensor<T> ReadTensor<T>(string name) where T : struct
    {
        ThrowIfDisposed();
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (!_weightMap.TryGetValue(name, out var shardName))
            throw new KeyNotFoundException(
                $"Tensor '{name}' not present in sharded weight map. Known shards: " +
                $"[{string.Join(", ", _weightMap.Values.Distinct())}].");

        var shard = _shards[shardName];
        // Suppress the per-shard EnforceBeforeLoad — the sharded
        // reader's Open already counted one operation, and reading
        // many tensors out of one sharded reader shouldn't drain the
        // user's trial budget per tensor.
        // Note: SafetensorsReader.ReadTensor doesn't itself call
        // EnforceBeforeLoad (it's only on Open) so the wrapping is
        // belt-and-braces — kept for forward-compat in case future
        // changes add per-read enforcement.
        using (PersistenceGuard.InternalOperation())
        {
            return shard.ReadTensor<T>(name);
        }
    }

    /// <summary>
    /// Returns the raw byte payload of <paramref name="name"/>'s
    /// tensor from the appropriate shard. See
    /// <see cref="SafetensorsReader.ReadRawBytes"/>.
    /// </summary>
    public byte[] ReadRawBytes(string name)
    {
        ThrowIfDisposed();
        if (name is null) throw new ArgumentNullException(nameof(name));
        if (!_weightMap.TryGetValue(name, out var shardName))
            throw new KeyNotFoundException($"Tensor '{name}' not present in sharded weight map.");
        using (PersistenceGuard.InternalOperation())
        {
            return _shards[shardName].ReadRawBytes(name);
        }
    }

    private static (Dictionary<string, string> WeightMap, Dictionary<string, string> Metadata) ParseIndex(string indexPath)
    {
        string json = File.ReadAllText(indexPath);
        var weightMap = new Dictionary<string, string>(StringComparer.Ordinal);
        var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("Sharded index root must be a JSON object.");

            if (!root.TryGetProperty("weight_map", out var wm) || wm.ValueKind != JsonValueKind.Object)
                throw new InvalidDataException("Sharded index missing required 'weight_map' object.");
            foreach (var prop in wm.EnumerateObject())
            {
                if (prop.Value.ValueKind != JsonValueKind.String)
                    throw new InvalidDataException(
                        $"Sharded index weight_map['{prop.Name}'] must be a string (shard filename).");
                weightMap[prop.Name] = prop.Value.GetString() ?? string.Empty;
            }

            if (root.TryGetProperty("metadata", out var meta) && meta.ValueKind == JsonValueKind.Object)
            {
                foreach (var prop in meta.EnumerateObject())
                {
                    if (prop.Value.ValueKind == JsonValueKind.String)
                        metadata[prop.Name] = prop.Value.GetString() ?? string.Empty;
                    // Numeric / nested values (e.g. total_size = int)
                    // are coerced to their JSON representation so the
                    // metadata dict stays string→string for round-trip
                    // simplicity.
                    else
                        metadata[prop.Name] = prop.Value.GetRawText();
                }
            }
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException("Sharded index is not valid JSON: " + ex.Message, ex);
        }
        return (weightMap, metadata);
    }

    private void ThrowIfDisposed()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(ShardedSafetensorsReader));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var s in _shards.Values) s.Dispose();
    }
}
