// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Licensing;
using AiDotNet.Tensors.Serialization.Safetensors;

namespace AiDotNet.Tensors.Serialization.Distributed;

/// <summary>
/// Distributed-checkpoint format machinery layered on top of
/// <see cref="ShardedSafetensorsWriter"/> and
/// <see cref="ShardedSafetensorsReader"/>. Provides the three
/// operations FSDP / ZeRO checkpoints care about:
/// <list type="bullet">
///   <item><see cref="Save"/> — write a state-dict as N shards.</item>
///   <item><see cref="Load"/> — read every tensor from a sharded
///   directory back into a flat dict.</item>
///   <item><see cref="Reshard"/> — open an N-shard checkpoint and
///   write it as M shards (any M ≥ 1).</item>
/// </list>
/// </summary>
/// <remarks>
/// <para><b>Why this lives in the tensor layer:</b></para>
/// <para>
/// FSDP / ZeRO require a process-group abstraction (issue #215) for
/// the cross-rank shard-discovery dance. The format machinery —
/// "split a state-dict into N shards on disk; consolidate back; reshard
/// to a different count" — is independent of how shards travel between
/// ranks. We ship the format ops here so callers without a real
/// distributed runtime (single-process eval, off-line resharding,
/// CI checkpoint validation) get a working tool, and the upstream
/// <c>AiDotNet</c> package's FSDP integration plugs in by collecting
/// per-rank shards into a single <see cref="DistributedCheckpoint.Save"/>
/// call.
/// </para>
/// <para><b>Resharding:</b> the implementation walks the input
/// tensor-by-tensor and re-bin-packs into the new shard count.
/// Tensors are not split across shards — the granularity is the
/// individual tensor, matching FSDP's <c>flat_param</c> view where
/// each shard owns a complete tensor. For models where one tensor
/// is itself bigger than the requested shard size (e.g. a 30B-vocab
/// embedding under a 1 GB shard budget), the affected tensor lives in
/// its own shard regardless of the budget.
/// </para>
/// </remarks>
public static class DistributedCheckpoint
{
    /// <summary>
    /// Saves a state-dict to <paramref name="outputDir"/> as
    /// <paramref name="numShards"/> safetensors shards. The shard
    /// count is a soft target — if any single tensor exceeds
    /// <c>totalBytes / numShards</c>, the actual shard count may be
    /// higher.
    /// </summary>
    /// <returns>Number of shards actually emitted.</returns>
    public static int Save(string outputDir, IReadOnlyDictionary<string, IPersistableTensor> stateDict, int numShards)
    {
        if (outputDir is null) throw new ArgumentNullException(nameof(outputDir));
        if (stateDict is null) throw new ArgumentNullException(nameof(stateDict));
        if (numShards <= 0) throw new ArgumentOutOfRangeException(nameof(numShards));

        // Compute total byte count to target a shard size of
        // ceil(total / numShards). The previous formula
        // (total / numShards + 1) overshoots by 1 when total is a
        // multiple of numShards, producing larger-than-requested
        // shards on evenly divisible inputs. The (total + n - 1) / n
        // form is the standard ceil-div, with a 1-byte floor for
        // total == 0 to avoid feeding ShardedSafetensorsWriter a
        // zero shard size.
        long total = 0;
        foreach (var kv in stateDict) total += kv.Value.ByteCount;
        long shardSize = total <= 0 ? 1 : (total + numShards - 1) / numShards;

        var w = new ShardedSafetensorsWriter(outputDir, "model", shardSize);
        foreach (var kv in stateDict)
            kv.Value.AddTo(w, kv.Key);
        return w.Save();
    }

    /// <summary>
    /// Loads every tensor from a sharded checkpoint at
    /// <paramref name="indexPath"/> into a flat dict keyed by tensor
    /// name. Counts as one <see cref="PersistenceGuard.EnforceBeforeLoad"/>
    /// (the inner sharded reader's per-shard opens are suppressed).
    /// </summary>
    public static Dictionary<string, byte[]> Load(string indexPath)
    {
        // EnforceBeforeLoad is called by ShardedSafetensorsReader.Open.
        if (indexPath is null) throw new ArgumentNullException(nameof(indexPath));
        using var r = ShardedSafetensorsReader.Open(indexPath);
        var dict = new Dictionary<string, byte[]>(StringComparer.Ordinal);
        foreach (var name in r.Names) dict[name] = r.ReadRawBytes(name);
        return dict;
    }

    /// <summary>
    /// Opens an existing sharded checkpoint at
    /// <paramref name="srcIndexPath"/> and rewrites it to
    /// <paramref name="dstDir"/> with <paramref name="newShardCount"/>
    /// shards. All tensors from the source land in some shard of the
    /// destination — the on-disk layout reorganises but the
    /// state-dict contents are bit-identical.
    /// </summary>
    /// <returns>Number of shards actually written (≥ 1).</returns>
    public static int Reshard(string srcIndexPath, string dstDir, int newShardCount)
    {
        if (srcIndexPath is null) throw new ArgumentNullException(nameof(srcIndexPath));
        if (dstDir is null) throw new ArgumentNullException(nameof(dstDir));
        if (newShardCount <= 0) throw new ArgumentOutOfRangeException(nameof(newShardCount));

        // The sharded reader counts one EnforceBeforeLoad; the
        // sharded writer's Save counts one EnforceBeforeSave. From
        // the user's POV one Reshard call = two trial-counter ticks.
        using var src = ShardedSafetensorsReader.Open(srcIndexPath);

        long total = 0;
        foreach (var kv in src.Entries) total += kv.Value.ByteLength;
        // ceil(total / newShardCount), 1-byte floor for total == 0.
        long shardSize = total <= 0 ? 1 : (total + newShardCount - 1) / newShardCount;

        var w = new ShardedSafetensorsWriter(dstDir, "model", shardSize);
        foreach (var kv in src.Entries)
        {
            var entry = kv.Value;
            var bytes = src.ReadRawBytes(kv.Key);
            w.AddRaw(kv.Key, entry.Dtype, entry.Shape, bytes);
        }
        // Carry over user metadata (omit the auto-injected
        // total_size — the writer recomputes it for the new layout).
        foreach (var kv in src.Metadata)
            if (kv.Key != "total_size") w.Metadata[kv.Key] = kv.Value;
        return w.Save();
    }
}

/// <summary>
/// Type-erased view of a tensor that can be persisted to a
/// safetensors writer. <see cref="DistributedCheckpoint.Save"/> takes
/// a dict keyed by name to one of these — the typed wrapper
/// <see cref="PersistableTensor{T}"/> covers all the standard CLR
/// element types; raw byte payloads use
/// <see cref="RawPersistableTensor"/>.
/// </summary>
public interface IPersistableTensor
{
    /// <summary>Bytes of the payload — used for shard-size planning.</summary>
    long ByteCount { get; }

    /// <summary>Calls <see cref="ShardedSafetensorsWriter.Add{T}"/> or AddRaw with the matching type info.</summary>
    void AddTo(ShardedSafetensorsWriter writer, string name);
}

/// <summary>Typed persistable wrapper for any <see cref="Tensor{T}"/>.</summary>
public sealed class PersistableTensor<T> : IPersistableTensor where T : struct
{
    private readonly Tensor<T> _t;

    /// <summary>Wraps <paramref name="tensor"/> for persistence.</summary>
    public PersistableTensor(Tensor<T> tensor)
    {
        _t = tensor ?? throw new ArgumentNullException(nameof(tensor));
    }

    /// <inheritdoc />
    // Widen Length to long BEFORE the multiply — Tensor<T>.Length is
    // int and MarshalSize() returns int, so int × int overflows for
    // any tensor of more than ~2 GB / sizeof(T). The previous form
    // wrapped that overflow into a negative long, badly skewing the
    // sharded writer's bin-packing.
    public long ByteCount => (long)_t.Length * MarshalSize();

    /// <inheritdoc />
    public void AddTo(ShardedSafetensorsWriter writer, string name) => writer.Add(name, _t);

    private int MarshalSize()
    {
        var t = typeof(T);
        if (t == typeof(float)) return 4;
        if (t == typeof(double)) return 8;
        if (t == typeof(long)) return 8;
        if (t == typeof(int)) return 4;
        if (t == typeof(short)) return 2;
        if (t == typeof(sbyte)) return 1;
        if (t == typeof(byte)) return 1;
        if (t == typeof(bool)) return 1;
        return Marshal.SizeOf<T>();
    }
}

/// <summary>Raw-bytes persistable wrapper — for sub-byte / FP8 payloads.</summary>
public sealed class RawPersistableTensor : IPersistableTensor
{
    private readonly SafetensorsDtype _dtype;
    private readonly long[] _shape;
    private readonly byte[] _bytes;

    /// <summary>Wraps a raw byte payload with the supplied dtype + shape.</summary>
    public RawPersistableTensor(SafetensorsDtype dtype, long[] shape, byte[] bytes)
    {
        _dtype = dtype;
        _shape = shape ?? throw new ArgumentNullException(nameof(shape));
        _bytes = bytes ?? throw new ArgumentNullException(nameof(bytes));
    }

    /// <inheritdoc />
    public long ByteCount => _bytes.Length;

    /// <inheritdoc />
    public void AddTo(ShardedSafetensorsWriter writer, string name)
        => writer.AddRaw(name, _dtype, _shape, _bytes);
}
