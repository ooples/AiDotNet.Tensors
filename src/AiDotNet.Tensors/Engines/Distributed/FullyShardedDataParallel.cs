// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Sharding strategy for <see cref="FullyShardedDataParallel{T}"/>.
/// Mirrors <c>torch.distributed.fsdp.ShardingStrategy</c>.
/// </summary>
public enum ShardingStrategy
{
    /// <summary>No sharding — equivalent to DDP. Each rank holds the
    /// full parameter + full grad + full optimizer state.</summary>
    NoShard,

    /// <summary>Shard parameters + grads + optimizer state across the
    /// group. Equivalent to ZeRO-3. Maximum memory savings; highest
    /// communication cost.</summary>
    FullShard,

    /// <summary>Shard params + grads, replicate optimizer state per
    /// node. Equivalent to ZeRO-2 with hybrid placement.</summary>
    HybridShard,

    /// <summary>Shard only the optimizer state. Equivalent to ZeRO-1.
    /// Smallest memory savings; lowest communication cost.</summary>
    ShardOptimizerOnly,
}

/// <summary>
/// Configuration for <see cref="FullyShardedDataParallel{T}"/>.
/// </summary>
public sealed class FsdpOptions
{
    /// <summary>Sharding strategy. Default <see cref="ShardingStrategy.FullShard"/>.</summary>
    public ShardingStrategy Strategy { get; set; } = ShardingStrategy.FullShard;

    /// <summary>If true, the all-gather for the next layer's parameters
    /// is started before the current layer's forward finishes. PyTorch
    /// FSDP's <c>forward_prefetch</c>.</summary>
    public bool ForwardPrefetch { get; set; } = true;

    /// <summary>If true, the all-gather for the previous layer's
    /// parameters (needed for backward) is started before the current
    /// layer's backward begins.</summary>
    public bool BackwardPrefetch { get; set; } = true;

    /// <summary>Mixed-precision: keep params + grads in fp16 / bf16,
    /// cast to fp32 for the reduce. <c>null</c> = full precision.</summary>
    public string? ParamDtype { get; set; }

    /// <summary>If true, parameters are offloaded to host memory between
    /// uses (and re-uploaded on demand). Major memory savings at the
    /// cost of host-device transfer time.</summary>
    public bool CpuOffload { get; set; }
}

/// <summary>
/// FSDP wrapper. Holds a parameter set sharded across the process group
/// per <see cref="FsdpOptions.Strategy"/>; on forward/backward, the user
/// requests a full parameter via <see cref="GatherParameter"/> (which
/// all-gathers the shards into a freshly-allocated tensor) and releases
/// it via <see cref="ReleaseParameter"/> when no longer needed.
/// Gradients are reduce-scattered back to per-rank shards via
/// <see cref="ReduceScatterGradients"/>.
///
/// <para>This class is the framework-side surface; layer integration
/// (auto-gather on forward, auto-reduce-scatter at backward end) is the
/// caller's responsibility — same as PyTorch's manual-FSDP path. The
/// <c>fully_shard</c>-style auto-instrumentation is a follow-up that
/// composes this class with the autograd hook surface.</para>
/// </summary>
public sealed class FullyShardedDataParallel<T>
{
    private readonly IProcessGroup _group;
    private readonly FsdpOptions _options;
    private readonly List<ShardedParameter<T>> _shards = new();

    /// <summary>Constructs.</summary>
    public FullyShardedDataParallel(IProcessGroup group, FsdpOptions? options = null)
    {
        _group = group ?? throw new ArgumentNullException(nameof(group));
        _options = options ?? new FsdpOptions();
    }

    /// <summary>The process group params are sharded across.</summary>
    public IProcessGroup Group => _group;

    /// <summary>Configuration.</summary>
    public FsdpOptions Options => _options;

    /// <summary>Number of registered sharded parameters.</summary>
    public int ParameterCount => _shards.Count;

    /// <summary>
    /// Registers a full-size parameter and shards it across the group's
    /// ranks. <paramref name="fullParam"/> must be the same on every
    /// rank (typically loaded from a single checkpoint then broadcast);
    /// the registered shard is the slice this rank owns.
    /// </summary>
    /// <returns>Shard handle this rank holds.</returns>
    public ShardedParameter<T> RegisterParameter(Tensor<T> fullParam)
    {
        if (fullParam is null) throw new ArgumentNullException(nameof(fullParam));
        var shard = ShardedParameter<T>.Create(fullParam, _group, _options);
        _shards.Add(shard);
        return shard;
    }

    /// <summary>
    /// All-gathers the shards of <paramref name="param"/> from every rank
    /// and returns the full-size parameter. Caller must call
    /// <see cref="ReleaseParameter"/> when done so the gathered tensor
    /// can be reclaimed (FullShard strategies hold only the shard
    /// outside of the gather window).
    /// </summary>
    public Tensor<T> GatherParameter(ShardedParameter<T> param)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (_options.Strategy == ShardingStrategy.NoShard) return param.LocalShard;

        // All-gather: each rank contributes its shard; result is the
        // concatenation across ranks. We allocate the full tensor, then
        // every rank writes its own slice.
        var full = new Tensor<T>((int[])param.FullShape.Clone());
        var perRank = new List<Tensor<T>>(_group.WorldSize);
        for (int r = 0; r < _group.WorldSize; r++)
        {
            // Per-rank slice of the same shape as LocalShard.
            perRank.Add(new Tensor<T>((int[])param.LocalShard._shape.Clone()));
        }
        _group.AllGather(param.LocalShard, perRank);
        // Concatenate into full.
        int offset = 0;
        var dst = full.AsWritableSpan();
        for (int r = 0; r < perRank.Count; r++)
        {
            var src = perRank[r].AsSpan();
            int copyLen = Math.Min(src.Length, dst.Length - offset);
            if (copyLen <= 0) break;
            src.Slice(0, copyLen).CopyTo(dst.Slice(offset, copyLen));
            offset += copyLen;
        }
        return full;
    }

    /// <summary>Marks a previously-gathered parameter as no longer
    /// needed; the framework can reclaim its memory. No-op for
    /// <see cref="ShardingStrategy.NoShard"/>.</summary>
    public void ReleaseParameter(ShardedParameter<T> param)
    {
        // In a real GPU integration this would return the gathered
        // buffer to the workspace pool. The CPU path relies on GC.
        _ = param;
    }

    /// <summary>
    /// Reduce-scatter the full-size <paramref name="grad"/> back into
    /// per-rank shards. Each rank ends up with the reduction of the
    /// full grad's slice that corresponds to its parameter shard.
    /// </summary>
    /// <returns>The local-rank gradient shard.</returns>
    public Tensor<T> ReduceScatterGradients(ShardedParameter<T> param, Tensor<T> fullGrad)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (fullGrad is null) throw new ArgumentNullException(nameof(fullGrad));
        if (_options.Strategy == ShardingStrategy.NoShard)
        {
            // DDP-equivalent: just all-reduce the full grad in place.
            _group.AllReduce(fullGrad, ReduceOp.Avg);
            return fullGrad;
        }

        // Slice fullGrad into WorldSize equal-sized chunks, reduce-scatter
        // so rank R ends up with its chunk's reduction.
        var perRankInputs = new List<Tensor<T>>(_group.WorldSize);
        int chunkLen = param.LocalShard.Length;
        var srcSpan = fullGrad.AsSpan();
        for (int r = 0; r < _group.WorldSize; r++)
        {
            var chunk = new Tensor<T>((int[])param.LocalShard._shape.Clone());
            var dstSpan = chunk.AsWritableSpan();
            int from = r * chunkLen;
            int copyLen = Math.Min(chunkLen, srcSpan.Length - from);
            if (copyLen > 0) srcSpan.Slice(from, copyLen).CopyTo(dstSpan.Slice(0, copyLen));
            perRankInputs.Add(chunk);
        }
        var output = new Tensor<T>((int[])param.LocalShard._shape.Clone());
        _group.ReduceScatter(perRankInputs, output, ReduceOp.Avg);
        return output;
    }
}

/// <summary>
/// One sharded parameter — holds the full-shape metadata + this rank's
/// shard. The shard is the contiguous slice of the flattened parameter
/// at rank R: <c>full[R * shardLen : (R + 1) * shardLen]</c> reshaped to
/// the slice's natural shape (or padded to the chunk size when
/// numel doesn't divide WorldSize).
/// </summary>
public sealed class ShardedParameter<T>
{
    /// <summary>Original full-tensor shape. Recovered by GatherParameter.</summary>
    public int[] FullShape { get; }

    /// <summary>This rank's shard.</summary>
    public Tensor<T> LocalShard { get; }

    private ShardedParameter(int[] fullShape, Tensor<T> shard)
    {
        FullShape = fullShape;
        LocalShard = shard;
    }

    internal static ShardedParameter<T> Create(Tensor<T> full, IProcessGroup group, FsdpOptions options)
    {
        var fullShape = (int[])full._shape.Clone();
        if (options.Strategy == ShardingStrategy.NoShard)
        {
            // No-shard: every rank holds the full param.
            return new ShardedParameter<T>(fullShape, full);
        }

        int totalLen = full.Length;
        int chunk = (totalLen + group.WorldSize - 1) / group.WorldSize;
        int from = group.Rank * chunk;
        int actual = Math.Min(chunk, Math.Max(0, totalLen - from));

        // Allocate a chunk-sized shard (padded to make every rank's
        // shard the same size — required by ReduceScatter alignment).
        var shard = new Tensor<T>(new[] { chunk });
        if (actual > 0)
        {
            full.AsSpan().Slice(from, actual).CopyTo(shard.AsWritableSpan().Slice(0, actual));
        }
        return new ShardedParameter<T>(fullShape, shard);
    }
}
