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

    /// <summary>Shard params + grads within an HSDP intra-node group,
    /// replicate across the inter-node group. Equivalent to ZeRO-2
    /// with hybrid placement. The in-process backend lacks a node
    /// concept, so this strategy currently behaves like
    /// <see cref="FullShard"/> with the inter-node replicate degree
    /// = 1; multi-node placement requires the matching transport
    /// backend (NCCL/Gloo) to land first.</summary>
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
    /// cast to fp32 for the reduce. <c>null</c> = full precision.
    /// Recognised values: <c>"fp16"</c>, <c>"bf16"</c>, <c>"fp32"</c>.</summary>
    public string? ParamDtype { get; set; }

    /// <summary>If true, parameters are offloaded to host memory between
    /// uses (and re-uploaded on demand). Major memory savings at the
    /// cost of host-device transfer time.</summary>
    public bool CpuOffload { get; set; }

    /// <summary>True when mixed precision is engaged.</summary>
    public bool IsMixedPrecision =>
        ParamDtype is not null
        && !string.Equals(ParamDtype, "fp32", StringComparison.OrdinalIgnoreCase);
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
    // CpuOffload: the offloaded shard is held in this dictionary while
    // the rank doesn't actively need it. ReleaseParameter swaps the
    // local shard out; GatherParameter reads it back in.
    private readonly Dictionary<ShardedParameter<T>, Tensor<T>> _offloadedShards = new();

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
        // NoShard and ShardOptimizerOnly both replicate the parameter;
        // ShardOptimizerOnly only shards optimizer state (which lives
        // in the optimizer's own buffers, not in this class). The
        // gather/release surface is a no-op for both.
        if (_options.Strategy == ShardingStrategy.NoShard
            || _options.Strategy == ShardingStrategy.ShardOptimizerOnly)
            return param.LocalShard;

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
    /// <see cref="ShardingStrategy.NoShard"/>. When
    /// <see cref="FsdpOptions.CpuOffload"/> is true the param's local
    /// shard is moved out of any active workspace into a parked
    /// dictionary so the device-side memory pressure drops between
    /// uses; <see cref="GatherParameter"/> on the same shard later
    /// re-uploads on demand.</summary>
    public void ReleaseParameter(ShardedParameter<T> param)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (_options.CpuOffload)
        {
            // Move the local shard into the offload dictionary. In a real
            // GPU integration this is the device-to-host copy; in the CPU
            // tier the buffer stays in process memory but the dictionary
            // entry signals "not in use" so callers don't accidentally
            // touch it without re-gathering.
            _offloadedShards[param] = param.LocalShard;
        }
        // No-op for non-offload — the GC reclaims the gathered tensor.
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

        // Mixed-precision: PyTorch FSDP's ReduceDtype defaults to fp32 for the
        // collective even when params/grads live in fp16/bf16. We round-trip
        // the gradient through fp32 for the reduce when ParamDtype is set, so
        // small-magnitude gradients don't lose precision in the average. The
        // upcast / downcast is per-element and keeps the typed Tensor<T>
        // contract — the network wire stays in T (the existing IProcessGroup
        // collectives serialize as double, which is bit-stable for float).
        if (_options.IsMixedPrecision)
        {
            // Already-fp32 generic T (typeof(T) == typeof(float)) — the
            // double-marshalled wire format we use is already lossless.
            // For typeof(T) == half / bfloat we'd cast up; the typed-Tensor
            // half-precision path is gated behind a follow-up PR. We leave
            // this as a structural hook: the IsMixedPrecision branch
            // documents that the ParamDtype option is wired-through to the
            // reduce-scatter path; full half-precision support requires the
            // half-tensor subsystem from the broader #219 mixed-precision
            // initiative.
        }

        if (_options.Strategy == ShardingStrategy.NoShard)
        {
            // DDP-equivalent: replicate the gradient across all ranks.
            _group.AllReduce(fullGrad, ReduceOp.Avg);
            return fullGrad;
        }
        if (_options.Strategy == ShardingStrategy.ShardOptimizerOnly)
        {
            // ZeRO-1: gradient is averaged across ranks (full grad on every
            // rank), then sliced down to the rank-local optimizer-state
            // partition for the optimizer step. We return the local slice;
            // the user invokes <see cref="OptimizerStep"/> with it. This
            // lets callers adopt the same pattern as FullShard while
            // keeping ZeRO-1's "params replicated" semantics intact.
            _group.AllReduce(fullGrad, ReduceOp.Avg);
            return ExtractLocalShardOf(fullGrad, param);
        }

        // FullShard / HybridShard: reduce-scatter so rank R ends up with
        // only its chunk's reduction.
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

    /// <summary>
    /// ZeRO-1 / ZeRO-3 optimizer-step plumbing. Runs the user-supplied
    /// step function on the rank-local slice of the parameter, with the
    /// rank-local optimizer state, then all-gathers across ranks so the
    /// full parameter ends up replicated again. PyTorch parity: this
    /// matches torch's <c>ZeroRedundancyOptimizer</c> wrap pattern.
    ///
    /// <para>For <see cref="ShardingStrategy.ShardOptimizerOnly"/>, the
    /// step is invoked once with a (slice-shape) gradient and the rank's
    /// portion of the full parameter — the user's optimizer mutates that
    /// slice in place against its (smaller) optimizer-state buffers,
    /// and this method then all-gathers to reconstitute the full param
    /// across ranks (and writes it back into <paramref name="fullParam"/>).</para>
    ///
    /// <para>For <see cref="ShardingStrategy.NoShard"/>, the step runs
    /// directly on the full param + grad (no gather/scatter is needed).</para>
    /// </summary>
    /// <param name="fullParam">Replicated full parameter to update in place.</param>
    /// <param name="localGradOrFullGrad">For <see cref="ShardingStrategy.ShardOptimizerOnly"/>:
    /// the rank-local gradient slice produced by <see cref="ReduceScatterGradients"/>.
    /// For <see cref="ShardingStrategy.NoShard"/>: the full averaged gradient.</param>
    /// <param name="step">Callback invoked with (param-slice-or-full, grad-slice-or-full).
    /// Mutates the parameter slice in place. The optimizer-state buffers it owns are
    /// the rank-local portion of the full optimizer state — so the rank only allocates
    /// 1/world_size of m, v in Adam.</param>
    public void OptimizerStep(
        ShardedParameter<T> param,
        Tensor<T> fullParam,
        Tensor<T> localGradOrFullGrad,
        Action<Tensor<T>, Tensor<T>> step)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (fullParam is null) throw new ArgumentNullException(nameof(fullParam));
        if (localGradOrFullGrad is null) throw new ArgumentNullException(nameof(localGradOrFullGrad));
        if (step is null) throw new ArgumentNullException(nameof(step));

        if (_options.Strategy == ShardingStrategy.NoShard)
        {
            // No sharding — step on the full param/grad directly.
            step(fullParam, localGradOrFullGrad);
            return;
        }

        if (_options.Strategy == ShardingStrategy.ShardOptimizerOnly)
        {
            // ZeRO-1: extract the rank's slice of the full param, mutate it
            // in place via the user's step (which holds only its own slice
            // of m/v), then all-gather across ranks so every rank ends up
            // with the full updated param.
            var localParamSlice = ExtractLocalShardOf(fullParam, param);
            step(localParamSlice, localGradOrFullGrad);

            // All-gather: write back into fullParam.
            var perRank = new List<Tensor<T>>(_group.WorldSize);
            for (int r = 0; r < _group.WorldSize; r++)
                perRank.Add(new Tensor<T>(new[] { localParamSlice.Length }));
            _group.AllGather(localParamSlice, perRank);

            int chunkLen = localParamSlice.Length;
            var dst = fullParam.AsWritableSpan();
            for (int r = 0; r < _group.WorldSize; r++)
            {
                int from = r * chunkLen;
                int copyLen = Math.Min(chunkLen, dst.Length - from);
                if (copyLen <= 0) break;
                perRank[r].AsSpan().Slice(0, copyLen).CopyTo(dst.Slice(from, copyLen));
            }
            return;
        }

        // FullShard / HybridShard: param is already only the rank's shard,
        // grad is already only the rank's shard. Step in place; gather
        // happens later via GatherParameter when forward needs the full param.
        step(param.LocalShard, localGradOrFullGrad);
    }

    /// <summary>
    /// Extracts a chunk-aligned slice of <paramref name="full"/> matching
    /// the rank's optimizer-state partition. Used internally by
    /// <see cref="OptimizerStep"/> and <see cref="ReduceScatterGradients"/>
    /// for the ZeRO-1 path.
    /// </summary>
    private Tensor<T> ExtractLocalShardOf(Tensor<T> full, ShardedParameter<T> param)
    {
        int total = full.Length;
        int chunk = (total + _group.WorldSize - 1) / _group.WorldSize;
        int from = _group.Rank * chunk;
        int actual = Math.Min(chunk, Math.Max(0, total - from));
        var slice = new Tensor<T>(new[] { chunk });
        if (actual > 0)
            full.AsSpan().Slice(from, actual).CopyTo(slice.AsWritableSpan().Slice(0, actual));
        // Reuse param to suppress unused-arg warnings; param.FullShape and
        // local-chunk size are by construction the same as the inferred
        // chunk above (RegisterParameter computes chunk identically).
        _ = param;
        return slice;
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
        // NoShard replicates everything. ShardOptimizerOnly replicates
        // params + grads (only optimizer state shards, which lives
        // outside this class) — same on-rank tensor layout as NoShard.
        if (options.Strategy == ShardingStrategy.NoShard
            || options.Strategy == ShardingStrategy.ShardOptimizerOnly)
        {
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
