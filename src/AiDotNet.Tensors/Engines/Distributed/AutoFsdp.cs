// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Auto-instrumented FSDP — wires <see cref="FullyShardedDataParallel{T}"/>
/// into the <see cref="GradientTape{T}.ComputeGradients"/> result so the
/// user no longer has to call
/// <see cref="FullyShardedDataParallel{T}.ReduceScatterGradients"/> by
/// hand at every backward. PyTorch-side parity is the
/// <c>torch.distributed.fsdp.fully_shard</c> decorator: a one-line
/// instrumentation that lifts a parameter set into the sharded world.
///
/// <para><b>Design.</b> The wrapper records every parameter you register
/// in a (full-param ↦ shard-handle) dictionary. After
/// <see cref="GradientTape{T}.ComputeGradients"/> returns, you pass the
/// gradient dictionary to <see cref="ProcessGradients"/> which runs the
/// FSDP-strategy-appropriate collective on every registered parameter
/// in one pass — <see cref="FullyShardedDataParallel{T}.ReduceScatterGradients"/>
/// for full/hybrid shards, <c>AllReduce</c> for ZeRO-1 / NoShard — and
/// stores the rank-local gradient slice in <see cref="LocalGradient"/>.
/// This avoids the fire-multiple-times pitfall of mid-walk hooks: the
/// reduce-scatter sees a single final gradient per parameter, not a
/// partial accumulation.</para>
///
/// <para>Usage:
/// <code>
/// var fsdp = new FullyShardedDataParallel&lt;float&gt;(group, options);
/// using var tape = new GradientTape&lt;float&gt;();
/// var auto = new AutoFsdp&lt;float&gt;(fsdp, tape);
///
/// // Register every full-size parameter once.
/// var sharded = auto.Shard(weights);
///
/// // Run forward + backward as usual.
/// var loss = ComputeLoss(weights, x, y);
/// var grads = tape.ComputeGradients(loss, sources: new[] { weights });
/// auto.ProcessGradients(grads);
///
/// // Pull the rank-local gradient slice for the optimizer step.
/// var localGrad = auto.LocalGradient(sharded);
/// fsdp.OptimizerStep(sharded, weights, localGrad, (p, g) =&gt; sgd.Step(p, g));
/// </code>
/// </para>
/// </summary>
/// <typeparam name="T">Element type — float / double / etc.</typeparam>
public sealed class AutoFsdp<T>
{
    private readonly FullyShardedDataParallel<T> _fsdp;
    private readonly GradientTape<T> _tape;
    private readonly Dictionary<ShardedParameter<T>, Tensor<T>> _localGrads = new();
    private readonly Dictionary<Tensor<T>, ShardedParameter<T>> _byFullParam = new();

    /// <summary>Constructs an auto-instrumenter bound to a tape.
    /// The tape reference is retained so future API additions (e.g.
    /// pre-forward all-gather instrumentation) can plumb through it
    /// without an API change.</summary>
    public AutoFsdp(FullyShardedDataParallel<T> fsdp, GradientTape<T> tape)
    {
        _fsdp = fsdp ?? throw new ArgumentNullException(nameof(fsdp));
        _tape = tape ?? throw new ArgumentNullException(nameof(tape));
    }

    /// <summary>The bound FSDP instance.</summary>
    public FullyShardedDataParallel<T> Fsdp => _fsdp;

    /// <summary>The bound tape.</summary>
    public GradientTape<T> Tape => _tape;

    /// <summary>
    /// Registers a full-size parameter with the underlying FSDP. After
    /// <see cref="GradientTape{T}.ComputeGradients"/> returns, call
    /// <see cref="ProcessGradients"/> to fan the FSDP collective across
    /// every registered parameter and stage the rank-local slice for
    /// <see cref="LocalGradient"/>.
    /// </summary>
    public ShardedParameter<T> Shard(Tensor<T> fullParam)
    {
        if (fullParam is null) throw new ArgumentNullException(nameof(fullParam));
        var sharded = _fsdp.RegisterParameter(fullParam);
        _byFullParam[fullParam] = sharded;
        return sharded;
    }

    /// <summary>
    /// Runs the FSDP-strategy-appropriate collective for every registered
    /// parameter that has a gradient in <paramref name="grads"/>. For
    /// <see cref="ShardingStrategy.ShardOptimizerOnly"/> this is an
    /// AllReduce-Avg followed by a slice; for
    /// <see cref="ShardingStrategy.FullShard"/> /
    /// <see cref="ShardingStrategy.HybridShard"/> it's a ReduceScatter-Avg.
    /// Stores the rank-local slice in <see cref="LocalGradient"/>.
    /// </summary>
    /// <param name="grads">The dictionary returned by
    /// <see cref="GradientTape{T}.ComputeGradients"/>.</param>
    public void ProcessGradients(IReadOnlyDictionary<Tensor<T>, Tensor<T>> grads)
    {
        if (grads is null) throw new ArgumentNullException(nameof(grads));
        foreach (var kv in _byFullParam)
        {
            var fullParam = kv.Key;
            var sharded = kv.Value;
            if (!grads.TryGetValue(fullParam, out var grad)) continue;
            var localOrFull = _fsdp.ReduceScatterGradients(sharded, grad);
            _localGrads[sharded] = localOrFull;
        }
    }

    /// <summary>
    /// Returns the rank-local gradient slice produced by the most recent
    /// <see cref="ProcessGradients"/> call. Throws if no
    /// <see cref="ProcessGradients"/> has fired yet for this parameter.
    /// </summary>
    public Tensor<T> LocalGradient(ShardedParameter<T> sharded)
    {
        if (sharded is null) throw new ArgumentNullException(nameof(sharded));
        if (!_localGrads.TryGetValue(sharded, out var grad))
            throw new InvalidOperationException(
                "No local gradient recorded yet for this sharded parameter. " +
                "Ensure ProcessGradients has run after ComputeGradients.");
        return grad;
    }

    /// <summary>True if the local gradient for <paramref name="sharded"/>
    /// has been populated by <see cref="ProcessGradients"/>.</summary>
    public bool HasLocalGradient(ShardedParameter<T> sharded) =>
        sharded is not null && _localGrads.ContainsKey(sharded);

    /// <summary>Clears all recorded local gradients. Call between training
    /// steps so a stale slice can't leak into the next step's optimizer
    /// call. PyTorch parity with <c>optimizer.zero_grad()</c>.</summary>
    public void ZeroLocalGradients() => _localGrads.Clear();
}
