// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>
/// Strategy for averaging gradients across DDP ranks. Mirrors the
/// implicit "sum + divide" PyTorch DDP performs.
/// </summary>
public enum DdpReduction
{
    /// <summary>Sum then divide by WorldSize. Matches PyTorch DDP.</summary>
    Mean,

    /// <summary>Sum only — used when the loss is already
    /// world-size-scaled (some FSDP setups).</summary>
    Sum,
}

/// <summary>
/// Configuration for <see cref="DistributedDataParallel{T}"/>. Mirrors
/// PyTorch's DDP keyword args.
/// </summary>
public sealed class DdpOptions
{
    /// <summary>Maximum bytes to coalesce into a single all-reduce call.
    /// Larger buckets ⇒ fewer collectives ⇒ better wire utilization, at
    /// the cost of less overlap with backward. PyTorch default is 25 MB
    /// — we follow.</summary>
    public int BucketSizeBytes { get; set; } = 25 * 1024 * 1024;

    /// <summary>If true, reduces grads as soon as a bucket fills (overlaps
    /// with backward); if false, waits until backward completes for every
    /// parameter before reducing. PyTorch's
    /// <c>gradient_as_bucket_view=True</c>.</summary>
    public bool OverlapBackward { get; set; } = true;

    /// <summary>How to combine per-rank grads. Mean = average; Sum =
    /// raw sum (caller already scaled the loss).</summary>
    public DdpReduction Reduction { get; set; } = DdpReduction.Mean;

    /// <summary>If true, parameters that did not receive a gradient on
    /// some ranks are skipped during reduction. Necessary when models
    /// have unused branches (e.g. classifier with multiple heads where
    /// only one runs per batch).</summary>
    public bool FindUnusedParameters { get; set; }

    /// <summary>If true, the parameter set + reduce-order is locked
    /// after the first iteration. Subsequent steps skip sanity checks
    /// and run the cached reduce plan directly. PyTorch's
    /// <c>static_graph=True</c> — saves a couple of µs per step on
    /// tight inner loops, mandatory for some compile paths.</summary>
    public bool StaticGraph { get; set; }
}

/// <summary>
/// Distributed Data Parallel — wraps a model parameter set and provides
/// the bucketed gradient all-reduce at the end of each backward pass.
///
/// <para>The wrapper itself doesn't run the model — your training loop
/// computes the forward pass and gradients exactly as in single-GPU mode.
/// You then call <see cref="ReduceGradients"/> once per training step
/// (typically right before <c>optimizer.step()</c>); DDP buckets the
/// flattened gradients, fires async all-reduce per bucket, and waits.</para>
///
/// <para>The bucketing strategy matches PyTorch:
/// parameters are walked in registration order; each one's flattened
/// element count is added to the current bucket; once the bucket would
/// exceed <see cref="DdpOptions.BucketSizeBytes"/>, the bucket is closed
/// and a new one starts. The result is a stable per-iteration plan that
/// can be cached when <see cref="DdpOptions.StaticGraph"/> is true.</para>
/// </summary>
public sealed class DistributedDataParallel<T>
{
    private readonly IProcessGroup _group;
    private readonly DdpOptions _options;
    private readonly List<Tensor<T>> _params = new();
    private readonly List<int> _bucketBoundaries = new();
    private bool _planFrozen;

    /// <summary>Constructs a DDP wrapper.</summary>
    public DistributedDataParallel(IProcessGroup group, DdpOptions? options = null)
    {
        _group = group ?? throw new ArgumentNullException(nameof(group));
        _options = options ?? new DdpOptions();
    }

    /// <summary>The process group this wrapper reduces over.</summary>
    public IProcessGroup Group => _group;

    /// <summary>Configuration.</summary>
    public DdpOptions Options => _options;

    /// <summary>Parameters in registration order.</summary>
    public IReadOnlyList<Tensor<T>> Parameters => _params;

    /// <summary>Bucket boundaries — index <c>i</c> holds the parameter
    /// count cumulative through bucket <c>i</c>. The last entry equals
    /// <see cref="Parameters"/>.<c>Count</c>.</summary>
    public IReadOnlyList<int> BucketBoundaries => _bucketBoundaries;

    /// <summary>
    /// Registers a model parameter. Call before the first training step;
    /// re-registering after the plan has frozen (StaticGraph=true) throws.
    /// Parameters are bucketed in registration order.
    /// </summary>
    public void RegisterParameter(Tensor<T> param)
    {
        if (param is null) throw new ArgumentNullException(nameof(param));
        if (_planFrozen)
            throw new InvalidOperationException(
                "Cannot register parameters after the bucket plan has frozen (StaticGraph=true).");
        // If a previous step already built a bucket plan, the new
        // parameter would silently be dropped from every bucket
        // boundary check the next ReduceGradients does. Invalidate
        // the plan so EnsurePlan rebuilds at the next reduce.
        if (_bucketBoundaries.Count > 0) _bucketBoundaries.Clear();
        _params.Add(param);
    }

    /// <summary>Registers many parameters at once.</summary>
    public void RegisterParameters(IEnumerable<Tensor<T>> parameters)
    {
        foreach (var p in parameters) RegisterParameter(p);
    }

    /// <summary>
    /// Reduces the gradients for every registered parameter across the
    /// process group, in bucketed all-reduce calls. Pass the gradient
    /// tensors in the same registration order as the parameters; the
    /// length must match.
    /// </summary>
    /// <param name="grads">Per-parameter gradient tensors. Same length and
    /// order as <see cref="Parameters"/>. Each tensor is reduced in place.
    /// When <see cref="DdpOptions.FindUnusedParameters"/> is true, entries
    /// may be <c>null</c> — the wrapper substitutes a zero-tensor of the
    /// parameter's shape so the cross-rank average stays correct (a rank
    /// with no gradient contributes zero, ranks with a gradient contribute
    /// their value).</param>
    public void ReduceGradients(IReadOnlyList<Tensor<T>> grads)
    {
        if (grads is null) throw new ArgumentNullException(nameof(grads));
        if (grads.Count != _params.Count)
            throw new ArgumentException(
                $"Expected {_params.Count} grads (one per registered param); got {grads.Count}.",
                nameof(grads));

        // FindUnusedParameters: substitute null entries with zero
        // tensors of the matching parameter's shape so the bucket
        // reduce treats the rank's contribution as zero. Without this
        // pass, a null grad would NRE in the bucket walker.
        var resolved = new Tensor<T>[_params.Count];
        for (int i = 0; i < _params.Count; i++)
        {
            if (grads[i] is not null) { resolved[i] = grads[i]; continue; }
            if (!_options.FindUnusedParameters)
                throw new ArgumentException(
                    $"Gradient for parameter {i} is null and FindUnusedParameters is false.",
                    nameof(grads));
            resolved[i] = new Tensor<T>((int[])_params[i]._shape.Clone());
        }

        EnsurePlan();

        // Walk the bucket plan and reduce each bucket as one logical
        // all-reduce. We flatten + concat the bucket's grads into a single
        // contiguous buffer so the wire transfer is one collective rather
        // than one per parameter. After the reduce, we copy back into the
        // per-parameter tensors. The flatten/scatter work is amortised by
        // the network round-trip savings.
        int paramStart = 0;
        for (int b = 0; b < _bucketBoundaries.Count; b++)
        {
            int paramEnd = _bucketBoundaries[b];
            ReduceBucket(resolved, paramStart, paramEnd);
            paramStart = paramEnd;
        }
    }

    private void ReduceBucket(IReadOnlyList<Tensor<T>> grads, int start, int end)
    {
        // Sum the lengths to size the flat buffer.
        int totalLen = 0;
        for (int i = start; i < end; i++) totalLen += grads[i].Length;
        if (totalLen == 0) return;

        var flat = new Tensor<T>(new[] { totalLen });
        var flatSpan = flat.AsWritableSpan();
        int offset = 0;
        for (int i = start; i < end; i++)
        {
            var g = grads[i];
            g.AsSpan().CopyTo(flatSpan.Slice(offset, g.Length));
            offset += g.Length;
        }

        var op = _options.Reduction == DdpReduction.Mean ? ReduceOp.Avg : ReduceOp.Sum;
        _group.AllReduce(flat, op);

        // Scatter back.
        offset = 0;
        for (int i = start; i < end; i++)
        {
            var g = grads[i];
            flatSpan.Slice(offset, g.Length).CopyTo(g.AsWritableSpan());
            offset += g.Length;
        }
    }

    /// <summary>
    /// Computes the bucket plan if not already computed. Idempotent —
    /// after the first call the plan is reused, and after StaticGraph
    /// freezes it, parameter registration is locked.
    /// </summary>
    private void EnsurePlan()
    {
        if (_bucketBoundaries.Count > 0) return;
        if (_params.Count == 0)
            throw new InvalidOperationException("DDP has no registered parameters.");

        int elementBytes = ApproxElementBytes();
        int bucketCap = Math.Max(elementBytes, _options.BucketSizeBytes / Math.Max(1, elementBytes));
        int currentBucketElements = 0;

        for (int i = 0; i < _params.Count; i++)
        {
            int paramLen = _params[i].Length;
            if (currentBucketElements > 0 && currentBucketElements + paramLen > bucketCap)
            {
                _bucketBoundaries.Add(i);
                currentBucketElements = 0;
            }
            currentBucketElements += paramLen;
        }
        _bucketBoundaries.Add(_params.Count);

        if (_options.StaticGraph) _planFrozen = true;
    }

    private static int ApproxElementBytes()
    {
        try { return System.Runtime.CompilerServices.Unsafe.SizeOf<T>(); }
        catch { return 4; /* fallback to float-equivalent */ }
    }
}
