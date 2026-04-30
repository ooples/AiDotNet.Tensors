// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Distributed;

/// <summary>One pipeline-parallel stage — owns its sub-model's forward
/// + backward functions. The pipeline scheduler chains stages
/// rank-to-rank: stage S is hosted by rank S.
/// </summary>
public abstract class PipelineStage<T>
{
    /// <summary>Stage index (0-based). Equals this stage's owner rank.</summary>
    public int StageIndex { get; }

    /// <summary>Total number of stages in the pipeline.</summary>
    public int NumStages { get; }

    /// <summary>Constructs.</summary>
    protected PipelineStage(int stageIndex, int numStages)
    {
        StageIndex = stageIndex;
        NumStages = numStages;
    }

    /// <summary>
    /// Forward pass for one micro-batch. Receives the input from the
    /// previous stage (or the user-provided input on stage 0); returns
    /// the activation passed to the next stage (or to the loss on the
    /// last stage).
    /// </summary>
    public abstract Tensor<T> Forward(Tensor<T> input);

    /// <summary>
    /// Backward pass — receives the gradient from the next stage and
    /// returns the gradient passed back to the previous stage.
    /// </summary>
    public abstract Tensor<T> Backward(Tensor<T> gradOutput);
}

/// <summary>
/// GPipe-style pipeline-parallel scheduler. Splits each batch into
/// <c>numMicroBatches</c> micro-batches; runs a fill-drain schedule:
/// every micro-batch flows through every stage, with stage S running
/// micro-batch M only after stage S-1 has finished M.
///
/// <para>Communication: stage outputs are sent to the next-rank stage
/// via the process group's <see cref="IProcessGroup.Send{T}"/> /
/// <see cref="IProcessGroup.Recv{T}"/> primitives. Each (micro-batch,
/// activation) pair is a single point-to-point.</para>
/// </summary>
public sealed class GPipeSchedule<T>
{
    private readonly IProcessGroup _group;
    private readonly PipelineStage<T> _stage;

    /// <summary>Constructs.</summary>
    public GPipeSchedule(IProcessGroup group, PipelineStage<T> stage)
    {
        _group = group ?? throw new ArgumentNullException(nameof(group));
        _stage = stage ?? throw new ArgumentNullException(nameof(stage));
        if (group.WorldSize != stage.NumStages)
            throw new ArgumentException(
                $"WorldSize {group.WorldSize} must equal NumStages {stage.NumStages} for 1-stage-per-rank GPipe.");
        if (group.Rank != stage.StageIndex)
            throw new ArgumentException(
                $"This rank ({group.Rank}) must own stage index {group.Rank}; got stage {stage.StageIndex}.");
    }

    /// <summary>
    /// Runs one full GPipe iteration: forward through every micro-batch
    /// across every stage, then backward through every micro-batch in
    /// reverse stage order. Returns the per-micro-batch loss tensors on
    /// the last rank, empty on others.
    /// </summary>
    /// <param name="microBatches">Per-micro-batch input. Required on
    /// stage 0; ignored elsewhere.</param>
    /// <param name="lossInitialGrads">Per-micro-batch initial gradient
    /// (typically ones for the loss seed). Required on the last stage.</param>
    public Tensor<T>[] Run(int numMicroBatches,
        IReadOnlyList<Tensor<T>>? microBatches, IReadOnlyList<Tensor<T>>? lossInitialGrads)
    {
        if (numMicroBatches <= 0)
            throw new ArgumentOutOfRangeException(nameof(numMicroBatches),
                "numMicroBatches must be positive — middle ranks (microBatches=null and " +
                "lossInitialGrads=null) can't infer the count from edge inputs.");
        int M = numMicroBatches;

        bool isFirst = _group.Rank == 0;
        bool isLast = _group.Rank == _group.WorldSize - 1;

        // Validate edge inputs against the explicit count.
        if (isFirst && (microBatches is null || microBatches.Count != M))
            throw new ArgumentException(
                $"Stage 0 must receive exactly {M} microBatches; got {microBatches?.Count ?? 0}.",
                nameof(microBatches));
        if (isLast && (lossInitialGrads is null || lossInitialGrads.Count != M))
            throw new ArgumentException(
                $"Last stage must receive exactly {M} loss initial gradients; got {lossInitialGrads?.Count ?? 0}.",
                nameof(lossInitialGrads));

        // Forward fill-drain: every micro-batch flows through every stage.
        var fwdActs = new Tensor<T>[M];
        for (int m = 0; m < M; m++)
        {
            Tensor<T> input;
            if (isFirst)
            {
                input = microBatches![m];
            }
            else
            {
                input = ReceiveActivation(m);
            }
            fwdActs[m] = _stage.Forward(input);
            if (!isLast) _group.Send(fwdActs[m], dst: _group.Rank + 1, tag: m);
        }

        // Backward fill-drain in reverse.
        var grads = new Tensor<T>[M];
        for (int m = M - 1; m >= 0; m--)
        {
            Tensor<T> gradOut;
            if (isLast)
            {
                gradOut = lossInitialGrads![m];
            }
            else
            {
                gradOut = ReceiveGrad(m);
            }
            grads[m] = _stage.Backward(gradOut);
            if (!isFirst) _group.Send(grads[m], dst: _group.Rank - 1, tag: 100_000 + m);
            // Free intermediate ranks' fwdActs as soon as backward
            // for that micro-batch is done — only the last rank
            // returns them, so non-last ranks don't need to retain.
            if (!isLast) fwdActs[m] = null!;
        }

        return isLast ? fwdActs : Array.Empty<Tensor<T>>();
    }

    /// <summary>Backwards-compat overload — infers M from edge
    /// inputs. Throws on middle ranks since neither edge input is
    /// available there.</summary>
    [Obsolete("Use the overload that takes numMicroBatches explicitly. The legacy " +
        "edge-only inference deadlocks middle ranks where both lists are null.")]
    public Tensor<T>[] Run(IReadOnlyList<Tensor<T>>? microBatches, IReadOnlyList<Tensor<T>>? lossInitialGrads)
    {
        int M = (microBatches?.Count) ?? (lossInitialGrads?.Count) ?? 0;
        if (M == 0)
            throw new InvalidOperationException(
                "Cannot infer numMicroBatches on middle ranks; pass it explicitly to the " +
                "Run(int, ...) overload.");
        return Run(M, microBatches, lossInitialGrads);
    }

    private Tensor<T> ReceiveActivation(int m)
    {
        // We don't know the shape ahead of time in this minimal scaffold;
        // for the in-process backend the Send already enqueues the full
        // tensor object so the Recv copy-loop reads its actual shape from
        // the queued object. We allocate a placeholder of size 1 and let
        // the InProcessGroup recv copy work — its CopyTo asserts shape
        // equality, so a real implementation would either prepend a
        // shape or pass a typed contract. Here we trust the queued
        // object's shape via a custom recv path:
        var msg = _group.RecvDiscoverShape<T>(_group.Rank - 1, m);
        return msg;
    }

    private Tensor<T> ReceiveGrad(int m)
    {
        var msg = _group.RecvDiscoverShape<T>(_group.Rank + 1, 100_000 + m);
        return msg;
    }
}

/// <summary>
/// 1F1B (one-forward-one-backward) interleaved pipeline schedule.
/// After steady state, each stage alternates between a forward
/// micro-batch and a backward micro-batch — reduces peak activation
/// memory vs GPipe's "all forwards then all backwards" pattern.
/// </summary>
public sealed class OneForwardOneBackwardSchedule<T>
{
    private readonly IProcessGroup _group;
    private readonly PipelineStage<T> _stage;

    /// <summary>Constructs.</summary>
    public OneForwardOneBackwardSchedule(IProcessGroup group, PipelineStage<T> stage)
    {
        _group = group ?? throw new ArgumentNullException(nameof(group));
        _stage = stage ?? throw new ArgumentNullException(nameof(stage));
        if (group.WorldSize != stage.NumStages)
            throw new ArgumentException("WorldSize must equal NumStages for 1F1B.");
        if (group.Rank != stage.StageIndex)
            throw new ArgumentException("This rank must own its stage index for 1F1B.");
    }

    /// <summary>
    /// Runs one full 1F1B iteration. The schedule:
    /// <list type="bullet">
    ///   <item>Warmup: stage S runs (NumStages - S) forward-only steps.</item>
    ///   <item>Steady state: alternate one forward, one backward.</item>
    ///   <item>Cooldown: drain remaining backwards.</item>
    /// </list>
    /// </summary>
    public Tensor<T>[] Run(int numMicroBatches,
        IReadOnlyList<Tensor<T>>? microBatches, IReadOnlyList<Tensor<T>>? lossInitialGrads)
    {
        if (numMicroBatches <= 0)
            throw new ArgumentOutOfRangeException(nameof(numMicroBatches),
                "numMicroBatches must be positive — middle ranks can't infer it from edge inputs.");
        int M = numMicroBatches;
        bool isFirst = _group.Rank == 0;
        bool isLast = _group.Rank == _group.WorldSize - 1;
        int S = _stage.StageIndex;
        int W = _stage.NumStages;
        if (isFirst && (microBatches is null || microBatches.Count != M))
            throw new ArgumentException(
                $"Stage 0 must receive exactly {M} microBatches.", nameof(microBatches));
        if (isLast && (lossInitialGrads is null || lossInitialGrads.Count != M))
            throw new ArgumentException(
                $"Last stage must receive exactly {M} loss initial gradients.", nameof(lossInitialGrads));

        // Warmup: this stage runs (W - S) forward steps before the first backward.
        int warmup = Math.Min(W - S, M);
        var fwdActs = new Tensor<T>[M];
        for (int m = 0; m < warmup; m++)
        {
            var input = isFirst ? microBatches![m] : _group.RecvDiscoverShape<T>(_group.Rank - 1, m);
            fwdActs[m] = _stage.Forward(input);
            if (!isLast) _group.Send(fwdActs[m], dst: _group.Rank + 1, tag: m);
        }

        // Steady state: alternate F and B until all forwards are done.
        int fwdIdx = warmup, bwdIdx = 0;
        while (fwdIdx < M)
        {
            // Backward for the oldest pending micro-batch.
            var gradOut = isLast ? lossInitialGrads![bwdIdx]
                                 : _group.RecvDiscoverShape<T>(_group.Rank + 1, 100_000 + bwdIdx);
            var gradIn = _stage.Backward(gradOut);
            if (!isFirst) _group.Send(gradIn, dst: _group.Rank - 1, tag: 100_000 + bwdIdx);
            bwdIdx++;

            // Forward for the next micro-batch.
            var input = isFirst ? microBatches![fwdIdx]
                                : _group.RecvDiscoverShape<T>(_group.Rank - 1, fwdIdx);
            fwdActs[fwdIdx] = _stage.Forward(input);
            if (!isLast) _group.Send(fwdActs[fwdIdx], dst: _group.Rank + 1, tag: fwdIdx);
            fwdIdx++;
        }

        // Cooldown: drain remaining backwards.
        while (bwdIdx < M)
        {
            var gradOut = isLast ? lossInitialGrads![bwdIdx]
                                 : _group.RecvDiscoverShape<T>(_group.Rank + 1, 100_000 + bwdIdx);
            var gradIn = _stage.Backward(gradOut);
            if (!isFirst) _group.Send(gradIn, dst: _group.Rank - 1, tag: 100_000 + bwdIdx);
            // 1F1B's main memory-saving property: drop activations as
            // each backward finishes on intermediate ranks. Only the
            // last rank returns them.
            if (!isLast) fwdActs[bwdIdx] = null!;
            bwdIdx++;
        }

        return isLast ? fwdActs : Array.Empty<Tensor<T>>();
    }

    /// <summary>Legacy overload — see GPipeSchedule.Run for rationale.</summary>
    [Obsolete("Use the overload that takes numMicroBatches explicitly.")]
    public Tensor<T>[] Run(IReadOnlyList<Tensor<T>>? microBatches, IReadOnlyList<Tensor<T>>? lossInitialGrads)
    {
        int M = (microBatches?.Count) ?? (lossInitialGrads?.Count) ?? 0;
        if (M == 0)
            throw new InvalidOperationException(
                "Cannot infer numMicroBatches on middle ranks; pass it explicitly.");
        return Run(M, microBatches, lossInitialGrads);
    }
}
