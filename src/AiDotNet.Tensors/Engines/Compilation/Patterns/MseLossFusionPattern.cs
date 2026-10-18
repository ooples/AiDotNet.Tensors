// Copyright (c) AiDotNet. All rights reserved.
// Built-in fusion: ReduceMean(Square(Subtract(p, y))) over every axis -> one mean-squared-error node.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the mean squared error as it is usually written - <c>ReduceMean(Square(Subtract(p, y)))</c> over every axis,
/// the square recorded as <c>Multiply(d, d)</c> - and emits one <c>FusedMseLoss</c> node. On the GPU engine its forward
/// is one launch and its backward another (<see cref="DirectGpu.IFusedMseLossBackend"/>); composed, the loss and its
/// gradient were about ten nodes of a captured step graph, every parity network's loss among them.
/// </summary>
/// <remarks>
/// Only graphs recorded on <see cref="DirectGpuTensorEngine"/> in float are fused. The difference and the square must
/// have no other consumer. The fused node keeps the mean's output tensor and shape (rank 0, or all ones with keepDims),
/// and its backward gives the target a gradient only when one is required, which the composed backward did not know
/// (it negated the difference's gradient for the batch's targets every step).
/// </remarks>
internal sealed class MseLossFusionPattern : IFusionPattern
{
    public string Name => "FusedMseLoss";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records FusedMseLossBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (nodes[index] is not LazyNode<float> { OpType: LazyNodeType.Subtract } sub
            || sub.RecordingEngine is not DirectGpuTensorEngine || sub.Input1 is null
            || !PatternHelpers.SameShape(sub.Input0, sub.Input1))
            return false;
        // The square: Multiply(d, d), the difference's only consumer (it is listed once per input slot).
        if (!consumerCounts.TryGetValue(sub, out int subConsumers) || subConsumers != 2) return false;
        var square = PatternHelpers.FindNextFloat(nodes, index + 1, nodes.Count - 1, alreadyRemoved,
            n => PatternHelpers.ConsumesOutput(n, sub));
        if (square is null || !PatternHelpers.IsMultiply(square.OpType)
            || !ReferenceEquals(square.Input0, sub.Output) || !ReferenceEquals(square.Input1, sub.Output))
            return false;
        // The mean over every axis, the square's only consumer.
        if (!consumerCounts.TryGetValue(square, out int squareConsumers) || squareConsumers != 1) return false;
        var mean = PatternHelpers.FindNextFloat(nodes, IndexOf(nodes, square, index + 1) + 1, nodes.Count - 1, alreadyRemoved,
            n => PatternHelpers.ConsumesOutput(n, square));
        if (mean is null || mean.OpType != LazyNodeType.ReduceMean || !ReducesEveryAxis(mean, square.Output.Rank))
            return false;

        var predictions = sub.Input0;
        var targets = sub.Input1;
        var outputShape = (int[])mean.Output._shape.Clone();
        var node = new LazyNode<float>(LazyNodeType.Custom, "FusedMseLoss", new[] { predictions, targets }, mean.Output,
            (eng, result) =>
            {
                Tensor<float>? loss = eng is DirectGpuTensorEngine gpu ? gpu.TryFusedMseLoss(predictions, targets, outputShape) : null;
                if (loss is null)
                {
                    var diff = eng.TensorSubtract(predictions, targets);
                    var squared = eng.TensorMultiply(diff, diff);
                    var axes = new int[squared.Rank];
                    for (int a = 0; a < axes.Length; a++) axes[a] = a;
                    loss = eng.ReduceMean(squared, axes, keepDims: outputShape.Length == squared.Rank);
                }
                DirectGpuTensorEngine.CopyResultInto(eng, loss, result);
            },
            BackwardFunctions<float>.FusedMseLossBackward, null)
        {
            RecordingEngine = sub.RecordingEngine,
        };
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(sub);
        alreadyRemoved.Add(square);
        alreadyRemoved.Add(mean);
        return true;
    }

    private static int IndexOf(IReadOnlyList<ILazyNode> nodes, ILazyNode node, int from)
    {
        for (int i = from; i < nodes.Count; i++)
            if (ReferenceEquals(nodes[i], node)) return i;
        return nodes.Count;
    }

    // ReduceMean saves { axes, keepDims }: every axis of the square, each once.
    private static bool ReducesEveryAxis(LazyNode<float> mean, int rank)
    {
        if (mean.SavedState is not { Length: >= 1 } state || state[0] is not int[] axes || axes.Length != rank) return false;
        var seen = new bool[rank];
        foreach (int a in axes)
        {
            int axis = a < 0 ? a + rank : a;
            if (axis < 0 || axis >= rank || seen[axis]) return false;
            seen[axis] = true;
        }
        return true;
    }
}
