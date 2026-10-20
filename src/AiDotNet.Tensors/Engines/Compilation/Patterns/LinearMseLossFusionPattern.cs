// Copyright (c) AiDotNet. All rights reserved.
// Built-in fusion: a linear head into the mean squared error -> one node whose forward and backward are one launch each.
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches <c>FusedLinear(x, w, b)</c> with no activation whose only consumer is the mean squared error
/// <c>ReduceMean(Square(Subtract(p, y)))</c> over every axis (the square recorded as <c>Multiply(d, d)</c>), and emits one
/// <c>FusedLinearMseLoss</c> node. On the GPU engine its forward is one launch that never writes the head's output, and
/// its backward one launch for dx, dw and db (<see cref="DirectGpu.IFusedLinearMseLossBackend"/>). Composed - the head's
/// GEMM, the fused loss and its gradient, the two gradient GEMMs, the bias's column sum and a copy - they were seven
/// nodes and ~11 us of a parity LSTM's ~63 us captured step.
/// </summary>
/// <remarks>
/// Only graphs recorded on <see cref="DirectGpuTensorEngine"/> in float, with a head the backend's kernels take
/// (<see cref="DirectGpu.IFusedLinearMseLossBackend.SupportsLinearMseLoss"/>), are fused: a larger head keeps the
/// FusedLinear and FusedMseLoss nodes. The head's output, the difference and the square must have no other consumer, and
/// the targets must be produced before the head (the fused node takes the head's place). When a launch is declined at
/// run time anyway (a non-contiguous view, say), the node composes FusedLinear and the loss, and its backward the loss's
/// gradient and the linear's backward.
/// Registered ahead of <see cref="MseLossFusionPattern"/>, which would otherwise take the loss alone.
/// </remarks>
internal sealed class LinearMseLossFusionPattern : IFusionPattern
{
    public string Name => "FusedLinearMseLoss";

    /// <inheritdoc />
    public bool IsDifferentiable => true;   // records FusedLinearMseLossBackward

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        // The head: a bias-or-not FusedLinear with no activation (SavedState carries the activation when there is one).
        if (nodes[index] is not LazyNode<float> { OpType: LazyNodeType.FusedLinear } linear
            || linear.RecordingEngine is not DirectGpuTensorEngine gpu || linear.SavedState is not null
            || linear.InputCount is not (2 or 3) || linear.Input1 is null
            || linear.Input0.Rank != 2 || linear.Input1.Rank != 2 || linear.Output.Rank != 2
            || !gpu.SupportsFusedLinearMseLoss(linear.Input0._shape[0], linear.Input0._shape[1], linear.Input1._shape[1]))
            return false;
        if (!consumerCounts.TryGetValue(linear, out int linearConsumers) || linearConsumers != 1) return false;

        // The difference with the targets, in either order (the loss and the head's gradients are the same).
        var sub = PatternHelpers.FindNextFloat(nodes, index + 1, nodes.Count - 1, alreadyRemoved,
            n => PatternHelpers.ConsumesOutput(n, linear));
        if (sub is not { OpType: LazyNodeType.Subtract } || sub.Input1 is null || !PatternHelpers.SameShape(sub.Input0, sub.Input1))
            return false;
        Tensor<float> targets;
        if (ReferenceEquals(sub.Input0, linear.Output) && !ReferenceEquals(sub.Input1, linear.Output)) targets = sub.Input1;
        else if (ReferenceEquals(sub.Input1, linear.Output) && !ReferenceEquals(sub.Input0, linear.Output)) targets = sub.Input0;
        else return false;
        if (!ProducedBefore(nodes, targets, index)) return false;

        // The square: Multiply(d, d), the difference's only consumer (it is listed once per input slot).
        int subIndex = IndexOf(nodes, sub, index + 1);
        if (!consumerCounts.TryGetValue(sub, out int subConsumers) || subConsumers != 2) return false;
        var square = PatternHelpers.FindNextFloat(nodes, subIndex + 1, nodes.Count - 1, alreadyRemoved,
            n => PatternHelpers.ConsumesOutput(n, sub));
        if (square is null || !PatternHelpers.IsMultiply(square.OpType)
            || !ReferenceEquals(square.Input0, sub.Output) || !ReferenceEquals(square.Input1, sub.Output))
            return false;
        // The mean over every axis, the square's only consumer.
        if (!consumerCounts.TryGetValue(square, out int squareConsumers) || squareConsumers != 1) return false;
        var mean = PatternHelpers.FindNextFloat(nodes, IndexOf(nodes, square, subIndex + 1) + 1, nodes.Count - 1, alreadyRemoved,
            n => PatternHelpers.ConsumesOutput(n, square));
        if (mean is null || mean.OpType != LazyNodeType.ReduceMean || !ReducesEveryAxis(mean, square.Output.Rank))
            return false;

        var x = linear.Input0;
        var w = linear.Input1;
        var bias = linear.InputCount == 3 ? linear.Input2 : null;
        var outputShape = (int[])mean.Output._shape.Clone();
        var inputs = bias is not null ? new[] { x, w, bias, targets } : new[] { x, w, targets };
        var node = new LazyNode<float>(LazyNodeType.Custom, "FusedLinearMseLoss", inputs, mean.Output,
            (eng, result) =>
            {
                Tensor<float>? loss = eng is DirectGpuTensorEngine gpu ? gpu.TryFusedLinearMseLoss(x, w, bias, targets, outputShape) : null;
                if (loss is null)
                {
                    var predictions = eng.FusedLinear(x, w, bias, FusedActivationType.None);
                    loss = eng is DirectGpuTensorEngine lossGpu ? lossGpu.TryFusedMseLoss(predictions, targets, outputShape) : null;
                    if (loss is null)
                    {
                        var diff = eng.TensorSubtract(predictions, targets);
                        loss = eng.ReduceMean(eng.TensorMultiply(diff, diff), new[] { 0, 1 }, keepDims: outputShape.Length == 2);
                    }
                }
                DirectGpuTensorEngine.CopyResultInto(eng, loss, result);
            },
            BackwardFunctions<float>.FusedLinearMseLossBackward, null)
        {
            RecordingEngine = linear.RecordingEngine,
        };
        fused = node;
        PatternHelpers.SetLazySource(node);
        alreadyRemoved.Add(sub);
        alreadyRemoved.Add(square);
        alreadyRemoved.Add(mean);
        return true;
    }

    // The targets are a graph input, or the output of a node ahead of the head (the fused node runs in its place).
    private static bool ProducedBefore(IReadOnlyList<ILazyNode> nodes, Tensor<float> tensor, int index)
    {
        if (tensor.LazySource is not ILazyNode source || source.IsRealized) return true;
        for (int i = 0; i < index; i++)
            if (ReferenceEquals(nodes[i], source)) return true;
        return false;
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
