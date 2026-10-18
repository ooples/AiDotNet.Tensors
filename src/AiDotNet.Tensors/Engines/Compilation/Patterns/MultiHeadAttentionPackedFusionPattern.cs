// Copyright (c) AiDotNet. All rights reserved.
// Built-in fusion: three column blocks of one packed QKV projection feeding a head-interleaved attention -> one packed
// attention node that reads the blocks in place.
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the in-projection split of a Transformer encoder layer as it is usually written (torch's
/// <c>in_proj</c> then <c>chunk(3)</c>): <c>Narrow(qkv, -1, s, w)</c> -> <c>Reshape([B, S, w])</c> for each of query, key
/// and value, all three feeding one recorded fused <c>MultiHeadAttentionCore</c>, and emits one
/// <c>MultiHeadAttentionPacked</c> node (<see cref="DirectGpuTensorEngine.BuildPackedAttentionNode"/>) whose kernels
/// read q, k and v straight out of the [B * S, 3w] projection and write their gradients straight into its gradient.
/// </summary>
/// <remarks>
/// Fires at the first block's Narrow (the pass emits nodes in order, and the projection precedes it). The three blocks
/// must exactly cover the projection's columns (starts 0, w and 2w in any order), every Narrow and Reshape must have
/// the attention as its only consumer, and the attention must be self-attention recorded on the GPU engine.
/// </remarks>
internal sealed class MultiHeadAttentionPackedFusionPattern : IFusionPattern
{
    public string Name => "MultiHeadAttentionPacked";

    /// <inheritdoc />
    public bool IsDifferentiable => true;

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        if (!IsLastAxisNarrow(nodes[index], out var first, out _, out int width)) return false;
        var packed = first.Input0;
        if (packed.Rank != 2 || packed._shape[1] != 3 * width) return false;

        // The first block's reshape, then the attention consuming it.
        if (!SingleConsumer(consumerCounts, first)) return false;
        var firstReshape = PatternHelpers.FindNextFloat(nodes, index + 1, nodes.Count - 1, alreadyRemoved,
            n => PatternHelpers.ConsumesOutput(n, first));
        if (firstReshape is null || firstReshape.OpType != LazyNodeType.Reshape || !SingleConsumer(consumerCounts, firstReshape))
            return false;
        var attention = PatternHelpers.FindNextFloat(nodes, index + 1, nodes.Count - 1, alreadyRemoved,
            n => PatternHelpers.ConsumesOutput(n, firstReshape));
        if (attention is null || attention.OpType != LazyNodeType.Custom || attention.OpName != "MultiHeadAttentionCore"
            || attention.InputCount != 3 || attention.Input1 is null || attention.Input2 is null)
            return false;

        // Each of query, key and value: Reshape(Narrow(packed, last axis, start, width)), used only here.
        var removed = new List<ILazyNode> { attention };
        var starts = new int[3];
        var operands = new[] { attention.Input0, attention.Input1, attention.Input2 };
        for (int i = 0; i < 3; i++)
        {
            if (operands[i].LazySource is not LazyNode<float> { OpType: LazyNodeType.Reshape } reshape
                || alreadyRemoved.Contains(reshape) || !SingleConsumer(consumerCounts, reshape)
                || reshape.Output._shape.Length != 3
                || !IsLastAxisNarrow(reshape.Input0.LazySource, out var narrow, out int start, out int length)
                || alreadyRemoved.Contains(narrow) || !SingleConsumer(consumerCounts, narrow)
                || !ReferenceEquals(narrow.Input0, packed) || length != width)
                return false;
            starts[i] = start;
            removed.Add(reshape);
            removed.Add(narrow);
        }
        // The blocks exactly cover the projection's columns.
        var covered = new bool[3];
        foreach (int s in starts)
        {
            if (s % width != 0 || s / width > 2 || covered[s / width]) return false;
            covered[s / width] = true;
        }

        var node = DirectGpuTensorEngine.BuildPackedAttentionNode(attention, packed, starts[0], starts[1], starts[2]);
        if (node is null) return false;
        foreach (var r in removed)
        {
            if (r is LazyNode<float> f && !ReferenceEquals(f, attention)) f.Output.LazySource = null;
            alreadyRemoved.Add(r);
        }
        PatternHelpers.SetLazySource(node);
        fused = node;
        return true;
    }

    private static bool SingleConsumer(IReadOnlyDictionary<ILazyNode, int> consumerCounts, ILazyNode node)
        => consumerCounts.TryGetValue(node, out int count) && count == 1;

    // A float column block of its input: Narrow on the last axis (saves { dim, start, length }), or the TensorSlice the
    // GPU engine records that Narrow as (saves { start[] }, output shape = lengths) with every other axis whole.
    private static bool IsLastAxisNarrow(ILazyNode? node, out LazyNode<float> narrow, out int start, out int length)
    {
        narrow = null!;
        start = 0; length = 0;
        if (node is not LazyNode<float> { OpType: LazyNodeType.Custom } candidate
            || candidate.RecordingEngine is not DirectGpuTensorEngine || candidate.SavedState is not { } state)
            return false;
        int rank = candidate.Input0.Rank;
        if (candidate.OpName == "Narrow")
        {
            if (state.Length < 3 || state[0] is not int dim || state[1] is not int s || state[2] is not int l
                || (dim < 0 ? dim + rank : dim) != rank - 1)
                return false;
            start = s; length = l;
        }
        else if (candidate.OpName == "TensorSlice")
        {
            if (state.Length < 1 || state[0] is not int[] starts || starts.Length != rank
                || candidate.OutputShape.Length != rank)
                return false;
            for (int d = 0; d < rank - 1; d++)
                if (starts[d] != 0 || candidate.OutputShape[d] != candidate.Input0._shape[d]) return false;
            start = starts[rank - 1]; length = candidate.OutputShape[rank - 1];
        }
        else
        {
            return false;
        }
        narrow = candidate;
        return true;
    }
}
