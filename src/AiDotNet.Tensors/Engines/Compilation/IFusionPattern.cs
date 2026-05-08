// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A graph-rewriting fusion pattern. Each implementation matches a
/// specific shape of consecutive lazy nodes and emits a single fused
/// replacement node. The <see cref="CpuFusionPass"/> iterates a
/// registry of these patterns; adding a new pattern is a new
/// implementation + one registration in <see cref="FusionPatternRegistry"/>
/// — no modifications to the pass itself.
/// </summary>
internal interface IFusionPattern
{
    /// <summary>
    /// Short name used for diagnostics (e.g. "Linear", "LoRA").
    /// </summary>
    string Name { get; }

    /// <summary>
    /// Attempts to fuse a chain of nodes starting at <paramref name="index"/>.
    /// Returns true on a successful match. On match the implementation must:
    /// <list type="bullet">
    ///   <item>Set <paramref name="fused"/> to the replacement node (or
    ///   <c>null</c> if the fusion produced nothing visible — uncommon).</item>
    ///   <item>Add every node it consumed to <paramref name="alreadyRemoved"/>
    ///   so subsequent iterations skip them.</item>
    ///   <item>Rewire the LazySource on the final-output tensor of the chain
    ///   to the new fused node (so downstream consumers find it).</item>
    /// </list>
    /// Returns false otherwise; output parameters are undefined in that case.
    /// </summary>
    bool TryFuse(
        IReadOnlyList<ILazyNode> nodes,
        int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused);
}
