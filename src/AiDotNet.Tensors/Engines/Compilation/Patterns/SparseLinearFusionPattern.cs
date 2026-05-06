// Copyright (c) AiDotNet. All rights reserved.
// Issue #301 — sparse-linear chain detection for graph-mode auto-fusion.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the dense-with-mask sparse-linear pattern and rewrites it
/// to <see cref="LazyNodeType.FusedSparseLinear"/>. The CSR-format
/// kernel ships now (see
/// <see cref="AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedSparseLinear"/>);
/// the chain-walker that detects mask sentinels in graph mode is
/// landing in a follow-up — the pattern is registered now so the
/// registry slot is reserved.
/// </summary>
internal sealed class SparseLinearFusionPattern : IFusionPattern
{
    public string Name => "SparseLinear";

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        return false;
    }
}
