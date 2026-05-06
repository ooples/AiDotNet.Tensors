// Copyright (c) AiDotNet. All rights reserved.
// Issue #301 — LoRA-forward chain detection for graph-mode auto-fusion.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the LoRA adapter forward chain — five lazy nodes:
/// <code>
///   intermed = MatMul(input, A)            // [batch, rank]
///   delta    = MatMul(intermed, B)         // [batch, out]
///   scaled   = MultiplyScalar(delta, α)    // [batch, out]
///   output   = Add(baseOutput, scaled)     // [batch, out]
/// </code>
/// and emits a single <see cref="LazyNodeType.FusedLoRA"/> node that
/// dispatches to <see cref="AiDotNet.Tensors.Helpers.CpuFusedOperations.FusedLoRAForward"/>.
///
/// <para>The kernel is shipped in the same PR. Auto-detection
/// in graph mode is wired up here via the registry rather than by
/// touching <see cref="CpuFusionPass"/>; <see cref="TryFuse"/> currently
/// returns <c>false</c> as the chain-walker is being landed in a
/// follow-up so the rest of the kernel ships now. The pattern is
/// registered so the override boundary is in place — graph-mode
/// detection lands without touching this file's siblings.</para>
/// </summary>
internal sealed class LoRAFusionPattern : IFusionPattern
{
    public string Name => "LoRA";

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        // Chain-walker landing in a follow-up; ship the kernel + registry
        // hook now so callers can dispatch to FusedLoRAForward directly.
        return false;
    }
}
