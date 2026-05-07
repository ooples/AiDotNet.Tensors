// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// CPU fusion pass that pattern-matches sequences of lazy nodes and
/// replaces them with fused operations.
///
/// <para>OCP-compliant: this class iterates a registry of
/// <see cref="IFusionPattern"/> instances; adding a new fusion pattern
/// is a new <see cref="IFusionPattern"/> implementation registered with
/// <see cref="FusionPatternRegistry"/>, with no modification to this
/// pass.</para>
///
/// <para>Patterns registered out of the box (see
/// <see cref="FusionPatternRegistry"/> static constructor):</para>
/// <list type="bullet">
///   <item><c>LinearFusionPattern</c> — MatMul + BiasAdd (+ optional activation) → FusedLinear[ReLU/GELU/Sigmoid]</item>
///   <item><c>LoRAFusionPattern</c> (issue #301) — MatMul + MatMul + MultiplyScalar + Add → FusedLoRA. Activates only when the chain matches LoRA's exact 4-node shape with single-consumer fan-out at each intermediate.</item>
///   <item><c>SparseLinearFusionPattern</c> (issue #301) — registered for the dense-mask CSR pattern; activates when the upstream graph emits the sentinel sparse layout.</item>
///   <item><c>DDIMStepFusionPattern</c> (issue #301) — registered for the DDIM sampler-update chain; activates when the (noiseScale, subtract, divide, x0Scale, …) sequence is detected.</item>
/// </list>
///
/// <para>A registered pattern's <c>TryFuse</c> may legitimately return <c>false</c>
/// for any input that does not match its specific shape; emission depends on
/// the runtime graph structure, not on whether the pattern is registered.</para>
/// </summary>
internal sealed class CpuFusionPass : ILazyGraphOptimizationPass
{
    public string Name => "CpuFusion";

    public List<ILazyNode> Run(List<ILazyNode> nodes)
    {
        // Build consumer-count map: how many downstream nodes use each node's output.
        var consumerCounts = new Dictionary<ILazyNode, int>();
        foreach (var node in nodes)
        {
            foreach (var inputNode in node.GetInputNodes())
            {
                if (consumerCounts.ContainsKey(inputNode))
                    consumerCounts[inputNode]++;
                else
                    consumerCounts[inputNode] = 1;
            }
        }

        var result = new List<ILazyNode>(nodes.Count);
        var removed = new HashSet<ILazyNode>();
        var patterns = FusionPatternRegistry.Patterns;

        for (int i = 0; i < nodes.Count; i++)
        {
            if (removed.Contains(nodes[i]))
                continue;

            // Try each pattern in registration order. First match wins —
            // patterns register in priority order (most specific first).
            bool didFuse = false;
            for (int p = 0; p < patterns.Count; p++)
            {
                if (patterns[p].TryFuse(nodes, i, consumerCounts, removed, out var fused))
                {
                    if (fused != null)
                        result.Add(fused);
                    didFuse = true;
                    break;
                }
            }
            if (!didFuse)
                result.Add(nodes[i]);
        }

        return result;
    }
}
