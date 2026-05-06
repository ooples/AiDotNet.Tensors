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
/// pass. The built-in patterns registered out of the box cover:</para>
/// <list type="bullet">
///   <item>MatMul + BiasAdd (+ optional activation) → FusedLinear[ReLU/GELU/Sigmoid]</item>
///   <item>MatMul + MatMul + MultiplyScalar + Add → FusedLoRA  (issue #301)</item>
///   <item>SparseLinear pattern via dense-mask sentinel → FusedSparseLinear (issue #301)</item>
///   <item>DDIM sampler-update chain → FusedDDIMStep  (issue #301)</item>
/// </list>
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
