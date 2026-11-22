// Copyright (c) CrowdTrainer LLC. All rights reserved.

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

    private readonly bool _forTraining;
    private readonly HashSet<ILazyNode>? _observableOutputs;

    /// <summary>Creates a fusion pass that preserves training and observable-output contracts.</summary>
    /// <param name="forTraining">True when compiling a TRAINING plan: inference-only patterns (whose fused node has
    /// no backward) are skipped, so the matched ops keep their backward steps. Measured before this: a matmul by a
    /// sparse constant selection matrix was rewritten to FusedSparseLinear inside a training plan and every gradient
    /// upstream of it came out exactly zero (the compiled HRE attention never trained its Q/K/V projections).</param>
    /// <param name="observableOutputs">Nodes whose outputs must remain independently observable after execution.</param>
    public CpuFusionPass(bool forTraining = false, HashSet<ILazyNode>? observableOutputs = null)
    {
        _forTraining = forTraining;
        _observableOutputs = observableOutputs;
    }

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

        // An in-place write and the earlier readers of its target are never fused: the write's ordering edge names those
        // readers, and a fused node that replaced one (recorded at order 0, so it resolved the target's producer to the
        // later write) ran after the write and read the updated value. Patterns skip every node in the set they are
        // handed, so the blocked nodes start in it and are told apart from fused-away ones below.
        // An observable output is read after the plan runs. Inference: count that read as one more consumer - every
        // pattern refuses to absorb an interior node with another consumer, so an observable node can still END a fused
        // chain (the fused node writes the chain's last output tensor, the one the caller observes) but is never fused
        // away. Training keeps every observable node out of fusion: a retained output that ends a fused linear chain
        // is not left holding the step's values there.
        var blocked = new HashSet<ILazyNode>();
        if (_observableOutputs is not null)
            foreach (var observable in _observableOutputs)
            {
                if (_forTraining) blocked.Add(observable);
                else consumerCounts[observable] = consumerCounts.TryGetValue(observable, out int readers) ? readers + 1 : 1;
            }
        foreach (var node in nodes)
        {
            // Already run (a nested scope's write realized it as its earlier reader): a fused replacement would be a new,
            // unrealized node that recomputes from inputs the write has since changed.
            if (node.IsRealized) blocked.Add(node);
            // Recorded under a NoGradScope: the node has no backward on purpose, and a fused replacement would bring
            // one back (FusedLinear's), sending gradient into what the forward ran detached. BatchNorm's auxiliary
            // batch-statistic nodes carry no backward because statistics are non-differentiable outputs absorbed by
            // BatchNorm+ReLU fusion.
            if (_forTraining && (node.RecordedUnderNoGrad
                || (node is ISignableLazyNode { BackwardDelegate: null }
                    && node.OpName is not ("BatchNormBatchMean" or "BatchNormBatchVariance"))
                || node is ISignableLazyNode { DifferentiableInputIndices: not null })) blocked.Add(node);
            if (node is not IInPlaceReadOrdering ordered || ordered.EarlierReaders.Count == 0) continue;
            blocked.Add(node);
            foreach (var reader in ordered.EarlierReaders) blocked.Add(reader);
        }

        var result = new List<ILazyNode>(nodes.Count);
        var removed = new HashSet<ILazyNode>(blocked);
        var patterns = FusionPatternRegistry.Patterns;

        for (int i = 0; i < nodes.Count; i++)
        {
            if (blocked.Contains(nodes[i]))
            {
                result.Add(nodes[i]);
                continue;
            }
            if (removed.Contains(nodes[i]))
                continue;

            // Try each pattern in registration order. First match wins —
            // patterns register in priority order (most specific first).
            bool didFuse = false;
            for (int p = 0; p < patterns.Count; p++)
            {
                if (_forTraining && !patterns[p].IsDifferentiable)
                    continue;
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
