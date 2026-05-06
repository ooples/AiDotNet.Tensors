// Copyright (c) AiDotNet. All rights reserved.
// Built-in fusion: MatMul + BiasAdd (+ optional activation) → FusedLinear.

using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Matches the canonical Linear / Dense-layer chain:
/// <c>MatMul(x, W) → BroadcastAdd(_, bias) → optional [ReLU|GELU|Sigmoid|...]</c>
/// and emits a single FusedLinear node that calls
/// <see cref="IEngine.FusedLinear{T}"/>.
/// </summary>
internal sealed class LinearFusionPattern : IFusionPattern
{
    public string Name => "Linear";

    public bool TryFuse(
        IReadOnlyList<ILazyNode> nodes, int index,
        IReadOnlyDictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> alreadyRemoved,
        out ILazyNode? fused)
    {
        fused = null;
        var matmul = nodes[index];
        if (matmul.OpType != LazyNodeType.MatMul)
            return false;

        // MatMul output must have exactly one consumer (the Add).
        if (consumerCounts.TryGetValue(matmul, out var matmulFanOut) && matmulFanOut > 1)
            return false;

        // Look ahead for Add (bias) consuming this MatMul's output.
        ILazyNode? addNode = null;
        int addIndex = -1;
        for (int j = index + 1; j < nodes.Count && j <= index + 3; j++)
        {
            if (alreadyRemoved.Contains(nodes[j])) continue;
            if ((nodes[j].OpType == LazyNodeType.Add || nodes[j].OpType == LazyNodeType.BroadcastAdd)
                && PatternHelpers.IsConsumerOf(nodes[j], matmul))
            {
                addNode = nodes[j];
                addIndex = j;
                break;
            }
        }
        if (addNode == null) return false;

        // Optional activation: only consume if Add has exactly one consumer
        // (otherwise external readers would be stranded).
        FusedActivationType activation = FusedActivationType.None;
        ILazyNode? activationNode = null;

        if (!consumerCounts.TryGetValue(addNode, out var addFanOut) || addFanOut <= 1)
        {
            for (int j = addIndex + 1; j < nodes.Count && j <= addIndex + 2; j++)
            {
                if (alreadyRemoved.Contains(nodes[j])) continue;
                if (PatternHelpers.IsConsumerOf(nodes[j], addNode))
                {
                    ActivationRegistry.TryGetActivationType(nodes[j].OpType, out activation);
                    if (activation != FusedActivationType.None)
                    {
                        if (consumerCounts.TryGetValue(nodes[j], out var actFan) && actFan > 1)
                            break; // shared activation — leave it standalone
                        activationNode = nodes[j];
                        break;
                    }
                }
            }
        }

        fused = FusedLinearNodeBuilder.TryBuild(matmul, addNode, activationNode, activation);
        if (fused == null) return false;

        PatternHelpers.SetLazySource(fused);
        alreadyRemoved.Add(matmul);
        alreadyRemoved.Add(addNode);
        if (activationNode != null) alreadyRemoved.Add(activationNode);
        return true;
    }

    /// <summary>Builds typed FusedLinear LazyNodes (handles float / double).</summary>
    private static class FusedLinearNodeBuilder
    {
        internal static ILazyNode? TryBuild(
            ILazyNode matmul, ILazyNode addNode,
            ILazyNode? activationNode, FusedActivationType activation)
        {
            if (matmul is LazyNode<float> mF && addNode is LazyNode<float> aF)
                return BuildTyped(mF, aF, activationNode as LazyNode<float>, activation);
            if (matmul is LazyNode<double> mD && addNode is LazyNode<double> aD)
                return BuildTyped(mD, aD, activationNode as LazyNode<double>, activation);
            return null;
        }

        private static LazyNode<T>? BuildTyped<T>(
            LazyNode<T> matmul, LazyNode<T> add,
            LazyNode<T>? activationNode, FusedActivationType activation)
        {
            var finalOutput = activationNode != null ? activationNode.Output : add.Output;

            var input = matmul.Input0;
            var weights = matmul.Input1 ?? matmul.Input0;

            Tensor<T> bias;
            if (ReferenceEquals(add.Input0, matmul.Output))
                bias = add.Input1 ?? add.Input0;
            else
                bias = add.Input0;

            // Only fuse if bias is 1D (broadcast bias add pattern).
            if (bias._shape.Length != 1) return null;

            var matmulOutputCols = matmul.OutputShape[matmul.OutputShape.Length - 1];
            if (bias._shape[0] != matmulOutputCols) return null;

            // Clear LazySource on intermediate outputs so auto-materialize
            // doesn't try to realize the now-removed nodes.
            matmul.Output.LazySource = null;
            add.Output.LazySource = null;

            var nodeType = ActivationRegistry.GetFusedLinearNodeType(activation);

            var capturedInput = input;
            var capturedWeights = weights;
            var capturedBias = bias;
            var capturedActivation = activation;

            object[]? savedState = activation != FusedActivationType.None
                ? new object[] { activation }
                : null;

            return new LazyNode<T>(
                nodeType,
                "FusedLinear",
                new[] { capturedInput, capturedWeights, capturedBias },
                finalOutput,
                (eng, output) =>
                {
                    var eager = eng.FusedLinear(capturedInput, capturedWeights, capturedBias, capturedActivation);
                    eager.AsSpan().CopyTo(output.AsWritableSpan());
                },
                BackwardFunctions<T>.FusedLinearWithActivationBackward,
                savedState);
        }
    }
}
