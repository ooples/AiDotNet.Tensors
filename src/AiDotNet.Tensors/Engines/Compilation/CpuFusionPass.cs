using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// CPU fusion pass that pattern-matches sequences of lazy nodes and replaces them
/// with fused operations. Uses the same fusion patterns as KernelFusionManager
/// but applied to the lazy computation graph.
///
/// Current fusion patterns:
/// - MatMul + BiasAdd → FusedLinear (single BLAS call)
/// - MatMul + BiasAdd + ReLU → FusedLinearReLU
/// - MatMul + BiasAdd + Sigmoid → FusedLinearSigmoid
/// - MatMul + BiasAdd + GELU → FusedLinearGELU
///
/// Future patterns (Phase 3+):
/// - Conv2D + BatchNorm + ReLU → FusedConvBNReLU
/// - Elementwise chains (Add + Mul → FMA)
/// </summary>
internal sealed class CpuFusionPass : ILazyGraphOptimizationPass
{
    public string Name => "CpuFusion";

    public List<ILazyNode> Run(List<ILazyNode> nodes)
    {
        // Build consumer-count map: how many downstream nodes use each node's output
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

        for (int i = 0; i < nodes.Count; i++)
        {
            if (removed.Contains(nodes[i]))
                continue;

            // Try to fuse MatMul + Add + Activation → FusedLinear
            if (TryFuseLinear(nodes, i, consumerCounts, removed, out var fused))
            {
                if (fused != null)
                    result.Add(fused);
                continue;
            }

            result.Add(nodes[i]);
        }

        return result;
    }

    /// <summary>
    /// Tries to fuse a MatMul node with a subsequent BiasAdd and optional activation
    /// into a single FusedLinear node. Returns true if the current node was consumed.
    /// </summary>
    private static bool TryFuseLinear(
        List<ILazyNode> nodes, int index,
        Dictionary<ILazyNode, int> consumerCounts,
        HashSet<ILazyNode> removed,
        out ILazyNode? fused)
    {
        fused = null;
        var matmul = nodes[index];
        if (matmul.OpType != LazyNodeType.MatMul)
            return false;

        // MatMul output must have exactly one consumer (the Add)
        if (consumerCounts.ContainsKey(matmul) && consumerCounts[matmul] > 1)
            return false;

        // Look ahead for Add (bias) consuming this MatMul's output
        ILazyNode? addNode = null;
        int addIndex = -1;
        for (int j = index + 1; j < nodes.Count && j <= index + 3; j++)
        {
            if (removed.Contains(nodes[j]))
                continue;
            if ((nodes[j].OpType == LazyNodeType.Add || nodes[j].OpType == LazyNodeType.BroadcastAdd)
                && IsConsumerOf(nodes[j], matmul))
            {
                addNode = nodes[j];
                addIndex = j;
                break;
            }
        }

        if (addNode == null)
            return false;

        // Check if Add output has exactly one consumer (potential activation)
        FusedActivationType activation = FusedActivationType.None;
        ILazyNode? activationNode = null;
        int activationIndex = -1;

        if (!consumerCounts.ContainsKey(addNode) || consumerCounts[addNode] <= 1)
        {
            for (int j = addIndex + 1; j < nodes.Count && j <= addIndex + 2; j++)
            {
                if (removed.Contains(nodes[j]))
                    continue;
                if (IsConsumerOf(nodes[j], addNode))
                {
                    // OCP-compliant: lookup via ActivationRegistry instead of switch
                    ActivationRegistry.TryGetActivationType(nodes[j].OpType, out activation);

                    if (activation != FusedActivationType.None)
                    {
                        activationNode = nodes[j];
                        activationIndex = j;
                        break;
                    }
                }
            }
        }

        // Build fused node using the typed FusedLinearNodeBuilder
        fused = FusedLinearNodeBuilder.TryBuild(matmul, addNode, activationNode, activation);
        if (fused == null)
            return false;

        // Mark consumed nodes for removal
        removed.Add(matmul);
        removed.Add(addNode);
        if (activationNode != null)
            removed.Add(activationNode);

        return true;
    }

    private static bool IsConsumerOf(ILazyNode consumer, ILazyNode producer)
    {
        foreach (var input in consumer.GetInputNodes())
        {
            if (ReferenceEquals(input, producer))
                return true;
        }
        return false;
    }

    /// <summary>
    /// Builds typed FusedLinear LazyNodes. Separate class to handle the generic type parameter.
    /// </summary>
    private static class FusedLinearNodeBuilder
    {
        internal static ILazyNode? TryBuild(
            ILazyNode matmul, ILazyNode addNode,
            ILazyNode? activationNode, FusedActivationType activation)
        {
            // Try float
            if (matmul is LazyNode<float> matmulF && addNode is LazyNode<float> addF)
                return BuildTyped(matmulF, addF, activationNode as LazyNode<float>, activation);
            // Try double
            if (matmul is LazyNode<double> matmulD && addNode is LazyNode<double> addD)
                return BuildTyped(matmulD, addD, activationNode as LazyNode<double>, activation);
            return null;
        }

        private static LazyNode<T>? BuildTyped<T>(
            LazyNode<T> matmul, LazyNode<T> add,
            LazyNode<T>? activationNode, FusedActivationType activation)
        {
            // Determine the final output: activation output if present, else add output
            var finalOutput = activationNode != null ? activationNode.Output : add.Output;

            // Inputs: input tensor, weight tensor, bias tensor
            var input = matmul.Input0;
            var weights = matmul.Input1 ?? matmul.Input0;

            // The bias is the "other" input of the add node (the one that isn't the matmul output)
            Tensor<T> bias;
            if (ReferenceEquals(add.Input0, matmul.Output))
                bias = add.Input1 ?? add.Input0;
            else
                bias = add.Input0;

            // Only fuse if bias is 1D (broadcast bias add pattern).
            // An elementwise add with same-shaped tensors is NOT a bias add.
            if (bias._shape.Length != 1)
                return null;

            // Verify bias dim matches matmul output columns
            var matmulOutputCols = matmul.OutputShape[matmul.OutputShape.Length - 1];
            if (bias._shape[0] != matmulOutputCols)
                return null;

            var nodeType = ActivationRegistry.GetFusedLinearNodeType(activation);

            var capturedInput = input;
            var capturedWeights = weights;
            var capturedBias = bias;
            var capturedActivation = activation;

            object[]? savedState = activation != FusedActivationType.None
                ? new object[] { activation }
                : null;

            var fusedNode = new LazyNode<T>(
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

            // Clear LazySource on removed intermediates. The fused node only writes
            // into finalOutput — intermediate buffers are never populated, so leaving
            // a LazySource would auto-materialize into a buffer the fused node doesn't
            // write to, returning uninitialized data. Clearing is safer: accessing an
            // intermediate will return its (uninitialized) buffer rather than silently
            // running the wrong realization.
            matmul.Output.LazySource = null;
            add.Output.LazySource = null;

            return fusedNode;
        }
    }
}
