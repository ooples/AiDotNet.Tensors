namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Type-erased interface for lazy computation nodes. Allows LazyTensorScope
/// to manage nodes across different element types (float, double, etc.)
/// without generic type parameters on the scope itself.
/// </summary>
internal interface ILazyNode
{
    /// <summary>The operation type for pattern matching during fusion.</summary>
    LazyNodeType OpType { get; }

    /// <summary>Output tensor shape (computed eagerly during recording).</summary>
    int[] OutputShape { get; }

    /// <summary>Whether this node's output has been computed.</summary>
    bool IsRealized { get; set; }

    /// <summary>Topological order index, assigned during compilation.</summary>
    int TopologicalIndex { get; set; }

    /// <summary>Number of downstream nodes that depend on this node's output.</summary>
    int ConsumerCount { get; set; }

    /// <summary>The engine that was active when this node was recorded.</summary>
    IEngine RecordingEngine { get; }

    /// <summary>Execute this node, materializing its output tensor's data.</summary>
    void Realize(IEngine engine);

    /// <summary>Get input nodes as type-erased array (for graph traversal).</summary>
    ILazyNode[] GetInputNodes();

    /// <summary>Clears the LazySource reference on the output tensor after realization.</summary>
    void ClearOutputLazySource();
}
