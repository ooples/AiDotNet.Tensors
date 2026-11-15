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

    /// <summary>
    /// The node's position in the order its trace recorded it (1, 2, ...; 0 for a node no trace recorded, such as one a
    /// fusion pass built). A node reads each input as it was when the node was recorded: an in-place write recorded
    /// later is not its producer (see <see cref="LazyNodeOrder.ProducerAsOf"/>).
    /// </summary>
    long RecordOrder { get; set; }

    /// <summary>For an in-place write: the producer of its target before the write (null when the target came from
    /// outside the trace). For any other node: null.</summary>
    ILazyNode? PriorWriter { get; }

    /// <summary>Whether the node writes one of its inputs in place.</summary>
    bool WritesInPlace { get; }

    /// <summary>
    /// The engine that recorded this node: the scope's bound engine (the one the op was invoked on), stamped when the
    /// scope publishes the node. A read of an unrealized output realizes it here - defaulting to the process-wide
    /// AiDotNetEngine.Current realized a CpuEngine model on whatever engine happened to be global.
    /// </summary>
    IEngine RecordingEngine { get; set; }


    /// <summary>Execute this node, materializing its output tensor's data.</summary>
    void Realize(IEngine engine);

    /// <summary>Get input nodes as type-erased array (for graph traversal).</summary>
    ILazyNode[] GetInputNodes();

    /// <summary>Clears the LazySource reference on the output tensor after realization.</summary>
    void ClearOutputLazySource();

    /// <summary>
    /// When this node writes in place to a tensor that no node of the trace produced (a persistent tensor: a running
    /// statistic, an optimizer moment), adds to <paramref name="restores"/> a copy of that tensor and how to put it
    /// back, and returns true; false for any other node. <paramref name="saved"/> holds the tensors already copied.
    /// </summary>
    bool SavePersistentWrite(HashSet<object> saved, List<Action> restores);

    /// <summary>
    /// Detaches this node from its output when its trace is abandoned, restoring any producer
    /// that the node temporarily replaced.
    /// </summary>
    void RestoreOutputLazySource();

    /// <summary>Adds every tensor storage structurally retained by this node.</summary>
    void AddStorageLeases(TensorStorageLeaseSet leases);
}

/// <summary>Resolves a node's producers by record order.</summary>
internal static class LazyNodeOrder
{
    /// <summary>
    /// The producer a node recorded at <paramref name="order"/> read: <paramref name="latest"/> (the input's current
    /// <c>LazySource</c>) stepped back past every in-place write recorded after the reader. Without this, a node that
    /// read a tensor before a later in-place update of it (an MoE reading its selection bias, then the bias nudged
    /// from the MoE's loads) took the update as its producer: a cycle, and the topological sort dropped every node on
    /// it from the compiled plan.
    /// </summary>
    internal static ILazyNode? ProducerAsOf(ILazyNode? latest, long order)
    {
        if (order <= 0) return latest;
        while (latest is { RecordOrder: > 0, WritesInPlace: true } writer && writer.RecordOrder > order)
            latest = writer.PriorWriter;
        return latest;
    }
}
