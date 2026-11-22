namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>Compiles a single forward dispatch without recursively realizing its producers.</summary>
internal interface ICompiledForwardNode
{
    Action<IEngine> CompileForwardAction();
    void DetachDeviceBindings(HashSet<DirectGpu.IGpuBuffer> released, HashSet<IntPtr> handles, HashSet<object> retained);
}

/// <summary>Protects persistent or in-place device state from post-step intermediate detachment.</summary>
internal interface IDeviceStateProtectionNode
{
    void ProtectDeviceState(HashSet<object> keys);
}

/// <summary>A node whose leaf inputs are not of the plan's element type, so the plan's leaf walk cannot see them.</summary>
internal interface IForeignLeafSource
{
    /// <summary>Migrates each leaf input another backend placed to <paramref name="consumer"/> (CrossBackendTensors).</summary>
    void ResolveForeignLeaves(DirectGpu.IDirectGpuBackend consumer);
}

/// <summary>Describes the gradient boundary of a node crossing element types.</summary>
internal interface ICrossTypeNodeMetadata
{
    Type InputElementType { get; }
    Type OutputElementType { get; }
    bool HasBackward { get; }
    object? DifferentiableInputTensor { get; }
}

/// <summary>
/// Type-erased interface for lazy computation nodes. Allows LazyTensorScope
/// to manage nodes across different element types (float, double, etc.)
/// without generic type parameters on the scope itself.
/// </summary>
internal interface ILazyNode
{
    /// <summary>The operation type for pattern matching during fusion.</summary>
    LazyNodeType OpType { get; }

    /// <summary>The operation's name as recorded (used in diagnostics and objective signatures).</summary>
    string OpName { get; }

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
    /// Whether one of the node's inputs is <paramref name="tensor"/> or shares its storage (a view of it, or it a view of
    /// the input): what a later in-place write of <paramref name="tensor"/> must run after.
    /// </summary>
    bool ReadsStorageOf(object tensor);

    /// <summary>
    /// The engine that recorded this node: the scope's bound engine (the one the op was invoked on), stamped when the
    /// scope publishes the node. A read of an unrealized output realizes it here - defaulting to the process-wide
    /// AiDotNetEngine.Current realized a CpuEngine model on whatever engine happened to be global.
    /// </summary>
    IEngine RecordingEngine { get; set; }

    /// <summary>
    /// Whether the node was recorded inside a <see cref="Autodiff.NoGradScope{T}"/>. Such a node keeps its forward (a
    /// replay refreshes its value) but carries no backward, so a compiled plan stops gradient at it exactly where the
    /// eager tape records nothing. A training fusion pass never rewrites it: the fused node would carry a backward.
    /// </summary>
    bool RecordedUnderNoGrad { get; set; }

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

/// <summary>An in-place write ordered after the earlier readers of its target (see <c>LazyTensorScope.RecordInPlace</c>).</summary>
internal interface IInPlaceReadOrdering
{
    /// <summary>The nodes that read the write's target before it was recorded; empty when none did.</summary>
    IReadOnlyList<ILazyNode> EarlierReaders { get; }
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
