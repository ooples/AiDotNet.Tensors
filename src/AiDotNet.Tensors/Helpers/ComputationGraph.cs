namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Records a sequence of tensor operations during a forward pass for later
/// optimization and compiled execution. This is the graph capture component
/// of the JIT compilation pipeline (similar to torch.compile / tf.function).
/// </summary>
/// <remarks>
/// <para>
/// Usage:
/// <code>
/// var graph = new ComputationGraph();
///
/// // First forward pass: capture mode
/// graph.BeginCapture();
/// int input = graph.RecordInput(new[] { 1, 3, 256, 256 });
/// int conv1 = graph.RecordOp(OpType.Conv2D, [input], new[] { 1, 64, 256, 256 },
///     new OpParams { Stride = 1, Padding = 1 });
/// int relu1 = graph.RecordOp(OpType.ReLU, [conv1], new[] { 1, 64, 256, 256 });
/// graph.RecordOutput(relu1);
/// graph.EndCapture();
///
/// // Optimize the graph
/// var plan = graph.Optimize();
///
/// // Subsequent forward passes: execute compiled graph
/// var workspace = plan.CreateWorkspace&lt;float&gt;();
/// // ... execute operations using workspace slots
/// </code>
/// </para>
/// </remarks>
public sealed class ComputationGraph
{
    private readonly List<GraphNode> _nodes = new();
    private readonly List<int> _inputNodeIds = new();
    private readonly List<int> _outputNodeIds = new();
    private bool _capturing;
    private bool _finalized;

    /// <summary>Operation types supported by the computation graph.</summary>
    public enum OpType
    {
        // Element-wise arithmetic
        Add, Subtract, Multiply, Divide, Negate,
        AddScalar, MultiplyScalar, DivideScalar,

        // Activations
        ReLU, Sigmoid, Tanh, GELU, Mish, Swish, LeakyReLU, SiLU, ELU, Softmax,

        // Element-wise math
        Exp, Log, Sqrt, Abs, Pow, Sin, Cos,

        // Convolution
        Conv2D, Conv2DInto,

        // Normalization
        BatchNorm, LayerNorm, GroupNorm,

        // Fused operations
        FusedConv2DBiasActivation, FusedGroupNormActivation,
        FusedLinearBiasActivation,

        // Pooling
        MaxPool2D, AvgPool2D,

        // Linear algebra
        MatMul, BatchMatMul, Linear,

        // Reduction
        Sum, Mean, Max, Min,

        // Reshape
        Reshape, Transpose, Permute, Flatten,

        // Attention
        ScaledDotProductAttention, MultiHeadAttention, FlashAttention,

        // Scatter / Gather (sparse and attention operations)
        ScatterAdd, ScatterMean, ScatterMax, ScatterSoftmax,
        GatherNd, ScatterNd,

        // Other
        Concat, Split, Residual, Upsample, Interpolate,
        Embedding, Dropout, Custom
    }

    /// <summary>Parameters for an operation (convolution stride, padding, etc.).</summary>
    public sealed class OpParams
    {
        public int Stride { get; set; } = 1;
        public int Padding { get; set; }
        public int Dilation { get; set; } = 1;
        public int Groups { get; set; } = 1;
        public double Epsilon { get; set; } = 1e-5;
        public double Alpha { get; set; } = 0.01; // LeakyReLU slope
        public int Axis { get; set; } = -1;
        public FusedActivationType FusedActivation { get; set; } = FusedActivationType.None;
        public string? CustomName { get; set; }
    }

    /// <summary>Activation types for fused operations.</summary>
    public enum FusedActivationType
    {
        None, ReLU, Sigmoid, Tanh, GELU, SiLU, Mish, Swish, LeakyReLU
    }

    /// <summary>A node in the computation graph.</summary>
    public sealed class GraphNode
    {
        /// <summary>Unique node ID.</summary>
        public int Id { get; }

        /// <summary>Operation type.</summary>
        public OpType Type { get; }

        /// <summary>Input node IDs.</summary>
        public int[] InputIds { get; }

        /// <summary>Output tensor shape.</summary>
        public int[] OutputShape { get; }

        /// <summary>Operation parameters.</summary>
        public OpParams? Params { get; }

        /// <summary>Whether this is an external input node.</summary>
        public bool IsInput { get; }

        /// <summary>Whether this is a graph output node.</summary>
        public bool IsOutput { get; set; }

        /// <summary>Whether this operation can execute in-place.</summary>
        public bool CanExecuteInPlace { get; }

        internal GraphNode(int id, OpType type, int[] inputIds, int[] outputShape,
            OpParams? parms, bool isInput, bool canInPlace)
        {
            Id = id;
            Type = type;
            InputIds = inputIds;
            OutputShape = outputShape;
            Params = parms;
            IsInput = isInput;
            CanExecuteInPlace = canInPlace;
        }
    }

    /// <summary>Gets all nodes in the graph.</summary>
    public IReadOnlyList<GraphNode> Nodes => _nodes;

    /// <summary>Gets the IDs of input nodes.</summary>
    public IReadOnlyList<int> InputNodeIds => _inputNodeIds;

    /// <summary>Gets the IDs of output nodes.</summary>
    public IReadOnlyList<int> OutputNodeIds => _outputNodeIds;

    /// <summary>Gets whether the graph is currently capturing operations.</summary>
    public bool IsCapturing => _capturing;

    /// <summary>Gets whether the graph has been finalized.</summary>
    public bool IsFinalized => _finalized;

    /// <summary>Gets the number of nodes in the graph.</summary>
    public int NodeCount => _nodes.Count;

    /// <summary>Begins capturing operations into the graph.</summary>
    public void BeginCapture()
    {
        if (_capturing)
            throw new InvalidOperationException("Already capturing.");
        if (_finalized)
            throw new InvalidOperationException("Graph is finalized. Create a new graph.");

        _capturing = true;
    }

    /// <summary>Records an external input tensor.</summary>
    /// <param name="shape">Shape of the input tensor.</param>
    /// <returns>Node ID for referencing in subsequent operations.</returns>
    public int RecordInput(int[] shape)
    {
        ThrowIfNotCapturing();

        int id = _nodes.Count;
        _nodes.Add(new GraphNode(id, OpType.Custom, Array.Empty<int>(), (int[])shape.Clone(),
            null, isInput: true, canInPlace: false));
        _inputNodeIds.Add(id);
        return id;
    }

    /// <summary>Records an operation and returns the output node ID.</summary>
    public int RecordOp(OpType type, int[] inputIds, int[] outputShape, OpParams? parms = null)
    {
        ThrowIfNotCapturing();

        // Validate all input IDs reference existing nodes
        for (int i = 0; i < inputIds.Length; i++)
        {
            if (inputIds[i] < 0 || inputIds[i] >= _nodes.Count)
                throw new ArgumentOutOfRangeException(nameof(inputIds),
                    $"Input ID {inputIds[i]} at index {i} does not reference an existing node (0 to {_nodes.Count - 1}).");
        }

        bool canInPlace = IsInPlaceEligible(type);
        int id = _nodes.Count;
        _nodes.Add(new GraphNode(id, type, (int[])inputIds.Clone(), (int[])outputShape.Clone(),
            parms, isInput: false, canInPlace: canInPlace));
        return id;
    }

    /// <summary>Marks a node as a graph output.</summary>
    public void RecordOutput(int nodeId)
    {
        ThrowIfNotCapturing();
        if (nodeId < 0 || nodeId >= _nodes.Count)
            throw new ArgumentOutOfRangeException(nameof(nodeId));

        _nodes[nodeId].IsOutput = true;
        _outputNodeIds.Add(nodeId);
    }

    /// <summary>Ends capture and finalizes the graph.</summary>
    public void EndCapture()
    {
        if (!_capturing)
            throw new InvalidOperationException("Not currently capturing.");

        _capturing = false;
        _finalized = true;
    }

    /// <summary>
    /// Optimizes the captured graph using the MemoryPlanner to compute
    /// the minimum workspace layout with buffer aliasing and in-place scheduling.
    /// </summary>
    /// <returns>An optimized memory plan for this graph.</returns>
    public MemoryPlanner.MemoryPlan Optimize()
    {
        if (!_finalized)
            throw new InvalidOperationException("Graph must be finalized before optimization.");

        var planner = new MemoryPlanner();

        // Map graph node IDs to planner tensor IDs
        var nodeToTensor = new int[_nodes.Count];

        for (int i = 0; i < _nodes.Count; i++)
        {
            var node = _nodes[i];

            if (node.IsInput)
            {
                nodeToTensor[i] = planner.AddExternalInput(node.OutputShape);
            }
            else
            {
                var inputs = new int[node.InputIds.Length];
                for (int j = 0; j < node.InputIds.Length; j++)
                    inputs[j] = nodeToTensor[node.InputIds[j]];

                nodeToTensor[i] = planner.AddOp(
                    node.Type.ToString(),
                    inputs,
                    node.OutputShape,
                    canInPlace: node.CanExecuteInPlace);
            }
        }

        // Pin output tensors so their slots are never recycled
        foreach (int outputNodeId in _outputNodeIds)
        {
            planner.MarkOutput(nodeToTensor[outputNodeId]);
        }

        return planner.Plan();
    }

    /// <summary>Determines if an operation type can execute in-place.</summary>
    private static bool IsInPlaceEligible(OpType type)
    {
        return type switch
        {
            OpType.ReLU or OpType.Sigmoid or OpType.Tanh or OpType.GELU or
            OpType.Mish or OpType.Swish or OpType.SiLU or OpType.ELU or
            OpType.LeakyReLU or OpType.Negate or OpType.Abs or
            OpType.Exp or OpType.Log or OpType.Sqrt or
            OpType.AddScalar or OpType.MultiplyScalar or OpType.DivideScalar
                => true,
            _ => false,
        };
    }

    private void ThrowIfNotCapturing()
    {
        if (!_capturing)
            throw new InvalidOperationException("Not in capture mode. Call BeginCapture() first.");
    }
}
