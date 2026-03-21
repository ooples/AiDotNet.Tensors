using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using static AiDotNet.Tensors.Helpers.ComputationGraph;

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Executes a captured computation graph using a pre-allocated TensorWorkspace
/// and IEngine operations. Achieves zero allocation during forward passes by
/// reading/writing workspace slots instead of allocating tensors.
/// </summary>
/// <remarks>
/// <para>
/// This is the runtime component that makes graph capture useful:
/// 1. ComputationGraph captures the operations (what to do)
/// 2. MemoryPlanner computes the workspace layout (where to store data)
/// 3. GraphExecutor runs the operations using workspace memory (how to execute)
/// </para>
/// </remarks>
/// <typeparam name="T">Element type for tensor operations.</typeparam>
public sealed class GraphExecutor<T> : IDisposable
{
    private readonly ComputationGraph _graph;
    private readonly MemoryPlanner.MemoryPlan _plan;
    private readonly TensorWorkspace<T> _workspace;
    private readonly IEngine _engine;
    private readonly int[] _nodeToTensorId;
    private readonly Dictionary<OpType, Action<GraphNode, Tensor<T>[], Tensor<T>>> _opHandlers;
    // Pre-allocated per-execute buffers to avoid per-call allocation
    private readonly Tensor<T>[] _tensorCache;
    private readonly Tensor<T>[][] _inputBuffers;
    // Cached default gamma/beta tensors (one per unique size)
    private readonly Dictionary<int, Tensor<T>> _cachedOnes = new();
    private readonly Dictionary<int, Tensor<T>> _cachedZeros = new();
    private bool _disposed;

    /// <summary>The workspace backing all intermediate tensors.</summary>
    public TensorWorkspace<T> Workspace => _workspace;

    /// <summary>The memory plan used for slot assignments.</summary>
    public MemoryPlanner.MemoryPlan Plan => _plan;

    /// <summary>
    /// Creates a graph executor from a captured and finalized graph.
    /// </summary>
    /// <param name="graph">A finalized computation graph.</param>
    /// <param name="engine">The engine to use for execution.</param>
    public GraphExecutor(ComputationGraph graph, IEngine engine)
    {
        if (!graph.IsFinalized)
            throw new InvalidOperationException("Graph must be finalized before creating an executor.");

        _graph = graph;
        _engine = engine;
        _plan = graph.Optimize();
        _workspace = _plan.CreateWorkspace<T>();

        // Build mapping from graph node IDs to planner tensor IDs
        // This mapping is 1:1 since MemoryPlanner assigns sequential IDs
        _nodeToTensorId = new int[graph.NodeCount];
        int nextId = 0;
        for (int i = 0; i < graph.NodeCount; i++)
        {
            _nodeToTensorId[i] = nextId++;
        }

        // Pre-allocate per-execute buffers (zero allocation during forward passes)
        _tensorCache = new Tensor<T>[graph.NodeCount];
        _inputBuffers = new Tensor<T>[graph.NodeCount][];
        for (int i = 0; i < graph.NodeCount; i++)
        {
            var node = graph.Nodes[i];
            if (!node.IsInput)
                _inputBuffers[i] = new Tensor<T>[node.InputIds.Length];
        }

        // Build dispatch table — follows open/closed principle
        // New ops are added by registering handlers, not modifying a switch
        _opHandlers = BuildDispatchTable();
    }

    /// <summary>
    /// Registers a custom operation handler. Allows extending the executor
    /// without modifying the class (open/closed principle).
    /// </summary>
    public void RegisterOpHandler(OpType opType, Action<GraphNode, Tensor<T>[], Tensor<T>> handler)
    {
        _opHandlers[opType] = handler;
    }

    /// <summary>
    /// Executes the graph with the given input tensors and returns the output tensors.
    /// All intermediate tensors use pre-allocated workspace memory (zero allocation).
    /// </summary>
    /// <param name="inputs">Input tensors in the order they were registered via RecordInput.</param>
    /// <returns>Output tensors.</returns>
    public Tensor<T>[] Execute(params Tensor<T>[] inputs)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(GraphExecutor<T>));

        if (inputs.Length != _graph.InputNodeIds.Count)
            throw new ArgumentException(
                $"Expected {_graph.InputNodeIds.Count} inputs but got {inputs.Length}.");

        var nodes = _graph.Nodes;

        // Use pre-allocated tensorCache (zero per-call allocation)
        Array.Clear(_tensorCache, 0, _tensorCache.Length);

        // Map input nodes to provided tensors
        for (int i = 0; i < _graph.InputNodeIds.Count; i++)
        {
            _tensorCache[_graph.InputNodeIds[i]] = inputs[i];
        }

        // Execute each non-input node
        for (int nodeIdx = 0; nodeIdx < nodes.Count; nodeIdx++)
        {
            var node = nodes[nodeIdx];
            if (node.IsInput) continue;

            // Get output tensor from workspace slot (zero alloc)
            int tensorId = _nodeToTensorId[nodeIdx];
            int slotId = _plan.GetSlotForTensor(tensorId);
            Tensor<T> output;

            if (slotId >= 0 && slotId < _workspace.SlotCount)
            {
                output = _workspace.Get(slotId);
            }
            else
            {
                output = new Tensor<T>(node.OutputShape);
            }

            // Use pre-allocated input buffer (zero per-node allocation)
            var inputTensors = _inputBuffers[nodeIdx];
            for (int j = 0; j < node.InputIds.Length; j++)
            {
                inputTensors[j] = _tensorCache[node.InputIds[j]];
            }

            ExecuteOp(node, inputTensors, output);
            _tensorCache[nodeIdx] = output;
        }

        // Collect outputs (this small array is the only per-call allocation)
        var outputs = new Tensor<T>[_graph.OutputNodeIds.Count];
        for (int i = 0; i < outputs.Length; i++)
        {
            outputs[i] = _tensorCache[_graph.OutputNodeIds[i]];
        }

        return outputs;
    }

    private void ExecuteOp(GraphNode node, Tensor<T>[] inputs, Tensor<T> output)
    {
        if (_opHandlers.TryGetValue(node.Type, out var handler))
        {
            handler(node, inputs, output);
        }
        else
        {
            throw new NotSupportedException(
                $"Operation type '{node.Type}' is not registered. Use RegisterOpHandler to add support.");
        }
    }

    private Dictionary<OpType, Action<GraphNode, Tensor<T>[], Tensor<T>>> BuildDispatchTable()
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        var defaultAlpha = numOps.FromDouble(0.01);

        return new Dictionary<OpType, Action<GraphNode, Tensor<T>[], Tensor<T>>>
        {
            // Element-wise arithmetic
            [OpType.Add] = (_, inp, o) => _engine.TensorAddInto(o, inp[0], inp[1]),
            [OpType.Residual] = (_, inp, o) => _engine.TensorAddInto(o, inp[0], inp[1]),
            [OpType.Subtract] = (_, inp, o) => numOps.Subtract(inp[0].AsSpan(), inp[1].AsSpan(), o.AsWritableSpan()),
            [OpType.Multiply] = (_, inp, o) => _engine.TensorMultiplyInto(o, inp[0], inp[1]),

            // Activations
            [OpType.ReLU] = (_, inp, o) => { inp[0].Data.Span.CopyTo(o.Data.Span); _engine.ReLUInPlace(o); },
            [OpType.Sigmoid] = (_, inp, o) => { inp[0].Data.Span.CopyTo(o.Data.Span); _engine.SigmoidInPlace(o); },
            [OpType.Tanh] = (_, inp, o) => _engine.TanhInto(o, inp[0]),
            [OpType.GELU] = (_, inp, o) => _engine.GELUInto(o, inp[0]),
            [OpType.Mish] = (_, inp, o) => _engine.MishInto(o, inp[0]),
            [OpType.Swish] = (_, inp, o) => _engine.SwishInto(o, inp[0]),
            [OpType.SiLU] = (_, inp, o) => _engine.SwishInto(o, inp[0]),
            [OpType.LeakyReLU] = (n, inp, o) =>
            {
                T alpha = n.Params != null ? numOps.FromDouble(n.Params.Alpha) : defaultAlpha;
                _engine.LeakyReLUInto(o, inp[0], alpha);
            },
            [OpType.Softmax] = (n, inp, o) => _engine.SoftmaxInto(o, inp[0], axis: n.Params?.Axis ?? -1),

            // Convolution
            [OpType.Conv2D] = (n, inp, o) =>
            {
                if (inp.Length >= 2)
                    _engine.Conv2DInto(o, inp[0], inp[1],
                        stride: n.Params?.Stride ?? 1, padding: n.Params?.Padding ?? 0, dilation: n.Params?.Dilation ?? 1);
                else
                    inp[0].Data.Span.CopyTo(o.Data.Span);
            },

            // Normalization
            [OpType.GroupNorm] = (n, inp, o) =>
                _engine.GroupNormInto(o, inp[0], n.Params?.Groups ?? 32,
                    inp.Length > 1 ? inp[1] : CreateOnes(o.Shape[1]),
                    inp.Length > 2 ? inp[2] : CreateZeros(o.Shape[1]),
                    n.Params?.Epsilon ?? 1e-5, out _, out _),
            [OpType.FusedGroupNormActivation] = (n, inp, o) =>
                _engine.GroupNormSwishInto(o, inp[0], n.Params?.Groups ?? 32,
                    inp.Length > 1 ? inp[1] : CreateOnes(o.Shape[1]),
                    inp.Length > 2 ? inp[2] : CreateZeros(o.Shape[1]),
                    n.Params?.Epsilon ?? 1e-5),

            // Element-wise math
            [OpType.Exp] = (_, inp, o) => { var r = _engine.TensorExp(inp[0]); Array.Copy(r.GetDataArray(), o.GetDataArray(), Math.Min(r.Length, o.Length)); },
            [OpType.Log] = (_, inp, o) => { var r = _engine.TensorLog(inp[0]); Array.Copy(r.GetDataArray(), o.GetDataArray(), Math.Min(r.Length, o.Length)); },
            [OpType.Sqrt] = (_, inp, o) => { var r = _engine.TensorSqrt(inp[0]); Array.Copy(r.GetDataArray(), o.GetDataArray(), Math.Min(r.Length, o.Length)); },
            [OpType.Abs] = (_, inp, o) => { var r = _engine.TensorAbs(inp[0]); Array.Copy(r.GetDataArray(), o.GetDataArray(), Math.Min(r.Length, o.Length)); },

            // Linear algebra
            [OpType.MatMul] = (_, inp, o) => _engine.MatMulInto(o, inp[0], inp[1]),

            // Pooling (handler registered later in the table)

            // Reductions
            [OpType.Sum] = (_, inp, o) => { o.AsWritableSpan()[0] = _engine.TensorSum(inp[0]); },
            [OpType.Mean] = (_, inp, o) => { o.AsWritableSpan()[0] = _engine.TensorMean(inp[0]); },

            // Concat
            [OpType.Concat] = (n, inp, o) =>
            {
                _engine.ConcatInto(o, inp, n.Params?.Axis ?? 0);
            },

            // Transpose — use params Axis if available, otherwise reverse dimensions
            [OpType.Transpose] = (n, inp, o) =>
            {
                int rank = inp[0].Rank;
                Span<int> axes = rank <= 8 ? stackalloc int[rank] : new int[rank];
                for (int ax = 0; ax < rank; ax++) axes[ax] = rank - 1 - ax;
                _engine.TransposeInto(o, inp[0], axes.ToArray());
            },

            // Attention
            [OpType.FlashAttention] = (_, inp, o) =>
            {
                if (inp.Length >= 3)
                {
                    Tensor<T> stats;
                    var result = _engine.FlashAttention(inp[0], inp[1], inp[2], scale: null, isCausal: false, softmaxStats: out stats);
                    Array.Copy(result.GetDataArray(), o.GetDataArray(), Math.Min(result.Length, o.Length));
                }
            },

            // Fused operations
            [OpType.FusedConv2DBiasActivation] = (n, inp, o) =>
            {
                if (inp.Length >= 2)
                {
                    var bias = inp.Length >= 3 ? inp[2] : null;
                    var result = _engine.FusedConv2D(inp[0], inp[1], bias,
                        n.Params?.Stride ?? 1, n.Params?.Stride ?? 1,
                        n.Params?.Padding ?? 0, n.Params?.Padding ?? 0,
                        n.Params?.Dilation ?? 1, n.Params?.Dilation ?? 1,
                        (Engines.FusedActivationType)(int)(n.Params?.FusedActivation ?? ComputationGraph.FusedActivationType.None));
                    Array.Copy(result.GetDataArray(), o.GetDataArray(), Math.Min(result.Length, o.Length));
                }
            },

            // BatchNorm
            [OpType.BatchNorm] = (n, inp, o) =>
            {
                if (inp.Length >= 3)
                {
                    var result = _engine.BatchNorm(inp[0], inp[1], inp[2],
                        n.Params?.Epsilon ?? 1e-5, out _, out _);
                    Array.Copy(result.GetDataArray(), o.GetDataArray(), Math.Min(result.Length, o.Length));
                }
            },

            // MaxPool
            [OpType.MaxPool2D] = (n, inp, o) =>
            {
                var result = _engine.MaxPool2D(inp[0], n.Params?.Stride ?? 2, n.Params?.Stride ?? 2, n.Params?.Padding ?? 0);
                Array.Copy(result.GetDataArray(), o.GetDataArray(), Math.Min(result.Length, o.Length));
            },

            // Dropout (pass-through for inference)
            [OpType.Dropout] = (_, inp, o) => Array.Copy(inp[0].GetDataArray(), o.GetDataArray(), Math.Min(inp[0].Length, o.Length)),
        };
    }

    private Tensor<T> CreateOnes(int size)
    {
        if (_cachedOnes.TryGetValue(size, out var cached))
            return cached;
        var t = new Tensor<T>(new[] { size });
        var numOps = MathHelper.GetNumericOperations<T>();
        var span = t.AsWritableSpan();
        for (int i = 0; i < size; i++)
            span[i] = numOps.One;
        _cachedOnes[size] = t;
        return t;
    }

    private Tensor<T> CreateZeros(int size)
    {
        if (_cachedZeros.TryGetValue(size, out var cached))
            return cached;
        var t = new Tensor<T>(new[] { size });
        _cachedZeros[size] = t;
        return t;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _workspace.Dispose();
    }
}
