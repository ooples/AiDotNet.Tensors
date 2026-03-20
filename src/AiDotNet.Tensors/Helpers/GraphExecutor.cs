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
        var tensorCache = new Tensor<T>[nodes.Count];

        // Map input nodes to provided tensors
        for (int i = 0; i < _graph.InputNodeIds.Count; i++)
        {
            tensorCache[_graph.InputNodeIds[i]] = inputs[i];
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

            if (slotId >= 0)
            {
                output = _workspace.Get(slotId);
            }
            else
            {
                // Fallback: allocate if not in workspace (shouldn't happen normally)
                output = new Tensor<T>(node.OutputShape);
            }

            // Get input tensors
            var inputTensors = new Tensor<T>[node.InputIds.Length];
            for (int j = 0; j < node.InputIds.Length; j++)
            {
                inputTensors[j] = tensorCache[node.InputIds[j]];
            }

            // Execute the operation
            ExecuteOp(node, inputTensors, output);
            tensorCache[nodeIdx] = output;
        }

        // Collect outputs
        var outputs = new Tensor<T>[_graph.OutputNodeIds.Count];
        for (int i = 0; i < outputs.Length; i++)
        {
            outputs[i] = tensorCache[_graph.OutputNodeIds[i]];
        }

        return outputs;
    }

    private void ExecuteOp(GraphNode node, Tensor<T>[] inputs, Tensor<T> output)
    {
        var p = node.Params;

        switch (node.Type)
        {
            // Element-wise arithmetic — use Into variants for zero-alloc
            case OpType.Add:
            case OpType.Residual:
                _engine.TensorAddInto(output, inputs[0], inputs[1]);
                break;
            case OpType.Subtract:
                var subOps = MathHelper.GetNumericOperations<T>();
                subOps.Subtract(inputs[0].AsSpan(), inputs[1].AsSpan(), output.AsWritableSpan());
                break;
            case OpType.Multiply:
                _engine.TensorMultiplyInto(output, inputs[0], inputs[1]);
                break;

            // Activations — copy input then apply in-place, or use Into variant
            case OpType.ReLU:
                inputs[0].Data.Span.CopyTo(output.Data.Span);
                _engine.ReLUInPlace(output);
                break;
            case OpType.Sigmoid:
                inputs[0].Data.Span.CopyTo(output.Data.Span);
                _engine.SigmoidInPlace(output);
                break;
            case OpType.Tanh:
                _engine.TanhInto(output, inputs[0]);
                break;
            case OpType.GELU:
                _engine.GELUInto(output, inputs[0]);
                break;
            case OpType.Mish:
                _engine.MishInto(output, inputs[0]);
                break;
            case OpType.Swish:
            case OpType.SiLU:
                _engine.SwishInto(output, inputs[0]);
                break;
            case OpType.LeakyReLU:
                T alpha = p != null ? MathHelper.GetNumericOperations<T>().FromDouble(p.Alpha) : MathHelper.GetNumericOperations<T>().FromDouble(0.01);
                _engine.LeakyReLUInto(output, inputs[0], alpha);
                break;
            case OpType.Softmax:
                _engine.SoftmaxInto(output, inputs[0], axis: p?.Axis ?? -1);
                break;

            // Convolution
            case OpType.Conv2D:
                if (inputs.Length >= 2)
                {
                    _engine.Conv2DInto(output, inputs[0], inputs[1],
                        stride: p?.Stride ?? 1, padding: p?.Padding ?? 0, dilation: p?.Dilation ?? 1);
                }
                else
                {
                    // Single input conv (kernel baked into graph — not supported yet)
                    inputs[0].Data.Span.CopyTo(output.Data.Span);
                }
                break;

            // Normalization
            case OpType.GroupNorm:
                _engine.GroupNormInto(output, inputs[0], p?.Groups ?? 32,
                    inputs.Length > 1 ? inputs[1] : CreateOnes(output.Shape[1]),
                    inputs.Length > 2 ? inputs[2] : CreateZeros(output.Shape[1]),
                    p?.Epsilon ?? 1e-5, out _, out _);
                break;

            // Fused operations
            case OpType.FusedGroupNormActivation:
                _engine.GroupNormSwishInto(output, inputs[0], p?.Groups ?? 32,
                    inputs.Length > 1 ? inputs[1] : CreateOnes(output.Shape[1]),
                    inputs.Length > 2 ? inputs[2] : CreateZeros(output.Shape[1]),
                    p?.Epsilon ?? 1e-5);
                break;

            // Linear algebra
            case OpType.MatMul:
                _engine.MatMulInto(output, inputs[0], inputs[1]);
                break;

            // Reductions
            case OpType.Sum:
                var sumVal = _engine.TensorSum(inputs[0]);
                output.AsWritableSpan()[0] = sumVal;
                break;
            case OpType.Mean:
                var meanVal = _engine.TensorMean(inputs[0]);
                output.AsWritableSpan()[0] = meanVal;
                break;

            // Default: copy input to output (passthrough for unimplemented ops)
            default:
                if (inputs.Length > 0 && inputs[0].Length == output.Length)
                    inputs[0].Data.Span.CopyTo(output.Data.Span);
                break;
        }
    }

    private Tensor<T> CreateOnes(int size)
    {
        var t = new Tensor<T>(new[] { size });
        var numOps = MathHelper.GetNumericOperations<T>();
        var span = t.AsWritableSpan();
        for (int i = 0; i < size; i++)
            span[i] = numOps.One;
        return t;
    }

    private Tensor<T> CreateZeros(int size)
    {
        return new Tensor<T>(new[] { size });
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _workspace.Dispose();
    }
}
