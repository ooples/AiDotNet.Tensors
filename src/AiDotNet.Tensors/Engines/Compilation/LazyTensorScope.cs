using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Manages a lazy computation graph. When active, tensor operations create LazyNodes
/// instead of executing. On Realize(), the graph is optimized (fused) and executed.
///
/// Lifecycle: Enable() → record ops → Realize() or Dispose() → compiled execution
///
/// For training: CompileTraining() produces a CompiledTrainingPlan that caches the
/// compiled forward+backward for zero-overhead replay across training steps.
/// </summary>
internal sealed class LazyTensorScope : IDisposable
{
    private readonly LazyTensorScope? _parent;
    private readonly List<ILazyNode> _nodes = new();
    private readonly IEngine _engine;
    private bool _disposed;
    private bool _realized;

    internal LazyTensorScope(LazyTensorScope? parent)
    {
        _parent = parent;
        _engine = AiDotNetEngine.Current;
    }

    /// <summary>Number of lazy operations recorded.</summary>
    internal int NodeCount => _nodes.Count;

    /// <summary>
    /// Records a unary operation as a lazy node. Returns a tensor whose data
    /// is not yet computed — it will be materialized during Realize().
    /// </summary>
    internal Tensor<T> RecordUnary<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> input,
        int[] outputShape,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        // Allocate output tensor with correct shape but uninitialized data
        var output = TensorAllocator.RentUninitialized<T>(outputShape);

        var node = new LazyNode<T>(opType, opName, input, output, execute, backwardFn, savedState);
        output.LazySource = node;
        _nodes.Add(node);

        return output;
    }

    /// <summary>Records a binary operation as a lazy node.</summary>
    internal Tensor<T> RecordBinary<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> input0,
        Tensor<T> input1,
        int[] outputShape,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        var output = TensorAllocator.RentUninitialized<T>(outputShape);

        var node = new LazyNode<T>(opType, opName, input0, input1, output, execute, backwardFn, savedState);
        output.LazySource = node;
        _nodes.Add(node);

        return output;
    }

    /// <summary>Records a variadic operation as a lazy node.</summary>
    internal Tensor<T> RecordVariadic<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T>[] inputs,
        int[] outputShape,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        var output = TensorAllocator.RentUninitialized<T>(outputShape);

        var node = new LazyNode<T>(opType, opName, inputs, output, execute, backwardFn, savedState);
        output.LazySource = node;
        _nodes.Add(node);

        return output;
    }

    /// <summary>
    /// Materializes all lazy nodes in the graph. Walks in topological order
    /// (inputs before outputs) and executes each node that hasn't been realized.
    /// In the future, this will run the fusion pass and compiled plan instead.
    /// </summary>
    internal void Realize()
    {
        if (_realized) return;
        _realized = true;

        // Run graph compiler: optimization passes (fusion, DCE) + topological sort
        var compiler = new LazyGraphCompiler();
        var optimized = compiler.Compile(_nodes);

        // Suspend graph mode so execute delegates call eager paths (no re-entry)
        var savedScope = GraphMode.Current;
        GraphMode.SetCurrent(null);
        try
        {
            foreach (var node in optimized)
            {
                if (!node.IsRealized)
                    node.Realize(_engine);
            }
        }
        finally
        {
            GraphMode.SetCurrent(savedScope);
        }

        // Clear lazy source references so tensors behave normally after realization
        foreach (var node in optimized)
        {
            node.ClearOutputLazySource();
        }
    }

    /// <summary>Gets all recorded nodes (for graph compiler).</summary>
    internal IReadOnlyList<ILazyNode> Nodes => _nodes;

    /// <summary>
    /// Compiles the lazy graph into an inference plan for zero-overhead replay.
    /// Call this instead of Realize() when you want to cache and replay the plan.
    /// </summary>
    internal CompiledInferencePlan<T> CompileInference<T>()
    {
        return CompiledInferencePlan<T>.Compile(this, _engine);
    }

    /// <summary>
    /// Compiles the lazy graph into a training plan with forward + backward steps.
    /// The plan can be replayed for zero-overhead training iterations.
    /// </summary>
    internal CompiledTrainingPlan<T> CompileTraining<T>(Tensor<T>[] parameters)
    {
        return CompiledTrainingPlan<T>.Compile(this, _engine, parameters);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        // Auto-realize on dispose if not yet done (safety net)
        if (!_realized)
            Realize();

        // Restore parent scope
        GraphMode.SetCurrent(_parent);
    }
}
