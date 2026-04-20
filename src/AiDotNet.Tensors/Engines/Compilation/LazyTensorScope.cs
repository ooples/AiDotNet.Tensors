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

    /// <summary>Records a cross-type operation (input type differs from output type).</summary>
    internal Tensor<TOut> RecordCrossType<TIn, TOut>(
        LazyNodeType opType,
        string opName,
        Tensor<TIn> input,
        int[] outputShape,
        Action<IEngine, Tensor<TOut>> execute)
    {
        var output = TensorAllocator.RentUninitialized<TOut>(outputShape);
        var node = new CrossTypeLazyNode<TIn, TOut>(opType, opName, input, output, execute);
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

    /// <summary>
    /// Records a pure metadata-view op (contiguous <c>Reshape</c>, <c>Squeeze</c>,
    /// <c>Unsqueeze</c>, <c>Permute</c> on already-contiguous tensors, ...).
    /// Unlike <see cref="RecordUnary"/>, the output tensor is supplied by the
    /// caller and shares storage with the input — a fresh output buffer would
    /// defeat the whole point of a view. The recorded execute delegate is a
    /// no-op because writes to the producer's buffer are live-visible through
    /// the view.
    ///
    /// This exists so that a <c>Func&lt;Tensor&lt;T&gt;&gt;</c> forward that
    /// ends in a pure-view op still hands the compiler a tensor with a
    /// <see cref="Tensor{T}.LazySource"/>, which lets
    /// <see cref="CompiledInferencePlan{T}.Compile"/> pick up the caller's
    /// actual return value as <c>_finalOutput</c> instead of guessing via
    /// the last-step heuristic.
    /// </summary>
    internal Tensor<T> RecordView<T>(
        LazyNodeType opType,
        string opName,
        Tensor<T> input,
        Tensor<T> view)
    {
        // The caller already built the view with shared storage — we just
        // attach it to the graph so the compile step sees a producing node.
        var node = new LazyNode<T>(opType, opName, input, view,
            execute: (_, _) => { /* storage is shared; nothing to compute */ });
        view.LazySource = node;
        _nodes.Add(node);
        return view;
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
    /// <remarks>
    /// Falls back to the last-step heuristic for <c>_finalOutput</c>. Prefer
    /// <see cref="CompileInference{T}(Tensor{T})"/> when you have the caller's
    /// returned tensor — the explicit path is correct even when the forward
    /// ends in a pure-view op or when optimization passes reorder steps.
    /// </remarks>
    internal CompiledInferencePlan<T> CompileInference<T>()
    {
        MarkCompiled();
        return CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput: null);
    }

    /// <summary>
    /// Compiles the lazy graph with an explicit output tensor. The plan's
    /// <c>Execute()</c> returns <paramref name="explicitOutput"/> instead of
    /// the last optimized step's output buffer — fixes issue #228.
    /// </summary>
    internal CompiledInferencePlan<T> CompileInference<T>(Tensor<T> explicitOutput)
    {
        MarkCompiled();
        return CompiledInferencePlan<T>.Compile(this, _engine, explicitOutput);
    }

    /// <summary>
    /// Compiles the lazy graph into a training plan with forward + backward steps.
    /// The plan can be replayed for zero-overhead training iterations.
    /// </summary>
    /// <remarks>
    /// Falls back to the last forward step's output as the loss tensor. Prefer
    /// the explicit-loss overload when you have the caller's returned loss —
    /// a forward+loss lambda ending in a view op (e.g. <c>loss.Reshape([])</c>
    /// to scalarize) is only correct via the explicit path.
    /// </remarks>
    internal CompiledTrainingPlan<T> CompileTraining<T>(Tensor<T>[] parameters)
    {
        MarkCompiled();
        return CompiledTrainingPlan<T>.Compile(this, _engine, parameters, explicitLoss: null);
    }

    /// <summary>
    /// Compiles the lazy graph into a training plan, threading the caller's
    /// returned loss tensor through as the explicit loss output (issue #228).
    /// </summary>
    internal CompiledTrainingPlan<T> CompileTraining<T>(Tensor<T>[] parameters, Tensor<T> explicitLoss)
    {
        MarkCompiled();
        return CompiledTrainingPlan<T>.Compile(this, _engine, parameters, explicitLoss);
    }

    /// <summary>
    /// Marks this scope as compiled so Dispose() won't auto-realize the graph.
    /// Called by CompileInference/CompileTraining which handle the graph themselves.
    /// </summary>
    internal void MarkCompiled()
    {
        _realized = true;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;

        try
        {
            // Auto-realize on dispose if not yet done (safety net)
            if (!_realized)
                Realize();
        }
        finally
        {
            // Always restore parent scope, even if Realize() throws
            GraphMode.SetCurrent(_parent);
        }
    }
}
