using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// Backward for a cross-type lazy node: the upstream gradient arrives in the OUTPUT element type
/// (<typeparamref name="TOut"/>) and must be turned into the INPUT-typed gradient
/// (<typeparamref name="TIn"/>) so it can be deposited into the input's (different-dtype) grad space.
/// For a mixed-precision cast the local Jacobian is the identity, so the implementation just casts the
/// gradient across the dtype boundary (see <see cref="Autodiff.MixedPrecisionCast"/>); for a genuine
/// cross-type op (e.g. complex magnitude) it carries that op's real Jacobian. Returns the input-typed
/// gradient; <c>null</c> means "no contribution" (treated as a stop edge by the backward walk).
/// </summary>
internal delegate Tensor<TIn>? CrossTypeBackwardFunction<TIn, TOut>(
    Tensor<TOut> gradOutput,
    Tensor<TIn> input,
    Tensor<TOut> output,
    object[] savedState,
    IEngine engine);

/// <summary>
/// A lazy computation node for cross-type operations where the input and output
/// tensor element types differ (e.g., Tensor&lt;Complex&lt;T&gt;&gt; → Tensor&lt;T&gt;).
/// Used for operations like ComplexMagnitude, ComplexPhase, and FFT — and, with a
/// <see cref="BackwardFn"/>, the differentiable FP32↔FP16 cast that bridges the two grad spaces in
/// mixed-precision (#555) training.
/// </summary>
internal sealed class CrossTypeLazyNode<TIn, TOut> : ILazyNode
{
    public LazyNodeType OpType { get; }
    public string OpName { get; }
    public int[] OutputShape { get; }
    public bool IsRealized { get; set; }
    public int TopologicalIndex { get; set; } = -1;
    public int ConsumerCount { get; set; }
    public IEngine RecordingEngine { get; } = AiDotNetEngine.Current;

    public readonly Tensor<TIn> Input;
    public readonly Tensor<TOut> Output;
    public readonly Action<IEngine, Tensor<TOut>> Execute;

    // ═══ Backward integration (mixed-precision #555) ═══
    /// <summary>Cross-type backward, or null for a forward-only node (FFT/Complex realize-only ops).</summary>
    public readonly CrossTypeBackwardFunction<TIn, TOut>? BackwardFn;
    public readonly object[]? SavedState;

    public CrossTypeLazyNode(
        LazyNodeType opType,
        string opName,
        Tensor<TIn> input,
        Tensor<TOut> output,
        Action<IEngine, Tensor<TOut>> execute,
        CrossTypeBackwardFunction<TIn, TOut>? backwardFn = null,
        object[]? savedState = null)
    {
        OpType = opType;
        OpName = opName;
        Input = input;
        Output = output;
        OutputShape = output._shape;
        Execute = execute;
        BackwardFn = backwardFn;
        SavedState = savedState;
    }

    /// <summary>
    /// Runs the cross-type backward: turns the output-typed gradient into the input-typed gradient.
    /// Returns null when the node has no backward (forward-only) — the walk treats that as a stop edge.
    /// </summary>
    public Tensor<TIn>? Backward(Tensor<TOut> gradOutput, IEngine engine)
    {
        if (BackwardFn is null) return null;
        return BackwardFn(gradOutput, Input, Output, SavedState ?? System.Array.Empty<object>(), engine);
    }

    public void Realize(IEngine engine)
    {
        if (IsRealized) return;
        IsRealized = true; // Set BEFORE executing to prevent re-entrant recursion

        // Issue #238: the Execute delegate re-invokes the engine op (e.g.,
        // `eng.NativeComplexIFFTReal(ci)`), which checks GraphMode.IsActive.
        // When Realize is triggered via EnsureMaterialized while tracing is
        // still in progress (scope not yet disposed, nobody else suppressed
        // GraphMode), the re-invoked op records ANOTHER CrossTypeLazyNode,
        // returns a new lazy tensor, and the lambda's `r.AsSpan()` forces
        // Realize on that new node → infinite recursion until StackOverflow.
        //
        // LazyNode<T>.Realize already suspends GraphMode for exactly this
        // reason — mirror that pattern here so cross-type ops (FFT,
        // ComplexMagnitude, ComplexPhase, IFFTReal) are protected too.
        var savedScope = GraphMode.Current;
        GraphMode.SetCurrent(null);
        try
        {
            // Realize input first if it's also lazy
            if (Input is { LazySource: ILazyNode inputNode } && !inputNode.IsRealized)
                inputNode.Realize(engine);

            Execute(engine, Output);
        }
        catch
        {
            // Roll back so a retry or diagnostic access doesn't observe
            // uninitialised output data.
            IsRealized = false;
            throw;
        }
        finally
        {
            GraphMode.SetCurrent(savedScope);
        }
    }

    public ILazyNode[] GetInputNodes()
    {
        if (Input?.LazySource is ILazyNode node)
            return new[] { node };
        return Array.Empty<ILazyNode>();
    }

    public void ClearOutputLazySource()
    {
        Output.LazySource = null;
    }
}
