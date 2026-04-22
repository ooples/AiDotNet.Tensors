using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A lazy computation node for cross-type operations where the input and output
/// tensor element types differ (e.g., Tensor&lt;Complex&lt;T&gt;&gt; → Tensor&lt;T&gt;).
/// Used for operations like ComplexMagnitude, ComplexPhase, and FFT.
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

    public CrossTypeLazyNode(
        LazyNodeType opType,
        string opName,
        Tensor<TIn> input,
        Tensor<TOut> output,
        Action<IEngine, Tensor<TOut>> execute)
    {
        OpType = opType;
        OpName = opName;
        Input = input;
        Output = output;
        OutputShape = output._shape;
        Execute = execute;
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
