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

        // Realize input first if it's also lazy
        if (Input is { LazySource: ILazyNode inputNode } && !inputNode.IsRealized)
            inputNode.Realize(engine);

        Execute(engine, Output);
        IsRealized = true;
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
