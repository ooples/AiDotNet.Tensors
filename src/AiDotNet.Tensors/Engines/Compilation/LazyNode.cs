using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A node in the lazy computation graph. Created when GraphMode is active instead
/// of executing the operation immediately. Stores everything needed to execute the
/// operation later (during Realize) or to fuse it with adjacent operations.
/// </summary>
/// <typeparam name="T">The numeric element type.</typeparam>
internal sealed class LazyNode<T> : ILazyNode
{
    // ═══ Operation identity ═══
    public LazyNodeType OpType { get; }
    public string OpName { get; }

    // ═══ Input graph edges (inline for 1-3, overflow for 4+) ═══
    public readonly Tensor<T> Input0;
    public readonly Tensor<T>? Input1;
    public readonly Tensor<T>? Input2;
    public readonly Tensor<T>[]? InputsOverflow;
    public readonly byte InputCount;

    // ═══ Output (shape computed eagerly, data computed lazily) ═══
    public int[] OutputShape { get; }
    public readonly Tensor<T> Output;

    // ═══ Execution delegate — the actual operation to run ═══
    public readonly Action<IEngine, Tensor<T>> Execute;

    // ═══ Backward integration ═══
    public readonly BackwardFunction<T>? BackwardFn;
    public readonly object[]? SavedState;

    // ═══ Graph metadata ═══
    public bool IsRealized { get; set; }
    public int TopologicalIndex { get; set; } = -1;
    public int ConsumerCount { get; set; }

    /// <summary>Creates a unary lazy node (one input).</summary>
    public LazyNode(
        LazyNodeType opType,
        string opName,
        Tensor<T> input,
        Tensor<T> output,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        OpType = opType;
        OpName = opName;
        Input0 = input;
        InputCount = 1;
        Output = output;
        OutputShape = output._shape;
        Execute = execute;
        BackwardFn = backwardFn;
        SavedState = savedState;
    }

    /// <summary>Creates a binary lazy node (two inputs).</summary>
    public LazyNode(
        LazyNodeType opType,
        string opName,
        Tensor<T> input0,
        Tensor<T> input1,
        Tensor<T> output,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        OpType = opType;
        OpName = opName;
        Input0 = input0;
        Input1 = input1;
        InputCount = 2;
        Output = output;
        OutputShape = output._shape;
        Execute = execute;
        BackwardFn = backwardFn;
        SavedState = savedState;
    }

    /// <summary>Creates a variadic lazy node (3+ inputs).</summary>
    public LazyNode(
        LazyNodeType opType,
        string opName,
        Tensor<T>[] inputs,
        Tensor<T> output,
        Action<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backwardFn = null,
        object[]? savedState = null)
    {
        OpType = opType;
        OpName = opName;
        Input0 = inputs[0];
        Input1 = inputs.Length > 1 ? inputs[1] : null;
        Input2 = inputs.Length > 2 ? inputs[2] : null;
        // Store only the overflow elements (index 3+) to avoid duplicating Input0/1/2
        // in RealizeInputs() and GetInputNodes().
        if (inputs.Length > 3)
        {
            var overflow = new Tensor<T>[inputs.Length - 3];
            Array.Copy(inputs, 3, overflow, 0, overflow.Length);
            InputsOverflow = overflow;
        }
        InputCount = (byte)(inputs.Length > 3 ? 0xFF : inputs.Length);
        Output = output;
        OutputShape = output._shape;
        Execute = execute;
        BackwardFn = backwardFn;
        SavedState = savedState;
    }

    /// <summary>Materialize this node's output by executing the operation.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Realize(IEngine engine)
    {
        if (IsRealized) return;

        // Mark as realized BEFORE executing to prevent re-entrant auto-materialization.
        // The output tensor's AsSpan()/AsWritableSpan() checks LazySource.IsRealized —
        // if we don't set this first, the execute delegate writing to the output triggers
        // an infinite recursion: AsWritableSpan → auto-materialize → Realize → Execute → AsWritableSpan.
        IsRealized = true;

        try
        {
            // Ensure inputs are realized first
            RealizeInputs(engine);

            // Execute the operation
            Execute(engine, Output);
        }
        catch
        {
            // Roll back so a retry or diagnostic access doesn't see stale/uninitialized data.
            IsRealized = false;
            throw;
        }
    }

    private void RealizeInputs(IEngine engine)
    {
        if (Input0.LazySource is ILazyNode n0 && !n0.IsRealized) n0.Realize(engine);
        if (Input1?.LazySource is ILazyNode n1 && !n1.IsRealized) n1.Realize(engine);
        if (Input2?.LazySource is ILazyNode n2 && !n2.IsRealized) n2.Realize(engine);
        if (InputsOverflow != null)
            foreach (var inp in InputsOverflow)
                if (inp.LazySource is ILazyNode n && !n.IsRealized) n.Realize(engine);
    }

    /// <summary>Get input nodes for graph traversal.</summary>
    public ILazyNode[] GetInputNodes()
    {
        var nodes = new List<ILazyNode>();
        if (Input0.LazySource is ILazyNode n0) nodes.Add(n0);
        if (Input1?.LazySource is ILazyNode n1) nodes.Add(n1);
        if (Input2?.LazySource is ILazyNode n2) nodes.Add(n2);
        if (InputsOverflow != null)
            foreach (var inp in InputsOverflow)
                if (inp.LazySource is ILazyNode n) nodes.Add(n);
        return nodes.ToArray();
    }

    /// <summary>Clears the LazySource reference on the output tensor after realization.</summary>
    public void ClearOutputLazySource()
    {
        Output.LazySource = null;
    }

    /// <summary>Get input tensors as array for backward compatibility.</summary>
    public Tensor<T>[] GetInputsArray()
    {
        if (InputsOverflow != null)
        {
            // Reconstruct full array from inline slots + overflow
            var all = new Tensor<T>[3 + InputsOverflow.Length];
            all[0] = Input0;
            all[1] = Input1 ?? Input0;
            all[2] = Input2 ?? Input0;
            Array.Copy(InputsOverflow, 0, all, 3, InputsOverflow.Length);
            return all;
        }
        return InputCount switch
        {
            1 => new[] { Input0 },
            2 => new[] { Input0, Input1 ?? Input0 },
            3 => new[] { Input0, Input1 ?? Input0, Input2 ?? Input0 },
            _ => new[] { Input0 }
        };
    }
}
