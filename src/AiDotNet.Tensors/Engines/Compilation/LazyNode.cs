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

    // ═══ External prerequisite (in-place ops) ═══
    // When an in-place op records itself as Input0=target=Output=target, the
    // resulting GetInputNodes() lookup would resolve Input0.LazySource back to
    // THIS node (self-loop), and the prior producer of `target` would be lost.
    // RecordInPlace captures that prior producer here so topo-sort/DCE/fusion
    // see a real predecessor edge instead of a self-cycle, and so RealizeInputs
    // visits the prior producer before mutating its output buffer.
    private readonly ILazyNode? _externalPrerequisite;

    // ═══ Graph metadata ═══
    public bool IsRealized { get; set; }
    public int TopologicalIndex { get; set; } = -1;
    public int ConsumerCount { get; set; }
    public IEngine RecordingEngine { get; } = AiDotNetEngine.Current;

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
        object[]? savedState = null,
        ILazyNode? externalPrerequisite = null)
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
        // Snapshot the prior producer of an in-place target BEFORE the caller
        // overwrites target.LazySource with this node. See _externalPrerequisite
        // docstring above. Self-references (rare; happens when caller passes
        // `this` after construction) are filtered out at query time.
        _externalPrerequisite = externalPrerequisite;
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

        // Suppress GraphMode so Execute delegates call eager engine paths
        // instead of re-recording into the active scope.
        var savedScope = GraphMode.Current;
        GraphMode.SetCurrent(null);
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
        finally
        {
            GraphMode.SetCurrent(savedScope);
        }
    }

    private void RealizeInputs(IEngine engine)
    {
        // Realize the external prerequisite first when present. For in-place
        // ops this is the prior producer of `target` captured at record time —
        // resolving via Input0.LazySource alone would return THIS node after
        // RecordInPlace overwrites target.LazySource (self-loop).
        if (_externalPrerequisite is { IsRealized: false } pre && !ReferenceEquals(pre, this))
            pre.Realize(engine);
        // In-place op (Input0 IS the mutated target/output): the prior producer
        // of the target is _externalPrerequisite (realized above). Input0.LazySource
        // instead resolves to the LATEST in-place writer of the target — for an
        // EARLIER op in an in-place chain that is a LATER node, so following it
        // would recurse into a not-yet-realized successor. Skip it.
        bool input0IsInPlaceTarget = ReferenceEquals(Input0, Output);
        if (!input0IsInPlaceTarget && Input0.LazySource is ILazyNode n0 && !n0.IsRealized && !ReferenceEquals(n0, this)) n0.Realize(engine);
        if (Input1?.LazySource is ILazyNode n1 && !n1.IsRealized && !ReferenceEquals(n1, this)) n1.Realize(engine);
        if (Input2?.LazySource is ILazyNode n2 && !n2.IsRealized && !ReferenceEquals(n2, this)) n2.Realize(engine);
        if (InputsOverflow != null)
            foreach (var inp in InputsOverflow)
                if (inp.LazySource is ILazyNode n && !n.IsRealized && !ReferenceEquals(n, this)) n.Realize(engine);
    }

    /// <summary>Get input nodes for graph traversal.</summary>
    public ILazyNode[] GetInputNodes()
    {
        var nodes = new List<ILazyNode>();
        // External prerequisite first (when present) — see _externalPrerequisite
        // docstring. Filter out self-references so DCE / topo-sort / fusion do
        // not see a phantom self-edge for in-place ops where Input0 == Output.
        if (_externalPrerequisite is ILazyNode pre && !ReferenceEquals(pre, this))
            nodes.Add(pre);
        // In-place op (Input0 IS the mutated target/output): its prior producer is
        // _externalPrerequisite (added above). Input0.LazySource resolves to the
        // LATEST in-place writer of the target tensor; for the FIRST op in an
        // in-place chain that is a LATER node — a false back-edge that turns the
        // chain into a topo-sort cycle (e.g. M1↔A1 for runningVar *= m; += v), so
        // Kahn's reordering pass schedules NEITHER and silently DROPS the whole
        // in-place chain from the compiled plan. The dropped EMA never executes,
        // leaving the persistent running-stat tensor stale (BatchNorm clone bug).
        bool input0IsInPlaceTarget = ReferenceEquals(Input0, Output);
        if (!input0IsInPlaceTarget && Input0.LazySource is ILazyNode n0 && !ReferenceEquals(n0, this)) nodes.Add(n0);
        if (Input1?.LazySource is ILazyNode n1 && !ReferenceEquals(n1, this)) nodes.Add(n1);
        if (Input2?.LazySource is ILazyNode n2 && !ReferenceEquals(n2, this)) nodes.Add(n2);
        if (InputsOverflow != null)
            foreach (var inp in InputsOverflow)
                if (inp.LazySource is ILazyNode n && !ReferenceEquals(n, this)) nodes.Add(n);
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
