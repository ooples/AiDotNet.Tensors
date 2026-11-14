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

    /// <summary>
    /// True when the node's replay updates host state of its own every step (an unseeded random op drawing its step's
    /// seed). A captured CUDA graph does not re-run host code, so a plan holding such a node is never graph-captured.
    /// </summary>
    internal bool ReplaysHostState { get; set; }

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
    public long RecordOrder { get; set; }
    public bool WritesInPlace => ReferenceEquals(Input0, Output);
    public ILazyNode? PriorWriter => WritesInPlace ? _externalPrerequisite : null;

    // An in-place write's earlier readers of its target (recorded after the target's prior producer): the write must
    // run after them, or a reader scheduled later would see the updated value (write-after-read).
    private ILazyNode[]? _earlierReaders;

    internal void SetEarlierReaders(ILazyNode[] readers) => _earlierReaders = readers.Length == 0 ? null : readers;

    /// <summary>Whether <paramref name="tensor"/> is one of the node's inputs.</summary>
    internal bool Reads(Tensor<T> tensor)
    {
        if (ReferenceEquals(Input0, tensor) || ReferenceEquals(Input1, tensor) || ReferenceEquals(Input2, tensor)) return true;
        if (InputsOverflow is not null)
            foreach (var input in InputsOverflow)
                if (ReferenceEquals(input, tensor)) return true;
        return false;
    }

    private ILazyNode? ProducerOf(Tensor<T>? input) => LazyNodeOrder.ProducerAsOf(input?.LazySource, RecordOrder);
    public int ConsumerCount { get; set; }
    public IEngine RecordingEngine { get; set; } = AiDotNetEngine.Current;

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
        if (GraphMode.IsProbe) throw new ObjectiveProbeAbortedException(OpName);

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

            // Execute the operation, each input showing the producer it had when this node was recorded: a read of
            // an input auto-realizes its LazySource, which for a tensor updated in place later in the trace is that
            // later write. Realizing this node (an eager op reading its output mid-trace) then ran the write early,
            // and the compiled plan ran it again: an MoE's selection-bias nudge landed twice in the first step.
            var shown = ShowInputsAsRecorded();
            try
            {
                Execute(engine, Output);
            }
            finally
            {
                if (shown is not null)
                    foreach (var (input, latest) in shown) input.LazySource = latest;
            }
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

    // Points every input whose LazySource is a write recorded after this node back at its producer as of this node;
    // returns what to restore (null when nothing moved, the common case).
    private List<(Tensor<T> Input, ILazyNode? Latest)>? ShowInputsAsRecorded()
    {
        List<(Tensor<T>, ILazyNode?)>? shown = null;
        void Show(Tensor<T>? input)
        {
            if (input is null || ReferenceEquals(input, Output) || input.LazySource is not { } latest) return;
            var asOf = ProducerOf(input);
            if (ReferenceEquals(asOf, latest)) return;
            if (shown is not null)
                foreach (var (seen, _) in shown)
                    if (ReferenceEquals(seen, input)) return;
            (shown ??= new()).Add((input, latest));
            input.LazySource = asOf;
        }
        Show(Input0);
        Show(Input1);
        Show(Input2);
        if (InputsOverflow is not null)
            foreach (var input in InputsOverflow) Show(input);
        return shown;
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
        if (!input0IsInPlaceTarget && ProducerOf(Input0) is { IsRealized: false } n0 && !ReferenceEquals(n0, this)) n0.Realize(engine);
        if (ProducerOf(Input1) is { IsRealized: false } n1 && !ReferenceEquals(n1, this)) n1.Realize(engine);
        if (ProducerOf(Input2) is { IsRealized: false } n2 && !ReferenceEquals(n2, this)) n2.Realize(engine);
        if (InputsOverflow != null)
            foreach (var inp in InputsOverflow)
                if (ProducerOf(inp) is { IsRealized: false } n && !ReferenceEquals(n, this)) n.Realize(engine);
        // A write realizes the target's earlier readers first, so they read the value before it.
        if (_earlierReaders is not null)
            foreach (var reader in _earlierReaders)
                if (!reader.IsRealized && !ReferenceEquals(reader, this)) reader.Realize(engine);
    }

    /// <summary>The prior producer of an in-place target (see <c>_externalPrerequisite</c>), or null.</summary>
    internal ILazyNode? ExternalPrerequisite => _externalPrerequisite;

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
        if (!input0IsInPlaceTarget && ProducerOf(Input0) is { } n0 && !ReferenceEquals(n0, this)) nodes.Add(n0);
        if (ProducerOf(Input1) is { } n1 && !ReferenceEquals(n1, this)) nodes.Add(n1);
        if (ProducerOf(Input2) is { } n2 && !ReferenceEquals(n2, this)) nodes.Add(n2);
        if (InputsOverflow != null)
            foreach (var inp in InputsOverflow)
                if (ProducerOf(inp) is { } n && !ReferenceEquals(n, this)) nodes.Add(n);
        if (_earlierReaders is not null)
            foreach (var reader in _earlierReaders)
                if (!ReferenceEquals(reader, this)) nodes.Add(reader);
        return nodes.ToArray();
    }

    public bool SavePersistentWrite(HashSet<object> saved, List<Action> restores)
    {
        if (!ReferenceEquals(Input0, Output)) return false;
        // The target's producer before its chain of in-place writes; none means it came from outside the trace.
        var prior = _externalPrerequisite;
        while (prior is LazyNode<T> earlier && ReferenceEquals(earlier.Output, Output)) prior = earlier._externalPrerequisite;
        if (prior is not null) return false;
        if (saved.Add(Output))
        {
            // Detached from this node for the copy, so reading it (from the host or a device) does not run the write.
            var target = Output;
            var source = target.LazySource;
            target.LazySource = null;
            Tensor<T> copy;
            try { copy = target.Clone(); }
            finally { target.LazySource = source; }
            var engine = RecordingEngine;
            restores.Add(() => DirectGpuTensorEngine.CopyResultInto(engine, copy, target));
        }
        return true;
    }

    /// <summary>Clears the LazySource reference on the output tensor after realization.</summary>
    public void ClearOutputLazySource()
    {
        Output.LazySource = null;
    }

    /// <summary>
    /// Removes this abandoned node from its output without severing a producer from an enclosing
    /// scope. In-place recording temporarily replaces that producer with this node.
    /// </summary>
    public void RestoreOutputLazySource()
    {
        // Compilation takes ownership of output materialization by clearing LazySource before
        // later validation/specialization stages finish. If one of those stages throws, the
        // displaced producer is still the source this in-place node must restore even though the
        // node is no longer attached to the tensor. Reverse-order scope abandonment then walks an
        // in-place chain back to the producer owned by the enclosing scope.
        if (ReferenceEquals(Output.LazySource, this)
            || (Output.LazySource is null && _externalPrerequisite is not null))
        {
            Output.LazySource = _externalPrerequisite;
        }
    }

    public void AddStorageLeases(TensorStorageLeaseSet leases)
    {
        leases.Add(Input0);
        if (Input1 is not null) leases.Add(Input1);
        if (Input2 is not null) leases.Add(Input2);
        if (InputsOverflow is not null)
            for (int i = 0; i < InputsOverflow.Length; i++) leases.Add(InputsOverflow[i]);
        leases.Add(Output);
        leases.AddSavedState(SavedState);
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
