using System.Diagnostics;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines.Compilation.Serialization;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A compiled inference plan — a flat array of pre-resolved execution steps.
/// Zero overhead per-operation: no graph traversal, no shape validation, no allocation.
/// All buffers are pre-allocated at compile time and reused across replays.
///
/// Implements ICompiledPlan for public API access and IDisposable for GCHandle cleanup.
/// </summary>
internal sealed class CompiledInferencePlan<T> : ICompiledPlan<T>
{
    private readonly CompiledStep<T>[] _steps;
    private readonly Tensor<T> _finalOutput;
    private readonly IEngine _engine;
    private readonly int[] _compiledInputShape;
    // Tensors the plan was traced against — the array is the primary storage
    // so SetInputs generalizes cleanly to multi-input graphs. Today Compile()
    // captures exactly one entry (steps[0].Inputs[0]); discovering all leaf
    // inputs of a LazyTensorScope for true multi-input plans is a future
    // Compile() refactor. After a ThenAsync rebind, the first entry's storage
    // is aliased to the upstream plan's final output so downstream steps read
    // data written by the upstream steps without a boundary copy. First entry
    // is also serialized so a deserialized plan can be re-stitched or re-bound.
    // Array is never null but may be empty (degenerate empty-plan case).
    private readonly Tensor<T>[] _compiledInputTensors;
    private readonly List<GCHandle> _pinnedHandles = new();

    // Source plans that were stitched to form this plan — empty for plans
    // constructed from a scope compile, length 2 for each ThenAsync result.
    // Held for two reasons:
    //   1. Keep A and B GC-reachable through the stitched plan's lifetime,
    //      matching the xmldoc's "caller retains ownership" contract — the
    //      caller cannot accidentally let the sources be collected while
    //      the stitched plan is alive and in use.
    //   2. Let Execute() detect the "caller disposed a source prematurely"
    //      bug and throw a crisp ObjectDisposedException instead of running
    //      over freed GC-handle memory.
    // Initialized to an empty array (not null) so the Execute loop can
    // unconditionally iterate.
    private readonly CompiledInferencePlan<T>[] _sourcePlans;

    private bool _disposed;

    /// <summary>
    /// Upstream final-output tensor that this plan was last stitched onto,
    /// or <c>null</c> if this plan has never been a stitch target.
    /// <see cref="ThenAsync"/> rebinds this plan's captured input to the
    /// upstream's output, so a second stitch targeting this plan must
    /// either reuse the same upstream tensor (idempotent; fine — the
    /// associativity case <c>(A.Then B).Then C</c> vs <c>A.Then(B.Then C)</c>
    /// rebinds every intermediate to the same target) or go through a fresh
    /// compiled plan. Pointing the rebind at a different upstream would
    /// silently rewire any earlier stitched pipeline — the bug we're
    /// guarding against.
    /// </summary>
    private Tensor<T>? _lastStitchUpstream;

    /// <summary>
    /// Whether this plan has been disposed. Internal because
    /// <see cref="ThenAsync"/>'s stitched-plan Execute uses it to detect
    /// a caller who disposed a source plan while the stitched plan is
    /// still alive.
    /// </summary>
    internal bool IsDisposed => _disposed;

    /// <summary>
    /// The output buffer produced by this plan's last step. Internal because
    /// <see cref="ICompiledPlan{T}.ThenAsync"/>'s stitching machinery needs to
    /// route data from this buffer into the next plan's captured input.
    /// </summary>
    internal Tensor<T> FinalOutputBuffer => _finalOutput;

    /// <summary>
    /// The captured-at-trace-time input tensor of this plan (first entry of
    /// <c>_compiledInputTensors</c>). Internal for the same reason as
    /// <see cref="FinalOutputBuffer"/>: stitching needs to rebind the
    /// downstream plan's storage to point at the upstream plan's output.
    /// Null for empty plans (no steps to consume an input). Multi-input
    /// plans — when Compile() eventually supports them — expose additional
    /// captured tensors through <see cref="SetInputs"/>; stitching remains
    /// a single-input operation since rebinding "the" input is ambiguous
    /// for N>1.
    /// </summary>
    internal Tensor<T>? CompiledInputTensor =>
        _compiledInputTensors.Length > 0 ? _compiledInputTensors[0] : null;

    private CompiledInferencePlan(
        CompiledStep<T>[] steps,
        Tensor<T> finalOutput,
        IEngine engine,
        int[] inputShape,
        Tensor<T>? compiledInputTensor,
        List<GCHandle>? handles = null,
        CompiledInferencePlan<T>[]? sourcePlans = null)
    {
        _steps = steps;
        _finalOutput = finalOutput;
        _engine = engine;
        _compiledInputShape = inputShape;
        // Compile() / CreateFromDeserialized currently capture one input,
        // so the array has length 0 or 1 in practice. Storing it as an array
        // lets SetInputs generalize cleanly when multi-input Compile lands.
        _compiledInputTensors = compiledInputTensor is null
            ? Array.Empty<Tensor<T>>()
            : new[] { compiledInputTensor };
        _sourcePlans = sourcePlans ?? Array.Empty<CompiledInferencePlan<T>>();
        if (handles is not null)
            _pinnedHandles.AddRange(handles);
    }

    /// <summary>Number of compiled steps.</summary>
    public int StepCount => _steps.Length;

    /// <summary>Checks whether this plan was compiled for the given input shape.</summary>
    public bool IsValid(int[] inputShape)
    {
        if (inputShape.Length != _compiledInputShape.Length) return false;
        for (int i = 0; i < inputShape.Length; i++)
            if (inputShape[i] != _compiledInputShape[i]) return false;
        return true;
    }

    /// <inheritdoc/>
    public Task SaveAsync(Stream stream, CancellationToken cancellationToken = default)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>));
        cancellationToken.ThrowIfCancellationRequested();

        // Current on-disk format captures a single input tensor. When
        // Compile() starts producing multi-input plans, bump the format
        // version and extend the writer to emit every entry.
        var firstInput = _compiledInputTensors.Length > 0 ? _compiledInputTensors[0] : null;
        InferencePlanWriter.Write(stream, _steps, _finalOutput, _compiledInputShape, firstInput);
        return Task.CompletedTask;
    }

    /// <inheritdoc/>
    public bool IsCompatibleWith(PlanCompatibilityInfo info)
    {
        return info.GetIncompatibilityReason<T>() is null;
    }

    /// <summary>
    /// Internal factory for the deserialization path — constructs a plan from
    /// pre-built steps without running the compiler or optimization passes.
    /// The steps' closures were reconstructed by the op registry; the tensor
    /// buffers were reconstituted from the tensor table. This is the "zero
    /// warmup" entry point: load from disk → ready to Execute() immediately.
    /// </summary>
    internal static CompiledInferencePlan<T> CreateFromDeserialized(
        CompiledStep<T>[] steps,
        Tensor<T> finalOutput,
        IEngine engine,
        int[] inputShape,
        Tensor<T>? compiledInputTensor)
    {
        // Run TryBuildSpecializedForward on the deserialized steps so the
        // loaded plan uses the same SIMD-tuned kernels as the original.
        // Without this, the loaded plan's generic engine-method closures
        // produce slightly different floating-point results due to
        // accumulation-order differences (6th decimal place on MatMul).
        // Graph-level optimization passes are NOT re-run — the saved plan
        // was already optimized before serialization.
        var pinnedHandles = new List<GCHandle>();
        var specialized = new CompiledStep<T>[steps.Length];
        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            var spec = CompiledTrainingPlan<T>.TryBuildSpecializedForward(step, pinnedHandles);
            if (spec != null)
            {
                var output = step.OutputBuffer;
                specialized[i] = new CompiledStep<T>(
                    step.OpName,
                    (eng, o) => spec(eng),
                    output,
                    step.Inputs,
                    step.BackwardFn,
                    step.SavedState);
            }
            else
            {
                specialized[i] = step;
            }
        }

        // Clear LazySource on output tensors (same as Compile does).
        foreach (var step in specialized)
            step.OutputBuffer.LazySource = null;

        var actualFinalOutput = specialized.Length > 0
            ? specialized[specialized.Length - 1].OutputBuffer
            : finalOutput;

        return new CompiledInferencePlan<T>(
            specialized, actualFinalOutput, engine, inputShape,
            handles: pinnedHandles, compiledInputTensor: compiledInputTensor);
    }

    // ── Internal accessors for serialization ────────────────────────────
    // Note: CompiledInputTensor is already defined above for ThenAsync stitching.
    internal CompiledStep<T>[] Steps => _steps;
    internal int[] CompiledInputShape => _compiledInputShape;

    /// <inheritdoc/>
    /// <remarks>
    /// <para>
    /// Implementation: we rebind <paramref name="next"/>'s captured input
    /// tensor to share backing storage with <c>this.FinalOutputBuffer</c>
    /// (via <see cref="TensorBase{T}.RebindStorageFrom"/>), then splice the
    /// two plans' step arrays into one flat array —
    /// <c>[this.steps, next.steps]</c>. No boundary step. The upstream plan's
    /// last step writes into the same <see cref="Vector{T}"/> the downstream
    /// plan's first step reads from, so data flows through without any
    /// per-execute copy. Satisfies the issue's "same object reference,
    /// no copy" semantic to the strongest degree possible given the existing
    /// Tensor contract (Tensor OBJECT identity is not restored — the source
    /// and target Tensor instances stay distinct — but they share a single
    /// backing Vector and TensorStorage after rebind).
    /// </para>
    /// <para>
    /// <b>Side effect on <paramref name="next"/>:</b> after this call,
    /// <paramref name="next"/>'s captured input tensor is aliased to
    /// <c>this.FinalOutputBuffer</c>. Running <paramref name="next"/>
    /// standalone afterwards will read data produced by <c>this</c>'s last
    /// execution, not whatever the caller might write into
    /// <paramref name="next"/>'s input slot. The documented contract is:
    /// once you stitch a plan into a pipeline, prefer to drive it through
    /// the stitched plan's <see cref="Execute"/>.
    /// </para>
    /// </remarks>
    public ICompiledPlan<T> ThenAsync(ICompiledPlan<T> next)
    {
        if (next is null) throw new ArgumentNullException(nameof(next));

        // Stitching needs to splice each plan's step array — that's a
        // concrete-type concern, not interface-level. Reject foreign
        // implementations cleanly rather than guessing.
        if (next is not CompiledInferencePlan<T> nextPlan)
            throw new NotSupportedException(
                $"{nameof(ThenAsync)} requires the next plan to be a built-in CompiledInferencePlan<T>. " +
                $"Got {next.GetType().FullName}. Third-party implementers can opt in by " +
                "providing their own concrete-type stitcher.");

        // Both sources must still be alive; stitching a disposed plan would
        // splice references to freed GCHandles.
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>),
            "Cannot stitch from a disposed plan.");
        if (nextPlan._disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>),
            "Cannot stitch onto a disposed plan.");

        if (_steps.Length == 0)
            throw new ArgumentException(
                "Cannot stitch from an empty plan (no steps to feed into next).", nameof(next));
        if (nextPlan._steps.Length == 0)
            throw new ArgumentException(
                "Cannot stitch into an empty plan (no steps to consume the upstream output).", nameof(next));
        // Stitching is a single-input operation; multi-input downstream plans
        // would need to pick which input to rebind and that's ambiguous. Today
        // Compile() only produces 1-input plans so the first clause always
        // holds, but the explicit length check keeps the contract honest when
        // multi-input Compile lands later.
        if (nextPlan._compiledInputTensors.Length == 0)
            throw new ArgumentException(
                "Next plan has no captured input tensor — it cannot be stitched onto.", nameof(next));
        if (nextPlan._compiledInputTensors.Length > 1)
            throw new NotSupportedException(
                "ThenAsync stitching is only supported for single-input plans. " +
                $"Next plan has {nextPlan._compiledInputTensors.Length} captured inputs.");
        var nextPlanInput = nextPlan._compiledInputTensors[0];

        // Reject fan-out reuse from a DIFFERENT upstream. Stitching rebinds
        // nextPlan's input storage to this plan's output; calling Then again
        // on the same nextPlan from a different upstream would silently
        // rewire any earlier stitched pipeline because nextPlan is shared by
        // reference. Idempotent re-stitch to the same upstream is allowed —
        // the associativity case (A.Then B).Then C vs A.Then(B.Then C) ends
        // up rebinding every intermediate to the same target tensor.
        if (nextPlan._lastStitchUpstream is not null &&
            !ReferenceEquals(nextPlan._lastStitchUpstream, _finalOutput))
        {
            throw new InvalidOperationException(
                "This plan has already been stitched onto a different upstream. " +
                "Reusing it as a stitch target with a new upstream would rebind " +
                "its input storage and silently invalidate the earlier pipeline. " +
                "Recompile the downstream plan to get a fresh instance for each " +
                "distinct upstream.");
        }

        // Validate at stitch time, not at execute time, per acceptance
        // criterion #3. Compare _shape arrays element-wise so the error
        // message can name the mismatching dim.
        var thisOut  = _finalOutput._shape;
        var nextIn   = nextPlanInput._shape;
        if (thisOut.Length != nextIn.Length || !ShapesEqual(thisOut, nextIn))
            throw new ArgumentException(
                $"Cannot stitch: this plan's output shape [{string.Join(", ", thisOut)}] " +
                $"does not match next plan's input shape [{string.Join(", ", nextIn)}]. " +
                "Stitching requires shape-equal boundary tensors.",
                nameof(next));

        // Rebind nextPlan's captured input to share storage with this plan's
        // final output. After this point, writes to _finalOutput are
        // immediately visible to nextPlan's first step's closure reads — no
        // boundary memcpy, no intermediate materialization. RebindStorageFrom
        // validates shape/contiguity/offset-zero and throws descriptively
        // if the invariants don't hold (pre-filtered above, so in practice
        // this is a belt-and-suspenders check).
        nextPlanInput.RebindStorageFrom(_finalOutput);
        nextPlan._lastStitchUpstream = _finalOutput;

        // Splice: [thisSteps..., nextSteps...]. No boundary step — the
        // rebind above is a one-shot stitch-time operation, not a
        // per-execute copy.
        //
        // Cross-engine handling: if the two plans were compiled against
        // different engines (e.g. plan A with CpuEngine, plan B with
        // DirectGpuEngine), running B's steps with A's engine produces
        // wrong results — each op is dispatched to the wrong kernel set.
        // The previous implementation handed the stitched plan's single
        // _engine to every step's Execute, so cross-engine stitches silently
        // computed garbage. Wrap each of nextPlan's steps in a shim that
        // substitutes its original engine. Plan A's steps keep their
        // engine binding via the outer loop.
        var combined = new CompiledStep<T>[_steps.Length + nextPlan._steps.Length];
        Array.Copy(_steps, 0, combined, 0, _steps.Length);
        bool crossEngine = !ReferenceEquals(_engine, nextPlan._engine);
        if (crossEngine)
        {
            Trace.WriteLine(
                $"[CompiledInferencePlan] Stitching across engines: " +
                $"{_engine.GetType().Name} → {nextPlan._engine.GetType().Name}. " +
                "Next plan's steps will run against their original engine; " +
                "the boundary tensor is aliased via RebindStorageFrom, so the " +
                "device-transfer cost is paid by the first cross-engine read.");
            var nextEngine = nextPlan._engine;
            for (int i = 0; i < nextPlan._steps.Length; i++)
            {
                var original = nextPlan._steps[i];
                var originalExecute = original.Execute;
                // The outer loop passes _engine (plan A's) to this delegate;
                // we ignore it and route to nextEngine (plan B's original).
                Action<IEngine, Tensor<T>> rewrapped = (_, output) => originalExecute(nextEngine, output);
                combined[_steps.Length + i] = new CompiledStep<T>(
                    original.OpName,
                    rewrapped,
                    original.OutputBuffer,
                    original.Inputs,
                    original.BackwardFn,
                    original.SavedState);
            }
        }
        else
        {
            Array.Copy(nextPlan._steps, 0, combined, _steps.Length, nextPlan._steps.Length);
        }

        // The stitched plan inherits this plan's input shape (callers re-use
        // their existing IsValid checks) but reports next plan's final
        // output as its result. Pinned handles stay with the originals —
        // the stitched plan owns no new pins, so its Dispose is a no-op
        // for handles. The sourcePlans array holds strong references to
        // every LEAF plan (not just the immediate operands): if this or
        // nextPlan is itself a previously-stitched result, we flatten its
        // leaves into the new array so the Execute disposed-source check
        // sees every backing plan. Without this, A.Then(B).Then(C) would
        // store only [stitchedAB, C] and disposing A silently leaves the
        // outer pipeline running against freed GCHandles.
        var leftSources = _sourcePlans.Length == 0
            ? new[] { this }
            : _sourcePlans;
        var rightSources = nextPlan._sourcePlans.Length == 0
            ? new[] { nextPlan }
            : nextPlan._sourcePlans;
        var stitchedSources = new CompiledInferencePlan<T>[leftSources.Length + rightSources.Length];
        Array.Copy(leftSources, 0, stitchedSources, 0, leftSources.Length);
        Array.Copy(rightSources, 0, stitchedSources, leftSources.Length, rightSources.Length);

        return new CompiledInferencePlan<T>(
            steps: combined,
            finalOutput: nextPlan._finalOutput,
            engine: _engine,
            inputShape: (int[])_compiledInputShape.Clone(),
            compiledInputTensor: _compiledInputTensors.Length > 0 ? _compiledInputTensors[0] : null,
            handles: null,
            sourcePlans: stitchedSources);
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>
    /// Executes the compiled plan. Runs each step's delegate in order.
    /// All buffers are pre-allocated - zero allocation during execution.
    /// Throws ObjectDisposedException if the plan has been disposed
    /// OR if any stitched source plan has been disposed (stitched plans
    /// share step references with their sources; running over a disposed
    /// source's freed GCHandles would be undefined behaviour).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Execute()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>));

        // Stitched-plan guard: the steps array holds references into each
        // source plan's pre-allocated buffers. If the caller disposed A or
        // B while the stitched plan is still alive, their pinned GCHandles
        // are freed and running over them silently corrupts memory. Raise
        // a clean error instead. Iteration is zero-cost for the common
        // non-stitched case because _sourcePlans is the sentinel empty array.
        for (int i = 0; i < _sourcePlans.Length; i++)
        {
            if (_sourcePlans[i]._disposed)
                throw new ObjectDisposedException(
                    nameof(CompiledInferencePlan<T>),
                    $"Stitched plan's source at index {i} has been disposed. " +
                    "The caller must keep all plans passed to ThenAsync alive " +
                    "for the lifetime of the stitched result.");
        }

        var steps = _steps;
        var engine = _engine;
        for (int i = 0; i < steps.Length; i++)
        {
            steps[i].Execute(engine, steps[i].OutputBuffer);
        }
        return _finalOutput;
    }

    /// <inheritdoc/>
    public void ExecuteInto(Tensor<T> output)
    {
        if (output is null) throw new ArgumentNullException(nameof(output));
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>));

        // Validate shape up front so we throw before any step runs. Same
        // shape guards RebindStorageFrom would raise, but surfaced here so
        // the exception trace points at the user's call site.
        ValidateShapesMatch(_finalOutput, output, nameof(output));

        // Run the plan into its internal buffer, then copy the final tensor's
        // data into the caller's buffer. Copy (not rebind) because many
        // specialized kernel paths capture array references at compile time
        // via GetDataArray() closure — a post-compile storage rebind would
        // not reach those closures, leaving the caller's buffer untouched.
        // The extra copy is one memcpy of the output-tensor size per call;
        // under CUDA graph capture it's recorded as a device memcpy node and
        // replays correctly every iteration.
        Execute();
        _finalOutput.AsSpan().CopyTo(output.AsWritableSpan());
    }

    /// <inheritdoc/>
    public void SetInputs(Tensor<T>[] inputs)
    {
        if (inputs is null) throw new ArgumentNullException(nameof(inputs));
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>));
        // The uniform length-match check below handles every case: zero-input
        // plans accept SetInputs(Array.Empty<Tensor<T>>()) as a no-op, N-input
        // plans require exactly N, and the error names the expected count.
        // No special case for empty plans — callers get one consistent
        // contract across 0-, 1-, and future N-input graphs.
        if (inputs.Length != _compiledInputTensors.Length)
            throw new ArgumentException(
                $"This plan was compiled with {_compiledInputTensors.Length} captured input(s); " +
                $"got {inputs.Length}.",
                nameof(inputs));

        // Copy (not rebind): the specialized kernels capture each input's
        // array reference at compile time (see
        // CompiledTrainingPlan.TryBuildSpecializedForward), so a storage
        // rebind after compile is invisible to them. Copying refreshes the
        // captured buffer in-place — the kernels read the new data next
        // Execute and stay graph-capture-safe.
        for (int i = 0; i < inputs.Length; i++)
        {
            var src = inputs[i] ?? throw new ArgumentException(
                $"inputs[{i}] is null.", nameof(inputs));
            ValidateShapesMatch(_compiledInputTensors[i], src, $"inputs[{i}]");
            src.AsSpan().CopyTo(_compiledInputTensors[i].AsWritableSpan());
        }
    }

    private static void ValidateShapesMatch(Tensor<T> expected, Tensor<T> actual, string paramName)
    {
        if (expected._shape.Length != actual._shape.Length)
            throw new ArgumentException(
                $"{paramName} rank {actual._shape.Length} != plan rank {expected._shape.Length}.",
                paramName);
        for (int i = 0; i < expected._shape.Length; i++)
        {
            if (expected._shape[i] != actual._shape[i])
                throw new ArgumentException(
                    $"{paramName} shape [{string.Join(", ", actual._shape)}] " +
                    $"!= plan shape [{string.Join(", ", expected._shape)}].",
                    paramName);
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        foreach (var handle in _pinnedHandles)
        {
            if (handle.IsAllocated)
                handle.Free();
        }
        _pinnedHandles.Clear();
    }

    /// <summary>
    /// Per-step timing instrumentation for systematic-debugging Phase 1
    /// (identify which compiled steps dominate plan cost). Runs the same
    /// step-loop as <see cref="Execute"/> but times each step individually
    /// with Stopwatch and accumulates per-step ticks across <paramref name="iters"/>
    /// measurement iterations. Warms up for <paramref name="warmup"/> full
    /// plan runs first so JIT / cache effects are amortised.
    ///
    /// <para>Internal because this is a diagnostic tool — not part of the
    /// public ICompiledPlan contract. The bookkeeping per-step (~20 ns for
    /// two Stopwatch.GetTimestamp calls) would be visible on workloads with
    /// sub-µs steps, so we DON'T fold this into the production Execute path.</para>
    ///
    /// <para>Returns a tuple per step: the OpName of the underlying lazy op
    /// (same string the LazyNode carried), and the average wall-clock ms
    /// that step took across the measurement iterations.</para>
    /// </summary>
    internal (string OpName, double AvgMs)[] ProfilePerStep(int warmup, int iters)
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>));
        if (warmup < 0) throw new ArgumentOutOfRangeException(nameof(warmup));
        if (iters <= 0) throw new ArgumentOutOfRangeException(nameof(iters));

        var steps = _steps;
        var engine = _engine;

        // Warm up — run the full plan N times so JIT / CPU cache / branch
        // predictor / allocator state matches steady-state conditions.
        // Guard stitched sources before EACH warmup iteration (same contract
        // as the measurement loop below): profiling a stitched plan whose
        // source was disposed during warmup used to crash reading freed
        // GCHandles instead of throwing the clean ObjectDisposedException.
        for (int w = 0; w < warmup; w++)
        {
            for (int s = 0; s < _sourcePlans.Length; s++)
            {
                if (_sourcePlans[s]._disposed)
                    throw new ObjectDisposedException(
                        nameof(CompiledInferencePlan<T>),
                        $"Stitched plan's source at index {s} has been disposed during warmup.");
            }
            for (int i = 0; i < steps.Length; i++)
                steps[i].Execute(engine, steps[i].OutputBuffer);
        }

        var tickSums = new long[steps.Length];
        for (int it = 0; it < iters; it++)
        {
            // Match Execute's stitched-disposal safety even in the profiling
            // path — a caller profiling a stitched plan whose source was
            // disposed mid-run should see the same crisp error, not a crash.
            for (int s = 0; s < _sourcePlans.Length; s++)
            {
                if (_sourcePlans[s]._disposed)
                    throw new ObjectDisposedException(
                        nameof(CompiledInferencePlan<T>),
                        $"Stitched plan's source at index {s} has been disposed during profiling.");
            }

            for (int k = 0; k < steps.Length; k++)
            {
                long t0 = Stopwatch.GetTimestamp();
                steps[k].Execute(engine, steps[k].OutputBuffer);
                long t1 = Stopwatch.GetTimestamp();
                tickSums[k] += (t1 - t0);
            }
        }

        double tickToMs = 1000.0 / Stopwatch.Frequency;
        var result = new (string, double)[steps.Length];
        for (int k = 0; k < steps.Length; k++)
            result[k] = (steps[k].OpName, tickSums[k] * tickToMs / iters);
        return result;
    }

    /// <summary>
    /// Compiles a lazy tensor scope into an inference plan.
    /// Runs optimization passes, pre-allocates all buffers, and builds step array.
    /// </summary>
    /// <param name="scope">The traced lazy graph.</param>
    /// <param name="engine">Execution engine.</param>
    /// <param name="explicitOutput">
    /// The tensor the caller's forward lambda returned, or <c>null</c> to
    /// fall back to the last-optimized-step heuristic. Passing the caller's
    /// actual output tensor is what fixes issue #228: a forward that ends in
    /// a pure-view op (<c>Reshape</c> on contiguous data, <c>Squeeze</c>, ...)
    /// or has host-side control flow that conditionally appends such a view
    /// no longer leaks the wrong tensor through <c>Execute()</c>.
    /// </param>
    internal static CompiledInferencePlan<T> Compile(
        LazyTensorScope scope, IEngine engine, Tensor<T>? explicitOutput)
    {
        var compiler = new LazyGraphCompiler();
        var optimized = compiler.Compile(scope.Nodes);

        var steps = new List<CompiledStep<T>>();
        foreach (var node in optimized)
        {
            if (node is LazyNode<T> typed)
            {
                steps.Add(new CompiledStep<T>(
                    typed.OpName,
                    typed.Execute,
                    typed.Output,
                    typed.GetInputsArray(),
                    typed.BackwardFn,
                    typed.SavedState));
            }
        }

        // Track GCHandles for cleanup on Dispose
        var pinnedHandles = new List<GCHandle>();

        // Determine input shape from the first step's inputs (for IsValid check)
        // and capture the tensor reference itself for Then() stitching + serialization.
        var inputTensor = steps.Count > 0 && steps[0].Inputs.Length > 0
            ? steps[0].Inputs[0]
            : null;
        var inputShape = inputTensor is not null
            ? (int[])inputTensor._shape.Clone()
            : Array.Empty<int>();

        // Build specialized forward actions (same optimization as CompiledTrainingPlan)
        var specializedSteps = new CompiledStep<T>[steps.Count];
        for (int i = 0; i < steps.Count; i++)
        {
            var step = steps[i];

            // Transpose optimization: use fast Data.Span path when input is contiguous
            // with zero offset, fall back to eng.TensorTranspose for views/slices
            if (step.OpType == OpType.TensorTranspose && step.Inputs.Length == 1 && step.Inputs[0].Rank == 2)
            {
                var capturedInput = step.Inputs[0];
                var capturedOutput = step.OutputBuffer;
                bool canUseFastPath = capturedInput.IsContiguous && capturedInput._storageOffset == 0;

                if (canUseFastPath)
                {
                    // Fast path: direct data access (zero-offset contiguous tensor)
                    int rows = capturedInput._shape[0];
                    int cols = capturedInput._shape[1];
                    specializedSteps[i] = new CompiledStep<T>(
                        step.OpName,
                        (eng, o) =>
                        {
                            var src = capturedInput.GetDataArray();
                            var dst = capturedOutput.GetDataArray();
                            for (int r = 0; r < rows; r++)
                                for (int c = 0; c < cols; c++)
                                    dst[c * rows + r] = src[r * cols + c];
                        },
                        step.OutputBuffer,
                        step.Inputs,
                        step.BackwardFn,
                        step.SavedState);
                }
                else
                {
                    // Safe path: use engine transpose for views/slices with offset
                    specializedSteps[i] = new CompiledStep<T>(
                        step.OpName,
                        (eng, o) =>
                        {
                            var transposed = eng.TensorTranspose(capturedInput);
                            transposed.AsSpan().CopyTo(capturedOutput.AsWritableSpan());
                        },
                        step.OutputBuffer,
                        step.Inputs,
                        step.BackwardFn,
                        step.SavedState);
                }
                continue;
            }

            var specialized = CompiledTrainingPlan<T>.TryBuildSpecializedForward(step, pinnedHandles);
            if (specialized != null)
            {
                // Wrap the specialized action as a CompiledStep with the optimized execute
                var output = step.OutputBuffer;
                var action = specialized;
                specializedSteps[i] = new CompiledStep<T>(
                    step.OpName,
                    (eng, o) => action(eng),
                    output,
                    step.Inputs,
                    step.BackwardFn,
                    step.SavedState);
            }
            else
            {
                specializedSteps[i] = step;
            }
        }

        // Run CPU-level optimization passes (spectral decomposition, dataflow fusion)
        var optimizedSteps = RunCpuOptimizationPasses(specializedSteps, engine);

        // Clear LazySource on all compiled output tensors to prevent auto-materialization
        // from re-triggering lazy graph execution after compilation
        foreach (var step in optimizedSteps)
            step.OutputBuffer.LazySource = null;

        // Prefer the caller's returned tensor (#228). When the forward ends in
        // a pure-view op, explicitOutput shares storage with one of the step
        // output buffers — writes to the producer buffer flow through the view
        // at Execute() time. When the forward returns a regular lazy-node
        // output, explicitOutput is that node's output buffer itself.
        //
        // Legacy no-explicit-output path still uses the last-step heuristic so
        // internal callers (AutoTracer, tests constructing scopes directly)
        // keep working unchanged.
        Tensor<T> finalOutput;
        if (explicitOutput is not null)
        {
            explicitOutput.LazySource = null;
            finalOutput = explicitOutput;
        }
        else
        {
            finalOutput = optimizedSteps.Length > 0
                ? optimizedSteps[optimizedSteps.Length - 1].OutputBuffer
                : new Tensor<T>(new int[] { 0 });
        }
        return new CompiledInferencePlan<T>(optimizedSteps, finalOutput, engine, inputShape, inputTensor, pinnedHandles);
    }

    /// <summary>
    /// Runs CPU-level optimization passes on the compiled steps.
    /// Currently: spectral decomposition (Phase A) and dataflow fusion (Phase B).
    /// Each pass is independently toggleable via TensorCodecOptions.
    /// </summary>
    private static CompiledStep<T>[] RunCpuOptimizationPasses(CompiledStep<T>[] steps, IEngine engine)
    {
        // A/B testing showed that optimization passes have net NEGATIVE value for small-to-medium
        // inference plans. The raw compiled plan (direct BLAS/SIMD dispatch) is already 300-4500x
        // faster than eager. Passes add compile-time overhead that doesn't pay off at runtime.
        //
        // Only run passes for large plans (20+ steps) where fusion can amortize the overhead,
        // or when the plan contains specific patterns that passes target (Conv+BN, attention).
        const int MinStepsForPasses = 20;
        bool hasConvOrAttention = false;
        for (int i = 0; i < steps.Length && !hasConvOrAttention; i++)
        {
            bool isHeavyOp = steps[i].OpType is OpType.Conv2D or OpType.DepthwiseConv2D
                or OpType.BatchNorm or OpType.TensorMatMul;
            if (isHeavyOp && steps.Length >= 8)
                hasConvOrAttention = true;
        }

        if (steps.Length < MinStepsForPasses && !hasConvOrAttention)
            return steps;

        ICpuOptimizationPass[] passes =
        {
            new ConstantFoldingPass(),
            new ForwardCSEPass(),
            new ConvBnFusionPass(),
            new DiffusionFusionPass(), // Patterns 11-14: GroupNorm+SiLU, Conv+Bias+SiLU, Add+GroupNorm (#181)
            new PointwiseFusionPass(),
            new AttentionFusionPass(),
            new BlasBatchPass(),
            new SpectralDecompositionPass(),
            new DataflowFusionPass(),
            new MixedPrecisionPass(),
            new OperatorReorderingPass(), // Reorder for cache locality (#182)
            new MemoryPlanningPass(),     // Buffer reuse via lifetime analysis (#182)
            new TileSchedulingPass(),     // L1/L2 tile sizing for GEMM/Conv (#182)
        };

        var current = steps;
        foreach (var pass in passes)
        {
            if (!pass.IsEnabled) continue;
            var optimized = pass.TryOptimize(current, engine);
            if (optimized != null)
                current = optimized;
        }
        return current;
    }
}
