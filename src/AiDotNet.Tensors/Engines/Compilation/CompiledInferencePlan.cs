using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.Optimization;
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
    // Reference to the tensor the plan was traced against — distinct from
    // _compiledInputShape which is just a shape vector. Used by Then() so the
    // stitched-plan's boundary copy knows where to deposit the upstream
    // output. Null only for the degenerate empty-plan case (zero steps).
    private readonly Tensor<T>? _compiledInputTensor;
    private readonly List<GCHandle> _pinnedHandles = new();
    private bool _disposed;

    /// <summary>
    /// The output buffer produced by this plan's last step. Internal because
    /// <see cref="ICompiledPlan{T}.Then"/>'s stitching machinery needs to
    /// route data from this buffer into the next plan's captured input.
    /// </summary>
    internal Tensor<T> FinalOutputBuffer => _finalOutput;

    /// <summary>
    /// The captured-at-trace-time input tensor of this plan. Internal for the
    /// same reason as <see cref="FinalOutputBuffer"/>: stitching needs to
    /// know which buffer the next plan's first step reads from. Null for
    /// empty plans (no steps to consume an input).
    /// </summary>
    internal Tensor<T>? CompiledInputTensor => _compiledInputTensor;

    private CompiledInferencePlan(
        CompiledStep<T>[] steps,
        Tensor<T> finalOutput,
        IEngine engine,
        int[] inputShape,
        Tensor<T>? compiledInputTensor,
        List<GCHandle>? handles = null)
    {
        _steps = steps;
        _finalOutput = finalOutput;
        _engine = engine;
        _compiledInputShape = inputShape;
        _compiledInputTensor = compiledInputTensor;
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
    /// <remarks>
    /// Implementation: we splice the two plans' step arrays with a boundary
    /// step that copies <c>this.FinalOutputBuffer</c> into
    /// <c>nextPlan.CompiledInputTensor</c> in place. Both buffers were
    /// pre-allocated at compile time, so the copy moves data within existing
    /// storage — no <see cref="Tensor{T}"/> allocation between A and B,
    /// satisfying the issue's "zero materialization" acceptance criterion.
    /// True buffer aliasing (same storage object) is a future enhancement —
    /// would require Tensor surgery to swap underlying <c>_data</c>
    /// references on already-constructed tensors, which the closures inside
    /// each step's <c>Execute</c> delegate already captured at trace time.
    /// </remarks>
    public ICompiledPlan<T> Then(ICompiledPlan<T> next)
    {
        if (next is null) throw new ArgumentNullException(nameof(next));

        // Stitching needs to splice each plan's step array — that's a
        // concrete-type concern, not interface-level. Reject foreign
        // implementations cleanly rather than guessing.
        if (next is not CompiledInferencePlan<T> nextPlan)
            throw new NotSupportedException(
                $"{nameof(Then)} requires the next plan to be a built-in CompiledInferencePlan<T>. " +
                $"Got {next.GetType().FullName}. Third-party implementers can opt in by " +
                "providing their own concrete-type stitcher.");

        if (_steps.Length == 0)
            throw new ArgumentException(
                "Cannot stitch from an empty plan (no steps to feed into next).", nameof(next));
        if (nextPlan._steps.Length == 0)
            throw new ArgumentException(
                "Cannot stitch into an empty plan (no steps to consume the upstream output).", nameof(next));
        if (nextPlan._compiledInputTensor is null)
            throw new ArgumentException(
                "Next plan has no captured input tensor — it cannot be stitched onto.", nameof(next));

        // Validate at stitch time, not at execute time, per acceptance
        // criterion #3. Compare _shape arrays element-wise rather than
        // SequenceEqual so the error message can name the mismatching dim.
        var thisOut  = _finalOutput._shape;
        var nextIn   = nextPlan._compiledInputTensor._shape;
        if (thisOut.Length != nextIn.Length || !ShapesEqual(thisOut, nextIn))
            throw new ArgumentException(
                $"Cannot stitch: this plan's output shape [{string.Join(", ", thisOut)}] " +
                $"does not match next plan's input shape [{string.Join(", ", nextIn)}]. " +
                "Stitching requires shape-equal boundary tensors.",
                nameof(next));

        // Boundary step: copy this._finalOutput → next._compiledInputTensor.
        // The Execute delegate ignores its `output` parameter because the
        // boundary writes to a fixed buffer (next's captured input), not to
        // a per-step output buffer. CompiledStep's contract permits this —
        // OutputBuffer is also next._compiledInputTensor for diagnostics.
        var fromBuffer = _finalOutput;
        var toBuffer   = nextPlan._compiledInputTensor;
        var boundaryStep = new CompiledStep<T>(
            opName: "stitch.boundary",
            execute: (eng, _) => fromBuffer.AsSpan().CopyTo(toBuffer.AsWritableSpan()),
            outputBuffer: toBuffer,
            inputs: new[] { fromBuffer });

        // Splice: [thisSteps..., boundary, nextSteps...]
        var combined = new CompiledStep<T>[_steps.Length + 1 + nextPlan._steps.Length];
        Array.Copy(_steps, 0, combined, 0, _steps.Length);
        combined[_steps.Length] = boundaryStep;
        Array.Copy(nextPlan._steps, 0, combined, _steps.Length + 1, nextPlan._steps.Length);

        // The stitched plan inherits this plan's input shape (callers re-use
        // their existing IsValid checks) but reports next plan's final
        // output as its result. Pinned handles stay with the originals —
        // the stitched plan owns no new pins, so its Dispose is a no-op
        // for handles. The originals must outlive the stitched plan; the
        // xmldoc on Then() makes that ownership contract explicit.
        return new CompiledInferencePlan<T>(
            steps: combined,
            finalOutput: nextPlan._finalOutput,
            engine: _engine,
            inputShape: (int[])_compiledInputShape.Clone(),
            compiledInputTensor: _compiledInputTensor,
            handles: null);
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
    /// Throws ObjectDisposedException if the plan has been disposed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Tensor<T> Execute()
    {
        if (_disposed) throw new ObjectDisposedException(nameof(CompiledInferencePlan<T>));
        var steps = _steps;
        var engine = _engine;
        for (int i = 0; i < steps.Length; i++)
        {
            steps[i].Execute(engine, steps[i].OutputBuffer);
        }
        return _finalOutput;
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
    /// Compiles a lazy tensor scope into an inference plan.
    /// Runs optimization passes, pre-allocates all buffers, and builds step array.
    /// </summary>
    internal static CompiledInferencePlan<T> Compile(LazyTensorScope scope, IEngine engine)
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
        // and capture the tensor reference itself for Then() stitching.
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

        // Use the last optimized step's output (may differ from original after fusion/spectral passes)
        var finalOutput = optimizedSteps.Length > 0 ? optimizedSteps[optimizedSteps.Length - 1].OutputBuffer : new Tensor<T>(new int[] { 0 });
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
            new PointwiseFusionPass(),
            new AttentionFusionPass(),
            new BlasBatchPass(),
            new SpectralDecompositionPass(),
            new DataflowFusionPass(),
            new MixedPrecisionPass(),
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
