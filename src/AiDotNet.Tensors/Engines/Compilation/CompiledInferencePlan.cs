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
    private readonly List<GCHandle> _pinnedHandles = new();
    private bool _disposed;

    private CompiledInferencePlan(CompiledStep<T>[] steps, Tensor<T> finalOutput, IEngine engine, int[] inputShape, List<GCHandle>? handles = null)
    {
        _steps = steps;
        _finalOutput = finalOutput;
        _engine = engine;
        _compiledInputShape = inputShape;
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
        var inputShape = steps.Count > 0 && steps[0].Inputs.Length > 0
            ? (int[])steps[0].Inputs[0]._shape.Clone()
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
        return new CompiledInferencePlan<T>(optimizedSteps, finalOutput, engine, inputShape, pinnedHandles);
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
