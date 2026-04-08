using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>
/// A compiled inference plan — a flat array of pre-resolved execution steps.
/// Zero overhead per-operation: no graph traversal, no shape validation, no allocation.
/// All buffers are pre-allocated at compile time and reused across replays.
///
/// Usage:
///   var plan = CompiledInferencePlan.Compile(scope);
///   var output = plan.Execute(engine);  // first call
///   var output2 = plan.Execute(engine); // replay — same speed, zero alloc
/// </summary>
internal sealed class CompiledInferencePlan<T>
{
    private readonly CompiledStep<T>[] _steps;
    private readonly Tensor<T> _finalOutput;
    private readonly IEngine _engine;

    private CompiledInferencePlan(CompiledStep<T>[] steps, Tensor<T> finalOutput, IEngine engine)
    {
        _steps = steps;
        _finalOutput = finalOutput;
        _engine = engine;
    }

    /// <summary>Number of compiled steps.</summary>
    internal int StepCount => _steps.Length;

    /// <summary>
    /// Executes the compiled plan. Runs each step's delegate in order.
    /// All buffers are pre-allocated — zero allocation during execution.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal Tensor<T> Execute()
    {
        var steps = _steps;
        var engine = _engine;
        for (int i = 0; i < steps.Length; i++)
        {
            steps[i].Execute(engine, steps[i].OutputBuffer);
        }
        return _finalOutput;
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

        // Build specialized forward actions (same optimization as CompiledTrainingPlan)
        var specializedSteps = new CompiledStep<T>[steps.Count];
        for (int i = 0; i < steps.Count; i++)
        {
            var step = steps[i];

            // Zero-copy transpose: replace output buffer with strided view, no-op execute
            if (step.OpType == OpType.TensorTranspose && step.Inputs.Length == 1 && step.Inputs[0].Rank == 2)
            {
                var originalOutput = step.OutputBuffer;
                var view = step.Inputs[0].Transpose();
                specializedSteps[i] = new CompiledStep<T>(
                    step.OpName,
                    (eng, o) => { }, // no-op — data accessed via stride permutation
                    view,            // output IS the strided view
                    step.Inputs,
                    step.BackwardFn,
                    step.SavedState);

                // Rewrite downstream steps: replace any input reference to the original
                // OutputBuffer with the new view so they read from the strided view
                for (int j = i + 1; j < steps.Count; j++)
                {
                    var downstream = steps[j];
                    bool rewritten = false;
                    var newInputs = new Tensor<T>[downstream.Inputs.Length];
                    for (int k = 0; k < downstream.Inputs.Length; k++)
                    {
                        if (ReferenceEquals(downstream.Inputs[k], originalOutput))
                        {
                            newInputs[k] = view;
                            rewritten = true;
                        }
                        else
                        {
                            newInputs[k] = downstream.Inputs[k];
                        }
                    }
                    if (rewritten)
                    {
                        steps[j] = new CompiledStep<T>(
                            downstream.OpName, downstream.Execute, downstream.OutputBuffer,
                            newInputs, downstream.BackwardFn, downstream.SavedState);
                    }
                }
                continue;
            }

            var specialized = CompiledTrainingPlan<T>.TryBuildSpecializedForward(step);
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
        return new CompiledInferencePlan<T>(optimizedSteps, finalOutput, engine);
    }

    /// <summary>
    /// Runs CPU-level optimization passes on the compiled steps.
    /// Currently: spectral decomposition (Phase A) and dataflow fusion (Phase B).
    /// Each pass is independently toggleable via TensorCodecOptions.
    /// </summary>
    private static CompiledStep<T>[] RunCpuOptimizationPasses(CompiledStep<T>[] steps, IEngine engine)
    {
        ICpuOptimizationPass[] passes =
        {
            new SpectralDecompositionPass(),
            new DataflowFusionPass(),
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
