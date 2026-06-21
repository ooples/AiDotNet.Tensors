using System.Runtime.CompilerServices;
using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 4.5: Constant folding — detect operations whose inputs don't change
/// between steps and precompute them once at compile time.
///
/// A tensor is "constant" if:
/// - It's not in the trainable parameter set
/// - It's not derived from the model input (first step's input)
/// - All its inputs are also constant
///
/// Constant subgraphs are evaluated once during compilation and their outputs
/// are stored as precomputed tensors, eliminating redundant computation.
/// </summary>
internal sealed class ConstantFoldingPass : ICpuOptimizationPass
{
    public string Name => "ConstantFolding";

    public bool IsEnabled => TensorCodecOptions.Current.EnableConstantFolding;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || steps.Length == 0) return null;

        // Build set of all tensors produced by some step's output.
        // Any tensor that appears as a step input but is NOT produced by any step
        // is a graph input (model input, weight, etc.) and must be treated as dynamic.
        var producedByStep = new HashSet<object>();
        foreach (var step in steps)
            producedByStep.Add(step.OutputBuffer);

        var dynamicTensors = new HashSet<object>();
        foreach (var step in steps)
        {
            foreach (var inp in step.Inputs)
            {
                // If this input is not produced by any step, it's a graph-level input
                // (model input or a mutable weight) — treat it as dynamic.
                if (!producedByStep.Contains(inp))
                    dynamicTensors.Add(inp);
            }
        }

        // Propagate: any op with at least one dynamic input produces a dynamic output
        bool anyFolded = false;
        var foldedSteps = new List<CompiledStep<T>>(steps.Length);

        foreach (var step in steps)
        {
            bool hasDynamicInput = false;
            foreach (var inp in step.Inputs)
            {
                if (dynamicTensors.Contains(inp))
                {
                    hasDynamicInput = true;
                    break;
                }
            }

            if (hasDynamicInput)
            {
                // Output is dynamic — keep the step
                dynamicTensors.Add(step.OutputBuffer);
                foldedSteps.Add(step);
            }
            else
            {
                // All inputs are constant — fold to a RUN-ONCE step: execute the op on its FIRST runtime
                // replay and skip on every subsequent replay (the value persists in OutputBuffer). This
                // preserves the optimization (a constant subgraph is computed once, not re-run across a
                // denoising loop's 50+ replays) WITHOUT the previous design's compile-time side effect.
                //
                // The old pass called step.Execute(engine, step.OutputBuffer) HERE, at compile time. That
                // was wrong on two counts: (1) the plan's intermediate input buffers are not materialized
                // until the plan runs, so the op read garbage and wrote garbage into a graph tensor; and
                // (2) it made BUILDING the plan mutate the graph's tensors, which corrupts the verify-then-
                // trust inference gate — that gate runs the eager forward over the SAME graph to compare it
                // against the compiled candidate, so any compile-time write poisons the eager reference
                // (observed as non-deterministic cross-attention diffusion inference). Executing lazily at
                // runtime, when inputs are valid, fixes both: compilation has no graph side effects.
                var inner = step;
                bool computed = false;
                foldedSteps.Add(new CompiledStep<T>(
                    step.OpName,
                    (eng, o) =>
                    {
                        // Benign race under concurrent inference: a double-compute writes the same constant
                        // value into the same buffer, so torn identical writes are harmless.
                        if (computed) return;
                        inner.Execute(eng, o);
                        computed = true;
                    },
                    step.OutputBuffer,
                    step.Inputs,
                    step.BackwardFn,
                    step.SavedState));
                anyFolded = true;
                // Output stays CONSTANT (not added to dynamicTensors) so a downstream op reading it can
                // also fold to run-once — the chain collapses to one compute on the first replay.
            }
        }

        return anyFolded ? foldedSteps.ToArray() : null;
    }
}
