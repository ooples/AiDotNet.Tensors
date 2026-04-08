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
                // All inputs are constant — evaluate once and replace with precomputed result
                step.Execute(engine, step.OutputBuffer);
                // Skip this step in the compiled plan (output is already computed)
                anyFolded = true;
                // Don't add to foldedSteps — the output tensor already has the right data
            }
        }

        return anyFolded ? foldedSteps.ToArray() : null;
    }
}
