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

        // Identify all tensors that are model inputs (first step's inputs that aren't weights)
        var dynamicTensors = new List<object>();

        // The first step's primary input (index 0) is the model input — it's dynamic
        if (steps[0].Inputs.Length > 0)
            dynamicTensors.Add(steps[0].Inputs[0]);

        // Propagate: any op whose input is dynamic produces a dynamic output
        bool anyFolded = false;
        var foldedSteps = new List<CompiledStep<T>>(steps.Length);

        foreach (var step in steps)
        {
            bool hasDynamicInput = false;
            foreach (var inp in step.Inputs)
            {
                foreach (var dyn in dynamicTensors)
                {
                    if (ReferenceEquals(inp, dyn))
                    {
                        hasDynamicInput = true;
                        break;
                    }
                }
                if (hasDynamicInput) break;
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
