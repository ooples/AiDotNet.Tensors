using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 5.1: Tensor lifetime analysis and buffer reuse for compiled inference plans.
///
/// Analyzes the dataflow graph to determine when each intermediate tensor's buffer
/// is last consumed. Buffers whose lifetime has ended can be reused by later ops,
/// reducing peak memory allocation.
///
/// IMPORTANT: Only safe for inference plans (no backward pass). Training plans
/// need all intermediate tensors alive for backward, so this pass must not be
/// applied to CompiledTrainingPlan.
/// </summary>
internal sealed class MemoryPlannerPass : ICpuOptimizationPass
{
    public string Name => "MemoryPlanner";

    public bool IsEnabled => true;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || steps.Length < 3) return null;

        // Phase 1: Compute last-use index for each tensor
        var lastUse = ComputeLastUseIndices(steps);

        // Phase 2: Build pool of reusable buffers and rewrite references.
        // Key insight: when we reuse buffer A for step j's output, ALL subsequent
        // steps that reference step j's output as an input must be updated to
        // point at buffer A instead.
        var availablePool = new Dictionary<int, Queue<Tensor<T>>>();
        var reuseMap = new Dictionary<Tensor<T>, Tensor<T>>();
        int reusedCount = 0;

        var result = new CompiledStep<T>[steps.Length];

        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];

            // Rewrite this step's inputs to use remapped buffers
            var newInputs = step.Inputs;
            bool anyRemapped = false;
            for (int inp = 0; inp < newInputs.Length; inp++)
            {
                if (reuseMap.TryGetValue(newInputs[inp], out var remapped))
                {
                    if (!anyRemapped)
                    {
                        newInputs = (Tensor<T>[])newInputs.Clone();
                        anyRemapped = true;
                    }
                    newInputs[inp] = remapped;
                }
            }

            // Return consumed buffers to pool (after this step reads them for the last time)
            foreach (var inp in step.Inputs)
            {
                var actualInp = reuseMap.TryGetValue(inp, out var r) ? r : inp;
                if (lastUse.TryGetValue(inp, out int lastIdx) && lastIdx == i
                    && actualInp.Length > 0)
                {
                    int count = actualInp.Length;
                    if (!availablePool.ContainsKey(count))
                        availablePool[count] = new Queue<Tensor<T>>();
                    availablePool[count].Enqueue(actualInp);
                }
            }

            // Try to reuse a buffer for this step's output
            var output = step.OutputBuffer;
            // Don't reuse for the last step's output (returned to caller)
            if (i < steps.Length - 1
                && availablePool.TryGetValue(output.Length, out var pool)
                && pool.Count > 0)
            {
                var reusedBuffer = pool.Dequeue();
                if (!ReferenceEquals(reusedBuffer, output))
                {
                    reuseMap[output] = reusedBuffer;
                    output = reusedBuffer;
                    reusedCount++;
                }
            }

            // Build the (possibly rewritten) step
            if (anyRemapped || !ReferenceEquals(output, step.OutputBuffer))
            {
                var capturedExec = step.Execute;
                var capturedOutput = output;
                result[i] = new CompiledStep<T>(
                    step.OpName,
                    (eng, o) => capturedExec(eng, capturedOutput),
                    output,
                    newInputs,
                    null, // No backward for inference plans
                    step.SavedState);
            }
            else
            {
                result[i] = step;
            }
        }

        return reusedCount > 0 ? result : null;
    }

    /// <summary>
    /// Computes the last step index at which each tensor is consumed as an input.
    /// </summary>
    internal static Dictionary<object, int> ComputeLastUseIndices<T>(CompiledStep<T>[] steps)
    {
        var lastUse = new Dictionary<object, int>();
        for (int i = 0; i < steps.Length; i++)
            foreach (var inp in steps[i].Inputs)
                lastUse[inp] = i;
        return lastUse;
    }
}
