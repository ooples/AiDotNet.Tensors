using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 5.1: Tensor lifetime analysis and buffer reuse for compiled plans.
///
/// Analyzes the dataflow graph to determine when each intermediate tensor's buffer
/// is last consumed. Buffers whose lifetime has ended can be reused by later ops,
/// reducing peak memory allocation.
///
/// Example: In a 10-layer MLP, layer 3's activation is consumed by layer 4's backward.
/// After layer 4's backward runs, layer 3's buffer can be reused for layer 7's activation.
///
/// This pass runs AFTER other optimization passes and modifies buffer assignments
/// without changing the computation graph structure.
/// </summary>
internal sealed class MemoryPlannerPass : ICpuOptimizationPass
{
    public string Name => "MemoryPlanner";

    public bool IsEnabled => true; // Always enabled — pure memory optimization, no approximation

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (steps.Length < 3) return null; // Not enough steps to benefit

        // Phase 1: Compute last-use index for each tensor
        var lastUse = new Dictionary<object, int>(); // tensor → last step index that reads it

        for (int i = 0; i < steps.Length; i++)
        {
            // Inputs are read at this step
            foreach (var inp in steps[i].Inputs)
                lastUse[inp] = i;

            // Output is "born" at this step — will be read by later consumers
            // If not used by any later step, it's the final output (don't reuse)
        }

        // Phase 2: Build a pool of reusable buffers keyed by element count.
        // When a tensor's lifetime ends (step index > lastUse), its buffer becomes available.
        var availableBuffers = new Dictionary<int, Queue<Tensor<T>>>(); // elementCount → available tensors
        int reusedCount = 0;

        // We can't modify the steps' OutputBuffer references directly (they're readonly).
        // Instead, create new steps that write into reused buffers.
        var optimizedSteps = new CompiledStep<T>[steps.Length];
        var reuseMap = new Dictionary<object, Tensor<T>>(); // original output → reused buffer

        for (int i = 0; i < steps.Length; i++)
        {
            // Return consumed buffers to the pool
            foreach (var inp in steps[i].Inputs)
            {
                if (lastUse.TryGetValue(inp, out int lastIdx) && lastIdx == i)
                {
                    // This input is consumed for the last time at this step.
                    // Its buffer can be reused after this step completes.
                    if (inp is Tensor<T> tensor && tensor.Length > 0)
                    {
                        int count = tensor.Length;
                        if (!availableBuffers.ContainsKey(count))
                            availableBuffers[count] = new Queue<Tensor<T>>();
                        availableBuffers[count].Enqueue(tensor);
                    }
                }
            }

            // Try to reuse a buffer for this step's output
            var step = steps[i];
            int outputLen = step.OutputBuffer.Length;

            // Don't reuse for the final step's output (it's returned to the caller)
            if (i < steps.Length - 1
                && availableBuffers.TryGetValue(outputLen, out var pool) && pool.Count > 0)
            {
                var reusedBuffer = pool.Dequeue();
                if (!ReferenceEquals(reusedBuffer, step.OutputBuffer))
                {
                    // Create a new step that writes into the reused buffer
                    var capturedExec = step.Execute;
                    var capturedReused = reusedBuffer;
                    optimizedSteps[i] = new CompiledStep<T>(
                        step.OpName,
                        (eng, o) => capturedExec(eng, capturedReused),
                        reusedBuffer,
                        step.Inputs,
                        step.BackwardFn,
                        step.SavedState);
                    reuseMap[step.OutputBuffer] = reusedBuffer;
                    reusedCount++;
                    continue;
                }
            }

            optimizedSteps[i] = step;
        }

        return reusedCount > 0 ? optimizedSteps : null;
    }
}
