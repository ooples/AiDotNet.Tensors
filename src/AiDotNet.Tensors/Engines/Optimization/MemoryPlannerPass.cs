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

    /// <summary>
    /// Disabled until downstream input reference rewriting is implemented.
    /// The current implementation replaces output buffers but doesn't update
    /// later steps that reference those outputs as inputs, causing stale reads.
    /// </summary>
    public bool IsEnabled => false;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || steps.Length < 3) return null;

        // Phase 1: Compute last-use index for each tensor
        var lastUse = ComputeLastUseIndices(steps);

        // Phase 2: Analyze potential savings (for diagnostics only until rewriting is done)
        int reusableBuffers = 0;
        long savedBytes = 0;
        for (int i = 0; i < steps.Length; i++)
        {
            foreach (var inp in steps[i].Inputs)
            {
                if (lastUse.TryGetValue(inp, out int lastIdx) && lastIdx == i
                    && inp is Tensor<T> tensor)
                {
                    reusableBuffers++;
                    savedBytes += tensor.Length * System.Runtime.InteropServices.Marshal.SizeOf<T>();
                }
            }
        }

        // Return null — no transformation applied until input rewriting is implemented.
        // The analysis above can be used by ProfilingCompiler to report potential savings.
        return null;
    }

    /// <summary>
    /// Computes the last step index at which each tensor is consumed as an input.
    /// Tensors consumed after this index are safe to reclaim.
    /// </summary>
    internal static Dictionary<object, int> ComputeLastUseIndices<T>(CompiledStep<T>[] steps)
    {
        var lastUse = new Dictionary<object, int>();
        for (int i = 0; i < steps.Length; i++)
        {
            foreach (var inp in steps[i].Inputs)
                lastUse[inp] = i;
        }
        return lastUse;
    }
}
