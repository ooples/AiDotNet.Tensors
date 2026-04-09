using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 5.1: Tensor lifetime analysis and buffer reuse for compiled inference plans.
///
/// Only safe for inference plans (no backward pass). Training plans need all
/// intermediate tensors alive for backward.
/// </summary>
internal sealed class MemoryPlannerPass : ICpuOptimizationPass
{
    public string Name => "MemoryPlanner";

    public bool IsEnabled => true;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || steps.Length < 3) return null;

        var lastUse = ComputeLastUseIndices(steps);

        var availablePool = new Dictionary<string, Queue<Tensor<T>>>();
        var reuseMap = new Dictionary<Tensor<T>, Tensor<T>>();
        var enqueuedThisStep = new HashSet<Tensor<T>>();
        int reusedCount = 0;

        var result = new CompiledStep<T>[steps.Length];

        for (int i = 0; i < steps.Length; i++)
        {
            var step = steps[i];
            enqueuedThisStep.Clear();

            // Rewrite inputs to use remapped buffers
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

            // Return consumed buffers to pool (keyed by shape string, not just length)
            foreach (var inp in step.Inputs)
            {
                var actualInp = reuseMap.TryGetValue(inp, out var r) ? r : inp;
                if (lastUse.TryGetValue(inp, out int lastIdx) && lastIdx == i
                    && !enqueuedThisStep.Contains(actualInp))
                {
                    string shapeKey = ShapeKey(actualInp._shape);
                    if (!availablePool.ContainsKey(shapeKey))
                        availablePool[shapeKey] = new Queue<Tensor<T>>();
                    availablePool[shapeKey].Enqueue(actualInp);
                    enqueuedThisStep.Add(actualInp);
                }
            }

            // Try to reuse a buffer with matching SHAPE for this step's output
            var output = step.OutputBuffer;
            if (i < steps.Length - 1)
            {
                string outKey = ShapeKey(output._shape);
                if (availablePool.TryGetValue(outKey, out var pool) && pool.Count > 0)
                {
                    var reusedBuffer = pool.Dequeue();
                    if (!ReferenceEquals(reusedBuffer, output))
                    {
                        reuseMap[output] = reusedBuffer;
                        output = reusedBuffer;
                        reusedCount++;
                    }
                }
            }

            if (anyRemapped || !ReferenceEquals(output, step.OutputBuffer))
            {
                var capturedExec = step.Execute;
                var capturedOutput = output;
                result[i] = new CompiledStep<T>(
                    step.OpName,
                    (eng, o) => capturedExec(eng, capturedOutput),
                    output,
                    newInputs,
                    null,
                    step.SavedState);
            }
            else
            {
                result[i] = step;
            }
        }

        return reusedCount > 0 ? result : null;
    }

    private static string ShapeKey(int[] shape)
    {
        if (shape.Length == 1) return shape[0].ToString();
        return string.Join(",", shape);
    }

    internal static Dictionary<object, int> ComputeLastUseIndices<T>(CompiledStep<T>[] steps)
    {
        var lastUse = new Dictionary<object, int>();
        for (int i = 0; i < steps.Length; i++)
            foreach (var inp in steps[i].Inputs)
                lastUse[inp] = i;
        return lastUse;
    }
}
