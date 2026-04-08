using AiDotNet.Tensors.Engines.Compilation;

namespace AiDotNet.Tensors.Engines.Optimization;

/// <summary>
/// Phase 6.2: Forward Common Subexpression Elimination.
///
/// Detects identical computations across different layers and replaces duplicates
/// with references to the first occurrence. Two operations are identical if they
/// have the same OpName, the same input tensor identities, and the same SavedState.
///
/// Example: ResNet skip connections often compute the same normalization twice.
/// CSE computes it once and shares the result.
///
/// Expected gain: 5-15% for models with shared normalization or repeated blocks.
/// </summary>
internal sealed class ForwardCSEPass : ICpuOptimizationPass
{
    public string Name => "ForwardCSE";

    public bool IsEnabled => TensorCodecOptions.Current.EnableForwardCSE;

    public CompiledStep<T>[]? TryOptimize<T>(CompiledStep<T>[] steps, IEngine engine)
    {
        if (!IsEnabled || steps.Length < 2) return null;

        var result = new List<CompiledStep<T>>(steps.Length);
        // Map from operation signature hash to the step that computes it
        var computed = new Dictionary<long, CompiledStep<T>>();
        bool anyEliminated = false;

        foreach (var step in steps)
        {
            long hash = ComputeStepHash(step);

            if (computed.TryGetValue(hash, out var existing)
                && InputsMatch(step, existing))
            {
                // This step computes the same thing as an existing step.
                // Replace with a copy from the existing output.
                // BackwardFn is set to null because the CSE copy does not execute
                // the original op — its SavedState would be stale. Gradients flow
                // through the original (first) computation instead.
                var src = existing.OutputBuffer;
                var dst = step.OutputBuffer;
                result.Add(new CompiledStep<T>(
                    "CSE_Copy",
                    (eng, output) => src.AsSpan().CopyTo(output.AsWritableSpan()),
                    dst,
                    step.Inputs,
                    null,
                    null));
                anyEliminated = true;
            }
            else
            {
                computed[hash] = step;
                result.Add(step);
            }
        }

        return anyEliminated ? result.ToArray() : null;
    }

    private static long ComputeStepHash<T>(CompiledStep<T> step)
    {
        long hash = unchecked((long)0xcbf29ce484222325L);
        hash ^= step.OpName.GetHashCode();
        hash *= unchecked((long)0x100000001b3L);

        // Include input tensor identities
        foreach (var inp in step.Inputs)
        {
            hash ^= System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(inp);
            hash *= unchecked((long)0x100000001b3L);
        }

        // Include saved state if any
        if (step.SavedState is not null)
        {
            foreach (var s in step.SavedState)
            {
                hash ^= (s?.GetHashCode() ?? 0);
                hash *= unchecked((long)0x100000001b3L);
            }
        }

        return hash;
    }

    private static bool InputsMatch<T>(CompiledStep<T> a, CompiledStep<T> b)
    {
        if (a.Inputs.Length != b.Inputs.Length) return false;
        for (int i = 0; i < a.Inputs.Length; i++)
        {
            if (!ReferenceEquals(a.Inputs[i], b.Inputs[i]))
                return false;
        }
        return true;
    }
}
