using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

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
        var computed = new Dictionary<long, (CompiledStep<T> Step, int Index)>();
        bool anyEliminated = false;

        foreach (var step in steps)
        {
            long hash = ComputeStepHash(step);

            if (computed.TryGetValue(hash, out var found)
                && InputsMatch(step, found.Step)
                && !StorageRewrittenSince(result, found.Index, found.Step.OutputBuffer))
            {
                var existing = found.Step;
                // This step computes the same thing as an existing step.
                // Replace with a copy from the existing output.
                // BackwardFn is set to null because the CSE copy does not execute
                // the original op — its SavedState would be stale. Gradients flow
                // through the original (first) computation instead.
                var src = existing.OutputBuffer;
                var dst = step.OutputBuffer;
                result.Add(new CompiledStep<T>(
                    "CSE_Copy",
                    (eng, output) => AiDotNet.Tensors.Engines.DirectGpuTensorEngine.CopyResultInto(eng, src, output),
                    dst,
                    new[] { src },
                    null,
                    null));
                anyEliminated = true;
            }
            else
            {
                computed[hash] = (step, result.Count);
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

    /// <summary>
    /// This pass runs after memory planning, which reuses a buffer once its last original reader ran. A copy of an
    /// earlier step's output reads it LATER than any original reader did, so the buffer may already hold a later step's
    /// result (measured: a recomputed matmul copied from storage memory planning had donated to a step in between).
    /// The copy is only valid when no step after the source writes its storage.
    /// </summary>
    private static bool StorageRewrittenSince<T>(List<CompiledStep<T>> scheduled, int sourceIndex, Tensor<T> source)
    {
        for (int i = sourceIndex + 1; i < scheduled.Count; i++)
        {
            var written = scheduled[i].OutputBuffer;
            if (ReferenceEquals(written, source) || ReferenceEquals(written._storage, source._storage)) return true;
        }
        return false;
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
