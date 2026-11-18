using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    // Resolve tensors after pooling has assigned their storage. The old replay allocated the full derivative,
    // then copied it into this same planned gradient. No public kernel or higher-order derivative changes.
    internal static Action<IEngine>? BuildCpuLeakyReluBackward(CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradMap, Dictionary<Tensor<T>, int> writerCount,
        IEngine engine, HashSet<Tensor<T>>? required)
    {
        if (engine.GetType() != typeof(CpuEngine) || step.OpType != OpType.LeakyReLU
            || step.Inputs.Length != 1 || step.DifferentiableInputIndices is not null
            || (typeof(T) != typeof(float) && typeof(T) != typeof(double)))
            return null;
        var input = step.Inputs[0];
        if ((required is not null && !required.Contains(input))
            || !gradMap.TryGetValue(input, out var gradInput)
            || !gradMap.TryGetValue(step.OutputBuffer, out var gradOutput)
            || input.Length != gradOutput.Length || input.Length != gradInput.Length
            || !input.IsContiguous || !gradOutput.IsContiguous || !gradInput.IsContiguous)
            return null;
        // This probe must not materialize gradient placeholders before the final liveness assignment.
        if (input.GetCpuBackingForStridedRead(out _) is null
            || gradOutput.GetCpuBackingForStridedRead(out _) is null
            || gradInput.GetCpuBackingForStridedRead(out _) is null)
            return null;
        double slope = step.SavedState is { Length: > 0 } saved ? (double)saved[0] : .01;
        bool shared = writerCount.TryGetValue(input, out int writers) && writers > 1;
        return replayEngine =>
        {
            MaterializeIfGradPlaceholder(gradOutput);
            MaterializeIfGradPlaceholder(gradInput);
            bool add = AccumulatesInto(gradInput, shared);
            if (replayEngine.GetType() != typeof(CpuEngine))
            {
                var value = replayEngine.LeakyReluBackward(gradOutput, input, slope);
                if (add) replayEngine.TensorAddInto(gradInput, gradInput, value);
                else
                {
                    using var valueKeep = new KeepAliveScope(value);
                    using var targetKeep = new KeepAliveScope(gradInput);
                    value.AsSpanUnmarked().CopyTo(gradInput.AsWritableSpanUnmarked());
                }
            }
            else
            {
                using var inputKeep = new KeepAliveScope(input);
                using var upstreamKeep = new KeepAliveScope(gradOutput);
                using var destinationKeep = new KeepAliveScope(gradInput);
                if (typeof(T) == typeof(float))
                    WriteCpuLeakyReluGradient(((Tensor<float>)(object)input).AsSpanUnmarked(),
                        ((Tensor<float>)(object)gradOutput).AsSpanUnmarked(),
                        ((Tensor<float>)(object)gradInput).AsWritableSpanUnmarked(), (float)slope, add);
                else
                    WriteCpuLeakyReluGradient(((Tensor<double>)(object)input).AsSpanUnmarked(),
                        ((Tensor<double>)(object)gradOutput).AsSpanUnmarked(),
                        ((Tensor<double>)(object)gradInput).AsWritableSpanUnmarked(), slope, add);
            }
            input.Grad = gradInput;
        };
    }

    private static void WriteCpuLeakyReluGradient(ReadOnlySpan<float> input, ReadOnlySpan<float> upstream,
        Span<float> destination, float slope, bool add)
    {
        // Exact aliases are elementwise-safe. Partial overlaps must snapshot both sources before any write,
        // including independently constructed Memory wrappers that do not share a tensor storage identity.
        if (input.Overlaps(destination, out int inputOffset) && inputOffset != 0)
        {
            input = input.ToArray();
            CpuStorageAllocationDiagnostics.Record<float>("numeric.compiled_leaky_alias_snapshot", input.Length);
        }
        if (upstream.Overlaps(destination, out int upstreamOffset) && upstreamOffset != 0)
        {
            upstream = upstream.ToArray();
            CpuStorageAllocationDiagnostics.Record<float>("numeric.compiled_leaky_alias_snapshot", upstream.Length);
        }
        for (int i = 0; i < destination.Length; i++)
        {
            float value = input[i] > 0f ? upstream[i] : upstream[i] * slope;
            destination[i] = add ? destination[i] + value : value;
        }
    }

    private static void WriteCpuLeakyReluGradient(ReadOnlySpan<double> input, ReadOnlySpan<double> upstream,
        Span<double> destination, double slope, bool add)
    {
        if (input.Overlaps(destination, out int inputOffset) && inputOffset != 0)
        {
            input = input.ToArray();
            CpuStorageAllocationDiagnostics.Record<double>("numeric.compiled_leaky_alias_snapshot", input.Length);
        }
        if (upstream.Overlaps(destination, out int upstreamOffset) && upstreamOffset != 0)
        {
            upstream = upstream.ToArray();
            CpuStorageAllocationDiagnostics.Record<double>("numeric.compiled_leaky_alias_snapshot", upstream.Length);
        }
        for (int i = 0; i < destination.Length; i++)
        {
            double value = input[i] > 0.0 ? upstream[i] : upstream[i] * slope;
            destination[i] = add ? destination[i] + value : value;
        }
    }
}
