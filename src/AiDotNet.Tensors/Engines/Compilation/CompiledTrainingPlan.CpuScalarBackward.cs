using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    // Spectral normalization multiplies a matrix by [1,1]. The generic VJP allocates
    // two matrix-sized products. Its explicit two-axis ReduceSum uses Tensor.Sum's
    // single row-major accumulator; keep that association, including float product rounding.
    internal static Action<IEngine>? BuildCpuScalarMultiplyBackward(CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradients, Dictionary<Tensor<T>, int> writers,
        IEngine engine, HashSet<Tensor<T>>? required)
    {
        if (typeof(T) != typeof(float) || engine.Unwrap().GetType() != typeof(CpuEngine) || !ReferenceEquals(engine, engine.Unwrap())
            || step.OpType != OpType.TensorBroadcastMultiply || step.Inputs.Length != 2
            || step.DifferentiableInputIndices is not null) return null;
        var matrix = step.Inputs[0];
        var scalar = step.Inputs[1];
        // Both axes must actually be reduced. A singleton matrix axis uses a different
        // canonical single-axis SIMD reduction and retains the existing implementation.
        if (matrix.Rank != 2 || matrix._shape[0] <= 1 || matrix._shape[1] <= 1
            || scalar.Rank != 2 || scalar._shape[0] != 1 || scalar._shape[1] != 1
            || (required is not null && (!required.Contains(matrix) || !required.Contains(scalar)))
            || !gradients.TryGetValue(matrix, out var matrixGradient)
            || !gradients.TryGetValue(scalar, out var scalarGradient)
            || !gradients.TryGetValue(step.OutputBuffer, out var upstream)
            || matrixGradient.Length != matrix.Length || upstream.Length != matrix.Length
            || scalarGradient.Length != 1) return null;
        foreach (var tensor in new[] { matrix, scalar, matrixGradient, scalarGradient, upstream })
            if (!tensor.IsContiguous || tensor.GetCpuBackingForStridedRead(out _) is null) return null;
        bool matrixShared = writers.TryGetValue(matrix, out int mw) && mw > 1;
        bool scalarShared = writers.TryGetValue(scalar, out int sw) && sw > 1;
        return _ =>
        {
            MaterializeIfGradPlaceholder(upstream);
            MaterializeIfGradPlaceholder(matrixGradient);
            MaterializeIfGradPlaceholder(scalarGradient);
            using var matrixKeep = new KeepAliveScope(matrix);
            using var scalarKeep = new KeepAliveScope(scalar);
            using var upstreamKeep = new KeepAliveScope(upstream);
            using var matrixGradientKeep = new KeepAliveScope(matrixGradient);
            using var scalarGradientKeep = new KeepAliveScope(scalarGradient);
            var values = ((Tensor<float>)(object)matrix).AsSpanUnmarked();
            var seeds = ((Tensor<float>)(object)upstream).AsSpanUnmarked();
            float scale = ((Tensor<float>)(object)scalar).AsSpanUnmarked()[0];
            var destination = ((Tensor<float>)(object)matrixGradient).AsWritableSpanUnmarked();
            var scalarDestination = ((Tensor<float>)(object)scalarGradient).AsWritableSpanUnmarked();
            // Finish all reads needed by the scalar derivative before an aliased destination
            // is written. Upstream's exact alias is elementwise safe; partial overlap is not.
            float sum = 0f;
            for (int i = 0; i < values.Length; i++)
            {
                float product = seeds[i] * values[i];
                sum += product;
            }
            if (seeds.Overlaps(destination, out int offset) && offset != 0) seeds = seeds.ToArray();
            bool addMatrix = AccumulatesInto(matrixGradient, matrixShared);
            WriteScaledGradient(seeds, destination, scale, addMatrix);
            bool addScalar = AccumulatesInto(scalarGradient, scalarShared);
            scalarDestination[0] = addScalar ? scalarDestination[0] + sum : sum;
            matrix.Grad = matrixGradient;
            scalar.Grad = scalarGradient;
        };
    }

    internal static Action<IEngine>? BuildCpuFullMeanBackward(CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradients, Dictionary<Tensor<T>, int> writers,
        IEngine engine, HashSet<Tensor<T>>? required)
    {
        if (engine.Unwrap().GetType() != typeof(CpuEngine) || !ReferenceEquals(engine, engine.Unwrap()) || step.OpType != OpType.Mean
            || step.Inputs.Length != 1 || step.DifferentiableInputIndices is not null
            || (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            || step.SavedState is not { Length: >= 2 } saved || saved[0] is not int[] axes || axes.Length == 0) return null;
        var input = step.Inputs[0];
        if (input.Length == 0 || (required is not null && !required.Contains(input))
            || !gradients.TryGetValue(input, out var destination)
            || !gradients.TryGetValue(step.OutputBuffer, out var upstream)
            || destination.Length != input.Length || upstream.Length != 1
            || !destination.IsContiguous || !upstream.IsContiguous) return null;
        var reduced = new bool[input.Rank];
        foreach (int raw in axes)
        {
            int axis = raw < 0 ? raw + input.Rank : raw;
            if ((uint)axis >= (uint)input.Rank) return null;
            reduced[axis] = true;
        }
        if (Array.IndexOf(reduced, false) >= 0
            || destination.GetCpuBackingForStridedRead(out _) is null
            || upstream.GetCpuBackingForStridedRead(out _) is null) return null;
        bool shared = writers.TryGetValue(input, out int count) && count > 1;
        // CpuEngine.ReduceMeanBackward first rounds 1/count to T, then multiplies by dy.
        float floatScale = 1f / input.Length;
        double doubleScale = 1d / input.Length;
        return _ =>
        {
            MaterializeIfGradPlaceholder(upstream);
            MaterializeIfGradPlaceholder(destination);
            using var upstreamKeep = new KeepAliveScope(upstream);
            using var destinationKeep = new KeepAliveScope(destination);
            bool add = AccumulatesInto(destination, shared);
            if (typeof(T) == typeof(float))
            {
                float value = ((Tensor<float>)(object)upstream).AsSpanUnmarked()[0] * floatScale;
                var target = ((Tensor<float>)(object)destination).AsWritableSpanUnmarked();
                if (!add) target.Fill(value);
                else for (int i = 0; i < target.Length; i++) target[i] += value;
            }
            else
            {
                double value = ((Tensor<double>)(object)upstream).AsSpanUnmarked()[0] * doubleScale;
                var target = ((Tensor<double>)(object)destination).AsWritableSpanUnmarked();
                if (!add) target.Fill(value);
                else for (int i = 0; i < target.Length; i++) target[i] += value;
            }
            input.Grad = destination;
        };
    }

    private static void WriteScaledGradient(ReadOnlySpan<float> source, Span<float> target, float scale, bool add)
    {
        int at = 0;
#if NET5_0_OR_GREATER
        int width = System.Numerics.Vector<float>.Count;
        var multiplier = new System.Numerics.Vector<float>(scale);
        for (; at + width <= source.Length; at += width)
        {
            var value = new System.Numerics.Vector<float>(source.Slice(at, width)) * multiplier;
            if (add) value += new System.Numerics.Vector<float>(target.Slice(at, width));
            value.CopyTo(target.Slice(at, width));
        }
#endif
        for (; at < source.Length; at++)
        {
            float value = source[at] * scale;
            target[at] = add ? target[at] + value : value;
        }
    }
}
