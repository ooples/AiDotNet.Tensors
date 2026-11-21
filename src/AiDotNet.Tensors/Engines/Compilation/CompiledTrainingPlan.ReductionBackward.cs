using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    // For a contiguous block of reduced axes the input is [outer,reduced,inner], and the upstream gradient
    // is [outer,inner]. This broadcasts it into the plan's existing gradient instead of allocating TensorTile's
    // entire input-sized tensor and then copying that tensor into the accumulator on every replay.
    internal static Action<IEngine>? BuildCpuReducedGradientBackward(
        CompiledStep<T> step, Dictionary<Tensor<T>, Tensor<T>> gradMap,
        Dictionary<Tensor<T>, int> writerCount, IEngine engine)
    {
        if (step.OpType != OpType.ReduceSum || step.Inputs.Length != 1 || engine.SupportsGpu
            || engine.Is<DirectGpuTensorEngine>() || !engine.Is<CpuEngine>()
            || (typeof(T) != typeof(float) && typeof(T) != typeof(double))
            || step.SavedState is not { Length: >= 2 } saved || saved[0] is not int[] axes || saved[1] is not bool)
            return null;
        var input = step.Inputs[0];
        if (!gradMap.TryGetValue(input, out var gradIn) || !gradMap.TryGetValue(step.OutputBuffer, out var gradOut)
            || !gradIn.IsContiguous || !gradOut.IsContiguous || input.Length == 0)
            return null;
        // Probe the backing without materializing gradient placeholders or capturing arrays before pooling rebinds
        // them. Native storage retains the canonical engine fallback; contiguous array slices retain their offsets.
        if (gradIn.GetCpuBackingForStridedRead(out _) is null || gradOut.GetCpuBackingForStridedRead(out _) is null)
            return null;

        int rank = input.Rank;
        var reduced = new bool[rank];
        if (axes.Length == 0) for (int axis = 0; axis < rank; axis++) reduced[axis] = true;
        else foreach (int raw in axes)
        {
            int axis = raw < 0 ? raw + rank : raw;
            if ((uint)axis >= (uint)rank) return null;
            reduced[axis] = true;
        }
        int first = Array.IndexOf(reduced, true), last = Array.LastIndexOf(reduced, true);
        if (first < 0) return null;
        for (int axis = first; axis <= last; axis++) if (!reduced[axis]) return null;
        int outer = 1, repeats = 1, inner = 1;
        for (int axis = 0; axis < first; axis++) outer = checked(outer * input._shape[axis]);
        for (int axis = first; axis <= last; axis++) repeats = checked(repeats * input._shape[axis]);
        for (int axis = last + 1; axis < rank; axis++) inner = checked(inner * input._shape[axis]);
        if ((long)outer * inner != gradOut.Length || (long)outer * repeats * inner != gradIn.Length) return null;
        bool shared = writerCount.TryGetValue(input, out int writers) && writers > 1;

        return _ =>
        {
            MaterializeIfGradPlaceholder(gradIn);
            MaterializeIfGradPlaceholder(gradOut);
            bool accumulate = AccumulatesInto(gradIn, shared);
            if (typeof(T) == typeof(float))
            {
                var source = ElementTyped<float>(gradOut).GetCpuBackingForStridedRead(out int sourceOffset);
                var destination = ElementTyped<float>(gradIn).GetCpuBackingForContiguousWrite(out int destinationOffset);
                if (source is null || destination is null) throw new InvalidOperationException("Reduced gradient lost its contiguous CPU backing.");
                BroadcastReducedInto(source, sourceOffset, destination, destinationOffset, gradIn.Length, repeats, inner, accumulate);
            }
            else
            {
                var source = ElementTyped<double>(gradOut).GetCpuBackingForStridedRead(out int sourceOffset);
                var destination = ElementTyped<double>(gradIn).GetCpuBackingForContiguousWrite(out int destinationOffset);
                if (source is null || destination is null) throw new InvalidOperationException("Reduced gradient lost its contiguous CPU backing.");
                BroadcastReducedInto(source, sourceOffset, destination, destinationOffset, gradIn.Length, repeats, inner, accumulate);
            }
            input.Grad = gradIn;
        };
    }

    private static void BroadcastReducedInto(float[] source, int sourceOffset, float[] destination, int destinationOffset,
        int length, int repeats, int inner, bool accumulate)
    {
        const int chunkSize = 65536;
        int chunks = (length - 1) / chunkSize + 1;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, length, chunk =>
        {
            int at = chunk * chunkSize, end = at + Math.Min(chunkSize, length - at);
            while (at < end)
            {
                int row = at / (repeats * inner), column = at % inner;
                int count = Math.Min(end - at, inner == 1 ? (row + 1) * repeats - at : inner - column);
                int from = sourceOffset + row * inner + column, to = destinationOffset + at;
                if (!accumulate)
                {
                    if (inner == 1) destination.AsSpan(to, count).Fill(source[from]);
                    else source.AsSpan(from, count).CopyTo(destination.AsSpan(to, count));
                }
                else
                {
                    int width = System.Numerics.Vector<float>.Count, i = 0;
                    for (; i + width <= count; i += width)
                    {
                        var value = inner == 1 ? new System.Numerics.Vector<float>(source[from]) : new System.Numerics.Vector<float>(source, from + i);
                        (new System.Numerics.Vector<float>(destination, to + i) + value).CopyTo(destination, to + i);
                    }
                    for (; i < count; i++) destination[to + i] += source[from + (inner == 1 ? 0 : i)];
                }
                at += count;
            }
        }, deterministicSafe: true);
    }

    private static void BroadcastReducedInto(double[] source, int sourceOffset, double[] destination, int destinationOffset,
        int length, int repeats, int inner, bool accumulate)
    {
        const int chunkSize = 65536;
        int chunks = (length - 1) / chunkSize + 1;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, length, chunk =>
        {
            int at = chunk * chunkSize, end = at + Math.Min(chunkSize, length - at);
            while (at < end)
            {
                int row = at / (repeats * inner), column = at % inner;
                int count = Math.Min(end - at, inner == 1 ? (row + 1) * repeats - at : inner - column);
                int from = sourceOffset + row * inner + column, to = destinationOffset + at;
                if (!accumulate)
                {
                    if (inner == 1) destination.AsSpan(to, count).Fill(source[from]);
                    else source.AsSpan(from, count).CopyTo(destination.AsSpan(to, count));
                }
                else
                {
                    int width = System.Numerics.Vector<double>.Count, i = 0;
                    for (; i + width <= count; i += width)
                    {
                        var value = inner == 1 ? new System.Numerics.Vector<double>(source[from]) : new System.Numerics.Vector<double>(source, from + i);
                        (new System.Numerics.Vector<double>(destination, to + i) + value).CopyTo(destination, to + i);
                    }
                    for (; i < count; i++) destination[to + i] += source[from + (inner == 1 ? 0 : i)];
                }
                at += count;
            }
        }, deterministicSafe: true);
    }
}
