using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// Differentiates each slot's product without division or value-dependent masks. Prefix and suffix products
    /// exclude exactly one factor, preserving mixed derivatives even when several factors are zero.
    /// Slots with the same number of source factors share a batch; total intermediate storage is linear in the
    /// number of participating factors rather than destination length times maximum collision count.
    /// </summary>
    private static void ScatterReduceProdWithZerosRecorded(Tensor<T> gradOutput, Tensor<T> tensor, Tensor<T> source,
        int[] idxData, bool includeSelf,
        int outerSize, int innerSize, int dstAxis, int srcAxis, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var slots = new List<int>?[tensor.Length];
        for (int outer = 0; outer < outerSize; outer++)
        for (int i = 0; i < srcAxis; i++)
        for (int inner = 0; inner < innerSize; inner++)
        {
            int src = (outer * srcAxis + i) * innerSize + inner;
            int target = idxData[src];
            if (target < 0 || target >= dstAxis) continue;
            int dst = (outer * dstAxis + target) * innerSize + inner;
            (slots[dst] ??= new List<int>()).Add(src);
        }

        var groups = new Dictionary<int, List<int>>();
        var untouched = new List<int>();
        for (int dst = 0; dst < slots.Length; dst++)
        {
            if (slots[dst] is not { } factors) { untouched.Add(dst); continue; }
            if (!groups.TryGetValue(factors.Count, out var group))
                groups[factors.Count] = group = new List<int>();
            group.Add(dst);
        }
        var selfParts = new List<Tensor<T>>();
        var selfPositions = new List<int>(untouched);
        if (untouched.Count > 0)
        {
            var to = new int[untouched.Count];
            for (int i = 0; i < to.Length; i++) to[i] = i;
            selfParts.Add(SparseMap(gradOutput, new[] { to.Length }, to, untouched.ToArray(), null));
        }
        var sourceParts = new List<Tensor<T>>();
        var sourceTargets = new List<int>();
        var ops = MathHelper.GetNumericOperations<T>();
        foreach (var entry in groups)
        {
            int depth = entry.Key, count = entry.Value.Count;
            var destinations = entry.Value.ToArray();
            var positions = new int[count];
            for (int j = 0; j < count; j++) positions[j] = j;
            var shape = new[] { count };
            var ones = new Tensor<T>(shape);
            ones.Fill(ops.One);
            var upstream = SparseMap(gradOutput, shape, positions, destinations, null);
            var factors = new Tensor<T>[depth];
            var sourcePositions = new int[depth][];
            var prefix = new Tensor<T>[depth + 1];
            prefix[0] = includeSelf ? SparseMap(tensor, shape, positions, destinations, null) : ones;
            for (int i = 0; i < depth; i++)
            {
                var from = new int[count];
                for (int j = 0; j < count; j++) from[j] = slots[destinations[j]]![i];
                sourcePositions[i] = from;
                factors[i] = SparseMap(source, shape, positions, from, null);
                prefix[i + 1] = engine.TensorMultiply(prefix[i], factors[i]);
            }

            var suffix = ones;
            for (int i = depth - 1; i >= 0; i--)
            {
                var contribution = engine.TensorMultiply(upstream, engine.TensorMultiply(prefix[i], suffix));
                sourceParts.Add(contribution);
                sourceTargets.AddRange(sourcePositions[i]);
                suffix = engine.TensorMultiply(factors[i], suffix);
            }
            if (includeSelf)
            {
                selfParts.Add(engine.TensorMultiply(upstream, suffix));
                selfPositions.AddRange(destinations);
            }
        }

        Tensor<T> Assemble(List<Tensor<T>> parts, List<int> targets, int[] shape)
        {
            if (parts.Count == 0) return new Tensor<T>((int[])shape.Clone());
            var packed = parts.Count == 1 ? parts[0] : engine.TensorConcatenate(parts.ToArray(), 0);
            var from = new int[targets.Count];
            for (int i = 0; i < from.Length; i++) from[i] = i;
            return SparseMap(packed, (int[])shape.Clone(), targets.ToArray(), from, null);
        }
        DifferentiableOps.AccumulateGrad(grads, tensor, Assemble(selfParts, selfPositions, tensor._shape), engine);
        DifferentiableOps.AccumulateGrad(grads, source, Assemble(sourceParts, sourceTargets, source._shape), engine);
    }
}
