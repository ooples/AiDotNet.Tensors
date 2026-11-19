using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private static void ImproveGradientPacking(List<int> tenants, int[] firstWrite, int[] lastUse,
        List<Tensor<T>> gradients, ref int[] assignment, ref List<int> capacities)
    {
        if (Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_WEIGHTED_GRADIENT_PACKING") != "1") return;
        var intervals = new List<GradientBufferPacking.Interval>(tenants.Count);
        foreach (int g in tenants)
            intervals.Add(new GradientBufferPacking.Interval(g, gradients[g].Length, firstWrite[g], lastUse[g]));
        long Capacity(List<int> values)
        {
            long total = 0;
            foreach (int value in values) total += value;
            return total;
        }
        long best = Capacity(capacities);
        for (int variant = 0; variant < 2; variant++)
        {
            var packed = GradientBufferPacking.Pack(gradients.Count, intervals, variant == 1);
            long size = Capacity(packed.Capacities);
            if (size >= best) continue;
            best = size;
            assignment = packed.BankOf;
            capacities = packed.Capacities;
        }
    }
}
