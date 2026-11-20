using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    // Diagnostic-only export of the planner's actual integer liveness metadata. No tensor payload or execution changes.
    private static void ExportGradientPacking(List<int> tenants, int[] firstWrite, int[] lastUse,
        List<Tensor<T>> gradients, int[] arrayOf, List<int> capacities)
    {
        string? directory = Environment.GetEnvironmentVariable("AIDOTNET_GRAD_POOL_PLAN_EXPORT");
        if (string.IsNullOrEmpty(directory)) return;
        var json = new StringBuilder("{\"schema\":\"gradient-packing-audit-v1\",\"tenants\":[");
        bool first = true;
        foreach (int g in tenants)
        {
            if (!first) json.Append(',');
            first = false;
            json.Append('[').Append(g).Append(',').Append(gradients[g].Length).Append(',')
                .Append(firstWrite[g]).Append(',').Append(lastUse[g]).Append(',').Append(arrayOf[g]).Append(']');
        }
        json.Append("],\"capacities\":[");
        for (int i = 0; i < capacities.Count; i++)
        {
            if (i != 0) json.Append(',');
            json.Append(capacities[i]);
        }
        json.Append("]}");
        Directory.CreateDirectory(directory);
        File.WriteAllText(Path.Combine(directory, Guid.NewGuid().ToString("N") + ".json"), json.ToString());
    }
}
