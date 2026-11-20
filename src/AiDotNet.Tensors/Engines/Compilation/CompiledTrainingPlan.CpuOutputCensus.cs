using System.Runtime.CompilerServices;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    // Metadata only, under the existing opt-in plan storage diagnostic. This does
    // not realize tensors, obtain spans, rebind storage, or install an observer.
    private void TraceCpuOutputCensus()
    {
        if (_engine.SupportsGpu || _forwardSteps is not { } steps) return;
        var arrays = new HashSet<object>();
        var operations = new SortedDictionary<string, (int Count, long Bytes)>();
        var consumers = new Dictionary<Tensor<T>, int>();
        var producers = new Dictionary<Tensor<T>, CompiledStep<T>>();
        foreach (var step in steps)
        {
            producers[step.OutputBuffer] = step;
            foreach (var input in step.Inputs)
                consumers[input] = consumers.TryGetValue(input, out int count) ? count + 1 : 1;
            if (step.OutputBuffer.GetBackingArrayForCacheLookupUnsafe() is not { } array || !arrays.Add(array)) continue;
            string key = step.OpName ?? step.OpType.ToString();
            operations.TryGetValue(key, out var row);
            operations[key] = (row.Count + 1, row.Bytes + (long)array.Length * Unsafe.SizeOf<T>());
        }
        foreach (var operation in operations.OrderByDescending(row => row.Value.Bytes))
            AiDotNetEngine.LogDiagnostic($"[PLAN-FORWARD] plan={_planId} op={operation.Key} arrays={operation.Value.Count} bytes={operation.Value.Bytes}");

        var opportunities = new SortedDictionary<string, (int Count, long Bytes)>();
        foreach (var step in steps)
        {
            if (step.OpType != OpType.LeakyReLU || step.Inputs.Length != 1) continue;
            var input = step.Inputs[0];
            string source = producers.TryGetValue(input, out var producer) ? producer.OpName : "external";
            bool single = consumers.TryGetValue(input, out int count) && count == 1;
            bool retained = _retainedOutputs.Contains(input) || _retainedOutputs.Contains(step.OutputBuffer)
                || _namedRetainedOutputs.Values.Contains(input) || _namedRetainedOutputs.Values.Contains(step.OutputBuffer);
            bool distinct = input.GetBackingArrayForCacheLookupUnsafe() is { } ia
                && step.OutputBuffer.GetBackingArrayForCacheLookupUnsafe() is { } oa && !ReferenceEquals(ia, oa);
            string key = source + "/single=" + single + "/retained=" + retained + "/distinct=" + distinct;
            opportunities.TryGetValue(key, out var row);
            opportunities[key] = (row.Count + 1, row.Bytes + (long)input.Length * Unsafe.SizeOf<T>());
        }
        foreach (var opportunity in opportunities.OrderByDescending(row => row.Value.Bytes))
            AiDotNetEngine.LogDiagnostic($"[PLAN-LEAKY] plan={_planId} source={opportunity.Key} count={opportunity.Value.Count} logical_bytes={opportunity.Value.Bytes}");
    }
}
