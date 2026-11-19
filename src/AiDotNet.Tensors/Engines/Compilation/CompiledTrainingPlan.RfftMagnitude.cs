using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private static Action<IEngine>? BuildRfftMagnitudeBackward(CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradMap, Dictionary<Tensor<T>, int> writerCount, HashSet<Tensor<T>>? required)
    {
        if (step.OpType != OpType.TensorRfftMagnitude || step.Inputs.Length != 1
            || step.SavedState is not { Length: 1 } saved || saved[0] is not double epsilon
            || !gradMap.TryGetValue(step.OutputBuffer, out var gradOutput)) return null;
        var input = step.Inputs[0];
        if (required is not null && !required.Contains(input)) return _ => { };
        if (!gradMap.TryGetValue(input, out var gradInput)) return null;
        bool shared = writerCount.TryGetValue(input, out int writers) && writers > 1;
        return engine =>
        {
            MaterializeIfGradPlaceholder(gradOutput);
            MaterializeIfGradPlaceholder(gradInput);
            bool add = AccumulatesInto(gradInput, shared);
            engine.TensorRfftMagnitudeBackwardInto(gradOutput, input, gradInput, epsilon, add);
            input.Grad = gradInput;
        };
    }
}
