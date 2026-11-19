// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private static Action<IEngine>? BuildWeightNormBackward(CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradMap, Dictionary<Tensor<T>, int> writerCount,
        HashSet<Tensor<T>>? required)
    {
        if (step.OpType != OpType.TensorWeightNorm || step.Inputs.Length != 2
            || step.SavedState is not { Length: 1 } saved || saved[0] is not double epsilon
            || !gradMap.TryGetValue(step.OutputBuffer, out var gradOutput))
            return null;
        var direction = step.Inputs[0];
        var gain = step.Inputs[1];
        bool needDirection = required is null || required.Contains(direction);
        bool needGain = required is null || required.Contains(gain);
        Tensor<T>? gradDirection = null, gradGain = null;
        if (needDirection && !gradMap.TryGetValue(direction, out gradDirection)) return null;
        if (needGain && !gradMap.TryGetValue(gain, out gradGain)) return null;
        bool tied = ReferenceEquals(direction, gain);
        bool sharedDirection = tied || (writerCount.TryGetValue(direction, out int dvWriters) && dvWriters > 1);
        bool sharedGain = tied || (writerCount.TryGetValue(gain, out int dgWriters) && dgWriters > 1);
        return engine =>
        {
            // Pooling may rebind these tensors after delegate construction; resolve
            // storage and first-write state only during the actual action.
            MaterializeIfGradPlaceholder(gradOutput);
            if (gradDirection is not null) MaterializeIfGradPlaceholder(gradDirection);
            if (gradGain is not null) MaterializeIfGradPlaceholder(gradGain);
            bool addDirection = gradDirection is not null && AccumulatesInto(gradDirection, sharedDirection);
            bool addGain = gradGain is not null && (tied && gradDirection is not null
                ? ClaimTiedGain() : AccumulatesInto(gradGain, sharedGain));
            engine.TensorWeightNormBackwardInto(gradOutput, direction, gain, gradDirection, gradGain,
                epsilon, addDirection, addGain);
            if (gradDirection is not null) direction.Grad = gradDirection;
            if (gradGain is not null) gain.Grad = gradGain;

            bool ClaimTiedGain()
            {
                Autodiff.DifferentiableOps.ClaimFirstWrite(gradGain!);
                return true;
            }
        };
    }
}
