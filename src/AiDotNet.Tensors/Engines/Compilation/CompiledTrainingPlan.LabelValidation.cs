using System;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private bool _fusedCeLabelsValidated;
    // Validate the separate finite label-error word before any optimizer writes. Gradient sanitization may repair
    // NaN gradients, but cannot make an invalid label acceptable. These plans capture forward/backward; the update
    // stays outside the graph so its validation also precedes weight decay and moment updates.
    private void ThrowOnInvalidCurrentFusedCeLabels()
    {
        if (!_usesFusedCe || _fusedCeLabelsValidated) return;
        if (_engine.TryGet<DirectGpuTensorEngine>(out var gpu) && gpu.TakeFusedCeLabelError())
            throw new ArgumentOutOfRangeException("targetIds",
                "A fused cross-entropy target id on this compiled step was outside [0, vocab). The step was not applied.");
        _fusedCeLabelsValidated = true;
    }
}
