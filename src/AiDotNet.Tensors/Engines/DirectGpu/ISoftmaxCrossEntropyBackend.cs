namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: IEngine.TensorSoftmaxCrossEntropy as fused kernels -- a per-row pass, a deterministic total and
/// a one-pass backward -- instead of the composed log-softmax, multiply, reductions and supervised-row count (about forty
/// kernels, memsets and copies per training step for a [64, 10] classifier head on CUDA).
/// </summary>
[BackendSpecificRoute("CUDA fused soft-target softmax cross-entropy; every backend runs the composed log-softmax and reductions natively",
    nameof(IDirectGpuBackend.MaxAxis), nameof(IDirectGpuBackend.SumAxis), nameof(IDirectGpuBackend.Exp), nameof(IDirectGpuBackend.Log))]
internal interface ISoftmaxCrossEntropyBackend
{
    /// <summary>Forward over [rows, cols]: writes <paramref name="rowStats"/> (4 * rows floats, kept for the backward),
    /// <paramref name="loss"/>[0] and <paramref name="count"/>[0] (the supervised-row count, at least one). False --
    /// nothing launched -- when unavailable.</summary>
    bool TrySoftmaxCrossEntropyForward(IGpuBuffer logits, IGpuBuffer target, IGpuBuffer rowStats, IGpuBuffer loss,
        IGpuBuffer count, int rows, int cols);

    /// <summary>Backward from the forward's <paramref name="rowStats"/> and <paramref name="count"/>:
    /// <c>gradLogits = gradOutput[0] / count · (softmax · Σ target − target)</c>, overwriting or, when
    /// <paramref name="add"/>, added. False -- nothing launched -- when unavailable.</summary>
    bool TrySoftmaxCrossEntropyBackward(IGpuBuffer logits, IGpuBuffer target, IGpuBuffer rowStats, IGpuBuffer count,
        IGpuBuffer gradOutput, IGpuBuffer gradLogits, int rows, int cols, bool add);
}
