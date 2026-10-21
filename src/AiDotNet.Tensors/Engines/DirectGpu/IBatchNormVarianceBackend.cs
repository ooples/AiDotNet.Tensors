namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: batch-norm training forward and backward that keep the batch statistics as mean and biased
/// variance (what IEngine.BatchNorm reports), with each channel's slice held on-chip so the input is read once. False --
/// nothing launched -- when the shape is outside the variant (channel slice over 8192 elements, spatial not a multiple
/// of four); the caller then runs <see cref="IDirectGpuBackend.BatchNorm"/> / <see cref="IDirectGpuBackend.BatchNormBackward"/>.
/// </summary>
[BackendSpecificRoute("CUDA register-resident batch-norm training forward/backward on mean + variance (one input read, no inverse-std conversion launches); every backend runs BatchNorm and BatchNormBackward natively",
    nameof(IDirectGpuBackend.BatchNorm), nameof(IDirectGpuBackend.BatchNormBackward))]
internal interface IBatchNormVarianceBackend
{
    /// <summary>Training forward over [batch, channels, spatial]: every element of <paramref name="output"/> and of
    /// <paramref name="saveMean"/> / <paramref name="saveVariance"/> (biased variance) is written. With
    /// <paramref name="relu"/> the output is ReLU of the normalized values (a BatchNorm followed by a ReLU); with a
    /// <paramref name="residual"/> as well (same shape as the input), ReLU(BN(x) + residual).</summary>
    bool TryBatchNormTrainingWithVariance(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveVariance, int batch, int channels, int spatial, float epsilon, bool relu = false,
        IGpuBuffer? residual = null);

    /// <summary>Backward from the saved mean and biased variance: every element of the three gradients is written. With
    /// <paramref name="reluOutput"/> (the output of a BatchNorm followed by a ReLU), <paramref name="gradOutput"/> is the
    /// ReLU's output gradient and is masked where that output is not positive. With <paramref name="gradResidual"/> (the
    /// forward added a residual before the ReLU) the masked gradient -- the residual's gradient -- is written there too.
    /// With <paramref name="beta"/> and <paramref name="reluOutput"/> but no residual, the backend may rebuild the mask from
    /// the input instead of reading <paramref name="reluOutput"/>, valid only when the forward ran this backend's
    /// <see cref="TryBatchNormTrainingWithVariance"/>. <paramref name="gradOutput2"/>, when given, is a second contribution
    /// to the output gradient, summed as gradOutput + gradOutput2 before anything else.</summary>
    bool TryBatchNormBackwardFromVariance(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMean,
        IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatial, float epsilon, IGpuBuffer? reluOutput = null, IGpuBuffer? gradResidual = null,
        IGpuBuffer? beta = null, IGpuBuffer? gradOutput2 = null);
}
