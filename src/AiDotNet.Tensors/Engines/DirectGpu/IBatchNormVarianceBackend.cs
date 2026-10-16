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
    /// <paramref name="saveMean"/> / <paramref name="saveVariance"/> (biased variance) is written.</summary>
    bool TryBatchNormTrainingWithVariance(IGpuBuffer input, IGpuBuffer output, IGpuBuffer gamma, IGpuBuffer beta,
        IGpuBuffer saveMean, IGpuBuffer saveVariance, int batch, int channels, int spatial, float epsilon);

    /// <summary>Backward from the saved mean and biased variance: every element of the three gradients is written.</summary>
    bool TryBatchNormBackwardFromVariance(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gamma, IGpuBuffer saveMean,
        IGpuBuffer saveVariance, IGpuBuffer gradInput, IGpuBuffer gradGamma, IGpuBuffer gradBeta,
        int batch, int channels, int spatial, float epsilon);
}
