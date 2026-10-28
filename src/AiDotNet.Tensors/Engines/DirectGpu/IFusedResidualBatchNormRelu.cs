// Copyright (c) CrowdTrainer LLC. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// relu(batchnorm_inference(input + residual)) with running statistics (#1154): one fused kernel on CUDA, the three
/// device ops in sequence elsewhere.
/// </summary>
internal interface IFusedResidualBatchNormRelu
{
    /// <summary>Writes the result to <paramref name="output"/>; false when this backend declines the shape.</summary>
    bool TryResidualBatchNormRelu(IGpuBuffer input, IGpuBuffer residual, IGpuBuffer output, IGpuBuffer gamma,
        IGpuBuffer beta, IGpuBuffer runningMean, IGpuBuffer runningVar, int batch, int channels, int spatialSize,
        float epsilon);
}

/// <summary>The composed form every backend without the fused kernel runs.</summary>
internal static class PortableResidualBatchNormRelu
{
    public static bool Run(IDirectGpuBackend backend, IGpuBuffer input, IGpuBuffer residual, IGpuBuffer output,
        IGpuBuffer gamma, IGpuBuffer beta, IGpuBuffer runningMean, IGpuBuffer runningVar, int batch, int channels,
        int spatialSize, float epsilon)
    {
        long count = (long)batch * channels * spatialSize;
        if (batch <= 0 || channels <= 0 || spatialSize <= 0 || count > int.MaxValue) return false;
        int n = (int)count;
        using var sum = backend.AllocateBuffer(n);
        using var saveMean = backend.AllocateBuffer(channels);
        using var saveInvVar = backend.AllocateBuffer(channels);
        backend.Add(input, residual, sum, n);
        // Inference: normalize with the running statistics, which training = false leaves untouched.
        backend.BatchNorm(sum, output, gamma, beta, runningMean, runningVar, saveMean, saveInvVar,
            batch, channels, spatialSize, epsilon, momentum: 0f, training: false);
        backend.Relu(output, output, n);
        return true;
    }
}