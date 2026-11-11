// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The model-op kernels (TorchModelOpsKernels), compiled by hipRTC into their own precise-math module (HipBackend.cs)
// and launched through the batch launcher.
namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : ITorchModelOpsBackend
{
    private IntPtr _torchModelOpsModule;

    void ITorchModelOpsBackend.ModelActivationBackward(int op, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int n, float p0, float p1)
        => PortableTorchModelOps.ActivationBackward(LaunchBatch, op, gradOutput, input, gradInput, n, p0, p1);

    void ITorchModelOpsBackend.ModelFakeQuantize(IGpuBuffer input, IGpuBuffer partial, IGpuBuffer minMax, IGpuBuffer output, IGpuBuffer mask, int n,
        bool derive, bool symmetric, bool hasZeroPoint, float qMin, float qMax, float fixedStep, float zeroPoint)
        => PortableTorchModelOps.FakeQuantize(LaunchBatch, input, partial, minMax, output, mask, n, derive, symmetric, hasZeroPoint,
            qMin, qMax, fixedStep, zeroPoint);

    void ITorchModelOpsBackend.ModelTopKSoftmax(IGpuBuffer input, IGpuBuffer scratchValues, IGpuBuffer scratchIndices, IGpuBuffer scratchWeights, IGpuBuffer output, int rows, int width, int k)
        => PortableTorchModelOps.TopKSoftmax(LaunchBatch, input, scratchValues, scratchIndices, scratchWeights, output, rows, width, k);

    void ITorchModelOpsBackend.ModelTimeEmbedding(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer output, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbedding(LaunchBatch, timesteps, frequencies, output, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelTimeEmbeddingBackward(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer gradOutput,
        IGpuBuffer gradTimesteps, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbeddingBackward(LaunchBatch, timesteps, frequencies, gradOutput, gradTimesteps, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelCrfForward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer start, IGpuBuffer end, IGpuBuffer alpha,
        IGpuBuffer logZ, int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfForward(LaunchBatch, emissions, transitions, start, end, alpha, logZ, batch, seqLen, tags);

    void ITorchModelOpsBackend.ModelCrfBackward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer end, IGpuBuffer alpha,
        IGpuBuffer gradLogZ, IGpuBuffer beta, IGpuBuffer gradEmissions, IGpuBuffer gradTransitions, IGpuBuffer gradStart, IGpuBuffer gradEnd,
        int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfBackward(LaunchBatch, emissions, transitions, end, alpha, gradLogZ, beta, gradEmissions,
            gradTransitions, gradStart, gradEnd, batch, seqLen, tags);
}
