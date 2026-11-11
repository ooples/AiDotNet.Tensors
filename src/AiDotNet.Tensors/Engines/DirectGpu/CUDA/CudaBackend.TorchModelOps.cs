// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The model-op kernels (TorchModelOpsKernels), compiled by NVRTC into their own precise-math module (CudaBackend.cs)
// and launched through the batch launcher.
namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : ITorchModelOpsBackend
{
    private IntPtr _torchModelOpsModule;

    void ITorchModelOpsBackend.ModelActivationBackward(int op, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int n, float p0, float p1)
        => PortableTorchModelOps.ActivationBackward(LaunchBatchKernel, op, gradOutput, input, gradInput, n, p0, p1);

    void ITorchModelOpsBackend.ModelFakeQuantize(IGpuBuffer input, IGpuBuffer partial, IGpuBuffer minMax, IGpuBuffer output, IGpuBuffer mask, int n,
        bool derive, bool symmetric, bool hasZeroPoint, float qMin, float qMax, float fixedStep, float zeroPoint)
        => PortableTorchModelOps.FakeQuantize(LaunchBatchKernel, input, partial, minMax, output, mask, n, derive, symmetric, hasZeroPoint,
            qMin, qMax, fixedStep, zeroPoint);

    void ITorchModelOpsBackend.ModelTopKSoftmax(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output, int rows, int width, int k)
        => PortableTorchModelOps.TopKSoftmax(LaunchBatchKernel, input, scratch, output, rows, width, k);

    void ITorchModelOpsBackend.ModelTimeEmbedding(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer output, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbedding(LaunchBatchKernel, timesteps, frequencies, output, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelTimeEmbeddingBackward(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer gradOutput,
        IGpuBuffer gradTimesteps, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbeddingBackward(LaunchBatchKernel, timesteps, frequencies, gradOutput, gradTimesteps, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelCrfForward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer start, IGpuBuffer end, IGpuBuffer alpha,
        IGpuBuffer logZ, int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfForward(LaunchBatchKernel, emissions, transitions, start, end, alpha, logZ, batch, seqLen, tags);

    void ITorchModelOpsBackend.ModelCrfBackward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer end, IGpuBuffer alpha, IGpuBuffer logZ,
        IGpuBuffer gradLogZ, IGpuBuffer beta, IGpuBuffer gradEmissions, IGpuBuffer gradTransitions, IGpuBuffer gradStart, IGpuBuffer gradEnd,
        int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfBackward(LaunchBatchKernel, emissions, transitions, end, alpha, logZ, gradLogZ, beta, gradEmissions,
            gradTransitions, gradStart, gradEnd, batch, seqLen, tags);
}
