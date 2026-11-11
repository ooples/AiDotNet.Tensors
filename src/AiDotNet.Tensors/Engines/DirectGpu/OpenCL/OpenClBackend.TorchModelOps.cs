// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The model-op kernels (TorchModelOpsTranslation.OpenCl), built with strict math into their own program
// (OpenClBackend.cs). Launched with work groups of exactly 256: the CRF kernels run one sequence per group.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : ITorchModelOpsBackend
{
    private void LaunchModelOp(string name, int threads, params object[] args)
    {
        if (threads <= 0) return;
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new GpuCapabilityUnavailableException($"OpenCL kernel not found: {name}. The model-op module may have failed to compile.");
        for (int i = 0; i < args.Length; i++)
        {
            uint index = (uint)i;
            switch (args[i])
            {
                case IGpuBuffer buffer: kernel.SetArg(index, buffer.Handle); break;
                case int value: kernel.SetArg(index, value); break;
                case float value: kernel.SetArg(index, value); break;
                default:
                    throw new System.ArgumentException($"Unsupported argument type {args[i]?.GetType().Name ?? "null"} for {name}.");
            }
        }
        kernel.Execute1D(threads, PortableTorchModelOps.CrfGroupSize);
    }

    void ITorchModelOpsBackend.ModelActivationBackward(int op, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int n, float p0, float p1)
        => PortableTorchModelOps.ActivationBackward(LaunchModelOp, op, gradOutput, input, gradInput, n, p0, p1);

    void ITorchModelOpsBackend.ModelFakeQuantize(IGpuBuffer input, IGpuBuffer partial, IGpuBuffer minMax, IGpuBuffer output, IGpuBuffer mask, int n,
        bool derive, bool symmetric, bool hasZeroPoint, float qMin, float qMax, float fixedStep, float zeroPoint)
        => PortableTorchModelOps.FakeQuantize(LaunchModelOp, input, partial, minMax, output, mask, n, derive, symmetric, hasZeroPoint,
            qMin, qMax, fixedStep, zeroPoint);

    void ITorchModelOpsBackend.ModelTopKSoftmax(IGpuBuffer input, IGpuBuffer scratchValues, IGpuBuffer scratchIndices, IGpuBuffer scratchWeights, IGpuBuffer output, int rows, int width, int k)
        => PortableTorchModelOps.TopKSoftmax(LaunchModelOp, input, scratchValues, scratchIndices, scratchWeights, output, rows, width, k);

    void ITorchModelOpsBackend.ModelTimeEmbedding(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer output, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbedding(LaunchModelOp, timesteps, frequencies, output, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelTimeEmbeddingBackward(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer gradOutput,
        IGpuBuffer gradTimesteps, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbeddingBackward(LaunchModelOp, timesteps, frequencies, gradOutput, gradTimesteps, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelCrfForward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer start, IGpuBuffer end, IGpuBuffer alpha,
        IGpuBuffer logZ, int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfForward(LaunchModelOp, emissions, transitions, start, end, alpha, logZ, batch, seqLen, tags);

    void ITorchModelOpsBackend.ModelCrfBackward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer end, IGpuBuffer alpha,
        IGpuBuffer gradLogZ, IGpuBuffer beta, IGpuBuffer gradEmissions, IGpuBuffer gradTransitions, IGpuBuffer gradStart, IGpuBuffer gradEnd,
        int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfBackward(LaunchModelOp, emissions, transitions, end, alpha, gradLogZ, beta, gradEmissions,
            gradTransitions, gradStart, gradEnd, batch, seqLen, tags);
}
