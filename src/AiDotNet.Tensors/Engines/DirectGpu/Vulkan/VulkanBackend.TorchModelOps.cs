// Copyright (c) CrowdTrainer LLC. All rights reserved.
// The model-op kernels (TorchModelOpsTranslation.Glsl): one compute shader per kernel, buffers bound in argument order
// and every int / float scalar one push-constant word. They compute in float: no shaderFloat64 / shaderInt64 needed.
namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : ITorchModelOpsBackend
{
    private void LaunchModelOp(string name, int threads, params object[] args)
    {
        if (threads <= 0) return;
        if (!TorchModelOpsTranslation.Glsl.TryGetValue(name, out var source))
            throw new GpuCapabilityUnavailableException($"Vulkan model-op kernel not found: {name}");
        if (!IsGlslCompilerAvailable)
            throw new GpuCapabilityUnavailableException($"Vulkan model-op kernel {name} needs the runtime GLSL compiler (libshaderc).");
        var buffers = new System.Collections.Generic.List<IGpuBuffer>();
        var push = new System.Collections.Generic.List<uint>();
        foreach (var arg in args)
        {
            switch (arg)
            {
                case IGpuBuffer buffer: buffers.Add(buffer); break;
                case int value: push.Add(Bits(value)); break;
                case float value: push.Add(FloatBits(value)); break;
                default:
                    throw new System.ArgumentException($"Unsupported argument type {arg?.GetType().Name ?? "null"} for {name}.");
            }
        }
        GlslNaryOp(source, buffers.ToArray(), threads, push.ToArray(), 256);
    }

    void ITorchModelOpsBackend.ModelActivationBackward(int op, IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradInput, int n, float p0, float p1)
        => PortableTorchModelOps.ActivationBackward(LaunchModelOp, op, gradOutput, input, gradInput, n, p0, p1);

    void ITorchModelOpsBackend.ModelFakeQuantize(IGpuBuffer input, IGpuBuffer partial, IGpuBuffer minMax, IGpuBuffer output, IGpuBuffer mask, int n,
        bool derive, bool symmetric, bool hasZeroPoint, float qMin, float qMax, float fixedStep, float zeroPoint)
        => PortableTorchModelOps.FakeQuantize(LaunchModelOp, input, partial, minMax, output, mask, n, derive, symmetric, hasZeroPoint,
            qMin, qMax, fixedStep, zeroPoint);

    void ITorchModelOpsBackend.ModelTopKSoftmax(IGpuBuffer input, IGpuBuffer scratch, IGpuBuffer output, int rows, int width, int k)
        => PortableTorchModelOps.TopKSoftmax(LaunchModelOp, input, scratch, output, rows, width, k);

    void ITorchModelOpsBackend.ModelTimeEmbedding(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer output, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbedding(LaunchModelOp, timesteps, frequencies, output, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelTimeEmbeddingBackward(IGpuBuffer timesteps, IGpuBuffer frequencies, IGpuBuffer gradOutput,
        IGpuBuffer gradTimesteps, int batch, int halfDim, int dim)
        => PortableTorchModelOps.TimeEmbeddingBackward(LaunchModelOp, timesteps, frequencies, gradOutput, gradTimesteps, batch, halfDim, dim);

    void ITorchModelOpsBackend.ModelCrfForward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer start, IGpuBuffer end, IGpuBuffer alpha,
        IGpuBuffer logZ, int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfForward(LaunchModelOp, emissions, transitions, start, end, alpha, logZ, batch, seqLen, tags);

    void ITorchModelOpsBackend.ModelCrfBackward(IGpuBuffer emissions, IGpuBuffer transitions, IGpuBuffer end, IGpuBuffer alpha, IGpuBuffer logZ,
        IGpuBuffer gradLogZ, IGpuBuffer beta, IGpuBuffer gradEmissions, IGpuBuffer gradTransitions, IGpuBuffer gradStart, IGpuBuffer gradEnd,
        int batch, int seqLen, int tags)
        => PortableTorchModelOps.CrfBackward(LaunchModelOp, emissions, transitions, end, alpha, logZ, gradLogZ, beta, gradEmissions,
            gradTransitions, gradStart, gradEnd, batch, seqLen, tags);
}
