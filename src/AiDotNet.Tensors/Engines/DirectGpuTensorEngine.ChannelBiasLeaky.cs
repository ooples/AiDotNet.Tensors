using System.Diagnostics.CodeAnalysis;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>Explicit inference-only composition; no graph provenance or input mutation is used.</summary>
    internal bool TryChannelBiasLeakyInference<T>(Tensor<T> input, Tensor<T> bias, T alpha,
        [NotNullWhen(true)] out Tensor<T>? output)
    {
        output = null;
        if (Environment.GetEnvironmentVariable("AIDOTNET_GPU_INFERENCE_CHANNEL_BIAS_LEAKY") == "0"
            || this.Unwrap().GetType() != typeof(DirectGpuTensorEngine)
            || typeof(T) != typeof(float) || Autodiff.GradientTape<T>.Current is not null
            || Compilation.GraphMode.IsActive
            || ResidentStepActive || InferenceCaptureActive || Gpu.GpuBufferReleaseDeferral.IsActive
            || Gpu.AutocastScope.IsEnabled || input.Rank != 4 || input.Length == 0
            || !input.IsContiguous || input._storageOffset != 0
            || input.GraphVectorIdentity is not Vector<T> vector || vector.Length != input.Length
            || bias.Rank != 1 || !bias.IsContiguous || bias._storageOffset != 0
            || input.Shape[1] != bias.Length)
            return false;
        float slope = ToFloatScalar(alpha);
        int batch = input.Shape[0], channels = input.Shape[1];
        if (!(slope > 0) || float.IsInfinity(slope) || batch <= 0 || channels <= 0)
            return false;
        using var operation = EnterGpuOp();
        if (!operation.Held || !TryGetBackend(out var backend) || backend is not IChannelBiasLeakyBackend fused)
            return false;
        if (backend is IGpuGraphCapture capture && capture.IsStreamCapturing()) return false;
        try
        {
            using var source = GetOrAllocateBuffer(backend, input);
            using var biases = GetWeightBufferPreferResident(backend, bias, PersistentTensorRole.Biases);
            using var destination = AllocateFullyWrittenOutputBuffer(backend, input.Length);
            int spatial = input.Length / checked(batch * channels);
            if (!fused.TryChannelBiasLeaky(source.Buffer, biases.Buffer, destination.Buffer, batch, channels, spatial, slope))
                return false;
            output = DeferTensorResult<T>(backend, destination.Buffer, input.Length, input.Shape.ToArray());
            destination.RelinquishOwnership();
            return true;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("ChannelBiasLeakyInference", capabilityGap);
            return false;
        }
    }
}
