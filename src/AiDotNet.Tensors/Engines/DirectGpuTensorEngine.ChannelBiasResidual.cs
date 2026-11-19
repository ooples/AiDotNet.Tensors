using System.Diagnostics.CodeAnalysis;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    /// <summary>Explicit inference composition; raw residual and optional activated carry have independent storage.</summary>
    internal bool TryChannelBiasResidualInference<T>(Tensor<T> input, Tensor<T> bias, Tensor<T> residual,
        T alpha, bool writeActivation, [NotNullWhen(true)] out Tensor<T>? output, out Tensor<T>? activated)
    {
        output = null;
        activated = null;
        if (Environment.GetEnvironmentVariable("AIDOTNET_GPU_INFERENCE_CHANNEL_BIAS_RESIDUAL") == "0"
            || this.Unwrap().GetType() != typeof(DirectGpuTensorEngine)
            || typeof(T) != typeof(float) || Autodiff.GradientTape<T>.Current is not null
            || Compilation.GraphMode.IsActive || ResidentStepActive || InferenceCaptureActive
            || Gpu.GpuBufferReleaseDeferral.IsActive || Gpu.AutocastScope.IsEnabled
            || !IsWholeContiguous(input) || !IsWholeContiguous(residual)
            || input.Rank != 4 || input.Length == 0 || !input._shape.AsSpan().SequenceEqual(residual._shape)
            || bias.Rank != 1 || !bias.IsContiguous || bias._storageOffset != 0 || input.Shape[1] != bias.Length)
            return false;
        float slope = ToFloatScalar(alpha);
        int batch = input.Shape[0], channels = input.Shape[1];
        if (!(slope > 0) || float.IsInfinity(slope) || batch <= 0 || channels <= 0) return false;
        using var operation = EnterGpuOp();
        if (!operation.Held || !TryGetBackend(out var backend) || backend is not IChannelBiasResidualBackend fused)
            return false;
        if (backend is IGpuGraphCapture capture && capture.IsStreamCapturing()) return false;
        Tensor<T>? sum = null;
        Tensor<T>? activation = null;
        try
        {
            using var source = GetOrAllocateBuffer(backend, input);
            using var biases = GetWeightBufferPreferResident(backend, bias, PersistentTensorRole.Biases);
            using var carry = GetOrAllocateBuffer(backend, residual);
            using var destination = AllocateFullyWrittenOutputBuffer(backend, input.Length);
            using var activatedDestination = writeActivation ? AllocateFullyWrittenOutputBuffer(backend, input.Length) : default;
            int spatial = input.Length / checked(batch * channels);
            if (!fused.TryChannelBiasResidual(source.Buffer, biases.Buffer, carry.Buffer, destination.Buffer,
                writeActivation ? activatedDestination.Buffer : destination.Buffer, batch, channels, spatial, slope, writeActivation))
                return false;
            sum = DeferTensorResult<T>(backend, destination.Buffer, input.Length, input.Shape.ToArray());
            destination.RelinquishOwnership();
            if (writeActivation)
            {
                activation = DeferTensorResult<T>(backend, activatedDestination.Buffer, input.Length, input.Shape.ToArray());
                activatedDestination.RelinquishOwnership();
            }
            output = sum;
            activated = activation;
            sum = null;
            activation = null;
            return true;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("ChannelBiasResidualInference", capabilityGap);
            return false;
        }
        finally
        {
            sum?.Dispose();
            activation?.Dispose();
        }
    }

    private static bool IsWholeContiguous<T>(Tensor<T> tensor)
        => tensor.IsContiguous && tensor._storageOffset == 0
            && tensor.GraphVectorIdentity is Vector<T> vector && vector.Length == tensor.Length;
}
