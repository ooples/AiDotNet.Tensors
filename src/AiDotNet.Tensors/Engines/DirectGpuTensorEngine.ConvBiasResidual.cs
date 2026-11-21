using System.Diagnostics.CodeAnalysis;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    internal bool TryConvBiasResidualInference<T>(Tensor<T> input, Tensor<T> kernel, Tensor<T> bias,
        Tensor<T> residual, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW,
        T alpha, bool writeActivation, [NotNullWhen(true)] out Tensor<T>? output, out Tensor<T>? activated)
    {
        output = null; activated = null;
        if (Environment.GetEnvironmentVariable("AIDOTNET_GPU_INFERENCE_CONV_BIAS_RESIDUAL") != "1"
            || this.Unwrap().GetType() != typeof(DirectGpuTensorEngine) || typeof(T) != typeof(float)
            || Autodiff.GradientTape<T>.Current is not null || Compilation.GraphMode.IsActive
            || ResidentStepActive || InferenceCaptureActive || Gpu.GpuBufferReleaseDeferral.IsActive || Gpu.AutocastScope.IsEnabled
            || input.Rank != 4 || kernel.Rank != 4 || residual.Rank != 4 || bias.Rank != 1
            || !IsWholeContiguous(input) || !IsWholeContiguous(kernel) || !IsWholeContiguous(residual) || !IsWholeContiguous(bias)
            || input.Length == 0 || residual.Length == 0 || kernel.Shape[1] != input.Shape[1]
            || bias.Length != kernel.Shape[0] || residual.Shape[0] != input.Shape[0] || residual.Shape[1] != kernel.Shape[0]) return false;
        using var operation = EnterGpuOp();
        if (!operation.Held || !TryGetBackend(out var backend) || backend is not IConvBiasResidualBackend composition
            || backend is IGpuGraphCapture capture && capture.IsStreamCapturing()) return false;
        Tensor<T>? sum = null, activation = null;
        try
        {
            using var source = GetOrAllocateBuffer(backend, input);
            using var weights = GetWeightBufferPreferResident(backend, kernel, PersistentTensorRole.Weights);
            using var biases = GetWeightBufferPreferResident(backend, bias, PersistentTensorRole.Biases);
            using var carry = GetOrAllocateBuffer(backend, residual);
            using var destination = AllocateFullyWrittenOutputBuffer(backend, residual.Length);
            using var activatedDestination = writeActivation ? AllocateFullyWrittenOutputBuffer(backend, residual.Length) : default;
            var args = new ConvBiasResidualArgs(source.Buffer, weights.Buffer, biases.Buffer, carry.Buffer, destination.Buffer,
                writeActivation ? activatedDestination.Buffer : destination.Buffer, input.Shape[0], input.Shape[1], input.Shape[2], input.Shape[3],
                kernel.Shape[0], residual.Shape[2], residual.Shape[3], kernel.Shape[2], kernel.Shape[3],
                strideH, strideW, padH, padW, dilationH, dilationW, ToFloatScalar(alpha), writeActivation);
            if (!composition.TryConvBiasResidual(args)) return false;
            sum = DeferTensorResult<T>(backend, destination.Buffer, residual.Length, residual.Shape.ToArray());
            destination.RelinquishOwnership();
            if (writeActivation)
            {
                activation = DeferTensorResult<T>(backend, activatedDestination.Buffer, residual.Length, residual.Shape.ToArray());
                activatedDestination.RelinquishOwnership();
            }
            output = sum; activated = activation; sum = null; activation = null;
            return true;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("ConvBiasResidualInference", capabilityGap);
            return false;
        }
        finally { sum?.Dispose(); activation?.Dispose(); }
    }
}
