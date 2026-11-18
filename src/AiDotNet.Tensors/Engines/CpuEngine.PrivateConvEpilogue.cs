using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using FloatVector = System.Numerics.Vector<float>;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    // Only a private, fully written CPU inference convolution destination may use this
    // internal entry point. Bias and residual retain their separate FP32 addition order.
    internal void AddPrivateConvolutionBiasResidual(Tensor<float> destination, Tensor<float>? bias, Tensor<float>? residual)
    {
        if (!NoGradScope<float>.IsSuppressed || GraphMode.IsActive)
            throw new InvalidOperationException("Private convolution epilogues require eager no-grad execution.");
        if (destination.Rank != 4 || !destination.IsContiguous)
            throw new ArgumentException("Contiguous NCHW convolution destination required.", nameof(destination));
        int channels = destination.Shape[1], rows = checked(destination.Shape[0] * channels);
        int spatial = checked(destination.Shape[2] * destination.Shape[3]);
        if (bias is not null && (bias.Length != channels || !bias.IsContiguous)) throw new ArgumentException("Channel bias shape mismatch.", nameof(bias));
        if (residual is not null && (residual.Length != destination.Length || !residual.IsContiguous)) throw new ArgumentException("Residual shape mismatch.", nameof(residual));
        if (bias is null && residual is null) return;
        var output = destination.GetCpuBackingForContiguousWrite(out int outputOffset)
            ?? throw new InvalidOperationException("Private CPU output requires managed backing.");
        int biasOffset = 0, residualOffset = 0;
        var b = bias?.GetCpuBackingForStridedRead(out biasOffset);
        var r = residual?.GetCpuBackingForStridedRead(out residualOffset);
        if (bias is not null && b is null || residual is not null && r is null)
            throw new InvalidOperationException("Private CPU epilogue inputs require managed backing.");
        CpuParallelSettings.ParallelForOrSerial(0, rows, destination.Length, row =>
        {
            int from = outputOffset + row * spatial, other = residualOffset + row * spatial;
            float channelBias = b is null ? 0f : b[biasOffset + row % channels];
            int column = 0;
#if NET5_0_OR_GREATER
            int width = FloatVector.Count;
            var vb = new FloatVector(channelBias);
            for (; column <= spatial - width; column += width)
            {
                var value = new FloatVector(output.AsSpan(from + column, width));
                if (b is not null) value += vb;
                if (r is not null) value += new FloatVector(r.AsSpan(other + column, width));
                value.CopyTo(output.AsSpan(from + column, width));
            }
#endif
            for (; column < spatial; column++)
            {
                float value = output[from + column];
                if (b is not null) value += channelBias;
                if (r is not null) value += r[other + column];
                output[from + column] = value;
            }
        }, deterministicSafe: true);
        destination.IncrementVersion();
    }
}
