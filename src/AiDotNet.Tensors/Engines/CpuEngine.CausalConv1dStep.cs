using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// One decode position of a causal depthwise convolution (the short convolution of Mamba / Gated DeltaNet blocks) with
// the previous K - 1 inputs carried in a fixed history tensor shifted in place, so a compiled decode plan keeps the
// history in one persistent buffer.
public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> CausalConv1dStep<T>(Tensor<T> history, Tensor<T> input, Tensor<T> weight, Tensor<T>? bias, bool silu)
    {
        var (taps, batch, channels) = ValidateCausalConv1dStep(history, input, weight, bias);
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(this);
            var (ch, cx, cw, cb) = (history, input, weight, bias);
            var inputs = bias is null ? new[] { history, input, weight } : new[] { history, input, weight, bias };
            return scope.RecordVariadicWritingInPlace(LazyNodeType.Custom, "CausalConv1dStep", inputs, new[] { batch, channels }, history,
                (eng, output) => DirectGpuTensorEngine.CopyResultInto(eng, eng.CausalConv1dStep(ch, cx, cw, cb, silu), output));
        }
        return CausalConv1dStepHost(history, input, weight, bias, silu, taps, batch, channels);
    }

    /// <summary>
    /// The composed implementation of <see cref="CausalConv1dStep{T}"/> from this engine's ops (the reference, and the
    /// fallback of device engines without the fused kernel).
    /// </summary>
    private protected Tensor<T> CausalConv1dStepHost<T>(Tensor<T> history, Tensor<T> input, Tensor<T> weight, Tensor<T>? bias,
        bool silu, int taps, int batch, int channels)
    {
        Tensor<T> Tap(int m) => Reshape(TensorSlice(weight, new[] { 0, m }, new[] { channels, 1 }), new[] { 1, channels });
        Tensor<T> Past(int m) => Reshape(TensorSlice(history, new[] { m - 1, 0, 0 }, new[] { 1, batch, channels }), new[] { batch, channels });
        var x = Reshape(input, new[] { batch, channels });
        var y = TensorMultiply(x, Tap(0));
        for (int m = 1; m <= taps; m++) y = TensorAdd(y, TensorMultiply(Past(m), Tap(m)));
        if (bias is not null) y = TensorAdd(y, Reshape(bias, new[] { 1, channels }));
        if (silu) y = Swish(y);
        if (taps > 0)
        {
            var shifted = taps == 1
                ? Reshape(x, new[] { 1, batch, channels })
                : TensorConcatenate(new[] { Reshape(x, new[] { 1, batch, channels }),
                    TensorSlice(history, new[] { 0, 0, 0 }, new[] { taps - 1, batch, channels }) }, 0);
            ((Compilation.ISignatureTensor)history).CopyLeafFrom(shifted);
        }
        return y;
    }

    private protected static (int Taps, int Batch, int Channels) ValidateCausalConv1dStep<T>(Tensor<T> history, Tensor<T> input,
        Tensor<T> weight, Tensor<T>? bias)
    {
        if (history is null) throw new ArgumentNullException(nameof(history));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (history.Rank != 3) throw new ArgumentException("history must be [K - 1, batch, channels].", nameof(history));
        if (!history.IsContiguous) throw new ArgumentException("history is updated in place and must be contiguous.", nameof(history));
        int taps = history.Shape[0], batch = history.Shape[1], channels = history.Shape[2];
        if (weight.Rank != 2 || weight.Shape[0] != channels || weight.Shape[1] != taps + 1)
            throw new ArgumentException($"weight must be [channels, K] = [{channels}, {taps + 1}].", nameof(weight));
        if (input.Length != batch * channels) throw new ArgumentException($"input must hold batch x channels = {batch * channels} values.", nameof(input));
        if (bias is not null && bias.Length != channels) throw new ArgumentException($"bias must hold {channels} values.", nameof(bias));
        return (taps, batch, channels);
    }
}
