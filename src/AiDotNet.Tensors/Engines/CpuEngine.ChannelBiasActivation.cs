using System;
using System.Runtime.CompilerServices;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// <c>output[n, c, ...] = act(input[n, c, ...] + bias[c])</c> with <c>act</c> = ReLU or identity, in one pass
    /// parallel over the (n, c) planes. This is the epilogue of a convolution followed by a channel-bias add and a
    /// ReLU: a compiled training plan runs it in place of the separate <c>TensorChannelBiasAdd</c> and <c>ReLU</c>
    /// steps, which each made a full pass (the bias add on one thread, through a temporary it then copied).
    /// </summary>
    /// <remarks>
    /// Bit-identical to the two separate ops: the sum is <c>input + bias</c> (the channel-bias add's operand order),
    /// and ReLU is <c>max(x, +0)</c> exactly as the ReLU kernel computes it, so a NaN or -0 sum becomes +0.
    /// </remarks>
    internal unsafe void ChannelBiasActivationInto(Tensor<float> output, Tensor<float> input, Tensor<float> bias, bool relu)
    {
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (bias == null) throw new ArgumentNullException(nameof(bias));
        if (input.Rank < 2 || bias.Rank != 1 || bias._shape[0] != input._shape[1] || output.Length != input.Length)
            throw new ArgumentException("ChannelBiasActivationInto needs input [N, C, ...], bias [C] and an output of the input's size.");
        int batch = input._shape[0], channels = input._shape[1];
        int spatial = input.Length / Math.Max(1, batch * channels);
        var src = input.GetCpuBackingForStridedRead(out int srcOff);
        var b = bias.GetCpuBackingForStridedRead(out int bOff);
        var dst = output.GetCpuBackingForContiguousWrite(out int dstOff);
        if (src is null || b is null || dst is null || !input.IsContiguous || !bias.IsContiguous)
            throw new ArgumentException("ChannelBiasActivationInto needs contiguous CPU tensors.");
        int planes = batch * channels;
        if (planes == 0 || spatial == 0) return;
        CpuParallelSettings.ParallelForOrSerial(0, planes, (long)planes * spatial, [MethodImpl(Compatibility.MethodImplHelper.Hot)] (int plane) =>
        {
            float bc = b[bOff + plane % channels];
            fixed (float* ps = &src[srcOff + plane * spatial])
            fixed (float* pd = &dst[dstOff + plane * spatial])
            {
                int i = 0;
#if NET5_0_OR_GREATER
                if (Avx.IsSupported)
                {
                    var vb = Vector256.Create(bc);
                    var zero = Vector256<float>.Zero;
                    for (; i + 8 <= spatial; i += 8)
                    {
                        var v = Avx.Add(Avx.LoadVector256(ps + i), vb);
                        Avx.Store(pd + i, relu ? Avx.Max(v, zero) : v);
                    }
                }
#endif
                for (; i < spatial; i++)
                {
                    float v = ps[i] + bc;
                    pd[i] = relu ? (v > 0 ? v : 0) : v;
                }
            }
        });
    }

    /// <summary>
    /// <c>output = act(conv2d(input, kernel) + bias[c])</c>, <c>act</c> = ReLU or identity: the forward of a compiled
    /// Conv2D -> TensorChannelBiasAdd [-> ReLU] chain, which needs only the activation (its backward reads the
    /// activation, never the raw convolution). When the float convolution runs on the SIMD 3x3 route, the bias and
    /// ReLU are applied as each output chunk is stored, so the convolution output is never written and re-read;
    /// otherwise the convolution is written into <paramref name="output"/> and the epilogue runs over it in place.
    /// </summary>
    /// <remarks>
    /// Bit-identical to <c>Conv2DInto</c> into a separate buffer followed by <see cref="ChannelBiasActivationInto"/>:
    /// the convolution takes the same route and summation either way, and the fused store applies
    /// <c>sum + bias</c> then <c>max(x, +0)</c> exactly as the separate epilogue does.
    /// </remarks>
    internal void Conv2DBiasActivationInto(
        Tensor<float> output, Tensor<float> input, Tensor<float> kernel, Tensor<float> bias, bool relu,
        int[] stride, int[] padding, int[] dilation)
    {
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        if (bias == null) throw new ArgumentNullException(nameof(bias));
        if (stride == null || stride.Length != 2 || padding == null || padding.Length != 2
            || dilation == null || dilation.Length != 2)
            throw new ArgumentException("Conv2DBiasActivationInto needs two-element stride, padding and dilation.");
        if (input.Rank != 4 || kernel.Rank != 4 || output.Rank != 4 || bias.Rank != 1
            || kernel._shape[1] != input._shape[1] || bias._shape[0] != kernel._shape[0]
            || output._shape[0] != input._shape[0] || output._shape[1] != kernel._shape[0])
            throw new ArgumentException("Conv2DBiasActivationInto needs input [N, C, H, W], kernel [O, C, kH, kW], bias [O] and output [N, O, oH, oW].");

#if !NET471
        int strideH = stride[0], strideW = stride[1], padH = padding[0], padW = padding[1];
        int dilationH = dilation[0], dilationW = dilation[1];
        int kernelHeight = kernel._shape[2], kernelWidth = kernel._shape[3];
        int outputHeight = (input._shape[2] + 2 * padH - (dilationH * (kernelHeight - 1) + 1)) / strideH + 1;
        int outputWidth = (input._shape[3] + 2 * padW - (dilationW * (kernelWidth - 1) + 1)) / strideW + 1;
        // The route Conv2DInto takes for a plain NCHW float input (Conv2DIntoImpl -> DispatchFloatConv2D ->
        // Conv2DWithIm2ColFloat), with the epilogue handed to it; any other layout or geometry keeps the two-pass form.
        bool adaptiveRoute = input.IsContiguous && kernel.IsContiguous && output.IsContiguous && bias.IsContiguous
            && input.Layout == LinearAlgebra.TensorLayout.Nchw
            && output._shape[2] == outputHeight && output._shape[3] == outputWidth
            && ShouldUseAdaptiveFloatConv2D(input.Layout, strideH, strideW, padH, padW, dilationH, dilationW);
        if (adaptiveRoute)
        {
            // False: the chosen strategy wrote the plain convolution, so the epilogue still has to run below.
            if (Conv2DWithIm2ColFloat(input, kernel, output,
                    input._shape[0], input._shape[1], input._shape[2], input._shape[3],
                    kernel._shape[0], kernelHeight, kernelWidth, strideH, padH, dilationH,
                    outputHeight, outputWidth, bias, relu))
                return;
        }
        else
        {
            Conv2DInto(output, input, kernel, stride, padding, dilation);
        }
#else
        Conv2DInto(output, input, kernel, stride, padding, dilation);
#endif
        ChannelBiasActivationInto(output, output, bias, relu);
    }

    /// <summary>
    /// Backward of <see cref="ChannelBiasActivationInto"/>: from the output gradient <paramref name="gradOutput"/>
    /// and the forward output <paramref name="output"/>, writes the pre-activation gradient
    /// <c>gradInput = relu ? (output &gt; 0 ? gradOutput : 0) : gradOutput</c> and the bias gradient
    /// <c>gradBias[c] = Σ_n Σ_s gradInput[n, c, s]</c> in one pass over each channel.
    /// </summary>
    /// <remarks>
    /// Bit-identical to the separate steps it replaces in a compiled plan: the ReLU backward mask (strict
    /// greater-than on the output, +0 for inactive lanes), the channel-bias add's identity gradient landing on a
    /// zeroed buffer (<c>0 + g</c>, so -0 becomes +0), and the bias reduction's order (one running float sum per
    /// channel, batch outer and spatial inner, starting from +0). <paramref name="accumulateBias"/> adds that sum to
    /// the existing bias gradient, as gradient accumulation into a shared bias does; otherwise it is stored.
    /// Channels are independent, so the result does not depend on the thread count.
    /// </remarks>
    internal unsafe void ChannelBiasActivationBackwardInto(
        Tensor<float> gradInput, Tensor<float> gradBias, Tensor<float> gradOutput, Tensor<float> output,
        bool relu, bool accumulateBias)
    {
        if (gradInput == null) throw new ArgumentNullException(nameof(gradInput));
        if (gradBias == null) throw new ArgumentNullException(nameof(gradBias));
        if (gradOutput == null) throw new ArgumentNullException(nameof(gradOutput));
        if (output == null) throw new ArgumentNullException(nameof(output));
        if (gradOutput.Rank < 2 || gradBias.Length != gradOutput._shape[1]
            || gradInput.Length != gradOutput.Length || output.Length != gradOutput.Length)
            throw new ArgumentException("ChannelBiasActivationBackwardInto needs matching [N, C, ...] tensors and a [C] bias gradient.");
        int batch = gradOutput._shape[0], channels = gradOutput._shape[1];
        int spatial = gradOutput.Length / Math.Max(1, batch * channels);
        var g = gradOutput.GetCpuBackingForStridedRead(out int gOff);
        var y = output.GetCpuBackingForStridedRead(out int yOff);
        var dz = gradInput.GetCpuBackingForContiguousWrite(out int dzOff);
        var db = gradBias.GetCpuBackingForContiguousWrite(out int dbOff);
        if (g is null || y is null || dz is null || db is null || !gradOutput.IsContiguous || !output.IsContiguous)
            throw new ArgumentException("ChannelBiasActivationBackwardInto needs contiguous CPU tensors.");
        if (channels == 0) return;
        if (batch == 0 || spatial == 0)
        {
            // No elements: the bias gradient is the empty sum.
            for (int c = 0; c < channels; c++) db[dbOff + c] = accumulateBias ? db[dbOff + c] + 0f : 0f;
            return;
        }
        CpuParallelSettings.ParallelForOrSerial(0, channels, (long)batch * channels * spatial, [MethodImpl(Compatibility.MethodImplHelper.Hot)] (int c) =>
        {
            float acc = 0f;
            for (int n = 0; n < batch; n++)
            {
                int off = (n * channels + c) * spatial;
                fixed (float* pg = &g[gOff + off])
                fixed (float* py = &y[yOff + off])
                fixed (float* pz = &dz[dzOff + off])
                {
                    int i = 0;
#if NET5_0_OR_GREATER
                    if (Avx.IsSupported)
                    {
                        var zero = Vector256<float>.Zero;
                        for (; i + 8 <= spatial; i += 8)
                        {
                            var v = Avx.LoadVector256(pg + i);
                            if (relu)
                                v = Avx.And(v, Avx.Compare(Avx.LoadVector256(py + i), zero,
                                    FloatComparisonMode.OrderedGreaterThanSignaling));
                            v = Avx.Add(zero, v);
                            Avx.Store(pz + i, v);
                            acc += v.GetElement(0); acc += v.GetElement(1); acc += v.GetElement(2); acc += v.GetElement(3);
                            acc += v.GetElement(4); acc += v.GetElement(5); acc += v.GetElement(6); acc += v.GetElement(7);
                        }
                    }
#endif
                    for (; i < spatial; i++)
                    {
                        float v = 0f + (relu ? (py[i] > 0 ? pg[i] : 0f) : pg[i]);
                        pz[i] = v;
                        acc += v;
                    }
                }
            }
            db[dbOff + c] = accumulateBias ? db[dbOff + c] + acc : acc;
        });
    }
}
