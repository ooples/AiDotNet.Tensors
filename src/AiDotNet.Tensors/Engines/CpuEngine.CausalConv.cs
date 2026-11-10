using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Channels-last causal depthwise 1-D convolution, the short convolution of H3/Mamba/GLA-style sequence mixers:
    /// <code>
    ///   y[..., t, c] = sum_{j &lt; K} kernel[j, c] * input[..., t - j, c]     (input[..., t &lt; 0, c] = 0)
    /// </code>
    /// over an input <c>[..., seqLen, channels]</c> (every leading axis is an independent sequence) and a kernel
    /// <c>[K, channels]</c> whose tap <c>j</c> reads <c>j</c> steps back. One tape node whose backward needs only the
    /// input and kernel, so a K-tap convolution keeps one output-sized activation alive for backward instead of the
    /// K shifted copies, K products and K - 1 partial sums a shift-multiply-add composition records.
    /// </summary>
    /// <param name="input">The sequences, <c>[..., seqLen, channels]</c>, rank 2 or more.</param>
    /// <param name="kernel">The taps, <c>[K, channels]</c>; row <c>j</c> weights the input <c>j</c> steps back.</param>
    /// <returns>The convolved sequences, the input's shape.</returns>
    public virtual Tensor<T> CausalDepthwiseConv1D<T>(Tensor<T> input, Tensor<T> kernel)
    {
        var (rows, seqLen, channels, taps) = CausalConvDims(input, kernel);
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var capturedInput = input;
            var capturedKernel = kernel;
            return scope.RecordBinary(LazyNodeType.Custom, "CausalDepthwiseConv1D", input, kernel, (int[])input._shape.Clone(),
                (eng, result) =>
                {
                    var conv = eng as CpuEngine ?? new CpuEngine();
                    DirectGpuTensorEngine.CopyResultInto(eng, conv.CausalDepthwiseConv1D(capturedInput, capturedKernel), result);
                },
                CausalDepthwiseConv1DBackward<T>);
        }

        var output = new Tensor<T>(input._shape);
        if (output.Length > 0)
        {
            var x = HostArray(input, out int xOff);
            var w = HostArray(kernel, out int wOff);
            CausalConvForwardHost(x, xOff, w, wOff, output.GetDataArray(), rows, seqLen, channels, taps);
        }
        DifferentiableOps.RecordBinary("CausalDepthwiseConv1D", output, input, kernel, CausalDepthwiseConv1DBackward<T>);
        return output;
    }

    /// <summary>Gradient of <see cref="CausalDepthwiseConv1D{T}"/> with respect to its input:
    /// <c>gradInput[..., t, c] = sum_j kernel[j, c] * gradOutput[..., t + j, c]</c>.</summary>
    public virtual Tensor<T> CausalDepthwiseConv1DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel)
    {
        var (rows, seqLen, channels, taps) = CausalConvDims(gradOutput, kernel);
        var gradInput = new Tensor<T>(gradOutput._shape);
        if (gradInput.Length > 0)
        {
            var g = HostArray(gradOutput, out int gOff);
            var w = HostArray(kernel, out int wOff);
            CausalConvBackwardInputHost(g, gOff, w, wOff, gradInput.GetDataArray(), rows, seqLen, channels, taps);
        }
        return gradInput;
    }

    /// <summary>Gradient of <see cref="CausalDepthwiseConv1D{T}"/> with respect to its kernel:
    /// <c>gradKernel[j, c] = sum_{..., t} gradOutput[..., t, c] * input[..., t - j, c]</c>, summed in a fixed order.</summary>
    public virtual Tensor<T> CausalDepthwiseConv1DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int taps)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (taps < 1) throw new ArgumentOutOfRangeException(nameof(taps), "A causal convolution needs at least one tap.");
        if (!ShapesEqual(gradOutput._shape, input._shape))
            throw new ArgumentException("CausalDepthwiseConv1D gradient and input shapes differ.", nameof(gradOutput));
        if (input.Rank < 2) throw new ArgumentException("CausalDepthwiseConv1D input must be [..., seqLen, channels].", nameof(input));
        int channels = input._shape[input.Rank - 1], seqLen = input._shape[input.Rank - 2];
        int rows = seqLen * channels == 0 ? 0 : input.Length / (seqLen * channels);
        var gradKernel = new Tensor<T>(new[] { taps, channels });
        if (input.Length > 0)
        {
            var g = HostArray(gradOutput, out int gOff);
            var x = HostArray(input, out int xOff);
            CausalConvBackwardKernelHost(g, gOff, x, xOff, gradKernel.GetDataArray(), rows, seqLen, channels, taps);
        }
        return gradKernel;
    }

    internal static void CausalDepthwiseConv1DBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // A non-CpuEngine IEngine has no causal-convolution kernels; the managed CPU ones compute the same sums.
        var conv = engine as CpuEngine ?? new CpuEngine();
        if (DifferentiableOps.IsGradientRequired(inputs[0]))
            DifferentiableOps.AccumulateGrad(grads, inputs[0], conv.CausalDepthwiseConv1DBackwardInput(gradOutput, inputs[1]), engine);
        if (DifferentiableOps.IsGradientRequired(inputs[1]))
            DifferentiableOps.AccumulateGrad(grads, inputs[1],
                conv.CausalDepthwiseConv1DBackwardKernel(gradOutput, inputs[0], inputs[1]._shape[0]), engine);
    }

    private static (int Rows, int SeqLen, int Channels, int Taps) CausalConvDims<T>(Tensor<T> input, Tensor<T> kernel)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (kernel is null) throw new ArgumentNullException(nameof(kernel));
        if (input.Rank < 2)
            throw new ArgumentException($"CausalDepthwiseConv1D input must be [..., seqLen, channels]; got rank {input.Rank}.", nameof(input));
        if (kernel.Rank != 2)
            throw new ArgumentException($"CausalDepthwiseConv1D kernel must be [taps, channels]; got rank {kernel.Rank}.", nameof(kernel));
        int channels = input._shape[input.Rank - 1], seqLen = input._shape[input.Rank - 2];
        if (kernel._shape[1] != channels)
            throw new ArgumentException($"CausalDepthwiseConv1D kernel has {kernel._shape[1]} channels; the input has {channels}.", nameof(kernel));
        if (kernel._shape[0] < 1)
            throw new ArgumentException("CausalDepthwiseConv1D kernel needs at least one tap.", nameof(kernel));
        int rows = seqLen * channels == 0 ? 0 : input.Length / (seqLen * channels);
        return (rows, seqLen, channels, kernel._shape[0]);
    }

    // The tensor's elements in row-major order starting at `offset`: GetDataArray hands back either the live backing
    // array (where a view starts at its storage offset) or a logical copy (starting at 0).
    private static T[] HostArray<T>(Tensor<T> tensor, out int offset)
    {
        var c = tensor.IsContiguous ? tensor : tensor.Contiguous();
        var data = c.GetDataArray();
        offset = data.Length == c.Length ? 0 : c._storageOffset;
        return data;
    }

    // Each (row, step) writes its own output row, so rows of the flattened [rows * seqLen, channels] grid split freely.
    private static void CausalConvForwardHost<T>(T[] x, int xOff, T[] w, int wOff, T[] y,
        int rows, int seqLen, int channels, int taps)
    {
        if (x is float[] xF && w is float[] wF && y is float[] yF)
            CausalConvForwardHostFloat(xF, xOff, wF, wOff, yF, rows, seqLen, channels, taps);
        else if (x is double[] xD && w is double[] wD && y is double[] yD)
            CausalConvForwardHostDouble(xD, xOff, wD, wOff, yD, rows, seqLen, channels, taps);
        else
            CausalConvForwardHostGeneric(x, xOff, w, wOff, y, rows, seqLen, channels, taps);
    }

    // Channel run per kernel-gradient work item: a multiple of every SIMD width, small enough to spread a few taps.
    private const int CausalConvChannelBlock = 256;

    private static void CausalConvForwardHostFloat(float[] x, int xOff, float[] w, int wOff, float[] y,
        int rows, int seqLen, int channels, int taps)
    {
        // Channels innermost: each tap adds one contiguous weight row times one input row into the output row. Every
        // element still sums its taps in ascending order from zero, so the result matches the scalar loop exactly.
        CpuParallelSettings.ParallelForOrSerial(0, rows * seqLen, (long)rows * seqLen * channels * taps, rt =>
        {
            int t = rt % seqLen, outBase = rt * channels;
            int maxJ = Math.Min(taps - 1, t);
            Array.Clear(y, outBase, channels);
            for (int j = 0; j <= maxJ; j++)
                MultiplyAccumulate(y, outBase, w, wOff + j * channels, x, xOff + outBase - j * channels, channels);
        }, deterministicSafe: true);
    }

    private static void CausalConvBackwardInputHostFloat(float[] g, int gOff, float[] w, int wOff, float[] dx,
        int rows, int seqLen, int channels, int taps)
    {
        CpuParallelSettings.ParallelForOrSerial(0, rows * seqLen, (long)rows * seqLen * channels * taps, rt =>
        {
            int t = rt % seqLen, rowBase = rt * channels;
            int maxJ = Math.Min(taps - 1, seqLen - 1 - t);
            Array.Clear(dx, rowBase, channels);
            for (int j = 0; j <= maxJ; j++)
                MultiplyAccumulate(dx, rowBase, w, wOff + j * channels, g, gOff + rowBase + j * channels, channels);
        }, deterministicSafe: true);
    }

    // One (tap, channel block) per work item; each element sums rows then steps in ascending order: deterministic.
    private static void CausalConvBackwardKernelHostFloat(float[] g, int gOff, float[] x, int xOff, float[] dw,
        int rows, int seqLen, int channels, int taps)
    {
        int blocks = (channels + CausalConvChannelBlock - 1) / CausalConvChannelBlock;
        CpuParallelSettings.ParallelForOrSerial(0, taps * blocks, (long)rows * seqLen * channels * taps, item =>
        {
            int j = item / blocks, c0 = (item % blocks) * CausalConvChannelBlock;
            int len = Math.Min(CausalConvChannelBlock, channels - c0), dst = j * channels + c0;
            Array.Clear(dw, dst, len);
            for (int r = 0; r < rows; r++)
            {
                int rowBase = r * seqLen * channels + c0;
                for (int t = j; t < seqLen; t++)
                    MultiplyAccumulate(dw, dst, g, gOff + rowBase + t * channels, x, xOff + rowBase + (t - j) * channels, len);
            }
        }, deterministicSafe: true);
    }

    /// <summary>acc[accOff..+n] += a[aOff..+n] * b[bOff..+n], a multiply then an add per element (no fused rounding).</summary>
    private static void MultiplyAccumulate(float[] acc, int accOff, float[] a, int aOff, float[] b, int bOff, int n)
    {
        int width = System.Numerics.Vector<float>.Count, i = 0;
        for (; i <= n - width; i += width)
            (new System.Numerics.Vector<float>(acc, accOff + i)
                + new System.Numerics.Vector<float>(a, aOff + i) * new System.Numerics.Vector<float>(b, bOff + i)).CopyTo(acc, accOff + i);
        for (; i < n; i++) acc[accOff + i] += a[aOff + i] * b[bOff + i];
    }

    private static void CausalConvForwardHostDouble(double[] x, int xOff, double[] w, int wOff, double[] y,
        int rows, int seqLen, int channels, int taps)
    {
        CpuParallelSettings.ParallelForOrSerial(0, rows * seqLen, (long)rows * seqLen * channels * taps, rt =>
        {
            int t = rt % seqLen, outBase = rt * channels;
            int maxJ = Math.Min(taps - 1, t);
            Array.Clear(y, outBase, channels);
            for (int j = 0; j <= maxJ; j++)
                MultiplyAccumulate(y, outBase, w, wOff + j * channels, x, xOff + outBase - j * channels, channels);
        }, deterministicSafe: true);
    }

    private static void CausalConvBackwardInputHostDouble(double[] g, int gOff, double[] w, int wOff, double[] dx,
        int rows, int seqLen, int channels, int taps)
    {
        CpuParallelSettings.ParallelForOrSerial(0, rows * seqLen, (long)rows * seqLen * channels * taps, rt =>
        {
            int t = rt % seqLen, rowBase = rt * channels;
            int maxJ = Math.Min(taps - 1, seqLen - 1 - t);
            Array.Clear(dx, rowBase, channels);
            for (int j = 0; j <= maxJ; j++)
                MultiplyAccumulate(dx, rowBase, w, wOff + j * channels, g, gOff + rowBase + j * channels, channels);
        }, deterministicSafe: true);
    }

    private static void CausalConvBackwardKernelHostDouble(double[] g, int gOff, double[] x, int xOff, double[] dw,
        int rows, int seqLen, int channels, int taps)
    {
        int blocks = (channels + CausalConvChannelBlock - 1) / CausalConvChannelBlock;
        CpuParallelSettings.ParallelForOrSerial(0, taps * blocks, (long)rows * seqLen * channels * taps, item =>
        {
            int j = item / blocks, c0 = (item % blocks) * CausalConvChannelBlock;
            int len = Math.Min(CausalConvChannelBlock, channels - c0), dst = j * channels + c0;
            Array.Clear(dw, dst, len);
            for (int r = 0; r < rows; r++)
            {
                int rowBase = r * seqLen * channels + c0;
                for (int t = j; t < seqLen; t++)
                    MultiplyAccumulate(dw, dst, g, gOff + rowBase + t * channels, x, xOff + rowBase + (t - j) * channels, len);
            }
        }, deterministicSafe: true);
    }

    /// <summary>acc[accOff..+n] += a[aOff..+n] * b[bOff..+n], a multiply then an add per element (no fused rounding).</summary>
    private static void MultiplyAccumulate(double[] acc, int accOff, double[] a, int aOff, double[] b, int bOff, int n)
    {
        int width = System.Numerics.Vector<double>.Count, i = 0;
        for (; i <= n - width; i += width)
            (new System.Numerics.Vector<double>(acc, accOff + i)
                + new System.Numerics.Vector<double>(a, aOff + i) * new System.Numerics.Vector<double>(b, bOff + i)).CopyTo(acc, accOff + i);
        for (; i < n; i++) acc[accOff + i] += a[aOff + i] * b[bOff + i];
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; float and double take the typed loops above.")]
    private static void CausalConvForwardHostGeneric<T>(T[] x, int xOff, T[] w, int wOff, T[] y,
        int rows, int seqLen, int channels, int taps)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        CpuParallelSettings.ParallelForOrSerial(0, rows * seqLen, (long)rows * seqLen * channels * taps, rt =>
        {
            int t = rt % seqLen, outBase = rt * channels;
            int maxJ = Math.Min(taps - 1, t);
            for (int c = 0; c < channels; c++)
            {
                T acc = ops.Zero;
                for (int j = 0; j <= maxJ; j++)
                    acc = ops.Add(acc, ops.Multiply(w[wOff + j * channels + c], x[xOff + outBase - j * channels + c]));
                y[outBase + c] = acc;
            }
        }, deterministicSafe: true);
    }

    private static void CausalConvBackwardInputHost<T>(T[] g, int gOff, T[] w, int wOff, T[] dx,
        int rows, int seqLen, int channels, int taps)
    {
        if (g is float[] gF && w is float[] wF && dx is float[] dxF)
            CausalConvBackwardInputHostFloat(gF, gOff, wF, wOff, dxF, rows, seqLen, channels, taps);
        else if (g is double[] gD && w is double[] wD && dx is double[] dxD)
            CausalConvBackwardInputHostDouble(gD, gOff, wD, wOff, dxD, rows, seqLen, channels, taps);
        else
            CausalConvBackwardInputHostGeneric(g, gOff, w, wOff, dx, rows, seqLen, channels, taps);
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; float and double take the typed loops above.")]
    private static void CausalConvBackwardInputHostGeneric<T>(T[] g, int gOff, T[] w, int wOff, T[] dx,
        int rows, int seqLen, int channels, int taps)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        CpuParallelSettings.ParallelForOrSerial(0, rows * seqLen, (long)rows * seqLen * channels * taps, rt =>
        {
            int t = rt % seqLen, rowBase = rt * channels;
            int maxJ = Math.Min(taps - 1, seqLen - 1 - t);
            for (int c = 0; c < channels; c++)
            {
                T acc = ops.Zero;
                for (int j = 0; j <= maxJ; j++)
                    acc = ops.Add(acc, ops.Multiply(w[wOff + j * channels + c], g[gOff + rowBase + j * channels + c]));
                dx[rowBase + c] = acc;
            }
        }, deterministicSafe: true);
    }

    // One (tap, channel) cell per work item, summed over rows then steps in ascending order: deterministic.
    private static void CausalConvBackwardKernelHost<T>(T[] g, int gOff, T[] x, int xOff, T[] dw,
        int rows, int seqLen, int channels, int taps)
    {
        if (g is float[] gF && x is float[] xF && dw is float[] dwF)
            CausalConvBackwardKernelHostFloat(gF, gOff, xF, xOff, dwF, rows, seqLen, channels, taps);
        else if (g is double[] gD && x is double[] xD && dw is double[] dwD)
            CausalConvBackwardKernelHostDouble(gD, gOff, xD, xOff, dwD, rows, seqLen, channels, taps);
        else
            CausalConvBackwardKernelHostGeneric(g, gOff, x, xOff, dw, rows, seqLen, channels, taps);
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; float and double take the typed loops above.")]
    private static void CausalConvBackwardKernelHostGeneric<T>(T[] g, int gOff, T[] x, int xOff, T[] dw,
        int rows, int seqLen, int channels, int taps)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        CpuParallelSettings.ParallelForOrSerial(0, taps * channels, (long)rows * seqLen * channels * taps, jc =>
        {
            int j = jc / channels, c = jc % channels;
            T acc = ops.Zero;
            for (int r = 0; r < rows; r++)
            {
                int rowBase = r * seqLen * channels;
                for (int t = j; t < seqLen; t++)
                    acc = ops.Add(acc, ops.Multiply(g[gOff + rowBase + t * channels + c], x[xOff + rowBase + (t - j) * channels + c]));
            }
            dw[jc] = acc;
        }, deterministicSafe: true);
    }
}
