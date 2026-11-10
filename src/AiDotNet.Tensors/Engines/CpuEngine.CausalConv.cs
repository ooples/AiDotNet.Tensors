using System;
using System.Collections.Generic;
using System.Threading.Tasks;
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

    // The sums below run on float and double through typed arithmetic the JIT specializes per struct; other element
    // types go through INumericOperations one element at a time (GenericCausalMath).
    private static void CausalConvForwardHost<T>(T[] x, int xOff, T[] w, int wOff, T[] y,
        int rows, int seqLen, int channels, int taps)
    {
        if (x is float[] xf && w is float[] wf && y is float[] yf)
            CausalConvForward<float, FloatCausalMath>(xf, xOff, wf, wOff, yf, rows, seqLen, channels, taps, default);
        else if (x is double[] xd && w is double[] wd && y is double[] yd)
            CausalConvForward<double, DoubleCausalMath>(xd, xOff, wd, wOff, yd, rows, seqLen, channels, taps, default);
        else
            CausalConvForward(x, xOff, w, wOff, y, rows, seqLen, channels, taps, new GenericCausalMath<T>(MathHelper.GetNumericOperations<T>()));
    }

    private static void CausalConvBackwardInputHost<T>(T[] g, int gOff, T[] w, int wOff, T[] dx,
        int rows, int seqLen, int channels, int taps)
    {
        if (g is float[] gf && w is float[] wf && dx is float[] df)
            CausalConvBackwardInput<float, FloatCausalMath>(gf, gOff, wf, wOff, df, rows, seqLen, channels, taps, default);
        else if (g is double[] gd && w is double[] wd && dx is double[] dd)
            CausalConvBackwardInput<double, DoubleCausalMath>(gd, gOff, wd, wOff, dd, rows, seqLen, channels, taps, default);
        else
            CausalConvBackwardInput(g, gOff, w, wOff, dx, rows, seqLen, channels, taps, new GenericCausalMath<T>(MathHelper.GetNumericOperations<T>()));
    }

    private static void CausalConvBackwardKernelHost<T>(T[] g, int gOff, T[] x, int xOff, T[] dw,
        int rows, int seqLen, int channels, int taps)
    {
        if (g is float[] gf && x is float[] xf && dw is float[] df)
            CausalConvBackwardKernel<float, FloatCausalMath>(gf, gOff, xf, xOff, df, rows, seqLen, channels, taps, default);
        else if (g is double[] gd && x is double[] xd && dw is double[] dd)
            CausalConvBackwardKernel<double, DoubleCausalMath>(gd, gOff, xd, xOff, dd, rows, seqLen, channels, taps, default);
        else
            CausalConvBackwardKernel(g, gOff, x, xOff, dw, rows, seqLen, channels, taps, new GenericCausalMath<T>(MathHelper.GetNumericOperations<T>()));
    }

    // Each (row, step) writes its own output row, so rows of the flattened [rows * seqLen, channels] grid split freely.
    private static void CausalConvForward<TNum, TMath>(TNum[] x, int xOff, TNum[] w, int wOff, TNum[] y,
        int rows, int seqLen, int channels, int taps, TMath math)
        where TMath : struct, ICausalMath<TNum>
    {
        Parallel.For(0, rows * seqLen, rt =>
        {
            int t = rt % seqLen, outBase = rt * channels;
            int maxJ = Math.Min(taps - 1, t);
            for (int c = 0; c < channels; c++)
            {
                TNum acc = math.Zero;
                for (int j = 0; j <= maxJ; j++)
                    acc = math.MultiplyAdd(acc, w[wOff + j * channels + c], x[xOff + outBase - j * channels + c]);
                y[outBase + c] = acc;
            }
        });
    }

    private static void CausalConvBackwardInput<TNum, TMath>(TNum[] g, int gOff, TNum[] w, int wOff, TNum[] dx,
        int rows, int seqLen, int channels, int taps, TMath math)
        where TMath : struct, ICausalMath<TNum>
    {
        Parallel.For(0, rows * seqLen, rt =>
        {
            int t = rt % seqLen, rowBase = rt * channels;
            int maxJ = Math.Min(taps - 1, seqLen - 1 - t);
            for (int c = 0; c < channels; c++)
            {
                TNum acc = math.Zero;
                for (int j = 0; j <= maxJ; j++)
                    acc = math.MultiplyAdd(acc, w[wOff + j * channels + c], g[gOff + rowBase + j * channels + c]);
                dx[rowBase + c] = acc;
            }
        });
    }

    // One (tap, channel) cell per work item, summed over rows then steps in ascending order: deterministic.
    private static void CausalConvBackwardKernel<TNum, TMath>(TNum[] g, int gOff, TNum[] x, int xOff, TNum[] dw,
        int rows, int seqLen, int channels, int taps, TMath math)
        where TMath : struct, ICausalMath<TNum>
    {
        Parallel.For(0, taps * channels, jc =>
        {
            int j = jc / channels, c = jc % channels;
            TNum acc = math.Zero;
            for (int r = 0; r < rows; r++)
            {
                int rowBase = r * seqLen * channels;
                for (int t = j; t < seqLen; t++)
                    acc = math.MultiplyAdd(acc, g[gOff + rowBase + t * channels + c], x[xOff + rowBase + (t - j) * channels + c]);
            }
            dw[jc] = acc;
        });
    }

    // acc + a * b in the element type's own arithmetic (no fused multiply-add, matching INumericOperations).
    private interface ICausalMath<TNum>
    {
        TNum Zero { get; }
        TNum MultiplyAdd(TNum acc, TNum a, TNum b);
    }

    private readonly struct FloatCausalMath : ICausalMath<float>
    {
        public float Zero => 0f;
        public float MultiplyAdd(float acc, float a, float b) => acc + (a * b);
    }

    private readonly struct DoubleCausalMath : ICausalMath<double>
    {
        public double Zero => 0.0;
        public double MultiplyAdd(double acc, double a, double b) => acc + (a * b);
    }

    [ScalarKernel("Element types other than float and double have no typed arithmetic; the causal convolution falls back to INumericOperations per element.")]
    private readonly struct GenericCausalMath<TNum> : ICausalMath<TNum>
    {
        private readonly Interfaces.INumericOperations<TNum> _ops;

        public GenericCausalMath(Interfaces.INumericOperations<TNum> ops) => _ops = ops;

        public TNum Zero => _ops.Zero;
        public TNum MultiplyAdd(TNum acc, TNum a, TNum b) => _ops.Add(acc, _ops.Multiply(a, b));
    }
}
