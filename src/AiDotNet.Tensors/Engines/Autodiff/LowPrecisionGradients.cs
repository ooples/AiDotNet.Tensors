using System;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Float32 evaluation of the mean-scaled gradients (losses, mean, average pools) for Half and BFloat16.
/// </summary>
/// <remarks>
/// These gradients carry a 1/N or 2/N factor. In Half that factor is subnormal once N passes 16,384 and the count
/// itself is +Inf past 65,504, so scaling in the storage type rounds twice and loses several units in the last place
/// (or zeroes the gradient). PyTorch evaluates them in float32 opmath and rounds once; so do these helpers. The scale
/// is formed in double, every element is computed in float32/double from the widened operands, and the result is
/// rounded to T once. Float, double and the other element types never come here.
/// </remarks>
internal static class LowPrecisionGradients
{
    internal static bool Applies<T>() => CpuEngine.IsLowPrecisionFloat<T>();

    internal static float[] Widen<T>(Tensor<T> tensor)
    {
        var contiguous = tensor.IsContiguous ? tensor : tensor.Contiguous();
        return CpuEngine.WidenToFloat(contiguous.GetFlattenedData(), contiguous.Length);
    }

    internal static Tensor<T> Narrow<T>(float[] values, int[] shape)
        => new Tensor<T>(CpuEngine.NarrowFromFloat<T>(values), (int[])shape.Clone());

    /// <summary>The scalar upstream gradient of a mean-reduced loss, widened.</summary>
    internal static double Upstream<T>(Tensor<T> gradOutput) => Widen(gradOutput)[0];

    /// <summary><c>term(a[i], b[i]) * scale</c> for every element, rounded to T once, shaped like <paramref name="a"/>.</summary>
    internal static Tensor<T> Pairwise<T>(Tensor<T> a, Tensor<T> b, double scale, Func<double, double, double> term)
    {
        var x = Widen(a);
        var y = Widen(b);
        var result = new float[x.Length];
        for (int i = 0; i < x.Length; i++) result[i] = (float)(term(x[i], y[i]) * scale);
        return Narrow<T>(result, a._shape);
    }

    /// <summary>A tensor of <paramref name="shape"/> filled with <paramref name="value"/>, rounded to T once.</summary>
    internal static Tensor<T> Filled<T>(int[] shape, double value)
    {
        int length = 1;
        foreach (int d in shape) length *= d;
        var result = new float[length];
        float v = (float)value;
        for (int i = 0; i < length; i++) result[i] = v;
        return Narrow<T>(result, shape);
    }

    /// <summary>
    /// AvgPool1D backward: each output gradient divided by the kernel size and added onto its window, in float32.
    /// </summary>
    internal static Tensor<T> AvgPool1DBackward<T>(Tensor<T> gradOutput, int[] inShape, int kernelSize, int stride)
    {
        int planes = inShape[0] * inShape[1], length = inShape[2];
        int outLength = (length - kernelSize) / stride + 1;
        var g = Widen(gradOutput);
        var result = new float[planes * length];
        for (int p = 0; p < planes; p++)
            for (int o = 0; o < outLength; o++)
            {
                float share = (float)((double)g[p * outLength + o] / kernelSize);
                int start = p * length + o * stride;
                for (int k = 0; k < kernelSize; k++) result[start + k] += share;
            }
        return Narrow<T>(result, inShape);
    }

    /// <summary>AdaptiveAvgPool2D backward through the float32 kernel, rounded once.</summary>
    internal static Tensor<T> AdaptiveAvgPool2DBackward<T>(Tensor<T> gradOutput, int[] inShape, int outH, int outW)
    {
        int planes = inShape[0] * inShape[1], inH = inShape[2], inW = inShape[3];
        var g = Widen(gradOutput);
        var result = new float[planes * inH * inW];
        CpuEngine.AdaptiveAvgPool2DBackwardFloat(g, 0, result, 0, planes, inH, inW, outH, outW, accumulate: false);
        return Narrow<T>(result, inShape);
    }
}
