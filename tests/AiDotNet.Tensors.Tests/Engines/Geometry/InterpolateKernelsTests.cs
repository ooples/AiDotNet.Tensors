using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Geometry;

/// <summary>
/// Nearest and linear-family Interpolate run over per-axis source tables in parallel over planes (InterpolateKernels).
/// They replaced per-element kernels that recomputed every coordinate through a division and a floor and read through
/// INumericOperations on one thread. The new kernels keep the arithmetic and its order, so forward and backward must
/// equal the per-element reference bit for bit, for every spatial rank, up- and downsampling, both corner conventions,
/// float and double.
/// </summary>
public class InterpolateKernelsTests
{
    private static readonly CpuEngine Engine = new();

    public static TheoryData<int[], int[], InterpolateMode, bool> Cases() => new()
    {
        { new[] { 2, 3, 7 }, new[] { 15 }, InterpolateMode.Nearest, false },
        { new[] { 2, 3, 7 }, new[] { 4 }, InterpolateMode.Linear, false },
        { new[] { 2, 3, 7 }, new[] { 13 }, InterpolateMode.Linear, true },
        { new[] { 2, 5, 9, 13 }, new[] { 18, 26 }, InterpolateMode.Nearest, false },
        { new[] { 1, 4, 25, 38 }, new[] { 13, 19 }, InterpolateMode.Nearest, false },
        { new[] { 2, 5, 9, 13 }, new[] { 20, 7 }, InterpolateMode.Bilinear, false },
        { new[] { 2, 5, 9, 13 }, new[] { 20, 7 }, InterpolateMode.Bilinear, true },
        { new[] { 1, 2, 1, 6 }, new[] { 1, 11 }, InterpolateMode.Bilinear, false },
        { new[] { 1, 3, 4, 5, 6 }, new[] { 7, 3, 9 }, InterpolateMode.Nearest, false },
        { new[] { 1, 3, 4, 5, 6 }, new[] { 7, 3, 9 }, InterpolateMode.Trilinear, false },
        { new[] { 1, 3, 4, 5, 6 }, new[] { 2, 8, 6 }, InterpolateMode.Trilinear, true },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void Float_ForwardAndBackward_EqualThePerElementReference(int[] shape, int[] sizes, InterpolateMode mode, bool alignCorners)
    {
        var random = new Random(shape.Length * 31 + sizes[0]);
        var input = new Tensor<float>(shape);
        for (int i = 0; i < input.Length; i++) input[i] = (float)(random.NextDouble() * 2 - 1);
        var output = Engine.Interpolate(input, sizes, mode, alignCorners);

        var gradOutput = new Tensor<float>(output.Shape.ToArray());
        for (int i = 0; i < gradOutput.Length; i++) gradOutput[i] = random.Next(5) == 0 ? 0f : (float)(random.NextDouble() * 2 - 1);
        var gradInput = InterpolateKernels.TryBackward(gradOutput, input, mode, alignCorners)!;

        var (expected, expectedGrad) = Reference(Read(input), shape, Read(gradOutput), output.Shape.ToArray(), mode, alignCorners);
        for (int i = 0; i < expected.Length; i++) Assert.Equal((float)expected[i], output[i]);
        for (int i = 0; i < expectedGrad.Length; i++) Assert.Equal(expectedGrad[i], gradInput[i]);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void Double_Forward_EqualsThePerElementReference(int[] shape, int[] sizes, InterpolateMode mode, bool alignCorners)
    {
        var random = new Random(shape.Length * 17 + sizes[0]);
        var input = new Tensor<double>(shape);
        for (int i = 0; i < input.Length; i++) input[i] = random.NextDouble() * 2 - 1;
        var output = Engine.Interpolate(input, sizes, mode, alignCorners);

        var values = new double[input.Length];
        for (int i = 0; i < values.Length; i++) values[i] = input[i];
        var (expected, _) = Reference(values, shape, new double[output.Length], output.Shape.ToArray(), mode, alignCorners);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], output[i]);
    }

    private static double[] Read(Tensor<float> tensor)
    {
        var values = new double[tensor.Length];
        for (int i = 0; i < values.Length; i++) values[i] = tensor[i];
        return values;
    }

    /// <summary>The per-element kernels as they were: forward in double, backward accumulating in float.</summary>
    private static (double[] Forward, float[] Backward) Reference(double[] src, int[] srcShape, double[] gout, int[] dstShape,
        InterpolateMode mode, bool alignCorners)
    {
        int spatial = srcShape.Length - 2;
        int planes = srcShape[0] * srcShape[1];
        var srcDims = new int[spatial];
        var dstDims = new int[spatial];
        for (int i = 0; i < spatial; i++) { srcDims[i] = srcShape[2 + i]; dstDims[i] = dstShape[2 + i]; }
        var stride = new int[spatial];
        stride[spatial - 1] = 1;
        for (int i = spatial - 2; i >= 0; i--) stride[i] = stride[i + 1] * srcDims[i + 1];
        int srcPlane = 1, dstPlane = 1;
        for (int i = 0; i < spatial; i++) { srcPlane *= srcDims[i]; dstPlane *= dstDims[i]; }
        var forward = new double[planes * dstPlane];
        var backward = new float[planes * srcPlane];
        int corners = 1 << spatial;
        var idx = new int[spatial];

        for (int p = 0; p < planes; p++)
        {
            for (int k = 0; k < dstPlane; k++)
            {
                int tmp = k;
                for (int i = spatial - 1; i >= 0; i--) { idx[i] = tmp % dstDims[i]; tmp /= dstDims[i]; }
                var lo = new int[spatial];
                var hi = new int[spatial];
                var frac = new double[spatial];
                for (int i = 0; i < spatial; i++)
                {
                    if (mode == InterpolateMode.Nearest)
                    {
                        double s = dstDims[i] > 1 ? (double)idx[i] * srcDims[i] / dstDims[i] : 0.0;
                        int si = Math.Min((int)Math.Floor(s), srcDims[i] - 1);
                        lo[i] = hi[i] = si;
                        continue;
                    }

                    double c = dstDims[i] <= 1 ? 0.0
                        : alignCorners ? (double)idx[i] * (srcDims[i] - 1) / (dstDims[i] - 1)
                        : ((idx[i] + 0.5) * srcDims[i] / dstDims[i]) - 0.5;
                    int l = Math.Max((int)Math.Floor(c), 0);
                    int h = l + 1;
                    if (h >= srcDims[i]) { h = srcDims[i] - 1; l = Math.Min(l, h); }
                    lo[i] = l; hi[i] = h; frac[i] = Math.Min(Math.Max(c - l, 0), 1);
                }

                double g = gout[(p * dstPlane) + k];
                if (mode == InterpolateMode.Nearest)
                {
                    int off = 0;
                    for (int i = 0; i < spatial; i++) off += lo[i] * stride[i];
                    forward[(p * dstPlane) + k] = src[(p * srcPlane) + off];
                    if (g != 0) backward[(p * srcPlane) + off] += (float)g;
                    continue;
                }

                double acc = 0;
                for (int corner = 0; corner < corners; corner++)
                {
                    int off = 0;
                    double w = 1.0;
                    for (int i = 0; i < spatial; i++)
                    {
                        bool takeHi = ((corner >> i) & 1) == 1;
                        off += (takeHi ? hi[i] : lo[i]) * stride[i];
                        w *= takeHi ? frac[i] : (1.0 - frac[i]);
                    }

                    acc += w * src[(p * srcPlane) + off];
                    if (g != 0) backward[(p * srcPlane) + off] += (float)(w * g);
                }

                forward[(p * dstPlane) + k] = acc;
            }
        }

        return (forward, backward);
    }
}
