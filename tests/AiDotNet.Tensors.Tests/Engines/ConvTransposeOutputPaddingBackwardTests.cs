// Copyright (c) CrowdTrainer LLC. All rights reserved.
// A transposed convolution's output padding adds rows at the far edge of its output that the kernel taps write
// (PyTorch's output_padding / output_size). The engine backward ops take no output-padding argument and used to demand
// that the gradient's extent invert EXACTLY to the input's, so every output-padded transposed convolution's gradient was
// refused ("does not invert to a whole input extent"), on the tape and when called directly, and the GPU engine passed
// an output padding of 0 to its kernels. These tests pin the backward of output-padded forwards, 2D and 3D, on the CPU
// engine and the GPU engine, against a direct fp64 transposed convolution, and pin that extents no forward produces are
// still refused.

using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

[Collection("DirectGpuSerial")]
public class ConvTransposeOutputPaddingBackwardTests : IDisposable
{
    private readonly IEngine _prior = AiDotNetEngine.Current;

    public void Dispose() => AiDotNetEngine.Current = _prior;

    private static double[] Fill(int count, int salt)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = (((i * 37 + salt * 101) % 97) - 48) / 64.0;
        return values;
    }

    private static Tensor<T> Make<T>(int[] shape, double[] values)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return new Tensor<T>(values.Select(v => ops.FromDouble(v)).ToArray(), shape);
    }

    private static double[] Read<T>(Tensor<T> t)
    {
        var ops = AiDotNet.Tensors.Helpers.MathHelper.GetNumericOperations<T>();
        return t.ToArray().Select(v => ops.ToDouble(v)).ToArray();
    }

    // The geometry of one case: spatial extents of the input, kernel, stride, padding and output padding per axis.
    public sealed record Case(int Batch, int InChannels, int OutChannels, int[] In, int[] Kernel, int[] Stride, int[] Padding, int[] OutputPadding)
    {
        public int Rank => In.Length;
        public int[] Out => Enumerable.Range(0, Rank).Select(a => (In[a] - 1) * Stride[a] - 2 * Padding[a] + Kernel[a] + OutputPadding[a]).ToArray();
        public int[] InputShape => new[] { Batch, InChannels }.Concat(In).ToArray();
        public int[] KernelShape => new[] { InChannels, OutChannels }.Concat(Kernel).ToArray();
        public int[] OutputShape => new[] { Batch, OutChannels }.Concat(Out).ToArray();
    }

    // y[n, co, i·s − p + k] += x[n, ci, i]·w[ci, co, k] over every axis, and the matching gradients for an upstream dy.
    private static (double[] Y, double[] Dx, double[] Dw) Reference(Case c, double[] x, double[] w, double[] dy)
    {
        var y = new double[dy.Length];
        var dx = new double[x.Length];
        var dw = new double[w.Length];
        int rank = c.Rank;
        var inSpatial = c.In;
        var outSpatial = c.Out;
        int inPlane = inSpatial.Aggregate(1, (a, b) => a * b), outPlane = outSpatial.Aggregate(1, (a, b) => a * b);
        int kPlane = c.Kernel.Aggregate(1, (a, b) => a * b);
        var position = new int[rank];
        var tap = new int[rank];
        for (int n = 0; n < c.Batch; n++)
            for (int ci = 0; ci < c.InChannels; ci++)
                for (int ip = 0; ip < inPlane; ip++)
                {
                    for (int a = rank - 1, r = ip; a >= 0; a--) { position[a] = r % inSpatial[a]; r /= inSpatial[a]; }
                    for (int co = 0; co < c.OutChannels; co++)
                        for (int kp = 0; kp < kPlane; kp++)
                        {
                            for (int a = rank - 1, r = kp; a >= 0; a--) { tap[a] = r % c.Kernel[a]; r /= c.Kernel[a]; }
                            int op = 0;
                            bool inside = true;
                            for (int a = 0; a < rank; a++)
                            {
                                int o = position[a] * c.Stride[a] - c.Padding[a] + tap[a];
                                if (o < 0 || o >= outSpatial[a]) { inside = false; break; }
                                op = op * outSpatial[a] + o;
                            }
                            if (!inside) continue;
                            int xi = (n * c.InChannels + ci) * inPlane + ip, yi = (n * c.OutChannels + co) * outPlane + op;
                            int wi = (ci * c.OutChannels + co) * kPlane + kp;
                            y[yi] += x[xi] * w[wi];
                            dx[xi] += dy[yi] * w[wi];
                            dw[wi] += x[xi] * dy[yi];
                        }
                }
        return (y, dx, dw);
    }

    public static IEnumerable<object[]> Cases()
    {
        // Stride 2, padding 1, 3-tap kernels: an output padding of 1 adds a row a tap writes, as output_size does for
        // an even target. The mixed cases pad one axis and not the other.
        foreach (var op in new[] { new[] { 0, 0 }, new[] { 1, 1 }, new[] { 1, 0 }, new[] { 0, 1 } })
            yield return new object[] { new Case(2, 3, 4, new[] { 5, 4 }, new[] { 3, 3 }, new[] { 2, 2 }, new[] { 1, 1 }, op) };
        // Stride 3 allows output paddings 1 and 2.
        yield return new object[] { new Case(1, 2, 3, new[] { 4, 3 }, new[] { 3, 2 }, new[] { 3, 3 }, new[] { 1, 0 }, new[] { 2, 1 }) };
        // 3D, as VFIT's upSplit uses it: time unstrided, space strided by 2 with an output padding.
        foreach (var op in new[] { new[] { 0, 0, 0 }, new[] { 0, 1, 1 }, new[] { 0, 1, 0 } })
            yield return new object[] { new Case(2, 3, 2, new[] { 3, 4, 3 }, new[] { 3, 3, 3 }, new[] { 1, 2, 2 }, new[] { 1, 1, 1 }, op) };
    }

    private static Tensor<T> Forward<T>(IEngine engine, Case c, Tensor<T> x, Tensor<T> w)
        => c.Rank == 2 ? engine.ConvTranspose2D(x, w, c.Stride, c.Padding, c.OutputPadding) : engine.ConvTranspose3D(x, w, c.Stride, c.Padding, c.OutputPadding);

    private static void AssertClose(double[] expected, double[] actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double error = Math.Abs(expected[i] - actual[i]) / Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(error <= tolerance, $"{label}: element {i} expected {expected[i]}, got {actual[i]}");
        }
    }

    // The tape's gradients of Σ dy·y for both operands, and the direct backward calls on the same upstream gradient.
    private static void AssertBackward<T>(IEngine engine, Case c, double tolerance, string label)
    {
        AiDotNetEngine.Current = engine;
        double[] xv = Fill(c.InputShape.Aggregate(1, (a, b) => a * b), 1), wv = Fill(c.KernelShape.Aggregate(1, (a, b) => a * b), 2);
        double[] dyv = Fill(c.OutputShape.Aggregate(1, (a, b) => a * b), 3);
        var (y, dx, dw) = Reference(c, xv, wv, dyv);
        var x = Make<T>(c.InputShape, xv);
        var w = Make<T>(c.KernelShape, wv);
        var dy = Make<T>(c.OutputShape, dyv);

        Dictionary<Tensor<T>, Tensor<T>> grads;
        using (var tape = new GradientTape<T>())
        {
            var output = Forward(engine, c, x, w);
            Assert.Equal(c.OutputShape, output.Shape.ToArray());
            AssertClose(y, Read(output), tolerance, $"{label} forward");
            var loss = engine.ReduceSum(engine.TensorMultiply(output, dy), null);
            grads = tape.ComputeGradients(loss, new[] { x, w });
        }
        AssertClose(dx, Read(grads[x]), tolerance, $"{label} tape data gradient");
        AssertClose(dw, Read(grads[w]), tolerance, $"{label} tape weight gradient");

        var directDx = c.Rank == 2 ? engine.ConvTranspose2DBackwardInput(dy, w, c.InputShape, c.Stride, c.Padding)
            : engine.ConvTranspose3DBackwardInput(dy, w, c.InputShape, c.Stride, c.Padding);
        var directDw = c.Rank == 2 ? engine.ConvTranspose2DBackwardKernel(dy, x, c.KernelShape, c.Stride, c.Padding)
            : engine.ConvTranspose3DBackwardKernel(dy, x, c.KernelShape, c.Stride, c.Padding);
        AssertClose(dx, Read(directDx), tolerance, $"{label} direct data gradient");
        AssertClose(dw, Read(directDw), tolerance, $"{label} direct weight gradient");
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void CpuBackward_CoversTheOutputPadding_Double(Case c) => AssertBackward<double>(new CpuEngine(), c, 1e-12, "CPU double");

    [Theory]
    [MemberData(nameof(Cases))]
    public void CpuBackward_CoversTheOutputPadding_Float(Case c) => AssertBackward<float>(new CpuEngine(), c, 1e-5, "CPU float");

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void GpuBackward_CoversTheOutputPadding(Case c)
    {
        DirectGpuTensorEngine? gpu = null;
        try
        {
            gpu = new DirectGpuTensorEngine();
        }
        catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or InvalidOperationException or NotSupportedException)
        {
            Skip.If(true, $"The GPU engine did not construct: {ex.GetType().Name}.");
        }
        using (gpu)
        {
            Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "No GPU backend is available.");
            // Through IEngine: the GPU engine implements these explicitly, so a class-typed call would reach the CPU base.
            AssertBackward<float>(gpu, c, 1e-4, "GPU");
        }
    }

    // A gradient with another batch or other channels than the input and kernel describe is refused on every engine,
    // before a kernel sized from the input reads past the gradient's end.
    [SkippableTheory]
    [InlineData(false, 1, 3)]   // one sample where the input has two
    [InlineData(false, 2, 4)]   // four channels where the kernel writes three
    [InlineData(true, 1, 3)]
    [InlineData(true, 2, 4)]
    public void Backward_RefusesAMismatchedBatchOrChannels(bool onGpu, int batch, int channels)
    {
        IEngine engine = new CpuEngine();
        DirectGpuTensorEngine? gpu = null;
        if (onGpu)
        {
            try { gpu = new DirectGpuTensorEngine(); }
            catch (Exception ex) when (ex is DllNotFoundException or TypeInitializationException or InvalidOperationException or NotSupportedException)
            {
                Skip.If(true, $"The GPU engine did not construct: {ex.GetType().Name}.");
            }
            Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "No GPU backend is available.");
            engine = gpu;
        }
        using (gpu)
        {
            var kernel = Make<float>(new[] { 2, 3, 3, 3 }, Fill(54, 10));
            var input = Make<float>(new[] { 2, 2, 5, 4 }, Fill(80, 11));
            var dy = Make<float>(new[] { batch, channels, 10, 8 }, Fill(batch * channels * 80, 12));
            int[] s = { 2, 2 }, p = { 1, 1 };
            Assert.Throws<ArgumentException>(() => engine.ConvTranspose2DBackwardInput(dy, kernel, input.Shape.ToArray(), s, p));
            Assert.Throws<ArgumentException>(() => engine.ConvTranspose2DBackwardKernel(dy, input, kernel.Shape.ToArray(), s, p));
            var kernel3 = Make<float>(new[] { 2, 3, 3, 3, 3 }, Fill(162, 13));
            var input3 = Make<float>(new[] { 2, 2, 3, 5, 4 }, Fill(240, 14));
            var dy3 = Make<float>(new[] { batch, channels, 3, 10, 8 }, Fill(batch * channels * 240, 15));
            int[] s3 = { 1, 2, 2 }, p3 = { 1, 1, 1 };
            Assert.Throws<ArgumentException>(() => engine.ConvTranspose3DBackwardInput(dy3, kernel3, input3.Shape.ToArray(), s3, p3));
            Assert.Throws<ArgumentException>(() => engine.ConvTranspose3DBackwardKernel(dy3, input3, kernel3.Shape.ToArray(), s3, p3));
        }
    }

    // The relation is evaluated in 64 bits: with an input extent of 5 and a stride of 2^30, (5 − 1)·2^30 wraps to 0 in
    // 32 bits, so a gradient of extent 3 (= 0 + k) would have passed as output padding 0.
    [Fact]
    public void CpuBackward_RefusesAnExtentThatOnlyFitsAfterA32BitWrap()
    {
        IEngine engine = new CpuEngine();
        var kernel = Make<double>(new[] { 1, 1, 3, 3 }, Fill(9, 16));
        var dy = Make<double>(new[] { 1, 1, 3, 3 }, Fill(9, 17));
        int[] s = { 1 << 30, 1 << 30 }, p = { 0, 0 };
        Assert.Throws<ArgumentException>(() => engine.ConvTranspose2DBackwardInput(dy, kernel, new[] { 1, 1, 5, 5 }, s, p));
        var kernel3 = Make<double>(new[] { 1, 1, 3, 3, 3 }, Fill(27, 18));
        var dy3 = Make<double>(new[] { 1, 1, 3, 3, 3 }, Fill(27, 19));
        int[] s3 = { 1 << 30, 1 << 30, 1 << 30 }, p3 = { 0, 0, 0 };
        Assert.Throws<ArgumentException>(() => engine.ConvTranspose3DBackwardInput(dy3, kernel3, new[] { 1, 1, 5, 5, 5 }, s3, p3));
    }

    // Extents no forward produces are still refused: past the largest output padding (s − 1), or short of none.
    [Theory]
    [InlineData(2, 11)]   // (5 − 1)·2 − 2 + 3 = 9; output paddings 0 and 1 give 9 and 10, so 11 is out of reach
    [InlineData(2, 8)]
    [InlineData(3, 16)]   // (5 − 1)·3 − 2 + 3 = 13; 13 to 15 are reachable
    public void CpuBackward_RefusesAnExtentNoForwardProduces(int stride, int extent)
    {
        IEngine engine = new CpuEngine();
        var kernel = Make<double>(new[] { 2, 3, 3, 3 }, Fill(54, 4));
        var input = Make<double>(new[] { 1, 2, 5, 5 }, Fill(50, 5));
        var dy = Make<double>(new[] { 1, 3, extent, extent }, Fill(3 * extent * extent, 6));
        int[] s = { stride, stride }, p = { 1, 1 };
        Assert.Throws<ArgumentException>(() => engine.ConvTranspose2DBackwardInput(dy, kernel, input.Shape.ToArray(), s, p));
        Assert.Throws<ArgumentException>(() => engine.ConvTranspose2DBackwardKernel(dy, input, kernel.Shape.ToArray(), s, p));
        var kernel3 = Make<double>(new[] { 2, 3, 3, 3, 3 }, Fill(162, 7));
        var input3 = Make<double>(new[] { 1, 2, 5, 5, 5 }, Fill(250, 8));
        var dy3 = Make<double>(new[] { 1, 3, extent, extent, extent }, Fill(3 * extent * extent * extent, 9));
        int[] s3 = { stride, stride, stride }, p3 = { 1, 1, 1 };
        Assert.Throws<ArgumentException>(() => engine.ConvTranspose3DBackwardInput(dy3, kernel3, input3.Shape.ToArray(), s3, p3));
        Assert.Throws<ArgumentException>(() => engine.ConvTranspose3DBackwardKernel(dy3, input3, kernel3.Shape.ToArray(), s3, p3));
    }
}
