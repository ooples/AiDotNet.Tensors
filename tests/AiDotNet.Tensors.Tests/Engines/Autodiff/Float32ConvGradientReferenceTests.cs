using System;
using System.Collections.Generic;
#if NET8_0_OR_GREATER
using System.Runtime.Intrinsics.X86;
#endif
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Float32 convolution forward and gradients against exact double-precision references, on the encoder shapes
/// RAFT uses at a 64x64 input. 0.135.0 produced a WRONG float32 weight gradient on AVX-512 hardware: a RAFT
/// context-encoder 3x3 conv weight came out +7.72e-3 on an AMD EPYC Zen 5 (AVX-512) runner where the exact value
/// is -1.9116e-2 (finite differences agree to 0.000% in double, and float32 matches on non-AVX-512 CPUs). The
/// forward was identical, so the defect is in a backward path that only AVX-512 dispatch reaches.
/// </summary>
public sealed class Float32ConvGradientReferenceTests
{
    private readonly ITestOutputHelper _output;

    public Float32ConvGradientReferenceTests(ITestOutputHelper output) => _output = output;

    // Relative to the reference's largest magnitude: float32 sums of up to 64*3*3*batch*plane products stay well
    // inside this, while a wrong sign, a dropped term or a mis-indexed tile is O(1).
    private const double Tolerance = 1e-4;

    public static IEnumerable<object[]> RaftShapes() => new[]
    {
        //            batch, in, size, out, kernel, stride, pad
        new object[] { 1, 3, 64, 64, 7, 2, 3 },
        new object[] { 2, 3, 64, 64, 7, 2, 3 },
        new object[] { 1, 64, 32, 64, 3, 1, 1 },
        new object[] { 2, 64, 32, 64, 3, 1, 1 },
        new object[] { 1, 64, 32, 96, 3, 2, 1 },
        new object[] { 2, 64, 32, 96, 3, 2, 1 },
        new object[] { 2, 96, 16, 128, 3, 2, 1 },
        new object[] { 2, 128, 8, 8, 3, 1, 1 },
    };

    [Theory]
    [MemberData(nameof(RaftShapes))]
    public void Conv2D_ForwardAndGradients_MatchExactReference(int batch, int inChannels, int size, int outChannels, int kernel, int stride, int pad)
    {
        LogIsa();
        var engine = new CpuEngine();
        var (x, w, _, g, outSize) = Inputs(batch, inChannels, size, outChannels, kernel, stride, pad);

        var y = engine.Conv2D(x, w, stride, pad, 1);
        var dw = engine.Conv2DBackwardKernel(g, x, new[] { outChannels, inChannels, kernel, kernel }, new[] { stride, stride }, new[] { pad, pad }, new[] { 1, 1 });
        var dx = engine.Conv2DBackwardInput(g, w, new[] { batch, inChannels, size, size }, new[] { stride, stride }, new[] { pad, pad }, new[] { 1, 1 });

        var reference = Reference(x, w, null, g, batch, inChannels, size, outChannels, kernel, stride, pad, outSize, relu: false);
        AssertClose("forward", y, reference.Y);
        AssertClose("dW", dw, reference.DW);
        AssertClose("dX", dx, reference.DX);
    }

    [Theory]
    [MemberData(nameof(RaftShapes))]
    public void FusedConv2DReluTapeGradients_MatchExactReference(int batch, int inChannels, int size, int outChannels, int kernel, int stride, int pad)
    {
        LogIsa();
        var engine = new CpuEngine();
        var (x, w, bias, g, outSize) = Inputs(batch, inChannels, size, outChannels, kernel, stride, pad);

        Tensor<float> y;
        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            y = engine.FusedConv2D(x, w, bias, stride, stride, pad, pad, 1, 1, FusedActivationType.ReLU);
            var loss = engine.ReduceSum(engine.TensorMultiply(y, g), null);
            grads = tape.ComputeGradients(loss, new[] { x, w, bias });
        }

        var reference = Reference(x, w, bias, g, batch, inChannels, size, outChannels, kernel, stride, pad, outSize, relu: true);
        AssertClose("forward", y, reference.Y);
        AssertClose("dW", grads[w], reference.DW);
        AssertClose("dB", grads[bias], reference.DB);
        AssertClose("dX", grads[x], reference.DX);
    }

    // The defect is AVX-512-only, so a pass is meaningful only alongside the ISA it ran on.
    private void LogIsa()
    {
#if NET8_0_OR_GREATER
        _output.WriteLine($"Avx512F={Avx512F.IsSupported} Avx2={Avx2.IsSupported} Vector512={System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated} cores={Environment.ProcessorCount}");
#else
        _output.WriteLine($"cores={Environment.ProcessorCount} (ISA probes need .NET 8+)");
#endif
    }

    private static (Tensor<float> X, Tensor<float> W, Tensor<float> Bias, Tensor<float> G, int OutSize) Inputs(
        int batch, int inChannels, int size, int outChannels, int kernel, int stride, int pad)
    {
        var rng = new Random(20261011);
        int outSize = (size + 2 * pad - kernel) / stride + 1;
        return (Random(rng, batch, inChannels, size, size), Random(rng, outChannels, inChannels, kernel, kernel),
            Random(rng, outChannels), Random(rng, batch, outChannels, outSize, outSize), outSize);
    }

    private static Tensor<float> Random(Random rng, params int[] shape)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() * 2 - 1));
        return t;
    }

    private static (double[] Y, double[] DW, double[] DB, double[] DX) Reference(
        Tensor<float> x, Tensor<float> w, Tensor<float>? bias, Tensor<float> g,
        int batch, int inChannels, int size, int outChannels, int kernel, int stride, int pad, int outSize, bool relu)
    {
        var y = new double[batch * outChannels * outSize * outSize];
        var dw = new double[outChannels * inChannels * kernel * kernel];
        var db = new double[outChannels];
        var dx = new double[batch * inChannels * size * size];
        for (int b = 0; b < batch; b++)
        for (int o = 0; o < outChannels; o++)
        for (int oh = 0; oh < outSize; oh++)
        for (int ow = 0; ow < outSize; ow++)
        {
            double pre = bias is null ? 0.0 : bias[o];
            for (int c = 0; c < inChannels; c++)
            for (int kh = 0; kh < kernel; kh++)
            for (int kw = 0; kw < kernel; kw++)
            {
                int ih = oh * stride - pad + kh, iw = ow * stride - pad + kw;
                if (ih < 0 || iw < 0 || ih >= size || iw >= size) continue;
                pre += (double)x[b, c, ih, iw] * w[o, c, kh, kw];
            }
            bool pass = !relu || pre > 0;
            y[((b * outChannels + o) * outSize + oh) * outSize + ow] = pass ? pre : 0.0;
            double gv = pass ? g[b, o, oh, ow] : 0.0;
            db[o] += gv;
            for (int c = 0; c < inChannels; c++)
            for (int kh = 0; kh < kernel; kh++)
            for (int kw = 0; kw < kernel; kw++)
            {
                int ih = oh * stride - pad + kh, iw = ow * stride - pad + kw;
                if (ih < 0 || iw < 0 || ih >= size || iw >= size) continue;
                dw[((o * inChannels + c) * kernel + kh) * kernel + kw] += gv * x[b, c, ih, iw];
                dx[((b * inChannels + c) * size + ih) * size + iw] += gv * w[o, c, kh, kw];
            }
        }
        return (y, dw, db, dx);
    }

    private void AssertClose(string what, Tensor<float> actual, double[] expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 0.0, worst = 0.0;
        int worstIndex = -1;
        foreach (var e in expected) scale = Math.Max(scale, Math.Abs(e));
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(actual.GetFlat(i) - expected[i]);
            if (err > worst) { worst = err; worstIndex = i; }
        }
        double relative = worst / Math.Max(scale, 1e-12);
        _output.WriteLine($"{what}: worst relative error {relative:E2} at flat {worstIndex}");
        Assert.True(relative <= Tolerance,
            $"{what} disagrees with the exact reference: worst relative error {relative:E3} at flat {worstIndex} " +
            $"(actual {actual.GetFlat(worstIndex):E5}, exact {expected[worstIndex]:E5}).");
    }
}
