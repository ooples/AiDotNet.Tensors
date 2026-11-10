using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// <see cref="CpuEngine.CausalDepthwiseConv1D{T}"/>: the forward against a direct loop, and the tape gradients of the
/// input and kernel against the shift-multiply-add composition it replaces, which the tape differentiates op by op.
/// </summary>
public sealed class CausalDepthwiseConv1DTests
{
    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    // y[r, t, c] = sum_j w[j, c] * x[r, t - j, c], straight from the definition.
    private static float[] Direct(float[] x, float[] w, int rows, int seqLen, int channels, int taps)
    {
        var y = new float[x.Length];
        for (int r = 0; r < rows; r++)
            for (int t = 0; t < seqLen; t++)
                for (int c = 0; c < channels; c++)
                {
                    double acc = 0;
                    for (int j = 0; j < taps && j <= t; j++)
                        acc += w[j * channels + c] * x[(r * seqLen + t - j) * channels + c];
                    y[(r * seqLen + t) * channels + c] = (float)acc;
                }
        return y;
    }

    // The composition the fused op replaces: sum_j shift_j(x) * w[j], each shift a zero pad and a narrow on the
    // sequence axis, each tap row broadcast over the leading axes.
    private static Tensor<float> Composed(CpuEngine e, Tensor<float> x, Tensor<float> w)
    {
        int rank = x.Rank, seqAxis = rank - 2, seqLen = x.Shape[seqAxis], channels = x.Shape[rank - 1];
        int taps = w.Shape[0];
        var rowShape = Enumerable.Repeat(1, rank).ToArray();
        rowShape[rank - 1] = channels;
        Tensor<float>? y = null;
        for (int j = 0; j < taps; j++)
        {
            Tensor<float> shifted;
            if (j == 0) shifted = x;
            else
            {
                var padShape = x.Shape.ToArray();
                padShape[seqAxis] = Math.Min(j, seqLen);
                var pad = new Tensor<float>(padShape);
                shifted = j >= seqLen ? pad : e.TensorConcatenate([pad, e.TensorNarrow(x, seqAxis, 0, seqLen - j)], seqAxis);
            }
            var term = e.TensorMultiply(shifted, e.Reshape(e.TensorNarrow(w, 0, j, 1), rowShape));
            y = y is null ? term : e.TensorAdd(y, term);
        }
        return y!;
    }

    private static void AssertClose(float[] expected, float[] actual, string name, double rel = 1e-5)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = Math.Max(expected.Select(v => (double)Math.Abs(v)).DefaultIfEmpty(0).Max(), 1e-6);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= rel * scale, $"{name}[{i}]: expected {expected[i]}, got {actual[i]}");
    }

    public static TheoryData<int[], int> Shapes => new()
    {
        { new[] { 2, 7, 5 }, 3 },        // ordinary
        { new[] { 3, 2, 4, 6 }, 4 },     // two leading axes; taps equal seqLen
        { new[] { 1, 2, 3 }, 4 },        // more taps than steps
        { new[] { 2, 9, 4 }, 1 },        // one tap: a per-channel scale
        { new[] { 5, 3 }, 2 },           // rank 2: one sequence
        { new[] { 2, 6, 37 }, 3 },       // channels past the SIMD width with a scalar tail
        { new[] { 2, 5, 300 }, 3 },      // channels past one kernel-gradient channel block (256)
    };

    // The double path against the definitions of the three sums, in double: y, the input gradient and the kernel
    // gradient of a loss sum(y * r).
    [Theory]
    [MemberData(nameof(Shapes))]
    public void DoublePath_MatchesDefinitions(int[] shape, int taps)
    {
        var e = new CpuEngine();
        int channels = shape[^1], seqLen = shape[^2], rows = shape.Aggregate(1, (a, b) => a * b) / (seqLen * channels);
        var rng = new Random(7);
        var x = new Tensor<double>(shape);
        var g = new Tensor<double>(shape);
        var w = new Tensor<double>(new[] { taps, channels });
        for (int i = 0; i < x.Length; i++) { x[i] = rng.NextDouble() - 0.5; g[i] = rng.NextDouble() - 0.5; }
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextDouble() - 0.5;

        var y = e.CausalDepthwiseConv1D(x, w).ToArray();
        var dx = e.CausalDepthwiseConv1DBackwardInput(g, w).ToArray();
        var dw = e.CausalDepthwiseConv1DBackwardKernel(g, x, taps).ToArray();
        var xs = x.ToArray(); var gs = g.ToArray(); var ws = w.ToArray();
        var dwExpected = new double[taps * channels];
        for (int r = 0; r < rows; r++)
            for (int t = 0; t < seqLen; t++)
                for (int c = 0; c < channels; c++)
                {
                    int at = (r * seqLen + t) * channels + c;
                    double yExpected = 0, dxExpected = 0;
                    for (int j = 0; j < taps; j++)
                    {
                        if (t - j >= 0)
                        {
                            yExpected += ws[j * channels + c] * xs[at - j * channels];
                            dwExpected[j * channels + c] += gs[at] * xs[at - j * channels];
                        }
                        if (t + j < seqLen) dxExpected += ws[j * channels + c] * gs[at + j * channels];
                    }
                    Assert.True(Math.Abs(yExpected - y[at]) <= 1e-12, $"y[{at}]: expected {yExpected}, got {y[at]}");
                    Assert.True(Math.Abs(dxExpected - dx[at]) <= 1e-12, $"dx[{at}]: expected {dxExpected}, got {dx[at]}");
                }
        for (int i = 0; i < dwExpected.Length; i++)
            Assert.True(Math.Abs(dwExpected[i] - dw[i]) <= 1e-12, $"dw[{i}]: expected {dwExpected[i]}, got {dw[i]}");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Forward_MatchesDirectLoop(int[] shape, int taps)
    {
        var e = new CpuEngine();
        int channels = shape[^1], seqLen = shape[^2], rows = shape.Aggregate(1, (a, b) => a * b) / (seqLen * channels);
        var x = Rand(shape, 1);
        var w = Rand([taps, channels], 2);
        var y = e.CausalDepthwiseConv1D(x, w);
        Assert.Equal(shape, y.Shape.ToArray());
        AssertClose(Direct(x.ToArray(), w.ToArray(), rows, seqLen, channels, taps), y.ToArray(), "y");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void TapeGradients_MatchShiftMultiplyAddComposition(int[] shape, int taps)
    {
        var e = new CpuEngine();
        var x = Rand(shape, 3);
        var w = Rand([taps, shape[^1]], 4);
        var r = Rand(shape, 5);
        var axes = Enumerable.Range(0, shape.Length).ToArray();

        (float[] Y, float[] Dx, float[] Dw) Run(bool fused)
        {
            using var tape = new GradientTape<float>();
            var y = fused ? e.CausalDepthwiseConv1D(x, w) : Composed(e, x, w);
            var loss = e.ReduceSum(e.TensorMultiply(y, r), axes, keepDims: false);
            var g = tape.ComputeGradients(loss, [x, w]);
            return (y.ToArray(), g[x].ToArray(), g[w].ToArray());
        }

        var reference = Run(fused: false);
        var fusedRun = Run(fused: true);
        AssertClose(reference.Y, fusedRun.Y, "y");
        AssertClose(reference.Dx, fusedRun.Dx, "dx");
        AssertClose(reference.Dw, fusedRun.Dw, "dw");
    }

    [Fact]
    public void NonContiguousInput_IsReadInLogicalOrder()
    {
        var e = new CpuEngine();
        var source = Rand([2, 6, 5], 6);                       // [rows, channels, seqLen] storage
        var x = e.TensorPermute(source, [0, 2, 1]);            // [rows, seqLen, channels] view
        var w = Rand([3, 6], 7);
        var expected = e.CausalDepthwiseConv1D(x.Contiguous(), w).ToArray();
        AssertClose(expected, e.CausalDepthwiseConv1D(x, w).ToArray(), "y");
    }

    [Fact]
    public void KernelChannelMismatch_Throws()
    {
        var e = new CpuEngine();
        Assert.Throws<ArgumentException>(() => e.CausalDepthwiseConv1D(Rand([1, 4, 3], 1), Rand([2, 4], 2)));
    }
}
