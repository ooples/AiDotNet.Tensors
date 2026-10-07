using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The float adaptive average pool kernels (forward and backward) and the compiled plan's specialized backward,
/// which writes the input gradient straight into the plan's buffer. The kernels are meant to be bit-identical to the
/// per-element scalar loops they replaced, so the references below are those loops, copied verbatim; the shapes
/// include overlapping (non-dividing) windows, where the backward's addition order matters, and upsampling ones.
/// </summary>
[Collection("EngineCurrentGlobalState")]
public class AdaptiveAvgPoolKernelTests
{
    private static Tensor<float> Rnd(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static void AssertBitEqual(ReadOnlySpan<float> expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}: element {i} expected {expected[i]:R} got {actual[i]:R}");
    }

    /// <summary>The scalar forward bin loop the kernel replaced.</summary>
    private static float[] ReferenceForward(float[] x, int planes, int iH, int iW, int oH, int oW)
    {
        var y = new float[planes * oH * oW];
        for (int bc = 0; bc < planes; bc++)
        {
            int inBase = bc * iH * iW, outBase = bc * oH * oW;
            for (int oh = 0; oh < oH; oh++)
            {
                int startH = (int)Math.Floor((double)oh * iH / oH);
                int endH = (int)Math.Ceiling((double)(oh + 1) * iH / oH);
                for (int ow = 0; ow < oW; ow++)
                {
                    int startW = (int)Math.Floor((double)ow * iW / oW);
                    int endW = (int)Math.Ceiling((double)(ow + 1) * iW / oW);
                    float sum = 0f;
                    int count = 0;
                    for (int ih = startH; ih < endH; ih++)
                        for (int iw = startW; iw < endW; iw++)
                        {
                            sum += x[inBase + ih * iW + iw];
                            count++;
                        }
                    y[outBase + oh * oW + ow] = sum / count;
                }
            }
        }
        return y;
    }

    /// <summary>The scalar backward loop the kernel replaced (a cleared plane, then window-order additions).</summary>
    private static float[] ReferenceBackward(float[] g, int planes, int iH, int iW, int oH, int oW)
    {
        var d = new float[planes * iH * iW];
        for (int p = 0; p < planes; p++)
            for (int oh = 0; oh < oH; oh++)
            {
                int hs = (int)Math.Floor((double)oh * iH / oH), he = (int)Math.Ceiling((double)(oh + 1) * iH / oH);
                for (int ow = 0; ow < oW; ow++)
                {
                    int ws = (int)Math.Floor((double)ow * iW / oW), we = (int)Math.Ceiling((double)(ow + 1) * iW / oW);
                    float v = g[(p * oH + oh) * oW + ow] / ((he - hs) * (we - ws));
                    for (int ih = hs; ih < he; ih++)
                        for (int iw = ws; iw < we; iw++) d[p * iH * iW + ih * iW + iw] += v;
                }
            }
        return d;
    }

    public static TheoryData<int, int, int, int, int, int> Shapes => new()
    {
        { 64, 32, 14, 14, 4, 4 },   // the parity CNN head: overlapping 4-wide windows
        { 2, 3, 5, 7, 3, 2 },       // overlapping, different per axis
        { 2, 2, 6, 6, 3, 3 },       // dividing: disjoint 2x2 windows
        { 1, 2, 3, 3, 5, 5 },       // upsampling: more outputs than inputs
        { 3, 1, 7, 1, 2, 1 },       // a column
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void ForwardMatchesScalarLoop(int n, int c, int iH, int iW, int oH, int oW)
    {
        var engine = new CpuEngine();
        var x = Rnd(new[] { n, c, iH, iW }, 1);
        x[0] = -0f; x[1] = float.NaN;
        var expected = ReferenceForward(x.GetDataArray(), n * c, iH, iW, oH, oW);
        var y = new Tensor<float>(new[] { n, c, oH, oW });
        for (int i = 0; i < y.Length; i++) y[i] = 77f;
        engine.AdaptiveAvgPool2DInto(y, x, oH, oW);
        AssertBitEqual(expected, y.AsSpan(), "forward");
    }

    [Theory]
    [MemberData(nameof(Shapes))]
    public void BackwardMatchesScalarLoop(int n, int c, int iH, int iW, int oH, int oW)
    {
        var g = Rnd(new[] { n, c, oH, oW }, 2);
        g[0] = -0f;
        var expected = ReferenceBackward(g.GetDataArray(), n * c, iH, iW, oH, oW);
        var d = new float[n * c * iH * iW];
        for (int i = 0; i < d.Length; i++) d[i] = 77f;   // stale contents must be fully replaced
        CpuEngine.AdaptiveAvgPool2DBackwardFloat(g.GetDataArray(), 0, d, 0, n * c, iH, iW, oH, oW, accumulate: false);
        AssertBitEqual(expected, d, "backward");

        // Accumulating adds the separately computed plane gradient onto what is there.
        var prior = Rnd(new[] { n * c * iH * iW }, 3).GetDataArray();
        var acc = (float[])prior.Clone();
        CpuEngine.AdaptiveAvgPool2DBackwardFloat(g.GetDataArray(), 0, acc, 0, n * c, iH, iW, oH, oW, accumulate: true);
        var expectedAcc = new float[prior.Length];
        for (int i = 0; i < prior.Length; i++) expectedAcc[i] = prior[i] + expected[i];
        AssertBitEqual(expectedAcc, acc, "accumulated backward");
    }

    [Fact]
    public void BackwardHonoursOffsets()
    {
        const int planes = 6, iH = 5, iW = 7, oH = 3, oW = 2;
        var g = Rnd(new[] { 3 + planes * oH * oW }, 4).GetDataArray();
        var expected = ReferenceBackward(g.AsSpan(3).ToArray(), planes, iH, iW, oH, oW);
        var d = new float[5 + planes * iH * iW + 4];
        for (int i = 0; i < d.Length; i++) d[i] = 9f;
        CpuEngine.AdaptiveAvgPool2DBackwardFloat(g, 3, d, 5, planes, iH, iW, oH, oW, accumulate: false);
        AssertBitEqual(expected, d.AsSpan(5, planes * iH * iW), "offset backward");
        for (int i = 0; i < 5; i++) Assert.Equal(9f, d[i]);
        for (int i = 5 + planes * iH * iW; i < d.Length; i++) Assert.Equal(9f, d[i]);
    }

    [Fact]
    public void KernelsAreIndependentOfThreadCount()
    {
        var engine = new CpuEngine();
        var x = Rnd(new[] { 64, 32, 14, 14 }, 5);
        var g = Rnd(new[] { 64, 32, 4, 4 }, 6);
        int prior = CpuParallelSettings.MaxDegreeOfParallelism;
        try
        {
            CpuParallelSettings.MaxDegreeOfParallelism = 1;
            var y1 = new Tensor<float>(new[] { 64, 32, 4, 4 });
            engine.AdaptiveAvgPool2DInto(y1, x, 4, 4);
            var d1 = new float[x.Length];
            CpuEngine.AdaptiveAvgPool2DBackwardFloat(g.GetDataArray(), 0, d1, 0, 64 * 32, 14, 14, 4, 4, false);
            CpuParallelSettings.MaxDegreeOfParallelism = Math.Max(2, Environment.ProcessorCount);
            var y2 = new Tensor<float>(new[] { 64, 32, 4, 4 });
            engine.AdaptiveAvgPool2DInto(y2, x, 4, 4);
            var d2 = new float[x.Length];
            CpuEngine.AdaptiveAvgPool2DBackwardFloat(g.GetDataArray(), 0, d2, 0, 64 * 32, 14, 14, 4, 4, false);
            AssertBitEqual(y1.AsSpan(), y2.AsSpan(), "forward");
            AssertBitEqual(d1, d2, "backward");
        }
        finally
        {
            CpuParallelSettings.MaxDegreeOfParallelism = prior;
        }
    }

    /// <summary>
    /// The compiled plan's specialized backward against the eager tape (generic backward), bit for bit, over several
    /// steps -- with the pooled tensor read only by the pool and, in the shared case, also by a second op (whose
    /// gradient must be added to, not overwritten).
    /// </summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledPlanMatchesTape(bool shared)
    {
        var priorEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = new CpuEngine();
        try
        {
            var p = Rnd(new[] { 3, 4, 14, 14 }, 7);
            var coef = Rnd(new[] { 3, 4, 4, 4 }, 8);
            var coef2 = Rnd(new[] { 3, 4, 14, 14 }, 9);
            Tensor<float> Forward(IEngine e)
            {
                var h = e.TensorMultiply(p, p);   // the pooled tensor is an activation with its own gradient buffer
                var loss = e.ReduceSum(e.TensorMultiply(e.AdaptiveAvgPool2D(h, 4, 4), coef), null);
                if (shared) loss = e.TensorAdd(loss, e.ReduceSum(e.TensorMultiply(h, coef2), null));
                return loss;
            }
            float[] tape;
            using (var t = new GradientTape<float>())
            {
                var loss = Forward(new CpuEngine());
                tape = t.ComputeGradients(loss, new[] { p })[p].GetFlattenedData();
            }
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Forward(new CpuEngine());
                plan = scope.CompileTraining(new[] { p });
            }
            try
            {
                for (int step = 0; step < 3; step++)
                {
                    plan.Step();
                    AssertBitEqual(tape, plan.Gradients[0].AsSpan(), $"step {step}");
                }
            }
            finally
            {
                plan.Dispose();
            }
        }
        finally
        {
            AiDotNetEngine.Current = priorEngine;
        }
    }
}
