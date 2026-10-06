using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Simd;

/// <summary>
/// Covers the CPU training hot paths that were rewritten for parallel dispatch: the 3x3 stride-1 conv's flat
/// (batch item, channel block) loop and Tensor.Sum's keep-one-axis reduction (every bias gradient).
/// </summary>
public class CpuTrainingHotPathTests
{
    private static float[] RandomFloats(int length, int seed)
    {
        var rng = new Random(seed);
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() - 0.5);
        return data;
    }

#if !NET471 // SimdConvHelper is not compiled for net471 (AiDotNet.Tensors.csproj removes it there).
    /// <summary>
    /// Output channel counts pick each variant: 16 -> Block4, 6 -> Block2, 3 -> per-channel. Batch &gt; 1 so tasks
    /// span batch items, and small spatial sizes so each task sits below the old per-task parallel gate.
    /// </summary>
    [Theory]
    [InlineData(4, 1, 28, 16)]
    [InlineData(8, 16, 14, 16)]
    [InlineData(3, 5, 9, 6)]
    [InlineData(5, 4, 11, 3)]
    [InlineData(1, 8, 32, 32)]
    public unsafe void Conv3x3Stride1_BatchedTasks_MatchNaiveReference(int batch, int inC, int hw, int outC)
    {
        var input = RandomFloats(batch * inC * hw * hw, 11);
        var kernel = RandomFloats(outC * inC * 9, 23);
        var output = new float[batch * outC * hw * hw];
        // Poison the output: every element must be written.
        for (int i = 0; i < output.Length; i++) output[i] = float.NaN;

        fixed (float* ip = input, kp = kernel, op = output)
            SimdConvHelper.Conv3x3Stride1(ip, kp, op, batch, inC, hw, hw, outC, 1, 1, 1, 1);

        for (int b = 0; b < batch; b++)
        for (int oc = 0; oc < outC; oc++)
        for (int oh = 0; oh < hw; oh++)
        for (int ow = 0; ow < hw; ow++)
        {
            double expected = 0;
            for (int ic = 0; ic < inC; ic++)
            for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
            {
                int ih = oh + kh - 1, iw = ow + kw - 1;
                if (ih < 0 || ih >= hw || iw < 0 || iw >= hw) continue;
                expected += input[((b * inC + ic) * hw + ih) * hw + iw] * kernel[((oc * inC + ic) * 3 + kh) * 3 + kw];
            }
            float actual = output[((b * outC + oc) * hw + oh) * hw + ow];
            Assert.True(Math.Abs(expected - actual) < 1e-4,
                $"b={b} oc={oc} oh={oh} ow={ow}: expected {expected}, got {actual}");
        }
    }

#endif

    /// <summary>
    /// The keep-one-axis path must equal summing each kept index's elements in source row-major order, bit for bit.
    /// </summary>
    [Theory]
    [InlineData(new[] { 64, 16, 28, 28 }, 1)]
    [InlineData(new[] { 3, 5, 7 }, 0)]
    [InlineData(new[] { 3, 5, 7 }, 2)]
    [InlineData(new[] { 2, 3, 4, 5, 6 }, 3)]
    public void Sum_KeepOneAxis_IsBitIdenticalToRowMajorAccumulation(int[] shape, int keep)
    {
        var tensor = new Tensor<float>(shape);
        var data = RandomFloats(tensor.Length, 5);
        for (int i = 0; i < data.Length; i++) tensor[i] = data[i];
        var axes = new int[shape.Length - 1];
        for (int d = 0, a = 0; d < shape.Length; d++) if (d != keep) axes[a++] = d;

        var result = tensor.Sum(axes);

        int outer = 1, inner = 1, kept = shape[keep];
        for (int d = 0; d < keep; d++) outer *= shape[d];
        for (int d = keep + 1; d < shape.Length; d++) inner *= shape[d];
        Assert.Equal(new[] { kept }, result.Shape.ToArray());
        for (int c = 0; c < kept; c++)
        {
            float expected = 0f;
            for (int o = 0; o < outer; o++)
                for (int j = 0; j < inner; j++)
                    expected += data[(o * kept + c) * inner + j];
            Assert.Equal(expected, result[c]);
        }
    }

    /// <summary>
    /// Adaptive average pool backward on non-dividing bins (7 -> 4 overlaps windows) under a tape, twice, so the
    /// second step reuses a recycled arena buffer: each input gradient must equal the sum over the windows that
    /// contain it of upstream / window size, with no residue from the first step.
    /// </summary>
    [Fact]
    public void AdaptiveAvgPool2DBackward_NonDividingBins_MatchesWindowSums_AcrossSteps()
    {
        var engine = new CpuEngine();
        int batch = 3, channels = 5, inH = 7, inW = 7, outH = 4, outW = 4;
        // One arena across both steps, Reset between them like a training loop: step 2's gradient buffer is the
        // recycled step-1 buffer, so a missing per-plane clear shows up as step-1 residue.
        using var arena = TensorArena.Create();
        for (int step = 0; step < 2; step++)
        {
            if (step > 0) arena.Reset();
            var input = new Tensor<float>(new[] { batch, channels, inH, inW });
            var data = RandomFloats(input.Length, 31 + step);
            for (int i = 0; i < data.Length; i++) input[i] = data[i];
            var weights = new Tensor<float>(new[] { batch, channels, outH, outW });
            var w = RandomFloats(weights.Length, 77 + step);
            for (int i = 0; i < w.Length; i++) weights[i] = w[i];

            Tensor<float> inputGrad;
            using (var tape = new GradientTape<float>())
            {
                var pooled = engine.AdaptiveAvgPool2D(input, outH, outW);
                var loss = engine.ReduceSum(engine.TensorMultiply(pooled, weights), null);
                var grads = tape.ComputeGradients(loss, new[] { input });
                inputGrad = grads[input];
            }

            var expected = new double[input.Length];
            for (int plane = 0; plane < batch * channels; plane++)
            for (int oh = 0; oh < outH; oh++)
            for (int ow = 0; ow < outW; ow++)
            {
                int hs = (int)Math.Floor((double)oh * inH / outH), he = (int)Math.Ceiling((double)(oh + 1) * inH / outH);
                int ws = (int)Math.Floor((double)ow * inW / outW), we = (int)Math.Ceiling((double)(ow + 1) * inW / outW);
                double g = w[(plane * outH + oh) * outW + ow] / ((he - hs) * (we - ws));
                for (int ih = hs; ih < he; ih++)
                    for (int iw = ws; iw < we; iw++)
                        expected[(plane * inH + ih) * inW + iw] += g;
            }
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - inputGrad[i]) < 1e-5,
                    $"step {step} [{i}]: expected {expected[i]}, got {inputGrad[i]}");
        }
    }

    [Fact]
    public void Sum_KeepOneAxis_Double_IsBitIdenticalToRowMajorAccumulation()
    {
        int[] shape = { 8, 6, 5, 5 };
        var tensor = new Tensor<double>(shape);
        var rng = new Random(9);
        for (int i = 0; i < tensor.Length; i++) tensor[i] = rng.NextDouble() - 0.5;

        var result = tensor.Sum(new[] { 0, 2, 3 });

        for (int c = 0; c < 6; c++)
        {
            double expected = 0d;
            for (int o = 0; o < 8; o++)
                for (int j = 0; j < 25; j++)
                    expected += tensor[(o * 6 + c) * 25 + j];
            Assert.Equal(expected, result[c]);
        }
    }
}
