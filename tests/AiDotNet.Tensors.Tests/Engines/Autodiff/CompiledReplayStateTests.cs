using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// State a compiled replay must carry from its forward to its backward and from one node to the next: the slopes an
/// unseeded RReLU drew, the argmax a pool refreshes for the op that reads its indices, the array arguments the op was
/// traced with. Also the topDb clip of AmplitudeToDB, the element pairing of binary cross-entropy and the float
/// arithmetic of RoIPool's windows, which the tape and the GPU kernels must agree on.
/// </summary>
public class CompiledReplayStateTests
{
    private static Tensor<float> Values(int[] shape, params float[] values) => new(values, shape);

    private static float[] TapeGradient(Func<IEngine, Tensor<float>> loss, Tensor<float> parameter)
    {
        var engine = new CpuEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var tape = new GradientTape<float>();
            var grads = tape.ComputeGradients(loss(engine), new[] { parameter });
            return grads[parameter].GetFlattenedData();
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    private static ICompiledTrainingPlan<float> Compile(Func<IEngine, Tensor<float>> loss, Tensor<float> parameter,
        out Tensor<float> traced)
    {
        var engine = new CpuEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var scope = GraphMode.EnableTraining(new[] { parameter });
            traced = loss(engine);
            return scope.CompileTraining(new[] { parameter });
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-4f * (1f + Math.Abs(expected[i])),
                $"{what}: element {i} expected {expected[i]:R}, got {actual[i]:R}");
    }

    /// <summary>
    /// AmplitudeToDB with topDb replaces every output below peak - topDb by peak - topDb, so that output's gradient
    /// belongs to the peak. Sum loss on [1, 0.0001] with topDb 20: the second output (-80 dB) is clipped to -20 dB,
    /// and the peak gets both outputs' gradient, 2 * 20 / ln 10 = 17.37.
    /// </summary>
    [Fact]
    public void AmplitudeToDB_TopDbClip_SendsTheClippedGradientToThePeak()
    {
        var x = Values(new[] { 2 }, 1f, 1e-4f);
        Func<IEngine, Tensor<float>> loss = e => e.ReduceSum(e.AmplitudeToDB(x, 1e-10f, 20f), null);
        double perOutput = 20.0 / Math.Log(10.0);
        var tape = TapeGradient(loss, x);
        AssertClose(new[] { (float)(2 * perOutput), 0f }, tape, "tape");

        using var plan = Compile(loss, x, out _);
        plan.Step();
        AssertClose(tape, plan.Gradients[0].AsSpan(), "compiled");
    }

    /// <summary>
    /// Unseeded training RReLU draws new slopes each step. The compiled backward re-runs the op under a tape, so it
    /// must reuse the slopes the forward drew: for x &lt; 0 the gradient of sum(y) is y / x.
    /// </summary>
    [Fact]
    public void Rrelu_UnseededTraining_CompiledBackwardUsesTheForwardSlopes()
    {
        var x = Values(new[] { 8 }, -1f, -2f, -0.5f, -3f, 1f, -1.5f, -0.25f, 2f);
        Tensor<float>? y = null;
        Func<IEngine, Tensor<float>> loss = e =>
        {
            y = e.TensorRrelu(x, 0.1, 0.4, training: true, seed: null);
            return e.ReduceSum(y, null);
        };

        using var plan = Compile(loss, x, out _);
        var firstSlopes = new float[x.Length];
        for (int step = 0; step < 2; step++)
        {
            plan.Step();
            var grad = plan.Gradients[0].AsSpan();
            var output = y!.AsSpan();
            for (int i = 0; i < x.Length; i++)
            {
                float slope = x[i] > 0 ? 1f : output[i] / x[i];
                if (x[i] < 0)
                    Assert.True(slope >= 0.1f - 1e-6f && slope <= 0.4f + 1e-6f, $"step {step}: slope {slope} at {i}");
                Assert.True(Math.Abs(grad[i] - slope) <= 1e-6f,
                    $"step {step}, element {i}: the forward applied slope {slope:R}, the backward {grad[i]:R}");
                if (step == 0) firstSlopes[i] = slope;
                else if (x[i] < 0) Assert.NotEqual(firstSlopes[i], slope);
            }
        }
    }

    /// <summary>
    /// MaxUnpool reads only the pool's indices. Each compiled step must pool again and refresh them before the unpool
    /// runs: the input below moves its maximum between steps, and the compiled gradient must follow it.
    /// </summary>
    [Fact]
    public void MaxPool2DIndices_RefreshBeforeTheirConsumer_InEveryCompiledStep()
    {
        var x = Values(new[] { 1, 1, 2, 2 }, 4f, 1f, 2f, 3f);
        var weights = Values(new[] { 1, 1, 2, 2 }, 1f, 2f, 3f, 4f);
        Func<IEngine, Tensor<float>> loss = e =>
        {
            var pooled = e.MaxPool2DWithTensorIndices(x, new[] { 2, 2 }, new[] { 2, 2 }, out var indices);
            var unpooled = e.TensorMaxUnpool(pooled, indices, new[] { 2, 2 });
            return e.ReduceSum(e.TensorMultiply(unpooled, weights), null);
        };

        using var plan = Compile(loss, x, out _);
        plan.Step();
        AssertClose(TapeGradient(loss, x), plan.Gradients[0].AsSpan(), "step 0 (maximum at 0)");

        // Move the maximum to the last pixel. A stale index would still route the gradient through pixel 0.
        x[0, 0, 0, 0] = 0f;
        x[0, 0, 1, 1] = 9f;
        plan.Step();
        var expected = TapeGradient(loss, x);
        Assert.Equal(4f, expected[3]);
        AssertClose(expected, plan.Gradients[0].AsSpan(), "step 1 (maximum at 3)");
    }

    /// <summary>
    /// A compiled op re-runs with the arguments it was traced with. Changing the pad array afterwards (to another pad
    /// of the same total size, so the shapes still agree) must not change the replay.
    /// </summary>
    [Fact]
    public void PadNd_ArgumentsAreFixedAtTraceTime()
    {
        var x = Values(new[] { 1, 3 }, 1f, 2f, 3f);
        var weights = Values(new[] { 1, 5 }, 1f, 2f, 3f, 4f, 5f);
        var pad = new[] { 1, 1 };
        Func<IEngine, Tensor<float>> loss = e => e.ReduceSum(e.TensorMultiply(e.PadNd(x, pad, PadMode.Constant, 0f), weights), null);
        var expected = TapeGradient(loss, x);
        AssertClose(new[] { 2f, 3f, 4f }, expected, "tape");

        using var plan = Compile(loss, x, out _);
        pad[0] = 2;
        pad[1] = 0;
        plan.Step();
        AssertClose(expected, plan.Gradients[0].AsSpan(), "compiled after the caller changed pad");
    }

    /// <summary>Binary cross-entropy pairs predictions and targets element by element, so their counts must agree.</summary>
    [Fact]
    public void BinaryCrossEntropy_RejectsMismatchedCounts()
    {
        var engine = new CpuEngine();
        var p = Values(new[] { 2, 1 }, 0.2f, 0.7f);
        var t = Values(new[] { 3 }, 0f, 1f, 1f);
        Assert.Throws<ArgumentException>(() => engine.TensorBinaryCrossEntropy(p, t, 1e-7f));
    }

    /// <summary>
    /// RoIPool on float tensors rounds the scaled box in float, as the GPU kernels and torchvision do: 0.7f * 5 is
    /// 3.5 in float (4 after rounding), 3.4999999 in double (3). Column 3 holds the image's maximum, so a double
    /// window would pick it. The float window starts at column 4, and the backward sends the gradient to the
    /// pixel the forward picked.
    /// </summary>
    [Fact]
    public void RoIPool_Float_UsesFloatWindows_InForwardAndBackward()
    {
        const int size = 8;
        var input = new Tensor<float>(new[] { 1, 1, size, size });
        for (int i = 0; i < input.Length; i++) input[i] = i % size == 3 ? 100f : i;
        var boxes = Values(new[] { 1, 5 }, 0f, 0.7f, 0.7f, 1.2f, 1.2f);
        var engine = new CpuEngine();
        var pooled = engine.RoIPool(input, boxes, 1, 1, 5f);
        // Float window: x and y from round(3.5) = 4 to round(6) = 6. Its maximum is row 6, column 6.
        Assert.Equal(6 * size + 6, pooled[0]);

        var grad = TapeGradient(e => e.ReduceSum(e.RoIPool(input, boxes, 1, 1, 5f), null), input);
        for (int i = 0; i < grad.Length; i++)
            Assert.Equal(i == 6 * size + 6 ? 1f : 0f, grad[i]);
    }
}
