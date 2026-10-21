using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The compiled training plan's shortcut pair (CompiledTrainingPlan.ShortcutPair.cs) without a GPU: an engine that
/// reports SupportsGpu (so the plan takes its device-generic backward) and computes the pair's input gradient as the
/// two separate input gradients plus an add. Covers the detection, the fused backward action and its fallback when the
/// engine declines, on CI hosts where the CUDA pair kernel (ConvShortcutPairPlanTests) never runs.
/// </summary>
[Collection("VulkanGlobalState")]
public class ShortcutPairPlanCpuTests
{
    private sealed class PairEngine : CpuEngine, IConvShortcutPairEngine
    {
        public bool Supports = true;
        public bool Decline;
        public int Calls;

        public override bool SupportsGpu => true;

        bool IConvShortcutPairEngine.SupportsConv2DBackwardInputShortcutPair(int[] inputShape, int[] kernel3, int[] kernel1,
            int outH, int outW)
            => Supports && kernel3[1] == inputShape[1] && kernel1[1] == inputShape[1]
               && inputShape[2] == 2 * outH && inputShape[3] == 2 * outW;

        Tensor<float>? IConvShortcutPairEngine.TryConv2DBackwardInputShortcutPair(Tensor<float> gradOutput3,
            Tensor<float> kernel3, Tensor<float> gradOutput1, Tensor<float> kernel1, int[] inputShape)
        {
            Calls++;
            if (Decline) return null;
            var main = Conv2DBackwardInput(gradOutput3, kernel3, inputShape, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
            var shortcut = Conv2DBackwardInput(gradOutput1, kernel1, inputShape, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 1, 1 });
            return TensorAdd(main, shortcut);
        }
    }

    private static Tensor<float> Make(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        int n = 1;
        foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < data.Length; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return new Tensor<float>(data, shape);
    }

    // h = relu(conv0(x)); a 3x3 stride-2 conv and a projection of h (shortcutPad 0 is the 1x1/s2/p0 shortcut the pair
    // admits), added, sum of squares. Returns each step's parameter gradients.
    private static float[][][] Train(PairEngine engine, bool fuse, bool shortcutFirst, int shortcutPad = 0, int steps = 3)
    {
        const int batch = 2, inC = 3, c = 4, k = 6, size = 8;
        var previousEngine = AiDotNetEngine.Current;
        var previousFlag = Environment.GetEnvironmentVariable("AIDOTNET_CONV_SHORTCUT_FUSION");
        AiDotNetEngine.Current = engine;
        Environment.SetEnvironmentVariable("AIDOTNET_CONV_SHORTCUT_FUSION", fuse ? null : "0");
        try
        {
            var x = Make(new[] { batch, inC, size, size }, 1, 1f);
            var w0 = Make(new[] { c, inC, 3, 3 }, 2, 0.3f);
            var w3 = Make(new[] { k, c, 3, 3 }, 3, 0.1f);
            var w1 = Make(new[] { k, c, 1, 1 }, 4, 0.2f);
            void Loss()
            {
                var h = engine.ReLU(engine.Conv2D(x, w0, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }));
                Tensor<float> Main() => engine.Conv2D(h, w3, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
                Tensor<float> Shortcut() => engine.Conv2D(h, w1, new[] { 2, 2 }, new[] { shortcutPad, shortcutPad }, new[] { 1, 1 });
                Tensor<float> main, shortcut;
                if (shortcutFirst) { shortcut = Shortcut(); main = Main(); }
                else { main = Main(); shortcut = Shortcut(); }
                if (shortcutPad != 0) shortcut = engine.Crop(shortcut, 0, 0, size / 2, size / 2);
                var s = engine.TensorAdd(main, shortcut);
                engine.ReduceSum(engine.TensorMultiply(s, s), new[] { 0, 1, 2, 3 }, keepDims: false);
            }
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Loss();
                plan = scope.CompileTraining(new[] { w0, w3, w1 });
            }
            var grads = new float[steps][][];
            using (plan)
                for (int st = 0; st < steps; st++)
                {
                    plan.Step();
                    grads[st] = new[] { plan.Gradients[0].ToArray(), plan.Gradients[1].ToArray(), plan.Gradients[2].ToArray() };
                }
            return grads;
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_SHORTCUT_FUSION", previousFlag);
            AiDotNetEngine.Current = previousEngine;
        }
    }

    private static void AssertClose(float[][][] expected, float[][][] actual)
    {
        string[] names = { "dW0 (through h's gradient)", "dW3", "dW1" };
        for (int st = 0; st < expected.Length; st++)
            for (int p = 0; p < 3; p++)
            {
                float[] a = expected[st][p], b = actual[st][p];
                Assert.Equal(a.Length, b.Length);
                double scale = 0;
                foreach (var v in a) scale = Math.Max(scale, Math.Abs(v));
                Assert.True(scale > 0, $"step {st} {names[p]} is all zero");
                for (int i = 0; i < a.Length; i++)
                    Assert.True(Math.Abs(a[i] - b[i]) <= 1e-4 * scale,
                        $"step {st} {names[p]}[{i}]: unfused {a[i]} vs pair {b[i]} (max |grad| {scale})");
            }
    }

    [Theory]
    [InlineData(false)]   // the shortcut is the later step
    [InlineData(true)]    // the shortcut is the earlier step
    public void ThePairMatchesTheUnfusedPlan(bool shortcutFirst)
    {
        var unfusedEngine = new PairEngine();
        var unfused = Train(unfusedEngine, fuse: false, shortcutFirst);
        Assert.Equal(0, unfusedEngine.Calls);

        var engine = new PairEngine();
        var fused = Train(engine, fuse: true, shortcutFirst);
        Assert.True(engine.Calls >= 3, $"the plan called the pair {engine.Calls} times over 3 steps");
        AssertClose(unfused, fused);
    }

    [Fact]
    public void ADeclinedPairFallsBackToBothBackwards()
    {
        var unfused = Train(new PairEngine(), fuse: false, shortcutFirst: false);
        var engine = new PairEngine { Decline = true };
        var fallback = Train(engine, fuse: true, shortcutFirst: false);
        Assert.True(engine.Calls >= 3, $"the plan called the pair {engine.Calls} times over 3 steps");
        AssertClose(unfused, fallback);
    }

    [Fact]
    public void AnUnsupportedPairIsNotFused()
    {
        var engine = new PairEngine { Supports = false };
        var grads = Train(engine, fuse: true, shortcutFirst: false);
        Assert.Equal(0, engine.Calls);
        AssertClose(Train(new PairEngine(), fuse: false, shortcutFirst: false), grads);
    }

    [Fact]
    public void APaddedProjectionIsNotAPair()
    {
        var engine = new PairEngine();
        var grads = Train(engine, fuse: true, shortcutFirst: false, shortcutPad: 1);
        Assert.Equal(0, engine.Calls);
        AssertClose(Train(new PairEngine(), fuse: false, shortcutFirst: false, shortcutPad: 1), grads);
    }
}
