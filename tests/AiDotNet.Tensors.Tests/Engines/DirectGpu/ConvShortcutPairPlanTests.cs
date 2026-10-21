using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The compiled training plan's shortcut pair (CompiledTrainingPlan.ShortcutPair.cs) on a downsampling block's shape:
/// h = relu(conv0(x)), then a 3x3 stride-2 convolution and a 1x1 stride-2 projection of h. With the pair the plan
/// computes h's gradient in one launch; every parameter gradient must match the unfused plan's
/// (AIDOTNET_CONV_SHORTCUT_FUSION=0) over several steps, and the pair must actually run (the wiring, not just the
/// kernel).
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ConvShortcutPairPlanTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly ITestOutputHelper _out;

    public ConvShortcutPairPlanTests(DirectGpuTensorEngineTestFixture fixture, ITestOutputHelper output)
    {
        _fixture = fixture;
        _out = output;
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

    // Returns the parameter gradients of each step, and how many pair launches the plan made.
    private (float[][][] Grads, long PairLaunches) Train(DirectGpuTensorEngine gpu, bool fuse, int batch, int inC, int c, int k,
        int size, int steps)
    {
        var previousEngine = AiDotNetEngine.Current;
        var previousFlag = Environment.GetEnvironmentVariable("AIDOTNET_CONV_SHORTCUT_FUSION");
        AiDotNetEngine.Current = gpu;
        Environment.SetEnvironmentVariable("AIDOTNET_CONV_SHORTCUT_FUSION", fuse ? null : "0");
        try
        {
            var x = Make(new[] { batch, inC, size, size }, 1, 1f).Gpu();
            var w0 = Make(new[] { c, inC, 3, 3 }, 2, 0.3f).Gpu();
            var w3 = Make(new[] { k, c, 3, 3 }, 3, 0.1f).Gpu();
            var w1 = Make(new[] { k, c, 1, 1 }, 4, 0.2f).Gpu();
            Tensor<float> Loss()
            {
                var h = gpu.ReLU(gpu.Conv2D(x, w0, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 }));
                var main = gpu.Conv2D(h, w3, new[] { 2, 2 }, new[] { 1, 1 }, new[] { 1, 1 });
                var shortcut = gpu.Conv2D(h, w1, new[] { 2, 2 }, new[] { 0, 0 }, new[] { 1, 1 });
                var s = gpu.TensorAdd(main, shortcut);
                return gpu.ReduceSum(gpu.TensorMultiply(s, s), new[] { 0, 1, 2, 3 }, keepDims: false);
            }
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                Loss();
                plan = scope.CompileTraining(new[] { w0, w3, w1 });
            }
            long before = DirectGpuTensorEngine.ShortcutPairLaunches;
            var grads = new float[steps][][];
            using (plan)
                for (int st = 0; st < steps; st++)
                {
                    plan.Step();
                    grads[st] = new[] { plan.Gradients[0].ToArray(), plan.Gradients[1].ToArray(), plan.Gradients[2].ToArray() };
                }
            return (grads, DirectGpuTensorEngine.ShortcutPairLaunches - before);
        }
        finally
        {
            Environment.SetEnvironmentVariable("AIDOTNET_CONV_SHORTCUT_FUSION", previousFlag);
            AiDotNetEngine.Current = previousEngine;
        }
    }

    [SkippableTheory]
    [InlineData(2, 4, 16, 24, 16)]    // output width 8
    [InlineData(2, 8, 32, 40, 64)]    // output width 32 (one row per block); K off the eight-channel step
    [InlineData(3, 4, 12, 16, 8)]     // output width 4
    public void ThePairMatchesTheUnfusedPlanAndRuns(int batch, int inC, int c, int k, int size)
    {
        Skip.IfNot(_fixture.IsAvailable, "No DirectGpu backend available.");
        var gpu = _fixture.Engine!;
        Skip.IfNot(gpu.SupportsConv2DBackwardInputShortcutPair(new[] { batch, c, size, size }, new[] { k, c, 3, 3 },
            new[] { k, c, 1, 1 }, size / 2, size / 2), "The backend does not compute the shortcut pair.");
        const int steps = 3;   // the second and later steps replay (captured where the backend supports it)
        var (unfused, offLaunches) = Train(gpu, fuse: false, batch, inC, c, k, size, steps);
        var (fused, onLaunches) = Train(gpu, fuse: true, batch, inC, c, k, size, steps);
        Assert.Equal(0, offLaunches);
        Assert.True(onLaunches >= 1, "the compiled plan never took the shortcut pair");
        string[] names = { "dW0 (through h's gradient)", "dW3", "dW1" };
        double worst = 0;
        for (int st = 0; st < steps; st++)
            for (int p = 0; p < 3; p++)
            {
                float[] a = unfused[st][p], b = fused[st][p];
                double scale = 0;
                foreach (var v in a) scale = Math.Max(scale, Math.Abs(v));
                for (int i = 0; i < a.Length; i++)
                {
                    double err = Math.Abs(a[i] - b[i]) / Math.Max(1e-6, scale);
                    worst = Math.Max(worst, err);
                    Assert.True(err < 1e-4, $"step {st} {names[p]}[{i}]: unfused {a[i]} vs pair {b[i]} (max |grad| {scale})");
                }
            }
        _out.WriteLine($"pair launches {onLaunches}; max |unfused - pair| / max|grad| = {worst:E2}");
    }
}
