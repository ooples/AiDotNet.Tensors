// Copyright (c) CrowdTrainer LLC. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Loss scaling inside the compiled plan (PyTorch's GradScaler): the loss gradient is seeded with the scale, the
/// gradients are unscaled and checked before the clip and the optimizer, a non-finite step is skipped and the scale
/// backs off, finite steps grow it every interval. On the host and on the GPU (where the check and the update run on
/// the device), and matmul autocast down-converts GEMMs to FP16/BF16 with FP32 accumulation.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class CompiledPlanLossScalingTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public CompiledPlanLossScalingTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private const int In = 16, Out = 8, Rows = 12;

    private static float[] Data(int n, double a, double b)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)Math.Sin(a * i + b);
        return v;
    }

    // loss = sum(tanh(x W) * c) for `steps` SGD steps; returns W and the plan's scaler state.
    private (float[] W, LossScaleState? State) Train(bool onGpu, LossScalingOptions? scaling, int steps, float[]? c = null)
    {
        IEngine engine = onGpu ? _fixture.Engine! : new CpuEngine();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Data(In * Out, 0.37, 0.1)));
            if (onGpu) w = w.Gpu();
            var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(Data(Rows * In, 0.21, 0.3)));
            var ct = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(c ?? Data(Rows * Out, 0.13, 0.7)));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReduceSum(engine.TensorMultiply(engine.Tanh(engine.TensorMatMul(x, w)), ct), null);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.05f);
                var scaled = (ICompiledTrainingPlanLossScaling)plan;
                scaled.ConfigureLossScaling(scaling);
                for (int s = 0; s < steps; s++) plan.Step();
                return (w.ToArray(), scaled.ReadLossScaleState());
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    private void SkipUnlessAvailable(bool onGpu) => Skip.If(onGpu && !_fixture.IsAvailable, "No GPU device.");

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ScaledTraining_MatchesUnscaled_AndGrowsTheScale(bool onGpu)
    {
        SkipUnlessAvailable(onGpu);
        const int steps = 10;
        var (reference, none) = Train(onGpu, null, steps);
        Assert.Null(none);
        var options = new LossScalingOptions { InitialScale = 1024f, GrowthInterval = 4 };
        var (w, state) = Train(onGpu, options, steps);
        // A power-of-two scale and its reciprocal are exact: the scaled trajectory equals the unscaled one up to the
        // rounding of a few products (stated tolerance 1e-5 absolute on weights of order 1).
        for (int i = 0; i < w.Length; i++) Assert.Equal(reference[i], w[i], 5);
        var s = Assert.NotNull(state);
        Assert.Equal(4096f, s.Scale);          // grew at steps 4 and 8
        Assert.Equal(2, s.GrowthTracker);
        Assert.Equal(steps, s.Steps);
        Assert.Equal(0, s.OverflowSteps);
    }

    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonFiniteGradient_SkipsTheStep_AndBacksOff(bool onGpu)
    {
        SkipUnlessAvailable(onGpu);
        var c = Data(Rows * Out, 0.13, 0.7);
        c[5] = float.PositiveInfinity;
        var (initial, _) = Train(onGpu, null, 0, c);
        var (w, state) = Train(onGpu, new LossScalingOptions(), 3, c);
        Assert.Equal(initial, w);                                   // every step skipped
        var s = Assert.NotNull(state);
        Assert.Equal(65536f / 8f, s.Scale);                         // backed off three times
        Assert.Equal(3, s.OverflowSteps);
        Assert.Equal(3, s.Steps);
        Assert.Equal(0, s.GrowthTracker);
    }

    [SkippableTheory]
    [InlineData(PrecisionMode.Float16)]
    [InlineData(PrecisionMode.BFloat16)]
    public void MatMulAutocast_RoundsTheProduct_AccumulatesInFp32(PrecisionMode precision)
    {
        SkipUnlessAvailable(true);
        var engine = _fixture.Engine!;
        const int m = 64, k = 256, n = 64;
        var a = new Tensor<float>(new[] { m, k }, new Vector<float>(Data(m * k, 0.731, 0.2))).Gpu();
        var b = new Tensor<float>(new[] { k, n }, new Vector<float>(Data(k * n, 0.293, 0.5))).Gpu();
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var exact = engine.TensorMatMul(a, b).ToArray();
            float[] mixed;
            using (engine.EnterMatMulAutocast(precision)) mixed = engine.TensorMatMul(a, b).ToArray();
            // Elementwise rounding bound: each operand rounds to the reduced format with unit roundoff u (2^-11 FP16,
            // 2^-8 BF16), so a product term errs by at most (2u + u^2)|a||b|, and the FP32 accumulation over K adds at most
            // K * 2^-24 of the absolute sum: |mixed - exact| <= ((2u + u^2) + 2 K 2^-24) * sum_k |a_ik||b_kj|.
            var ha = a.ToArray();
            var hb = b.ToArray();
            double u = precision == PrecisionMode.Float16 ? Math.Pow(2, -11) : Math.Pow(2, -8);
            double factor = 2 * u + u * u + 2.0 * k * Math.Pow(2, -24);
            double any = 0, worst = 0;
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double abs = 0;
                    for (int kk = 0; kk < k; kk++) abs += Math.Abs(ha[i * k + kk] * (double)hb[kk * n + j]);
                    double d = Math.Abs(exact[i * n + j] - mixed[i * n + j]);
                    any += d;
                    worst = Math.Max(worst, d / (factor * abs));
                }
            Assert.True(worst <= 1.0, $"{precision}: an element exceeds the rounding bound by {worst:F2}x");
            Skip.If(engine.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.IGemmAutocastBackend,
                "The backend has no matmul autocast (FP32 GEMM is the documented fallback).");
            Assert.True(any > 0, $"{precision} autocast produced bit-identical FP32 results: the reduced precision did not run");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
