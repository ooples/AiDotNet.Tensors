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

    // A loss-scaled step on the GPU is captured into the step graph and replayed like an unscaled one: the unscale, the
    // finiteness check and the scale update are device kernels inside the graph. Two parameters (a bias as well as the
    // weight) and Adam, as a network's step has.
    [SkippableTheory]
    [InlineData(OptimizerType.SGD, PrecisionMode.Float32)]
    [InlineData(OptimizerType.Adam, PrecisionMode.Float32)]
    [InlineData(OptimizerType.Adam, PrecisionMode.Float16)]
    [InlineData(OptimizerType.Adam, PrecisionMode.BFloat16)]
    public void ScaledGpuStep_IsCapturedAndReplayed(OptimizerType optimizer, PrecisionMode autocast)
    {
        SkipUnlessAvailable(onGpu: true);
        var engine = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Data(In * Out, 0.37, 0.1))).Gpu();
            var bias = new Tensor<float>(new[] { Out }, new Vector<float>(Data(Out, 0.5, 0.2))).Gpu();
            var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(Data(Rows * In, 0.21, 0.3)));
            var ct = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(Data(Rows * Out, 0.13, 0.7)));
            ICompiledTrainingPlan<float> plan;
            // The step runs under the caller's autocast scope, as a mixed-precision training step does.
            using var precision = autocast == PrecisionMode.Float32 ? null : new AutocastScope(autocast);
            using (var scope = GraphMode.Enable())
            {
                var z = engine.Tanh(engine.FusedLinear(x, w, bias, FusedActivationType.None));
                engine.ReduceSum(engine.TensorMultiply(z, ct), null);
                plan = scope.CompileTraining(new[] { w, bias });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(optimizer, learningRate: 0.01f);
                ((ICompiledTrainingPlanLossScaling)plan).ConfigureLossScaling(new LossScalingOptions { InitialScale = 1024f });
                long before = CompiledTrainingPlan<float>.GraphReplayCount;
                for (int s = 0; s < 10; s++) plan.Step();
                long replays = CompiledTrainingPlan<float>.GraphReplayCount - before;
                var compiled = (CompiledTrainingPlan<float>)plan;
                Assert.True(replays > 0,
                    $"a loss-scaled GPU step never replayed its graph (capture failure: {compiled.GraphCaptureFailureReason ?? "(none)"})");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
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
        var engine = TestRequire.NotNull(_fixture.Engine, "the fixture's engine");
        const int m = 512, k = 1024, n = 512;   // large enough that cuBLAS takes tensor cores (FAST_* permits, not requires)
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
            // FAST_16F/_16BF permit the reduced precision rather than require it: cuBLAS may keep an FP32 kernel (it does
            // for BF16 at some shapes on GA102), which the bound above also admits. FP16 at this shape takes tensor cores.
            if (precision == PrecisionMode.Float16)
                Assert.True(any > 0, "FP16 autocast produced bit-identical FP32 results: the reduced precision did not run");
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// A captured step graph copies the device loss-gradient seed into the loss gradient in its prologue. A scale the
    /// host advanced (a step unscaled without a device scaler, the gradient-exchange path) was not written back to that
    /// seed, so the next replay scaled the gradients by the old scale while the unscale divided by the new one.
    /// </summary>
    [SkippableFact]
    public void A_host_side_scale_change_rewrites_the_device_seed_a_captured_graph_reads()
    {
        SkipUnlessAvailable(onGpu: true);
        var engine = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Data(In * Out, 0.37, 0.1))).Gpu();
            var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(Data(Rows * In, 0.21, 0.3)));
            var ct = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(Data(Rows * Out, 0.13, 0.7)));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReduceSum(engine.TensorMultiply(engine.Tanh(engine.TensorMatMul(x, w)), ct), null);
                plan = scope.CompileTraining(new[] { w });
            }
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
                ((ICompiledTrainingPlanLossScaling)plan).ConfigureLossScaling(new LossScalingOptions { InitialScale = 1024f });
                for (int s = 0; s < 3; s++) plan.Step();
                var compiled = (CompiledTrainingPlan<float>)plan;
                Skip.If(compiled.ReadDeviceLossSeedForTests() is null, "The loss-gradient seed is not on the device.");

                float backedOff = compiled.AdvanceHostLossScaleForTests(nonFinite: true);
                Assert.Equal(512f, backedOff);
                var seed = compiled.ReadDeviceLossSeedForTests()!;
                Assert.All(seed, v => Assert.Equal(backedOff, v));
            }
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    /// <summary>
    /// A teardown that fails after destroying the step graph must not leave its handle set: the plan's finalizer, armed
    /// because a stage failed, would destroy that graph a second time. The handles used to be cleared only after every
    /// destroy had returned.
    /// </summary>
    [SkippableFact]
    public void A_graph_teardown_that_fails_midway_leaves_no_destroyed_handle_behind()
    {
        SkipUnlessAvailable(onGpu: true);
        var engine = _fixture.Engine!;
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var w = new Tensor<float>(new[] { In, Out }, new Vector<float>(Data(In * Out, 0.37, 0.1))).Gpu();
            var x = new Tensor<float>(new[] { Rows, In }, new Vector<float>(Data(Rows * In, 0.21, 0.3)));
            var ct = new Tensor<float>(new[] { Rows, Out }, new Vector<float>(Data(Rows * Out, 0.13, 0.7)));
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                engine.ReduceSum(engine.TensorMultiply(engine.Tanh(engine.TensorMatMul(x, w)), ct), null);
                plan = scope.CompileTraining(new[] { w });
            }
            var compiled = (CompiledTrainingPlan<float>)plan;
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.01f);
            for (int s = 0; s < 6; s++) plan.Step();
            Skip.If(compiled.GraphHandlesForTests.Step == IntPtr.Zero,
                $"The step was not captured ({compiled.GraphCaptureFailureReason ?? "no reason given"}).");

            // The parts after the fault have something to do: the eviction suspension is held while the graph lives.
            Assert.True(compiled.GraphEvictionSuspendedForTests, "the captured step holds no eviction suspension");
            bool fired = false;
            compiled.TestFaultAfterStepGraphDestroy = () =>
            {
                if (fired) return;
                fired = true;
                throw new InvalidOperationException("injected destroy fault");
            };
            Assert.Throws<InvalidOperationException>(() => plan.Dispose());
            Assert.Equal(IntPtr.Zero, compiled.GraphHandlesForTests.Step);
            // The rest of the teardown still ran: the eviction suspension is balanced, rather than left on for the
            // engine shared by the later tests.
            Assert.False(compiled.GraphEvictionSuspendedForTests, "the graph-lifetime eviction suspension was left on");
            compiled.TestFaultAfterStepGraphDestroy = null;
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }
}
