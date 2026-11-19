using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class WeightNormOptimizerTrajectoryTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public WeightNormOptimizerTrajectoryTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static void Verify(IEngine engine)
    {
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            var initial = new[] { .5f, -.75f, .25f, 1.25f, -.375f, -.625f, .875f, .125f, -.25f, .75f };
            var scales = new[] { .75f, -1.25f };
            var h = new[] { .125f, -.25f, .5f, .375f, -.625f, -.5f, .25f, .625f, -.125f, .375f };
            var direction = new Tensor<float>((float[])initial.Clone(), new[] { 2, 5 });
            var gain = new Tensor<float>((float[])scales.Clone(), new[] { 2 });
            var seed = new Tensor<float>(h, new[] { 2, 5 });
            var parameters = new[] { direction, gain };
            ICompiledTrainingPlan<float> plan;
            using (var graph = GraphMode.EnableTraining(parameters))
                plan = graph.CompileTraining(parameters,
                    engine.ReduceSum(engine.TensorMultiply(engine.TensorWeightNorm(direction, gain, .03125), seed)));
            using (plan)
            {
                Assert.IsType<CompiledTrainingPlan<float>>(plan).DisableGraphStep();
                plan.ConfigureOptimizer(OptimizerType.Adam, .003, .9, .999, 1e-8);
                var values = new[] { initial, scales };
                var m = new[] { new double[10], new double[2] };
                var s = new[] { new double[10], new double[2] };
                for (int step = 1; step <= 5; step++)
                {
                    var gradients = new[] { new double[10], new double[2] };
                    double expectedLoss = 0;
                    for (int row = 0; row < 2; row++)
                    {
                        double square = 0, dot = 0;
                        for (int j = 0; j < 5; j++)
                        {
                            int i = row * 5 + j;
                            square += (double)values[0][i] * values[0][i];
                            dot += (double)h[i] * values[0][i];
                        }
                        double inv = 1 / Math.Sqrt(square + .03125);
                        expectedLoss += dot * values[1][row] * inv;
                        gradients[1][row] = dot * inv;
                        for (int j = 0; j < 5; j++)
                        {
                            int i = row * 5 + j;
                            gradients[0][i] = values[1][row] * inv * (h[i] - values[0][i] * (dot * inv * inv));
                        }
                    }
                    var loss = plan.Step();
                    Assert.InRange(Math.Abs(loss.ToArray()[0] - expectedLoss), 0, 2e-5);
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var actualGradients = plan.Gradients[p].ToArray();
                        for (int i = 0; i < actualGradients.Length; i++)
                        {
                            Assert.InRange(Math.Abs(actualGradients[i] - gradients[p][i]), 0, 2e-5);
                            m[p][i] = .9 * m[p][i] + .1 * gradients[p][i];
                            s[p][i] = .999 * s[p][i] + .001 * gradients[p][i] * gradients[p][i];
                            values[p][i] -= (float)(.003 * (m[p][i] / (1 - Math.Pow(.9, step)))
                                / (Math.Sqrt(s[p][i] / (1 - Math.Pow(.999, step))) + 1e-8));
                        }
                        var actual = parameters[p].ToArray();
                        for (int i = 0; i < actual.Length; i++)
                            Assert.True(Math.Abs(actual[i] - values[p][i]) < 3e-5,
                                $"Adam update={step} parameter={p} index={i}: actual={actual[i]:G9}, expected={values[p][i]:G9}");
                    }
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void Cpu_weight_norm_replay_preserves_five_Adam_updates() => Verify(new CpuEngine());

    [SkippableFact]
    public void Native_gpu_weight_norm_replay_preserves_five_Adam_updates()
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        Verify(_fixture.Engine ?? throw new InvalidOperationException());
    }

    [SkippableTheory] [InlineData(false)] [InlineData(true)]
    public void Native_gpu_tied_direction_gain_adds_both_derivatives(bool compiled)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        var engine = _fixture.Engine ?? throw new InvalidOperationException();
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            var x = new Tensor<float>(new[] { .5f, -.75f, 1.25f }, new[] { 3 });
            engine.EnsureResidentInput(x);
            ICompiledTrainingPlan<float>? plan = null;
            if (compiled)
            {
                using var graph = GraphMode.EnableTraining(new[] { x });
                plan = graph.CompileTraining(new[] { x }, engine.ReduceSum(engine.TensorWeightNorm(x, x, .125)));
                Assert.IsType<CompiledTrainingPlan<float>>(plan).DisableGraphStep();
            }
            using (plan)
            {
                for (int step = 0; step < 3; step++)
                {
                    Tensor<float> gradient;
                    if (plan is not null) { plan.Step(); gradient = plan.Gradients[0]; }
                    else
                    {
                        using var tape = new AiDotNet.Tensors.Engines.Autodiff.GradientTape<float>();
                        gradient = tape.ComputeGradients(engine.ReduceSum(engine.TensorWeightNorm(x, x, .125)), new[] { x })[x];
                    }
                    var actual = gradient.ToArray();
                    for (int i = 0; i < x.Length; i++)
                    {
                        double v = x[i], inv = 1 / Math.Sqrt(v * v + .125);
                        double expected = 2 * v * inv - v * v * v * inv * inv * inv;
                        Assert.InRange(Math.Abs(actual[i] - expected), 0, 2e-5);
                    }
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
