using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public sealed class WeightNormSemanticReplayTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    public WeightNormSemanticReplayTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static void Verify<T>(IEngine engine, bool compiled, int selected)
    {
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var ops = MathHelper.GetNumericOperations<T>();
            const int rows = 3, columns = 257; // incomplete reduction tile
            const double epsilon = .03125, secondEpsilon = .125;
            var direction = new Tensor<T>(new[] { rows, 1, columns });
            var gain = new Tensor<T>(new[] { rows });
            var seed = new Tensor<T>(direction.Shape.ToArray());
            var secondSeed = new Tensor<T>(direction.Shape.ToArray());
            for (int i = 0; i < seed.Length; i++)
            {
                seed[i] = ops.FromDouble((i % 9 - 4) * .125);
                secondSeed[i] = ops.FromDouble((i % 7 - 3) * .0625);
            }
            var parameters = selected == 0 ? new[] { direction, gain } : selected == 1 ? new[] { direction } : new[] { gain };
            Tensor<T>? retained = null;
            Tensor<T> Objective()
            {
                retained = engine.TensorWeightNorm(direction, gain, epsilon);
                var second = engine.TensorWeightNorm(direction, gain, secondEpsilon);
                return engine.TensorAdd(engine.ReduceSum(engine.TensorMultiply(retained, seed)),
                    engine.ReduceSum(engine.TensorMultiply(second, secondSeed)));
            }
            void Feed(int step)
            {
                for (int i = 0; i < direction.Length; i++)
                    direction[i] = ops.FromDouble((i % 17 - 8) * .0625 + step * .03125);
                gain[0] = ops.FromDouble(.75 - step * .125);
                gain[1] = ops.FromDouble(step == 0 ? 0 : -.5);
                gain[2] = ops.FromDouble(-1.25 + step * .0625);
                if (engine.TryGet<DirectGpuTensorEngine>(out var gpu))
                {
                    gpu.EnsureResidentInput(direction); gpu.EnsureResidentInput(gain);
                    gpu.EnsureResidentInput(seed); gpu.EnsureResidentInput(secondSeed);
                }
            }
            Feed(0);
            ICompiledTrainingPlan<T>? plan = null;
            if (compiled)
            {
                using var graph = GraphMode.EnableTraining(parameters);
                var loss = Objective();
                Assert.NotNull(retained!.LazySource);
                Assert.Equal("TensorWeightNorm", retained.LazySource!.OpName);
                graph.RetainOutput(retained);
                plan = graph.CompileTraining(parameters, loss);
                Assert.IsType<CompiledTrainingPlan<T>>(plan).DisableGraphStep();
            }
            using (plan)
            {
                for (int step = 0; step < 3; step++)
                {
                    Feed(step);
                    using var tape = plan is null ? new GradientTape<T>() : null;
                    Tensor<T>[] actualGradients;
                    if (plan is not null) { plan.Step(); actualGradients = plan.Gradients; }
                    else
                    {
                        var gradients = tape!.ComputeGradients(Objective(), parameters);
                        actualGradients = new Tensor<T>[parameters.Length];
                        for (int p = 0; p < parameters.Length; p++) actualGradients[p] = gradients[parameters[p]];
                    }
                    var weights = retained!.ToArray();
                    var expectedDirection = new double[direction.Length];
                    var expectedGain = new double[rows];
                    double tolerance = typeof(T) == typeof(double) ? 3e-11 : 4e-5;
                    for (int row = 0; row < rows; row++)
                    {
                        double square = 0, dot = 0, secondDot = 0;
                        for (int j = 0; j < columns; j++)
                        {
                            int i = row * columns + j;
                            double v = ops.ToDouble(direction[i]);
                            square += v * v;
                            dot += ops.ToDouble(seed[i]) * v;
                            secondDot += ops.ToDouble(secondSeed[i]) * v;
                        }
                        double inv = 1 / Math.Sqrt(square + epsilon), inv2 = 1 / Math.Sqrt(square + secondEpsilon);
                        double g = ops.ToDouble(gain[row]);
                        expectedGain[row] = dot * inv + secondDot * inv2;
                        for (int j = 0; j < columns; j++)
                        {
                            int i = row * columns + j;
                            double v = ops.ToDouble(direction[i]);
                            double expectedWeight = v * (g * inv);
                            Assert.True(Math.Abs(ops.ToDouble(weights[i]) - expectedWeight) <= tolerance * (1 + Math.Abs(expectedWeight)),
                                $"step={step} retained weight[{i}] was overwritten or stale");
                            expectedDirection[i] = g * inv * (ops.ToDouble(seed[i]) - v * (dot * inv * inv))
                                + g * inv2 * (ops.ToDouble(secondSeed[i]) - v * (secondDot * inv2 * inv2));
                        }
                    }
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var expected = ReferenceEquals(parameters[p], direction) ? expectedDirection : expectedGain;
                        var actual = actualGradients[p].ToArray();
                        for (int i = 0; i < actual.Length; i++)
                            Assert.True(Math.Abs(ops.ToDouble(actual[i]) - expected[i]) <= tolerance * (1 + Math.Abs(expected[i])),
                                $"step={step} selected={selected} parameter={p} gradient[{i}] actual={ops.ToDouble(actual[i]):G17} expected={expected[i]:G17}");
                    }
                }
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Theory]
    [InlineData(false, 0)] [InlineData(true, 0)] [InlineData(false, 1)]
    [InlineData(true, 1)] [InlineData(false, 2)] [InlineData(true, 2)]
    public void Cpu_float_semantic_gradients_accumulate_and_honor_requested_sources(bool compiled, int selected)
        => Verify<float>(new CpuEngine(), compiled, selected);

    [Theory]
    [InlineData(false, 0)] [InlineData(true, 0)] [InlineData(true, 1)] [InlineData(true, 2)]
    public void Cpu_double_semantic_gradients_accumulate_and_honor_requested_sources(bool compiled, int selected)
        => Verify<double>(new CpuEngine(), compiled, selected);

    [SkippableTheory]
    [InlineData(false, 0)] [InlineData(true, 0)] [InlineData(false, 1)]
    [InlineData(true, 1)] [InlineData(false, 2)] [InlineData(true, 2)]
    public void Native_gpu_semantic_gradients_accumulate_and_honor_requested_sources(bool compiled, int selected)
    {
        Skip.IfNot(_fixture.IsAvailable, "No direct GPU backend.");
        Verify<float>(_fixture.Engine ?? throw new InvalidOperationException(), compiled, selected);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void Tied_direction_gain_accumulates_both_derivatives(bool compiled)
    {
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            var x = new Tensor<double>(new[] { .5, -.75, 1.25 }, new[] { 3 });
            using var graph = compiled ? GraphMode.EnableTraining(new[] { x }) : null;
            using var tape = compiled ? null : new GradientTape<double>();
            var loss = engine.ReduceSum(engine.TensorWeightNorm(x, x, .125));
            using var plan = graph?.CompileTraining(new[] { x }, loss);
            Tensor<double> gradient;
            if (plan is not null) { graph!.Dispose(); plan.Step(); gradient = plan.Gradients[0]; }
            else gradient = tape!.ComputeGradients(loss, new[] { x })[x];
            for (int i = 0; i < x.Length; i++)
            {
                double v = x[i], inv = 1 / Math.Sqrt(v * v + .125);
                double expected = 2 * v * inv - v * v * v * inv * inv * inv;
                Assert.InRange(Math.Abs(gradient[i] - expected), 0, 1e-11);
            }
        }
        finally { AiDotNetEngine.Current = previous; }
    }

    [Fact]
    public void WeightNorm_create_graph_preserves_mixed_second_derivative()
    {
        var engine = new CpuEngine();
        var previous = AiDotNetEngine.Current; AiDotNetEngine.Current = engine;
        try
        {
            var direction = new Tensor<double>(new[] { .5, -.75, 1.25 }, new[] { 1, 3 });
            var gain = new Tensor<double>(new[] { 1.5 }, new[] { 1 });
            using var outer = new GradientTape<double>();
            Tensor<double> first;
            using (var inner = new GradientTape<double>())
            {
                var loss = engine.ReduceSum(engine.TensorWeightNorm(direction, gain, .125));
                first = inner.ComputeGradients(loss, new[] { direction }, createGraph: true)[direction];
            }
            var second = outer.ComputeGradients(engine.ReduceSum(first), new[] { gain })[gain];
            double sum = .5 - .75 + 1.25, squares = .5 * .5 + .75 * .75 + 1.25 * 1.25;
            double inv = 1 / Math.Sqrt(squares + .125);
            double expected = 3 * inv - sum * sum * inv * inv * inv;
            Assert.InRange(Math.Abs(second[0] - expected), 0, 1e-10);
        }
        finally { AiDotNetEngine.Current = previous; }
    }
}
