using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// <see cref="WeightTiedLoop"/> on the CPU engine: the plain loop against the same body applied by hand, the adaptive
/// loop's exit distribution, entropy, expected loops and masked exit against their closed forms for a constant halting
/// logit, the argument and body-contract checks, and a compiled training step against the eager tape without
/// per-iteration checkpointing (checkpoint segments compile only on a DirectGpu engine; CompiledWeightTiedLoopTests covers them).
/// </summary>
public sealed class WeightTiedLoopCpuTests
{
    private static readonly CpuEngine Engine = new();

    private static Tensor<float> Filled(int[] shape, int seed, float scale, float offset = 0f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = offset + (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static Tensor<float> Halve(Tensor<float> h, int _) => Engine.TensorMultiplyScalar(h, 0.5f);

    // A halting head whose logit is the constant c for every token: z = sum(h * 0, axis 1) + c, shape [rows, 1].
    private static Func<Tensor<float>, Tensor<float>> ConstantLogit(float c)
        => h => Engine.TensorAddScalar(Engine.ReduceSum(Engine.TensorMultiplyScalar(h, 0f), new[] { 1 }, true), c);

    [Fact]
    public void Run_applies_the_body_once_per_iteration_with_the_iteration_number()
    {
        var x = Filled(new[] { 3, 4 }, 1, 2f);
        var seen = new List<int>();
        var y = WeightTiedLoop.Run(Engine, x, 3, (h, i) => { seen.Add(i); return Engine.TensorAddScalar(h, i + 1f); });
        Assert.Equal(new[] { 0, 1, 2 }, seen);
        var xs = x.ToArray();
        var ys = y.ToArray();
        for (int i = 0; i < xs.Length; i++) Assert.Equal(xs[i] + 6f, ys[i], 5);

        // Eagerly the checkpoint flag changes nothing.
        var yc = WeightTiedLoop.Run(Engine, x, 3, (h, i) => Engine.TensorAddScalar(h, i + 1f), checkpointEachIteration: true);
        Assert.Equal(ys, yc.ToArray());
    }

    [Fact]
    public void Run_and_RunAdaptive_reject_invalid_arguments_and_a_body_that_breaks_the_contract()
    {
        var x = Filled(new[] { 2, 3 }, 2, 1f);
        Func<Tensor<float>, int, Tensor<float>> body = Halve;
        var halt = ConstantLogit(0f);
        Assert.Throws<ArgumentNullException>(() => WeightTiedLoop.Run<float>(null!, x, 1, body));
        Assert.Throws<ArgumentNullException>(() => WeightTiedLoop.Run(Engine, null!, 1, body));
        Assert.Throws<ArgumentNullException>(() => WeightTiedLoop.Run(Engine, x, 1, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => WeightTiedLoop.Run(Engine, x, 0, body));
        Assert.Throws<InvalidOperationException>(() => WeightTiedLoop.Run(Engine, x, 2, (h, i) => null!));
        Assert.Throws<InvalidOperationException>(() =>
            WeightTiedLoop.Run(Engine, x, 2, (h, i) => Engine.Reshape(h, new[] { 3, 2 })));

        Assert.Throws<ArgumentNullException>(() => WeightTiedLoop.RunAdaptive<float>(null!, x, 2, body, halt));
        Assert.Throws<ArgumentNullException>(() => WeightTiedLoop.RunAdaptive(Engine, null!, 2, body, halt));
        Assert.Throws<ArgumentNullException>(() => WeightTiedLoop.RunAdaptive(Engine, x, 2, null!, halt));
        Assert.Throws<ArgumentNullException>(() => WeightTiedLoop.RunAdaptive(Engine, x, 2, body, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => WeightTiedLoop.RunAdaptive(Engine, x, 0, body, halt));
        Assert.Throws<ArgumentOutOfRangeException>(() => WeightTiedLoop.RunAdaptive(Engine, x, 2, body, halt, exitThreshold: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => WeightTiedLoop.RunAdaptive(Engine, x, 2, body, halt, exitThreshold: double.NaN));
    }

    /// <summary>
    /// A constant logit c gives lambda = sigmoid(c) every iteration, so for T = 3: p = (l, (1-l) l, (1-l)^2), the entropy
    /// is -sum p log p and the expected loop count sum t p_t. Without a mask every iteration runs.
    /// </summary>
    [Theory]
    [InlineData(0f)]
    [InlineData(1.5f)]
    [InlineData(-2f)]
    public void RunAdaptive_exit_distribution_entropy_and_expected_loops_match_the_closed_form(float c)
    {
        var x = Filled(new[] { 4, 3 }, 3, 2f);
        var r = WeightTiedLoop.RunAdaptive(Engine, x, 3, Halve, ConstantLogit(c));
        double l = 1.0 / (1.0 + Math.Exp(-c));
        var p = new[] { l, (1 - l) * l, (1 - l) * (1 - l) };
        Assert.Equal(3, r.ExitProbabilities.Count);
        Assert.Equal(3, r.States.Count);
        for (int t = 0; t < 3; t++)
            foreach (var v in r.ExitProbabilities[t].ToArray()) Assert.Equal(p[t], v, 5);
        Assert.Equal(-p.Sum(q => q * Math.Log(q)), r.Entropy.ToArray()[0], 4);
        Assert.Equal(p[0] + 2 * p[1] + 3 * p[2], r.ExpectedLoops.ToArray()[0], 4);
        Assert.Equal(3f, r.LoopsRun.ToArray()[0], 5);
        var xs = x.ToArray();
        var outs = r.Output.ToArray();
        for (int i = 0; i < xs.Length; i++) Assert.Equal(xs[i] * 0.125f, outs[i], 5);

        // ExpectedLoss: sum_t p_t L_t, here with L_t = t + 1 for every token.
        var losses = Enumerable.Range(0, 3).Select(t => Engine.TensorAddScalar(new Tensor<float>(new[] { 4 }), t + 1f)).ToArray();
        Assert.Equal(p[0] + 2 * p[1] + 3 * p[2], r.ExpectedLoss(losses).ToArray()[0], 4);
    }

    /// <summary>
    /// lambda = 0.5: the survival after iteration 1 is 0.5 and after iteration 2 is 0.25. A threshold of 0.4 stops every
    /// token after one iteration (0.5 is not above 0.6), 0.6 after two (0.5 &gt; 0.4, 0.25 is not).
    /// </summary>
    [Theory]
    [InlineData(0.4, 1)]
    [InlineData(0.6, 2)]
    [InlineData(0.99, 3)]
    public void RunAdaptive_masked_exit_keeps_each_tokens_state_from_the_iteration_it_exited(double threshold, int loops)
    {
        var x = Filled(new[] { 2, 5 }, 4, 2f);
        var r = WeightTiedLoop.RunAdaptive(Engine, x, 3, Halve, ConstantLogit(0f), threshold);
        Assert.Equal(loops, r.LoopsRun.ToArray()[0], 5);
        float scale = (float)Math.Pow(0.5, loops);
        var xs = x.ToArray();
        var outs = r.Output.ToArray();
        for (int i = 0; i < xs.Length; i++) Assert.Equal(xs[i] * scale, outs[i], 5);
    }

    [Fact]
    public void RunAdaptive_with_one_iteration_puts_all_exit_mass_on_it()
    {
        var x = Filled(new[] { 3, 2 }, 5, 1f);
        var r = WeightTiedLoop.RunAdaptive(Engine, x, 1, Halve, ConstantLogit(0.7f), checkpointEachIteration: true);
        Assert.All(r.ExitProbabilities[0].ToArray(), v => Assert.Equal(1f, v, 6));
        Assert.Equal(0f, r.Entropy.ToArray()[0], 6);
        Assert.Equal(1f, r.ExpectedLoops.ToArray()[0], 6);
        Assert.Equal(1f, r.LoopsRun.ToArray()[0], 6);
    }

    [Fact]
    public void ExpectedLoss_rejects_a_loss_list_that_does_not_match_the_loop()
    {
        var x = Filled(new[] { 2, 3 }, 6, 1f);
        var r = WeightTiedLoop.RunAdaptive(Engine, x, 2, Halve, ConstantLogit(0f));
        Assert.Throws<ArgumentNullException>(() => r.ExpectedLoss(null!));
        Assert.Throws<ArgumentException>(() => r.ExpectedLoss(new[] { new Tensor<float>(new[] { 2 }) }));
        Assert.Throws<ArgumentException>(() => r.ExpectedLoss(new[] { new Tensor<float>(new[] { 2 }), new Tensor<float>(new[] { 3 }) }));
    }

    // h -> h + 0.1 * silu((h * e_i) W1) W2 with a shared W1, W2 and a per-iteration scale e_i.
    private static Tensor<float> Body(IEngine e, Tensor<float>[] p, Tensor<float> h, int i)
    {
        var inner = e.TensorMatMul(e.TensorSiLU(e.TensorMatMul(e.TensorMultiply(h, p[2 + i]), p[0])), p[1]);
        return e.TensorAdd(h, e.TensorMultiplyScalar(inner, 0.1f));
    }

    private static Tensor<float>[] Parameters(int width, int hidden, int iterations)
    {
        var p = new List<Tensor<float>> { Filled(new[] { width, hidden }, 10, 1f), Filled(new[] { hidden, width }, 11, 1f) };
        for (int i = 0; i < iterations; i++) p.Add(Filled(new[] { 1, width }, 20 + i, 0.4f, 1f));
        return p.ToArray();
    }

    private static Tensor<float> Loss(IEngine e, Tensor<float>[] p, Tensor<float> x, int iterations, bool checkpoint)
    {
        var h = WeightTiedLoop.Run(e, x, iterations, (s, i) => Body(e, p, s, i), checkpoint);
        return e.ReduceSum(e.TensorMultiply(h, h), null);
    }

    /// <summary>
    /// The compiled CPU step gives the eager tape's loss and gradients: the shared block's gradient summed over every
    /// iteration, each scale's from its own iteration.
    /// </summary>
    [Fact]
    public void A_compiled_cpu_step_over_the_loop_matches_the_eager_tape()
    {
        const int Iterations = 3;
        const bool checkpoint = false;
        var x = Filled(new[] { 4, 6 }, 1, 2f);

        var eagerParams = Parameters(6, 8, Iterations);
        float eagerLoss;
        float[][] eagerGrads;
        using (var tape = new GradientTape<float>())
        {
            var loss = Loss(Engine, eagerParams, x, Iterations, checkpoint);
            var grads = tape.ComputeGradients(loss, sources: eagerParams);
            eagerLoss = loss.ToArray()[0];
            eagerGrads = eagerParams.Select(t => grads[t].ToArray()).ToArray();
        }

        var parameters = Parameters(6, 8, Iterations);
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = Loss(AiDotNetEngine.Current, parameters, x, Iterations, checkpoint);
            plan = (CompiledTrainingPlan<float>)scope.CompileTraining(parameters, loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0f);
            for (int step = 0; step < 2; step++)
            {
                float compiledLoss = plan.Step().ToArray()[0];
                Assert.True(Math.Abs(compiledLoss - eagerLoss) <= 1e-4f * (1 + Math.Abs(eagerLoss)),
                    $"step {step}: compiled loss {compiledLoss} vs eager {eagerLoss}");
                var grads = plan.Gradients.Select(g => g.ToArray()).ToArray();
                for (int k = 0; k < eagerGrads.Length; k++)
                    for (int j = 0; j < eagerGrads[k].Length; j++)
                        Assert.True(Math.Abs(grads[k][j] - eagerGrads[k][j]) <= 1e-4f * (1 + Math.Abs(eagerGrads[k][j])),
                            $"step {step}, parameter {k}[{j}]: {grads[k][j]} vs {eagerGrads[k][j]}");
            }
            Assert.Equal(0, plan.CheckpointSegmentCount);
            for (int i = 1; i < Iterations; i++)
                Assert.NotEqual(eagerGrads[2], eagerGrads[2 + i]);
        }
    }
}
