using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A compiled plan replaying a closure objective that builds new per-step data each call: it must train on each
/// step's data, never the trace's, and must refuse to replay when anything the plan froze has changed.
/// </summary>
public class ClosureObjectiveGuardTests
{
    private const int Rows = 5, Inner = 6, Cols = 4;

    private static Tensor<float> Filled(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows * cols; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5));
        return t;
    }

    /// <summary>Eager reference for loss = sum((x W)^2) * scale.</summary>
    private static float ReferenceLoss(Tensor<float> x, Tensor<float> w, float scale)
    {
        double sum = 0;
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
            {
                double y = 0;
                for (int k = 0; k < Inner; k++) y += x[r, k] * w[k, c];
                sum += y * y;
            }
        return (float)(sum * scale);
    }

    /// <summary>One SGD step of loss = sum((x W)^2) on a host copy: W -= lr * 2 x^T (x W).</summary>
    private static void ReferenceSgd(double[,] w, Tensor<float> x, double lr)
    {
        var y = new double[Rows, Cols];
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                for (int k = 0; k < Inner; k++) y[r, c] += x[r, k] * w[k, c];
        for (int k = 0; k < Inner; k++)
            for (int c = 0; c < Cols; c++)
            {
                double g = 0;
                for (int r = 0; r < Rows; r++) g += 2 * x[r, k] * y[r, c];
                w[k, c] -= lr * g;
            }
    }

    private static (CompiledTrainingPlan<float> plan, ClosureObjectiveGuard<float> guard) Compile(
        Func<Tensor<float>> objective, Tensor<float>[] parameters)
    {
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = objective();
            plan = scope.CompileTraining(parameters, loss);
        }
        var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason)
                    ?? throw new InvalidOperationException($"no guard was recorded: {reason}");
        guard.AttachTo(plan);
        return (plan, guard);
    }

    [Fact]
    public void Rebound_plan_matches_eager_on_fresh_data_each_step()
    {
        var engine = new CpuEngine();
        var w = Filled(Inner, Cols, 1);
        var parameters = new[] { w };
        var hostW = new double[Inner, Cols];
        for (int k = 0; k < Inner; k++) for (int c = 0; c < Cols; c++) hostW[k, c] = w[k, c];
        const float lr = 0.05f;

        int step = 0;
        Tensor<float>? current = null;
        Func<Tensor<float>> objective = () =>
        {
            current = Filled(Rows, Inner, 100 + step);   // new data, new tensor object, every call
            var y = engine.TensorMatMul(current, w);
            return engine.ReduceSum(engine.TensorMultiply(y, y));
        };

        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: lr);
            for (step = 0; step < 5; step++)
            {
                var result = guard.TryRebind(objective, parameters, out _);
                Assert.True(result == ClosureGuardResult.Matched, guard.LastReason);
                var data = current ?? throw new InvalidOperationException("the probe did not run the objective");

                var expectedW = new Tensor<float>(new[] { Inner, Cols });
                for (int k = 0; k < Inner; k++) for (int c = 0; c < Cols; c++) expectedW[k, c] = (float)hostW[k, c];
                float expected = ReferenceLoss(data, expectedW, 1f);

                float loss = plan.Step().ToArray()[0];
                Assert.True(Math.Abs(loss - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                    $"step {step}: plan loss {loss}, eager loss on this step's data {expected}");
                ReferenceSgd(hostW, data, lr);
            }
        }

        for (int k = 0; k < Inner; k++)
            for (int c = 0; c < Cols; c++)
                Assert.True(Math.Abs(w[k, c] - hostW[k, c]) <= 1e-4, $"w[{k},{c}] = {w[k, c]}, eager {hostW[k, c]}");
    }

    [Fact]
    public void Captured_scalar_change_is_a_mismatch_and_never_replays()
    {
        var engine = new CpuEngine();
        var w = Filled(Inner, Cols, 2);
        var x = Filled(Rows, Inner, 3);
        var parameters = new[] { w };

        int step = 0;
        Func<Tensor<float>> objective = () =>
        {
            float a = 1 + step;   // a per-step coefficient, like a sampled diffusion timestep's alpha
            return engine.ReduceSum(engine.TensorMultiplyScalar(engine.TensorMatMul(x, w), a));
        };

        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            step = 0;
            Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(objective, parameters, out _));
            step = 1;
            var result = guard.TryRebind(objective, parameters, out var probed);
            Assert.Equal(ClosureGuardResult.Mismatched, result);
            Assert.NotNull(probed);
            Assert.NotEqual(guard.Signature, probed);
            Assert.Contains("TensorMultiplyScalar", guard.LastReason);
        }
    }

    [Fact]
    public void Same_leaf_objects_rebind_as_a_no_op()
    {
        var engine = new CpuEngine();
        var w = Filled(Inner, Cols, 4);
        var x = Filled(Rows, Inner, 5);   // one persistent buffer the caller refills in place
        var parameters = new[] { w };

        int step = 0;
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.TensorMatMul(x, w);
            return engine.ReduceSum(engine.TensorMultiply(y, y));
        };

        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0f);
            for (step = 0; step < 4; step++)
            {
                var fresh = Filled(Rows, Inner, 50 + step);
                for (int i = 0; i < x.Length; i++) x.SetFlat(i, fresh.GetFlat(i));

                Assert.Equal(ClosureGuardResult.Matched, guard.TryRebind(objective, parameters, out _));
                float loss = plan.Step().ToArray()[0];
                float expected = ReferenceLoss(x, w, 1f);
                Assert.True(Math.Abs(loss - expected) <= 1e-4 * Math.Max(1, Math.Abs(expected)),
                    $"step {step}: plan loss {loss}, eager {expected}");
            }
        }
    }

    [Fact]
    public void Host_read_closure_returns_aborted()
    {
        var engine = new CpuEngine();
        var w = Filled(Inner, Cols, 6);
        var x = Filled(Rows, Inner, 7);
        var parameters = new[] { w };
        bool read = false;

        Func<Tensor<float>> objective = () =>
        {
            var y = engine.TensorMatMul(x, w);
            if (read) _ = y.ToArray();   // logging an intermediate
            return engine.ReduceSum(engine.TensorMultiply(y, y));
        };

        var (plan, guard) = Compile(objective, parameters);
        using (plan)
        {
            read = true;
            Assert.Equal(ClosureGuardResult.Aborted, guard.TryRebind(objective, parameters, out var probed));
            Assert.Null(probed);
            Assert.Contains("TensorMatMul", guard.LastReason);
        }
    }
}
