using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// An objective that reads its model output on the host, computes the loss there and records a custom backward with
/// DifferentiableOps.RecordUnary (AiDotNet's TransducerLoss): the loss is a host tensor no recorded op produces, and its
/// backward is a closure over this step's host arrays. A compiled plan cannot replay that, so the closure guard must not
/// let such an objective commit a plan.
/// </summary>
public class ClosureObjectiveHostLossTests
{
    private readonly ITestOutputHelper _out;
    public ClosureObjectiveHostLossTests(ITestOutputHelper output) => _out = output;

    private static Tensor<float> Filled(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows * cols; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5));
        return t;
    }

    /// <summary>loss = sum(y^2) for y = x W, computed on the host, with d loss / d y = 2 y recorded by hand.</summary>
    private static Tensor<float> HostLoss(IEngine engine, Tensor<float> x, Tensor<float> w)
    {
        var y = engine.TensorMatMul(x, w);
        var values = y.ToArray();
        double sum = 0;
        foreach (var v in values) sum += v * v;
        var loss = new Tensor<float>(new[] { 1 });
        loss[0] = (float)sum;
        DifferentiableOps.RecordUnary("HostSquaredSum", loss, y, (gradOutput, inputs, _, _, eng, grads) =>
        {
            float scale = gradOutput.ToArray()[0];
            var grad = new Tensor<float>((int[])inputs[0]._shape.Clone());
            for (int i = 0; i < values.Length; i++) grad.SetFlat(i, 2f * scale * values[i]);
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grad, eng);
        });
        return loss;
    }

    private static (ClosureObjectiveGuard<float>? guard, string? reason) CompileGuarded(Func<Tensor<float>> objective,
        Tensor<float>[] parameters)
    {
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var loss = objective();
            using var plan = scope.CompileTraining(parameters, loss);
        }
        var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
        return (guard, reason);
    }

    [Fact]
    public void Host_loss_with_a_recorded_backward_records_no_guard()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 4, 1);
        var x = Filled(5, 6, 2);
        var parameters = new[] { w };

        // No tape: the custom backward is not even recorded; the loss is still a host value no traced op produced.
        var (guard, reason) = CompileGuarded(() => HostLoss(engine, x, w), parameters);
        _out.WriteLine(reason);
        Assert.Null(guard);
        Assert.Contains("loss is not computed by a traced op", reason);

        // Under a tape (an eager training step around the trace): the custom record itself is named.
        using (new GradientTape<float>())
        {
            (guard, reason) = CompileGuarded(() => HostLoss(engine, x, w), parameters);
            _out.WriteLine(reason);
            Assert.Null(guard);
            Assert.Contains("records 'HostSquaredSum' on the gradient tape", reason);
        }
    }

    [Fact]
    public void Probe_of_a_host_loss_is_never_matched()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 4, 3);
        var x = Filled(5, 6, 4);
        var parameters = new[] { w };
        bool host = false;
        Func<Tensor<float>> objective = () =>
        {
            if (host) return HostLoss(engine, x, w);
            var y = engine.TensorMatMul(x, w);
            return engine.ReduceSum(engine.TensorMultiply(y, y));
        };
        var (guard, reason) = CompileGuarded(objective, parameters);
        Assert.True(guard is not null, reason);

        host = true;
        var result = guard!.TryRebind(objective, parameters, out _);
        Assert.NotEqual(ClosureGuardResult.Matched, result);
        using (new GradientTape<float>())
        {
            result = guard.TryRebind(objective, parameters, out _);
            Assert.NotEqual(ClosureGuardResult.Matched, result);
        }

        // A loss that is a constant host tensor reads nothing on the host, yet no traced op produces it.
        var constant = new Tensor<float>(new[] { 1 });
        Assert.Equal(ClosureGuardResult.Unsignable, guard.TryRebind(() => constant, parameters, out _));
        Assert.Contains("loss is not computed by a traced op", guard.LastReason);
    }

    [Fact]
    public void A_tape_around_an_ordinary_trace_still_records_a_guard()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 4, 5);
        var x = Filled(5, 6, 6);
        var parameters = new[] { w };
        using (new GradientTape<float>())
        {
            var (guard, reason) = CompileGuarded(() =>
            {
                var y = engine.Tanh(engine.TensorMatMul(x, w));
                return engine.ReduceSum(engine.TensorMultiply(y, y));
            }, parameters);
            Assert.True(guard is not null, reason);
        }
    }
}
