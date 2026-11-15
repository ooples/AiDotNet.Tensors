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

        // No tape: the custom backward is not even recorded. The read of the traced product is the first refusal.
        var (guard, reason) = CompileGuarded(() => HostLoss(engine, x, w), parameters);
        _out.WriteLine(reason);
        Assert.Null(guard);
        Assert.Contains("reads 'TensorMatMul' on the host while it is traced", reason);

        // Under a tape (an eager training step around the trace): the custom record itself is named.
        using (new GradientTape<float>())
        {
            (guard, reason) = CompileGuarded(() => HostLoss(engine, x, w), parameters);
            _out.WriteLine(reason);
            Assert.Null(guard);
            Assert.Contains("records 'HostSquaredSum' on the gradient tape", reason);
        }
    }

    // EmformerRNNT's shape: the host loss feeds a traced op (a mean over rows), so the loss is a traced node's output,
    // but that node's only input is a host leaf.
    [Fact]
    public void Traced_scaling_of_a_host_loss_records_no_guard()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 4, 7);
        var parameters = new[] { w };
        var host = new Tensor<float>(new[] { 1 });
        host[0] = 3f;
        var (guard, reason) = CompileGuarded(() => engine.TensorMultiplyScalar(host, 0.5f), parameters);
        Assert.Null(guard);
        Assert.Contains("reaches no trained parameter", reason);

        // The same through a view of the parameter does reach it.
        (guard, reason) = CompileGuarded(() => engine.ReduceSum(engine.Reshape(w, new[] { 24 })), parameters);
        Assert.True(guard is not null, reason);
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

    // Ops that compute values at trace time (CPU LayerNorm runs its forward as it records, reading its traced input)
    // are not host reads by the objective: a probe of the same objective traces from shapes alone.
    [Fact]
    public void LayerNorm_over_a_traced_input_still_records_a_guard()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 8, 9);
        var x = Filled(5, 6, 10);
        var gamma = new Tensor<float>(new[] { 8 });
        var beta = new Tensor<float>(new[] { 8 });
        for (int i = 0; i < 8; i++) gamma[i] = 1f;
        var parameters = new[] { w, gamma, beta };
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.LayerNorm(engine.TensorMatMul(x, w), gamma, beta, 1e-5, out _, out _);
            return engine.ReduceSum(engine.TensorMultiply(y, y));
        };
        var (guard, reason) = CompileGuarded(objective, parameters);
        Assert.True(guard is not null, reason);
        Assert.Equal(ClosureGuardResult.Matched, guard!.TryRebind(objective, parameters, out _));
    }

    // SeedTTSClone's RmsNorm: ReduceMean over the last axis of x*x. Its recorder computed the forward at record time,
    // which under a probe indexed the shape-only input (IndexOutOfRangeException) and aborted every step.
    [Fact]
    public void ReduceMean_over_a_traced_input_probes_and_matches()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 8, 11);
        var x = Filled(5, 6, 12);
        var parameters = new[] { w };
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.TensorMatMul(x, w);
            var meanSquare = engine.ReduceMean(engine.TensorMultiply(y, y), new[] { 1 }, keepDims: true);
            return engine.ReduceSum(meanSquare, null);
        };
        var (guard, reason) = CompileGuarded(objective, parameters);
        Assert.True(guard is not null, reason);
        var result = guard!.TryRebind(objective, parameters, out _);
        Assert.True(result == ClosureGuardResult.Matched, guard.LastReason);
    }

    // Recorders that copy an eager result into their own just-recorded output (BatchNorm on a non-4D input, GroupNorm)
    // write that node, not read a traced value: a Conformer's convolution module ([B, C, T] BatchNorm) and Wav2Vec2's
    // GroupNorm were declined as "reads 'BatchNorm' / 'GroupNorm' on the host while it is traced".
    [Fact]
    public void Normalizations_that_prefill_their_recorded_output_still_record_a_guard()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 12, 13);
        var x = Filled(4, 6, 14);
        var gamma = new Tensor<float>(new[] { 4 });   // a 3D BatchNorm input is [C, H, W]
        var beta = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) gamma[i] = 1f;
        var gamma4 = new Tensor<float>(new[] { 4 });
        var beta4 = new Tensor<float>(new[] { 4 });
        for (int i = 0; i < 4; i++) gamma4[i] = 1f;
        var parameters = new[] { w, gamma, beta, gamma4, beta4 };
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.Reshape(engine.TensorMatMul(x, w), new[] { 4, 3, 4 });       // [C, H, W], not 4D
            var b = engine.BatchNorm(y, gamma, beta, 1e-5, out _, out _);
            var g = engine.GroupNorm(engine.Reshape(b, new[] { 1, 4, 3, 4 }), 2, gamma4, beta4, 1e-5, out _, out _);
            return engine.ReduceSum(engine.TensorMultiply(g, g), null);
        };
        var (guard, reason) = CompileGuarded(objective, parameters);
        Assert.True(guard is not null, reason);
    }

    // The Conformer-transducer family's LSTM: the compiled LstmSequenceTrain node keeps its training workspace (the
    // transposed weights, gates and states its forward rewrites every step) in its closure. That is scratch, not
    // configuration: the objective was declined as holding "a LstmTrainWorkspace, whose value cannot be compared".
    [Fact]
    public void Lstm_training_workspace_is_scratch_to_the_signature()
    {
        var engine = new CpuEngine();
        int batch = 2, seq = 3, inF = 4, hidden = 5;
        var x = Filled(batch * seq, inF, 15);
        var proj = Filled(inF, inF, 16);
        var wIh = Filled(4 * hidden, inF, 17);
        var wHh = Filled(4 * hidden, hidden, 18);
        var bIh = new Tensor<float>(new[] { 4 * hidden });
        var bHh = new Tensor<float>(new[] { 4 * hidden });
        var parameters = new[] { proj, wIh, wHh, bIh, bHh };
        Func<Tensor<float>> objective = () =>
        {
            var input = engine.Reshape(engine.TensorMatMul(x, proj), new[] { batch, seq, inF });
            var h = engine.LstmSequenceForward(input, null, null, wIh, wHh, bIh, bHh, returnSequences: true);
            return engine.ReduceSum(engine.TensorMultiply(h, h), null);
        };
        var (guard, reason) = CompileGuarded(objective, parameters);
        Assert.True(guard is not null, reason);
        var result = guard!.TryRebind(objective, parameters, out _);
        Assert.True(result == ClosureGuardResult.Matched, guard.LastReason);
    }

    // The cache supplies MaskedFill's output shape without realizing the upstream LayerNorm on shape-only values.
    [Fact]
    public void A_probe_never_executes_a_node_an_eager_recorder_reads()
    {
        var engine = new CpuEngine();
        var w = Filled(6, 8, 19);
        var x = Filled(5, 6, 20);
        var gamma = new Tensor<float>(new[] { 8 });
        var beta = new Tensor<float>(new[] { 8 });
        for (int i = 0; i < 8; i++) gamma[i] = 1f;
        var mask = new bool[5 * 8];
        for (int i = 0; i < mask.Length; i += 3) mask[i] = true;
        var parameters = new[] { w, gamma, beta };
        Func<Tensor<float>> objective = () =>
        {
            var y = engine.LayerNorm(engine.TensorMatMul(x, w), gamma, beta, 1e-5, out _, out _);
            var masked = engine.TensorMaskedFill(y, mask, 0f);
            return engine.ReduceSum(engine.TensorMultiply(masked, masked), null);
        };
        var (guard, reason) = CompileGuarded(objective, parameters);
        Assert.True(guard is not null, reason);
        var result = guard!.TryRebind(objective, parameters, out _);
        _out.WriteLine($"{result}: {guard.LastReason}");
        Assert.True(result == ClosureGuardResult.Matched, guard.LastReason);
    }
}
