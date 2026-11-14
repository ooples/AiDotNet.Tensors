using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The guard's recording around CompileTraining and its refusals: a recording yields a guard only for exactly one
/// signable compilation, and every way a step's probe can fail to match reports a reason instead of replaying.
/// </summary>
public class ClosureObjectiveGuardRecordingTests
{
    private static Tensor<float> Filled(int rows, int cols, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { rows, cols });
        for (int i = 0; i < rows * cols; i++) t.SetFlat(i, (float)(rng.NextDouble() - 0.5));
        return t;
    }

    private static CompiledTrainingPlan<float> CompileOnce(Func<Tensor<float>> objective, Tensor<float>[] parameters)
    {
        using var scope = GraphMode.EnableTraining(parameters);
        var loss = objective();
        return scope.CompileTraining(parameters, loss);
    }

    private static (CompiledTrainingPlan<float> plan, ClosureObjectiveGuard<float> guard) Compile(
        Func<Tensor<float>> objective, Tensor<float>[] parameters)
    {
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        var plan = CompileOnce(objective, parameters);
        var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason)
                    ?? throw new InvalidOperationException($"no guard was recorded: {reason}");
        guard.AttachTo(plan);
        return (plan, guard);
    }

    [Fact]
    public void Take_without_an_open_recording_reports_it()
    {
        Assert.Null(ClosureObjectiveGuard<float>.TakeRecorded(out var reason));
        Assert.Equal("no recording is open", reason);
    }

    [Fact]
    public void Recording_with_no_compilation_yields_no_guard()
    {
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        Assert.Null(ClosureObjectiveGuard<float>.TakeRecorded(out var reason));
        Assert.Contains("no training plan compiled", reason);
    }

    [Fact]
    public void Two_compilations_in_one_recording_yield_no_guard()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 3, 1);
        var x = Filled(2, 4, 2);
        var parameters = new[] { w };

        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        CompileOnce(() => engine.ReduceSum(engine.TensorMatMul(x, w)), parameters);
        CompileOnce(() => engine.ReduceSum(engine.TensorMatMul(x, w)), parameters);

        Assert.Null(ClosureObjectiveGuard<float>.TakeRecorded(out var reason));
        Assert.Contains("more than one", reason);
    }

    [Fact]
    public void Nested_recordings_restore_the_outer_one_and_dispose_is_idempotent()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 3, 1);
        var x = Filled(2, 4, 2);
        var parameters = new[] { w };

        using var outer = ClosureObjectiveGuard<float>.BeginRecording();
        var inner = ClosureObjectiveGuard<float>.BeginRecording();
        CompileOnce(() => engine.ReduceSum(engine.TensorMatMul(x, w)), parameters);
        Assert.NotNull(ClosureObjectiveGuard<float>.TakeRecorded(out _));
        inner.Dispose();
        inner.Dispose();

        // The inner compilation was not the outer recording's.
        Assert.Null(ClosureObjectiveGuard<float>.TakeRecorded(out var reason));
        Assert.Contains("no training plan compiled", reason);
    }

    [Fact]
    public void Probe_of_an_objective_that_throws_is_aborted_with_the_cause()
    {
        var w = Filled(4, 3, 1);
        var result = ClosureObjectiveGuard<float>.Probe(
            () => throw new InvalidOperationException("batch source exhausted"), new[] { w },
            out var signature, out var leaves, out var reason);

        Assert.Equal(ClosureGuardResult.Aborted, result);
        Assert.Null(signature);
        Assert.Empty(leaves);
        Assert.Contains("batch source exhausted", reason);
    }

    [Fact]
    public void Rebind_reports_a_mismatch_when_the_step_changes_shape()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 3, 1);
        var parameters = new[] { w };
        int rows = 2;
        var (_, guard) = Compile(() => engine.ReduceSum(engine.TensorMatMul(Filled(rows, 4, rows), w)), parameters);

        rows = 3;
        var result = guard.TryRebind(() => engine.ReduceSum(engine.TensorMatMul(Filled(rows, 4, rows), w)), parameters, out var probed);

        Assert.Equal(ClosureGuardResult.Mismatched, result);
        Assert.NotNull(probed);
        Assert.NotNull(guard.LastReason);
        Assert.Equal(1, guard.LeafCount);
        Assert.False(guard.HasForeignLeaves);
    }

    [Fact]
    public void Rebind_reports_an_unsignable_step()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 3, 1);
        var x = Filled(2, 4, 2);
        var parameters = new[] { w };
        var (_, guard) = Compile(() => engine.ReduceSum(engine.TensorMatMul(x, w)), parameters);
        var opaque = new List<int>();

        var result = guard.TryRebind(() =>
        {
            var scope = GraphMode.Current!;
            return scope.RecordUnary(LazyNodeType.Custom, "OpaqueStateOp", x, new[] { 2, 4 },
                (e, o) => o.SetFlat(0, opaque.Count));
        }, parameters, out _);

        Assert.Equal(ClosureGuardResult.Unsignable, result);
        Assert.Contains("OpaqueStateOp", guard.LastReason);
    }

    [Fact]
    public void Leaves_that_cannot_be_copied_are_refused_with_a_reason()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 3, 1);
        var x = Filled(2, 4, 2);
        var (_, guard) = Compile(() => engine.ReduceSum(engine.TensorMatMul(x, w)), new[] { w });

        Assert.False(guard.TryCopyLeaves(Array.Empty<object>(), out var countReason));
        Assert.Contains("0 leaves", countReason);

        Assert.False(guard.TryCopyLeaves(new object[] { "not a tensor" }, out var typeReason));
        Assert.Contains("not a tensor", typeReason);

        Assert.False(guard.TryCopyLeaves(new object[] { new Tensor<double>(new[] { 2, 4 }) }, out var dtypeReason));
        Assert.Contains("can no longer be written", dtypeReason);

        Assert.False(guard.TryCopyLeaves(new object[] { Filled(3, 4, 3) }, out var lengthReason));
        Assert.Contains("can no longer be written", lengthReason);

        Assert.True(guard.TryCopyLeaves(new object[] { x }, out var sameReason));   // the compiled leaf itself
        Assert.Null(sameReason);
    }

    [Fact]
    public void Null_arguments_are_rejected()
    {
        var engine = new CpuEngine();
        var w = Filled(4, 3, 1);
        var x = Filled(2, 4, 2);
        var (_, guard) = Compile(() => engine.ReduceSum(engine.TensorMatMul(x, w)), new[] { w });

        Assert.Throws<ArgumentNullException>(() => guard.AttachTo(null!));
        Assert.Throws<ArgumentNullException>(() => guard.TryCopyLeaves(null!, out _));
        Assert.Throws<ArgumentNullException>(() => ClosureObjectiveGuard<float>.Probe(null!, new[] { w }, out _, out _, out _));
        Assert.Throws<ArgumentNullException>(() => ClosureObjectiveGuard<float>.Probe(() => x, null!, out _, out _, out _));
    }
}
