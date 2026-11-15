using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

public class ObjectiveAdversarialTests
{
    private static ObjectiveSignature Sign(object? value)
    {
        using var scope = GraphMode.EnableProbe();
        var x = new Tensor<float>(new[] { 1 });
        scope.RecordUnary(LazyNodeType.Custom, "CapturedValue", x, new[] { 1 },
            (e, o) => GC.KeepAlive(value));
        return ObjectiveSignature.Build(scope.Nodes, Array.Empty<Tensor<float>>(), out _);
    }

    [Fact]
    public void Null_is_distinct_from_the_literal_null_string()
    {
        var absent = Sign(null);
        var literal = Sign("null");
        Assert.True(absent.Signable, absent.UnsignableReason);
        Assert.True(literal.Signable, literal.UnsignableReason);
        Assert.NotEqual(absent, literal);
    }

    [Fact]
    public void Cyclic_and_deep_arrays_are_refused_without_recursing_forever()
    {
        var cycle = new object[1];
        cycle[0] = cycle;
        var signature = Sign(cycle);
        Assert.False(signature.Signable);
        Assert.Contains("cyclic", signature.UnsignableReason);
        object nested = 1;
        for (int i = 0; i < 20; i++) nested = new[] { nested };
        signature = Sign(nested);
        Assert.False(signature.Signable);
        Assert.Contains("deeper", signature.UnsignableReason);
    }

    [Fact]
    public void Array_geometry_and_lower_bounds_are_part_of_the_signature()
    {
        var row = Sign(new int[,] { { 1, 2 } });
        var column = Sign(new int[,] { { 1 }, { 2 } });
        Assert.True(row.Signable, row.UnsignableReason);
        Assert.True(column.Signable, column.UnsignableReason);
        Assert.NotEqual(row, column);
        var shifted = Array.CreateInstance(typeof(int), new[] { 2 }, new[] { 1 });
        shifted.SetValue(1, 1);
        shifted.SetValue(2, 2);
        Assert.NotEqual(Sign(new[] { 1, 2 }), Sign(shifted));
    }

    private static ObjectiveSignature ViewSignature(Tensor<float> w, bool useSecond)
    {
        var first = w.Slice(0, 0, 1);
        var second = w.Slice(0, 1, 2);
        var engine = new CpuEngine();
        using var scope = GraphMode.EnableProbe();
        var sum = engine.TensorAdd(first, second);
        engine.TensorMultiply(sum, useSecond ? second : first);
        return ObjectiveSignature.Build(scope.Nodes, new[] { w }, out _);
    }

    [Fact]
    public void Repeated_parameter_views_retain_their_individual_identity()
    {
        var w = new Tensor<float>(new[] { 2 });
        w.SetFlat(0, 2);
        w.SetFlat(1, 5);
        var first = ViewSignature(w, false);
        var second = ViewSignature(w, true);
        Assert.True(first.Signable, first.UnsignableReason);
        Assert.True(second.Signable, second.UnsignableReason);
        Assert.NotEqual(first, second);
        Assert.Equal(first, ViewSignature(w, false));
    }

    [Fact]
    public void Leaf_permutations_are_rejected_before_modifying_any_source()
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { 1 });
        var a = new Tensor<float>(new[] { 1 });
        var b = new Tensor<float>(new[] { 1 });
        w.SetFlat(0, 2);
        a.SetFlat(0, 3);
        b.SetFlat(0, 5);
        var parameters = new[] { w };
        bool swap = false;
        Tensor<float> Objective() => engine.ReduceSum(engine.TensorAdd(
            engine.TensorMultiply(swap ? b : a, w), swap ? a : b));
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            swap = true;
            Assert.Equal(ClosureGuardResult.Mismatched, guard!.TryRebind(Objective, parameters, out _));
            Assert.Equal(3f, a.GetFlat(0));
            Assert.Equal(5f, b.GetFlat(0));
            Assert.Contains("aliases", guard.LastReason);
        }
    }

    [Fact]
    public void Separate_external_leaves_sharing_storage_are_unsignable()
    {
        var engine = new CpuEngine();
        var x = new Tensor<float>(new[] { 2 });
        var first = x.Slice(0, 0, 1);
        var second = x.Slice(0, 1, 2);
        using var scope = GraphMode.EnableProbe();
        engine.TensorAdd(first, second);
        var signature = ObjectiveSignature.Build(scope.Nodes, Array.Empty<Tensor<float>>(), out _);
        Assert.False(signature.Signable);
        Assert.Contains("aliases another external leaf", signature.UnsignableReason);
    }

    [Fact]
    public void Identical_nodes_with_a_different_selected_loss_do_not_match()
    {
        var engine = new CpuEngine();
        var w = new Tensor<float>(new[] { 1 });
        w.SetFlat(0, 2);
        var parameters = new[] { w };
        bool second = false;
        Tensor<float> Objective()
        {
            var a = engine.ReduceSum(engine.TensorMultiplyScalar(w, 2f));
            var b = engine.ReduceSum(engine.TensorMultiplyScalar(w, 3f));
            return second ? b : a;
        }
        using var recording = ClosureObjectiveGuard<float>.BeginRecording();
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<float>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            Assert.Equal(ClosureGuardResult.Matched, guard!.TryRebind(Objective, parameters, out _));
            var replacement = new Tensor<float>(new[] { 1 });
            Assert.Equal(ClosureGuardResult.Mismatched, guard.TryRebind(Objective, new[] { replacement }, out _));
            Assert.Contains("parameters differ", guard.LastReason);
            second = true;
            Assert.Equal(ClosureGuardResult.Mismatched, guard.TryRebind(Objective, parameters, out _));
        }
    }
}
