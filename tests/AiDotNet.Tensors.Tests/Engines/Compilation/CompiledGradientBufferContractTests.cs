using System;
using System.Collections.Generic;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// The plan's clip, regularization and optimizer read <c>_gradients[p]</c>, the buffer configured at compile time. When
/// the backward replaces a parameter's live gradient-map entry instead of writing that buffer, the step commits the
/// replacement back into it (CommitReplacedParameterGradients); and <see cref="ICompiledTrainingPlan{T}.Gradients"/>
/// always hands out a copy of its array, so a caller cannot rebind an entry the step reads.
/// </summary>
[Collection("CompiledTrainingPlanSerial")]
public class CompiledGradientBufferContractTests
{
    private const BindingFlags Private = BindingFlags.NonPublic | BindingFlags.Instance;

    private static TField Field<TField>(object plan, string name) where TField : class
    {
        var field = plan.GetType().GetField(name, Private)
            ?? throw new InvalidOperationException($"CompiledTrainingPlan has no field {name}.");
        return field.GetValue(plan) as TField
            ?? throw new InvalidOperationException($"CompiledTrainingPlan.{name} is null or not a {typeof(TField).Name}.");
    }

    private static void Commit(object plan)
    {
        var method = plan.GetType().GetMethod("CommitReplacedParameterGradients", Private)
            ?? throw new InvalidOperationException("CompiledTrainingPlan has no CommitReplacedParameterGradients.");
        method.Invoke(plan, new object[] { new CpuEngine() });
    }
    private static ICompiledTrainingPlan<float> Compile(out Tensor<float> w)
    {
        var engine = new CpuEngine();
        w = new Tensor<float>(new float[] { 0.5f, -1f, 2f, 0.25f, -0.75f, 1.5f }, new[] { 2, 3 });
        using var scope = GraphMode.Enable();
        engine.ReduceSum(engine.TensorMultiply(w, w), null);
        return scope.CompileTraining(new[] { w });
    }

    [Fact]
    public void Gradients_IsACopyBeforeAndAfterTheFirstStep()
    {
        using var plan = Compile(out _);
        var first = plan.Gradients;
        Assert.NotSame(first, plan.Gradients);
        first[0] = new Tensor<float>(new[] { 2, 3 });
        Assert.NotSame(first[0], plan.Gradients[0]);

        plan.Step();
        var afterStep = plan.Gradients;
        Assert.NotSame(afterStep, plan.Gradients);
        var expected = new float[] { 1f, -2f, 4f, 0.5f, -1.5f, 3f };   // d(sum w^2)/dw = 2w
        Assert.Equal(expected, afterStep[0].ToArray());
    }

    [Fact]
    public void AReplacedLiveGradient_IsCommittedIntoThePlansOwnBuffer()
    {
        using var plan = Compile(out var w);
        plan.Step();
        var own = Field<Tensor<float>[]>(plan, "_gradients")[0];
        var live = Field<Dictionary<Tensor<float>, Tensor<float>>>(plan, "_liveGradientMap");

        var replacement = new Tensor<float>(new float[] { 9f, 8f, 7f, 6f, 5f, 4f }, new[] { 2, 3 });
        live[w] = replacement;
        Commit(plan);

        Assert.Equal(replacement.ToArray(), own.ToArray());
        Assert.Same(own, live[w]);
        Assert.Same(own, w.Grad);
        Assert.Same(own, plan.Gradients[0]);
    }

    [Fact]
    public void AReplacedLiveGradientOfTheWrongSize_Throws()
    {
        using var plan = Compile(out var w);
        plan.Step();
        var live = Field<Dictionary<Tensor<float>, Tensor<float>>>(plan, "_liveGradientMap");
        live[w] = new Tensor<float>(new[] { 4 });

        var ex = Assert.Throws<TargetInvocationException>(() => Commit(plan));
        Assert.IsType<InvalidOperationException>(ex.InnerException);
    }
}
