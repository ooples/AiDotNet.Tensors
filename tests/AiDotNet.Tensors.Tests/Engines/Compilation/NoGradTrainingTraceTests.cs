using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Ops a training forward runs inside a <see cref="NoGradScope{T}"/> must not propagate gradient through a compiled
/// training plan, exactly as they record nothing on the eager tape. The trace still records them (their values must
/// refresh on every replay); only the backward edge is dropped.
/// </summary>
public class NoGradTrainingTraceTests
{
    private static Tensor<float> Frozen() => new(new[] { 0.5f, -1f, 2f, 0.25f, 1.5f, -0.75f }, new[] { 3, 2 });
    private static Tensor<float> Bias() => new(new[] { 0.1f, -0.2f }, new[] { 2 });
    private static Tensor<float> Head() => new(new[] { 1f, -2f, 0.5f, 3f }, new[] { 2, 2 });
    private static Tensor<float> Input() => new(new[] { 1f, 2f, -1f, 0.5f, -0.5f, 3f }, new[] { 2, 3 });

    // loss = sum((x @ frozen [+ bias]) * head), with the frozen part computed under NoGradScope.
    private static Tensor<float> Loss(IEngine engine, Tensor<float> x, Tensor<float> frozen, Tensor<float>? bias, Tensor<float> head)
    {
        Tensor<float> memory;
        using (new NoGradScope<float>())
        {
            memory = engine.TensorMatMul(x, frozen);
            if (bias is not null) memory = engine.TensorAdd(memory, bias);
        }
        return engine.ReduceSum(engine.TensorMultiply(memory, head), null);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)] // matmul + 1-D bias add: the shape the training fusion pass rewrites to FusedLinear
    public void CompiledPlan_GivesNoGradientToOpsRecordedUnderNoGrad_LikeTheEagerTape(bool withBias)
    {
        var engine = new CpuEngine();
        var x = Input();

        // Eager reference: the tape sees only the head.
        var eagerFrozen = Frozen();
        var eagerBias = withBias ? Bias() : null;
        var eagerHead = Head();
        Dictionary<Tensor<float>, Tensor<float>> eager;
        using (var tape = new GradientTape<float>())
            eager = tape.ComputeGradients(Loss(engine, x, eagerFrozen, eagerBias, eagerHead));
        Assert.False(eager.ContainsKey(eagerFrozen));
        if (eagerBias is not null) Assert.False(eager.ContainsKey(eagerBias));

        var frozen = Frozen();
        var bias = withBias ? Bias() : null;
        var head = Head();
        var parameters = bias is null ? new[] { frozen, head } : new[] { frozen, bias, head };
        using var scope = GraphMode.EnableTraining(parameters);
        var loss = Loss(engine, x, frozen, bias, head);
        using var plan = scope.CompileTraining(parameters, loss);
        plan.ConfigureOptimizer(OptimizerType.Adam, 0.003f, 0.9f, 0.999f, 1e-8f, 0f);

        plan.Step();

        var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
        Assert.All(compiled.Gradients[0].ToArray(), g => Assert.Equal(0f, g));
        if (bias is not null) Assert.All(compiled.Gradients[1].ToArray(), g => Assert.Equal(0f, g));
        Assert.Equal(eager[eagerHead].ToArray(), compiled.Gradients[parameters.Length - 1].ToArray());

        // An Adam step on a zero gradient leaves the frozen weights exactly where they were.
        Assert.Equal(Frozen().ToArray(), frozen.ToArray());
        if (bias is not null) Assert.Equal(Bias().ToArray(), bias.ToArray());
        Assert.NotEqual(Head().ToArray(), head.ToArray());
    }

    [Fact]
    public void CompiledPlan_ReplaysOpsRecordedUnderNoGrad_WithTheirCurrentValues()
    {
        // Dropping the backward must not turn the no-grad region into a trace-time constant: a later step reads
        // whatever the frozen weights hold then (the head's gradient is x @ frozen).
        var engine = new CpuEngine();
        var x = Input();
        var frozen = Frozen();
        var head = Head();
        var parameters = new[] { head };
        using var scope = GraphMode.EnableTraining(parameters);
        var loss = Loss(engine, x, frozen, null, head);
        using var plan = scope.CompileTraining(parameters, loss);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0f, 0f, 0f, 0f, 0f);
        plan.Step();

        var doubled = Frozen().ToArray().Select(v => 2f * v).ToArray();
        doubled.CopyTo(frozen.AsWritableSpan());
        plan.Step();

        var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
        var expected = engine.TensorMatMul(x, new Tensor<float>(doubled, new[] { 3, 2 }));
        Assert.Equal(expected.ToArray(), compiled.Gradients[0].ToArray());
    }

    [Fact]
    public void CompiledPlan_StillDifferentiatesOpsRecordedOutsideTheNoGradScope()
    {
        // The same parameter used once under no-grad and once outside it gets exactly the outside use's gradient.
        var engine = new CpuEngine();
        var x = Input();
        var weight = Frozen();
        var parameters = new[] { weight };
        using var scope = GraphMode.EnableTraining(parameters);
        Tensor<float> detached;
        using (new NoGradScope<float>()) detached = engine.TensorMatMul(x, weight);
        var live = engine.TensorMatMul(x, weight);
        var loss = engine.ReduceSum(engine.TensorMultiply(detached, live), null);
        using var plan = scope.CompileTraining(parameters, loss);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0f, 0f, 0f, 0f, 0f);
        plan.Step();

        // d/dW sum(D * (xW)) with D held constant = x^T D.
        var d = engine.TensorMatMul(x, Frozen());
        var expected = engine.TensorMatMul(engine.TensorTranspose(x), d);
        var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
        var actual = compiled.Gradients[0].ToArray();
        var want = expected.ToArray();
        for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], actual[i], 4);
    }
}
