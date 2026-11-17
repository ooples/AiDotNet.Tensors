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
    [Fact]
    public void No_grad_parameters_are_absent_from_updates_even_with_nonzero_decoupled_weight_decay()
    {
        var engine = new CpuEngine();
        var frozen = new Tensor<float>(new[] { 2f }, new[] { 1 });
        var head = new Tensor<float>(new[] { 3f }, new[] { 1 });
        Tensor<float> Objective()
        {
            Tensor<float> data;
            using (new NoGradScope<float>()) data = engine.TensorMultiplyScalar(frozen, 2f);
            return engine.ReduceSum(engine.TensorMultiply(data, head), null, false);
        }
        using (var tape = new GradientTape<float>())
        {
            var gradients = tape.ComputeGradients(Objective());
            Assert.False(gradients.ContainsKey(frozen));
            Assert.Equal(4f, gradients[head].ToArray()[0]);
        }
        using var scope = GraphMode.EnableTraining(new[] { frozen, head });
        using var plan = scope.CompileTraining(new[] { frozen, head }, Objective());
        plan.ConfigureOptimizer(OptimizerType.AdamW, learningRate: 0.1, weightDecay: 0.2);
        Assert.Equal(12f, plan.Step().ToArray()[0]);
        Assert.Equal(2f, frozen.ToArray()[0]);
        Assert.Equal(2.84f, head.ToArray()[0], 5); // 3*(1 - .1*.2) - .1, with head gradient 4.
    }

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
    [InlineData(false, false)]
    [InlineData(true, false)] // matmul + 1-D bias add: the shape the training fusion pass rewrites to FusedLinear
    [InlineData(false, true)] // a GraphMode.Enable() (compatibility) trace compiled for training
    [InlineData(true, true)]
    public void CompiledPlan_GivesNoGradientToOpsRecordedUnderNoGrad_LikeTheEagerTape(bool withBias, bool compatibility)
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
        using var scope = compatibility ? GraphMode.Enable() : GraphMode.EnableTraining(parameters);
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

    // y = x @ W, then y += b in place under NoGradScope, then loss = sum(y * head). The eager tape records nothing for
    // the write: y keeps the matmul's history, so W gets x^T head (the write passes the gradient through unchanged), b
    // gets nothing, and head gets the written y. The compiled plan must reach the matmul through the write.
    private static Tensor<float> InPlaceLoss(IEngine engine, Tensor<float> x, Tensor<float> weight, Tensor<float> bias, Tensor<float> head)
    {
        var y = engine.TensorMatMul(x, weight);
        using (new NoGradScope<float>()) engine.TensorAddInPlace(y, bias);
        return engine.ReduceSum(engine.TensorMultiply(y, head), null);
    }

    private static Tensor<float> SquareBias() => new(new[] { 0.1f, -0.2f, 0.3f, 0.4f }, new[] { 2, 2 });

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledPlan_PassesGradientThroughAnInPlaceWriteRecordedUnderNoGrad_LikeTheEagerTape(bool compatibility)
    {
        var engine = new CpuEngine();
        var x = Input();

        var eagerWeight = Frozen();
        var eagerBias = SquareBias();
        var eagerHead = Head();
        Dictionary<Tensor<float>, Tensor<float>> eager;
        using (var tape = new GradientTape<float>())
            eager = tape.ComputeGradients(InPlaceLoss(engine, x, eagerWeight, eagerBias, eagerHead));
        Assert.True(eager.ContainsKey(eagerWeight));
        Assert.False(eager.ContainsKey(eagerBias));

        var weight = Frozen();
        var bias = SquareBias();
        var head = Head();
        var parameters = new[] { weight, bias, head };
        using var scope = compatibility ? GraphMode.Enable() : GraphMode.EnableTraining(parameters);
        var loss = InPlaceLoss(engine, x, weight, bias, head);
        using var plan = scope.CompileTraining(parameters, loss);
        plan.ConfigureOptimizer(OptimizerType.SGD, 0f, 0f, 0f, 0f, 0f);
        plan.Step();

        var compiled = Assert.IsType<CompiledTrainingPlan<float>>(plan);
        void AssertClose(float[] want, float[] actual)
        {
            Assert.Equal(want.Length, actual.Length);
            for (int i = 0; i < want.Length; i++) Assert.Equal(want[i], actual[i], 5);
        }
        AssertClose(eager[eagerWeight].ToArray(), compiled.Gradients[0].ToArray());
        Assert.All(compiled.Gradients[1].ToArray(), g => Assert.Equal(0f, g));
        AssertClose(eager[eagerHead].ToArray(), compiled.Gradients[2].ToArray());
    }
}
