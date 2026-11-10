using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// #1096 known follow-ups: the fused LSTM node's workspace order contract, and the opt-in input gradient of compiled
/// training plans (saliency maps and adversarial examples read the network input's gradient, which plans skip by
/// default because training never reads it).
/// </summary>
[Collection("CompilationGlobalState")]
public class Issue1096FollowUpTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    // The saved state belongs to the most recent completed forward. A backward before any forward, or one overlapping a
    // forward (or another backward) on the same workspace, would read state that is not this step's: it throws instead.
    [Fact]
    public void LstmWorkspace_RejectsBackwardWithoutACompletedForward_AndOverlap()
    {
        var ws = new CpuEngine.LstmTrainWorkspace(batch: 2, seqLen: 3, inFeatures: 4, hidden: 5);

        var never = Assert.Throws<InvalidOperationException>(() => ws.BeginBackward());
        Assert.Contains("never filled", never.Message);

        ws.BeginForward();
        Assert.Throws<InvalidOperationException>(() => ws.BeginForward());      // two forwards at once
        Assert.Throws<InvalidOperationException>(() => ws.BeginBackward());     // backward during a forward
        ws.EndForward();
        Assert.Equal(1, ws.CompletedForwards);

        ws.BeginBackward();
        Assert.Throws<InvalidOperationException>(() => ws.BeginForward());      // forward during a backward
        ws.EndBackward();
        // Repeated backwards of one forward are exact (the backward only overwrites its Wx scratch).
        ws.BeginBackward();
        ws.EndBackward();

        // A forward that threw part-way leaves partial state, which must not be backpropagated.
        ws.BeginForward();
        ws.AbortForward();
        var partial = Assert.Throws<InvalidOperationException>(() => ws.BeginBackward());
        Assert.Contains("partial", partial.Message);
        Assert.Equal(1, ws.CompletedForwards);

        // A completed forward makes the state valid again.
        ws.BeginForward();
        ws.EndForward();
        ws.BeginBackward();
        ws.EndBackward();
        Assert.Equal(2, ws.CompletedForwards);
    }

    // The forward wrapper itself must abort the workspace when the arithmetic throws, not only a caller that aborts it
    // by hand: an output span too short for the hidden outputs makes the core throw part-way through the recurrence.
    [Fact]
    public void LstmForward_ThatThrows_LeavesTheWorkspaceUnbackpropagatable_UntilAForwardCompletes()
    {
        const int batch = 2, seqLen = 3, inFeatures = 4, hidden = 5, G = 4 * hidden;
        var ws = new CpuEngine.LstmTrainWorkspace(batch, seqLen, inFeatures, hidden);
        float[] input = Filled(new[] { batch * seqLen * inFeatures }, 31, 1f).ToArray();
        float[] wIh = Filled(new[] { G * inFeatures }, 32, 0.5f).ToArray();
        float[] wHh = Filled(new[] { G * hidden }, 33, 0.5f).ToArray();
        var output = new float[batch * seqLen * hidden];

        void Forward(Span<float> outSpan) => CpuEngine.LstmTrainForwardCoreFloat(
            input, wIh, wHh, null, null, null, null, returnSequences: true, ws, outSpan);

        Forward(output);
        Assert.Equal(1, ws.CompletedForwards);

        Assert.ThrowsAny<Exception>(() => Forward(new float[hidden]));
        var partial = Assert.Throws<InvalidOperationException>(() => ws.BeginBackward());
        Assert.Contains("partial", partial.Message);
        Assert.Equal(1, ws.CompletedForwards);

        Forward(output);
        ws.BeginBackward();
        ws.EndBackward();
        Assert.Equal(2, ws.CompletedForwards);
    }

    // Two external leaves into one same-shape add (each with a single consumer): the reshape/add gradient aliasing
    // shares their gradient buffers and skips the add's backward, and the gradient pool may give same-shaped leaves one
    // buffer. Both leaves must still publish their own .Grad, equal to the eager tape's, on every step.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledInputGradient_TwoLeavesIntoOneAdd_EachGetsItsGradient(bool gradPool)
    {
        var engine = new CpuEngine();
        int[] shape = { 4, 16 };
        float[] eagerA, eagerB;
        {
            var a0 = Filled(shape, 21, 1f);
            var b0 = Filled(shape, 22, 1f);
            var w0 = Filled(new[] { 16, 8 }, 23, 0.3f);
            var y0 = Filled(new[] { 4, 8 }, 24, 1f);
            using var tape = new GradientTape<float>();
            var loss = engine.TensorMSELoss(engine.FusedLinear(engine.TensorAdd(a0, b0), w0, null, FusedActivationType.None), y0);
            var grads = tape.ComputeGradients(loss, sources: new[] { a0, b0, w0 });
            eagerA = grads[a0].ToArray();
            eagerB = grads[b0].ToArray();
        }

        var a = Filled(shape, 21, 1f);
        var b = Filled(shape, 22, 1f);
        var w = Filled(new[] { 16, 8 }, 23, 0.3f);
        var y = Filled(new[] { 4, 8 }, 24, 1f);
        ICompiledTrainingPlan<float> plan;
        bool? previous = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        string? previousPool = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL");
        CompiledTrainingPlan<float>.CompiledInputGradientsForTests = true;
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", gradPool ? "1" : null);
        try
        {
            using var scope = GraphMode.Enable();
            engine.TensorMSELoss(engine.FusedLinear(engine.TensorAdd(a, b), w, null, FusedActivationType.None), y);
            plan = scope.CompileTraining(new[] { w });
        }
        finally
        {
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previous;
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", previousPool);
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                foreach (var (leaf, expected, name) in new[] { (a, eagerA, "a"), (b, eagerB, "b") })
                {
                    if (leaf.Grad is not { } g)
                    {
                        Assert.Fail($"step {step}: leaf {name} got no .Grad");
                        return;
                    }
                    var actual = g.ToArray();
                    Assert.Equal(expected.Length, actual.Length);
                    for (int i = 0; i < expected.Length; i++)
                        Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                            $"step {step}: {name}.Grad[{i}] {actual[i]} vs eager {expected[i]}");
                }
            }
        }
    }

    // Same-shaped leaves whose gradients are written by separate backward actions (one linear each): the liveness pool
    // must not give them one buffer, or both .Grad would show the last writer's value.
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledInputGradient_LeavesOfSeparateLinears_KeepTheirOwnGradients(bool gradPool)
    {
        var engine = new CpuEngine();
        int[] shape = { 4, 16 };
        Tensor<float> Loss(Tensor<float> a, Tensor<float> b, Tensor<float> w1, Tensor<float> w2, Tensor<float> y) =>
            engine.TensorMSELoss(engine.TensorAdd(
                engine.FusedLinear(a, w1, null, FusedActivationType.None),
                engine.FusedLinear(b, w2, null, FusedActivationType.ReLU)), y);
        float[] eagerA, eagerB;
        {
            var a0 = Filled(shape, 31, 1f); var b0 = Filled(shape, 32, 1f);
            var w10 = Filled(new[] { 16, 8 }, 33, 0.3f); var w20 = Filled(new[] { 16, 8 }, 34, 0.3f);
            var y0 = Filled(new[] { 4, 8 }, 35, 1f);
            using var tape = new GradientTape<float>();
            var grads = tape.ComputeGradients(Loss(a0, b0, w10, w20, y0), sources: new[] { a0, b0, w10, w20 });
            eagerA = grads[a0].ToArray();
            eagerB = grads[b0].ToArray();
        }

        var a = Filled(shape, 31, 1f); var b = Filled(shape, 32, 1f);
        var w1 = Filled(new[] { 16, 8 }, 33, 0.3f); var w2 = Filled(new[] { 16, 8 }, 34, 0.3f);
        var y = Filled(new[] { 4, 8 }, 35, 1f);
        ICompiledTrainingPlan<float> plan;
        bool? previous = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        string? previousPool = Environment.GetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL");
        CompiledTrainingPlan<float>.CompiledInputGradientsForTests = true;
        Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", gradPool ? "1" : null);
        try
        {
            using var scope = GraphMode.Enable();
            Loss(a, b, w1, w2, y);
            plan = scope.CompileTraining(new[] { w1, w2 });
        }
        finally
        {
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previous;
            Environment.SetEnvironmentVariable("AIDOTNET_COMPILED_GRAD_POOL", previousPool);
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                foreach (var (leaf, expected, name) in new[] { (a, eagerA, "a"), (b, eagerB, "b") })
                {
                    if (leaf.Grad is not { } g)
                    {
                        Assert.Fail($"step {step}: leaf {name} got no .Grad");
                        return;
                    }
                    var actual = g.ToArray();
                    for (int i = 0; i < expected.Length; i++)
                        Assert.True(Math.Abs(expected[i] - actual[i]) <= 1e-5f * Math.Max(1f, Math.Abs(expected[i])),
                            $"step {step}: {name}.Grad[{i}] {actual[i]} vs eager {expected[i]}");
                }
                Assert.False(ReferenceEquals(a.Grad, b.Grad), "both leaves expose one gradient buffer");
            }
        }
    }

    // Conv2D -> channel-bias add -> ReLU, optionally into a tiling max pool: the chain the conv-epilogue fusion detects,
    // and with the pool, the pooled-fusion backward that runs at the pool's position.
    private static Tensor<float> ConvLoss(CpuEngine engine, Tensor<float> x, Tensor<float> w, Tensor<float> b, bool pooled)
    {
        var y = engine.FusedConv2D(x, w, b, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
        if (pooled) y = engine.MaxPool2D(y, 2);
        return engine.ReduceSum(engine.TensorMultiply(y, y), null);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CompiledConv_InputGradient_IsOptIn_AndMatchesTheEagerTape(bool requested, bool pooled)
    {
        var engine = new CpuEngine();
        int[] xShape = { 2, 32, 8, 8 }, wShape = { 32, 32, 3, 3 };
        var x0 = Filled(xShape, 11, 2f);
        var w0 = Filled(wShape, 12, 0.5f);
        var b0 = Filled(new[] { 32 }, 13, 0.5f);

        float[] eagerInputGrad;
        using (var tape = new GradientTape<float>())
        {
            var loss = ConvLoss(engine, x0, w0, b0, pooled);
            var grads = tape.ComputeGradients(loss, sources: new[] { x0, w0, b0 });
            eagerInputGrad = grads[x0].ToArray();
        }

        var x = Filled(xShape, 11, 2f);
        var w = Filled(wShape, 12, 0.5f);
        var b = Filled(new[] { 32 }, 13, 0.5f);
        ICompiledTrainingPlan<float> plan;
        bool? previous = CompiledTrainingPlan<float>.CompiledInputGradientsForTests;
        CompiledTrainingPlan<float>.CompiledInputGradientsForTests = requested;
        try
        {
            using var scope = GraphMode.Enable();
            ConvLoss(engine, x, w, b, pooled);
            plan = scope.CompileTraining(new[] { w, b });   // the input is not a parameter
        }
        finally
        {
            CompiledTrainingPlan<float>.CompiledInputGradientsForTests = previous;
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            for (int step = 0; step < 2; step++)
            {
                plan.Step();
                if (!requested)
                {
                    Assert.True(x.Grad is null, $"step {step}: the input received a gradient although none was requested");
                    continue;
                }
                if (x.Grad is not { } g)
                {
                    Assert.Fail($"step {step}: the requested input gradient was not set");
                    return;
                }
                var actual = g.ToArray();
                Assert.Equal(eagerInputGrad.Length, actual.Length);
                double maxAbs = 0;
                foreach (float v in eagerInputGrad) maxAbs = Math.Max(maxAbs, Math.Abs(v));
                for (int i = 0; i < actual.Length; i++)
                    Assert.True(Math.Abs(actual[i] - eagerInputGrad[i]) <= 1e-5 * Math.Max(1.0, maxAbs),
                        $"step {step}: input gradient [{i}] is {actual[i]}, the eager tape gives {eagerInputGrad[i]}");
            }
        }
    }
}
