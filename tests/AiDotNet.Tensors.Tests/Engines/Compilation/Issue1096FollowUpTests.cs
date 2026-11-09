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

    private static Tensor<float> ConvLoss(CpuEngine engine, Tensor<float> x, Tensor<float> w, Tensor<float> b)
    {
        var y = engine.FusedConv2D(x, w, b, 1, 1, 1, 1, 1, 1, FusedActivationType.ReLU);
        return engine.ReduceSum(engine.TensorMultiply(y, y), null);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CompiledConv_InputGradient_IsOptIn_AndMatchesTheEagerTape(bool requested)
    {
        var engine = new CpuEngine();
        int[] xShape = { 2, 32, 8, 8 }, wShape = { 32, 32, 3, 3 };
        var x0 = Filled(xShape, 11, 2f);
        var w0 = Filled(wShape, 12, 0.5f);
        var b0 = Filled(new[] { 32 }, 13, 0.5f);

        float[] eagerInputGrad;
        using (var tape = new GradientTape<float>())
        {
            var loss = ConvLoss(engine, x0, w0, b0);
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
            ConvLoss(engine, x, w, b);
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
                var g = x.Grad;
                Assert.True(g is not null, $"step {step}: the requested input gradient was not set");
                var actual = g!.ToArray();
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
