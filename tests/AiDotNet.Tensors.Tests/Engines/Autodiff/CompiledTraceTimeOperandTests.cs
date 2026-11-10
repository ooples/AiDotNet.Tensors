using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// An op recorded as a compiled node (<c>TryRecordCompiledOp</c>) runs once eagerly at trace time to materialize its
/// output. Its operands are usually still lazy there: the producer has been recorded but not run, so its buffer holds
/// whatever the allocator handed out. The engine's kernels read operands through accessors that deliberately do not
/// realize a lazy node, so the trace-time run pooled uninitialized memory, and a compiled training step built on that
/// trace returned a loss and gradients that changed from run to run (WGAN-GP's generator: a Conv/ReLU feeding
/// MaxPool2DWithTensorIndices gave a first-step loss of 1.32, 5.0e-5 or 4e11 depending on the recycled memory).
/// </summary>
public class CompiledTraceTimeOperandTests
{
    private static void Fill(Tensor<float> t, int seed)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        for (int i = 0; i < t.Length; i++) t.SetFlat(i, (float)(rng.NextDouble() * 0.6 - 0.3));
    }

    // WGAN-GP's generator stem: two Conv/bias/ReLU blocks, each pooled by MaxPool2DWithTensorIndices (what a
    // MaxPoolingLayer runs in training), weighted so every pooled cell reaches the loss with its own coefficient.
    private static Tensor<float> Loss(IEngine e, Tensor<float> x, Tensor<float> k1, Tensor<float> b1,
        Tensor<float> k2, Tensor<float> b2, Tensor<float> w)
    {
        var one = new[] { 1, 1 };
        var h = e.ReLU(e.TensorChannelBiasAdd(e.Conv2D(x, k1, one, one, one), b1));
        h = e.MaxPool2DWithTensorIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _);
        h = e.ReLU(e.TensorChannelBiasAdd(e.Conv2D(h, k2, one, one, one), b2));
        h = e.MaxPool2DWithTensorIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _);
        return e.ReduceSum(e.TensorMultiply(h, w), null, false);
    }

    [Fact]
    public void MaxPoolWithIndices_OnALazyProducer_StepsTheProducersValues()
    {
        var x = new Tensor<float>(new[] { 1, 1, 8, 8 });
        var k1 = new Tensor<float>(new[] { 32, 1, 3, 3 });
        var b1 = new Tensor<float>(new[] { 32 });
        var k2 = new Tensor<float>(new[] { 32, 32, 3, 3 });
        var b2 = new Tensor<float>(new[] { 32 });
        var w = new Tensor<float>(new[] { 1, 32, 2, 2 });
        Fill(x, 1); Fill(k1, 2); Fill(b1, 3); Fill(k2, 4); Fill(b2, 5); Fill(w, 6);
        AssertFirstStepsMatchTheTape(e => Loss(e, x, k1, b1, k2, b2, w), new[] { k1, b1, k2, b2 });
    }

    public enum TraceTimeOp { MaxPoolWithIndices, BatchNorm, LayerNorm, RMSNorm, GroupNorm, InstanceNorm }

    // Each op whose graph node is built from a trace-time eager run, fed by a Conv/bias/ReLU recorded just before it.
    [Theory]
    [InlineData(TraceTimeOp.MaxPoolWithIndices)]
    [InlineData(TraceTimeOp.BatchNorm)]
    [InlineData(TraceTimeOp.LayerNorm)]
    [InlineData(TraceTimeOp.RMSNorm)]
    [InlineData(TraceTimeOp.GroupNorm)]
    [InlineData(TraceTimeOp.InstanceNorm)]
    public void TraceTimeEagerOp_OnALazyProducer_StepsTheProducersValues(TraceTimeOp op)
    {
        var x = new Tensor<float>(new[] { 2, 1, 4, 4 });
        var k = new Tensor<float>(new[] { 8, 1, 3, 3 });
        var b = new Tensor<float>(new[] { 8 });
        var channelScale = new Tensor<float>(new[] { 8 });
        var channelShift = new Tensor<float>(new[] { 8 });
        var lastScale = new Tensor<float>(new[] { 4 });
        var lastShift = new Tensor<float>(new[] { 4 });
        Fill(x, 11); Fill(k, 12); Fill(b, 13); Fill(channelScale, 14); Fill(channelShift, 15); Fill(lastScale, 16);
        Fill(lastShift, 17);
        var w = new Tensor<float>(op == TraceTimeOp.MaxPoolWithIndices ? new[] { 2, 8, 2, 2 } : new[] { 2, 8, 4, 4 });
        Fill(w, 18);
        var one = new[] { 1, 1 };
        AssertFirstStepsMatchTheTape(e =>
        {
            var h = e.ReLU(e.TensorChannelBiasAdd(e.Conv2D(x, k, one, one, one), b));
            var y = op switch
            {
                TraceTimeOp.MaxPoolWithIndices => e.MaxPool2DWithTensorIndices(h, new[] { 2, 2 }, new[] { 2, 2 }, out _),
                TraceTimeOp.BatchNorm => e.BatchNorm(h, channelScale, channelShift, 1e-5, out _, out _),
                TraceTimeOp.LayerNorm => e.LayerNorm(h, lastScale, lastShift, 1e-5, out _, out _),
                TraceTimeOp.RMSNorm => e.RMSNorm(h, lastScale, 1e-5, out _),
                TraceTimeOp.GroupNorm => e.GroupNorm(h, 2, channelScale, channelShift, 1e-5, out _, out _),
                _ => e.InstanceNorm(h, channelScale, channelShift, 1e-5, out _, out _),
            };
            return e.ReduceSum(e.TensorMultiply(y, w), null, false);
        }, new[] { k, b });
    }

    private static void AssertFirstStepsMatchTheTape(Func<IEngine, Tensor<float>> loss, Tensor<float>[] parameters)
    {
        var engine = new CpuEngine();
        var prior = AiDotNetEngine.Current;
        bool priorFresh = TensorPool.ForceFreshAllocations;
        AiDotNetEngine.Current = engine;
        try
        {
            float expectedLoss;
            var expected = new float[parameters.Length][];
            using (var tape = new GradientTape<float>())
            {
                var l = loss(engine);
                expectedLoss = l.GetFlat(0);
                var grads = tape.ComputeGradients(l, parameters);
                for (int i = 0; i < parameters.Length; i++) expected[i] = grads[parameters[i]].ToArray();
            }
            Assert.NotEqual(0f, expectedLoss);

            // A compiled training step traces and runs its first step on fresh, uninitialized allocations.
            TensorPool.ForceFreshAllocations = true;
            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.EnableTraining(parameters))
            {
                loss(engine);
                plan = scope.CompileTraining(parameters);
            }
            using (plan)
            {
                // The first step runs on the trace's data, as a compiled training step's first call does.
                for (int step = 0; step < 2; step++)
                {
                    float actualLoss = plan.Step().GetFlat(0);
                    Assert.True(Math.Abs(expectedLoss - actualLoss) <= 1e-4f * (1 + Math.Abs(expectedLoss)),
                        $"step {step}: tape loss {expectedLoss:R}, compiled {actualLoss:R}");
                    for (int p = 0; p < parameters.Length; p++)
                    {
                        var actual = plan.Gradients[p].ToArray();
                        for (int i = 0; i < actual.Length; i++)
                            Assert.True(Math.Abs(expected[p][i] - actual[i]) <= 1e-4f * (1 + Math.Abs(expected[p][i])),
                                $"step {step}, parameter {p}[{i}]: tape {expected[p][i]:R}, compiled {actual[i]:R}");
                    }
                }
            }
        }
        finally
        {
            TensorPool.ForceFreshAllocations = priorFresh;
            AiDotNetEngine.Current = prior;
        }
    }
}
