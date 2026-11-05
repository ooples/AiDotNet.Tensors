using System;
using System.Collections;
using System.Reflection;
using System.Threading;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A tape's returned gradients are withheld from the cross-arena pool for one step (#1031). With the tape inside a
/// caller's per-step arena (AiDotNet's NeuralNetworkBase training step), the arena's Dispose withheld them but nothing
/// ever returned them: the thread-static withheld list grew by one full set of gradients per step (AiDotNet#2322).
/// </summary>
public class ArenaWithheldGradientsTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ReturnedGradients_SurviveUnrelatedArenaDisposalAndRents(bool callerArenaPerStep)
    {
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try
            {
                var engine = new CpuEngine();
                var w1 = Rnd(new[] { 256, 512 }, 1);
                var w2 = Rnd(new[] { 512, 256 }, 2);
                var x = Rnd(new[] { 256, 256 }, 3);
                Tensor<float> gradient;
                using (callerArenaPerStep ? TensorArena.Create() : null)
                {
                    using var tape = new GradientTape<float>();
                    var h = engine.TensorMatMul(engine.TensorMatMul(x, w1), w2);
                    var loss = engine.ReduceSum(engine.TensorMultiply(h, h), null);
                    gradient = tape.ComputeGradients(loss, new[] { w1, w2 })[w1];
                }
                Assert.True(WithheldCount() > 0, "The step must retain arena-backed gradients.");
                var expected = gradient.ToArray();

                // Neither an inference arena nor scratch rentals end the optimizer's use of these gradients.
                using (TensorArena.Create()) { }
                int rentals = TensorArena.CrossArenaRetainedArrayCount + 1;
                using (TensorArena.Create())
                {
                    for (int i = 0; i < rentals; i++)
                    {
                        var scratch = TensorAllocator.Rent<float>(new[] { 256, 512 });
                        for (int j = 0; j < scratch.Length; j++) scratch[j] = -123f;
                    }
                    Assert.Equal(expected, gradient.ToArray());
                }
                Assert.Equal(expected, gradient.ToArray());
            }
            catch (Exception e) { failure = e; }
        });
        thread.Start();
        thread.Join();
        if (failure is not null) throw new InvalidOperationException("Gradient lifetime regression failed.", failure);
    }

    private static int WithheldCount()
    {
        var field = typeof(TensorArena).GetField("t_withheldFromPool", BindingFlags.NonPublic | BindingFlags.Static)
            ?? throw new InvalidOperationException("TensorArena.t_withheldFromPool was renamed; update this test.");
        return field.GetValue(null) is ICollection list ? list.Count : 0;
    }

    private static Tensor<float> Rnd(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static int[] RunSteps(bool ownArenaPerStep, int steps)
    {
        var engine = new CpuEngine();
        // Large enough (>= 64K elements) for the arena to pool, and so withhold, the gradient buffers.
        var w1 = Rnd(new[] { 256, 512 }, 1);
        var w2 = Rnd(new[] { 512, 256 }, 2);
        var x = Rnd(new[] { 256, 256 }, 3);
        var counts = new int[steps];
        for (int step = 0; step < steps; step++)
        {
            void Step()
            {
                using var tape = new GradientTape<float>();
                var h = engine.TensorMatMul(engine.TensorMatMul(x, w1), w2);
                var loss = engine.ReduceSum(engine.TensorMultiply(h, h), null);
                var grads = tape.ComputeGradients(loss, new[] { w1, w2 });
                Assert.Equal(2, grads.Count);
            }

            if (ownArenaPerStep)
            {
                using var arena = TensorArena.Create();
                Step();
            }
            else
            {
                Step();
            }

            counts[step] = WithheldCount();
        }

        return counts;
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void WithheldGradients_StayBoundedAcrossSteps(bool callerArenaPerStep)
    {
        // A fresh thread: the withheld list is thread-static, so other tests on the pool thread cannot leave entries.
        int[]? counts = null;
        Exception? failure = null;
        var thread = new Thread(() =>
        {
            try { counts = RunSteps(callerArenaPerStep, steps: 6); }
            catch (Exception e) { failure = e; }
        });
        thread.Start();
        thread.Join();
        if (failure is not null) throw new InvalidOperationException("The training steps threw.", failure);

        // One step's handed-over gradients may be withheld at a time; later steps must not add to them.
        int bound = Math.Max(counts![0], 1);
        for (int step = 1; step < counts.Length; step++)
            Assert.True(counts[step] <= bound,
                $"Withheld arrays per step: [{string.Join(", ", counts)}]; step {step} holds {counts[step]}, more than the {bound} of one step.");
    }
}
