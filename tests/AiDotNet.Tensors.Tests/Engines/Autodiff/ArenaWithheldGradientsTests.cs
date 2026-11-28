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
                // A caller's arena withholds them for a step; a tape-owned arena quarantines them while they are held.
                Assert.True(RetainedCount() > 0, "The step must retain arena-backed gradients.");
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

    // What a finished step keeps out of the pool for its caller: the withheld list (a caller-owned arena's one-step
    // hold) plus the escape quarantine (a tape-owned arena keeps each result until the caller drops it).
    private static int RetainedCount() => WithheldCount() + TensorArena.QuarantinedCount;

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
                using (TensorArena.Create()) Step();
                counts[step] = WithheldCount();   // the arena's Dispose withholds the handed-over gradients
            }
            else
            {
                // A tape-owned arena quarantines the step's buffers until nothing references them: once the previous
                // step's results are collected, this step's Dispose moves them to the pool, leaving only its own.
                GC.Collect();
                GC.WaitForPendingFinalizers();
                Step();
                counts[step] = RetainedCount();
            }
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
            try { counts = RunSteps(callerArenaPerStep, steps: 8); }
            catch (Exception e) { failure = e; }
        });
        thread.Start();
        thread.Join();
        if (failure is not null) throw new InvalidOperationException("The training steps threw.", failure);

        // One step's handed-over gradients may be held at a time; later steps must not add to them. The leak this guards
        // (AiDotNet#2322) adds a set per step: [2, 4, 6, ...] for a caller's arena, +2 a step for a tape-owned one.
        //
        // The early steps are not a bound. What a step retains depends on process-wide pool state that tests running in
        // parallel change, so the first steps can hold fewer than the steady state: measured [1, 0, 2, 2, 2, 2] for a
        // caller's arena, and [5, 6, 6, 7, 7, 7] for a tape-owned one, which a bound taken from the first three steps
        // reported as growth. A tape-owned arena's steady count is 7 on this model: the step's own pooled forward
        // buffers (3 x 131072 + 2 x 65536 floats), quarantined until their readers are collected, plus the previous
        // step's two gradients (2 x 131072), which the caller still referenced at the last sweep and which leave at the
        // next. So the test checks growth: from the fourth step on, the count stays flat and never exceeds the fourth
        // step's.
        Assert.NotNull(counts);
        if (counts is null) return;
        string seen = $"Retained arrays per step: [{string.Join(", ", counts)}]";
        // Not vacuous: a run that never withheld anything would pass the bound without testing it.
        Assert.True(Array.Exists(counts, count => count > 0), $"{seen}; no step retained a gradient, so the bound was not exercised.");
        // A caller's arena withholds at most the two gradients (w1, w2) a step hands over.
        if (callerArenaPerStep)
            for (int step = 0; step < counts.Length; step++)
                Assert.True(counts[step] <= 2, $"{seen}; step {step} holds {counts[step]}, more than the 2 gradients of one step.");
        int steady = counts[3];
        for (int step = 3; step < counts.Length; step++)
            Assert.True(counts[step] <= steady, $"{seen}; step {step} holds {counts[step]}, more than the {steady} of step 3.");
        int lastMin = int.MaxValue, lastMax = int.MinValue;
        for (int step = counts.Length - 4; step < counts.Length; step++)
        {
            lastMin = Math.Min(lastMin, counts[step]);
            lastMax = Math.Max(lastMax, counts[step]);
        }
        Assert.True(lastMin == lastMax, $"{seen}; the last four steps are not flat.");
    }
}
