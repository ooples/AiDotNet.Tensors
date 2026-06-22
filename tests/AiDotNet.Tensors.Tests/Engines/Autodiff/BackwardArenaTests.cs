using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// #1662 lever #4 guard: per-step BACKWARD allocation must stay bounded when the per-step
/// <see cref="TensorArena"/> is active (the way real training runs). This locks in the fix
/// that routed Conv2D backward output gradients through the arena — before it, the conv
/// backward leaked ~1 MB/step of `new float[]` output buffers past the arena
/// (`--trainbench --block conv`: 1.072 MB/step, 76.6% recycled). After, ~0.1 MB/step.
///
/// <para>Measurement note: <see cref="GC.GetAllocatedBytesForCurrentThread"/> counts only the
/// test thread. The streaming backward and the output-grad rent run on this thread; conv's
/// internal im2col scratch uses ArrayPool on worker threads (not counted, and already pooled).
/// So this faithfully guards the main-thread output-grad allocation the fix addressed.</para>
/// </summary>
public class BackwardArenaTests
{
    private readonly CpuEngine _engine = new();

    [Fact]
    public void Conv2DBackward_PerStepAllocation_Recycled_ByArena()
    {
        var rng = new Random(0);
        // Same shape class as the trainbench conv probe that surfaced the leak.
        const int channels = 32, sp = 32, layers = 6;
        var x = Rand(new[] { 1, channels, sp, sp }, rng);
        var kernels = new Tensor<float>[layers];
        for (int l = 0; l < layers; l++)
            kernels[l] = Scaled(Rand(new[] { channels, channels, 3, 3 }, rng), 0.02f);

        void OneStep()
        {
            using var tape = new GradientTape<float>();
            var h = x;
            for (int l = 0; l < layers; l++)
            {
                var y = _engine.Conv2D(h, kernels[l], 1, 1, 1);
                y = _engine.GELU(y);
                h = _engine.TensorAdd(h, y);
            }
            var loss = _engine.ReduceSum(_engine.TensorMultiply(h, h));
            var sources = new List<Tensor<float>>(layers);
            for (int l = 0; l < layers; l++) sources.Add(kernels[l]);
            tape.ComputeGradientsStreaming(loss, sources, (src, g) =>
            {
                if (g is null || g.Length == 0) return;
                for (int i = 0; i < src.Length; i++) src[i] -= 1e-4f * g[i];
            });
        }

        const int warmup = 12, reps = 10;

        // Measure WITHOUT an arena: conv backward output grads are fresh allocations each step.
        // (Relative on-vs-off comparison is robust to the process-global AutoTensorCache state
        // left by other tests in the suite, which an absolute byte threshold is not.)
        for (int i = 0; i < warmup; i++) OneStep();
        long offStart = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < reps; i++) OneStep();
        long perStepOff = (GC.GetAllocatedBytesForCurrentThread() - offStart) / reps;

        // Measure WITH the per-step arena: the conv backward output grads (and other scratch)
        // are recycled via Reset(), so per-step allocation drops sharply.
        long perStepOn;
        using (var arena = TensorArena.Create())
        {
            for (int i = 0; i < warmup; i++) { arena.Reset(); OneStep(); }
            long onStart = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 0; i < reps; i++) { arena.Reset(); OneStep(); }
            perStepOn = (GC.GetAllocatedBytesForCurrentThread() - onStart) / reps;
        }

        // The arena must recycle the bulk of the per-step allocation. Pre-fix, conv backward
        // output grads bypassed the arena (raw new float[]) so the arena barely helped here;
        // after routing them through the arena, on-allocation is a small fraction of off.
        Assert.True(perStepOn * 3 < perStepOff,
            $"arena did not recycle conv backward allocation: on={perStepOn} B/step, off={perStepOff} B/step " +
            $"(expected on < off/3; conv backward output-grad arena routing may have regressed)");
    }

    private static Tensor<float> Rand(int[] s, Random r)
    {
        var t = new Tensor<float>(s);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(r.NextDouble() - 0.5);
        return t;
    }

    private static Tensor<float> Scaled(Tensor<float> t, float s)
    {
        for (int i = 0; i < t.Length; i++) t[i] *= s;
        return t;
    }
}
