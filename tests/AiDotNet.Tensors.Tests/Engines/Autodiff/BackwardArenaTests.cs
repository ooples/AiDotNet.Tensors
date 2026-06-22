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
    public void Conv2DBackward_PerStepAllocation_StaysBounded_WithArena()
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

        using var arena = TensorArena.Create();
        for (int i = 0; i < 5; i++) { arena.Reset(); OneStep(); } // warmup: fill the arena

        const int reps = 10;
        long start = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < reps; i++) { arena.Reset(); OneStep(); }
        long perStep = (GC.GetAllocatedBytesForCurrentThread() - start) / reps;

        // Pre-fix this path allocated ~1.07 MB/step on the main thread; post-fix ~0.1 MB/step.
        // 0.5 MB threshold sits firmly between the two so a regression of the conv arena
        // routing trips it, while leaving headroom for tape bookkeeping / machine variance.
        const long thresholdBytes = 512L * 1024;
        Assert.True(perStep < thresholdBytes,
            $"per-step conv backward allocation {perStep} bytes >= threshold {thresholdBytes} " +
            $"(arena routing for Conv2D backward output grads may have regressed)");
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
