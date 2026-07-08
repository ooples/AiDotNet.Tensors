using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression for the compiled-training-plan BACKWARD-REPLAY buffer-liveness bug.
///
/// <para><b>Bug:</b> a plan's forward-activation buffers were allocated from whatever transient
/// <see cref="TensorArena"/> the caller had open during the trace (the fused training path opens a
/// per-step arena and compiles the plan on the first step), then RETURNED to the shared pool when
/// that step's arena disposed. On a later step — run in its OWN fresh arena — a backward temporary
/// (e.g. the <c>ReduceSum</c> gradient broadcast, which tiles a fresh <c>[B, V]</c> buffer) re-rented
/// a still-live forward activation and overwrote it BEFORE its backward consumer read it. For a
/// large activation (≥ the ArrayPool bucket boundary) this silently zeroed gradients on replay,
/// freezing fused training at large vocab (fused LM top-1 collapsed to ≤ unigram while eager learned).</para>
///
/// <para><b>Fix:</b> <see cref="TensorArena.Suspend"/> wraps the trace/compile in
/// <see cref="CompiledModelCache{T}.GetOrCompileTraining"/> so a compiled plan owns its buffers
/// independently of any transient per-step arena and they can never be recycled underneath a replay.</para>
///
/// <para>This test mirrors the real path exactly: it compiles through
/// <see cref="CompiledModelCache{T}.GetOrCompileTraining"/> INSIDE a per-step arena and then replays
/// each subsequent step inside its OWN fresh per-step arena, checking the trained parameter's
/// gradient stays non-zero and finite on every replay step.</para>
/// </summary>
public class CompiledBackwardReplayBufferLivenessTests
{
    private readonly ITestOutputHelper _o;
    public CompiledBackwardReplayBufferLivenessTests(ITestOutputHelper o) { _o = o; }

    private static double L2(Tensor<float> t)
    {
        double s = 0; var a = t.GetDataArray(); for (int i = 0; i < t.Length; i++) { double v = a[i]; s += v * v; } return Math.Sqrt(s);
    }

    [Theory]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void FusedReplayGradientsSurvivePerStepArenas(int V)
    {
        AiDotNetEngine.Current = new CpuEngine();
        var engine = new CpuEngine();
        int B = 128, K = 128;
        var x = Tensor<float>.CreateRandom(new[] { B, K });
        var w = Tensor<float>.CreateRandom(new[] { K, V });
        var rng = new Random(3);
        var tgt = new Tensor<float>(new[] { B, V });
        for (int r = 0; r < B; r++) tgt[r, rng.Next(V)] = 1f;

        // The loss the LM/SequenceClassification path traces: softmax head -> clamp -> log -> CE.
        Func<Tensor<float>> forwardAndLoss = () =>
        {
            var mm = engine.TensorMatMul(x, w);
            var sm = engine.Softmax(mm, -1);                        // large [B,V] activation the Clamp backward reads
            var clamped = engine.TensorClamp(sm, 1e-7f, 1f);
            var lg = engine.TensorLog(clamped);
            var prod = engine.TensorMultiply(tgt, lg);
            var perRow = engine.ReduceSum(prod, new[] { 1 }, false); // ReduceSum backward tiles a fresh [B,V] temp
            var mean = engine.ReduceMean(perRow, new[] { 0 }, false);
            return engine.TensorNegate(mean);
        };
        var shapeKey = new[] { B, K };

        var cache = new CompiledModelCache<float>();
        double[] grads = new double[6];
        for (int step = 0; step < 6; step++)
        {
            // Each step in its OWN transient arena — exactly how the fused training path wraps a step.
            using var stepArena = TensorArena.Create();
            var plan = cache.GetOrCompileTraining(shapeKey, forwardAndLoss, new[] { w }); // compiles on step 0
            var loss = plan.Step();
            grads[step] = L2(plan.Gradients[0]);
            Assert.False(float.IsNaN(loss[0]), $"loss NaN at V={V} step {step}");
        }

        _o.WriteLine($"V={V} replay gradL2 per step: {string.Join(", ", Array.ConvertAll(grads, g => g.ToString("G4")))}");
        // Pre-fix: step 0 grad is fine but every REPLAY step (1..5) collapses to 0 for V ≥ 512.
        for (int step = 0; step < 6; step++)
            Assert.True(grads[step] > 0,
                $"REPLAY grad ZERO at V={V} step {step} — a forward activation was recycled after its step arena disposed " +
                $"(per-step grads: {string.Join(", ", Array.ConvertAll(grads, g => g.ToString("G4")))})");
    }
}
