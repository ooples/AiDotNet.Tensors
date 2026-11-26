using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A weight-tied loop (<see cref="WeightTiedLoop"/>) inside a compiled GPU training step. One shared block (W1, W2) runs
/// k times; around it, each iteration i has its own small adapter (a diagonal scale e_i and a rank-r pair A_i, B_i) that
/// only iteration i reads. The compiled step must give the eager tape's loss and gradients: the shared block's gradient
/// summed over all k uses, each adapter's gradient from its own iteration alone. It must do so in a captured graph that
/// replays, with each iteration checkpointed or not; checkpointing must keep the plan's activation memory from growing
/// with k as the plain plan's does. The adaptive loop (<see cref="WeightTiedLoop.RunAdaptive{T}"/>) must match the eager
/// tape the same way, with its masked, device-side exit inside the replayed graph.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CompiledWeightTiedLoopTests
{
    private const int Rank = 4;
    private readonly ITestOutputHelper _output;

    public CompiledWeightTiedLoopTests(ITestOutputHelper output) => _output = output;

    private static CudaBackend RequireCuda()
    {
        var backend = new CudaBackend();
        if (!backend.IsAvailable)
        {
            backend.Dispose();
            bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1"
                || Environment.GetEnvironmentVariable("AIDOTNET_EXPECT_GPU_BACKEND") == "CUDA";
            Assert.False(required, "The validation lane requires a working CUDA backend.");
            Skip.If(true, "No CUDA device available.");
        }
        return backend;
    }

    private static Tensor<float> Filled(int[] shape, int seed, float scale, float offset = 0f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = offset + (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    /// <summary>The loop's parameters: shared [W1, W2], then per iteration [e_i, A_i, B_i], then the halting head [w, b].</summary>
    private sealed class LoopModel
    {
        internal readonly int Rows, Width, Hidden, Iterations;
        internal readonly Tensor<float> X, Target;
        internal readonly Tensor<float>[] Parameters;

        internal LoopModel(int rows, int width, int hidden, int iterations, bool halting = false, int seed = 0)
        {
            Rows = rows; Width = width; Hidden = hidden; Iterations = iterations;
            X = Filled(new[] { rows, width }, 1 + seed, 2f);
            Target = Filled(new[] { rows, width }, 2 + seed, 1f);
            var p = new List<Tensor<float>>
            {
                Filled(new[] { width, hidden }, 10 + seed, 2f / (float)Math.Sqrt(width)),
                Filled(new[] { hidden, width }, 11 + seed, 2f / (float)Math.Sqrt(hidden)),
            };
            for (int i = 0; i < iterations; i++)
            {
                p.Add(Filled(new[] { 1, width }, 20 + 3 * i + seed, 0.2f, 1f));
                p.Add(Filled(new[] { width, Rank }, 21 + 3 * i + seed, 0.3f));
                p.Add(Filled(new[] { Rank, width }, 22 + 3 * i + seed, 0.3f));
            }
            if (halting)
            {
                p.Add(Filled(new[] { width, 1 }, 90 + seed, 1.5f / (float)Math.Sqrt(width)));
                p.Add(Filled(new[] { 1 }, 91 + seed, 0f, -1.2f));
            }
            Parameters = p.ToArray();
        }

        internal Tensor<float>[] Fresh() => Parameters.Select(t => new Tensor<float>(t.Shape.ToArray(), new Vector<float>(t.ToArray()))).ToArray();

        /// <summary>h + 0.1 * W2 silu(W1 (h e_i + h A_i B_i)).</summary>
        internal static Tensor<float> Body(IEngine e, Tensor<float>[] p, Tensor<float> h, int i)
        {
            var ei = p[2 + 3 * i]; var ai = p[3 + 3 * i]; var bi = p[4 + 3 * i];
            var adapted = e.TensorAdd(e.TensorMultiply(h, ei), e.TensorMatMul(e.TensorMatMul(h, ai), bi));
            var inner = e.TensorMatMul(e.TensorSiLU(e.TensorMatMul(adapted, p[0])), p[1]);
            return e.TensorAdd(h, e.TensorMultiplyScalar(inner, 0.1f));
        }

        internal Tensor<float> HaltLogit(IEngine e, Tensor<float>[] p, Tensor<float> h)
            => e.TensorAdd(e.TensorMatMul(h, p[p.Length - 2]), p[p.Length - 1]);

        internal Tensor<float> TokenLoss(IEngine e, Tensor<float> h)
        {
            var d = e.TensorSubtract(h, Target);
            return e.ReduceSum(e.TensorMultiply(d, d), new[] { 1 }, true);
        }

        internal Tensor<float> LoopLoss(IEngine e, Tensor<float>[] p, bool checkpoint)
        {
            var h = WeightTiedLoop.Run(e, X, Iterations, (s, i) => Body(e, p, s, i), checkpoint);
            return e.TensorMultiplyScalar(e.ReduceSum(TokenLoss(e, h), null), 1f / Rows);
        }

        internal (Tensor<float> Loss, AdaptiveLoopResult<float> Exit) AdaptiveLoss(IEngine e, Tensor<float>[] p, double threshold,
            bool checkpoint, float entropyWeight)
        {
            var exit = WeightTiedLoop.RunAdaptive(e, X, Iterations, (s, i) => Body(e, p, s, i), s => HaltLogit(e, p, s),
                threshold, checkpoint);
            var losses = exit.States.Select(s => TokenLoss(e, s)).ToArray();
            var loss = e.TensorSubtract(exit.ExpectedLoss(losses), e.TensorMultiplyScalar(exit.Entropy, entropyWeight));
            return (loss, exit);
        }
    }

    private static (float Abs, float Rel) MaxDiff(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        float abs = 0, rel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = Math.Abs(expected[i] - actual[i]);
            abs = Math.Max(abs, d);
            rel = Math.Max(rel, d / (1f + Math.Abs(expected[i])));
        }
        return (abs, rel);
    }

    private sealed class CompiledRun
    {
        internal float[] Losses = Array.Empty<float>();
        internal float[][] FirstGradients = Array.Empty<float[]>();
        internal float[][] LastGradients = Array.Empty<float[]>();
        internal long Replays, Recomputes, ScratchBytes;
        internal int Segments;
        internal float LoopsRun = float.NaN, Entropy = float.NaN, ExpectedLoops = float.NaN;
        internal string? CaptureFailure;
    }

    /// <summary>Traces <paramref name="trace"/>, compiles it, and runs it at learning rate 0 (every step is step 0).</summary>
    private static CompiledRun RunCompiled(Tensor<float>[] parameters, int steps,
        Func<Tensor<float>[], (Tensor<float> Loss, AdaptiveLoopResult<float>? Exit)> trace)
    {
        CompiledTrainingPlan<float> plan;
        AdaptiveLoopResult<float>? exit;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            (var loss, exit) = trace(parameters);
            if (exit is not null)
            {
                GradientRecording.RetainOutput(exit.LoopsRun);
                GradientRecording.RetainOutput(exit.Entropy);
                GradientRecording.RetainOutput(exit.ExpectedLoops);
            }
            plan = (CompiledTrainingPlan<float>)scope.CompileTraining(parameters, loss);
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0f);
            long before = CompiledTrainingPlan<float>.GraphReplayCount;
            var run = new CompiledRun();
            var losses = new List<float>();
            for (int s = 0; s < steps; s++)
            {
                losses.Add(plan.Step().ToArray()[0]);
                var grads = plan.Gradients.Select(g => g.ToArray()).ToArray();
                if (s == 0) run.FirstGradients = grads;
                run.LastGradients = grads;
            }
            run.Losses = losses.ToArray();
            run.Replays = CompiledTrainingPlan<float>.GraphReplayCount - before;
            run.Recomputes = plan.CheckpointRecomputeRuns;
            run.Segments = plan.CheckpointSegmentCount;
            run.ScratchBytes = plan.ScratchBufferBytes;
            run.CaptureFailure = plan.GraphCaptureFailureReason;
            if (exit is not null)
            {
                run.LoopsRun = exit.LoopsRun.ToArray()[0];
                run.Entropy = exit.Entropy.ToArray()[0];
                run.ExpectedLoops = exit.ExpectedLoops.ToArray()[0];
            }
            return run;
        }
    }

    private static (float Loss, float[][] Gradients) RunEager(Tensor<float>[] parameters,
        Func<Tensor<float>[], Tensor<float>> forward)
    {
        using var tape = new GradientTape<float>();
        var loss = forward(parameters);
        var grads = tape.ComputeGradients(loss, sources: parameters);
        return (loss.ToArray()[0], parameters.Select(p => grads[p].ToArray()).ToArray());
    }

    private void AssertParity(float eagerLoss, float[][] eagerGrads, CompiledRun compiled, int iterations, string label,
        int adaptersTrained = -1)
    {
        var lossDiff = compiled.Losses.Max(l => Math.Abs(l - eagerLoss));
        float worstAbs = 0, worstRel = 0;
        for (int p = 0; p < eagerGrads.Length; p++)
            foreach (var g in new[] { compiled.FirstGradients[p], compiled.LastGradients[p] })
            {
                var (abs, rel) = MaxDiff(eagerGrads[p], g);
                worstAbs = Math.Max(worstAbs, abs);
                worstRel = Math.Max(worstRel, rel);
            }
        _output.WriteLine($"{label}: eager loss {eagerLoss:R}, compiled losses [{string.Join(", ", compiled.Losses.Select(l => l.ToString("R")))}], "
            + $"max |loss diff| {lossDiff:E3}, max |grad diff| {worstAbs:E3} (rel {worstRel:E3}), replays {compiled.Replays}, "
            + $"segments {compiled.Segments}, recomputes {compiled.Recomputes}, scratch {compiled.ScratchBytes} B");
        Assert.True(compiled.Replays > 0, $"{label}: the step graph never replayed: {compiled.CaptureFailure}");
        Assert.True(lossDiff <= 1e-5f * (1 + Math.Abs(eagerLoss)), $"{label}: loss differs from eager by {lossDiff}");
        Assert.True(worstRel <= 1e-5f, $"{label}: a gradient differs from eager by {worstAbs} (rel {worstRel})");
        // Every iteration's adapter got a gradient of its own, and the shared block one summed over all iterations.
        if (adaptersTrained < 0) adaptersTrained = iterations;
        for (int i = 0; i < adaptersTrained; i++)
            Assert.True(compiled.FirstGradients[2 + 3 * i].Any(v => v != 0f), $"{label}: iteration {i}'s adapter got no gradient");
        for (int i = 1; i < adaptersTrained; i++)
            Assert.NotEqual(compiled.FirstGradients[2], compiled.FirstGradients[2 + 3 * i]);
    }

    [SkippableTheory]
    [InlineData(4, false)]
    [InlineData(4, true)]
    [InlineData(8, true)]
    public void WeightTiedLoop_CompiledStep_MatchesEagerGradients_SharedAndPerIteration(int iterations, bool checkpoint)
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var model = new LoopModel(rows: 128, width: 64, hidden: 256, iterations);
            var (eagerLoss, eagerGrads) = RunEager(model.Fresh(), p => model.LoopLoss(engine, p, checkpoint));

            var compiled = RunCompiled(model.Fresh(), steps: 8, p => (model.LoopLoss(engine, p, checkpoint), null));
            AssertParity(eagerLoss, eagerGrads, compiled, iterations, $"loop k={iterations} checkpoint={checkpoint}");
            if (checkpoint)
            {
                Assert.Equal(iterations, compiled.Segments);
                Assert.True(compiled.Recomputes > 0, "no iteration was recomputed");
            }
            else Assert.Equal(0, compiled.Segments);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [SkippableFact]
    public void WeightTiedLoop_SharedGradient_IsTheSumOverIterations()
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int k = 3;
            var model = new LoopModel(rows: 64, width: 32, hidden: 64, k);
            var (_, tied) = RunEager(model.Fresh(), p => model.LoopLoss(engine, p, checkpoint: false));
            // Untied: iteration i reads its own copy of W1, W2 (same values); the tied gradient is the copies' sum.
            var baseParams = model.Fresh();
            var copies = Enumerable.Range(0, k).Select(_ => new[] { Clone(baseParams[0]), Clone(baseParams[1]) }).ToArray();
            var sources = copies.SelectMany(c => c).ToArray();
            float[] sumW1, sumW2;
            using (var tape = new GradientTape<float>())
            {
                var h = model.X;
                for (int i = 0; i < k; i++)
                {
                    var p = (Tensor<float>[])baseParams.Clone();
                    p[0] = copies[i][0]; p[1] = copies[i][1];
                    h = LoopModel.Body(engine, p, h, i);
                }
                var loss = engine.TensorMultiplyScalar(engine.ReduceSum(model.TokenLoss(engine, h), null), 1f / model.Rows);
                var g = tape.ComputeGradients(loss, sources: sources);
                sumW1 = Sum(copies.Select(c => g[c[0]].ToArray()));
                sumW2 = Sum(copies.Select(c => g[c[1]].ToArray()));
            }
            var compiled = RunCompiled(model.Fresh(), steps: 4, p => (model.LoopLoss(engine, p, checkpoint: true), null));
            var (a1, r1) = MaxDiff(sumW1, tied[0]);
            var (a2, r2) = MaxDiff(sumW2, tied[1]);
            var (c1, cr1) = MaxDiff(sumW1, compiled.LastGradients[0]);
            var (c2, cr2) = MaxDiff(sumW2, compiled.LastGradients[1]);
            _output.WriteLine($"sum of untied copies vs tied eager: W1 {a1:E3} (rel {r1:E3}), W2 {a2:E3} (rel {r2:E3}); "
                + $"vs compiled checkpointed: W1 {c1:E3} (rel {cr1:E3}), W2 {c2:E3} (rel {cr2:E3})");
            Assert.True(Math.Max(r1, r2) <= 1e-5f && Math.Max(cr1, cr2) <= 1e-5f);
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }

        static Tensor<float> Clone(Tensor<float> t) => new(t.Shape.ToArray(), new Vector<float>(t.ToArray()));
        static float[] Sum(IEnumerable<float[]> parts)
        {
            float[]? total = null;
            foreach (var part in parts)
            {
                if (total is null) { total = (float[])part.Clone(); continue; }
                for (int i = 0; i < total.Length; i++) total[i] += part[i];
            }
            return total!;
        }
    }

    [SkippableFact]
    public void WeightTiedLoop_PerIterationCheckpointing_KeepsActivationMemoryFlatInK()
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var scratch = new Dictionary<(int, bool), long>();
            foreach (int k in new[] { 4, 8, 16 })
                foreach (bool checkpoint in new[] { false, true })
                {
                    var model = new LoopModel(rows: 1024, width: 128, hidden: 512, k);
                    backendProof.TryGetDeviceMemory(out ulong freeBefore, out _);
                    var run = RunCompiled(model.Fresh(), steps: 8, p => (model.LoopLoss(engine, p, checkpoint), null));
                    backendProof.TryGetDeviceMemory(out ulong freeAfter, out _);
                    scratch[(k, checkpoint)] = run.ScratchBytes;
                    _output.WriteLine($"k={k,2} checkpoint={checkpoint,-5}: plan scratch {run.ScratchBytes / 1048576.0:F2} MiB, "
                        + $"device free before {freeBefore / 1048576.0:F0} MiB / after the plan's dispose {freeAfter / 1048576.0:F0} MiB, "
                        + $"replays {run.Replays}, recomputes {run.Recomputes}");
                    Assert.True(run.Replays > 0, run.CaptureFailure);
                }
            long plainGrowth = scratch[(16, false)] - scratch[(4, false)];
            long checkpointGrowth = scratch[(16, true)] - scratch[(4, true)];
            _output.WriteLine($"scratch growth k=4 -> 16: plain {plainGrowth / 1048576.0:F2} MiB, checkpointed {checkpointGrowth / 1048576.0:F2} MiB");
            Assert.True(scratch[(16, true)] < scratch[(16, false)], "checkpointing did not lower the k=16 plan's scratch");
            Assert.True(checkpointGrowth * 2 < plainGrowth,
                $"checkpointed scratch grew {checkpointGrowth} B from k=4 to k=16, the plain plan's {plainGrowth} B");
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    [SkippableTheory]
    [InlineData(0.7, false)]
    [InlineData(0.7, true)]
    [InlineData(1.0, true)]
    public void AdaptiveExit_CompiledStep_MatchesEager_AndReplaysWithTheMaskOnTheDevice(double threshold, bool checkpoint)
    {
        using var backendProof = RequireCuda();
        using var engine = new DirectGpuTensorEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            const int k = 6;
            const float beta = 0.05f;
            var model = new LoopModel(rows: 128, width: 64, hidden: 256, k, halting: true);
            AdaptiveLoopResult<float>? eagerExit = null;
            var (eagerLoss, eagerGrads) = RunEager(model.Fresh(), p =>
            {
                var (loss, exit) = model.AdaptiveLoss(engine, p, threshold, checkpoint, beta);
                eagerExit = exit;
                return loss;
            });
            float eagerLoops = eagerExit!.LoopsRun.ToArray()[0];
            float eagerEntropy = eagerExit.Entropy.ToArray()[0];
            float eagerExpected = eagerExit.ExpectedLoops.ToArray()[0];

            // The exit distribution sums to 1 per token; a masked token's state stops changing at its exit, so the
            // iterations whose state changed match LoopsRun.
            var probs = eagerExit.ExitProbabilities.Select(t => t.ToArray()).ToArray();
            for (int r = 0; r < model.Rows; r++)
                Assert.InRange(probs.Sum(pt => (double)pt[r]), 1 - 1e-5, 1 + 1e-5);
            var states = new[] { model.X.ToArray() }.Concat(eagerExit.States.Select(s => s.ToArray())).ToArray();
            double changed = 0;
            for (int r = 0; r < model.Rows; r++)
                for (int t = 1; t <= k; t++)
                {
                    bool moved = false;
                    for (int c = 0; c < model.Width && !moved; c++)
                        moved = states[t][r * model.Width + c] != states[t - 1][r * model.Width + c];
                    if (moved) changed++;
                }
            double changedMean = changed / model.Rows;
            _output.WriteLine($"eager: loops run {eagerLoops:R} (state changed on {changedMean:R} iterations per token), "
                + $"expected loops {eagerExpected:R}, entropy {eagerEntropy:R}");
            Assert.True(Math.Abs(changedMean - eagerLoops) < 1e-6, "the mask's loop count is not the iterations that moved the state");
            if (threshold < 1) Assert.True(eagerLoops < k && eagerLoops > 1, $"threshold {threshold}: loops run {eagerLoops}");
            else Assert.Equal((float)k, eagerLoops);

            var compiled = RunCompiled(model.Fresh(), steps: 8, p =>
            {
                var (loss, exit) = model.AdaptiveLoss(engine, p, threshold, checkpoint, beta);
                return (loss, exit);
            });
            _output.WriteLine($"compiled: loops run {compiled.LoopsRun:R}, expected loops {compiled.ExpectedLoops:R}, entropy {compiled.Entropy:R}");
            // An iteration no token reaches (every token exited before it) leaves its adapter untrained, eagerly too.
            AssertParity(eagerLoss, eagerGrads, compiled, k, $"adaptive k={k} q={threshold} checkpoint={checkpoint}",
                adaptersTrained: 2);
            Assert.Equal(eagerLoops, compiled.LoopsRun);
            Assert.True(Math.Abs(eagerEntropy - compiled.Entropy) <= 1e-5f, $"entropy {eagerEntropy} vs {compiled.Entropy}");
            Assert.True(Math.Abs(eagerExpected - compiled.ExpectedLoops) <= 1e-5f, $"expected loops {eagerExpected} vs {compiled.ExpectedLoops}");
            // The halting head is trained (through p and the entropy term).
            Assert.True(compiled.FirstGradients[^2].Any(v => v != 0f) && compiled.FirstGradients[^1].Any(v => v != 0f));
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }
}
