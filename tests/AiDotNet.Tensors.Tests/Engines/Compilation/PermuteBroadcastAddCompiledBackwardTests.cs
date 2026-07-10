using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// AiDotNet #1841 regression: the compiled-plan backward for the
/// <c>TensorPermute + TensorMatMul + TensorBroadcastAdd</c> chain used by every
/// NBEATS / NHiTS block (paper §3.2 doubly-residual MLP: permute input to
/// column-major, matmul with weight, broadcast-add bias) produces exploding Adam
/// updates on the linked Tensors build. The junior dev's PR body reports
/// weights blowing up to ~1e13 while the reported loss barely moves — a clear
/// sign that the compiled gradient disagrees with the eager gradient.
///
/// <para>The eager gradient is the ground truth (finite-difference-verified in
/// other tests). This test compares the compiled backward output against the
/// eager backward output for the exact NBEATS block forward chain. If they
/// disagree beyond fp32 tolerance, the bug is confirmed and we know which op
/// (weight vs. bias) diverges.</para>
/// </summary>
public class PermuteBroadcastAddCompiledBackwardTests
{
    /// <summary>
    /// Minimal repro of the NBEATS block forward chain. Batch B=4, input dim
    /// inSize=3, output dim outSize=2 — small enough to reason about but
    /// exercises the exact op sequence:
    /// <c>x[B,in]  ─Permute[1,0]→  x[in,B]  ─MatMul(W[out,in], x[in,B])→  linear[out,B]  ─BroadcastAdd(linear, biasCol[out,1])→  y[out,B]</c>
    /// Then a ReduceSum-to-scalar loss so a single scalar backward has clear
    /// analytic gradients w.r.t. W and bias.
    /// </summary>
    [Fact]
    public void CompiledBackward_MatchesEager_ForNBEATSBlockForwardChain()
    {
        var engine = new CpuEngine();
        const int B = 4, inSize = 3, outSize = 2;

        // Deterministic parameters — hand-crafted so a mismatch reproduces every run.
        var input = new Tensor<float>(new float[]
        {
            1f, 2f, 3f,
            4f, 5f, 6f,
            7f, 8f, 9f,
            10f, 11f, 12f,
        }, new[] { B, inSize });
        var weight = new Tensor<float>(new float[]
        {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
        }, new[] { outSize, inSize });
        var bias = new Tensor<float>(new float[] { 0.01f, 0.02f }, new[] { outSize });

        // ── EAGER PATH via GradientTape<T> ────────────────────────────────
        Tensor<float> weightGradEager, biasGradEager;
        using (var tape = new GradientTape<float>())
        {
var x = engine.TensorPermute(input, new[] { 1, 0 });         // [in, B]
            var linear = engine.TensorMatMul(weight, x);                 // [out, B]
            var biasCol = engine.Reshape(bias, new[] { outSize, 1 });    // [out, 1]
            var y = engine.TensorBroadcastAdd(linear, biasCol);          // [out, B]
            var loss = engine.ReduceSum(y, null);                        // scalar
            var grads = tape.ComputeGradients(loss, new[] { weight, bias });
            weightGradEager = grads[weight];
            biasGradEager = grads[bias];
        }

        // ── COMPILED PATH via CompileTraining + plan.Step() ───────────────
        Tensor<float> weightGradCompiled, biasGradCompiled;
        {
            var weightC = new Tensor<float>((float[])weight.ToArray().Clone(), new[] { outSize, inSize });
            var biasC = new Tensor<float>((float[])bias.ToArray().Clone(), new[] { outSize });
            var inputC = new Tensor<float>((float[])input.ToArray().Clone(), new[] { B, inSize });

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var x = engine.TensorPermute(inputC, new[] { 1, 0 });
                var linear = engine.TensorMatMul(weightC, x);
                var biasCol = engine.Reshape(biasC, new[] { outSize, 1 });
                var y = engine.TensorBroadcastAdd(linear, biasCol);
                engine.ReduceSum(y, null);
                plan = scope.CompileTraining(new[] { weightC, biasC });
            }

            using (plan)
            {
                plan.Step();
                // The plan writes gradients into weightC.Grad / biasC.Grad after Step().
                Assert.NotNull(weightC.Grad);
                Assert.NotNull(biasC.Grad);
                weightGradCompiled = weightC.Grad!;
                biasGradCompiled = biasC.Grad!;
            }
        }

        // ── VERIFY: compiled must match eager within fp32 tolerance ───────
        AssertTensorNear(weightGradEager, weightGradCompiled, tol: 1e-5f,
            label: "weight gradient (NBEATS-block compiled backward vs. eager)");
        AssertTensorNear(biasGradEager, biasGradCompiled, tol: 1e-5f,
            label: "bias gradient (NBEATS-block compiled backward vs. eager)");
    }

    /// <summary>
    /// Same forward chain with an MSE-style loss (matches NBEATS's actual loss)
    /// so gradOut is not the trivial all-ones from ReduceSum-only. Exercises
    /// the broadcast-reduce backward with non-uniform gradient values.
    /// </summary>
    [Fact]
    public void CompiledBackward_MatchesEager_ForNBEATSBlockWithMseLoss()
    {
        var engine = new CpuEngine();
        const int B = 4, inSize = 3, outSize = 2;

        var input = new Tensor<float>(new float[]
        {
            1f, 2f, 3f,
            4f, 5f, 6f,
            7f, 8f, 9f,
            10f, 11f, 12f,
        }, new[] { B, inSize });
        var weight = new Tensor<float>(new float[]
        {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
        }, new[] { outSize, inSize });
        var bias = new Tensor<float>(new float[] { 0.01f, 0.02f }, new[] { outSize });
        // Target for MSE: same shape as forward output [out, B].
        var target = new Tensor<float>(new float[]
        {
            0.5f, 1.0f, 1.5f, 2.0f,
            0.6f, 1.1f, 1.6f, 2.1f,
        }, new[] { outSize, B });

        Tensor<float> weightGradEager, biasGradEager;
        using (var tape = new GradientTape<float>())
        {
var x = engine.TensorPermute(input, new[] { 1, 0 });
            var linear = engine.TensorMatMul(weight, x);
            var biasCol = engine.Reshape(bias, new[] { outSize, 1 });
            var y = engine.TensorBroadcastAdd(linear, biasCol);
            var diff = engine.TensorSubtract(y, target);
            var sq = engine.TensorMultiply(diff, diff);
            var loss = engine.ReduceMean(sq, new[] { 0, 1 }, keepDims: false);
            var grads = tape.ComputeGradients(loss, new[] { weight, bias });
            weightGradEager = grads[weight];
            biasGradEager = grads[bias];
        }

        Tensor<float> weightGradCompiled, biasGradCompiled;
        {
            var weightC = new Tensor<float>((float[])weight.ToArray().Clone(), new[] { outSize, inSize });
            var biasC = new Tensor<float>((float[])bias.ToArray().Clone(), new[] { outSize });
            var inputC = new Tensor<float>((float[])input.ToArray().Clone(), new[] { B, inSize });
            var targetC = new Tensor<float>((float[])target.ToArray().Clone(), new[] { outSize, B });

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var x = engine.TensorPermute(inputC, new[] { 1, 0 });
                var linear = engine.TensorMatMul(weightC, x);
                var biasCol = engine.Reshape(biasC, new[] { outSize, 1 });
                var y = engine.TensorBroadcastAdd(linear, biasCol);
                var diff = engine.TensorSubtract(y, targetC);
                var sq = engine.TensorMultiply(diff, diff);
                engine.ReduceMean(sq, new[] { 0, 1 }, keepDims: false);
                plan = scope.CompileTraining(new[] { weightC, biasC });
            }

            using (plan)
            {
                plan.Step();
                Assert.NotNull(weightC.Grad);
                Assert.NotNull(biasC.Grad);
                weightGradCompiled = weightC.Grad!;
                biasGradCompiled = biasC.Grad!;
            }
        }

        AssertTensorNear(weightGradEager, weightGradCompiled, tol: 1e-5f,
            label: "weight gradient (NBEATS-block + MSE compiled backward vs. eager)");
        AssertTensorNear(biasGradEager, biasGradCompiled, tol: 1e-5f,
            label: "bias gradient (NBEATS-block + MSE compiled backward vs. eager)");
    }

    /// <summary>
    /// Multi-step Adam training loop through the compiled + fused-optimizer path
    /// (the path <c>NBEATSModel.TryTrainGpuResident</c> uses via
    /// <c>CompiledTapeTrainingStep&lt;T&gt;.TryStepWithFusedOptimizer</c>).
    /// Runs 20 identical steps and asserts the loss trajectory is finite and
    /// bounded — if the fused Adam kernel corrupts state or the persistent-input
    /// mechanism feeds stale data, weights explode within a few steps and the
    /// loss goes non-finite (mirrors the NBEATS "1e13 weights" symptom).
    /// </summary>
    [Fact]
    public void CompiledFusedAdamLoop_LossStaysFinite_ForNBEATSBlockChain()
    {
        var engine = new CpuEngine();
        const int B = 4, inSize = 3, outSize = 2, steps = 20;

        var input = new Tensor<float>(new float[]
        {
            1f, 2f, 3f,
            4f, 5f, 6f,
            7f, 8f, 9f,
            10f, 11f, 12f,
        }, new[] { B, inSize });
        var weight = new Tensor<float>(new float[]
        {
            0.1f, 0.2f, 0.3f,
            0.4f, 0.5f, 0.6f,
        }, new[] { outSize, inSize });
        var bias = new Tensor<float>(new float[] { 0.01f, 0.02f }, new[] { outSize });
        var target = new Tensor<float>(new float[]
        {
            0.5f, 1.0f, 1.5f, 2.0f,
            0.6f, 1.1f, 1.6f, 2.1f,
        }, new[] { outSize, B });

        // Compile ONCE with graph mode; replay Step() `steps` times.
        // ConfigureOptimizerFloat wires Adam into the plan so plan.Step() runs
        // forward + backward + fused Adam update in one on-device call.
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var x = engine.TensorPermute(input, new[] { 1, 0 });
            var linear = engine.TensorMatMul(weight, x);
            var biasCol = engine.Reshape(bias, new[] { outSize, 1 });
            var y = engine.TensorBroadcastAdd(linear, biasCol);
            var diff = engine.TensorSubtract(y, target);
            var sq = engine.TensorMultiply(diff, diff);
            engine.ReduceMean(sq, new[] { 0, 1 }, keepDims: false);
            plan = scope.CompileTraining(new[] { weight, bias });
        }

        var losses = new float[steps];
        using (plan)
        {
            plan.ConfigureOptimizer(
                OptimizerType.Adam,
                learningRate: 0.01f,
                beta1: 0.9f, beta2: 0.999f, eps: 1e-8f, weightDecay: 0f);

            for (int s = 0; s < steps; s++)
            {
                var lossT = plan.Step();
                losses[s] = lossT[0];
            }
        }

        // Every step's loss must be finite. Non-finite anywhere = the "exploding
        // Adam updates" symptom. (NBEATS's guard fires on IsNaN or IsInfinity of
        // the reported step loss — this asserts exactly that.)
        for (int s = 0; s < steps; s++)
        {
            Assert.True(!float.IsNaN(losses[s]) && !float.IsInfinity(losses[s]),
                $"Step {s} loss is non-finite ({losses[s]:R}) — fused-Adam divergence on NBEATS-block chain. " +
                $"Prior losses: [{string.Join(", ", System.Linq.Enumerable.Range(0, s).Select(i => losses[i].ToString("R")))}]");
        }

        // NBEATS's stricter guard also fires when step_N > 1e3 AND step_N > step_0 * 1e3.
        for (int s = 1; s < steps; s++)
        {
            bool exploding = losses[s] > 1e3f && losses[s] > losses[0] * 1e3f;
            Assert.False(exploding,
                $"Step {s} loss ({losses[s]:R}) exceeds NBEATS's explosion guard (>1e3 AND >1000× step 0 loss {losses[0]:R}). " +
                $"Fused-Adam divergence confirmed.");
        }

        // Sanity: loss should trend DOWN over 20 Adam steps on this trivial fixture
        // (Adam on a small quadratic converges within a few steps).
        Assert.True(losses[steps - 1] < losses[0] * 1.1f,
            $"Loss did not decrease meaningfully over {steps} steps: {losses[0]:R} → {losses[steps - 1]:R}. " +
            $"Fused Adam step may be running but not updating weights correctly.");
    }

    /// <summary>
    /// Exact NBEATS doubly-residual stack pattern (paper §3.2): two blocks share
    /// the same Permute+MatMul+BroadcastAdd forward, block N's output is
    /// SUBTRACTED from the residual fed to block N+1, and per-block forecasts
    /// are SUMMED. The subtract-from-residual + sum-of-forecasts chain is what
    /// the junior dev's NBEATS PR comment blames for the "compiled fused
    /// plan produces exploding Adam updates" divergence — this test replays
    /// that exact op graph through the fused-Adam plan for 20 steps to catch
    /// any interaction the single-block tests above wouldn't surface.
    /// </summary>
    [Fact]
    public void CompiledFusedAdam_DoublyResidualStack_LossStaysFinite()
    {
        var engine = new CpuEngine();
        const int B = 4, L = 6, H = 3, steps = 20;
        // Two blocks with the NBEATS shape: input=[B,L] -> permute -> matmul(W_bc [L,L]) -> bias_bc [L]
        // for backcast, and matmul(W_fc [H,L]) -> bias_fc [H] for forecast. Deterministic init.
        var input = new Tensor<float>(new float[]
        {
            1f, 2f, 3f, 4f, 5f, 6f,
            2f, 3f, 4f, 5f, 6f, 7f,
            3f, 4f, 5f, 6f, 7f, 8f,
            4f, 5f, 6f, 7f, 8f, 9f,
        }, new[] { B, L });

        // Block 1
        var w1_bc = new Tensor<float>(RandArr(L * L, seed: 11), new[] { L, L });
        var b1_bc = new Tensor<float>(new float[L], new[] { L });
        var w1_fc = new Tensor<float>(RandArr(H * L, seed: 12), new[] { H, L });
        var b1_fc = new Tensor<float>(new float[H], new[] { H });
        // Block 2
        var w2_bc = new Tensor<float>(RandArr(L * L, seed: 21), new[] { L, L });
        var b2_bc = new Tensor<float>(new float[L], new[] { L });
        var w2_fc = new Tensor<float>(RandArr(H * L, seed: 22), new[] { H, L });
        var b2_fc = new Tensor<float>(new float[H], new[] { H });

        var target = new Tensor<float>(new float[]
        {
            1f, 1f, 1f, 1f,
            1f, 1f, 1f, 1f,
            1f, 1f, 1f, 1f,
        }, new[] { H, B });

        Tensor<float> Block(Tensor<float> resIn, Tensor<float> wBc, Tensor<float> bBc, Tensor<float> wFc, Tensor<float> bFc,
            out Tensor<float> newRes, out Tensor<float> fc)
        {
            var x = engine.TensorPermute(resIn, new[] { 1, 0 });                    // [L, B]
            var bcLin = engine.TensorMatMul(wBc, x);                                 // [L, B]
            var bcCol = engine.Reshape(bBc, new[] { L, 1 });
            var bc = engine.TensorBroadcastAdd(bcLin, bcCol);                        // [L, B]
            var bcT = engine.TensorPermute(bc, new[] { 1, 0 });                      // [B, L]
            newRes = engine.TensorSubtract(resIn, bcT);                              // residual chain

            var fcLin = engine.TensorMatMul(wFc, x);                                 // [H, B]
            var fcCol = engine.Reshape(bFc, new[] { H, 1 });
            fc = engine.TensorBroadcastAdd(fcLin, fcCol);                            // [H, B]
            return fc;
        }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            Block(input, w1_bc, b1_bc, w1_fc, b1_fc, out var res1, out var fc1);
            Block(res1, w2_bc, b2_bc, w2_fc, b2_fc, out _, out var fc2);
            var agg = engine.TensorAdd(fc1, fc2);                                    // [H, B] doubly-residual sum
            var diff = engine.TensorSubtract(agg, target);
            var sq = engine.TensorMultiply(diff, diff);
            engine.ReduceMean(sq, new[] { 0, 1 }, keepDims: false);
            plan = scope.CompileTraining(new[] { w1_bc, b1_bc, w1_fc, b1_fc, w2_bc, b2_bc, w2_fc, b2_fc });
        }

        var losses = new float[steps];
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f,
                beta1: 0.9f, beta2: 0.999f, eps: 1e-8f, weightDecay: 0f);
            for (int s = 0; s < steps; s++)
                losses[s] = plan.Step()[0];
        }

        for (int s = 0; s < steps; s++)
        {
            Assert.True(!float.IsNaN(losses[s]) && !float.IsInfinity(losses[s]),
                $"Step {s} loss non-finite ({losses[s]:R}) — doubly-residual stack triggered fused-Adam divergence. " +
                $"Trajectory: [{string.Join(", ", losses[..(s+1)].Select(v => v.ToString("R")))}]");
        }
        // Sanity: doubly-residual stack + Adam should converge on the constant target.
        Assert.True(losses[steps - 1] < losses[0],
            $"Loss did not decrease over {steps} steps on doubly-residual stack: {losses[0]:R} → {losses[steps - 1]:R}");

        // Weights must stay finite — the junior dev reported "1e13 weights"; assert every param is bounded.
        AssertAllFinite(w1_bc, "w1_bc"); AssertAllFinite(b1_bc, "b1_bc");
        AssertAllFinite(w1_fc, "w1_fc"); AssertAllFinite(b1_fc, "b1_fc");
        AssertAllFinite(w2_bc, "w2_bc"); AssertAllFinite(b2_bc, "b2_bc");
        AssertAllFinite(w2_fc, "w2_fc"); AssertAllFinite(b2_fc, "b2_fc");
    }

    private static float[] RandArr(int n, int seed)
    {
        var rng = new System.Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() - 0.5) * 0.2);
        return a;
    }

    private static void AssertAllFinite(Tensor<float> t, string label)
    {
        var arr = t.ToArray();
        for (int i = 0; i < arr.Length; i++)
            Assert.True(!float.IsNaN(arr[i]) && !float.IsInfinity(arr[i]) && System.MathF.Abs(arr[i]) < 1e10f,
                $"{label}[{i}] = {arr[i]:R} — fused Adam produced non-finite or exploding weight");
    }

    private static void AssertTensorNear(Tensor<float> expected, Tensor<float> actual, float tol, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        var e = expected.ToArray();
        var a = actual.ToArray();
        int mismatches = 0;
        float maxAbsDiff = 0f;
        int worstIdx = -1;
        for (int i = 0; i < e.Length; i++)
        {
            float diff = System.MathF.Abs(e[i] - a[i]);
            if (diff > maxAbsDiff) { maxAbsDiff = diff; worstIdx = i; }
            if (diff > tol) mismatches++;
        }
        if (mismatches > 0)
        {
            Assert.Fail($"{label}: {mismatches}/{e.Length} elements diverge beyond tol={tol}. " +
                        $"Worst @ idx={worstIdx}: expected={e[worstIdx]:R} actual={a[worstIdx]:R} diff={maxAbsDiff:R}");
        }
    }
}
