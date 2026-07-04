using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Regression for the fused compiled-TRAINING GroupNorm stale-statistics bug
/// (surfaced by ooples/AiDotNet #1750: ODISE's SD-UNet trained on the fused path
/// and DIVERGED — CrossEntropy loss 9231.58 -> 9244.10 over 30 Adam steps — while
/// the eager tape reached 9188.04; the fix makes the fused path reach 9188.02).
///
/// Root cause: the specialized GroupNorm forward in <c>CompiledTrainingPlan</c>
/// replayed <c>GroupNormInto(o, inp, …, out _, out _)</c>, DISCARDING the freshly
/// computed mean/variance. The GroupNorm BACKWARD reads mean/variance from the
/// captured <c>savedState</c>; because the forward never wrote the current step's
/// statistics back, the backward ran on stale, trace-time stats. Once the GroupNorm
/// input drifts (its upstream weights update every step), the saved mean/var no
/// longer match the input, so gradGamma/gradBeta/dL-dInput are mis-computed — the
/// same class of bug LayerNorm hit and had fixed in #1331.
///
/// The test traces the plan with one input, then DRIFTS the GroupNorm input before
/// stepping (mirroring an activation fed by an upstream weight that moved since the
/// trace). With lr=0 there is no multi-step compounding, so the compiled gradient
/// must equal a fresh eager-tape gradient computed on the drifted input — it only
/// does when the forward refreshes the saved statistics each replay.
/// </summary>
[Collection("CompilationGlobalState")]
public class GroupNormCompiledTrainingStalenessTests
{
    [Fact]
    public void GroupNorm_CompiledBackward_UsesFreshStats_AfterInputDrifts()
    {
        const int N = 1, C = 4, H = 4, W = 4, numGroups = 2;
        const double epsilon = 1e-5;

        var rng = new System.Random(1750);
        int len = N * C * H * W;
        var traceInput = new float[len]; // input present at trace time
        var driftInput = new float[len]; // a DIFFERENT input the step actually sees
        var tgtData = new float[len];
        var gData = new float[C];
        var bData = new float[C];
        for (int i = 0; i < len; i++)
        {
            traceInput[i] = (float)(rng.NextDouble() - 0.5);
            driftInput[i] = (float)(rng.NextDouble() - 0.5);
            tgtData[i] = (float)(rng.NextDouble() - 0.5);
        }
        for (int i = 0; i < C; i++) { gData[i] = (float)(0.8 + 0.4 * rng.NextDouble()); bData[i] = (float)(rng.NextDouble() - 0.5); }

        // ---- COMPILED: trace with traceInput, then drift the input in place, Step (lr=0) ----
        var engineF = new CpuEngine();
        var inputF = new Tensor<float>(new[] { N, C, H, W });
        var targetF = new Tensor<float>(new[] { N, C, H, W });
        var gammaF = new Tensor<float>(new[] { C });
        var betaF = new Tensor<float>(new[] { C });
        for (int i = 0; i < len; i++) { inputF[i] = traceInput[i]; targetF[i] = tgtData[i]; }
        for (int i = 0; i < C; i++) { gammaF[i] = gData[i]; betaF[i] = bData[i]; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var gn = engineF.GroupNorm(inputF, numGroups, gammaF, betaF, epsilon, out _, out _);
            var diff = engineF.TensorSubtract(gn, targetF);
            var sq = engineF.TensorMultiply(diff, diff);
            engineF.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        // Drift the GroupNorm input between the trace and the step. The forward replay
        // re-reads this buffer; the saved mean/variance MUST be recomputed from it.
        for (int i = 0; i < len; i++) inputF[i] = driftInput[i];

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }
        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad");

        // ---- EAGER reference: gradient on the DRIFTED input (the correct answer) ----
        var engineE = new CpuEngine();
        var inputE = new Tensor<float>(new[] { N, C, H, W });
        var targetE = new Tensor<float>(new[] { N, C, H, W });
        var gammaE = new Tensor<float>(new[] { C });
        var betaE = new Tensor<float>(new[] { C });
        for (int i = 0; i < len; i++) { inputE[i] = driftInput[i]; targetE[i] = tgtData[i]; }
        for (int i = 0; i < C; i++) { gammaE[i] = gData[i]; betaE[i] = bData[i]; }

        Tensor<float> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<float>())
        {
            var gn = engineE.GroupNorm(inputE, numGroups, gammaE, betaE, epsilon, out _, out _);
            var diff = engineE.TensorSubtract(gn, targetE);
            var sq = engineE.TensorMultiply(diff, diff);
            var loss = engineE.ReduceSum(sq, null);
            var g = tape.ComputeGradients(loss, sources: new[] { gammaE, betaE });
            eagerGradGamma = g[gammaE];
            eagerGradBeta = g[betaE];
        }

        // Stale trace-time statistics make the compiled gradient diverge sharply from
        // the drifted-input eager gradient; fresh statistics match to FP tolerance.
        const float tol = 1e-4f;
        var lines = new System.Collections.Generic.List<string>();
        float maxG = 0, maxB = 0;
        for (int p = 0; p < C; p++)
        {
            float dG = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            float dB = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            maxG = System.Math.Max(maxG, dG);
            maxB = System.Math.Max(maxB, dB);
            lines.Add($"  ch[{p}]  γ eager={eagerGradGamma[p]:F6} fused={fusedGradGamma[p]:F6} |Δ|={dG:E3}   β eager={eagerGradBeta[p]:F6} fused={fusedGradBeta[p]:F6} |Δ|={dB:E3}");
        }
        Assert.True(maxG < tol && maxB < tol,
            $"Compiled GroupNorm backward ran on STALE trace-time mean/variance after the input drifted " +
            $"(max|Δγ|={maxG:E3}, max|Δβ|={maxB:E3}, tol {tol:E0}):\n" + string.Join("\n", lines));
    }
}
