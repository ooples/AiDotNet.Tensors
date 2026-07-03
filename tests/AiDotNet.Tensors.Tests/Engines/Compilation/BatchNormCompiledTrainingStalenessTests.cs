using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Probe: does the compiled-TRAINING BatchNorm forward have the SAME stale-statistics
/// bug GroupNorm had (and LayerNorm had, fixed in #1331)? The specialized BatchNorm
/// forward routes through BatchNormInferenceUnsafe with the savedState mean/variance
/// and does not recompute/refresh them, so once the BatchNorm input drifts across
/// training steps the backward runs on stale trace-time stats. FireRedASR (Conformer
/// conv modules use BatchNorm) trains on the fused path and its training invariants
/// (LossStrictlyDecreases / DifferentInputs_AfterTraining / MoreData) fail on 0.106.1.
/// This test drifts the input after the trace and compares the compiled gradient to a
/// fresh eager-tape gradient on the drifted input (lr=0, no compounding).
/// </summary>
[Collection("CompilationGlobalState")]
public class BatchNormCompiledTrainingStalenessTests
{
    [Fact]
    public void BatchNorm_CompiledBackward_UsesFreshStats_AfterInputDrifts()
    {
        // Rank-3 [C,H,W] float hits the specialized channels-first BatchNorm forward
        // (BatchNormInferenceUnsafe, CompiledTrainingPlan ~5439) — the path FireRedASR's
        // Conformer conv-module BatchNorm takes. A rank-2 [batch,features] input misses
        // it (routes to the generic refreshing path) and would not expose the bug.
        const int C = 6, H = 4, W = 4;
        const double epsilon = 1e-5;
        int batch = C, features = C; // gamma/beta length == channels

        var rng = new System.Random(1750);
        int len = C * H * W;
        var traceInput = new float[len];
        var driftInput = new float[len];
        var tgtData = new float[len];
        var gData = new float[features];
        var bData = new float[features];
        for (int i = 0; i < len; i++)
        {
            traceInput[i] = (float)(rng.NextDouble() - 0.5);
            driftInput[i] = (float)(rng.NextDouble() * 2.0 - 1.0); // clearly different distribution
            tgtData[i] = (float)(rng.NextDouble() - 0.5);
        }
        for (int i = 0; i < features; i++) { gData[i] = (float)(0.8 + 0.4 * rng.NextDouble()); bData[i] = (float)(rng.NextDouble() - 0.5); }

        // ---- COMPILED: trace with traceInput, drift the input, Step (lr=0) ----
        var engineF = new CpuEngine();
        var inputF = new Tensor<float>(new[] { C, H, W });
        var targetF = new Tensor<float>(new[] { C, H, W });
        var gammaF = new Tensor<float>(new[] { features });
        var betaF = new Tensor<float>(new[] { features });
        for (int i = 0; i < len; i++) { inputF[i] = traceInput[i]; targetF[i] = tgtData[i]; }
        for (int i = 0; i < features; i++) { gammaF[i] = gData[i]; betaF[i] = bData[i]; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var bn = engineF.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            var diff = engineF.TensorSubtract(bn, targetF);
            var sq = engineF.TensorMultiply(diff, diff);
            engineF.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        for (int i = 0; i < len; i++) inputF[i] = driftInput[i];

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }
        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad");

        // ---- EAGER reference on the DRIFTED input ----
        var engineE = new CpuEngine();
        var inputE = new Tensor<float>(new[] { C, H, W });
        var targetE = new Tensor<float>(new[] { C, H, W });
        var gammaE = new Tensor<float>(new[] { features });
        var betaE = new Tensor<float>(new[] { features });
        for (int i = 0; i < len; i++) { inputE[i] = driftInput[i]; targetE[i] = tgtData[i]; }
        for (int i = 0; i < features; i++) { gammaE[i] = gData[i]; betaE[i] = bData[i]; }

        Tensor<float> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<float>())
        {
            var bn = engineE.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var diff = engineE.TensorSubtract(bn, targetE);
            var sq = engineE.TensorMultiply(diff, diff);
            var loss = engineE.ReduceSum(sq, null);
            var g = tape.ComputeGradients(loss, sources: new[] { gammaE, betaE });
            eagerGradGamma = g[gammaE];
            eagerGradBeta = g[betaE];
        }

        const float tol = 1e-4f;
        var lines = new System.Collections.Generic.List<string>();
        float maxG = 0, maxB = 0;
        for (int p = 0; p < features; p++)
        {
            float dG = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            float dB = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            maxG = System.Math.Max(maxG, dG);
            maxB = System.Math.Max(maxB, dB);
            lines.Add($"  ch[{p}]  γ eager={eagerGradGamma[p]:F6} fused={fusedGradGamma[p]:F6} |Δ|={dG:E3}   β eager={eagerGradBeta[p]:F6} fused={fusedGradBeta[p]:F6} |Δ|={dB:E3}");
        }
        Assert.True(maxG < tol && maxB < tol,
            $"Compiled BatchNorm backward ran on STALE trace-time mean/variance after the input drifted " +
            $"(max|Δγ|={maxG:E3}, max|Δβ|={maxB:E3}, tol {tol:E0}):\n" + string.Join("\n", lines));
    }
}
