using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// #74: AMSGrad fused plan-dispatch. Before this change CompiledTrainingPlan's
/// ConfigureOptimizer threw <c>NotSupportedException("Optimizer type AMSGrad …")</c>,
/// so AiDotNet's FeedForwardNeuralNetwork — whose default optimizer is AMSGrad-mode
/// Adam (chosen for the #1332 drift fix) — silently fell back to the eager tape for
/// EVERY training run (the "compiled does nothing" symptom the AIsEval/parity harness
/// surfaced).
///
/// Two layers of coverage:
///  - Kernel-direct: drive <see cref="FusedOptimizer.AMSGradUpdateSimd(float*,float*,float*,float*,float*,int,float,float,float,float,int)"/>
///    with a KNOWN gradient sequence (rise then fall, so vMax exceeds v and the max
///    path actually matters) and compare against an independent textbook AMSGrad
///    reference. This validates the AVX2 update against the canonical formula
///    unambiguously (no graph-gradient semantics involved).
///  - Plan-level: ConfigureOptimizer(AMSGrad) must dispatch (no NotSupportedException),
///    update parameters in place, equal Adam on step 1 (when vMax == v), and diverge
///    from Adam once vMax > v (proving the vMax buffer is consulted, not aliased).
/// </summary>
public class ConfigureOptimizerAMSGradTests
{
    private const float Lr = 0.05f, B1 = 0.9f, B2 = 0.999f, Eps = 1e-8f;

    // ---- kernel-direct parity ------------------------------------------------

    /// <summary>Independent textbook AMSGrad replay over an explicit gradient sequence.</summary>
    private static float[] ReferenceAMSGradFloat(float[] w0, float[][] grads, float lr, float b1, float b2, float eps)
    {
        var w = (float[])w0.Clone();
        var m = new float[w.Length];
        var v = new float[w.Length];
        var vMax = new float[w.Length];
        for (int t = 1; t <= grads.Length; t++)
        {
            float bc1 = 1f - MathF.Pow(b1, t);
            float bc2 = 1f - MathF.Pow(b2, t);
            var g = grads[t - 1];
            for (int i = 0; i < w.Length; i++)
            {
                m[i] = b1 * m[i] + (1f - b1) * g[i];
                v[i] = b2 * v[i] + (1f - b2) * g[i] * g[i];
                vMax[i] = MathF.Max(vMax[i], v[i]);
                w[i] -= lr * (m[i] / bc1) / (MathF.Sqrt(vMax[i] / bc2) + eps);
            }
        }
        return w;
    }

    private static double[] ReferenceAMSGradDouble(double[] w0, double[][] grads, double lr, double b1, double b2, double eps)
    {
        var w = (double[])w0.Clone();
        var m = new double[w.Length];
        var v = new double[w.Length];
        var vMax = new double[w.Length];
        for (int t = 1; t <= grads.Length; t++)
        {
            double bc1 = 1.0 - Math.Pow(b1, t);
            double bc2 = 1.0 - Math.Pow(b2, t);
            var g = grads[t - 1];
            for (int i = 0; i < w.Length; i++)
            {
                m[i] = b1 * m[i] + (1.0 - b1) * g[i];
                v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
                vMax[i] = Math.Max(vMax[i], v[i]);
                w[i] -= lr * (m[i] / bc1) / (Math.Sqrt(vMax[i] / bc2) + eps);
            }
        }
        return w;
    }

    /// <summary>Gradient sequence that rises (steps 1..4) then falls (steps 5..12), so
    /// the running max vMax holds the early peak while v decays — the regime where
    /// AMSGrad's max-denominator differs from plain Adam.</summary>
    private static float[][] RiseThenFallGrads(int n, int seed)
    {
        var rng = new Random(seed);
        var baseG = new float[n];
        for (int i = 0; i < n; i++) baseG[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var grads = new float[12][];
        for (int t = 0; t < 12; t++)
        {
            float scale = t < 4 ? 1.0f : 0.05f;
            var g = new float[n];
            for (int i = 0; i < n; i++) g[i] = baseG[i] * scale;
            grads[t] = g;
        }
        return grads;
    }

    [Fact]
    public unsafe void AMSGradUpdateSimd_Float_MatchesCanonicalFormula()
    {
        const int n = 16; // >= 8 so the AVX2 path runs
        var rng = new Random(3);
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        var grads = RiseThenFallGrads(n, seed: 5);

        var reference = ReferenceAMSGradFloat(w, grads, Lr, B1, B2, Eps);

        var m = new float[n];
        var v = new float[n];
        var vMax = new float[n];
        for (int t = 1; t <= grads.Length; t++)
        {
            var g = (float[])grads[t - 1].Clone(); // kernel reads grad in place
            fixed (float* pW = w, pG = g, pM = m, pV = v, pVMax = vMax)
                FusedOptimizer.AMSGradUpdateSimd(pW, pG, pM, pV, pVMax, n, Lr, B1, B2, Eps, t);
        }

        double maxAbs = 0;
        for (int i = 0; i < n; i++) maxAbs = Math.Max(maxAbs, Math.Abs(w[i] - reference[i]));
        Assert.True(maxAbs < 1e-4,
            $"AMSGradUpdateSimd(float) diverged from the canonical AMSGrad formula. max |Δ| = {maxAbs}");
    }

    [Fact]
    public unsafe void AMSGradUpdateSimd_Double_MatchesCanonicalFormula()
    {
        const int n = 16;
        var rng = new Random(3);
        var w = new double[n];
        for (int i = 0; i < n; i++) w[i] = rng.NextDouble() * 2.0 - 1.0;
        var gradsF = RiseThenFallGrads(n, seed: 5);
        var grads = new double[gradsF.Length][];
        for (int t = 0; t < gradsF.Length; t++)
        {
            grads[t] = new double[n];
            for (int i = 0; i < n; i++) grads[t][i] = gradsF[t][i];
        }

        var reference = ReferenceAMSGradDouble(w, grads, Lr, B1, B2, Eps);

        var m = new double[n];
        var v = new double[n];
        var vMax = new double[n];
        for (int t = 1; t <= grads.Length; t++)
        {
            var g = (double[])grads[t - 1].Clone();
            fixed (double* pW = w, pG = g, pM = m, pV = v, pVMax = vMax)
                FusedOptimizer.AMSGradUpdateSimd(pW, pG, pM, pV, pVMax, n, Lr, B1, B2, Eps, t);
        }

        double maxAbs = 0;
        for (int i = 0; i < n; i++) maxAbs = Math.Max(maxAbs, Math.Abs(w[i] - reference[i]));
        Assert.True(maxAbs < 1e-12,
            $"AMSGradUpdateSimd(double) diverged from the canonical AMSGrad formula. max |Δ| = {maxAbs}");
    }

    // ---- plan-level wiring ---------------------------------------------------

    private static float[] RunPlanFloat(float[] init, OptimizerType opt, int steps)
    {
        var engine = new CpuEngine();
        var weight = new Tensor<float>(new[] { init.Length });
        for (int i = 0; i < init.Length; i++) weight[i] = init[i];

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(opt, Lr, B1, B2, Eps);
            for (int s = 0; s < steps; s++) plan.Step();
        }
        return weight.GetDataArray().AsSpan(0, init.Length).ToArray();
    }

    private static float[] SeedFloat(int n, int seed)
    {
        var rng = new Random(seed);
        var w = new float[n];
        for (int i = 0; i < n; i++) w[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return w;
    }

    /// <summary>
    /// Direct regression for the NotSupportedException that demoted every
    /// FeedForwardNeuralNetwork to the eager tape: ConfigureOptimizer(AMSGrad) must
    /// dispatch and mutate the parameter in place.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_AMSGradFloat_StepUpdatesParameterInPlace()
    {
        var init = SeedFloat(11, seed: 7);
        var post = RunPlanFloat(init, OptimizerType.AMSGrad, steps: 1);
        double maxAbs = 0;
        for (int i = 0; i < init.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(post[i] - init[i]));
        Assert.True(maxAbs > 0,
            $"AMSGrad(float) Step() did not update the parameter in place (dispatch missing?). max |Δ| = {maxAbs}");
    }

    [Fact]
    public void ConfigureOptimizer_AMSGradDouble_StepUpdatesParameterInPlace()
    {
        var engine = new CpuEngine();
        var rng = new Random(7);
        var weight = new Tensor<double>(new[] { 11 });
        var init = new double[11];
        for (int i = 0; i < 11; i++) { init[i] = rng.NextDouble() * 2.0 - 1.0; weight[i] = init[i]; }

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(weight, weight);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { weight });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.AMSGrad, Lr, B1, B2, Eps);
            plan.Step();
        }

        var post = weight.GetDataArray();
        double maxAbs = 0;
        for (int i = 0; i < 11; i++) maxAbs = Math.Max(maxAbs, Math.Abs(post[i] - init[i]));
        Assert.True(maxAbs > 0,
            $"AMSGrad(double) Step() did not update the parameter in place. max |Δ| = {maxAbs}");
    }

    /// <summary>
    /// On step 1, vMax = max(0, v) = v, so AMSGrad is identical to Adam through the
    /// plan. Pins the step-1 update (m / v / bias-correction wiring) independently of
    /// the vMax path.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_AMSGradFloat_Step1_EqualsAdam()
    {
        var init = SeedFloat(11, seed: 13);
        var ams = RunPlanFloat(init, OptimizerType.AMSGrad, steps: 1);
        var adam = RunPlanFloat(init, OptimizerType.Adam, steps: 1);
        double maxAbs = 0;
        for (int i = 0; i < init.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(ams[i] - adam[i]));
        Assert.True(maxAbs < 1e-6,
            $"AMSGrad and Adam must be identical on step 1 (vMax == v). max |Δ| = {maxAbs}");
    }

    /// <summary>
    /// Proves the vMax buffer is consulted: with loss = sum(w²), w shrinks toward 0 so
    /// the gradient and thus v decay; once vMax > v, AMSGrad's larger denominator takes
    /// smaller steps than Adam, so the trajectories must diverge measurably. If vMax
    /// were ignored (AMSGrad aliased to Adam), this stays ~0.
    /// </summary>
    [Fact]
    public void ConfigureOptimizer_AMSGradFloat_DivergesFromAdam_OverManySteps()
    {
        var init = SeedFloat(11, seed: 21);
        const int steps = 40;
        var ams = RunPlanFloat(init, OptimizerType.AMSGrad, steps);
        var adam = RunPlanFloat(init, OptimizerType.Adam, steps);
        double maxAbs = 0;
        for (int i = 0; i < init.Length; i++) maxAbs = Math.Max(maxAbs, Math.Abs(ams[i] - adam[i]));
        Assert.True(maxAbs > 1e-5,
            $"AMSGrad never diverged from Adam over {steps} steps — vMax appears unused. max |Δ| = {maxAbs}");
        foreach (var x in ams)
            Assert.True(!float.IsNaN(x) && !float.IsInfinity(x), "AMSGrad produced a non-finite parameter");
    }
}
