using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Issue #350 third-order residual: PR #351's two-bug repair landed but the
/// AiDotNet GraFPrint min-repro (testconsole/BnFusedGradDiff.cs in
/// ooples/AiDotNet branch fix/pr-1290-cluster-4-5) still shows a
/// single-element gradient slot divergence on a 4-feature
/// BatchNormalizationLayer:
///
///   forward output: bit-for-bit identical (eager vs compiled)
///   gamma_grad after 1 Adam step:
///     [0]  eager 0.99000001  fused 0.99000001  (match)
///     [1]  eager 0.99000001  fused 1.00999999  (OPPOSITE direction)
///     [2]  eager 0.99000001  fused 0.99000001  (match)
///     [3]  eager 0.99000001  fused 0.99000001  (match)
///
/// The update magnitude is correct (|Δ|=0.01, lr-bounded by Adam's first step)
/// but the SIGN is flipped at one specific channel index. The 2D
/// BatchNormBackward code itself is correct and shared by both paths, so the
/// mis-routing must happen in how the compiled plan captures or accumulates
/// the gradient outputs of BatchNorm — not in the backward math.
///
/// These tests pin the contract: an Adam step on a graph whose ONLY
/// trainable parameters are gamma + beta of a BatchNorm op must move every
/// channel of gamma in the same direction the eager-mode equivalent does.
/// </summary>
public class BatchNormGradSlotResidualTests
{
    /// <summary>
    /// Pinpoint: capture bnF BEFORE plan.Step() (when it should hold the
    /// eager-time copied values) and AFTER plan.Step() (when forward replay
    /// has overwritten it). Compare both with the eager BN result.
    /// </summary>
    [Fact]
    public void Pinpoint_FusedBN_PreVsPostStep_VsEager()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;
        var rng = new System.Random(42);
        var inputData = new float[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() - 0.5);

        // EAGER
        var inputE = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var gammaE = new Tensor<float>(new[] { features });
        var betaE = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1f; betaE[i] = 0f; }
        var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
        var snapE = new float[bnE.Length];
        for (int i = 0; i < bnE.Length; i++) snapE[i] = bnE[i];

        // FUSED — capture pre/post Step
        var inputF = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var gammaF = new Tensor<float>(new[] { features });
        var betaF = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1f; betaF[i] = 0f; }

        Tensor<float> bnF;
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            var sqF = engine.TensorMultiply(bnF, bnF);
            engine.ReduceSum(sqF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        // Capture BEFORE Step
        var snapPreStep = new float[bnF.Length];
        for (int i = 0; i < bnF.Length; i++) snapPreStep[i] = bnF[i];

        using (plan)
        {
            plan.Step();
        }

        // Capture AFTER Step
        var snapPostStep = new float[bnF.Length];
        for (int i = 0; i < bnF.Length; i++) snapPostStep[i] = bnF[i];

        var lines = new System.Collections.Generic.List<string>();
        lines.Add("  per-element diff:  pre-Step vs eager,  post-Step vs eager");
        bool anyPreDiff = false, anyPostDiff = false;
        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < features; f++)
            {
                int i = b * features + f;
                float dPre = System.Math.Abs(snapPreStep[i] - snapE[i]);
                float dPost = System.Math.Abs(snapPostStep[i] - snapE[i]);
                if (dPre >= 1e-6f) anyPreDiff = true;
                if (dPost >= 1e-6f) anyPostDiff = true;
                lines.Add($"  [{b},{f}]  eager={snapE[i],12:F8}  preStep={snapPreStep[i],12:F8} (Δ={dPre,9:E2})  postStep={snapPostStep[i],12:F8} (Δ={dPost,9:E2})");
            }
        }
        Assert.False(anyPreDiff || anyPostDiff,
            $"BN forward output diverges. preDiff={anyPreDiff}, postDiff={anyPostDiff}\n" + string.Join("\n", lines));
    }

    /// <summary>
    /// Sanity: calling engine.BatchNorm twice with bit-identical input/gamma/beta
    /// MUST produce bit-identical output. If this fails, something deeper than
    /// compile-mode is wrong (probably TensorAllocator reusing dirty buffers).
    /// </summary>
    [Fact]
    public void Sanity_BatchNorm_Twice_SameInput_SameOutput()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;
        var rng = new System.Random(42);
        var inputData = new float[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() - 0.5);

        var input1 = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < input1.Length; i++) input1[i] = inputData[i];
        var gamma1 = new Tensor<float>(new[] { features });
        var beta1 = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gamma1[i] = 1f; beta1[i] = 0f; }
        var out1 = engine.BatchNorm(input1, gamma1, beta1, epsilon, out _, out _);
        var snap1 = new float[out1.Length];
        for (int i = 0; i < out1.Length; i++) snap1[i] = out1[i];

        var input2 = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < input2.Length; i++) input2[i] = inputData[i];
        var gamma2 = new Tensor<float>(new[] { features });
        var beta2 = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gamma2[i] = 1f; beta2[i] = 0f; }
        var out2 = engine.BatchNorm(input2, gamma2, beta2, epsilon, out _, out _);

        var lines = new System.Collections.Generic.List<string>();
        bool anyDiff = false;
        for (int b = 0; b < batch; b++)
        {
            for (int f = 0; f < features; f++)
            {
                int i = b * features + f;
                float d = System.Math.Abs(snap1[i] - out2[i]);
                if (d >= 1e-6f) anyDiff = true;
                lines.Add($"  [{b},{f}]  out1={snap1[i],12:F8}  out2={out2[i],12:F8}  |Δ|={d,12:E6}");
            }
        }
        Assert.False(anyDiff, "engine.BatchNorm produced different outputs for same input on consecutive calls:\n" + string.Join("\n", lines));
    }


    /// <summary>
    /// BN → square (no target / subtract) → ReduceSum. If THIS passes,
    /// the bug emerges from the subtract-with-non-trainable-target step
    /// in MSE; if it fails, the bug is in compile-mode multiply backward
    /// when one operand is a non-leaf intermediate (BN's output).
    /// </summary>
    [Fact]
    public void BatchNorm_Then_Square_ReduceSum_Backward_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;

        var rng = new System.Random(42);
        var inputData = new float[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() - 0.5);

        // EAGER
        var inputE = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var gammaE = new Tensor<float>(new[] { features });
        var betaE = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0f; betaE[i] = 0.0f; }

        Tensor<float> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<float>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var sqE = engine.TensorMultiply(bnE, bnE);
            var sumE = engine.ReduceSum(sqE, null);
            var grads = tape.ComputeGradients(sumE, sources: new[] { gammaE, betaE });
            eagerGradGamma = grads[gammaE];
            eagerGradBeta = grads[betaE];
        }

        // FUSED
        var inputF = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var gammaF = new Tensor<float>(new[] { features });
        var betaF = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1.0f; betaF[i] = 0.0f; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            var sqF = engine.TensorMultiply(bnF, bnF);
            engine.ReduceSum(sqF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }

        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad not set");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad not set");

        var lines = new System.Collections.Generic.List<string>();
        bool anyMismatch = false;
        for (int p = 0; p < features; p++)
        {
            float dG = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            float dB = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            if (dG >= 1e-4f || dB >= 1e-4f) anyMismatch = true;
            lines.Add($"  ch[{p}]  gamma_grad eager={eagerGradGamma[p],12:F6} fused={fusedGradGamma[p],12:F6} |Δ|={dG,12:E6}    beta_grad eager={eagerGradBeta[p],12:F6} fused={fusedGradBeta[p],12:F6} |Δ|={dB,12:E6}");
        }
        Assert.False(anyMismatch, "compiled BN→sq→Sum backward diverges:\n" + string.Join("\n", lines));
    }

    /// <summary>
    /// Even-more-isolated BN repro: BN forward → ReduceSum (no subtract,
    /// no square, no target tensor at all). dL/dy = 1 for every element of
    /// BN's output. If grads diverge here, the bug is purely in the compiled
    /// BN backward path, not in MSE / subtract / multiply chain interactions.
    /// </summary>
    [Fact]
    public void BatchNorm_Then_ReduceSum_Backward_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;

        var rng = new System.Random(42);
        var inputData = new float[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() - 0.5);

        // EAGER
        var inputE = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var gammaE = new Tensor<float>(new[] { features });
        var betaE = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0f; betaE[i] = 0.0f; }

        Tensor<float> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<float>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var sumE = engine.ReduceSum(bnE, null);
            var grads = tape.ComputeGradients(sumE, sources: new[] { gammaE, betaE });
            eagerGradGamma = grads[gammaE];
            eagerGradBeta = grads[betaE];
        }

        // FUSED
        var inputF = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var gammaF = new Tensor<float>(new[] { features });
        var betaF = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1.0f; betaF[i] = 0.0f; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            engine.ReduceSum(bnF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }

        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad not set");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad not set");

        var lines = new System.Collections.Generic.List<string>();
        bool anyMismatch = false;
        for (int p = 0; p < features; p++)
        {
            float dG = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            float dB = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            if (dG >= 1e-4f || dB >= 1e-4f) anyMismatch = true;
            lines.Add($"  ch[{p}]  gamma_grad eager={eagerGradGamma[p],12:F6} fused={fusedGradGamma[p],12:F6} |Δ|={dG,12:E6}    beta_grad eager={eagerGradBeta[p],12:F6} fused={fusedGradBeta[p],12:F6} |Δ|={dB,12:E6}");
        }
        Assert.False(anyMismatch, "compiled BN→ReduceSum backward diverges:\n" + string.Join("\n", lines));
    }

    /// <summary>
    /// Sanity-check the simpler graph (no BatchNorm at all): differences in
    /// the BN test could be in BN backward OR in upstream ops (MSE / sub /
    /// mul / sum). This pure-multiply graph isolates the upstream backward
    /// chain. If THIS passes, the bug is BN-specific; if THIS fails, the
    /// bug is in a more fundamental compile-mode op like TensorMultiply
    /// backward against the same operand twice (a common autodiff trap).
    /// </summary>
    [Fact]
    public void SquareSum_Backward_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int n = 4;
        var rng = new System.Random(42);

        // Eager
        var xE = new Tensor<float>(new[] { n });
        for (int i = 0; i < n; i++) xE[i] = (float)(rng.NextDouble() - 0.5);
        Tensor<float> eagerGrad;
        using (var tape = new GradientTape<float>())
        {
            var sq = engine.TensorMultiply(xE, xE);
            var loss = engine.ReduceSum(sq, null);
            eagerGrad = tape.ComputeGradients(loss, sources: new[] { xE })[xE];
        }

        // Fused — re-seed RNG so xF gets identical values
        var xF = new Tensor<float>(new[] { n });
        var rngF = new System.Random(42);
        for (int i = 0; i < n; i++) xF[i] = (float)(rngF.NextDouble() - 0.5);

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var sq = engine.TensorMultiply(xF, xF);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { xF });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }

        var fusedGrad = xF.Grad ?? throw new System.InvalidOperationException("xF.Grad not set after Step()");

        var lines = new System.Collections.Generic.List<string>();
        bool anyMismatch = false;
        for (int p = 0; p < n; p++)
        {
            float d = System.Math.Abs(eagerGrad[p] - fusedGrad[p]);
            if (d >= 1e-5f) anyMismatch = true;
            lines.Add($"  x[{p}]={xE[p],12:F8}  eager_grad={eagerGrad[p],12:F6}  fused_grad={fusedGrad[p],12:F6}  |Δ|={d,12:E6}");
        }
        Assert.False(anyMismatch, "compiled (x*x).ReduceSum backward diverges per-element:\n" + string.Join("\n", lines));
    }


    /// <summary>
    /// 4-feature, batch-4 BatchNorm in 2D mode. Compute MSE loss against
    /// a fixed target, run one Adam step under each path, compare per-element
    /// post-step parameter values. The eager path is run via tape;
    /// compile-mode runs through CompiledTrainingPlan.Step.
    /// </summary>
    [Fact]
    public void BatchNorm_2D_GradGammaSlot_AdamStep_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const float lr = 0.01f;
        const double epsilon = 1e-5;

        // Deterministic input + target — same data drives both paths so any
        // divergence is in the gradient routing, not the input.
        var rng = new System.Random(42);
        var inputData = new float[batch * features];
        var targetData = new float[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < targetData.Length; i++) targetData[i] = (float)(rng.NextDouble() - 0.5);

        // ================ EAGER path ================
        var inputE = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var targetE = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < targetE.Length; i++) targetE[i] = targetData[i];
        var gammaE = new Tensor<float>(new[] { features });
        var betaE = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0f; betaE[i] = 0.0f; }

        using (var tape = new GradientTape<float>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var diffE = engine.TensorSubtract(bnE, targetE);
            var sqE = engine.TensorMultiply(diffE, diffE);
            var lossE = engine.ReduceSum(sqE, null);

            var grads = tape.ComputeGradients(lossE, sources: new[] { gammaE, betaE });
            var gradGammaE = grads[gammaE];
            var gradBetaE = grads[betaE];

            // One Adam step (β1=0.9, β2=0.999, ε=1e-8, single-step bias correction).
            const float b1 = 0.9f, b2 = 0.999f, adamEps = 1e-8f;
            float bc1 = 1 - b1, bc2 = 1 - b2;
            float oneMinusB1 = 1 - b1, oneMinusB2 = 1 - b2;
            for (int p = 0; p < features; p++)
            {
                // gamma update
                float gG = gradGammaE[p];
                float mG = oneMinusB1 * gG;
                float vG = oneMinusB2 * gG * gG;
                float mHatG = mG / bc1;
                float vHatG = vG / bc2;
                gammaE[p] = gammaE[p] - lr * mHatG / ((float)System.Math.Sqrt(vHatG) + adamEps);

                // beta update
                float gB = gradBetaE[p];
                float mB = oneMinusB1 * gB;
                float vB = oneMinusB2 * gB * gB;
                float mHatB = mB / bc1;
                float vHatB = vB / bc2;
                betaE[p] = betaE[p] - lr * mHatB / ((float)System.Math.Sqrt(vHatB) + adamEps);
            }
        }

        // ================ FUSED / compiled path ================
        var inputF = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var targetF = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < targetF.Length; i++) targetF[i] = targetData[i];
        var gammaF = new Tensor<float>(new[] { features });
        var betaF = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1.0f; betaF[i] = 0.0f; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            var diffF = engine.TensorSubtract(bnF, targetF);
            var sqF = engine.TensorMultiply(diffF, diffF);
            engine.ReduceSum(sqF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: lr);
            plan.Step();
        }

        // ================ Per-element diff ================
        for (int p = 0; p < features; p++)
        {
            float dGamma = System.Math.Abs(gammaE[p] - gammaF[p]);
            float dBeta = System.Math.Abs(betaE[p] - betaF[p]);
            // Tolerance kept tight (1e-5) so a sign-flipped element shows up
            // immediately as |Δ|≈0.02, while genuine float-precision noise
            // (≤1e-6 typically) stays under.
            Assert.True(dGamma < 1e-5,
                $"gamma[{p}] eager={gammaE[p]:F8} fused={gammaF[p]:F8} |Δ|={dGamma:E6} — sign-flip indicates compiled BN gradient slotting bug");
            Assert.True(dBeta < 1e-5,
                $"beta[{p}] eager={betaE[p]:F8} fused={betaF[p]:F8} |Δ|={dBeta:E6}");
        }
    }

    /// <summary>
    /// Stronger version: inspect the gradient values BEFORE the optimizer
    /// runs. Eager grad collected from the tape directly; compile-mode grad
    /// collected via the introspection hook the AccumulateGrad-into-dict
    /// path leaves on each parameter (`tensor.Grad`). If any per-channel
    /// gradient component differs across modes for a deterministic input,
    /// the compile-mode backward is mis-routing.
    /// </summary>
    [Fact]
    public void BatchNorm_2D_GradGamma_BeforeOptimizer_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;

        var rng = new System.Random(42);
        var inputData = new float[batch * features];
        var targetData = new float[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < targetData.Length; i++) targetData[i] = (float)(rng.NextDouble() - 0.5);

        // EAGER grads via tape
        var inputE = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var targetE = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < targetE.Length; i++) targetE[i] = targetData[i];
        var gammaE = new Tensor<float>(new[] { features });
        var betaE = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0f; betaE[i] = 0.0f; }

        Tensor<float> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<float>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var diffE = engine.TensorSubtract(bnE, targetE);
            var sqE = engine.TensorMultiply(diffE, diffE);
            var lossE = engine.ReduceSum(sqE, null);
            var grads = tape.ComputeGradients(lossE, sources: new[] { gammaE, betaE });
            eagerGradGamma = grads[gammaE];
            eagerGradBeta = grads[betaE];
        }

        // FUSED grads via CompiledTrainingPlan — capture from gammaF.Grad
        // after Step() runs the backward but before the optimizer mutates
        // the param. Use a no-op SGD with lr=0 so the params don't move,
        // then read .Grad on the parameter tensor.
        var inputF = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var targetF = new Tensor<float>(new[] { batch, features });
        for (int i = 0; i < targetF.Length; i++) targetF[i] = targetData[i];
        var gammaF = new Tensor<float>(new[] { features });
        var betaF = new Tensor<float>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1.0f; betaF[i] = 0.0f; }

        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            var diffF = engine.TensorSubtract(bnF, targetF);
            var sqF = engine.TensorMultiply(diffF, diffF);
            engine.ReduceSum(sqF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        using (plan)
        {
            // lr=0 so params don't move — we only want the grad capture.
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }

        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad not set after Step()");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad not set after Step()");

        // Collect all per-channel diffs FIRST so the failure message dumps
        // every channel — single-element bugs are much easier to bisect when
        // you can see the whole gamma/beta vector, not just the first
        // mismatch the assertion happens to catch.
        var lines = new System.Collections.Generic.List<string>();
        bool anyMismatch = false;
        for (int p = 0; p < features; p++)
        {
            float dGamma = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            float dBeta = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            if (dGamma >= 1e-4f || dBeta >= 1e-4f) anyMismatch = true;
            lines.Add($"  ch[{p}]  gamma_grad eager={eagerGradGamma[p],12:F6} fused={fusedGradGamma[p],12:F6} |Δ|={dGamma,12:E6}    beta_grad eager={eagerGradBeta[p],12:F6} fused={fusedGradBeta[p],12:F6} |Δ|={dBeta,12:E6}");
        }
        Assert.False(anyMismatch, "compiled BN backward routes wrong per-channel gradients:\n" + string.Join("\n", lines));
    }
}
