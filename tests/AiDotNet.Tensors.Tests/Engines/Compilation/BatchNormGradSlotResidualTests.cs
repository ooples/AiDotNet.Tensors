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
[Collection("CompiledTrainingPlanProbeSerial")]
public class BatchNormGradSlotResidualTests
{
    /// <summary>
    /// Isolation: ONLY BN forward then ReduceSum (uniform gradOut = 1.0
    /// across BN's output). No subtract / multiply / target. If THIS
    /// diverges between eager and compiled per-element on T=double,
    /// the BN backward itself produces different gradients across the
    /// two paths despite both calling engine.BatchNormBackward. If it
    /// passes, the divergence in BatchNorm_2D_DOUBLE_GradGamma_* must
    /// come from upstream MultiplyBackward / SubtractBackward differences,
    /// NOT BN backward.
    /// </summary>
    [Fact]
    public void Isolation_BatchNorm_2D_DOUBLE_Then_ReduceSum_GradGamma_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;

        var rng = new System.Random(42);
        var inputData = new double[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() - 0.5;

        var inputE = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var gammaE = new Tensor<double>(new[] { features });
        var betaE = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }

        Tensor<double> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<double>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var sumE = engine.ReduceSum(bnE, null);
            var grads = tape.ComputeGradients(sumE, sources: new[] { gammaE, betaE });
            eagerGradGamma = grads[gammaE];
            eagerGradBeta = grads[betaE];
        }

        var inputF = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var gammaF = new Tensor<double>(new[] { features });
        var betaF = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1.0; betaF[i] = 0.0; }

        ICompiledTrainingPlan<double> plan;
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
        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad");

        var lines = new System.Collections.Generic.List<string>();
        bool any = false;
        const double tol = 1e-12;
        for (int p = 0; p < features; p++)
        {
            double dG = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            double dB = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            if (dG >= tol || dB >= tol) any = true;
            lines.Add($"  ch[{p}]  gammaG eager={eagerGradGamma[p]:F12} fused={fusedGradGamma[p]:F12} |Δ|={dG:E3}    betaG eager={eagerGradBeta[p]:F12} fused={fusedGradBeta[p]:F12} |Δ|={dB:E3}");
        }
        Assert.False(any, "Isolated BN→ReduceSum on double diverges:\n" + string.Join("\n", lines));
    }

    /// <summary>
    /// Bisect the divergence chain. Eager has BN→Sub→Mul→Sum (forward) +
    /// reverse backward. Compile has same. The eager-tape gradGamma is
    /// 8.76678034 (per the failing test). The compile gradGamma is
    /// 8.76678046. Δ ≈ 1.19e-7. Both should give the SAME value if every
    /// op produces bit-identical results. This test isolates the FORWARD
    /// chain — runs both eager and compile-mode forward for BN→Sub→Mul→Sum
    /// and asserts the diff and sq tensors are bit-identical.
    /// </summary>
    [Fact]
    public void Pinpoint_DOUBLE_BNSubMul_Forward_EagerVsCompile_BitIdentical()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;
        var rng = new System.Random(42);
        var inputData = new double[batch * features];
        var targetData = new double[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < targetData.Length; i++) targetData[i] = rng.NextDouble() - 0.5;

        // EAGER
        var inputE = new Tensor<double>(new[] { batch, features });
        var targetE = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) { inputE[i] = inputData[i]; targetE[i] = targetData[i]; }
        var gammaE = new Tensor<double>(new[] { features });
        var betaE = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }
        var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
        var diffE = engine.TensorSubtract(bnE, targetE);
        var sqE = engine.TensorMultiply(diffE, diffE);

        // COMPILE
        var inputF = new Tensor<double>(new[] { batch, features });
        var targetF = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) { inputF[i] = inputData[i]; targetF[i] = targetData[i]; }
        var gammaF = new Tensor<double>(new[] { features });
        var betaF = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1.0; betaF[i] = 0.0; }
        Tensor<double> bnF, diffF, sqF;
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            diffF = engine.TensorSubtract(bnF, targetF);
            sqF = engine.TensorMultiply(diffF, diffF);
            engine.ReduceSum(sqF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }
        // Wire StepProbe to capture diffF[0] at every phase boundary
        var ctpType3 = typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<double>).Assembly
            .GetType("AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan`1")!
            .MakeGenericType(typeof(double));
        var probeProp3 = ctpType3.GetProperty("StepProbe", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var phaseLog = new System.Collections.Generic.List<string>();
        System.Action<string> probe3 = phase =>
        {
            phaseLog.Add($"  {phase,-40} bnF[0]={bnF[0]:F18} diffF[0]={diffF[0]:F18} sqF[0]={sqF[0]:F18}");
        };
        probeProp3.SetValue(null, probe3);
        try
        {
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                plan.Step();
            }
        }
        finally
        {
            probeProp3.SetValue(null, null);
        }
        phaseLog.Add($"  POST-DISPOSE                             bnF[0]={bnF[0]:F18} diffF[0]={diffF[0]:F18} sqF[0]={sqF[0]:F18}");

        // EXPERIMENT: snapshot diffF four ways to find what fixes the Heisenbug.
        var diffF_directIndex = new double[batch * features];
        for (int i = 0; i < diffF.Length; i++) diffF_directIndex[i] = diffF[i];

        // (a) Force GC.Collect to test if it's a GC-related stale read
        var diffF_afterGC = new double[batch * features];
        System.GC.Collect();
        System.GC.WaitForPendingFinalizers();
        for (int i = 0; i < diffF.Length; i++) diffF_afterGC[i] = diffF[i];

        // (b) Memory barrier
        var diffF_afterBarrier = new double[batch * features];
        System.Threading.Thread.MemoryBarrier();
        for (int i = 0; i < diffF.Length; i++) diffF_afterBarrier[i] = diffF[i];

        // (c) Read via AsSpan instead of indexer
        var diffF_viaSpan = new double[batch * features];
        var diffSpan = diffF.AsSpan();
        for (int i = 0; i < diffF.Length; i++) diffF_viaSpan[i] = diffSpan[i];

        // (d) Read via GetDataArray (raw backing)
        var diffF_viaArray = new double[batch * features];
        var diffArr = (double[])(object)diffF.GetDataArray();
        for (int i = 0; i < diffF.Length; i++) diffF_viaArray[i] = diffArr[i];

        var lines = new System.Collections.Generic.List<string>();
        bool any = false;
        for (int i = 0; i < bnE.Length; i++)
        {
            double dBn = System.Math.Abs(bnE[i] - bnF[i]);
            double dDiff = System.Math.Abs(diffE[i] - diffF_directIndex[i]);
            double dDiffGC = System.Math.Abs(diffE[i] - diffF_afterGC[i]);
            double dDiffBar = System.Math.Abs(diffE[i] - diffF_afterBarrier[i]);
            double dDiffSpan = System.Math.Abs(diffE[i] - diffF_viaSpan[i]);
            double dDiffArr = System.Math.Abs(diffE[i] - diffF_viaArray[i]);
            double dSq = System.Math.Abs(sqE[i] - sqF[i]);
            if (dBn > 0 || dDiff > 0 || dDiffGC > 0 || dDiffBar > 0 || dDiffSpan > 0 || dDiffArr > 0 || dSq > 0) any = true;
            lines.Add($"  [{i}]  bn={dBn:E3} diff(idx)={dDiff:E3} diff(gc)={dDiffGC:E3} diff(bar)={dDiffBar:E3} diff(span)={dDiffSpan:E3} diff(arr)={dDiffArr:E3} sq={dSq:E3}");
        }
        // Read SubFwdDiag via reflection
        var ctpType2 = typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<double>).Assembly
            .GetType("AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan`1")!
            .MakeGenericType(typeof(double));
        var subDiagField = ctpType2.GetField("SubFwdDiag", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static);
        var subDiag = subDiagField?.GetValue(null) as string ?? "(unset)";

        Assert.False(any, $"Forward chain BN→Sub→Mul diverges (probe-mode bisect):\n  SUB-FWD diag: {subDiag}\n  eager bnE[0]={bnE[0]:F18} bnF[0]={bnF[0]:F18} targetE[0]={targetE[0]:F18} targetF[0]={targetF[0]:F18}\n  eager diffE[0]={diffE[0]:F18}  (manual: bnE[0]-targetE[0] = {bnE[0] - targetE[0]:F18})\n  diffF_directIndex[0]={diffF_directIndex[0]:F18}\n  diffF[0] (live indexer) = {diffF[0]:F18}\n  PHASE LOG:\n"
            + string.Join("\n", phaseLog) + "\n  DIFFS:\n"
            + string.Join("\n", lines));
    }

    /// <summary>
    /// Pinpoint #2: BatchNorm backward in EAGER tape mode vs DIRECT engine
    /// call. If the tape's backward differs from the direct engine call,
    /// the tape itself is the precision-losing path (engine routing, not
    /// kernel arithmetic).
    /// </summary>
    [Fact]
    public void Pinpoint_DOUBLE_BNBackward_TapeVsDirect_BitIdentical()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;
        var rng = new System.Random(42);
        var inputData = new double[batch * features];
        var targetData = new double[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < targetData.Length; i++) targetData[i] = rng.NextDouble() - 0.5;

        // Build identical inputs for both paths.
        var inputE = new Tensor<double>(new[] { batch, features });
        var targetE = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) { inputE[i] = inputData[i]; targetE[i] = targetData[i]; }
        var gammaE = new Tensor<double>(new[] { features });
        var betaE = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }

        // Path 1: EAGER TAPE (existing failing path)
        Tensor<double> tapeGradGamma;
        using (var tape = new GradientTape<double>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var diffE = engine.TensorSubtract(bnE, targetE);
            var sqE = engine.TensorMultiply(diffE, diffE);
            var lossE = engine.ReduceSum(sqE, null);
            tapeGradGamma = tape.ComputeGradients(lossE, sources: new[] { gammaE })[gammaE];
        }

        // Path 2: DIRECT — manually compute gradGamma without tape.
        // Forward + manual chain rule using engine ops.
        var bn2 = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out var meanD, out var varD);
        var diff2 = engine.TensorSubtract(bn2, targetE);
        var sq2 = engine.TensorMultiply(diff2, diff2);
        var loss2 = engine.ReduceSum(sq2, null);
        // dloss/dsq = 1 (broadcast)
        var gradSq2 = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < gradSq2.Length; i++) gradSq2[i] = 1.0;
        // dsq/ddiff = 2*diff via TensorMultiply
        var twoTimesDiff = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < twoTimesDiff.Length; i++) twoTimesDiff[i] = 2.0 * diff2[i];
        var gradDiff2 = engine.TensorMultiply(gradSq2, twoTimesDiff);
        // ddiff/dbn = 1 → gradBn = gradDiff
        var gradBn2 = gradDiff2;
        // dbn/dgamma via engine.BatchNormBackward
        engine.BatchNormBackward(gradBn2, inputE, gammaE, meanD, varD, epsilon, out var directGradGamma, out _);

        var lines = new System.Collections.Generic.List<string>();
        bool any = false;
        for (int p = 0; p < features; p++)
        {
            double d = System.Math.Abs(tapeGradGamma[p] - directGradGamma[p]);
            if (d >= 1e-12) any = true;
            lines.Add($"  ch[{p}]  tape={tapeGradGamma[p]:F18} direct={directGradGamma[p]:F18} |Δ|={d:E3}");
        }
        Assert.False(any, "BN gradGamma diverges between tape backward and direct engine call:\n"
            + string.Join("\n", lines));
    }

    /// <summary>
    /// Diagnostic: verify that calling engine.BatchNorm on a CpuEngine
    /// instance under an active GradientTape correctly binds the tape's
    /// _engine to that CpuEngine (not the global ambient
    /// DirectGpuTensorEngine).
    /// </summary>
    [Fact]
    public void Diagnostic_TapeBoundEngineIsCpuEngine_AfterBatchNormCall()
    {
        var engine = new CpuEngine();
        const int C = 4;
        var inputE = new Tensor<double>(new[] { 4, C });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = i * 0.1;
        var gammaE = new Tensor<double>(new[] { C });
        var betaE = new Tensor<double>(new[] { C });
        for (int i = 0; i < C; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }

        using var tape = new GradientTape<double>();
        var initialEngineType = tape.Engine.GetType().Name;
        var bnE = engine.BatchNorm(inputE, gammaE, betaE, 1e-5, out _, out _);
        var afterEngineType = tape.Engine.GetType().Name;

        Assert.Equal("CpuEngine", afterEngineType);
        // Initial may be DirectGpuTensorEngine on auto-detect-GPU systems;
        // we just confirm the bind happened.
    }

    /// <summary>
    /// Pinpoint: forward-only TensorSubtract bit-identity check. Eager
    /// engine.TensorSubtract vs compile-mode lazy execute (which calls
    /// engine.TensorSubtractInto under the hood). If THIS fails, the
    /// SubtractInto SIMD kernel diverges from the allocating Subtract
    /// SIMD kernel — that's where the MseChain residual 1e-7 comes from.
    /// </summary>
    [Fact]
    public void Pinpoint_DOUBLE_TensorSubtract_Forward_EagerVsCompile_BitIdentical()
    {
        var engine = new CpuEngine();
        const int n = 16;
        var rng = new System.Random(42);
        var xData = new double[n]; var tData = new double[n];
        for (int i = 0; i < n; i++) xData[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < n; i++) tData[i] = rng.NextDouble() - 0.5;

        // Eager
        var xE = new Tensor<double>(new[] { n });
        var tE = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) { xE[i] = xData[i]; tE[i] = tData[i]; }
        var diffE = engine.TensorSubtract(xE, tE);

        // Compile (forward only — read post-Step)
        var xF = new Tensor<double>(new[] { n });
        var tF = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) { xF[i] = xData[i]; tF[i] = tData[i]; }
        Tensor<double> diffF;
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            typeof(AiDotNet.Tensors.Engines.Compilation.LazyTensorScope)
                .GetMethod("BindEngineIfUnset", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!
                .Invoke(scope, new object[] { engine });
            diffF = engine.TensorSubtract(xF, tF);
            engine.ReduceSum(diffF, null);
            plan = scope.CompileTraining(new[] { xF });
        }

        // Wire the per-phase probe to capture diffF[12] AFTER each step.
        // Diff in compile-mode test runs has shown diffF[12] starts as the
        // correct double value (written by SUB-FWD) but ends up at
        // float-precision when the test reads it post-Step. This probe
        // pinpoints WHICH step writes the wrong value.
        var ctpType = typeof(AiDotNet.Tensors.LinearAlgebra.Tensor<double>).Assembly
            .GetType("AiDotNet.Tensors.Engines.Compilation.CompiledTrainingPlan`1")!
            .MakeGenericType(typeof(double));
        var probeProp = ctpType.GetProperty("StepProbe", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)!;
        var probeLog = new System.Collections.Generic.List<string>();
        System.Action<string> probe = phase =>
        {
            probeLog.Add($"  {phase,-40} diffF[12]={diffF[12]:F18}");
        };
        probeProp.SetValue(null, probe);
        try
        {
            using (plan)
            {
                plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
                plan.Step();
                probeLog.Add($"  AFTER-STEP-BEFORE-DISPOSE              diffF[12]={diffF[12]:F18}");
            }
            probeLog.Add($"  AFTER-DISPOSE                          diffF[12]={diffF[12]:F18}");
        }
        finally
        {
            probeProp.SetValue(null, null);
        }
        System.Console.WriteLine("=== STEP PROBE LOG ===");
        foreach (var l in probeLog) System.Console.WriteLine(l);

        // CRITICAL ORDER: snapshot diffF FIRST, before any other allocation
        // could pollute the underlying pool-backed buffer.
        var diffFSnap = new double[n];
        for (int i = 0; i < n; i++) diffFSnap[i] = diffF[i];

        // Direct TensorSubtractInto call — bypass the plan, mimic exactly
        // what the lazy execute does.
        var directOut = new Tensor<double>(new[] { n });
        engine.TensorSubtractInto(directOut, xF, tF);

        var lines = new System.Collections.Generic.List<string>();
        bool any = false;
        for (int i = 0; i < n; i++)
        {
            double dEvSnap = System.Math.Abs(diffE[i] - diffFSnap[i]);
            double dSnapNow = System.Math.Abs(diffFSnap[i] - diffF[i]);
            double dEvD = System.Math.Abs(diffE[i] - directOut[i]);
            if (dEvSnap > 0 || dSnapNow > 0 || dEvD > 0) any = true;
            lines.Add($"  [{i}]  eager={diffE[i]:F18} fusedSnap={diffFSnap[i]:F18} fusedNow={diffF[i]:F18} direct={directOut[i]:F18} |E-Snap|={dEvSnap:E3} |Snap-Now|={dSnapNow:E3} |E-D|={dEvD:E3}");
        }
        Assert.False(any, "TensorSubtract forward diverges between eager and compile-mode replay:\n"
            + string.Join("\n", lines));
    }

    /// <summary>
    /// Even-finer bisect: ONLY mul(x, x).ReduceSum() - no subtract, no
    /// target. Tests just the Multiply-with-self backward path on T=double.
    /// If this PASSES at 1e-12, the divergence in
    /// Isolation_DOUBLE_MseChain_GradX is in TensorSubtract; if it FAILS,
    /// the divergence is in TensorMultiply backward when both inputs are
    /// the same tensor reference (a documented autodiff trap — see the
    /// existing float-tolerance SquareSum_Backward test at line 545).
    /// </summary>
    [Fact]
    public void Isolation_DOUBLE_SquareSum_GradX_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int n = 16;
        var rng = new System.Random(42);
        var xData = new double[n];
        for (int i = 0; i < n; i++) xData[i] = rng.NextDouble() - 0.5;

        var xE = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) xE[i] = xData[i];

        Tensor<double> eagerGrad;
        using (var tape = new GradientTape<double>())
        {
            var sq = engine.TensorMultiply(xE, xE);
            var loss = engine.ReduceSum(sq, null);
            eagerGrad = tape.ComputeGradients(loss, sources: new[] { xE })[xE];
        }

        var xF = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) xF[i] = xData[i];

        ICompiledTrainingPlan<double> plan;
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
        var fusedGrad = xF.Grad ?? throw new System.InvalidOperationException("xF.Grad");

        var lines = new System.Collections.Generic.List<string>();
        bool any = false;
        const double tol = 1e-12;
        for (int i = 0; i < n; i++)
        {
            double d = System.Math.Abs(eagerGrad[i] - fusedGrad[i]);
            if (d >= tol) any = true;
            lines.Add($"  x[{i}]={xE[i]:F12} eager={eagerGrad[i]:F12} fused={fusedGrad[i]:F12} |Δ|={d:E3}");
        }
        Assert.False(any, "x*x→sum backward diverges:\n" + string.Join("\n", lines));
    }

    /// <summary>
    /// Bisect the upstream Sub→Multiply chain. BN forward is REPLACED with
    /// a leaf parameter `xF` — eliminates BN backward as a variable.
    /// MSE backward path: ReduceSum → Multiply(square) → Subtract(target).
    /// If THIS diverges per-element on T=double, the divergence is purely
    /// in the Subtract+Multiply backward chain — the multiply-with-self
    /// gradient is being accumulated through different intermediate
    /// allocations between tape and compiled paths.
    /// </summary>
    [Fact]
    public void Isolation_DOUBLE_MseChain_GradX_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int n = 16;
        var rng = new System.Random(42);
        var xData = new double[n];
        var tData = new double[n];
        for (int i = 0; i < n; i++) xData[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < n; i++) tData[i] = rng.NextDouble() - 0.5;

        var xE = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) xE[i] = xData[i];
        var tE = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) tE[i] = tData[i];

        Tensor<double> eagerGrad;
        using (var tape = new GradientTape<double>())
        {
            var diff = engine.TensorSubtract(xE, tE);
            var sq = engine.TensorMultiply(diff, diff);
            var loss = engine.ReduceSum(sq, null);
            eagerGrad = tape.ComputeGradients(loss, sources: new[] { xE })[xE];
        }

        var xF = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) xF[i] = xData[i];
        var tF = new Tensor<double>(new[] { n });
        for (int i = 0; i < n; i++) tF[i] = tData[i];

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var diff = engine.TensorSubtract(xF, tF);
            var sq = engine.TensorMultiply(diff, diff);
            engine.ReduceSum(sq, null);
            plan = scope.CompileTraining(new[] { xF });
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }
        var fusedGrad = xF.Grad ?? throw new System.InvalidOperationException("xF.Grad");

        var lines = new System.Collections.Generic.List<string>();
        bool any = false;
        const double tol = 1e-12;
        for (int i = 0; i < n; i++)
        {
            double d = System.Math.Abs(eagerGrad[i] - fusedGrad[i]);
            if (d >= tol) any = true;
            lines.Add($"  x[{i}]={xE[i]:F12} t[{i}]={tE[i]:F12} eager={eagerGrad[i]:F12} fused={fusedGrad[i]:F12} |Δ|={d:E3}");
        }
        Assert.False(any, "MSE chain backward diverges:\n" + string.Join("\n", lines));
    }

    /// <summary>
    /// Diagnostic: take the SAME inputF/gammaF/betaF that compile-mode
    /// captured in the lazy delegate, and run engine.BatchNorm on them
    /// directly RIGHT BEFORE plan.Step(). If THIS diverges from eager,
    /// the input data was mutated post-compile (root cause is buffer
    /// aliasing or a stale sticky GraphMode state). If it matches, the
    /// bug is in plan.Step's invocation context (re-entrant GraphMode,
    /// lazy result aliasing the input buffer, etc).
    /// </summary>
    [Fact]
    public void Diagnostic_BatchNorm_3D_DOUBLE_DirectCallAfterCompile_MatchesEager()
    {
        var engine = new CpuEngine();
        const int C = 8, H = 4, W = 4;
        const double epsilon = 1e-5;
        var rng = new System.Random(42);
        var inputData = new double[C * H * W];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() - 0.5;

        var inputE = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var gammaE = new Tensor<double>(new[] { C });
        var betaE = new Tensor<double>(new[] { C });
        for (int i = 0; i < C; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }
        var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
        var snapE = new double[bnE.Length];
        for (int i = 0; i < bnE.Length; i++) snapE[i] = bnE[i];

        var inputF = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var gammaF = new Tensor<double>(new[] { C });
        var betaF = new Tensor<double>(new[] { C });
        for (int i = 0; i < C; i++) { gammaF[i] = 1.0; betaF[i] = 0.0; }

        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            var bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            engine.ReduceSum(bnF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        // Snapshot inputF/gammaF/betaF data RIGHT NOW (post-Compile, pre-Step)
        var inSnap = new double[inputF.Length];
        for (int i = 0; i < inputF.Length; i++) inSnap[i] = inputF[i];
        var gSnap = new double[gammaF.Length];
        for (int i = 0; i < gammaF.Length; i++) gSnap[i] = gammaF[i];
        var bSnap = new double[betaF.Length];
        for (int i = 0; i < betaF.Length; i++) bSnap[i] = betaF[i];

        // Direct call — this is what plan.Step's lazy execute will do internally.
        // GraphMode is now disabled. If this diverges from snapE, the data was
        // already wrong by the time we got here.
        var bnDirect = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);

        var lines = new System.Collections.Generic.List<string>();
        bool dataDrift = false, outputDrift = false;
        const double tol = 1e-12;
        for (int i = 0; i < inputData.Length; i++)
            if (System.Math.Abs(inputData[i] - inSnap[i]) > tol) { dataDrift = true; lines.Add($"  inputF[{i}] orig={inputData[i]:F8} now={inSnap[i]:F8}"); }
        for (int i = 0; i < C; i++)
        {
            if (System.Math.Abs(1.0 - gSnap[i]) > tol) { dataDrift = true; lines.Add($"  gammaF[{i}] orig=1.0 now={gSnap[i]:F8}"); }
            if (System.Math.Abs(0.0 - bSnap[i]) > tol) { dataDrift = true; lines.Add($"  betaF[{i}] orig=0.0 now={bSnap[i]:F8}"); }
        }
        for (int i = 0; i < bnE.Length; i++)
        {
            double d = System.Math.Abs(snapE[i] - bnDirect[i]);
            if (d > tol) { outputDrift = true; lines.Add($"  bn[{i}] eager={snapE[i]:F8} direct={bnDirect[i]:F8} (Δ={d:E2})"); }
        }
        Assert.False(dataDrift || outputDrift,
            $"dataDrift={dataDrift}, outputDrift={outputDrift}. First mismatches:\n" + string.Join("\n", lines.Take(20)));
    }

    /// <summary>
    /// T=double rank-3 [C,H,W] forward-only diff. Pre-Step (right after
    /// CompileTraining returns) bnF should hold the eager-time-copied
    /// values. Post-Step bnF should be the result of the lazy execute
    /// delegate. Both should equal the standalone eager call. If post-Step
    /// diverges, the lazy execute's eng.BatchNorm replay gives a different
    /// output than the eager call did — pointing the bug at the rank-3
    /// double BN forward path under Step().
    /// </summary>
    [Fact]
    public void BatchNorm_3D_DOUBLE_CHW_ForwardReplay_MatchesEager()
    {
        var engine = new CpuEngine();
        const int C = 8, H = 4, W = 4;
        const double epsilon = 1e-5;
        var rng = new System.Random(42);
        var inputData = new double[C * H * W];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() - 0.5;

        var inputE = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var gammaE = new Tensor<double>(new[] { C });
        var betaE = new Tensor<double>(new[] { C });
        for (int i = 0; i < C; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }
        var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
        var snapE = new double[bnE.Length];
        for (int i = 0; i < bnE.Length; i++) snapE[i] = bnE[i];

        var inputF = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var gammaF = new Tensor<double>(new[] { C });
        var betaF = new Tensor<double>(new[] { C });
        for (int i = 0; i < C; i++) { gammaF[i] = 1.0; betaF[i] = 0.0; }

        Tensor<double> bnF;
        ICompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.Enable())
        {
            bnF = engine.BatchNorm(inputF, gammaF, betaF, epsilon, out _, out _);
            engine.ReduceSum(bnF, null);
            plan = scope.CompileTraining(new[] { gammaF, betaF });
        }

        var snapPre = new double[bnF.Length];
        for (int i = 0; i < bnF.Length; i++) snapPre[i] = bnF[i];

        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }

        var snapPost = new double[bnF.Length];
        for (int i = 0; i < bnF.Length; i++) snapPost[i] = bnF[i];

        var lines = new System.Collections.Generic.List<string>();
        bool preDiff = false, postDiff = false;
        const double tol = 1e-12;
        for (int c = 0; c < C; c++)
            for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++)
                {
                    int i = c * H * W + h * W + w;
                    double dPre = System.Math.Abs(snapE[i] - snapPre[i]);
                    double dPost = System.Math.Abs(snapE[i] - snapPost[i]);
                    if (dPre > tol) preDiff = true;
                    if (dPost > tol) postDiff = true;
                    if (dPre > tol || dPost > tol)
                        lines.Add($"  [{c},{h},{w}]  eager={snapE[i],14:F8} preStep={snapPre[i],14:F8} (Δ={dPre,9:E2})  postStep={snapPost[i],14:F8} (Δ={dPost,9:E2})");
                }
        Assert.False(preDiff || postDiff,
            $"BN forward diverges (preDiff={preDiff}, postDiff={postDiff}). First {lines.Count} mismatches:\n" +
            string.Join("\n", lines.Take(10)));
    }

    /// <summary>
    /// T=double, rank-3 [C, H, W] — the actual layout GraFPrint's 18
    /// BatchNorm layers see (and the Conv1×1 / Conv3×3 image-stack pyramid
    /// in general). If the 2D test passes within tolerance but THIS fails,
    /// the bug is rank-3 specific in the double compile-mode path.
    /// </summary>
    [Fact]
    public void BatchNorm_3D_DOUBLE_CHW_GradGamma_BeforeOptimizer_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int C = 8, H = 4, W = 4;
        const double epsilon = 1e-5;

        var rng = new System.Random(42);
        int len = C * H * W;
        var inputData = new double[len];
        var targetData = new double[len];
        for (int i = 0; i < len; i++) inputData[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < len; i++) targetData[i] = rng.NextDouble() - 0.5;

        var inputE = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var targetE = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < targetE.Length; i++) targetE[i] = targetData[i];
        var gammaE = new Tensor<double>(new[] { C });
        var betaE = new Tensor<double>(new[] { C });
        for (int i = 0; i < C; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }

        Tensor<double> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<double>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var diffE = engine.TensorSubtract(bnE, targetE);
            var sqE = engine.TensorMultiply(diffE, diffE);
            var lossE = engine.ReduceSum(sqE, null);
            var grads = tape.ComputeGradients(lossE, sources: new[] { gammaE, betaE });
            eagerGradGamma = grads[gammaE];
            eagerGradBeta = grads[betaE];
        }

        var inputF = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var targetF = new Tensor<double>(new[] { C, H, W });
        for (int i = 0; i < targetF.Length; i++) targetF[i] = targetData[i];
        var gammaF = new Tensor<double>(new[] { C });
        var betaF = new Tensor<double>(new[] { C });
        for (int i = 0; i < C; i++) { gammaF[i] = 1.0; betaF[i] = 0.0; }

        ICompiledTrainingPlan<double> plan;
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
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }

        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad not set");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad not set");

        var lines = new System.Collections.Generic.List<string>();
        bool anyMismatch = false;
        // Tolerance: 1e-9 catches anything above double-precision noise
        // (~1e-15) without flagging legitimate FMA reordering effects
        // that can creep up to ~1e-12 in batched reductions.
        const double tol = 1e-9;
        for (int p = 0; p < C; p++)
        {
            double dG = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            double dB = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            if (dG >= tol || dB >= tol) anyMismatch = true;
            lines.Add($"  ch[{p}]  gamma_grad eager={eagerGradGamma[p],14:F8} fused={fusedGradGamma[p],14:F8} |Δ|={dG,14:E6}    beta_grad eager={eagerGradBeta[p],14:F8} fused={fusedGradBeta[p],14:F8} |Δ|={dB,14:E6}");
        }
        Assert.False(anyMismatch, "T=double rank-3 [C,H,W] compiled BN backward routes wrong per-channel gradients:\n" + string.Join("\n", lines));
    }

    /// <summary>
    /// T=double version of the per-element gradient diff. The float-only
    /// BatchNormInferenceUnsafe layout fix can't help here (the
    /// specialization is gated on typeof(T) == typeof(float)), so if this
    /// fails the divergence is in a separate code path that affects the
    /// double-precision compile-mode BN backward — exactly the symptom
    /// GraFPrint hits in its T=double model-family invariant tests.
    /// </summary>
    [Fact]
    public void BatchNorm_2D_DOUBLE_GradGamma_BeforeOptimizer_MatchesEager_PerElement()
    {
        var engine = new CpuEngine();
        const int batch = 4, features = 4;
        const double epsilon = 1e-5;

        var rng = new System.Random(42);
        var inputData = new double[batch * features];
        var targetData = new double[batch * features];
        for (int i = 0; i < inputData.Length; i++) inputData[i] = rng.NextDouble() - 0.5;
        for (int i = 0; i < targetData.Length; i++) targetData[i] = rng.NextDouble() - 0.5;

        var inputE = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < inputE.Length; i++) inputE[i] = inputData[i];
        var targetE = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < targetE.Length; i++) targetE[i] = targetData[i];
        var gammaE = new Tensor<double>(new[] { features });
        var betaE = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gammaE[i] = 1.0; betaE[i] = 0.0; }

        Tensor<double> eagerGradGamma, eagerGradBeta;
        using (var tape = new GradientTape<double>())
        {
            var bnE = engine.BatchNorm(inputE, gammaE, betaE, epsilon, out _, out _);
            var diffE = engine.TensorSubtract(bnE, targetE);
            var sqE = engine.TensorMultiply(diffE, diffE);
            var lossE = engine.ReduceSum(sqE, null);
            var grads = tape.ComputeGradients(lossE, sources: new[] { gammaE, betaE });
            eagerGradGamma = grads[gammaE];
            eagerGradBeta = grads[betaE];
        }

        var inputF = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < inputF.Length; i++) inputF[i] = inputData[i];
        var targetF = new Tensor<double>(new[] { batch, features });
        for (int i = 0; i < targetF.Length; i++) targetF[i] = targetData[i];
        var gammaF = new Tensor<double>(new[] { features });
        var betaF = new Tensor<double>(new[] { features });
        for (int i = 0; i < features; i++) { gammaF[i] = 1.0; betaF[i] = 0.0; }

        ICompiledTrainingPlan<double> plan;
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
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 0.0f);
            plan.Step();
        }

        var fusedGradGamma = gammaF.Grad ?? throw new System.InvalidOperationException("gammaF.Grad not set");
        var fusedGradBeta = betaF.Grad ?? throw new System.InvalidOperationException("betaF.Grad not set");

        var lines = new System.Collections.Generic.List<string>();
        bool anyMismatch = false;
        for (int p = 0; p < features; p++)
        {
            double dG = System.Math.Abs(eagerGradGamma[p] - fusedGradGamma[p]);
            double dB = System.Math.Abs(eagerGradBeta[p] - fusedGradBeta[p]);
            if (dG >= 1e-8 || dB >= 1e-8) anyMismatch = true;
            lines.Add($"  ch[{p}]  gamma_grad eager={eagerGradGamma[p],14:F8} fused={fusedGradGamma[p],14:F8} |Δ|={dG,14:E6}    beta_grad eager={eagerGradBeta[p],14:F8} fused={fusedGradBeta[p],14:F8} |Δ|={dB,14:E6}");
        }
        Assert.False(anyMismatch, "T=double compiled BN backward routes wrong per-channel gradients:\n" + string.Join("\n", lines));
    }

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
