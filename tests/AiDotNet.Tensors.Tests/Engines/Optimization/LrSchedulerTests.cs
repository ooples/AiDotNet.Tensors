using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.Engines.Optimization.Schedulers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// LR-scheduler step-by-step tests. Reference values were computed against
/// the published PyTorch torch.optim.lr_scheduler implementation (see comments
/// near each Assert).
/// </summary>
public class LrSchedulerTests
{
    private static IOptimizer MakeOpt(double lr)
    {
        var opt = new SgdOptimizer();
        var w = new float[1]; var g = new float[1];
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = lr }).AddParameter(w, g);
        return opt;
    }

    [Fact]
    public void StepLr_DropsByGamma_EveryStepSize()
    {
        var opt = MakeOpt(1.0);
        var sched = new StepLr(opt, stepSize: 3, gamma: 0.1);
        // PyTorch reference (lr=1.0, step=3, γ=0.1): 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01
        double[] expected = { 1.0, 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01 };
        Assert.Equal(expected[0], sched.GetLastLr()[0], precision: 8);
        for (int i = 1; i < expected.Length; i++)
        {
            sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void MultiStepLr_DropsAtMilestones()
    {
        var opt = MakeOpt(1.0);
        var sched = new MultiStepLr(opt, milestones: new[] { 2, 5 }, gamma: 0.1);
        // PyTorch: 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01
        double[] expected = { 1.0, 1.0, 0.1, 0.1, 0.1, 0.01, 0.01 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void ExponentialLr_AppliesGammaPerEpoch()
    {
        var opt = MakeOpt(1.0);
        var sched = new ExponentialLr(opt, gamma: 0.5);
        // PyTorch: 1.0, 0.5, 0.25, 0.125
        double[] expected = { 1.0, 0.5, 0.25, 0.125 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void CosineAnnealingLr_GoesFrom_LrTo_EtaMin()
    {
        var opt = MakeOpt(1.0);
        var sched = new CosineAnnealingLr(opt, tMax: 10, etaMin: 0.0);
        // At epoch=0, lr = 1.0 (cos(0) = 1)
        Assert.Equal(1.0, sched.GetLastLr()[0], precision: 8);
        // At epoch=tMax (10), cos(π) = −1 → lr = ηmin
        for (int i = 0; i < 10; i++) sched.Step();
        Assert.Equal(0.0, sched.GetLastLr()[0], precision: 6);
    }

    [Fact]
    public void LinearLr_LinearWarmup()
    {
        var opt = MakeOpt(1.0);
        var sched = new LinearLr(opt, startFactor: 0.5, endFactor: 1.0, totalIters: 4);
        // PyTorch:
        //   epoch 0: 0.5
        //   epoch 1: 0.625
        //   epoch 2: 0.75
        //   epoch 3: 0.875
        //   epoch 4: 1.0 (end)
        double[] expected = { 0.5, 0.625, 0.75, 0.875, 1.0 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void ConstantLr_HoldsFactor_ThenSnapsBack()
    {
        var opt = MakeOpt(1.0);
        var sched = new ConstantLr(opt, factor: 0.25, totalIters: 3);
        // PyTorch:
        //   epoch 0..2: 0.25, 0.25, 0.25
        //   epoch 3+:   1.0
        double[] expected = { 0.25, 0.25, 0.25, 1.0, 1.0 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void PolynomialLr_GoesToZero_AfterTotalIters()
    {
        var opt = MakeOpt(1.0);
        var sched = new PolynomialLr(opt, totalIters: 4, power: 1.0);
        // Closed form: lr = (1 − t/4)
        double[] expected = { 1.0, 0.75, 0.5, 0.25, 0.0, 0.0 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void LambdaLr_AppliesUserFunction()
    {
        var opt = MakeOpt(1.0);
        var sched = new LambdaLr(opt, lrLambda: e => 1.0 / (1.0 + e));
        double[] expected = { 1.0 / 1, 1.0 / 2, 1.0 / 3, 1.0 / 4 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void MultiplicativeLr_Composes()
    {
        var opt = MakeOpt(1.0);
        var sched = new MultiplicativeLr(opt, lrLambda: e => 0.9);
        // PyTorch:  e=0 → 1.0
        //           e=1 → 1.0 · 0.9 = 0.9
        //           e=2 → 0.9 · 0.9 = 0.81
        //           e=3 → 0.729
        double[] expected = { 1.0, 0.9, 0.81, 0.729 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void CyclicLr_TriangularWave()
    {
        var opt = MakeOpt(0.0);
        var sched = new CyclicLr(opt, baseLr: 0.0, maxLr: 1.0,
                                 stepSizeUp: 2, stepSizeDown: 2,
                                 mode: CyclicMode.Triangular);
        //   epoch 0: 0.0
        //   epoch 1: 0.5
        //   epoch 2: 1.0
        //   epoch 3: 0.5
        //   epoch 4: 0.0
        double[] expected = { 0.0, 0.5, 1.0, 0.5, 0.0, 0.5 };
        for (int i = 0; i < expected.Length; i++)
        {
            if (i > 0) sched.Step();
            Assert.Equal(expected[i], sched.GetLastLr()[0], precision: 8);
        }
    }

    [Fact]
    public void OneCycleLr_RisesThenFalls()
    {
        var opt = MakeOpt(0.0); // OneCycle overwrites lr to maxLr/divFactor
        var sched = new OneCycleLr(opt, maxLr: 1.0, totalSteps: 10, pctStart: 0.3, cosineAnneal: false,
                                   divFactor: 25.0, finalDivFactor: 10000.0);
        double initialLr = sched.GetLastLr()[0];
        Assert.Equal(1.0 / 25.0, initialLr, precision: 6);

        double maxObserved = initialLr;
        for (int t = 0; t < 9; t++) { sched.Step(); maxObserved = Math.Max(maxObserved, sched.GetLastLr()[0]); }
        Assert.True(maxObserved > 0.9, $"max LR observed = {maxObserved}, should approach 1.0");
        // Final LR is annealed near zero
        Assert.True(sched.GetLastLr()[0] < initialLr / 10);
    }

    [Fact]
    public void ReduceLrOnPlateau_HalvesAfterPatience()
    {
        var opt = MakeOpt(1.0);
        var sched = new ReduceLrOnPlateau(opt, mode: "min", factor: 0.5, patience: 2, threshold: 0.0);
        // Worsening losses 1,2,3,4 — after patience=2 bad epochs (so 3rd bad), LR halves.
        sched.Step(1.0); // best = 1.0
        Assert.Equal(1.0, opt.ParamGroups[0].LearningRate, precision: 8);
        sched.Step(1.0);  // bad #1 (no strict improvement)
        sched.Step(1.0);  // bad #2
        sched.Step(1.0);  // bad #3 → reduces
        Assert.Equal(0.5, opt.ParamGroups[0].LearningRate, precision: 8);
    }

    [Fact]
    public void SequentialLr_DispatchesByMilestones()
    {
        var opt = MakeOpt(1.0);
        var warmup = new LinearLr(opt, startFactor: 0.1, endFactor: 1.0, totalIters: 3);
        var decay  = new ExponentialLr(opt, gamma: 0.5);
        var seq    = new SequentialLr(opt, new LrScheduler[] { warmup, decay }, new[] { 3 });

        // We expect the warmup to run for 3 epochs, then exponential decay takes over.
        Assert.Equal(0.1, warmup.GetLastLr()[0], precision: 8);
        seq.Step(); // epoch 0 → warmup epoch 1
        seq.Step(); // epoch 1 → warmup epoch 2
        seq.Step(); // epoch 2 → warmup epoch 3 (= 1.0)
        Assert.True(opt.ParamGroups[0].LearningRate > 0.5);
        seq.Step(); // epoch 3 → decay scheduler kicks in
        Assert.True(opt.ParamGroups[0].LearningRate <= 1.0);
    }

    [Fact]
    public void ChainedScheduler_CombinesEffects()
    {
        var opt = MakeOpt(1.0);
        var s1 = new ConstantLr(opt, factor: 0.5, totalIters: 2);
        var s2 = new ExponentialLr(opt, gamma: 0.9);
        var chain = new ChainedScheduler(opt, new LrScheduler[] { s1, s2 });
        chain.Step(); // both step once
        Assert.True(opt.ParamGroups[0].LearningRate > 0);
    }

    [Fact]
    public void ScheduleBuilder_BuildsSequentialPipeline()
    {
        var opt = MakeOpt(1.0);
        var built = ScheduleBuilder.For(opt)
            .Warmup(steps: 5, startFactor: 0.0)
            .Cosine(steps: 20)
            .LinearDecay(steps: 5, finalFactor: 0.0)
            .Build();
        Assert.IsType<SequentialLr>(built);
        var seq = (SequentialLr)built;
        // Stepping through warmup raises LR from 0 toward 1.
        for (int i = 0; i < 5; i++) seq.Step();
        Assert.True(opt.ParamGroups[0].LearningRate > 0.5);
    }

    [Fact]
    public void CosineAnnealingWarmRestarts_RestartsAtT0()
    {
        var opt = MakeOpt(1.0);
        var sched = new CosineAnnealingWarmRestarts(opt, t0: 4, tMult: 1, etaMin: 0.0);
        Assert.Equal(1.0, sched.GetLastLr()[0], precision: 6);
        for (int i = 0; i < 4; i++) sched.Step();
        // After 4 steps, t_cur was reset → lr is back near 1.0
        Assert.Equal(1.0, sched.GetLastLr()[0], precision: 6);
    }
}
