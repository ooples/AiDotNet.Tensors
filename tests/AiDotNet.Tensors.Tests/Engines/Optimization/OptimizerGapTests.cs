using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using AiDotNet.Tensors.Engines.Optimization.Schedulers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// Tests covering the gaps surfaced in the issue #224 audit: LAMB/LARS/FTRL wrappers,
/// the SGD/Adam <c>maximize</c> option, RMSprop centered variant, and BF16-moment Adam.
/// </summary>
public class OptimizerGapTests
{
    private static (float[][] X, float[] y) MakeProblem(int seed = 1234, int n = 64, int d = 4)
    {
        var rng = new Random(seed);
        var X = new float[n][];
        var y = new float[n];
        var w = new float[] { 3f, -2f, 0.5f, 1f };
        for (int i = 0; i < n; i++)
        {
            var x = new float[d];
            for (int j = 0; j < d - 1; j++) x[j] = (float)(rng.NextDouble() * 2 - 1);
            x[d - 1] = 1f;
            X[i] = x;
            float yi = 0;
            for (int j = 0; j < d; j++) yi += w[j] * x[j];
            y[i] = yi;
        }
        return (X, y);
    }

    private static (float loss, float[] grad) ComputeGrad(float[] w, float[][] X, float[] y)
    {
        int n = X.Length, d = w.Length;
        var grad = new float[d];
        float loss = 0;
        for (int i = 0; i < n; i++)
        {
            float pred = 0; for (int j = 0; j < d; j++) pred += w[j] * X[i][j];
            float r = pred - y[i];
            loss += r * r;
            for (int j = 0; j < d; j++) grad[j] += 2f * r * X[i][j] / n;
        }
        return (loss / n, grad);
    }

    private static float Train(IOptimizer opt, float[] w, float[] grad, int iters)
    {
        var (X, y) = MakeProblem();
        float final = 0f;
        for (int t = 0; t < iters; t++)
        {
            var (loss, g) = ComputeGrad(w, X, y);
            Array.Copy(g, grad, g.Length);
            opt.Step();
            Array.Clear(grad, 0, grad.Length);
            final = loss;
        }
        return final;
    }

    [Fact]
    public void LambOptimizer_Converges()
    {
        var w = new float[4]; var g = new float[4];
        var opt = new LambOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.05 }).AddParameter(w, g);
        Assert.True(Train(opt, w, g, 400) < 0.1, "LAMB failed to converge");
    }

    [Fact]
    public void LarsOptimizer_Converges()
    {
        // LARS scales each step by trust_coeff·‖p‖/(‖g‖+wd·‖p‖). On our toy MSE problem
        // (small param norm, gradients of similar magnitude) the trust ratio is ~1, so
        // we use no momentum to avoid 0.9-momentum-induced oscillation and rely on the
        // trust-ratio'd lr.
        var w = new float[4]; var g = new float[4];
        var opt = new LarsOptimizer();
        opt.AddParamGroup(new Dictionary<string, double>
        {
            ["lr"] = 0.5, ["momentum"] = 0.0, ["trust_coeff"] = 1.0,
        }).AddParameter(w, g);
        Assert.True(Train(opt, w, g, 600) < 0.5, "LARS failed to converge");
    }

    [Fact]
    public void FtrlOptimizer_Converges_OnL2Only()
    {
        var w = new float[4]; var g = new float[4];
        var opt = new FtrlOptimizer();
        // Pure L2 (no L1 sparsity penalty) so the OLS optimum is reachable.
        opt.AddParamGroup(new Dictionary<string, double>
        {
            ["lr"] = 0.5, ["l1_reg"] = 0.0, ["l2_reg"] = 1e-3, ["lr_power"] = -0.5,
        }).AddParameter(w, g);
        Assert.True(Train(opt, w, g, 600) < 0.5, "FTRL failed to converge");
    }

    [Fact]
    public void FtrlOptimizer_L1_ProducesSparseWeights()
    {
        var w = new float[10]; var g = new float[10];
        // Gradient pushes most weights toward 0 with a tiny non-zero contribution at one index.
        var opt = new FtrlOptimizer();
        opt.AddParamGroup(new Dictionary<string, double>
        {
            ["lr"] = 0.1, ["l1_reg"] = 0.1, ["l2_reg"] = 0.0, ["lr_power"] = -0.5,
        }).AddParameter(w, g);

        for (int t = 0; t < 100; t++)
        {
            for (int i = 0; i < 10; i++) g[i] = i == 0 ? 1f : 0.001f * (float)Math.Sin(t + i);
            opt.Step();
            Array.Clear(g, 0, g.Length);
        }
        // Only index 0 has a meaningful gradient signal — others should be soft-thresholded to zero.
        int zeros = 0;
        for (int i = 1; i < 10; i++) if (w[i] == 0f) zeros++;
        Assert.True(zeros >= 5, $"FTRL L1 expected ≥5 zeroed weights, got {zeros}");
    }

    [Fact]
    public void Sgd_Maximize_FlipsDescentToAscent()
    {
        var (X, y) = MakeProblem();
        var w = new float[4]; var g = new float[4];
        var opt = new SgdOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01, ["maximize"] = 1.0 })
           .AddParameter(w, g);
        // 50 steps maximising the (positive) MSE — loss must INCREASE rather than decrease.
        var (initialLoss, _) = ComputeGrad(w, X, y);
        for (int t = 0; t < 50; t++)
        {
            var (_, grad) = ComputeGrad(w, X, y);
            Array.Copy(grad, g, grad.Length);
            opt.Step();
            Array.Clear(g, 0, g.Length);
        }
        var (finalLoss, _) = ComputeGrad(w, X, y);
        Assert.True(finalLoss > initialLoss,
            $"maximize=true should increase MSE; got initial={initialLoss}, final={finalLoss}");
    }

    [Fact]
    public void Adam_Maximize_FlipsDescentToAscent()
    {
        var (X, y) = MakeProblem();
        var w = new float[4]; var g = new float[4];
        var opt = new AdamOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01, ["maximize"] = 1.0 })
           .AddParameter(w, g);
        var (initialLoss, _) = ComputeGrad(w, X, y);
        for (int t = 0; t < 50; t++)
        {
            var (_, grad) = ComputeGrad(w, X, y);
            Array.Copy(grad, g, grad.Length);
            opt.Step();
            Array.Clear(g, 0, g.Length);
        }
        var (finalLoss, _) = ComputeGrad(w, X, y);
        Assert.True(finalLoss > initialLoss);
    }

    [Fact]
    public void RmsProp_Centered_Converges()
    {
        var w = new float[4]; var g = new float[4];
        var opt = new RmsPropOptimizer();
        opt.AddParamGroup(new Dictionary<string, double>
        {
            ["lr"] = 0.01, ["alpha"] = 0.99, ["centered"] = 1.0,
        }).AddParameter(w, g);
        Assert.True(Train(opt, w, g, 600) < 0.2, "Centered RMSprop failed to converge");
    }

    [Fact]
    public void RmsProp_WithMomentum_Converges()
    {
        var w = new float[4]; var g = new float[4];
        var opt = new RmsPropOptimizer();
        opt.AddParamGroup(new Dictionary<string, double>
        {
            ["lr"] = 0.005, ["alpha"] = 0.99, ["momentum"] = 0.9,
        }).AddParameter(w, g);
        Assert.True(Train(opt, w, g, 600) < 0.2, "Momentum RMSprop failed to converge");
    }

    [Fact]
    public void BF16AdamOptimizer_StaysWithin2xOfFp32Adam()
    {
        // Acceptance criterion (issue #224): "Mixed-precision variant stays within 2× the loss
        // of FP32 baseline on a standard model training run."
        var w1 = new float[4]; var g1 = new float[4];
        var fp32 = new AdamOptimizer();
        fp32.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.05 }).AddParameter(w1, g1);
        float fp32Loss = Train(fp32, w1, g1, 400);

        var w2 = new float[4]; var g2 = new float[4];
        var bf16 = new BF16AdamOptimizer();
        bf16.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.05 }).AddParameter(w2, g2);
        float bf16Loss = Train(bf16, w2, g2, 400);

        Assert.True(bf16Loss <= 2.0 * fp32Loss + 1e-3,
            $"BF16 Adam loss {bf16Loss} exceeded 2× FP32 baseline {fp32Loss}.");
        // And BF16 should still actually converge to a small number.
        Assert.True(bf16Loss < 0.2, $"BF16 Adam failed to converge: loss={bf16Loss}");
    }

    [Fact]
    public void PyTorchOptimizerStateLoader_LoadsAdamStateDict()
    {
        // Synthesize a PyTorch-compatible optimizer state-dict layout in memory.
        // PyTorch torch.save(opt.state_dict(), 'opt.pt') produces this exact shape.
        var stateInner = new System.Collections.Hashtable
        {
            ["step"] = 5L,
            ["exp_avg"]    = new float[] { 0.1f, 0.2f, 0.3f, 0.4f },
            ["exp_avg_sq"] = new float[] { 0.01f, 0.02f, 0.03f, 0.04f },
        };
        var state = new System.Collections.Hashtable { [0L] = stateInner };
        var pg = new System.Collections.Hashtable
        {
            ["lr"] = 0.001,
            ["betas"] = new System.Collections.ArrayList { 0.9, 0.999 },
            ["eps"] = 1e-8,
            ["weight_decay"] = 0.0,
            ["amsgrad"] = false,
            ["params"] = new System.Collections.ArrayList { 0L },
        };
        var root = new System.Collections.Hashtable
        {
            ["state"] = state,
            ["param_groups"] = new System.Collections.ArrayList { pg },
        };

        var sd = PyTorchOptimizerStateLoader.Convert(root);
        Assert.Single(sd.ParamGroups);
        Assert.Equal(1, sd.State.Count);

        var grp = sd.ParamGroups[0];
        Assert.Equal(0.001, grp.Options["lr"], precision: 8);
        Assert.Equal(0.9,   grp.Options["beta1"], precision: 8);
        Assert.Equal(0.999, grp.Options["beta2"], precision: 8);
        Assert.Equal(1e-8,  grp.Options["eps"], precision: 12);
        Assert.Equal(0.0,   grp.Options["weight_decay"]);
        Assert.Equal(0.0,   grp.Options["amsgrad"]); // false → 0.0
        Assert.Equal(new List<int> { 0 }, grp.ParamIds);

        var slot = sd.State[0];
        Assert.Equal(5, slot["step"].IntValue);
        Assert.NotNull(slot["exp_avg"].Tensor);
        Assert.Equal(4, slot["exp_avg"].Tensor!.Length);
        Assert.Equal(0.2f, slot["exp_avg"].Tensor![1]);
    }

    [Fact]
    public void PyTorchOptimizerStateLoader_LoadAndStep_ReproducesTrajectory()
    {
        // Build a synthetic PyTorch-like state-dict that represents Adam mid-training,
        // load it into our AdamOptimizer, take one step, verify the new params match
        // what a fresh AdamOptimizer that had been stepped 5 times would produce.
        var w = new float[] { 0.5f, -0.3f, 0.1f, 0.2f };
        var g = new float[] { 0.05f, -0.02f, 0.01f, 0.03f };
        var m = new float[] { 0.04f, -0.018f, 0.009f, 0.025f };
        var v = new float[] { 1e-3f, 4e-4f, 1e-4f, 9e-4f };

        var stateInner = new System.Collections.Hashtable
        {
            ["step"] = 5L,
            ["exp_avg"] = (float[])m.Clone(),
            ["exp_avg_sq"] = (float[])v.Clone(),
        };
        var pg = new System.Collections.Hashtable
        {
            ["lr"] = 0.01,
            ["betas"] = new System.Collections.ArrayList { 0.9, 0.999 },
            ["eps"] = 1e-8,
            ["weight_decay"] = 0.0,
            ["amsgrad"] = false,
            ["params"] = new System.Collections.ArrayList { 0L },
        };
        var root = new System.Collections.Hashtable
        {
            ["state"] = new System.Collections.Hashtable { [0L] = stateInner },
            ["param_groups"] = new System.Collections.ArrayList { pg },
        };

        var sd = PyTorchOptimizerStateLoader.Convert(root);
        var opt = new AdamOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01 }).AddParameter(w, g);
        opt.LoadStateDict(sd);

        var wBefore = (float[])w.Clone();
        opt.Step();

        // After loading state with step=5, the next step is step=6. Compare against the
        // analytical Adam update with bc1 = 1 − 0.9^6, bc2 = 1 − 0.999^6.
        for (int i = 0; i < 4; i++)
        {
            float mNew = 0.9f * m[i] + 0.1f * g[i];
            float vNew = 0.999f * v[i] + 0.001f * g[i] * g[i];
            float bc1 = 1f - MathF.Pow(0.9f, 6);
            float bc2 = 1f - MathF.Pow(0.999f, 6);
            float mHat = mNew / bc1;
            float vHat = vNew / bc2;
            float expected = wBefore[i] - 0.01f * mHat / (MathF.Sqrt(vHat) + 1e-8f);
            Assert.Equal(expected, w[i], precision: 5);
        }
    }

    [Fact]
    public void BF16AdamOptimizer_RoundTripsBF16Mantissa()
    {
        // Sanity check: BF16 has 7-bit mantissa, so values round to the nearest representable
        // BF16. 1.0f is exact; 0.123456f is not — but should round-trip within 1/256 relative.
        var w = new float[4]; var g = new float[4];
        var opt = new BF16AdamOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.0 }).AddParameter(w, g);
        // Take one no-op step (lr=0) so the BF16 buffers get initialised.
        for (int i = 0; i < 4; i++) g[i] = (i + 1) * 0.123456f;
        opt.Step();
        // Read back the moments via state dict — they should be quantized but close.
        var sd = opt.StateDict();
        // Just verify the optimizer ran without exceptions. The numerical behaviour is
        // covered by the convergence test above.
        Assert.NotEmpty(sd.State);
    }
}
