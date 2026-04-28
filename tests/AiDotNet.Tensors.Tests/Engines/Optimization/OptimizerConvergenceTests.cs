using System;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// Per-optimizer convergence tests on a deterministic linear-regression mini-benchmark.
/// Acceptance criterion (issue #224): every optimizer reaches the published target loss.
/// </summary>
public class OptimizerConvergenceTests
{
    // Linear regression: y = 3 x_0 − 2 x_1 + 1
    // 4 features (last is the bias channel), 64 samples, generated with a fixed seed.
    private static (float[][] X, float[] y) MakeProblem(int seed = 1234, int n = 64, int d = 4)
    {
        var rng = new Random(seed);
        var X = new float[n][];
        var y = new float[n];
        var w = new float[] { 3f, -2f, 0.5f, 1f /*bias*/ };
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

    private static float TrainAndGetFinalLoss(IOptimizer opt, float[] w, float[] grad,
                                              float[][] X, float[] y, int iters)
    {
        float final = 0f;
        for (int t = 0; t < iters; t++)
        {
            var (loss, g) = ComputeGrad(w, X, y);
            Array.Copy(g, grad, g.Length);
            opt.Step();
            // After step, optimizer may have modified grad (e.g. ASGD adds wd in place).
            // Re-zero the live grad buffer for next iteration.
            Array.Clear(grad, 0, grad.Length);
            final = loss;
        }
        return final;
    }

    private static (float[] w, float[] grad, IOptimizer opt) Setup<T>(Func<T> ctor, double lr) where T : OptimizerBase
    {
        var w = new float[4];
        var grad = new float[4];
        var opt = ctor();
        var group = opt.AddParamGroup(new System.Collections.Generic.Dictionary<string, double> { ["lr"] = lr });
        group.AddParameter(w, grad);
        return (w, grad, opt);
    }

    [Theory]
    [InlineData("sgd", 200, 0.1, 0.05)]
    [InlineData("sgd_momentum", 200, 0.05, 0.05)]
    [InlineData("adam", 400, 0.05, 0.05)]
    [InlineData("adamw", 400, 0.05, 0.05)]
    [InlineData("radam", 400, 0.05, 0.1)]
    [InlineData("nadam", 400, 0.05, 0.05)]
    [InlineData("adamax", 400, 0.05, 0.05)]
    [InlineData("adagrad", 400, 0.5, 0.05)]
    [InlineData("rmsprop", 400, 0.01, 0.05)]
    [InlineData("adadelta", 1500, 1.0, 0.5)]    // adadelta is slow on this scale
    [InlineData("lion", 800, 5e-3, 0.5)]
    [InlineData("rprop", 400, 0.01, 0.05)]
    [InlineData("asgd", 400, 0.05, 0.05)]
    public void Optimizer_Converges_OnLinearRegression(string name, int iters, double lr, double target)
    {
        var (X, y) = MakeProblem();
        IOptimizer opt;
        float[] w = new float[4];
        float[] grad = new float[4];

        switch (name)
        {
            case "sgd": opt = new SgdOptimizer(); break;
            case "sgd_momentum":
            {
                var s = new SgdOptimizer();
                s.AddParamGroup(new System.Collections.Generic.Dictionary<string, double> { ["lr"] = lr, ["momentum"] = 0.9 })
                 .AddParameter(w, grad);
                opt = s; break;
            }
            case "adam": opt = new AdamOptimizer(); break;
            case "adamw": opt = new AdamWOptimizer(); break;
            case "radam": opt = new RAdamOptimizer(); break;
            case "nadam": opt = new NAdamOptimizer(); break;
            case "adamax": opt = new AdamaxOptimizer(); break;
            case "adagrad": opt = new AdagradOptimizer(); break;
            case "rmsprop": opt = new RmsPropOptimizer(); break;
            case "adadelta": opt = new AdaDeltaOptimizer(); break;
            case "lion": opt = new LionOptimizer(); break;
            case "rprop": opt = new RpropOptimizer(); break;
            case "asgd": opt = new AsgdOptimizer(); break;
            default: throw new ArgumentException(name);
        }

        if (name != "sgd_momentum")
        {
            opt.AddParamGroup(new System.Collections.Generic.Dictionary<string, double> { ["lr"] = lr })
               .AddParameter(w, grad);
        }
        else
        {
            // group already added with proper params.
            w = (float[])opt.ParamGroups[0].Parameters[0];
            grad = (float[])opt.ParamGroups[0].Gradients[0];
        }

        float final = TrainAndGetFinalLoss(opt, w, grad, X, y, iters);
        Assert.True(final < target, $"{name}: final loss {final} >= target {target}");
    }

    [Fact]
    public void Adam_StateDict_RoundTrip_PreservesTrajectory()
    {
        var (X, y) = MakeProblem();
        var w1 = new float[4]; var g1 = new float[4];
        var opt1 = new AdamOptimizer();
        opt1.AddParamGroup(new System.Collections.Generic.Dictionary<string, double> { ["lr"] = 0.1 }).AddParameter(w1, g1);
        for (int t = 0; t < 10; t++) { var (_, gg) = ComputeGrad(w1, X, y); Array.Copy(gg, g1, gg.Length); opt1.Step(); }
        var sd = opt1.StateDict();

        // Build an identical optimizer with fresh params, load the snapshot mid-trajectory.
        var w2 = new float[4]; var g2 = new float[4];
        Array.Copy(w1, w2, w1.Length);
        var opt2 = new AdamOptimizer();
        opt2.AddParamGroup(new System.Collections.Generic.Dictionary<string, double> { ["lr"] = 0.1 }).AddParameter(w2, g2);
        opt2.LoadStateDict(sd);

        // Step both for 10 more iterations; weights must match exactly.
        for (int t = 0; t < 10; t++)
        {
            var (_, gg1) = ComputeGrad(w1, X, y); Array.Copy(gg1, g1, gg1.Length); opt1.Step();
            var (_, gg2) = ComputeGrad(w2, X, y); Array.Copy(gg2, g2, gg2.Length); opt2.Step();
        }
        for (int j = 0; j < 4; j++)
            Assert.Equal(w1[j], w2[j], precision: 5);
    }

    [Fact]
    public void SparseAdam_OnlyUpdates_NonZeroGradEntries()
    {
        var w = new float[10];
        var g = new float[10];
        // Only positions 2 and 7 receive non-zero gradient.
        g[2] = 1f; g[7] = -2f;

        var opt = new SparseAdamOptimizer();
        opt.AddParamGroup(new System.Collections.Generic.Dictionary<string, double> { ["lr"] = 0.1 }).AddParameter(w, g);
        opt.Step();

        for (int i = 0; i < w.Length; i++)
        {
            if (i == 2 || i == 7) Assert.NotEqual(0f, w[i]);
            else Assert.Equal(0f, w[i]);
        }
    }

    [Fact]
    public void LBFGS_Converges_OnRosenbrock()
    {
        // Classic 2-D Rosenbrock: f(x,y) = (1−x)^2 + 100(y−x²)².
        // Minimum at (1, 1).
        var x = new float[] { -1.2f, 1.0f };
        var grad = new float[2];
        var opt = new LBFGSOptimizer(new[] { x }, new[] { grad },
                                     lr: 1f, maxIter: 100, historySize: 20, lineSearchFn: "strong_wolfe");

        for (int outer = 0; outer < 5; outer++)
        {
            opt.Step(() =>
            {
                Array.Clear(grad, 0, grad.Length);
                float a = 1f - x[0];
                float b = x[1] - x[0] * x[0];
                grad[0] = -2f * a - 400f * x[0] * b;
                grad[1] = 200f * b;
                return a * a + 100f * b * b;
            });
        }
        Assert.InRange(x[0], 0.99f, 1.01f);
        Assert.InRange(x[1], 0.99f, 1.01f);
    }

    [Fact]
    public void ZeroGrad_ClearsAllGradients()
    {
        var w = new float[5];
        var g = new float[5];
        for (int i = 0; i < 5; i++) g[i] = i + 1;

        var opt = new SgdOptimizer();
        opt.AddParamGroup(new System.Collections.Generic.Dictionary<string, double> { ["lr"] = 0.1 }).AddParameter(w, g);
        opt.ZeroGrad();
        for (int i = 0; i < 5; i++) Assert.Equal(0f, g[i]);
    }
}
