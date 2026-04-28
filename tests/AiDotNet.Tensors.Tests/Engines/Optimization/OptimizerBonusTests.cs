using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Optimization.Optimizers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Optimization;

/// <summary>
/// Tests for the issue #224 bonus items: Shampoo, D-Adaptation, Prodigy, FP8 Lion,
/// 2:4 SparseAdam, ZeRO sharding, and the CUDA-graph-safe attribution.
/// </summary>
public class OptimizerBonusTests
{
    // -------- problem helpers (copied/adapted from existing tests) --------
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
        var grad = new float[d]; float loss = 0;
        for (int i = 0; i < n; i++)
        {
            float pred = 0; for (int j = 0; j < d; j++) pred += w[j] * X[i][j];
            float r = pred - y[i]; loss += r * r;
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

    // -------------------------------- Shampoo --------------------------------

    [Fact]
    public void Shampoo_DiagonalFallback_ConvergesOn1DProblem()
    {
        // 1-D parameter triggers the diagonal-AdaGrad-style fallback path.
        var w = new float[4]; var g = new float[4];
        var opt = new ShampooOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.5, ["momentum"] = 0.9 })
           .AddParameter(w, g);
        Assert.True(Train(opt, w, g, 600) < 0.5, "Shampoo (diagonal fallback) failed to converge");
    }

    [Fact]
    public void Shampoo_FullPreconditioner_ConvergesOnMatrixProblem()
    {
        // Build a small 2x2 matrix MSE problem so Shampoo's full L^-1/4 g R^-1/4 path runs.
        // Target W*X = Y where W is the 2x2 we want to recover.
        var W = new float[] { 0.5f, -0.3f, 0.2f, 0.7f };  // ground truth
        var paramFlat = new float[4];
        var gradFlat  = new float[4];

        var rng = new Random(42);
        const int n = 32;
        var Xs = new float[n][];
        var Ys = new float[n][];
        for (int i = 0; i < n; i++)
        {
            var x = new float[] { (float)rng.NextGaussian(), (float)rng.NextGaussian() };
            Xs[i] = x;
            // y = W * x  (W as 2x2 row-major)
            Ys[i] = new[]
            {
                W[0] * x[0] + W[1] * x[1],
                W[2] * x[0] + W[3] * x[1],
            };
        }

        var opt = new ShampooOptimizer();
        opt.Add2DParameter(2, 2, paramFlat, gradFlat,
            new Dictionary<string, double>
            {
                ["lr"] = 0.05, ["momentum"] = 0.9, ["precondition_freq"] = 5.0,
            });

        for (int t = 0; t < 1000; t++)
        {
            // grad = 2/N · sum (W·x - y) ⊗ x   (row-major)
            Array.Clear(gradFlat, 0, gradFlat.Length);
            for (int i = 0; i < n; i++)
            {
                float pred0 = paramFlat[0] * Xs[i][0] + paramFlat[1] * Xs[i][1];
                float pred1 = paramFlat[2] * Xs[i][0] + paramFlat[3] * Xs[i][1];
                float r0 = pred0 - Ys[i][0]; float r1 = pred1 - Ys[i][1];
                gradFlat[0] += 2f * r0 * Xs[i][0] / n;
                gradFlat[1] += 2f * r0 * Xs[i][1] / n;
                gradFlat[2] += 2f * r1 * Xs[i][0] / n;
                gradFlat[3] += 2f * r1 * Xs[i][1] / n;
            }
            opt.Step();
        }
        // params should now be close to W.
        for (int j = 0; j < 4; j++) Assert.InRange(paramFlat[j], W[j] - 0.1f, W[j] + 0.1f);
    }

    // ---------------------------- D-Adaptation -------------------------------

    [Fact]
    public void DAdaptAdam_AdaptsLrUpward()
    {
        var w = new float[4]; var g = new float[4];
        var opt = new DAdaptAdamOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 1.0, ["d0"] = 1e-4 })
           .AddParameter(w, g);
        double initialD = opt.CurrentD.Length > 0 ? opt.CurrentD[0] : 1e-4;
        // Step a few times so d_hat can grow.
        Train(opt, w, g, 100);
        // d should have grown above its initial value.
        Assert.True(opt.CurrentD[0] > initialD,
            $"D-Adapt-Adam d should grow; initial={initialD}, final={opt.CurrentD[0]}");
    }

    [Fact]
    public void DAdaptAdam_Converges_NoManualLr()
    {
        // D-Adaptation grows d quickly via the dHat estimate. Without a growth cap it can
        // overshoot on simple problems; cap d at 5%/step so adaptation is stable.
        var w = new float[4]; var g = new float[4];
        var opt = new DAdaptAdamOptimizer();
        opt.AddParamGroup(new Dictionary<string, double>
        {
            ["lr"] = 1.0, ["d0"] = 1e-3, ["growth_rate"] = 1.05,
        }).AddParameter(w, g);
        float final = Train(opt, w, g, 2000);
        Assert.True(final < 1.0, $"D-Adapt-Adam loss did not improve enough: final={final}");
    }

    [Fact]
    public void Prodigy_Converges_NoManualLr()
    {
        var w = new float[4]; var g = new float[4];
        var opt = new ProdigyOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 1.0, ["d0"] = 1e-3 })
           .AddParameter(w, g);
        Assert.True(Train(opt, w, g, 1000) < 1.0, "Prodigy failed to converge");
    }

    // ------------------------------ FP8 Lion ---------------------------------

    [Fact]
    public void FP8Lion_Converges()
    {
        var w = new float[4]; var g = new float[4];
        var opt = new FP8LionOptimizer();
        opt.AddParamGroup(new Dictionary<string, double> { ["lr"] = 5e-3 })
           .AddParameter(w, g);
        Assert.True(Train(opt, w, g, 1500) < 0.5, "FP8 Lion failed to converge");
    }

    [Fact]
    public void FP8Lion_StaysWithin3xOfFp32Lion()
    {
        var w1 = new float[4]; var g1 = new float[4];
        var fp32 = new LionOptimizer();
        fp32.AddParamGroup(new Dictionary<string, double> { ["lr"] = 5e-3 }).AddParameter(w1, g1);
        float fp32Loss = Train(fp32, w1, g1, 1000);

        var w2 = new float[4]; var g2 = new float[4];
        var fp8 = new FP8LionOptimizer();
        fp8.AddParamGroup(new Dictionary<string, double> { ["lr"] = 5e-3 }).AddParameter(w2, g2);
        float fp8Loss = Train(fp8, w2, g2, 1000);

        Assert.True(fp8Loss <= 3.0 * fp32Loss + 1e-3,
            $"FP8 Lion loss {fp8Loss} exceeded 3× FP32 Lion baseline {fp32Loss}.");
    }

    // ---------------------------- 2:4 SparseAdam -----------------------------

    [Fact]
    public void SparseAdam24_OnlyUpdatesLivePositions()
    {
        // 8 params, 2 blocks of 4. Pattern: indices (0,1) live in block 0; (2,3) live in block 1.
        var w = new float[8]; var g = new float[8];
        // Pack pattern: byte 0 holds nibbles for (block0, block1).
        // Block 0 nibble: idx0=0, idx1=1 → 0x10? Wait — bits 0-1 = idx0, bits 2-3 = idx1.
        // So idx0=0 (00), idx1=1 (01) → nibble = (01 << 2) | 00 = 0x4
        // Block 1 nibble: idx0=2 (10), idx1=3 (11) → nibble = (11 << 2) | 10 = 0xE
        // Two nibbles per byte: low nibble = block0, high nibble = block1.
        var pattern = new byte[] { 0xE4 };

        // Gradient: each "live" position has 1.0 / "dead" 999 (the kernel must NOT read these).
        for (int i = 0; i < 8; i++) g[i] = 999f;
        g[0] = 1f; g[1] = 1f; g[6] = 1f; g[7] = 1f;

        var opt = new SparseAdam24Optimizer();
        opt.AddSparse24Parameter(w, g, pattern, new Dictionary<string, double> { ["lr"] = 0.1 });
        opt.Step();

        // Live positions (0, 1, 6, 7) should be moved by Adam; dead positions (2, 3, 4, 5) stay 0.
        Assert.NotEqual(0f, w[0]);
        Assert.NotEqual(0f, w[1]);
        Assert.NotEqual(0f, w[6]);
        Assert.NotEqual(0f, w[7]);
        Assert.Equal(0f, w[2]);
        Assert.Equal(0f, w[3]);
        Assert.Equal(0f, w[4]);
        Assert.Equal(0f, w[5]);
    }

    // --------------------------- ZeRO sharding -------------------------------

    [Fact]
    public void ZeroShardedOptimizer_PartitionsParamIdsAcrossRanks()
    {
        var inner = new AdamOptimizer();
        var grp = inner.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01 });
        for (int i = 0; i < 8; i++) grp.AddParameter(new float[4], new float[4]);

        var rank0 = new ZeroShardedOptimizer(inner, rank: 0, worldSize: 4);
        var rank1 = new ZeroShardedOptimizer(inner, rank: 1, worldSize: 4);
        var rank2 = new ZeroShardedOptimizer(inner, rank: 2, worldSize: 4);
        var rank3 = new ZeroShardedOptimizer(inner, rank: 3, worldSize: 4);

        Assert.Equal(new[] { 0, 4 }, rank0.LocalParamIds.ToArray());
        Assert.Equal(new[] { 1, 5 }, rank1.LocalParamIds.ToArray());
        Assert.Equal(new[] { 2, 6 }, rank2.LocalParamIds.ToArray());
        Assert.Equal(new[] { 3, 7 }, rank3.LocalParamIds.ToArray());
    }

    [Fact]
    public void ZeroShardedOptimizer_LocalStateDict_ContainsOnlyLocalParams()
    {
        var inner = new AdamOptimizer();
        var grp = inner.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01 });
        for (int i = 0; i < 4; i++) grp.AddParameter(new float[4], new float[4]);
        // Take one step so the state dict is populated.
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) inner.ParamGroups[0].Gradients[i][j] = 0.1f;
        inner.Step();

        var rank0 = new ZeroShardedOptimizer(inner, rank: 0, worldSize: 2);
        var local0 = rank0.LocalStateDict();
        Assert.Contains(0, local0.State.Keys);
        Assert.Contains(2, local0.State.Keys);
        Assert.DoesNotContain(1, local0.State.Keys);
        Assert.DoesNotContain(3, local0.State.Keys);
    }

    [Fact]
    public void ZeroShardedOptimizer_RoundTripsViaShards()
    {
        // Build optimizer A, step, shard, then load all shards into a fresh optimizer B,
        // verify B's state matches A's.
        var innerA = new AdamOptimizer();
        var grpA = innerA.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01 });
        for (int i = 0; i < 4; i++) grpA.AddParameter(new float[4], new float[4]);
        for (int i = 0; i < 4; i++) for (int j = 0; j < 4; j++) innerA.ParamGroups[0].Gradients[i][j] = 0.1f;
        innerA.Step();

        var rank0 = new ZeroShardedOptimizer(innerA, 0, 2);
        var rank1 = new ZeroShardedOptimizer(innerA, 1, 2);
        var shards = new[] { rank0.LocalStateDict(), rank1.LocalStateDict() };

        var innerB = new AdamOptimizer();
        var grpB = innerB.AddParamGroup(new Dictionary<string, double> { ["lr"] = 0.01 });
        for (int i = 0; i < 4; i++) grpB.AddParameter(new float[4], new float[4]);
        var shardedB = new ZeroShardedOptimizer(innerB, 0, 2);
        shardedB.LoadShardedStateDict(shards);

        var sdA = innerA.StateDict();
        var sdB = innerB.StateDict();
        Assert.Equal(sdA.State.Count, sdB.State.Count);
        foreach (var pid in sdA.State.Keys)
        {
            Assert.True(sdB.State.ContainsKey(pid));
            Assert.Equal(sdA.State[pid]["step"].IntValue, sdB.State[pid]["step"].IntValue);
            for (int j = 0; j < 4; j++)
                Assert.Equal(sdA.State[pid]["exp_avg"].Tensor![j],
                             sdB.State[pid]["exp_avg"].Tensor![j], precision: 6);
        }
    }

    // --------------------------- CUDA Graph safety ---------------------------

    [Theory]
    [InlineData(typeof(SgdOptimizer))]
    [InlineData(typeof(AdamOptimizer))]
    [InlineData(typeof(AdamWOptimizer))]
    [InlineData(typeof(RAdamOptimizer))]
    [InlineData(typeof(NAdamOptimizer))]
    [InlineData(typeof(AdamaxOptimizer))]
    [InlineData(typeof(AdagradOptimizer))]
    [InlineData(typeof(RmsPropOptimizer))]
    [InlineData(typeof(AdaDeltaOptimizer))]
    [InlineData(typeof(LionOptimizer))]
    [InlineData(typeof(AsgdOptimizer))]
    [InlineData(typeof(LambOptimizer))]
    [InlineData(typeof(LarsOptimizer))]
    [InlineData(typeof(BF16AdamOptimizer))]
    [InlineData(typeof(SparseAdam24Optimizer))]
    public void CapturedSafe_OptimizersCarryAttribute(Type optimizerType)
    {
        var attr = Attribute.GetCustomAttribute(optimizerType, typeof(CudaGraphSafeAttribute));
        Assert.NotNull(attr);
    }

    [Theory]
    [InlineData(typeof(SparseAdamOptimizer))]   // runtime zero-detection branches
    [InlineData(typeof(RpropOptimizer))]        // sign-change control flow
    [InlineData(typeof(FtrlOptimizer))]         // L1 soft-thresholding control flow
    [InlineData(typeof(ShampooOptimizer))]      // Jacobi convergence loop
    [InlineData(typeof(DAdaptAdamOptimizer))]   // d-update bound on sk_l1>0
    [InlineData(typeof(ProdigyOptimizer))]
    [InlineData(typeof(FP8LionOptimizer))]      // hysteresis on max-abs
    public void NotCaptureSafe_OptimizersDoNotCarryAttribute(Type optimizerType)
    {
        var attr = Attribute.GetCustomAttribute(optimizerType, typeof(CudaGraphSafeAttribute));
        Assert.Null(attr);
    }
}

internal static class RandomGaussianExtensions
{
    private static double _saved; private static bool _haveSaved;
    public static double NextGaussian(this Random rng)
    {
        if (_haveSaved) { _haveSaved = false; return _saved; }
        double u1 = rng.NextDouble(); double u2 = rng.NextDouble();
        if (u1 < 1e-12) u1 = 1e-12;
        double z0 = Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
        double z1 = Math.Sqrt(-2 * Math.Log(u1)) * Math.Sin(2 * Math.PI * u2);
        _saved = z1; _haveSaved = true;
        return z0;
    }
}
