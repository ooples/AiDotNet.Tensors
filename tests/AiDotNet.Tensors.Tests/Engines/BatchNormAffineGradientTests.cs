using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

// #639: validates the differentiable affine batch-norm op (BatchNormAffine) — the single fused
// op that replaces the ~6-primitive batch=1 BN training fallback. Forward must equal
// BatchNormInference; backward must equal the exact closed-form gradient of a linear loss
// L = Σ(y ⊙ w), since y = gamma·(x-mean)·inv + beta is affine in x/gamma/beta.
public class BatchNormAffineGradientTests
{
    private readonly CpuEngine _engine = new CpuEngine();

    private static Tensor<float> Rnd(int[] shape, int seed)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)Math.Sin(i * 0.7 + seed) * 0.5f;
        return new Tensor<float>(a, shape);
    }

    private static Tensor<float> PosRnd(int[] shape, int seed)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = 0.5f + 0.4f * (float)Math.Abs(Math.Sin(i * 1.3 + seed)); // in [0.5, 0.9]
        return new Tensor<float>(a, shape);
    }

    [Fact]
    public void BatchNormAffine_ForwardMatchesBatchNormInference()
    {
        int N = 2, C = 3, H = 4, W = 5;
        var x = Rnd(new[] { N, C, H, W }, 1);
        var gamma = Rnd(new[] { C }, 2);
        var beta = Rnd(new[] { C }, 3);
        var mean = Rnd(new[] { C }, 4);
        var variance = PosRnd(new[] { C }, 5);
        double eps = 1e-3;

        var a = _engine.BatchNormAffine(x, gamma, beta, mean, variance, eps).GetFlattenedData();
        var b = _engine.BatchNormInference(x, gamma, beta, mean, variance, eps).GetFlattenedData();
        Assert.Equal(b.Length, a.Length);
        for (int i = 0; i < a.Length; i++) Assert.Equal(b[i], a[i], 5);
    }

    [Fact]
    public void BatchNormAffine_Backward_MatchesClosedForm()
    {
        int N = 2, C = 3, H = 2, W = 2;
        var x = Rnd(new[] { N, C, H, W }, 1);
        var gamma = Rnd(new[] { C }, 2);
        var beta = Rnd(new[] { C }, 3);
        var mean = Rnd(new[] { C }, 4);
        var variance = PosRnd(new[] { C }, 5);
        var w = Rnd(new[] { N, C, H, W }, 6); // loss weights → dL/dy = w
        double eps = 1e-3;

        float[] gx, gg, gb;
        using (var tape = new GradientTape<float>())
        {
            var y = _engine.BatchNormAffine(x, gamma, beta, mean, variance, eps);
            var loss = _engine.ReduceSum(_engine.TensorMultiply(y, w), null);
            var grads = tape.ComputeGradients(loss, new[] { x, gamma, beta });
            gx = (float[])grads[x].GetFlattenedData().Clone();
            gg = (float[])grads[gamma].GetFlattenedData().Clone();
            gb = (float[])grads[beta].GetFlattenedData().Clone();
        }

        // Closed-form reference for L = Σ_{n,c,i} w·y, y = gamma·(x-mean)·inv + beta:
        //   dL/dx[n,c,i] = w · gamma[c]·inv[c]
        //   dL/dgamma[c] = Σ_{n,i} w · (x-mean[c])·inv[c]
        //   dL/dbeta[c]  = Σ_{n,i} w
        var xs = x.GetFlattenedData(); var ws = w.GetFlattenedData();
        var gm = gamma.GetFlattenedData(); var mn = mean.GetFlattenedData(); var vr = variance.GetFlattenedData();
        int hw = H * W, chw = C * hw;
        var refDx = new float[N * chw]; var refDg = new float[C]; var refDb = new float[C];
        for (int c = 0; c < C; c++)
        {
            float inv = 1f / MathF.Sqrt(vr[c] + (float)eps);
            float scale = gm[c] * inv;
            float sg = 0f, sb = 0f;
            for (int n = 0; n < N; n++)
                for (int i = 0; i < hw; i++)
                {
                    int idx = n * chw + c * hw + i;
                    refDx[idx] = ws[idx] * scale;
                    sg += ws[idx] * (xs[idx] - mn[c]) * inv;
                    sb += ws[idx];
                }
            refDg[c] = sg; refDb[c] = sb;
        }

        for (int i = 0; i < refDx.Length; i++) Assert.Equal(refDx[i], gx[i], 4);
        for (int c = 0; c < C; c++) Assert.Equal(refDg[c], gg[c], 3);
        for (int c = 0; c < C; c++) Assert.Equal(refDb[c], gb[c], 3);
    }
}
