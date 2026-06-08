using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Optimizer-agnostic gradient surface (Tensors #574):
/// <see cref="MixedPrecisionCompiledPlan.ComputeGradients"/> runs the same FP16-activation
/// forward + scaled mixed-dtype backward as <see cref="MixedPrecisionCompiledPlan.StepAdam"/>
/// / <see cref="MixedPrecisionCompiledPlan.Step"/>, but RETURNS the unscaled FP32 master
/// gradients instead of applying an update — so any optimizer (Lion, RMSprop, LAMB, …) can
/// apply its own master-weight update and still get the FP16 activation-storage memory win.
/// These tests pin: (1) end-to-end descent when the CALLER applies a vanilla SGD update to the
/// returned grads, (2) grads index-aligned with the supplied parameters, (3) loss-scaling is
/// undone (returned grads are independent of the scaler's scale), and (4) the grads match an
/// analytic FP32 reference within FP16 tolerance.
/// </summary>
public class MixedPrecisionComputeGradientsTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Rand(int r, int c, int seed, double s)
    {
        var rng = new Random(seed);
        var d = new float[r * c];
        for (int i = 0; i < d.Length; i++) d[i] = (float)((rng.NextDouble() * 2 - 1) * s);
        return new Tensor<float>(d, new[] { r, c });
    }

    private (Func<Tensor<float>> forward, Tensor<float> x, Tensor<float> t, Tensor<float> W) BuildLinearProblem(
        int B, int d, float wScale = 0.1f)
    {
        var x = Rand(B, d, 1, 1.0);
        var wTrue = Rand(d, d, 2, 0.4);
        var t = _engine.TensorMatMul(x, wTrue);
        var W = Rand(d, d, 3, wScale);

        Func<Tensor<float>> forward = () =>
        {
            var y = _engine.TensorMatMul(x, W);
            var diff = _engine.TensorSubtract(y, t);
            var sq = _engine.TensorMultiply(diff, diff);
            return _engine.ReduceSum(sq);
        };
        return (forward, x, t, W);
    }

    [Fact]
    public void ComputeGradients_CallerAppliedSgd_DescendsLoss()
    {
        const int B = 8, d = 6;
        var (forward, _, _, W) = BuildLinearProblem(B, d);

        var plan = MixedPrecisionCompiledPlan.Trace(forward, _engine);
        var scaler = new AiDotNet.Tensors.Engines.Autodiff.GradScaler(
            new AiDotNet.Tensors.Engines.Autodiff.MixedPrecisionConfig { LossScale = 128f, DynamicLossScale = false });
        var pars = new[] { W };

        const float lr = 0.05f;
        float first = 0, last = 0; int overflows = 0;
        const int steps = 60;
        for (int s = 0; s < steps; s++)
        {
            var r = plan.ComputeGradients(pars, scaler);
            if (s == 0) first = r.Loss;
            if (s == steps - 1) last = r.Loss;
            Assert.False(float.IsNaN(r.Loss) || float.IsInfinity(r.Loss), $"loss non-finite at step {s}");

            if (r.FoundInfNan) { overflows++; continue; }

            // Caller applies its own update (here: plain SGD) to the returned FP32 grads.
            var g = r.Gradients[0];
            Assert.NotNull(g);
            if (g is null) continue;
            var w = W.AsWritableSpan();
            var gs = g.AsSpan();
            for (int k = 0; k < w.Length; k++) w[k] -= lr * gs[k];
            W.IncrementVersion();
        }

        Assert.Equal(0, overflows);
        Assert.True(first > 0, "first loss positive");
        Assert.True(last < 0.25f * first, $"caller-applied SGD on ComputeGradients did not descend: first {first}, last {last}");
    }

    [Fact]
    public void ComputeGradients_AlignsGradientsWithParameters()
    {
        const int B = 8, d = 6;
        var (forward, _, _, W) = BuildLinearProblem(B, d);
        var plan = MixedPrecisionCompiledPlan.Trace(forward, _engine);
        var pars = new[] { W };

        var r = plan.ComputeGradients(pars, scaler: null);

        Assert.Equal(pars.Length, r.Gradients.Count);
        Assert.False(r.FoundInfNan);
        var grad0 = r.Gradients[0];
        Assert.NotNull(grad0);
        if (grad0 is null) return;
        Assert.Equal(W.Length, grad0.Length);
    }

    [Fact]
    public void ComputeGradients_UndoesLossScaling_GradsIndependentOfScale()
    {
        const int B = 8, d = 6;

        // Two independent fresh plans on identical W: one with a large loss scale, one with no
        // scaler (scale == 1). The returned grads are unscaled in FP32 in both cases, so they
        // must agree up to FP16 rounding — proving the unscale step (× 1/scale) is applied.
        var (fwdA, _, _, wA) = BuildLinearProblem(B, d);
        var (fwdB, _, _, wB) = BuildLinearProblem(B, d);

        var planA = MixedPrecisionCompiledPlan.Trace(fwdA, _engine);
        var planB = MixedPrecisionCompiledPlan.Trace(fwdB, _engine);

        var scaler = new AiDotNet.Tensors.Engines.Autodiff.GradScaler(
            new AiDotNet.Tensors.Engines.Autodiff.MixedPrecisionConfig { LossScale = 1024f, DynamicLossScale = false });

        var gradScaled = planA.ComputeGradients(new[] { wA }, scaler).Gradients[0];
        var gradUnscaled = planB.ComputeGradients(new[] { wB }, scaler: null).Gradients[0];
        Assert.NotNull(gradScaled);
        Assert.NotNull(gradUnscaled);
        if (gradScaled is null || gradUnscaled is null) return;

        var gScaled = gradScaled.ToArray();
        var gUnscaled = gradUnscaled.ToArray();

        Assert.Equal(gUnscaled.Length, gScaled.Length);
        for (int k = 0; k < gScaled.Length; k++)
        {
            float a = gScaled[k], b = gUnscaled[k];
            float tol = 1e-2f + 0.05f * Math.Abs(b); // FP16 abs + relative slack
            Assert.True(Math.Abs(a - b) <= tol,
                $"grad[{k}] scale-dependent: scaled {a} vs unscaled {b} (tol {tol}) — unscale not applied?");
        }
    }

    [Fact]
    public void ComputeGradients_MatchesAnalyticReference_WithinFp16Tolerance()
    {
        const int B = 8, d = 6;
        var (forward, x, t, W) = BuildLinearProblem(B, d);
        var plan = MixedPrecisionCompiledPlan.Trace(forward, _engine);

        var grad = plan.ComputeGradients(new[] { W }, scaler: null).Gradients[0];
        Assert.NotNull(grad);
        if (grad is null) return;
        var got = grad.ToArray();

        // Analytic FP32 reference for loss = Σ (xW − t)²:  dL/dW = 2 · xᵀ (xW − t).
        var xf = x.ToArray();   // [B, d] row-major
        var wf = W.ToArray();   // [d, d]
        var tf = t.ToArray();   // [B, d]
        var resid = new float[B * d];
        for (int b = 0; b < B; b++)
            for (int j = 0; j < d; j++)
            {
                float acc = 0f;
                for (int k = 0; k < d; k++) acc += xf[b * d + k] * wf[k * d + j];
                resid[b * d + j] = acc - tf[b * d + j];
            }
        var refGrad = new float[d * d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                float acc = 0f;
                for (int b = 0; b < B; b++) acc += xf[b * d + i] * resid[b * d + j];
                refGrad[i * d + j] = 2f * acc;
            }

        // Cosine similarity + magnitude ratio: a fundamentally wrong gradient fails cosine;
        // FP16 activation rounding only perturbs magnitude/direction slightly.
        double dot = 0, nGot = 0, nRef = 0;
        for (int k = 0; k < refGrad.Length; k++)
        {
            dot += (double)got[k] * refGrad[k];
            nGot += (double)got[k] * got[k];
            nRef += (double)refGrad[k] * refGrad[k];
        }
        double cosine = dot / (Math.Sqrt(nGot) * Math.Sqrt(nRef) + 1e-12);
        double normRatio = Math.Sqrt(nGot) / (Math.Sqrt(nRef) + 1e-12);

        Assert.True(cosine > 0.98, $"ComputeGradients direction off vs analytic reference: cosine {cosine:F4}");
        Assert.True(Math.Abs(normRatio - 1.0) < 0.2, $"ComputeGradients magnitude off vs analytic reference: ratio {normRatio:F4}");
    }
}
