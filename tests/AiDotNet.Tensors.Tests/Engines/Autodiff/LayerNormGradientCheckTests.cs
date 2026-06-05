using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Regression tests for the LayerNorm compiled-training fix (PR: "correct LayerNorm
/// mean/variance shape so the fused compiled-training path works for transformers").
///
/// Two guards:
///  1. SHAPE — mean/variance MUST be the multi-dim batchShape (the non-normalized
///     leading dims, e.g. [B,S] for a [B,S,D] input), NOT a flat [B*S]. The GPU
///     engine previously returned the flat shape, which is rank-inconsistent with
///     CpuEngine and mis-sized every gradient buffer in the compiled plan (crashing
///     transformer training with "Source array was not long enough").
///  2. GRADIENT — finite-difference vs analytic gradients for LayerNorm input/gamma/
///     beta on a rank-3 [B,S,D] input (the exact shape that broke). This is the
///     standing guard that the compiled/eager backward stays numerically correct, so
///     a regression cannot silently zero or corrupt gradients again.
///
/// Runs on whatever <see cref="AiDotNetEngine.Current"/> is configured (CPU in CI,
/// GPU when present) — the finite-difference check is backend-agnostic.
/// </summary>
public class LayerNormGradientCheckTests
{
    private readonly IEngine _engine = AiDotNetEngine.Current;

    private static Tensor<float> Make(int[] shape, Func<int, float> f)
    {
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = f(i);
        return new Tensor<float>(data, shape);
    }

    private static float Flat(Tensor<float> t, int i) => t.ToArray()[i];

    // Forward-only scalar loss L = sum( LayerNorm(input, gamma, beta) ). No tape active,
    // so this is a pure forward evaluation usable for finite differencing.
    private float ForwardLoss(Tensor<float> input, Tensor<float> gamma, Tensor<float> beta, double eps)
    {
        var y = _engine.LayerNorm(input, gamma, beta, eps, out _, out _);
        var s = _engine.ReduceSum(y, null);
        return Flat(s, 0);
    }

    [Fact]
    public void LayerNorm_MeanVariance_HaveBatchShape_NotFlattened()
    {
        // [B=2, S=3, D=4], normalize over the last dim (gamma length 4).
        var input = Make(new[] { 2, 3, 4 }, i => (float)Math.Sin(i * 0.7 + 1.0));
        var gamma = Make(new[] { 4 }, i => 1.0f + 0.1f * i);
        var beta = Make(new[] { 4 }, i => 0.05f * i);

        _ = _engine.LayerNorm(input, gamma, beta, 1e-5, out var mean, out var variance);

        // The normalized axis is the last dim (D=4); the statistics are per (B,S).
        Assert.Equal(new[] { 2, 3 }, mean.Shape.ToArray());
        Assert.Equal(new[] { 2, 3 }, variance.Shape.ToArray());
    }

    [Fact]
    public void LayerNorm_AnalyticGradients_MatchFiniteDifference_Rank3()
    {
        const double eps = 1e-5;
        var input = Make(new[] { 2, 3, 4 }, i => (float)Math.Sin(i * 0.7 + 1.0) * 0.5f);
        var gamma = Make(new[] { 4 }, i => 1.0f + 0.1f * i);
        var beta = Make(new[] { 4 }, i => 0.05f * i);

        // Analytic gradients via the autodiff tape (records LayerNorm + ReduceSum).
        Dictionary<Tensor<float>, Tensor<float>> grads;
        using (var tape = new GradientTape<float>())
        {
            var y = _engine.LayerNorm(input, gamma, beta, eps, out _, out _);
            var loss = _engine.ReduceSum(y, null);
            grads = tape.ComputeGradients(loss, new[] { input, gamma, beta });
        }
        var gInput = grads[input];
        var gGamma = grads[gamma];
        var gBeta = grads[beta];

        // Finite-difference a spread of elements of each tensor (not all — keep it fast).
        const float h = 1e-3f;
        const float tol = 5e-2f; // relative tolerance for single-precision finite differences

        void CheckTensor(Tensor<float> param, Tensor<float> analytic, int[] probeIdx)
        {
            var baseData = param.ToArray();
            foreach (int idx in probeIdx)
            {
                float orig = baseData[idx];
                var plus = (float[])baseData.Clone(); plus[idx] = orig + h;
                var minus = (float[])baseData.Clone(); minus[idx] = orig - h;
                float lp = ForwardLoss(
                    param == input ? new Tensor<float>(plus, param.Shape.ToArray()) : input,
                    param == gamma ? new Tensor<float>(plus, param.Shape.ToArray()) : gamma,
                    param == beta ? new Tensor<float>(plus, param.Shape.ToArray()) : beta, eps);
                float lm = ForwardLoss(
                    param == input ? new Tensor<float>(minus, param.Shape.ToArray()) : input,
                    param == gamma ? new Tensor<float>(minus, param.Shape.ToArray()) : gamma,
                    param == beta ? new Tensor<float>(minus, param.Shape.ToArray()) : beta, eps);
                float numeric = (lp - lm) / (2 * h);
                float exact = Flat(analytic, idx);
                float denom = Math.Max(1e-3f, Math.Max(Math.Abs(numeric), Math.Abs(exact)));
                Assert.True(Math.Abs(numeric - exact) / denom < tol,
                    $"grad mismatch at idx {idx}: numeric={numeric:F5} analytic={exact:F5}");
            }
        }

        // beta gradient is exactly 1 per output (sum of LayerNorm wrt beta) — strong, simple check.
        // (This guards the GPU bug where layernorm_grad_params was never launched, so gamma/beta
        // gradients were zero and LayerNorm's affine params never trained on GPU.)
        CheckTensor(beta, gBeta, new[] { 0, 1, 2, 3 });
        // gamma and input gradients exercise the through-normalization backward.
        CheckTensor(gamma, gGamma, new[] { 0, 2 });
        CheckTensor(input, gInput, new[] { 0, 5, 11, 17, 23 });
    }
}
