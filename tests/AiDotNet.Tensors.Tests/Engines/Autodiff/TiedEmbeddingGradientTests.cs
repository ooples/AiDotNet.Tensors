using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// A tied embedding table W [vocab, dim] is read twice: by an axis-0 gather (token lookup, whose backward may record a
/// sparse embedding gradient) and as the transposed LM-head weight (dense gradient). Its tape gradient must be the
/// SUM of both contributions. Checked against central finite differences, with a repeated token id.
/// </summary>
public class TiedEmbeddingGradientTests
{
    private static readonly CpuEngine Engine = new();

    private static Tensor<double> Loss(Tensor<double> w, Tensor<int> idx, Tensor<double> r)
    {
        var h = Engine.TensorGather(w, idx, 0);                 // [n, dim]
        var logits = Engine.TensorMatMulTransposed(h, w);        // [n, vocab] = h @ W^T
        return Engine.ReduceSum(Engine.TensorMultiply(logits, r), null);
    }

    [Fact]
    public void TiedTableGradient_IsGatherPlusHead()
    {
        const int vocab = 6, dim = 4;
        var rng = new Random(5);
        var w = new Tensor<double>(new[] { vocab, dim });
        for (int i = 0; i < w.Length; i++) w[i] = rng.NextDouble() - 0.5;
        var idx = new Tensor<int>(new[] { 2, 4, 2 }, new[] { 3 });   // token 2 repeats
        var r = new Tensor<double>(new[] { 3, vocab });
        for (int i = 0; i < r.Length; i++) r[i] = rng.NextDouble() - 0.5;

        double[] analytic;
        using (var tape = new GradientTape<double>())
        {
            var loss = Loss(w, idx, r);
            analytic = tape.ComputeGradients(loss, new[] { w })[w].ToArray();
        }

        const double eps = 1e-6;
        for (int i = 0; i < w.Length; i++)
        {
            double orig = w[i];
            w[i] = orig + eps; double lp = Loss(w, idx, r)[0];
            w[i] = orig - eps; double lm = Loss(w, idx, r)[0];
            w[i] = orig;
            double numeric = (lp - lm) / (2 * eps);
            Assert.True(Math.Abs(numeric - analytic[i]) < 1e-6,
                $"dW[{i / dim},{i % dim}] analytic {analytic[i]:G8} vs numeric {numeric:G8}");
        }
    }
}
