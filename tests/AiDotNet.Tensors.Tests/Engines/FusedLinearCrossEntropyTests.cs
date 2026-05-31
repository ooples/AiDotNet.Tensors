using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Correctness tests for the fused linear + cross-entropy-with-logits kernel
/// (<see cref="CpuEngine.FusedLinearCrossEntropyWithLogits{T}"/>, issue ooples/AiDotNet#1464).
/// Forward loss is checked against an independent (logits → log-softmax → NLL) reference; the custom
/// autodiff backward is checked against central finite differences of the loss for hidden/weight/bias.
/// </summary>
public class FusedLinearCrossEntropyTests
{
    private static double ReferenceLoss(double[] hidden, double[] weight, double[] bias, double[] target,
        int n, int d, int vocab)
    {
        double total = 0.0;
        for (int r = 0; r < n; r++)
        {
            var logit = new double[vocab];
            for (int v = 0; v < vocab; v++)
            {
                double s = bias[v];
                for (int j = 0; j < d; j++) s += hidden[r * d + j] * weight[j * vocab + v];
                logit[v] = s;
            }
            double max = logit[0];
            for (int v = 1; v < vocab; v++) if (logit[v] > max) max = logit[v];
            double sum = 0.0;
            for (int v = 0; v < vocab; v++) sum += Math.Exp(logit[v] - max);
            double lse = max + Math.Log(sum);
            for (int v = 0; v < vocab; v++) total += target[r * vocab + v] * (logit[v] - lse);
        }
        return -total / n;
    }

    private static double[] Gen(int len, int seed, double scale = 0.5)
    {
        var a = new double[len];
        for (int i = 0; i < len; i++) a[i] = Math.Sin(0.7 * (i + 1) + 1.3 * seed) * scale;
        return a;
    }

    // One-hot target per row.
    private static double[] OneHot(int n, int vocab, int seed)
    {
        var a = new double[n * vocab];
        for (int r = 0; r < n; r++) a[r * vocab + ((r * 7 + seed) % vocab)] = 1.0;
        return a;
    }

    [Fact]
    public void Forward_MatchesReference()
    {
        var engine = new CpuEngine();
        int n = 4, d = 5, vocab = 6;
        var hidden = new Tensor<double>(Gen(n * d, 1), new[] { n, d });
        var weight = new Tensor<double>(Gen(d * vocab, 2), new[] { d, vocab });
        var bias = new Tensor<double>(Gen(vocab, 3, 0.3), new[] { vocab });
        var target = new Tensor<double>(OneHot(n, vocab, 1), new[] { n, vocab });

        var loss = engine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, target);
        double got = ((double[])(object)loss.GetDataArray()!)[0];
        double expected = ReferenceLoss(
            (double[])(object)hidden.GetDataArray()!, (double[])(object)weight.GetDataArray()!,
            (double[])(object)bias.GetDataArray()!, (double[])(object)target.GetDataArray()!, n, d, vocab);
        Assert.True(Math.Abs(got - expected) < 1e-10, $"loss {got} vs reference {expected}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        int n = 3, d = 4, vocab = 5;
        var hidden = new Tensor<double>(Gen(n * d, 1), new[] { n, d });
        var weight = new Tensor<double>(Gen(d * vocab, 2), new[] { d, vocab });
        var bias = new Tensor<double>(Gen(vocab, 3, 0.3), new[] { vocab });
        var target = new Tensor<double>(OneHot(n, vocab, 1), new[] { n, vocab });

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var loss = engine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, target);
            grads = tape.ComputeGradients(loss, new[] { hidden, weight, bias });
        }

        const double eps = 1e-6;
        foreach (var input in new[] { hidden, weight, bias })
        {
            var data = (double[])(object)input.GetDataArray()!;
            var grad = grads[input];
            for (int i = 0; i < data.Length; i++)
            {
                double orig = data[i];
                data[i] = orig + eps;
                double lp = LossOf(engine, hidden, weight, bias, target);
                data[i] = orig - eps;
                double lm = LossOf(engine, hidden, weight, bias, target);
                data[i] = orig;
                double numeric = (lp - lm) / (2.0 * eps);
                double analytic = grad.GetFlat(i);
                double tol = 1e-5 + 1e-4 * Math.Abs(analytic);
                Assert.True(Math.Abs(numeric - analytic) < tol,
                    $"grad mismatch at element {i}: analytic={analytic}, finite-diff={numeric}");
            }
        }
    }

    private static double LossOf(CpuEngine engine, Tensor<double> hidden, Tensor<double> weight,
        Tensor<double> bias, Tensor<double> target)
    {
        var loss = engine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, target);
        return ((double[])(object)loss.GetDataArray()!)[0];
    }
}
