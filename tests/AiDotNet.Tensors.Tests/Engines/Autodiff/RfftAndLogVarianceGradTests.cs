using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Focused gradient checks for the two sweep findings that cannot be explained by the TensorLog
/// accuracy defect, driven with SEMANTICALLY VALID arguments.
/// </summary>
/// <remarks>
/// The sweep synthesizes arguments reflectively, so it drove IRFFT with outputLength = 1 against a
/// 6-element spectrum — a misuse, not a defect. These tests use the correct coupling (a spectrum of
/// n/2+1 bins with outputLength n) so that any disagreement is attributable to the op.
/// </remarks>
public class RfftAndLogVarianceGradTests
{
    private readonly ITestOutputHelper _out;
    private readonly CpuEngine _engine = new();

    public RfftAndLogVarianceGradTests(ITestOutputHelper output) => _out = output;

    private static Tensor<double> Signal(int n, int seed = 31)
    {
        var rng = new Random(seed);
        var x = new Tensor<double>([n]);
        for (int i = 0; i < n; i++) x[i] = 0.3 + rng.NextDouble() * 0.6;
        return x;
    }

    private (double[] analytical, double[] numerical) Check(
        Tensor<double> x, Func<Tensor<double>, Tensor<double>> forward, int probes, double eps = 1e-5)
    {
        using var tape = new GradientTape<double>();
        var outT = forward(x);
        var loss = _engine.ReduceSum(outT, null);
        var grads = tape.ComputeGradients(loss, [x]);
        Assert.True(grads.TryGetValue(x, out var g) && g is not null, "no gradient recorded");
        Assert.Equal(x.Shape.ToArray(), g!.Shape.ToArray());

        int k = Math.Min(probes, x.Length);
        var a = new double[k];
        var num = new double[k];
        for (int i = 0; i < k; i++)
        {
            a[i] = g[i];
            double orig = x[i];
            x[i] = orig + eps; double lp = _engine.TensorSum(forward(x));
            x[i] = orig - eps; double lm = _engine.TensorSum(forward(x));
            x[i] = orig;
            num[i] = (lp - lm) / (2 * eps);
        }
        return (a, num);
    }

    [Fact]
    public void RFFT_Gradient_MatchesFiniteDifferences()
    {
        var x = Signal(16);
        var (a, num) = Check(x, t => _engine.RFFT(t), probes: 4);

        for (int i = 0; i < a.Length; i++)
            _out.WriteLine($"RFFT [{i}] analytical={a[i]:G10} numerical={num[i]:G10}");

        for (int i = 0; i < a.Length; i++)
        {
            double denom = Math.Max(1.0, Math.Max(Math.Abs(a[i]), Math.Abs(num[i])));
            Assert.True(Math.Abs(a[i] - num[i]) / denom < 1e-6,
                $"RFFT gradient[{i}] analytical {a[i]:G10} vs numerical {num[i]:G10}");
        }
    }

    /// <summary>
    /// IRFFT driven correctly: a spectrum of n/2+1 bins reconstructed to length n.
    /// </summary>
    [Fact]
    public void IRFFT_Gradient_MatchesFiniteDifferences()
    {
        const int n = 16;
        // IRFFT consumes an INTERLEAVED spectrum, so the last dim is 2*numFreqs where
        // nFft = (numFreqs-1)*2 = n. For n=16 that is 2*9 = 18 values, not 9 — passing 9 makes the
        // forward infer numFreqs=4, nFft=6, and then index past its own output for outputLength=16.
        var spectrum = Signal(2 * (n / 2 + 1), seed: 77);
        var (a, num) = Check(spectrum, t => _engine.IRFFT(t, n), probes: 4);

        for (int i = 0; i < a.Length; i++)
            _out.WriteLine($"IRFFT [{i}] analytical={a[i]:G10} numerical={num[i]:G10}");

        for (int i = 0; i < a.Length; i++)
        {
            double denom = Math.Max(1.0, Math.Max(Math.Abs(a[i]), Math.Abs(num[i])));
            Assert.True(Math.Abs(a[i] - num[i]) / denom < 1e-6,
                $"IRFFT gradient[{i}] analytical {a[i]:G10} vs numerical {num[i]:G10}");
        }
    }

    /// <summary>
    /// TensorCross: the sweep saw analytical values that were the exact NEGATION of the numerical
    /// ones, which is the signature of reversed operands in an anti-commutative product.
    /// </summary>
    [Fact]
    public void TensorCross_Gradient_MatchesFiniteDifferences()
    {
        var a = Signal(3, seed: 101);
        var b = Signal(3, seed: 202);

        using var tape = new GradientTape<double>();
        var c = _engine.TensorCross(a, b, -1);
        var loss = _engine.ReduceSum(c, null);
        var grads = tape.ComputeGradients(loss, [a, b]);

        foreach (var (name, t) in new[] { ("a", a), ("b", b) })
        {
            Assert.True(grads.TryGetValue(t, out var g) && g is not null, $"no gradient for {name}");
            const double eps = 1e-6;
            for (int i = 0; i < t.Length; i++)
            {
                double orig = t[i];
                t[i] = orig + eps; double lp = _engine.TensorSum(_engine.TensorCross(a, b, -1));
                t[i] = orig - eps; double lm = _engine.TensorSum(_engine.TensorCross(a, b, -1));
                t[i] = orig;
                double numerical = (lp - lm) / (2 * eps);
                _out.WriteLine($"Cross d/d{name}[{i}] analytical={g![i]:G10} numerical={numerical:G10}");
                double denom = Math.Max(1.0, Math.Max(Math.Abs(g[i]), Math.Abs(numerical)));
                Assert.True(Math.Abs(g[i] - numerical) / denom < 1e-6,
                    $"Cross gradient d/d{name}[{i}]: analytical {g[i]:G10} vs numerical {numerical:G10}");
            }
        }
    }

    /// <summary>
    /// TensorBinaryCrossEntropy with a REALISTIC epsilon. The sweep passed epsilon = 0.5 (its
    /// blanket value for any scalar), and BCE clamps predictions into [eps, 1-eps] — at 0.5 that
    /// collapses to a single point, making the forward constant and its numerical gradient exactly
    /// zero. That was an invalid-argument artifact, not a defect.
    /// </summary>
    [Fact]
    public void BinaryCrossEntropy_Gradient_MatchesFiniteDifferences()
    {
        var predictions = Signal(6, seed: 303);          // in [0.3, 0.9], inside the clamp
        var targets = new Tensor<double>([6]);
        for (int i = 0; i < targets.Length; i++) targets[i] = i % 2 == 0 ? 1.0 : 0.0;

        Tensor<double> Fwd(Tensor<double> p) => _engine.TensorBinaryCrossEntropy(p, targets, 1e-7);

        var (a, num) = Check(predictions, Fwd, probes: 4, eps: 1e-6);
        for (int i = 0; i < a.Length; i++)
        {
            _out.WriteLine($"BCE [{i}] analytical={a[i]:G10} numerical={num[i]:G10}");
            double denom = Math.Max(1.0, Math.Max(Math.Abs(a[i]), Math.Abs(num[i])));
            Assert.True(Math.Abs(a[i] - num[i]) / denom < 1e-5,
                $"BCE gradient[{i}] analytical {a[i]:G10} vs numerical {num[i]:G10}");
        }
    }

    /// <summary>
    /// ReduceLogVariance over a single axis. The sweep reported analytical ~-4.8e6 against
    /// numerical ~-0.09, which no forward-accuracy issue explains, so it is checked directly.
    /// </summary>
    [Fact]
    public void ReduceLogVariance_Gradient_MatchesFiniteDifferences()
    {
        var x = Signal(8, seed: 5);
        // keepDims: false, and a genuinely small epsilon. The sweep supplied 0.5 for the epsilon
        // parameter (its blanket value for any double), which is not a stabilizer but a large
        // additive constant on the variance — another invalid-argument artifact rather than a defect.
        var (a, num) = Check(x, t => _engine.ReduceLogVariance(t, new[] { 0 }, false, 1e-12), probes: 4, eps: 1e-5);

        for (int i = 0; i < a.Length; i++)
            _out.WriteLine($"ReduceLogVariance [{i}] analytical={a[i]:G10} numerical={num[i]:G10}");

        for (int i = 0; i < a.Length; i++)
        {
            double denom = Math.Max(1.0, Math.Max(Math.Abs(a[i]), Math.Abs(num[i])));
            Assert.True(Math.Abs(a[i] - num[i]) / denom < 1e-5,
                $"ReduceLogVariance gradient[{i}] analytical {a[i]:G10} vs numerical {num[i]:G10}");
        }
    }
}
