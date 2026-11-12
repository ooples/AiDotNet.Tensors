using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Backward of the mean-reduction losses and average pools over 70,000+ Half elements. Their gradient carries a
/// 1/N (or 2/N) factor; in Half that factor is subnormal (below 6.1e-5) or, with the count itself converted, zero,
/// so scaling in Half rounds twice and loses up to several units in the last place. PyTorch (2.x) evaluates these
/// gradients in float32 opmath and rounds once; every element must be within one Half ulp (subnormal spacing 2^-24)
/// of the double reference computed from the same Half inputs. The upstream gradient is 7 so g/N is not a power-of-two
/// multiple of a single rounding.
/// </summary>
public class LowPrecisionBackwardScaleTests
{
    private const int N = 70_000;
    private const double G = 7.0;

    private readonly CpuEngine _engine = new CpuEngine();

    private static Half[] HalfData(int n, Func<int, double> f)
    {
        var data = new Half[n];
        for (int i = 0; i < n; i++) data[i] = (Half)f(i);
        return data;
    }

    private static Tensor<Half> T1(Half[] data, params int[] shape) => new Tensor<Half>(data, shape);

    private static Tensor<Half> Upstream(params int[] shape)
    {
        int n = 1;
        foreach (int d in shape) n *= d;
        return new Tensor<Half>(HalfData(n, _ => G), shape);
    }

    private static void AssertAllWithinUlp(Func<int, double> expected, Half[] actual, string what)
    {
        for (int i = 0; i < actual.Length; i++)
        {
            double e = expected(i), a = (double)actual[i];
            double ulp = e == 0 ? Math.Pow(2, -24) : Math.Max(Math.Pow(2, -24), Math.Pow(2, Math.Floor(Math.Log(Math.Abs(e), 2)) - 10));
            if (double.IsNaN(a) || Math.Abs(a - e) > ulp)
                Assert.Fail($"{what}[{i}]: got {a}, expected {e} (one ulp = {ulp})");
        }
    }

    private Half[] Run(BackwardFunction<Half> backward, Tensor<Half> upstream, Tensor<Half>[] inputs, object[] state)
    {
        var grads = new Dictionary<Tensor<Half>, Tensor<Half>>();
        backward(upstream, inputs, upstream, state, _engine, grads);
        return grads[inputs[0]].ToArray();
    }

    private static readonly Half[] P = HalfData(N, i => 1.0 + 0.5 * Math.Sin(0.37 * i));
    private static readonly Half[] Tg = HalfData(N, i => 0.5 + 0.25 * Math.Cos(0.11 * i));

    [Fact]
    public void MseLossBackward_Half_Over70000Elements()
    {
        var g = Run(BackwardFunctions<Half>.MSELossBackward, Upstream(1), new[] { T1(P, N), T1(Tg, N) }, Array.Empty<object>());
        AssertAllWithinUlp(i => 2.0 * ((double)P[i] - (double)Tg[i]) * G / N, g, "MSELossBackward");
    }

    [Fact]
    public void L1AndHuberLossBackward_Half_Over70000Elements()
    {
        var l1 = Run(BackwardFunctions<Half>.L1LossBackward, Upstream(1), new[] { T1(P, N), T1(Tg, N) }, Array.Empty<object>());
        AssertAllWithinUlp(i => Math.Sign((double)P[i] - (double)Tg[i]) * G / N, l1, "L1LossBackward");
        var huber = Run(BackwardFunctions<Half>.HuberLossBackward, Upstream(1), new[] { T1(P, N), T1(Tg, N) }, new object[] { 0.5 });
        AssertAllWithinUlp(i => Math.Max(-0.5, Math.Min(0.5, (double)P[i] - (double)Tg[i])) * G / N, huber, "HuberLossBackward");
    }

    [Fact]
    public void BceWithLogitsAndKlDivBackward_Half_Over70000Elements()
    {
        var bce = Run(BackwardFunctions<Half>.BCEWithLogitsLossBackward, Upstream(1), new[] { T1(P, N), T1(Tg, N) }, Array.Empty<object>());
        AssertAllWithinUlp(i => (1.0 / (1.0 + Math.Exp(-(double)P[i])) - (double)Tg[i]) * G / N, bce, "BCEWithLogitsLossBackward");
        var kl = Run(BackwardFunctions<Half>.KLDivLossBackward, Upstream(1), new[] { T1(P, N), T1(Tg, N) }, Array.Empty<object>());
        AssertAllWithinUlp(i => -(double)Tg[i] * G / N, kl, "KLDivLossBackward");
    }

    [Fact]
    public void MeanBackward_Half_Over70000Elements()
    {
        var g = Run(BackwardFunctions<Half>.MeanBackward, Upstream(1), new[] { T1(P, N) }, Array.Empty<object>());
        AssertAllWithinUlp(_ => G / N, g, "MeanBackward");
    }

    [Fact]
    public void AvgPool1DForwardAndBackward_Half_KernelOf70000()
    {
        var input = T1(P, 1, 1, N);
        double sum = 0;
        for (int i = 0; i < N; i++) sum += (double)P[i];
        var pooled = _engine.TensorAvgPool1D(input, N, N).ToArray();
        AssertAllWithinUlp(_ => sum / N, pooled, "TensorAvgPool1D");

        var g = Run(BackwardFunctions<Half>.AvgPool1DBackward, Upstream(1, 1, 1), new[] { input }, new object[] { N, N });
        AssertAllWithinUlp(_ => G / N, g, "AvgPool1DBackward");
    }

    [Fact]
    public void AdaptiveAvgPool2DBackward_Half_WindowOf75000()
    {
        const int h = 300, w = 250;
        var input = T1(HalfData(h * w, i => 1.0 + 0.5 * Math.Sin(0.37 * i)), 1, 1, h, w);
        var g = Run(BackwardFunctions<Half>.AdaptiveAvgPool2DBackward, Upstream(1, 1, 1, 1), new[] { input }, new object[] { 1, 1 });
        AssertAllWithinUlp(_ => G / (h * w), g, "AdaptiveAvgPool2DBackward");
    }
}
