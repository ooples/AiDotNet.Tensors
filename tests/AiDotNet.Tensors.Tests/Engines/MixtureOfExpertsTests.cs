using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The fused SwiGLU mixture-of-experts op (<see cref="MixtureOfExperts.SwiGluExperts"/>) on the host against an
/// independent per-token loop (no sort, no grouping), its per-sequence loads, the selection bias and tie rule, and every
/// gradient against central finite differences in double.
/// </summary>
public class MixtureOfExpertsTests
{
    internal static Tensor<T> Random<T>(int seed, double lo, double hi, params int[] shape)
    {
        var rng = new Random(seed);
        var t = new Tensor<T>(shape);
        var data = t.GetDataArray();
        for (int i = 0; i < data.Length; i++) data[i] = (T)Convert.ChangeType(lo + (hi - lo) * rng.NextDouble(), typeof(T));
        return t;
    }

    internal static double[] D<T>(Tensor<T> t) => t.ToArray().Select(v => Convert.ToDouble(v)).ToArray();

    /// <summary>The op written the obvious way: each token picks its experts and sums their gated outputs.</summary>
    internal static (double[] Y, double[] Loads) Naive(double[] x, double[] s, double[]? bias, double[] w1, double[] w3,
        double[] w2, int n, int d, int g, int f, int k, int b)
    {
        var y = new double[n * d];
        var loads = new double[b * g];
        for (int t = 0; t < n; t++)
        {
            var picked = Enumerable.Range(0, g)
                .OrderByDescending(e => s[t * g + e] + (bias?[e] ?? 0)).ThenBy(e => e).Take(k).ToArray();
            double sum = picked.Sum(e => s[t * g + e]);
            foreach (int e in picked)
            {
                loads[(t / (n / b)) * g + e]++;
                double gate = s[t * g + e] / sum;
                for (int c = 0; c < d; c++)
                {
                    double acc = 0;
                    for (int q = 0; q < f; q++)
                    {
                        double h = 0, u = 0;
                        for (int i = 0; i < d; i++)
                        {
                            h += x[t * d + i] * w1[(e * d + i) * f + q];
                            u += x[t * d + i] * w3[(e * d + i) * f + q];
                        }
                        acc += h / (1 + Math.Exp(-h)) * u * w2[(e * f + q) * d + c];
                    }
                    y[t * d + c] += gate * acc;
                }
            }
        }
        return (y, loads);
    }

    private static void AssertClose(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1, Math.Abs(expected[i])),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    [Theory]
    [InlineData(6, 4, 5, 3, 2, 1, false)]
    [InlineData(12, 7, 9, 5, 3, 3, true)]
    [InlineData(8, 3, 16, 2, 16, 2, false)]   // k = G: every expert, the largest top-k
    public void Forward_MatchesPerTokenLoop(int n, int d, int g, int f, int k, int b, bool withBias)
    {
        var e = new CpuEngine();
        var x = Random<float>(1, -1, 1, n, d);
        var s = Random<float>(2, 0.05, 0.95, n, g);
        var bias = withBias ? Random<float>(3, -0.5, 0.5, g) : null;
        var w1 = Random<float>(4, -0.5, 0.5, g, d, f);
        var w3 = Random<float>(5, -0.5, 0.5, g, d, f);
        var w2 = Random<float>(6, -0.5, 0.5, g, f, d);
        var y = MixtureOfExperts.SwiGluExperts(e, x, s, bias, w1, w3, w2, k, b, out var loads);
        var (ry, rl) = Naive(D(x), D(s), bias is null ? null : D(bias), D(w1), D(w3), D(w2), n, d, g, f, k, b);
        AssertClose(ry, D(y), 1e-5, "y");
        Assert.Equal(rl, D(loads));
        Assert.Equal(new[] { b, g }, loads.Shape.ToArray());
    }

    // The bias picks the experts; the gates still normalize the unbiased scores.
    [Fact]
    public void SelectionBias_PicksExperts_ButNotGates()
    {
        var e = new CpuEngine();
        var x = new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { 1.0 }));
        var s = new Tensor<double>(new[] { 1, 3 }, new Vector<double>(new[] { 0.9, 0.5, 0.2 }));
        var bias = new Tensor<double>(new[] { 3 }, new Vector<double>(new[] { 0.0, 0.0, 1.0 }));
        // Expert e computes silu(1) * 1 * (e + 1): w1 = w3 = 1, w2 = e + 1.
        var w1 = new Tensor<double>(new[] { 3, 1, 1 }, new Vector<double>(new[] { 1.0, 1, 1 }));
        var w2 = new Tensor<double>(new[] { 3, 1, 1 }, new Vector<double>(new[] { 1.0, 2, 3 }));
        var y = MixtureOfExperts.SwiGluExperts(e, x, s, bias, w1, w1, w2, 2, 1, out var loads);
        double silu = 1 / (1 + Math.Exp(-1));
        // Picked: expert 2 (1.2) and expert 0 (0.9); gates 0.2/1.1 and 0.9/1.1.
        Assert.Equal(silu * (0.2 / 1.1 * 3 + 0.9 / 1.1 * 1), y[0, 0], 12);
        Assert.Equal(new[] { 1.0, 0, 1 }, D(loads));
    }

    [Fact]
    public void Ties_GoToTheLowerExpert()
    {
        var e = new CpuEngine();
        var x = new Tensor<double>(new[] { 1, 1 }, new Vector<double>(new[] { 1.0 }));
        var s = new Tensor<double>(new[] { 1, 4 }, new Vector<double>(new[] { 0.5, 0.5, 0.5, 0.5 }));
        var w = new Tensor<double>(new[] { 4, 1, 1 }, new Vector<double>(new[] { 1.0, 1, 1, 1 }));
        MixtureOfExperts.SwiGluExperts(e, x, s, null, w, w, w, 2, 1, out var loads);
        Assert.Equal(new[] { 1.0, 1, 0, 0 }, D(loads));
    }

    [Fact]
    public void Gradients_MatchFiniteDifferences()
    {
        const int n = 6, d = 4, g = 5, f = 3, k = 2;
        var e = new CpuEngine();
        var x = Random<double>(11, -1, 1, n, d);
        var s = Random<double>(12, 0.05, 0.95, n, g);
        var bias = Random<double>(13, -0.2, 0.2, g);
        var w1 = Random<double>(14, -0.7, 0.7, g, d, f);
        var w3 = Random<double>(15, -0.7, 0.7, g, d, f);
        var w2 = Random<double>(16, -0.7, 0.7, g, f, d);
        var wy = Random<double>(17, -1, 1, n, d);
        var inputs = new[] { x, s, w1, w3, w2 };

        double Loss()
        {
            var y = MixtureOfExperts.SwiGluExperts(e, x, s, bias, w1, w3, w2, k, 1, out _);
            return y.ToArray().Zip(wy.ToArray(), (a, b) => a * b).Sum();
        }

        System.Collections.Generic.Dictionary<Tensor<double>, Tensor<double>> grads;
        using (var tape = new GradientTape<double>())
        {
            var y = MixtureOfExperts.SwiGluExperts(e, x, s, bias, w1, w3, w2, k, 1, out _);
            var loss = e.ReduceSum(e.TensorMultiply(y, wy), null);
            grads = tape.ComputeGradients(loss, inputs);
        }

        string[] names = { "x", "scores", "w1", "w3", "w2" };
        for (int which = 0; which < inputs.Length; which++)
        {
            var data = inputs[which].GetDataArray();
            var analytic = grads[inputs[which]].ToArray();
            for (int i = 0; i < data.Length; i++)
            {
                double keep = data[i];
                const double h = 1e-6;
                data[i] = keep + h; double up = Loss();
                data[i] = keep - h; double down = Loss();
                data[i] = keep;
                double numeric = (up - down) / (2 * h);
                Assert.True(Math.Abs(numeric - analytic[i]) <= 1e-6 * Math.Max(1, Math.Abs(numeric)),
                    $"d{names[which]}[{i}]: finite difference {numeric:R}, analytic {analytic[i]:R}");
            }
        }
    }

    [Fact]
    public void Rejects_NullTensors()
    {
        var e = new CpuEngine();
        var x = new Tensor<float>(new[] { 4, 3 });
        var s = new Tensor<float>(new[] { 4, 5 });
        var w = new Tensor<float>(new[] { 5, 3, 2 });
        var w2 = new Tensor<float>(new[] { 5, 2, 3 });
        Assert.Equal("engine", Assert.Throws<ArgumentNullException>(() => MixtureOfExperts.SwiGluExperts(null!, x, s, null, w, w, w2, 2, 1, out _)).ParamName);
        Assert.Equal("x", Assert.Throws<ArgumentNullException>(() => MixtureOfExperts.SwiGluExperts(e, null!, s, null, w, w, w2, 2, 1, out _)).ParamName);
        Assert.Equal("scores", Assert.Throws<ArgumentNullException>(() => MixtureOfExperts.SwiGluExperts(e, x, null!, null, w, w, w2, 2, 1, out _)).ParamName);
        Assert.Equal("w1", Assert.Throws<ArgumentNullException>(() => MixtureOfExperts.SwiGluExperts(e, x, s, null, null!, w, w2, 2, 1, out _)).ParamName);
        Assert.Equal("w3", Assert.Throws<ArgumentNullException>(() => MixtureOfExperts.SwiGluExperts(e, x, s, null, w, null!, w2, 2, 1, out _)).ParamName);
        Assert.Equal("w2", Assert.Throws<ArgumentNullException>(() => MixtureOfExperts.SwiGluExperts(e, x, s, null, w, w, null!, 2, 1, out _)).ParamName);
    }

    [SkippableFact]
    public void CpuBoundTape_DeviceMoeForward_DispatchesGradientsToCompatibleGpuEngine()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.IsGpuAvailable, "No DirectGpu backend available.");
        Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.IMoeKernelBackend moe && moe.MoeKernelsReady,
            "Fused MoE kernels are not available on this DirectGpu backend.");

        const int n = 8, d = 6, g = 4, f = 5, k = 2;
        var cpu = new CpuEngine();
        var x = Random<float>(21, -1, 1, n, d);
        var s = Random<float>(22, 0.05, 0.95, n, g);
        var bias = Random<float>(23, -0.2, 0.2, g);
        var w1 = Random<float>(24, -0.5, 0.5, g, d, f);
        var w3 = Random<float>(25, -0.5, 0.5, g, d, f);
        var w2 = Random<float>(26, -0.5, 0.5, g, f, d);
        var wy = Random<float>(27, -1, 1, n, d);
        var inputs = new[] { x, s, w1, w3, w2 };

        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> refGrads;
        using (var refTape = new GradientTape<float>())
        {
            var refY = MixtureOfExperts.SwiGluExperts(cpu, x, s, bias, w1, w3, w2, k, 2, out _);
            var refLoss = cpu.ReduceSum(cpu.TensorMultiply(refY, wy), null);
            refGrads = refTape.ComputeGradients(refLoss, inputs);
        }

        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gpuGrads;
        using (var cpuBoundTape = new GradientTape<float>())
        {
            cpuBoundTape.BindEngineIfUnset(cpu);
            var gpuY = MixtureOfExperts.SwiGluExperts(gpu, x, s, bias, w1, w3, w2, k, 2, out _);
            Assert.True(gpuY.IsGpuResident || gpuY.HasPendingGpuData,
                "SwiGluExperts fell back to the host instead of executing on the DirectGpu backend.");
            var gpuLoss = cpu.ReduceSum(cpu.TensorMultiply(gpuY, wy), null);
            gpuGrads = cpuBoundTape.ComputeGradients(gpuLoss, inputs);
        }

        foreach (var t in inputs)
            AssertClose(D(refGrads[t]), D(gpuGrads[t]), 1e-4, "grad");
    }

    [Fact]
    public void Rejects_InconsistentShapes()
    {
        var e = new CpuEngine();
        var x = new Tensor<float>(new[] { 4, 3 });
        var s = new Tensor<float>(new[] { 4, 5 });
        var w = new Tensor<float>(new[] { 5, 3, 2 });
        var w2 = new Tensor<float>(new[] { 5, 2, 3 });
        Assert.Throws<ArgumentException>(() => MixtureOfExperts.SwiGluExperts(e, x, new Tensor<float>(new[] { 4, 6 }), null, w, w, w2, 2, 1, out _));
        Assert.Throws<ArgumentException>(() => MixtureOfExperts.SwiGluExperts(e, x, s, null, w, w, w, 2, 1, out _));
        Assert.Throws<ArgumentException>(() => MixtureOfExperts.SwiGluExperts(e, x, s, new Tensor<float>(new[] { 4 }), w, w, w2, 2, 1, out _));
        Assert.Throws<ArgumentOutOfRangeException>(() => MixtureOfExperts.SwiGluExperts(e, x, s, null, w, w, w2, 6, 1, out _));
        Assert.Throws<ArgumentOutOfRangeException>(() => MixtureOfExperts.SwiGluExperts(e, x, s, null, w, w, w2, 2, 3, out _));
    }
}
