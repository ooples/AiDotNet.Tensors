using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The CPU engine's generic-T fallbacks for Half. They read each element through <c>Convert.ToDouble</c>, which throws
/// <see cref="InvalidCastException"/> for Half (not <c>IConvertible</c>); AiDotNet's Half randomized-smoothing tests
/// failed inside <see cref="CpuEngine.GenerateGaussianNoise{T}"/> this way.
/// </summary>
public class CpuEngineHalfFallbackTests
{
    [Fact]
    public void GenerateGaussianNoise_Half_IsFiniteWithTheRequestedMoments()
    {
        var engine = new CpuEngine();
        const int n = 20000;
        var noise = engine.GenerateGaussianNoise(n, (Half)2.0, (Half)0.5, 11);
        double[] values = Enumerable.Range(0, n).Select(i => (double)noise[i]).ToArray();
        Assert.All(values, v => Assert.False(double.IsNaN(v) || double.IsInfinity(v), $"sample {v} is not finite"));
        double mean = values.Average();
        double std = Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / (n - 1));
        // Standard error of the mean is 0.5 / sqrt(20000) = 0.0035; of the standard deviation about 0.0025.
        Assert.InRange(mean, 2.0 - 0.02, 2.0 + 0.02);
        Assert.InRange(std, 0.5 - 0.02, 0.5 + 0.02);
    }

    [Fact]
    public void GenerateGaussianNoise_Double_SameSeedSameSamples()
    {
        var engine = new CpuEngine();
        var a = engine.GenerateGaussianNoise(64, 0.0, 1.0, 7);
        var b = engine.GenerateGaussianNoise(64, 0.0, 1.0, 7);
        Assert.Equal(a.ToArray(), b.ToArray());
    }

    [Fact]
    public void ElementwiseFallbacks_Half_MatchTheDoubleFunctions()
    {
        var engine = new CpuEngine();
        double[] inputs = { -0.75, -0.25, 0.0, 0.3, 0.6 };
        var vector = new Vector<Half>(inputs.Select(v => (Half)v).ToArray());
        (Func<Vector<Half>, Vector<Half>> Op, Func<double, double> Reference, string Name)[] cases =
        {
            (engine.Sinh, Math.Sinh, "Sinh"), (engine.Cosh, Math.Cosh, "Cosh"),
            (engine.Asin, Math.Asin, "Asin"), (engine.Atan, Math.Atan, "Atan"),
            (engine.Exp2, x => Math.Pow(2.0, x), "Exp2"), (engine.Floor, Math.Floor, "Floor"),
        };
        foreach (var (op, reference, name) in cases)
        {
            var result = op(vector);
            for (int i = 0; i < inputs.Length; i++)
            {
                double expected = (double)(Half)reference((double)(Half)inputs[i]);
                Assert.True((double)result[i] == expected, $"{name}({inputs[i]}) = {(double)result[i]}, expected {expected}");
            }
        }
    }
}
