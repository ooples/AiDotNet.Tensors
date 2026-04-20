using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

public class Parity210SpecialMathTests
{
    private static CpuEngine E => new CpuEngine();
    private static Tensor<float> T(float[] data, params int[] shape) => new Tensor<float>(data, shape);
    private static bool Close(float a, float b, float eps = 1e-3f) => MathF.Abs(a - b) <= eps * (1 + MathF.Abs(a) + MathF.Abs(b));

    [Fact]
    public void Lgamma_AtIntegerMatchesFactorial()
    {
        // Γ(n) = (n-1)! → log Γ(n) = log((n-1)!)
        var x = T(new[] { 1f, 2f, 3f, 4f, 5f }, 5);
        var r = E.TensorLgamma(x);
        Assert.True(Close(0f, r[0], 1e-4f));        // log(0!) = 0
        Assert.True(Close(0f, r[1], 1e-4f));        // log(1!) = 0
        Assert.True(Close(MathF.Log(2f), r[2], 1e-3f));  // log(2!) = log 2
        Assert.True(Close(MathF.Log(6f), r[3], 1e-3f));  // log(3!) = log 6
        Assert.True(Close(MathF.Log(24f), r[4], 1e-3f)); // log(4!) = log 24
    }

    [Fact]
    public void Digamma_AtInteger_MatchesHarmonicSeries()
    {
        // ψ(n) = -γ + Σ_{k=1..n-1} 1/k; γ ≈ 0.5772156649
        float gamma = 0.5772156649f;
        // ψ(1) = -γ
        // ψ(2) = -γ + 1
        // ψ(3) = -γ + 1 + 1/2
        var x = T(new[] { 1f, 2f, 3f, 4f }, 4);
        var r = E.TensorDigamma(x);
        Assert.True(Close(-gamma, r[0], 1e-3f), $"ψ(1) expected ~{-gamma}, got {r[0]}");
        Assert.True(Close(-gamma + 1f, r[1], 1e-3f), $"ψ(2) expected ~{-gamma + 1}, got {r[1]}");
        Assert.True(Close(-gamma + 1.5f, r[2], 1e-3f), $"ψ(3) expected ~{-gamma + 1.5f}, got {r[2]}");
        Assert.True(Close(-gamma + 1f + 0.5f + 1f/3f, r[3], 1e-3f), $"ψ(4) got {r[3]}");
    }

    [Fact]
    public void Lgamma_LargeX_AgreesWithDirectLog()
    {
        // For x = 10, Γ(10) = 362880; log should be ~log(362880) ≈ 12.80.
        var x = T(new[] { 10f }, 1);
        var r = E.TensorLgamma(x);
        Assert.True(Close(MathF.Log(362880f), r[0], 5e-3f), $"got {r[0]}");
    }
}
