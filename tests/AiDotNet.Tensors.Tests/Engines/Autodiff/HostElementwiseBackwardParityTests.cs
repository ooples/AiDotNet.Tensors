using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

// The host fast paths of the Clamp, Log, Sqrt, Abs and Exp backwards must give the gradient the engine-op composition
// they replace gives, bit for bit, including at zeros, clamp bounds and (for Clamp) a non-finite input, which keeps the
// engine path.
public sealed class HostElementwiseBackwardParityTests
{
    private static Tensor<float> Values(int n, int seed, bool positive)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(new[] { 4, n / 4 });
        for (int i = 0; i < t.Length; i++)
        {
            float v = (float)(rng.NextDouble() * 4 - 2);
            t[i] = positive ? Math.Abs(v) + 0.25f : v;
        }
        if (!positive) { t[0] = 0f; t[1] = 0.5f; t[2] = -0.5f; }
        return t;
    }

    private static float[] RunBackward(string op, Tensor<float> x, Tensor<float> y, Tensor<float> g, object[] saved)
    {
        var engine = new CpuEngine();
        var grads = new Dictionary<Tensor<float>, Tensor<float>>(AiDotNet.Tensors.Engines.Autodiff.ReferenceEqualityComparer<Tensor<float>>.Instance);
        var inputs = new[] { x };
        switch (op)
        {
            case "exp": BackwardFunctions<float>.ExpBackward(g, inputs, y, saved, engine, grads); break;
            case "log": BackwardFunctions<float>.LogBackward(g, inputs, y, saved, engine, grads); break;
            case "sqrt": BackwardFunctions<float>.SqrtBackward(g, inputs, y, saved, engine, grads); break;
            case "abs": BackwardFunctions<float>.AbsBackward(g, inputs, y, saved, engine, grads); break;
            default: BackwardFunctions<float>.ClampBackward(g, inputs, y, saved, engine, grads); break;
        }
        return grads[x].ToArray();
    }

    // The engine composition each backward used before its host fast path.
    private static float[] EngineReference(string op, Tensor<float> x, Tensor<float> y, Tensor<float> g)
    {
        var e = new CpuEngine();
        switch (op)
        {
            case "exp": return e.TensorMultiply(g, y).ToArray();
            case "log": return e.TensorDivide(g, x).ToArray();
            case "sqrt": return e.TensorDivide(g, e.TensorMultiplyScalar(y, 2f)).ToArray();
            case "abs": return e.TensorMultiply(g, e.TensorSign(x)).ToArray();
            default:
                var clamped = e.TensorClamp(x, -0.5f, 0.5f);
                var absDiff = e.TensorAbs(e.TensorSubtract(clamped, x));
                var ones = e.TensorAddScalar(e.TensorMultiplyScalar(absDiff, 0f), 1f);
                var mask = e.TensorSubtract(ones, e.TensorSign(absDiff));
                return e.TensorMultiply(g, mask).ToArray();
        }
    }

    [Theory]
    [InlineData("exp")]
    [InlineData("log")]
    [InlineData("sqrt")]
    [InlineData("abs")]
    [InlineData("clamp")]
    public void HostFastPath_MatchesEngineComposition(string op)
    {
        var e = new CpuEngine();
        bool positive = op is "log" or "sqrt";
        var x = Values(4096, 7, positive);
        var g = Values(4096, 11, positive: false);
        var y = op switch
        {
            "exp" => e.TensorExp(x),
            "log" => e.TensorLog(x),
            "sqrt" => e.TensorSqrt(x),
            "abs" => e.TensorAbs(x),
            _ => e.TensorClamp(x, -0.5f, 0.5f),
        };
        var saved = op == "clamp" ? new object[] { -0.5, 0.5 } : Array.Empty<object>();
        Assert.Equal(EngineReference(op, x, y, g), RunBackward(op, x, y, g, saved));
    }

    [Fact]
    public void Clamp_NonFiniteInput_KeepsEnginePath()
    {
        var e = new CpuEngine();
        var x = Values(4096, 3, positive: false);
        x[5] = float.PositiveInfinity;
        x[6] = float.NaN;
        var g = Values(4096, 4, positive: false);
        var y = e.TensorClamp(x, -0.5f, 0.5f);
        var expected = EngineReference("clamp", x, y, g);
        var actual = RunBackward("clamp", x, y, g, new object[] { -0.5, 0.5 });
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
    }
}
