using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// The tensor-level acosh, asinh, exp2, expm1, log10, log1p, log2, rsqrt, tan and trunc: values against the double
/// reference, precision where the naive formula loses it, and the recorded gradient against a central difference.
/// </summary>
public class ElementwiseExtraOpsTests
{
    private readonly IEngine _engine = new CpuEngine();

    public static TheoryData<string> Ops => new() { "acosh", "asinh", "exp2", "expm1", "log10", "log1p", "log2", "rsqrt", "tan", "trunc" };

    private Tensor<double> Apply(string op, Tensor<double> x) => op switch
    {
        "acosh" => _engine.TensorAcosh(x), "asinh" => _engine.TensorAsinh(x), "exp2" => _engine.TensorExp2(x),
        "expm1" => _engine.TensorExpm1(x), "log10" => _engine.TensorLog10(x), "log1p" => _engine.TensorLog1p(x),
        "log2" => _engine.TensorLog2(x), "rsqrt" => _engine.TensorRsqrt(x), "tan" => _engine.TensorTan(x),
        "trunc" => _engine.TensorTrunc(x), _ => throw new ArgumentException(op),
    };

    private static double Reference(string op, double v) => op switch
    {
        "acosh" => Math.Log(v + Math.Sqrt(v * v - 1)), "asinh" => Math.Log(v + Math.Sqrt(v * v + 1)),
        "exp2" => Math.Pow(2, v), "expm1" => Math.Exp(v) - 1, "log10" => Math.Log10(v), "log1p" => Math.Log(1 + v),
        "log2" => Math.Log(v) / Math.Log(2), "rsqrt" => 1 / Math.Sqrt(v), "tan" => Math.Tan(v), "trunc" => Math.Truncate(v),
        _ => throw new ArgumentException(op),
    };

    // Inside every op's domain (acosh needs >= 1) and away from tan's poles and trunc's jumps.
    private static double[] Inputs(string op) => op == "acosh"
        ? new[] { 1.25, 1.5, 2.0, 3.7 }
        : new[] { 0.15, 0.4, 0.9, 1.3 };

    [Theory]
    [MemberData(nameof(Ops))]
    public void Values_MatchTheDoubleReference(string op)
    {
        var xs = Inputs(op);
        var y = Apply(op, new Tensor<double>((double[])xs.Clone(), new[] { xs.Length })).ToArray();
        for (int i = 0; i < xs.Length; i++) Assert.Equal(Reference(op, xs[i]), y[i], 12);
    }

    [Fact]
    public void Log1pAndExpm1_KeepTheDigitsOfATinyArgument()
    {
        // 1 + 1e-12 rounds away most of 1e-12's digits; log1p / expm1 must not.
        var x = new Tensor<double>(new[] { 1e-12, -3e-15 }, new[] { 2 });
        var l = _engine.TensorLog1p(x).ToArray();
        var e = _engine.TensorExpm1(x).ToArray();
        Assert.Equal(1e-12 - 0.5e-24, l[0], 25);
        Assert.Equal(1e-12 + 0.5e-24, e[0], 25);
        Assert.Equal(-3e-15, l[1], 28);
        Assert.Equal(-3e-15, e[1], 28);
    }

    [Theory]
    [MemberData(nameof(Ops))]
    public void Gradient_MatchesACentralDifference(string op)
    {
        var xs = Inputs(op);
        var x = new Tensor<double>((double[])xs.Clone(), new[] { xs.Length });
        double[] g;
        using (var tape = new GradientTape<double>())
        {
            var loss = _engine.ReduceSum(Apply(op, x), null);
            g = tape.ComputeGradients(loss, new[] { x })[x].ToArray();
        }
        const double h = 1e-6;
        for (int i = 0; i < xs.Length; i++)
        {
            double numeric = op == "trunc" ? 0 : (Reference(op, xs[i] + h) - Reference(op, xs[i] - h)) / (2 * h);
            Assert.Equal(numeric, g[i], 6);
        }
    }
}
