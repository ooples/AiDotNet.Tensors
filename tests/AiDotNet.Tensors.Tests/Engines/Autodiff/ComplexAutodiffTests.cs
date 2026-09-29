// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.Engines.DirectGpu;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// Native complex ops recorded nothing on a tape (and a null backward in a compiled graph), so no gradient flowed
/// through a complex computation. The convention is PyTorch's: for a real loss L of z = x + i·y the tape gradient is
/// ∂L/∂x + i·∂L/∂y. Seeding an op's output with g corresponds to L = Re(Σ conj(g)·f(z)), whose gradient is checked
/// against central finite differences in double precision.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class ComplexAutodiffTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public ComplexAutodiffTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<Complex<double>> RandC(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<Complex<double>>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = new Complex<double>(rng.NextDouble() - 0.5, rng.NextDouble() - 0.5);
        return t;
    }

    private static readonly int[] Shape = { 2, 8 };   // two FFT rows of length 8

    public static IEnumerable<object[]> Ops() =>
        new[] { "add", "scale", "conj", "multiply", "cross", "fft", "ifft", "chain" }.Select(o => new object[] { o });

    private static Tensor<Complex<double>> Apply(string op, IEngine e, Tensor<Complex<double>> a, Tensor<Complex<double>> b) => op switch
    {
        "add" => e.NativeComplexAdd(a, b),
        "scale" => e.NativeComplexScale(a, 0.7),
        "conj" => e.NativeComplexConjugate(a),
        "multiply" => e.NativeComplexMultiply(a, b),
        "cross" => e.NativeComplexCrossSpectral(a, b),
        "fft" => e.NativeComplexFFTComplex(a),
        "ifft" => e.NativeComplexIFFT(a),
        // conj(IFFT(FFT(a)·b)) + a: several ops, a shared operand, holomorphic and anti-holomorphic parts
        "chain" => e.NativeComplexAdd(e.NativeComplexConjugate(e.NativeComplexIFFT(e.NativeComplexMultiply(e.NativeComplexFFTComplex(a), b))), a),
        _ => throw new ArgumentException(op),
    };

    private static double Loss(string op, IEngine e, Tensor<Complex<double>> a, Tensor<Complex<double>> b, Tensor<Complex<double>> g)
    {
        var y = Apply(op, e, a, b);
        double l = 0;
        for (int i = 0; i < y.Length; i++) l += g[i].Real * y[i].Real + g[i].Imaginary * y[i].Imaginary;   // Re(conj(g)·y)
        return l;
    }

    [Theory]
    [MemberData(nameof(Ops))]
    public void Gradient_MatchesFiniteDifferences(string op)
    {
        IEngine e = new CpuEngine();
        var a = RandC(Shape, 1);
        var b = RandC(Shape, 2);
        var g = RandC(Shape, 3);

        Dictionary<Tensor<Complex<double>>, Tensor<Complex<double>>> grads;
        using (var tape = new GradientTape<Complex<double>>())
        {
            var y = Apply(op, e, a, b);
            grads = tape.ComputeGradients(y, new[] { a, b }, createGraph: false,
                seedOverride: new[] { new KeyValuePair<Tensor<Complex<double>>, Tensor<Complex<double>>>(y, g) });
        }

        const double h = 1e-6;
        foreach (var (input, label) in new[] { (a, "a"), (b, "b") })
        {
            bool used = op is "add" or "multiply" or "cross" or "chain" || label == "a";
            if (!used) continue;
            Assert.True(grads.TryGetValue(input, out var grad), $"{op}: no gradient for {label}");
            for (int i = 0; i < input.Length; i++)
            {
                var orig = input[i];
                input[i] = new Complex<double>(orig.Real + h, orig.Imaginary); double lp = Loss(op, e, a, b, g);
                input[i] = new Complex<double>(orig.Real - h, orig.Imaginary); double lm = Loss(op, e, a, b, g);
                double dRe = (lp - lm) / (2 * h);
                input[i] = new Complex<double>(orig.Real, orig.Imaginary + h); lp = Loss(op, e, a, b, g);
                input[i] = new Complex<double>(orig.Real, orig.Imaginary - h); lm = Loss(op, e, a, b, g);
                double dIm = (lp - lm) / (2 * h);
                input[i] = orig;
                Assert.True(Math.Abs(grad![i].Real - dRe) < 1e-6 && Math.Abs(grad[i].Imaginary - dIm) < 1e-6,
                    $"{op} d/d{label}[{i}]: tape ({grad[i].Real:G6}, {grad[i].Imaginary:G6}) vs finite differences ({dRe:G6}, {dIm:G6})");
            }
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Ops))]
    public void GpuEngine_RecordsTheSameGradients(string op)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        Dictionary<string, Complex<float>[]> Run(IEngine e)
        {
            var a = ToFloat(RandC(Shape, 1));
            var b = ToFloat(RandC(Shape, 2));
            var g = ToFloat(RandC(Shape, 3));
            using var tape = new GradientTape<Complex<float>>();
            var y = ApplyF(op, e, a, b);
            var grads = tape.ComputeGradients(y, new[] { a, b }, createGraph: false,
                seedOverride: new[] { new KeyValuePair<Tensor<Complex<float>>, Tensor<Complex<float>>>(y, g) });
            var r = new Dictionary<string, Complex<float>[]>();
            if (grads.TryGetValue(a, out var ga)) r["a"] = ga.ToArray();
            if (grads.TryGetValue(b, out var gb)) r["b"] = gb.ToArray();
            return r;
        }
        var cpu = Run(new CpuEngine());
        var gpu = Run(_fixture.Engine!);
        Assert.Equal(cpu.Keys.OrderBy(k => k), gpu.Keys.OrderBy(k => k));
        foreach (var k in cpu.Keys)
            for (int i = 0; i < cpu[k].Length; i++)
                Assert.True(Math.Abs(cpu[k][i].Real - gpu[k][i].Real) < 1e-4 && Math.Abs(cpu[k][i].Imaginary - gpu[k][i].Imaginary) < 1e-4,
                    $"{op} d/d{k}[{i}]: cpu {cpu[k][i].Real},{cpu[k][i].Imaginary} gpu {gpu[k][i].Real},{gpu[k][i].Imaginary}");
    }

    private static Tensor<Complex<float>> ToFloat(Tensor<Complex<double>> t)
    {
        var r = new Tensor<Complex<float>>(t.Shape.ToArray());
        for (int i = 0; i < t.Length; i++) r[i] = new Complex<float>((float)t[i].Real, (float)t[i].Imaginary);
        return r;
    }

    private static Tensor<Complex<float>> ApplyF(string op, IEngine e, Tensor<Complex<float>> a, Tensor<Complex<float>> b) => op switch
    {
        "add" => e.NativeComplexAdd(a, b),
        "scale" => e.NativeComplexScale(a, 0.7f),
        "conj" => e.NativeComplexConjugate(a),
        "multiply" => e.NativeComplexMultiply(a, b),
        "cross" => e.NativeComplexCrossSpectral(a, b),
        "fft" => e.NativeComplexFFTComplex(a),
        "ifft" => e.NativeComplexIFFT(a),
        "chain" => e.NativeComplexAdd(e.NativeComplexConjugate(e.NativeComplexIFFT(e.NativeComplexMultiply(e.NativeComplexFFTComplex(a), b))), a),
        _ => throw new ArgumentException(op),
    };
}
