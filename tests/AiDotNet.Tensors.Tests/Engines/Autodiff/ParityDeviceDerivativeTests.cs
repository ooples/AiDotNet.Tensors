using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Autodiff;

/// <summary>
/// The GPU engine differentiates the torch-parity element-wise ops with engine-op formulas (so a taped step stays on the
/// device) where the CPU engine uses host derivative lambdas. Both run here on the CPU engine: each formula must give the
/// CPU engine's own tape gradient, at points that include each formula's special cases (x = 0, |x| small, outside a clamp).
/// </summary>
public class ParityDeviceDerivativeTests
{
    private static readonly float[] Signed = { -2.3f, -1.1f, -0.4f, -1e-5f, 0f, 3e-5f, 0.25f, 0.9f, 1.7f, 2.6f };
    private static readonly float[] Positive = { 0.05f, 0.3f, 0.8f, 1.2f, 2.1f, 3.4f };
    private static readonly float[] Unit = { 0.02f, 0.05f, 0.2f, 0.5f, 0.71f, 0.93f, 0.97f };

    public static IEnumerable<object[]> UnaryCases() => new (TorchParityUnaryOp Op, float P0, float P1, float[] X)[]
    {
        (TorchParityUnaryOp.Celu, 1.5f, 0f, Signed),
        (TorchParityUnaryOp.Hardtanh, -0.5f, 0.7f, Signed),
        (TorchParityUnaryOp.LogSigmoid, 0f, 0f, Signed),
        (TorchParityUnaryOp.Softsign, 0f, 0f, Signed),
        (TorchParityUnaryOp.Erf, 0f, 0f, Signed),
        (TorchParityUnaryOp.Logit, 0.1f, 0.9f, Unit),
        (TorchParityUnaryOp.Logit, float.NegativeInfinity, float.PositiveInfinity, Unit),
        (TorchParityUnaryOp.Sinc, 0f, 0f, Signed),
        (TorchParityUnaryOp.Mvlgamma, 3f, 0f, new[] { 1.2f, 1.7f, 2.5f, 4.1f }),
        (TorchParityUnaryOp.Entr, 0f, 0f, Positive),
        (TorchParityUnaryOp.Erfcx, 0f, 0f, Signed),
        (TorchParityUnaryOp.Ndtr, 0f, 0f, Signed),
        (TorchParityUnaryOp.LogNdtr, 0f, 0f, Signed),
        (TorchParityUnaryOp.Ndtri, 0f, 0f, Unit),
        (TorchParityUnaryOp.BesselJ0, 0f, 0f, Signed),
        (TorchParityUnaryOp.BesselJ1, 0f, 0f, Signed),
        (TorchParityUnaryOp.BesselY0, 0f, 0f, Positive),
        (TorchParityUnaryOp.BesselY1, 0f, 0f, Positive),
        (TorchParityUnaryOp.ModifiedBesselI0, 0f, 0f, Signed),
        (TorchParityUnaryOp.ModifiedBesselI1, 0f, 0f, Signed),
        (TorchParityUnaryOp.ModifiedBesselK0, 0f, 0f, Positive),
        (TorchParityUnaryOp.ModifiedBesselK1, 0f, 0f, Positive),
        (TorchParityUnaryOp.ScaledModifiedBesselK0, 0f, 0f, Positive),
        (TorchParityUnaryOp.ScaledModifiedBesselK1, 0f, 0f, Positive),
        (TorchParityUnaryOp.SphericalBesselJ0, 0f, 0f, Signed),
    }.Select(c => new object[] { c.Op.ToString(), c.P0, c.P1, c.X });

    private static Tensor<float> Forward(IEngine e, TorchParityUnaryOp op, Tensor<float> x, float p0, float p1) => op switch
    {
        TorchParityUnaryOp.Celu => e.TensorCelu(x, p0),
        TorchParityUnaryOp.Hardtanh => e.TensorHardtanh(x, p0, p1),
        TorchParityUnaryOp.LogSigmoid => e.TensorLogSigmoid(x),
        TorchParityUnaryOp.Softsign => e.TensorSoftsign(x),
        TorchParityUnaryOp.Erf => e.TensorErf(x),
        TorchParityUnaryOp.Logit => e.TensorLogit(x, float.IsInfinity(p0) ? null : (double?)(double)(decimal)p0),
        TorchParityUnaryOp.Sinc => e.TensorSinc(x),
        TorchParityUnaryOp.Mvlgamma => e.TensorMvlgamma(x, (int)p0),
        TorchParityUnaryOp.Entr => e.TensorEntr(x),
        TorchParityUnaryOp.Erfcx => e.TensorErfcx(x),
        TorchParityUnaryOp.Ndtr => e.TensorNdtr(x),
        TorchParityUnaryOp.LogNdtr => e.TensorLogNdtr(x),
        TorchParityUnaryOp.Ndtri => e.TensorNdtri(x),
        TorchParityUnaryOp.BesselJ0 => e.TensorBesselJ0(x),
        TorchParityUnaryOp.BesselJ1 => e.TensorBesselJ1(x),
        TorchParityUnaryOp.BesselY0 => e.TensorBesselY0(x),
        TorchParityUnaryOp.BesselY1 => e.TensorBesselY1(x),
        TorchParityUnaryOp.ModifiedBesselI0 => e.TensorModifiedBesselI0(x),
        TorchParityUnaryOp.ModifiedBesselI1 => e.TensorModifiedBesselI1(x),
        TorchParityUnaryOp.ModifiedBesselK0 => e.TensorModifiedBesselK0(x),
        TorchParityUnaryOp.ModifiedBesselK1 => e.TensorModifiedBesselK1(x),
        TorchParityUnaryOp.ScaledModifiedBesselK0 => e.TensorScaledModifiedBesselK0(x),
        TorchParityUnaryOp.ScaledModifiedBesselK1 => e.TensorScaledModifiedBesselK1(x),
        TorchParityUnaryOp.SphericalBesselJ0 => e.TensorSphericalBesselJ0(x),
        _ => throw new ArgumentOutOfRangeException(nameof(op)),
    };

    [Theory]
    [MemberData(nameof(UnaryCases))]
    public void UnaryDerivative_MatchesTheCpuEnginesTapeGradient(string opName, float p0, float p1, float[] xs)
    {
        var op = Enum.Parse<TorchParityUnaryOp>(opName);
        var cpu = new CpuEngine();
        var x = new Tensor<float>((float[])xs.Clone(), new[] { xs.Length });
        float[] expected;
        Tensor<float> y;
        using (var tape = new GradientTape<float>())
        {
            y = Forward(cpu, op, x, p0, p1);
            expected = tape.ComputeGradients(cpu.ReduceSum(y, null), new[] { x })[x].ToArray();
        }
        float[] actual = DirectGpuTensorEngine.ParityUnaryDerivative(cpu, op, x, y, p0, p1).ToArray();
        AssertClose(op.ToString(), xs, expected, actual);
    }

    [Theory]
    [InlineData("Fmax")]
    [InlineData("Fmin")]
    [InlineData("Igamma")]
    [InlineData("Igammac")]
    public void BinaryDerivative_MatchesTheCpuEnginesTapeGradient(string opName)
    {
        var op = Enum.Parse<TorchParityBinaryOp>(opName);
        var cpu = new CpuEngine();
        bool gamma = op is TorchParityBinaryOp.Igamma or TorchParityBinaryOp.Igammac;
        float[] av = gamma ? new[] { 0.5f, 1.5f, 2f, 3.2f, 0.9f, 4f } : new[] { -1f, 0.5f, 2f, float.NaN, 1f, 3f };
        float[] bv = gamma ? new[] { 0.3f, 2f, 0f, 1.1f, -0.5f, 5f } : new[] { 0.5f, 0.5f, 1f, 2f, float.NaN, 4f };
        var a = new Tensor<float>(av, new[] { av.Length });
        var b = new Tensor<float>(bv, new[] { bv.Length });
        Dictionary<Tensor<float>, Tensor<float>> expected;
        Tensor<float> y;
        using (var tape = new GradientTape<float>())
        {
            y = op switch
            {
                TorchParityBinaryOp.Fmax => cpu.TensorFmax(a, b),
                TorchParityBinaryOp.Fmin => cpu.TensorFmin(a, b),
                TorchParityBinaryOp.Igamma => cpu.TensorIgamma(a, b),
                _ => cpu.TensorIgammac(a, b),
            };
            expected = tape.ComputeGradients(cpu.ReduceSum(y, null), new[] { a, b });
        }
        var actual = new Dictionary<Tensor<float>, Tensor<float>>();
        var ones = new Tensor<float>(Enumerable.Repeat(1f, av.Length).ToArray(), new[] { av.Length });
        DirectGpuTensorEngine.ParityBinaryBackward(ones, new[] { a, b }, y, new object[] { op }, cpu, actual);
        foreach (var (t, name) in new[] { (a, "a"), (b, "b") })
        {
            bool hasExpected = expected.TryGetValue(t, out var e) && e is not null;
            bool hasActual = actual.TryGetValue(t, out var g) && g is not null;
            float[] ev = hasExpected ? e!.ToArray() : new float[av.Length];
            float[] gv = hasActual ? g!.ToArray() : new float[av.Length];
            AssertClose($"{op} d/d{name}", av, ev, gv);
        }
    }

    private static void AssertClose(string what, float[] xs, float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i]) || float.IsNaN(actual[i]))
            {
                Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]),
                    $"{what} at x={xs[i]}: device formula {actual[i]}, CPU {expected[i]}");
                continue;
            }
            double tol = 2e-4 * Math.Max(1.0, Math.Abs(expected[i]));
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tol,
                $"{what} at x={xs[i]}: device formula {actual[i]:G7}, CPU {expected[i]:G7}");
        }
    }
}
