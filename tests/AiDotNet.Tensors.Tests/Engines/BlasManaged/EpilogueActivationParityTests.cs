using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Tensors #499: ActivationEpilogue (the BlasManaged GEMM-epilogue activation
/// path) was brought to parity with the MlpForward/FusedLinear dispatch tables —
/// it gained SELU, Softplus, HardSwish, HardSigmoid, HardTanh and the new ReLU6 /
/// SoftSign kernels (it previously stopped at ELU). These tests check the fp32 and
/// fp64 epilogue against an independent canonical formula so the two fused
/// activation paths stay numerically consistent.
/// </summary>
public class EpilogueActivationParityTests
{
    private const double SeluAlpha = 1.6732632423543772, SeluScale = 1.0507009873554805;

    private static double Reference(FusedActivationType act, double x) => act switch
    {
        FusedActivationType.ELU => x > 0 ? x : Math.Exp(x) - 1.0,
        FusedActivationType.SELU => SeluScale * (x > 0 ? x : SeluAlpha * (Math.Exp(x) - 1.0)),
        FusedActivationType.Softplus => x > 20 ? x : Math.Log(1.0 + Math.Exp(x)),
        FusedActivationType.Mish => x * Math.Tanh(x > 20 ? x : Math.Log(1.0 + Math.Exp(x))),
        FusedActivationType.HardSwish => x * Math.Max(0.0, Math.Min(1.0, (x + 3.0) / 6.0)),
        FusedActivationType.HardSigmoid => Math.Max(0.0, Math.Min(1.0, (x + 3.0) / 6.0)),
        FusedActivationType.HardTanh => Math.Max(-1.0, Math.Min(1.0, x)),
        FusedActivationType.ReLU6 => Math.Max(0.0, Math.Min(6.0, x)),
        FusedActivationType.SoftSign => x / (1.0 + Math.Abs(x)),
        _ => throw new ArgumentOutOfRangeException(nameof(act)),
    };

    public static TheoryData<FusedActivationType> NewActivations => new()
    {
        FusedActivationType.ELU,
        FusedActivationType.SELU,
        FusedActivationType.Softplus,
        FusedActivationType.Mish,
        FusedActivationType.HardSwish,
        FusedActivationType.HardSigmoid,
        FusedActivationType.HardTanh,
        FusedActivationType.ReLU6,
        FusedActivationType.SoftSign,
    };

    [Theory]
    [MemberData(nameof(NewActivations))]
    public void ApplyFp32_MatchesCanonicalFormula(FusedActivationType act)
    {
        var rng = new Random(20260530);
        const int n = 17; // odd length exercises the scalar tail past any SIMD block
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        var expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = Reference(act, data[i]);

        ActivationEpilogue.Apply<float>(data, ldc: n, m: 1, n: n, act);

        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expected[i] - data[i]) < 1e-4,
                $"{act} fp32: epilogue {data[i]} != canonical {expected[i]} at {i}.");
    }

    [Theory]
    [MemberData(nameof(NewActivations))]
    public void ApplyFp64_MatchesCanonicalFormula(FusedActivationType act)
    {
        var rng = new Random(20260530);
        const int n = 17;
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = rng.NextDouble() * 8.0 - 4.0;
        var expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = Reference(act, data[i]);

        ActivationEpilogue.Apply<double>(data, ldc: n, m: 1, n: n, act);

        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expected[i] - data[i]) < 1e-12,
                $"{act} fp64: epilogue {data[i]} != canonical {expected[i]} at {i}.");
    }

    // ---- parametric (FusedActivationParams) -------------------------------

    private static double ParamRef(FusedActivationType act, double x, FusedActivationParams p) => act switch
    {
        FusedActivationType.LeakyReLU => x >= 0 ? x : p.Alpha!.Value * x,
        FusedActivationType.ELU => x > 0 ? x : p.Alpha!.Value * (Math.Exp(x) - 1.0),
        FusedActivationType.CELU => x >= 0 ? x : p.Alpha!.Value * (Math.Exp(x / p.Alpha!.Value) - 1.0),
        FusedActivationType.ThresholdedReLU => x > p.Theta!.Value ? x : 0.0,
        FusedActivationType.ScaledTanh => p.Alpha!.Value * Math.Tanh(p.Beta!.Value * x),
        _ => throw new ArgumentOutOfRangeException(nameof(act)),
    };

    public static TheoryData<FusedActivationType, FusedActivationParams> ParametricCases => new()
    {
        { FusedActivationType.LeakyReLU, new FusedActivationParams { Alpha = 0.2f } },
        { FusedActivationType.ELU, new FusedActivationParams { Alpha = 2.0f } },
        { FusedActivationType.CELU, new FusedActivationParams { Alpha = 1.5f } },
        { FusedActivationType.ThresholdedReLU, new FusedActivationParams { Theta = 0.5f } },
        { FusedActivationType.ScaledTanh, new FusedActivationParams { Alpha = 1.7f, Beta = 0.66f } },
    };

    [Theory]
    [MemberData(nameof(ParametricCases))]
    public void ApplyFp32_ParametricHonorsParams(FusedActivationType act, FusedActivationParams p)
    {
        var rng = new Random(20260601);
        const int n = 17;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(rng.NextDouble() * 8.0 - 4.0);
        var expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = ParamRef(act, data[i], p);

        ActivationEpilogue.Apply<float>(data, ldc: n, m: 1, n: n, act, p);

        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expected[i] - data[i]) < 1e-4,
                $"{act} fp32 params: epilogue {data[i]} != {expected[i]} at {i}.");
    }

    [Theory]
    [MemberData(nameof(ParametricCases))]
    public void ApplyFp64_ParametricHonorsParams(FusedActivationType act, FusedActivationParams p)
    {
        var rng = new Random(20260601);
        const int n = 17;
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = rng.NextDouble() * 8.0 - 4.0;
        var expected = new double[n];
        for (int i = 0; i < n; i++) expected[i] = ParamRef(act, data[i], p);

        ActivationEpilogue.Apply<double>(data, ldc: n, m: 1, n: n, act, p);

        for (int i = 0; i < n; i++)
            Assert.True(Math.Abs(expected[i] - data[i]) < 1e-12,
                $"{act} fp64 params: epilogue {data[i]} != {expected[i]} at {i}.");
    }
}
