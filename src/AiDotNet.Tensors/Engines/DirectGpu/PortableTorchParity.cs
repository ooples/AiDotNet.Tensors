// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// <see cref="ITorchParityElementwiseBackend"/> over the FP32 kernels of <see cref="TorchParityF32Kernels"/> (#1125,
/// #1128), for backends without 64-bit floats: each member validates as CUDA's launcher does and launches the kernel
/// of the same contract through the backend's launcher.
/// </summary>
internal static class PortableTorchParity
{
    public static string UnaryKernel(TorchParityUnaryOp op) => op switch
    {
        TorchParityUnaryOp.Celu => "tp_celu",
        TorchParityUnaryOp.Hardtanh => "tp_hardtanh",
        TorchParityUnaryOp.LogSigmoid => "tp_logsigmoid",
        TorchParityUnaryOp.Softsign => "tp_softsign",
        TorchParityUnaryOp.Logit => "tp_logit",
        TorchParityUnaryOp.Sinc => "tp_sinc",
        TorchParityUnaryOp.Erf => "tp_erf",
        TorchParityUnaryOp.BesselJ0 => "tp_bessel_j0",
        TorchParityUnaryOp.BesselJ1 => "tp_bessel_j1",
        TorchParityUnaryOp.BesselY0 => "tp_bessel_y0",
        TorchParityUnaryOp.BesselY1 => "tp_bessel_y1",
        TorchParityUnaryOp.ModifiedBesselI0 => "tp_bessel_i0",
        TorchParityUnaryOp.ModifiedBesselI1 => "tp_bessel_i1",
        TorchParityUnaryOp.Erfcx => "tp_erfcx",
        TorchParityUnaryOp.Ndtr => "tp_ndtr",
        TorchParityUnaryOp.Ndtri => "tp_ndtri",
        TorchParityUnaryOp.Entr => "tp_entr",
        TorchParityUnaryOp.Signbit => "tp_signbit",
        TorchParityUnaryOp.IsPosInf => "tp_isposinf",
        TorchParityUnaryOp.IsNegInf => "tp_isneginf",
        TorchParityUnaryOp.LogNdtr => "tp_log_ndtr",
        TorchParityUnaryOp.ModifiedBesselK0 => "tp_bessel_k0",
        TorchParityUnaryOp.ModifiedBesselK1 => "tp_bessel_k1",
        TorchParityUnaryOp.ScaledModifiedBesselK0 => "tp_bessel_k0e",
        TorchParityUnaryOp.ScaledModifiedBesselK1 => "tp_bessel_k1e",
        TorchParityUnaryOp.SphericalBesselJ0 => "tp_spherical_bessel_j0",
        TorchParityUnaryOp.AiryAi => "tp_airy_ai",
        TorchParityUnaryOp.Mvlgamma => "tp_mvlgamma",
        TorchParityUnaryOp.Angle => "tp_angle",
        TorchParityUnaryOp.NonZero => "tp_nonzero",
        _ => throw new System.ArgumentOutOfRangeException(nameof(op)),
    };

    public static string BinaryKernel(TorchParityBinaryOp op) => op switch
    {
        TorchParityBinaryOp.Fmax => "tp_fmax",
        TorchParityBinaryOp.Fmin => "tp_fmin",
        TorchParityBinaryOp.Heaviside => "tp_heaviside",
        TorchParityBinaryOp.GreaterEqual => "tp_greater_equal",
        TorchParityBinaryOp.LessEqual => "tp_less_equal",
        TorchParityBinaryOp.Igamma => "tp_igamma",
        TorchParityBinaryOp.Igammac => "tp_igammac",
        TorchParityBinaryOp.FloorDivide => "tp_floor_divide",
        _ => throw new System.ArgumentOutOfRangeException(nameof(op)),
    };

    private static void RequireCapacity(IGpuBuffer buffer, long required, string parameter, string formula)
    {
        if (buffer is null) throw new System.ArgumentNullException(parameter);
        if (buffer.Size < required)
            throw new System.ArgumentException(
                $"{parameter} buffer too small: capacity={buffer.Size} elements, required={required} ({formula}).", parameter);
    }

    public static void Unary(PortableGenericFft.Launcher launch, TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output,
        int n, float p0, float p1)
    {
        if (n <= 0) return;
        RequireCapacity(input, n, nameof(input), "n");
        RequireCapacity(output, n, nameof(output), "n");
        launch(UnaryKernel(op), n, input, output, n, p0, p1);
    }

    public static void Binary(PortableGenericFft.Launcher launch, TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b,
        IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireCapacity(a, n, nameof(a), "n");
        RequireCapacity(b, n, nameof(b), "n");
        RequireCapacity(output, n, nameof(output), "n");
        launch(BinaryKernel(op), n, a, b, output, n);
    }

    public static void Polynomial(PortableGenericFft.Launcher launch, int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree,
        IGpuBuffer output, int n)
    {
        if (n <= 0) return;
        RequireCapacity(x, n, nameof(x), "n");
        RequireCapacity(degree, n, nameof(degree), "n");
        RequireCapacity(output, n, nameof(output), "n");
        launch("tp_polynomial", n, x, degree, output, n, kind, shifted ? 1 : 0);
    }
}
