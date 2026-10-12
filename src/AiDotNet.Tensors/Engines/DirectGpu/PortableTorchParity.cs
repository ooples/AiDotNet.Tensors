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

    /// <summary>A double as the float pair hi + lo the kernels take for values they must keep to double precision.</summary>
    private static (float Hi, float Lo) Split(double value)
    {
        float hi = (float)value;
        if (float.IsInfinity(hi) || float.IsNaN(hi)) return (hi, 0f);
        return (hi, (float)(value - hi));
    }

    public static void Range(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, double start, double span, double denominator)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        var (s0, s1) = Split(start); var (p0, p1) = Split(span); var (d0, d1) = Split(denominator);
        launch("tp_range", n, output, n, s0, s1, p0, p1, d0, d1);
    }

    public static void Logspace(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, double start, double span,
        double denominator, double logBase, bool single)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        var (s0, s1) = Split(start); var (p0, p1) = Split(span); var (d0, d1) = Split(denominator);
        var (l0, l1) = Split(Math.Log(logBase));
        launch("tp_logspace", n, output, n, s0, s1, p0, p1, d0, d1, l0, l1, single ? 1 : 0);
    }

    public static void Window(PortableGenericFft.Launcher launch, IGpuBuffer output, int n, int kind, int m, double alpha, double beta)
    {
        if (n <= 0) return;
        RequireCapacity(output, n, nameof(output), "n");
        launch("tp_window", n, output, n, kind, m, (float)alpha, (float)beta);
    }

    public static void TriIndices(PortableGenericFft.Launcher launch, IGpuBuffer output, int row, int col, int offset, bool lower, int total)
    {
        if (row <= 0 || total <= 0) return;
        RequireCapacity(output, 2L * total, nameof(output), "2*total");
        launch("tp_tri_indices", row, output, row, col, offset, lower ? 1 : 0, total);
    }

    public static void Diagflat(PortableGenericFft.Launcher launch, IGpuBuffer input, IGpuBuffer output, int n, int size, int offset)
    {
        if (size <= 0) return;
        int total = checked(size * size);
        RequireCapacity(output, total, nameof(output), "size*size");
        RequireCapacity(input, n, nameof(input), "n");
        launch("tp_diagflat", total, input, output, total, size, offset, n);
    }

    public static void DiagonalScatter(IDirectGpuBackend backend, PortableGenericFft.Launcher launch, IGpuBuffer src, IGpuBuffer output,
        int count, int[] otherExtents, int[] otherStrides, int stride1, int stride2, int start1, int start2, int diagonal)
    {
        if (otherExtents is null) throw new ArgumentNullException(nameof(otherExtents));
        if (otherStrides is null) throw new ArgumentNullException(nameof(otherStrides));
        if (otherStrides.Length != otherExtents.Length)
            throw new ArgumentException("otherStrides must hold one stride per other axis.", nameof(otherStrides));
        if (count <= 0) return;
        if (otherExtents.Length > 8) throw new ArgumentOutOfRangeException(nameof(otherExtents), "at most 8 other axes.");
        if (diagonal <= 0) throw new ArgumentOutOfRangeException(nameof(diagonal));
        RequireCapacity(src, count, nameof(src), "count");
        long furthest = (long)(diagonal - 1 + start1) * stride1 + (long)(diagonal - 1 + start2) * stride2;
        for (int k = 0; k < otherExtents.Length; k++) furthest += (long)(otherExtents[k] - 1) * otherStrides[k];
        RequireCapacity(output, furthest + 1, nameof(output), "furthest diagonal offset + 1");
        var geometry = new int[22];
        geometry[0] = otherExtents.Length;
        for (int k = 0; k < otherExtents.Length; k++) { geometry[1 + k] = otherExtents[k]; geometry[9 + k] = otherStrides[k]; }
        geometry[17] = stride1; geometry[18] = stride2; geometry[19] = start1; geometry[20] = start2; geometry[21] = diagonal;
        using var g = backend.AllocateIntBuffer(geometry);
        launch("tp_diagonal_scatter", count, src, output, g, count);
    }

    public static void QuantileRows(PortableGenericFft.Launcher launch, IGpuBuffer sorted, IGpuBuffer output, int rows, int length,
        double q, int mode, bool ignoreNan)
    {
        if (rows <= 0) return;
        RequireCapacity(sorted, (long)rows * length, nameof(sorted), "rows*length");
        RequireCapacity(output, rows, nameof(output), "rows");
        var (q0, q1) = Split(q);
        launch("tp_quantile_rows", rows, sorted, output, rows, length, q0, q1, mode, ignoreNan ? 1 : 0);
    }
}