namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Unary PyTorch-parity elementwise ops with a device kernel (#1112).</summary>
internal enum TorchParityUnaryOp
{
    Celu, Hardtanh, LogSigmoid, Softsign, Logit, Sinc, Erf,
    BesselJ0, BesselJ1, BesselY0, BesselY1, ModifiedBesselI0, ModifiedBesselI1, Erfcx, Ndtr, Ndtri, Entr, Signbit, IsPosInf, IsNegInf,
    LogNdtr, ModifiedBesselK0, ModifiedBesselK1, ScaledModifiedBesselK0, ScaledModifiedBesselK1, SphericalBesselJ0, AiryAi, Mvlgamma,
}

/// <summary>Binary (same-shape) PyTorch-parity elementwise ops with a device kernel (#1112).</summary>
internal enum TorchParityBinaryOp { Fmax, Fmin, Heaviside, GreaterEqual, LessEqual, Igamma, Igammac }

/// <summary>
/// Backends with device kernels for the PyTorch-parity elementwise ops that otherwise compute on the host. Optional:
/// a backend without it keeps the recorded host fallback (<c>GpuLaunchProbe.OnFallback</c>).
/// </summary>
/// <remarks>
/// Implemented by the CUDA backend. Not yet implemented, each tracked by its own follow-up issue: HIP (#1124),
/// Metal (#1125), OpenCL (#1126), Vulkan (#1127) and WebGPU (#1128). Until then those backends compute these ops on
/// the host.
/// </remarks>
internal interface ITorchParityElementwiseBackend
{
    /// <summary><paramref name="output"/>[i] = op(<paramref name="input"/>[i]; p0, p1) for i in [0, n).</summary>
    void TorchParityUnary(TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1);

    /// <summary><paramref name="output"/>[i] = op(<paramref name="a"/>[i], <paramref name="b"/>[i]) for i in [0, n).</summary>
    void TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n);

    /// <summary>
    /// <paramref name="output"/>[i] = P_degree[i](<paramref name="x"/>[i]) for the orthogonal-polynomial family
    /// <paramref name="kind"/> (0 Chebyshev T, 1 U, 2 V, 3 W, 4 Hermite H, 5 Hermite He, 6 Laguerre L, 7 Legendre P),
    /// optionally shifted (x -> 2x - 1).
    /// </summary>
    void TorchParityPolynomial(int kind, bool shifted, IGpuBuffer x, IGpuBuffer degree, IGpuBuffer output, int n);

    /// <summary><paramref name="output"/>[i] = start + i * span / denominator (arange / range pass the step and 1).</summary>
    void TorchParityRange(IGpuBuffer output, int n, double start, double span, double denominator);

    /// <summary><paramref name="output"/>[i] = logBase ^ (single ? start : start + i * span / denominator).</summary>
    void TorchParityLogspace(IGpuBuffer output, int n, double start, double span, double denominator, double logBase, bool single);

    /// <summary>
    /// A window over t = i / (m - 1): kind 0 Hann, 1 Hamming (alpha, beta), 2 Blackman, 3 Bartlett, 4 Kaiser (beta in
    /// <paramref name="alpha"/>); n = 1 gives [1].
    /// </summary>
    void TorchParityWindow(IGpuBuffer output, int n, int kind, int m, double alpha, double beta);

    /// <summary>The [2, total] lower (j - i &lt;= offset) or upper (j - i &gt;= offset) triangle indices of a row x col grid.</summary>
    void TorchParityTriIndices(IGpuBuffer output, int row, int col, int offset, bool lower, int total);
}
