namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Unary PyTorch-parity elementwise ops with a device kernel (#1112).</summary>
internal enum TorchParityUnaryOp
{
    Celu, Hardtanh, LogSigmoid, Softsign, Logit, Sinc, Erf,
    BesselJ0, BesselJ1, BesselY0, BesselY1, ModifiedBesselI0, ModifiedBesselI1, Erfcx, Ndtr, Ndtri, Entr, Signbit, IsPosInf, IsNegInf,
}

/// <summary>Binary (same-shape) PyTorch-parity elementwise ops with a device kernel (#1112).</summary>
internal enum TorchParityBinaryOp { Fmax, Fmin, Heaviside, GreaterEqual, LessEqual }

/// <summary>
/// Backends with device kernels for the PyTorch-parity elementwise ops that otherwise compute on the host. Optional:
/// a backend without it keeps the recorded host fallback.
/// </summary>
internal interface ITorchParityElementwiseBackend
{
    /// <summary><paramref name="output"/>[i] = op(<paramref name="input"/>[i]; p0, p1) for i in [0, n).</summary>
    void TorchParityUnary(TorchParityUnaryOp op, IGpuBuffer input, IGpuBuffer output, int n, float p0, float p1);

    /// <summary><paramref name="output"/>[i] = op(<paramref name="a"/>[i], <paramref name="b"/>[i]) for i in [0, n).</summary>
    void TorchParityBinary(TorchParityBinaryOp op, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output, int n);
}
