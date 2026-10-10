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
}
