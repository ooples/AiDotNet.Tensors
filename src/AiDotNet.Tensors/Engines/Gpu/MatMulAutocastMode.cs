namespace AiDotNet.Tensors.Engines.Gpu;

/// <summary>
/// How a matmul autocast scope (<see cref="DirectGpuTensorEngine.EnterMatMulAutocast(PrecisionMode, MatMulAutocastMode)"/>)
/// brings a GEMM's FP32 operands to 16 bits.
/// </summary>
public enum MatMulAutocastMode
{
    /// <summary>
    /// The default, PyTorch's <c>torch.autocast</c> semantics: both operands are rounded to FP16 or BF16 (round to
    /// nearest even) before every GEMM, which multiplies and accumulates them in FP32. Every product is that of the
    /// rounded operands, whatever the shape or the kernel cuBLAS picks.
    /// </summary>
    CastOperands = 0,

    /// <summary>
    /// The FP32 operands go to cuBLAS with a compute type that permits FP16/BF16 tensor-core math
    /// (<c>CUBLAS_COMPUTE_32F_FAST_16F</c> / <c>_FAST_16BF</c>). No cast kernels or 16-bit copies, but the rounding is
    /// cuBLAS's choice per shape: it keeps FP32 kernels where it prefers them (typically small GEMMs), so a model's
    /// results can depend on its batch size and the device.
    /// </summary>
    FastComputeType = 1,
}
