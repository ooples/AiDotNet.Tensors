using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// GPU-residency audit follow-ups: ops that CpuEngine exposes but DirectGpuTensorEngine
// did NOT override, so on a GPU device they silently ran on the CpuEngine base — which
// downloads every input from VRAM, computes on the host, and forces the next GPU op to
// re-upload (the "kernels missing on the GPU side fall back to CPU and the data keeps
// transferring back and forth" problem).
//
// These overrides keep the result GPU-resident by COMPOSING the op out of the existing
// GPU-resident primitive overrides (GEMM, scaled-add, etc.). That is also the maximum-
// performance path for this family: the heavy work is the GEMM, which already runs the
// tuned backend kernel; the only thing the fallback added was host transfers. Each
// primitive sub-op records its own backward on the tape, so the composed autodiff graph
// is gradient-equivalent to the fused CPU backward.
public partial class DirectGpuTensorEngine
{
    /// <inheritdoc/>
    public override Tensor<T> TensorAddMM<T>(Tensor<T> input, Tensor<T> a, Tensor<T> b, T alpha, T beta)
    {
        // Null contract matches the base (throw, don't silently no-op).
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));

        // Fall back to the CpuEngine base (which records the LazyNode in GraphMode / the
        // fused AddMMBackward on the tape, and validates shapes) when:
        //  - GraphMode is tracing or no GPU is available (TryGetBackend == false), or
        //  - the operands are not the 2-D matrices the GPU GEMM path handles.
        if (a.Rank != 2 || b.Rank != 2 || !TryGetBackend(out _))
        {
            return base.TensorAddMM(input, a, b, alpha, beta);
        }

        try
        {
            // result = alpha*(a@b) + beta*input, entirely on the device.
            // TensorMatMul and TensorAddScaled are GPU-resident overrides that each record
            // their own backward (MatMulBackward, AddScaledBackward), so AddScaled∘MatMul is
            // gradient-equivalent to the single fused AddMMBackward the base records.
            var mm = TensorMatMul(a, b);
            return TensorAddScaled(mm, input, alpha, beta);
        }
        catch (Exception)
        {
            // Any shape/dtype/GPU error: defer to the correct, fully-validated CPU path.
            return base.TensorAddMM(input, a, b, alpha, beta);
        }
    }
}
