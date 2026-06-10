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

    /// <inheritdoc/>
    public override Tensor<T> TensorMatMul2DWithPrePackedB<T>(
        Tensor<T> a, Tensor<T> b, BlasManaged.WeightPackHandle prePackedB)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (prePackedB is null) throw new ArgumentNullException(nameof(prePackedB));

        // prePackedB is a CPU managed-BLAS panel layout — meaningless on the GPU; `b` is the
        // original right-hand matrix. The CPU path downloads a/b to run the managed pre-packed
        // GEMM; route to the GPU GEMM (a@b) instead so the op stays device-resident. The pre-pack
        // was only a CPU speedup, so dropping it on the GPU changes nothing about the result.
        // Non-2-D / GraphMode / no-GPU defer to base (which also falls back to TensorMatMul for
        // unsupported dtypes).
        if (a.Rank != 2 || b.Rank != 2 || !TryGetBackend(out _))
            return base.TensorMatMul2DWithPrePackedB(a, b, prePackedB);

        try { return TensorMatMul(a, b); }
        catch (Exception) { return base.TensorMatMul2DWithPrePackedB(a, b, prePackedB); }
    }

    /// <inheritdoc/>
    public override T TensorVecDot<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));

        // VecDot(a,b) = sum(a[i]*b[i]) over two equal-length 1-D vectors, returning a scalar.
        // The CPU base downloads BOTH whole vectors to multiply-accumulate on the host; instead
        // do the elementwise multiply + full reduction on the GPU (the expensive O(n) part) and
        // download only the single scalar result. Defer non-conforming shapes / no-GPU to base.
        if (a.Rank != 1 || b.Rank != 1 || a.Length != b.Length || !TryGetBackend(out _))
            return base.TensorVecDot(a, b);

        try
        {
            var prod = TensorMultiply(a, b);           // GPU elementwise [n]
            var sum = ReduceSum(prod, null, false);    // GPU reduce-all -> 1 element
            var span = sum.AsSpan();
            if (span.Length != 1) return base.TensorVecDot(a, b);
            return span[0];
        }
        catch (Exception)
        {
            return base.TensorVecDot(a, b);
        }
    }
}
