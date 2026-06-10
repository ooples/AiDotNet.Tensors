using AiDotNet.Tensors.Engines.DirectGpu;
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

    /// <inheritdoc/>
    public override Tensor<T> TensorMultiDot<T>(Tensor<T>[] matrices)
    {
        if (matrices is null) throw new ArgumentNullException(nameof(matrices));
        if (matrices.Length == 0) throw new ArgumentException("MultiDot requires at least one matrix");
        if (matrices.Length == 1) return matrices[0];

        // The CPU base routes the chain through TensorEinsum (host). Multiply the chain with the
        // device-resident GPU GEMM instead. Require all-2-D with chained inner dims (the GPU GEMM
        // path); otherwise defer to base. GraphMode / no-GPU also defer.
        if (!TryGetBackend(out _))
            return base.TensorMultiDot(matrices);
        foreach (var mt in matrices)
            if (mt is null || mt.Rank != 2)
                return base.TensorMultiDot(matrices);
        for (int i = 1; i < matrices.Length; i++)
            if (matrices[i - 1]._shape[1] != matrices[i]._shape[0])
                return base.TensorMultiDot(matrices);

        try
        {
            // Classic matrix-chain DP to pick the parenthesization that minimizes total GEMM
            // FLOPs (same optimization the CPU einsum greedy-path does), then multiply on the GPU.
            int nM = matrices.Length;
            var dims = new int[nM + 1];
            dims[0] = matrices[0]._shape[0];
            for (int i = 0; i < nM; i++) dims[i + 1] = matrices[i]._shape[1];

            var cost = new long[nM, nM];
            var split = new int[nM, nM];
            for (int len = 2; len <= nM; len++)
                for (int i = 0; i + len - 1 < nM; i++)
                {
                    int j = i + len - 1;
                    cost[i, j] = long.MaxValue;
                    for (int s = i; s < j; s++)
                    {
                        long c = cost[i, s] + cost[s + 1, j] + (long)dims[i] * dims[s + 1] * dims[j + 1];
                        if (c < cost[i, j]) { cost[i, j] = c; split[i, j] = s; }
                    }
                }

            return MultiplyChainOnGpu(matrices, split, 0, nM - 1);
        }
        catch (Exception)
        {
            return base.TensorMultiDot(matrices);
        }
    }

    // Recursively multiply matrices[i..j] in the optimal order from the chain-DP split table,
    // using the GPU-resident TensorMatMul (which records MatMulBackward for each product).
    private Tensor<T> MultiplyChainOnGpu<T>(Tensor<T>[] m, int[,] split, int i, int j)
    {
        if (i == j) return m[i];
        int s = split[i, j];
        var left = MultiplyChainOnGpu(m, split, i, s);
        var right = MultiplyChainOnGpu(m, split, s + 1, j);
        return TensorMatMul(left, right);
    }

    /// <inheritdoc/>
    public override Tensor<T> BatchNormInference<T>(
        Tensor<T> x, Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, double epsilon)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (gamma is null) throw new ArgumentNullException(nameof(gamma));
        if (beta is null) throw new ArgumentNullException(nameof(beta));
        if (mean is null) throw new ArgumentNullException(nameof(mean));
        if (variance is null) throw new ArgumentNullException(nameof(variance));

        // The CPU base normalizes (x-mean)/sqrt(var+eps)*gamma+beta on the host, downloading x.
        // The backend BatchNorm kernel with training:false does exactly this using the PROVIDED
        // running mean/variance — a single on-device kernel. The op's contract is rank-4 NCHW;
        // defer non-rank-4 to base (it throws the proper error). Tape-active / no-GPU also defer
        // (BatchNormInference is an eval-mode op, so the tape is normally inactive).
        if (IsTapeActive<T>() || x.Rank != 4 || !TryGetBackend(out var backend))
            return base.BatchNormInference(x, gamma, beta, mean, variance, epsilon);

        var bufOut = default(OwnedBuffer);
        bool ownershipTransferred = false;
        try
        {
            int batch = x.Shape._dims[0];
            int channels = x.Shape._dims[1];
            int spatial = x.Length / (batch * channels);
            using var bufIn = GetOrAllocateBuffer(backend, x.GetDataArray());
            using var bufGamma = GetOrAllocateBuffer(backend, gamma.GetDataArray());
            using var bufBeta = GetOrAllocateBuffer(backend, beta.GetDataArray());
            using var bufMean = GetOrAllocateBuffer(backend, mean.GetDataArray());
            using var bufVar = GetOrAllocateBuffer(backend, variance.GetDataArray());
            bufOut = AllocateOutputBuffer(backend, x.Length);
            // saveMean/saveInvVar are training-only outputs; the inference path ignores them but
            // the kernel signature requires the buffers, so hand it scratch of size [channels].
            using var bufSaveMean = AllocateOutputBuffer(backend, channels);
            using var bufSaveInvVar = AllocateOutputBuffer(backend, channels);
            backend.BatchNorm(bufIn.Buffer, bufOut.Buffer, bufGamma.Buffer, bufBeta.Buffer,
                bufMean.Buffer, bufVar.Buffer, bufSaveMean.Buffer, bufSaveInvVar.Buffer,
                batch, channels, spatial, (float)epsilon, 0.1f, /*training:*/ false);
            var result = FinishGpuOp<T>(backend, bufOut, x.Length);
            ownershipTransferred = true;
            return new Tensor<T>(result, x.Shape._dims);
        }
        catch (Exception)
        {
            if (!ownershipTransferred) bufOut.Dispose();
            return base.BatchNormInference(x, gamma, beta, mean, variance, epsilon);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorInner<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));

        // Inner product contracts the LAST axis: a[M,K], b[N,K] -> [M,N] with out[i,j]=Σ_k a[i,k]b[j,k]
        // = a @ bᵀ — exactly TensorMatMulTransposed, which is GPU-resident (this is the Q·Kᵀ attention
        // score pattern). The CPU base routes through TensorEinsum (host). Handle the 2-D × 2-D case on
        // the GPU; higher-rank inner products need a host reshape/transpose dance, so defer those to base.
        if (a.Rank != 2 || b.Rank != 2
            || a._shape[a.Rank - 1] != b._shape[b.Rank - 1]
            || !TryGetBackend(out _))
        {
            return base.TensorInner(a, b);
        }

        try { return TensorMatMulTransposed(a, b); }
        catch (Exception) { return base.TensorInner(a, b); }
    }
}
