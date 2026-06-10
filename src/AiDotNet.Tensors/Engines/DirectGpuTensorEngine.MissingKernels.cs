using System.Collections.Generic;
using AiDotNet.Tensors;
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

        int ra = a.Rank, rb = b.Rank;
        // Inner product contracts the LAST axis of each operand (sizes must match) and produces
        // out-shape a.shape[:-1] ++ b.shape[:-1]. Flatten the free dims: a -> [Ma,K], b -> [Mb,K],
        // then out2d = a2 @ b2ᵀ (= TensorMatMulTransposed, GPU-resident) reshaped to the full out-shape.
        // Reshape preserves GPU residency (the view shares the cached buffer), so the whole chain stays
        // on the device — no host round-trip, unlike the CPU base's TensorEinsum path. The rank-1×rank-1
        // pure scalar dot (empty out-shape / rank-0) is left to base; TensorVecDot covers that case.
        if (ra < 1 || rb < 1 || (ra == 1 && rb == 1)
            || a._shape[ra - 1] != b._shape[rb - 1]
            || !TryGetBackend(out _))
        {
            return base.TensorInner(a, b);
        }

        try
        {
            int k = a._shape[ra - 1];
            int ma = a.Length / k;
            int mb = b.Length / k;
            var a2 = ra == 2 ? a : a.Reshape(ma, k);
            var b2 = rb == 2 ? b : b.Reshape(mb, k);
            var r2 = TensorMatMulTransposed(a2, b2);   // [ma, mb], device-resident

            if (ra == 2 && rb == 2) return r2;          // already the right shape

            var outShape = new int[(ra - 1) + (rb - 1)];
            for (int i = 0; i < ra - 1; i++) outShape[i] = a._shape[i];
            for (int i = 0; i < rb - 1; i++) outShape[ra - 1 + i] = b._shape[i];
            return r2.Reshape(outShape);
        }
        catch (Exception) { return base.TensorInner(a, b); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorDot<T>(Tensor<T> a, Tensor<T> b, int[] axesA, int[] axesB)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (axesA is null) throw new ArgumentNullException(nameof(axesA));
        if (axesB is null) throw new ArgumentNullException(nameof(axesB));

        int ra = a.Rank, rb = b.Rank, na = axesA.Length;

        // The CPU base routes through TensorEinsum (host). Implement the general contraction on the
        // GPU as permute(a)→reshape→GEMM→reshape, which stays device-resident (GPU TensorPermute +
        // residency-preserving Reshape + GPU TensorMatMul). Defer to base for: mismatched axis counts,
        // the no-contraction (outer-product, na==0) and full-contraction-to-scalar (rank-0 output)
        // edges, invalid/negative/duplicate axes, int-overflowing sizes, and no-GPU.
        if (na != axesB.Length || na == 0 || !TryGetBackend(out _))
            return base.TensorDot(a, b, axesA, axesB);

        var inA = new bool[ra];
        var inB = new bool[rb];
        long k = 1;
        for (int i = 0; i < na; i++)
        {
            int ax = axesA[i], bx = axesB[i];
            if (ax < 0 || ax >= ra || bx < 0 || bx >= rb || inA[ax] || inB[bx]
                || a._shape[ax] != b._shape[bx])
                return base.TensorDot(a, b, axesA, axesB);
            inA[ax] = true; inB[bx] = true;
            k *= a._shape[ax];
        }

        var freeA = new List<int>(ra - na);
        for (int i = 0; i < ra; i++) if (!inA[i]) freeA.Add(i);
        var freeB = new List<int>(rb - na);
        for (int i = 0; i < rb; i++) if (!inB[i]) freeB.Add(i);
        if (freeA.Count == 0 && freeB.Count == 0)
            return base.TensorDot(a, b, axesA, axesB);   // scalar (rank-0) result → base

        long faL = 1; foreach (var ax in freeA) faL *= a._shape[ax];
        long fbL = 1; foreach (var bx in freeB) fbL *= b._shape[bx];
        if (faL > int.MaxValue || fbL > int.MaxValue || k > int.MaxValue)
            return base.TensorDot(a, b, axesA, axesB);

        try
        {
            // Permute a to [free_a..., contracted...] and b to [contracted..., free_b...] so the
            // contraction becomes a plain 2-D GEMM aP[Fa,K] @ bP[K,Fb].
            var permA = new int[ra];
            { int p = 0; foreach (var ax in freeA) permA[p++] = ax; for (int i = 0; i < na; i++) permA[p++] = axesA[i]; }
            var permB = new int[rb];
            { int p = 0; for (int i = 0; i < na; i++) permB[p++] = axesB[i]; foreach (var bx in freeB) permB[p++] = bx; }

            var aP = IsIdentityPerm(permA) ? a : TensorPermute(a, permA);
            var bP = IsIdentityPerm(permB) ? b : TensorPermute(b, permB);
            var a2 = aP.Reshape((int)faL, (int)k);
            var b2 = bP.Reshape((int)k, (int)fbL);
            var r2 = TensorMatMul(a2, b2);   // [Fa, Fb], device-resident

            var outShape = new int[freeA.Count + freeB.Count];
            int oi = 0;
            foreach (var ax in freeA) outShape[oi++] = a._shape[ax];
            foreach (var bx in freeB) outShape[oi++] = b._shape[bx];
            return r2.Reshape(outShape);
        }
        catch (Exception) { return base.TensorDot(a, b, axesA, axesB); }
    }

    private static bool IsIdentityPerm(int[] perm)
    {
        for (int i = 0; i < perm.Length; i++) if (perm[i] != i) return false;
        return true;
    }

    // ----- Category F: shape / view / layout -----
    // The CPU base implements these as tensor.Transpose(perm), which returns a STRIDED view.
    // Feeding a strided view to a GPU op forces a host materialization (download + CPU permute +
    // re-upload). Routing through the GPU TensorPermute override physically reorders on the device
    // into a contiguous resident buffer instead — same permutation, no host round-trip.
    // (TensorAtLeast1D/2D/3D are deliberately NOT overridden: they delegate to Reshape, which already
    // preserves GPU residency, so the base is fine.)

    /// <inheritdoc/>
    public override Tensor<T> TensorSwapAxes<T>(Tensor<T> tensor, int axis1, int axis2)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (axis1 < 0) axis1 += rank;
        if (axis2 < 0) axis2 += rank;
        if (axis1 < 0 || axis1 >= rank || axis2 < 0 || axis2 >= rank || !TryGetBackend(out _))
            return base.TensorSwapAxes(tensor, axis1, axis2);

        try
        {
            var perm = new int[rank];
            for (int i = 0; i < rank; i++) perm[i] = i;
            perm[axis1] = axis2;
            perm[axis2] = axis1;
            return TensorPermute(tensor, perm);
        }
        catch (Exception) { return base.TensorSwapAxes(tensor, axis1, axis2); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMoveDim<T>(Tensor<T> tensor, int source, int destination)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (source < 0) source += rank;
        if (destination < 0) destination += rank;
        if (source < 0 || source >= rank || destination < 0 || destination >= rank || !TryGetBackend(out _))
            return base.TensorMoveDim(tensor, source, destination);

        try
        {
            // Build the SAME permutation the CPU base does (pull out `source`, insert at
            // `destination`), then materialize it on the GPU.
            var perm = new int[rank];
            int w = 0;
            for (int i = 0; i < rank; i++)
            {
                if (w == destination) { perm[w++] = source; }
                if (i != source)
                {
                    if (w == destination) { perm[w++] = source; }
                    perm[w++] = i;
                }
            }
            if (w < rank) perm[w++] = source;
            return TensorPermute(tensor, perm);
        }
        catch (Exception) { return base.TensorMoveDim(tensor, source, destination); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorRot90<T>(Tensor<T> tensor, int k = 1, int[]? axes = null)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        axes ??= new[] { 0, 1 };
        if (axes.Length != 2 || axes[0] == axes[1] || !TryGetBackend(out var backend))
            return base.TensorRot90(tensor, k, axes);

        int steps = ((k % 4) + 4) % 4;
        if (steps == 0)
            return base.TensorRot90(tensor, k, axes);

        try
        {
            int a0 = axes[0], a1 = axes[1];
            if (a0 < 0) a0 += tensor.Rank;
            if (a1 < 0) a1 += tensor.Rank;
            if (a0 < 0 || a0 >= tensor.Rank || a1 < 0 || a1 >= tensor.Rank)
                return base.TensorRot90(tensor, k, axes);

            // Each 90° step = swap the two axes + flip the (new) axes[0]. Use the MATERIALIZING GPU
            // permute (PermuteResidentGpu), NOT the view-returning TensorPermute: the strided permute
            // view shares the un-permuted buffer, and interleaving that with TensorFlip across multiple
            // steps fed the next op a stale/mis-laid-out buffer (the root-caused deep-chain divergence).
            // Materializing each permute hands TensorFlip a correctly-laid-out contiguous resident buffer.
            var result = tensor;
            for (int s = 0; s < steps; s++)
            {
                var perm = new int[result.Rank];
                for (int i = 0; i < result.Rank; i++) perm[i] = i;
                perm[a0] = a1;
                perm[a1] = a0;
                result = PermuteResidentGpu(backend, result, perm);
                result = TensorFlip(result, new[] { a0 });
            }
            return result;
        }
        catch (Exception) { return base.TensorRot90(tensor, k, axes); }
    }

    /// <inheritdoc/>
    public override Tensor<T>[] TensorTensorSplit<T>(Tensor<T> tensor, int[] indices, int dim = 0)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank || !TryGetBackend(out _))
            return base.TensorTensorSplit(tensor, indices, dim);

        // The CPU base slices each chunk on the host (SliceAlongAxis → GetDataArray loops). Extract
        // every chunk with the GPU-resident TensorSlice instead. The sections/H/V/D-split variants
        // all funnel through this virtual overload, so they ride along automatically. Empty chunks
        // (degenerate split points) fall back to base, which handles the zero-length-dim case.
        try
        {
            int dimSize = tensor._shape[dim];
            var result = new Tensor<T>[indices.Length + 1];
            int prev = 0;
            for (int i = 0; i <= indices.Length; i++)
            {
                int cur = i < indices.Length ? Math.Min(Math.Max(indices[i], prev), dimSize) : dimSize;
                int len = cur - prev;
                if (len <= 0)
                    return base.TensorTensorSplit(tensor, indices, dim);

                var start = new int[rank];
                var length = new int[rank];
                for (int k = 0; k < rank; k++) { start[k] = 0; length[k] = tensor._shape[k]; }
                start[dim] = prev;
                length[dim] = len;
                result[i] = TensorSlice(tensor, start, length);
                prev = cur;
            }
            return result;
        }
        catch (Exception) { return base.TensorTensorSplit(tensor, indices, dim); }
    }

    /// <inheritdoc/>
    public override Tensor<T>[] TensorSplit<T>(Tensor<T> tensor, int numSplits, int axis = 0)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (numSplits <= 0) throw new ArgumentException("Number of splits must be positive", nameof(numSplits));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;

        // Defer: GraphMode/no-GPU (TryGetBackend false → base records the lazy nodes), the packed
        // NCHWc8 layout (its slicing is physical-layout-specific), and non-divisible sizes (base
        // throws the proper error). TensorHSplit/VSplit/DSplit funnel through this virtual overload.
        if (axis < 0 || axis >= rank
            || tensor.Layout == LinearAlgebra.TensorLayout.Nchwc8
            || !TryGetBackend(out _))
            return base.TensorSplit(tensor, numSplits, axis);

        int axisSize = tensor._shape[axis];
        if (axisSize % numSplits != 0)
            return base.TensorSplit(tensor, numSplits, axis);

        try
        {
            int splitSize = axisSize / numSplits;
            var result = new Tensor<T>[numSplits];
            for (int i = 0; i < numSplits; i++)
            {
                var start = new int[rank];
                var length = new int[rank];
                for (int k = 0; k < rank; k++) { start[k] = 0; length[k] = tensor._shape[k]; }
                start[axis] = i * splitSize;
                length[axis] = splitSize;
                result[i] = TensorSlice(tensor, start, length);   // GPU-resident chunk
            }
            return result;
        }
        catch (Exception) { return base.TensorSplit(tensor, numSplits, axis); }
    }

    // ----- Category D: elementwise (wiring existing backend kernels) -----
    // (Reciprocal/Sin/Tanh in the audit are Vector<T> helpers, not Tensor ops — not GPU-residency
    // gaps. The Tensor<Bit>/bool predicates (Eq/IsNan/…) don't fit the float-buffer residency model
    // cleanly and are deferred.)

    /// <inheritdoc/>
    public override Tensor<T> TensorClampMin<T>(Tensor<T> tensor, T min)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        // Tape-active defers to base (preserves the boxed-double backward contract, like TensorClamp).
        if (IsTapeActive<T>() || !TryGetBackend(out var backend))
            return base.TensorClampMin(tensor, min);

        try
        {
            // clamp(x, min, +inf) == max(x, min) exactly (no MaxValue capping of large/inf inputs).
            float minF = ToFloatScalar(min);
            using var bufA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, tensor.Length);
            backend.Clamp(bufA.Buffer, bufOut.Buffer, minF, float.PositiveInfinity, tensor.Length);
            var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
            return new Tensor<T>(result, tensor.Shape._dims);
        }
        catch (Exception) { return base.TensorClampMin(tensor, min); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorClampMax<T>(Tensor<T> tensor, T max)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (IsTapeActive<T>() || !TryGetBackend(out var backend))
            return base.TensorClampMax(tensor, max);

        try
        {
            // clamp(x, -inf, max) == min(x, max) exactly.
            float maxF = ToFloatScalar(max);
            using var bufA = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, tensor.Length);
            backend.Clamp(bufA.Buffer, bufOut.Buffer, float.NegativeInfinity, maxF, tensor.Length);
            var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
            return new Tensor<T>(result, tensor.Shape._dims);
        }
        catch (Exception) { return base.TensorClampMax(tensor, max); }
    }

    // ----- Category C: gather / scatter (wiring existing backend kernels) -----
    // Gather ops are read-only (no atomics) so they wire safely. Scatter-add (atomicAdd) is
    // deliberately NOT wired here — a prior engine-level ScatterAdd wiring diverged from CPU
    // (see the note at DirectGpuTensorEngine.TensorGather), so it needs careful A/B validation
    // across the shape grid before re-adding; until then it stays on the correct CPU base.

    /// <inheritdoc/>
    public override Tensor<T> TensorTake<T>(Tensor<T> tensor, Tensor<int> indices)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (!TryGetBackend(out var backend))
            return base.TensorTake(tensor, indices);

        // Flat gather: dst[i] = flat(tensor)[indices[i]]. The backend Gather kernel with featureSize=1
        // is exactly this (output[i] = source[indices[i]]). Output shape = indices.shape.
        try
        {
            var src = tensor.IsContiguous ? tensor : tensor.Contiguous();
            int n = indices.Length;
            using var bufSrc = GetOrAllocateBuffer(backend, src.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(indices.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Gather(bufSrc.Buffer, bufIdx, bufOut.Buffer, n, 1);
            var result = FinishGpuOp<T>(backend, bufOut, n);
            var output = new Tensor<T>(result, (int[])indices._shape.Clone());
            Autodiff.DifferentiableOps.RecordUnary("TensorTake", output, tensor,
                Autodiff.BackwardFunctions<T>.TakeBackward,
                new object[] { indices, (int[])tensor._shape.Clone() });
            return output;
        }
        catch (Exception) { return base.TensorTake(tensor, indices); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorScatterAdd<T>(Tensor<T> destination, Tensor<int> indices, Tensor<T> updates, int axis = 0)
    {
        if (destination is null) throw new ArgumentNullException(nameof(destination));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (updates is null) throw new ArgumentNullException(nameof(updates));
        if (axis < 0) axis += destination.Rank;

        // index_add: result = copy(destination); result[index[i]] += updates[i]. The backend ScatterAdd
        // kernel is FLAT (featureSize=1), so only the 1-D / axis-0 case maps; higher-rank / featured /
        // non-zero-axis fall back to the (correct) base. Tape-active also defers (the scatter backward
        // stays on the base's recording). The prior engine wiring diverged because it did NOT seed the
        // output buffer with the base destination — scatter is an ADD onto the existing values — so the
        // base contribution was missing. Seed it with backend.Copy first, then atomic-add the updates.
        if (IsTapeActive<T>() || axis != 0
            || destination.Rank != 1 || updates.Rank != 1 || indices.Rank != 1
            || indices.Length != updates.Length
            || !TryGetBackend(out var backend))
            return base.TensorScatterAdd(destination, indices, updates, axis);

        try
        {
            int destSize = destination.Length;
            using var bufDest = GetOrAllocateBuffer(backend, destination.GetDataArray());
            using var bufUpd = GetOrAllocateBuffer(backend, updates.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(indices.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, destSize);
            backend.Copy(bufDest.Buffer, bufOut.Buffer, destSize);   // seed with the base (the += target)
            backend.ScatterAdd(bufUpd.Buffer, bufIdx, bufOut.Buffer, updates.Length, destSize);
            var result = FinishGpuOp<T>(backend, bufOut, destSize);
            return new Tensor<T>(result, (int[])destination._shape.Clone());
        }
        catch (Exception) { return base.TensorScatterAdd(destination, indices, updates, axis); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIndexAdd<T>(Tensor<T> tensor, int axis, Tensor<int> indices, Tensor<T> source)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (axis < 0) axis += tensor.Rank;

        // index_add: result = clone(tensor); result[index[i]] += source[i]. Same on-device path as
        // TensorScatterAdd (seed the output with the base, then atomic-add) — only the 1-D / axis-0
        // case the flat backend kernel covers; the rest defer to base.
        if (IsTapeActive<T>() || axis != 0
            || tensor.Rank != 1 || source.Rank != 1 || indices.Rank != 1
            || indices.Length != source.Length
            || !TryGetBackend(out var backend))
            return base.TensorIndexAdd(tensor, axis, indices, source);

        try
        {
            int destSize = tensor.Length;
            using var bufDest = GetOrAllocateBuffer(backend, tensor.GetDataArray());
            using var bufSrc = GetOrAllocateBuffer(backend, source.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(indices.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, destSize);
            backend.Copy(bufDest.Buffer, bufOut.Buffer, destSize);
            backend.ScatterAdd(bufSrc.Buffer, bufIdx, bufOut.Buffer, source.Length, destSize);
            var result = FinishGpuOp<T>(backend, bufOut, destSize);
            return new Tensor<T>(result, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorIndexAdd(tensor, axis, indices, source); }
    }

    // ----- Category F: repeat -----

    /// <inheritdoc/>
    public override Tensor<T> TensorRepeatInterleave<T>(Tensor<T> tensor, int repeats, int dim)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats), "repeats must be >= 1");
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;

        // The backend repeat_elements kernel writes the `repeats` copies immediately after each element
        // (output[outer, inner*repeats + r] = input[outer, inner]), which matches repeat_interleave only
        // when `dim` is the LAST axis (no inner block after it). Middle dims defer to base.
        if (dim != rank - 1 || dim < 0 || !TryGetBatchBackend(out var backend))
            return base.TensorRepeatInterleave(tensor, repeats, dim);

        try
        {
            var src = tensor.IsContiguous ? tensor : tensor.Contiguous();
            int innerSize = tensor._shape[dim];
            int outer = tensor.Length / innerSize;
            int outLen = tensor.Length * repeats;
            using var bufIn = GetOrAllocateBuffer(backend, src.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, outLen);
            backend.RepeatElements(bufIn.Buffer, bufOut.Buffer, outer, innerSize, repeats);
            var result = FinishGpuOp<T>(backend, bufOut, outLen);
            var outShape = (int[])tensor._shape.Clone();
            outShape[dim] = innerSize * repeats;
            var output = new Tensor<T>(result, outShape);
            Autodiff.DifferentiableOps.RecordUnary("TensorRepeatInterleave", output, tensor,
                Autodiff.BackwardFunctions<T>.RepeatInterleaveBackward, new object[] { repeats, dim });
            return output;
        }
        catch (Exception) { return base.TensorRepeatInterleave(tensor, repeats, dim); }
    }

    // ----- Category E: cosine similarity (last-axis reduction) -----

    /// <inheritdoc/>
    public override Tensor<T> TensorCosineSimilarity<T>(Tensor<T> x1, Tensor<T> x2, int dim = -1, double eps = 1e-8)
    {
        if (x1 is null) throw new ArgumentNullException(nameof(x1));
        if (x2 is null) throw new ArgumentNullException(nameof(x2));
        int rank = x1.Rank;
        if (dim < 0) dim += rank;

        // The backend CosineSimilarity kernel reduces the LAST axis (batchSize × axisLen); middle-dim
        // reductions and shape mismatches defer to base (which validates and handles the general case).
        if (dim != rank - 1 || dim < 0 || x2.Rank != rank || x1.Length != x2.Length
            || !ShapesEqual(x1._shape, x2._shape)
            || !TryGetBatchBackend(out var backend))
            return base.TensorCosineSimilarity(x1, x2, dim, eps);

        try
        {
            var a = x1.IsContiguous ? x1 : x1.Contiguous();
            var b = x2.IsContiguous ? x2 : x2.Contiguous();
            int axisLen = x1._shape[dim];
            int batchSize = x1.Length / axisLen;
            using var bufA = GetOrAllocateBuffer(backend, a.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, b.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, batchSize);
            backend.CosineSimilarity(bufA.Buffer, bufB.Buffer, bufOut.Buffer, batchSize, axisLen);
            var result = FinishGpuOp<T>(backend, bufOut, batchSize);
            var outShape = new int[rank - 1];
            for (int i = 0; i < rank - 1; i++) outShape[i] = x1._shape[i];
            return new Tensor<T>(result, outShape.Length == 0 ? new[] { 1 } : outShape);
        }
        catch (Exception) { return base.TensorCosineSimilarity(x1, x2, dim, eps); }
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    // ----- Stream-compaction family -----

    /// <inheritdoc/>
    public override Tensor<T> TensorMaskedSelect<T>(Tensor<T> tensor, Tensor<Bit> mask)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (typeof(T) != typeof(float) || !ShapesEqual(tensor._shape, mask._shape) || !TryGetBackend(out var backend))
            return base.TensorMaskedSelect(tensor, mask);
        try
        {
            // The mask is a CPU Bit tensor, so build the kept flat-index list on the host, then GATHER the
            // (resident) tensor values on the GPU — the tensor never leaves the device, only the count
            // selected elements come back. Order is preserved (indices ascending).
            var cm = mask.IsContiguous ? mask : (Tensor<Bit>)mask.Contiguous();
            var mspan = cm.GetDataArray();
            int n = tensor.Length;
            var kept = new List<int>();
            for (int i = 0; i < n; i++) if (mspan[i]) kept.Add(i);
            int count = kept.Count;
            if (count == 0) return new Tensor<T>(System.Array.Empty<T>(), new[] { 0 });

            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            using var inBuf = GetOrAllocateBuffer(backend, ct.GetDataArray());
            using var idxBuf = new OwnedBuffer(backend.AllocateIntBuffer(kept.ToArray()), true);
            var outBuf = AllocateOutputBuffer(backend, count);
            backend.Gather(inBuf.Buffer, idxBuf.Buffer, outBuf.Buffer, count, 1);
            var arr = FinishGpuOp<T>(backend, outBuf, count);
            return new Tensor<T>(arr, new[] { count });
        }
        catch (Exception) { return base.TensorMaskedSelect(tensor, mask); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorUnique<T>(Tensor<T> input, bool sorted = true)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        // Only the sorted form maps to the bitonic sort; unsorted / non-float defer to base.
        if (typeof(T) != typeof(float) || !sorted || input.Length == 0 || !TryGetBackend(out var backend))
            return base.TensorUnique(input, sorted);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int L = input.Length;
            var (vals, _) = BitonicSortRows(backend, (float[])(object)c.GetDataArray(), 1, L, false);
            // Dedup consecutive over the GPU-sorted values; collapse all NaN to one (torch.unique semantics).
            var uniq = new List<float>();
            for (int i = 0; i < L; i++)
            {
                if (i == 0) { uniq.Add(vals[i]); continue; }
                bool bothNan = float.IsNaN(vals[i]) && float.IsNaN(vals[i - 1]);
                if (!bothNan && vals[i] != vals[i - 1]) uniq.Add(vals[i]);
            }
            var arr = uniq.ToArray();
            return new Tensor<T>((T[])(object)arr, new[] { arr.Length });
        }
        catch (Exception) { return base.TensorUnique(input, sorted); }
    }

    // ----- Sort family (GPU bitonic sort) -----

    // Sorts numRows contiguous rows of length L on the GPU (NaN-aware, torch order) and returns the
    // sorted values and original-position indices in the SAME [numRows*L] layout. Rows are padded to a
    // power of two on the device (ascending pad = NaN -> +inf -> tail; descending pad = -inf -> tail);
    // the host drives the O(log^2 P) bitonic (k, j) schedule. Indices ride along as floats (exact for
    // the index range). Verified against the CPU reference.
    private (float[] vals, int[] idx) BitonicSortRows(IDirectGpuBackend backend, float[] data, int numRows, int L, bool descending)
    {
        int P = 1; while (P < L) P <<= 1;
        float padVal = descending ? float.NegativeInfinity : float.NaN;
        using var srcBuf = GetOrAllocateBuffer(backend, data);
        using var valBuf = AllocateOutputBuffer(backend, numRows * P);
        using var idxBuf = AllocateOutputBuffer(backend, numRows * P);
        backend.Fill(valBuf.Buffer, padVal, numRows * P);
        backend.CopyRows(srcBuf.Buffer, valBuf.Buffer, L, P, numRows, L);   // pad rows L -> P
        backend.IotaPad(idxBuf.Buffer, L, P, numRows);
        int desc = descending ? 1 : 0;
        for (int k = 2; k <= P; k <<= 1)
            for (int j = k >> 1; j > 0; j >>= 1)
                backend.BitonicStep(valBuf.Buffer, idxBuf.Buffer, P, k, j, numRows, desc);
        using var outVal = AllocateOutputBuffer(backend, numRows * L);
        using var outIdx = AllocateOutputBuffer(backend, numRows * L);
        backend.CopyRows(valBuf.Buffer, outVal.Buffer, P, L, numRows, L);   // un-pad P -> L (drop tail)
        backend.CopyRows(idxBuf.Buffer, outIdx.Buffer, P, L, numRows, L);
        backend.Synchronize();
        var vals = backend.DownloadBuffer(outVal.Buffer);
        var idxF = backend.DownloadBuffer(outIdx.Buffer);
        var idx = new int[numRows * L];
        for (int i = 0; i < idx.Length; i++) idx[i] = (int)idxF[i];
        return (vals, idx);
    }

    /// <inheritdoc/>
    public override (Tensor<T> Values, Tensor<int> Indices) TensorSort<T>(Tensor<T> input, int axis = -1, bool descending = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        int rank = input.Rank;
        int ax = axis < 0 ? axis + rank : axis;
        // GPU bitonic handles the contiguous last-axis case (torch default); other axes defer to base.
        if (typeof(T) != typeof(float) || ax != rank - 1 || !TryGetBackend(out var backend))
            return base.TensorSort(input, axis, descending);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int L = input._shape[rank - 1];
            int numRows = L == 0 ? 0 : input.Length / L;
            if (numRows == 0 || L == 0) return base.TensorSort(input, axis, descending);
            var data = (float[])(object)c.GetDataArray();
            var (vals, idx) = BitonicSortRows(backend, data, numRows, L, descending);
            var valuesT = new Tensor<T>((T[])(object)vals, (int[])input._shape.Clone());
            var indicesT = new Tensor<int>(idx, (int[])input._shape.Clone());
            return (valuesT, indicesT);
        }
        catch (Exception) { return base.TensorSort(input, axis, descending); }
    }

    /// <inheritdoc/>
    public override Tensor<int> TensorArgsort<T>(Tensor<T> input, int axis = -1, bool descending = false)
    {
        // Argsort = Sort's indices; the GPU Sort override above carries it.
        var (_, indices) = TensorSort(input, axis, descending);
        return indices;
    }

    /// <inheritdoc/>
    public override T TensorMedian<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0) throw new ArgumentException("Median requires a non-empty tensor");
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.TensorMedian(input);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int L = input.Length;
            var (vals, _) = BitonicSortRows(backend, (float[])(object)c.GetDataArray(), 1, L, false);
            int k = (L + 1) / 2;   // lower median (torch)
            return (T)(object)vals[k - 1];
        }
        catch (Exception) { return base.TensorMedian(input); }
    }

    /// <inheritdoc/>
    public override (T Value, int Index) TensorKthvalue<T>(Tensor<T> input, int k)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (k < 1 || k > input.Length) throw new ArgumentOutOfRangeException(nameof(k));
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.TensorKthvalue(input, k);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int L = input.Length;
            var (vals, idx) = BitonicSortRows(backend, (float[])(object)c.GetDataArray(), 1, L, false);
            return ((T)(object)vals[k - 1], idx[k - 1]);
        }
        catch (Exception) { return base.TensorKthvalue(input, k); }
    }

    /// <inheritdoc/>
    public override (T Value, int Count) TensorMode<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0) throw new ArgumentException("Mode requires a non-empty tensor");
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.TensorMode(input);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int L = input.Length;
            var (vals, _) = BitonicSortRows(backend, (float[])(object)c.GetDataArray(), 1, L, false);
            // Sorted ascending -> equal values are contiguous; longest run wins, ties -> smallest value
            // (the first run in ascending order). Strict '>' keeps the earliest (smallest-value) run.
            float bestVal = vals[0]; int bestCount = 1, run = 1;
            for (int i = 1; i < L; i++)
            {
                if (vals[i] == vals[i - 1]) run++;
                else { if (run > bestCount) { bestCount = run; bestVal = vals[i - 1]; } run = 1; }
            }
            if (run > bestCount) { bestCount = run; bestVal = vals[L - 1]; }
            return ((T)(object)bestVal, bestCount);
        }
        catch (Exception) { return base.TensorMode(input); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorTakeAlongDim<T>(Tensor<T> tensor, Tensor<int> indices, int dim)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        int rank = tensor.Rank;
        int d = dim < 0 ? dim + rank : dim;
        if (typeof(T) != typeof(float) || d < 0 || d >= rank || indices.Rank != rank || !TryGetBatchBackend(out var backend))
            return base.TensorTakeAlongDim(tensor, indices, dim);
        for (int k = 0; k < rank; k++)
            if (k != d && indices._shape[k] != tensor._shape[k])
                return base.TensorTakeAlongDim(tensor, indices, dim);

        // Gather along axis d on the device with the new TakeAlongDim kernel: view as
        // [outer, axisIn, inner] / [outer, axisOut, inner]. Indices upload as an int buffer.
        try
        {
            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            var ci = indices.IsContiguous ? indices : (Tensor<int>)indices.Contiguous();
            int axisIn = tensor._shape[d];
            int axisOut = indices._shape[d];
            int outerSize = 1; for (int k = 0; k < d; k++) outerSize *= tensor._shape[k];
            int innerSize = 1; for (int k = d + 1; k < rank; k++) innerSize *= tensor._shape[k];
            int outN = outerSize * axisOut * innerSize;

            using var bufIn = GetOrAllocateBuffer(backend, ct.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(ci.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, outN);
            backend.TakeAlongDim(bufIn.Buffer, bufIdx, bufOut.Buffer, outerSize, axisOut, innerSize, axisIn);
            var arr = FinishGpuOp<T>(backend, bufOut, outN);
            return new Tensor<T>(arr, (int[])indices._shape.Clone());
        }
        catch (Exception) { return base.TensorTakeAlongDim(tensor, indices, dim); }
    }

    /// <inheritdoc/>
    public override Tensor<int> TensorSearchSorted<T>(Tensor<T> sortedSequence, Tensor<T> values, bool right = false)
    {
        if (sortedSequence is null) throw new ArgumentNullException(nameof(sortedSequence));
        if (values is null) throw new ArgumentNullException(nameof(values));
        if (typeof(T) != typeof(float) || sortedSequence.Rank != 1 || !TryGetBackend(out var backend))
            return base.TensorSearchSorted(sortedSequence, values, right);
        try
        {
            var seq = sortedSequence.IsContiguous ? sortedSequence : (Tensor<T>)sortedSequence.Contiguous();
            var vals = values.IsContiguous ? values : (Tensor<T>)values.Contiguous();
            int seqLen = sortedSequence.Length;
            int nv = values.Length;
            using var bufSeq = GetOrAllocateBuffer(backend, seq.GetDataArray());
            using var bufVal = GetOrAllocateBuffer(backend, vals.GetDataArray());
            using var bufOut = AllocateOutputBuffer(backend, nv);
            backend.SearchSorted(bufSeq.Buffer, bufVal.Buffer, bufOut.Buffer, seqLen, nv, right ? 1 : 0);
            backend.Synchronize();
            var f = backend.DownloadBuffer(bufOut.Buffer);
            var result = new Tensor<int>((int[])values._shape.Clone());
            var dst = result.AsWritableSpan();
            for (int i = 0; i < nv; i++) dst[i] = (int)f[i];
            return result;
        }
        catch (Exception) { return base.TensorSearchSorted(sortedSequence, values, right); }
    }

    /// <inheritdoc/>
    public override Tensor<T>[] TensorMeshgrid<T>(Tensor<T>[] tensors, string indexing = "ij")
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length == 0) return System.Array.Empty<Tensor<T>>();
        foreach (var t in tensors)
        {
            if (t is null) throw new ArgumentNullException(nameof(tensors));
            if (t.Rank != 1) return base.TensorMeshgrid(tensors, indexing);
        }
        if (indexing != "ij" && indexing != "xy")
            throw new ArgumentException("indexing must be 'ij' or 'xy'");
        if (typeof(T) != typeof(float) || !TryGetBackend(out _))
            return base.TensorMeshgrid(tensors, indexing);

        // Each output grid k = input k reshaped onto its output axis (size n_k, 1s elsewhere) then
        // broadcast to the full grid shape — both GPU-resident ops, so the grids stay on the device.
        try
        {
            int d = tensors.Length;
            var outShape = new int[d];
            for (int i = 0; i < d; i++) outShape[i] = tensors[i]._shape[0];
            bool xy = indexing == "xy" && d >= 2;
            if (xy) { int tmp = outShape[0]; outShape[0] = outShape[1]; outShape[1] = tmp; }

            var result = new Tensor<T>[d];
            for (int k = 0; k < d; k++)
            {
                int axis = k;
                if (xy) { if (k == 0) axis = 1; else if (k == 1) axis = 0; }
                var rshape = new int[d];
                for (int i = 0; i < d; i++) rshape[i] = 1;
                rshape[axis] = tensors[k]._shape[0];
                result[k] = TensorBroadcastTo(tensors[k].Reshape(rshape), outShape);
            }
            return result;
        }
        catch (Exception) { return base.TensorMeshgrid(tensors, indexing); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorKron<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        // GPU path handles the classic 2-D Kronecker product; general N-D defers to base.
        if (typeof(T) != typeof(float) || a.Rank != 2 || b.Rank != 2 || !TryGetBackend(out var backend))
            return base.TensorKron(a, b);
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<T>)a.Contiguous();
            var cb = b.IsContiguous ? b : (Tensor<T>)b.Contiguous();
            int am = a._shape[0], an = a._shape[1], bp = b._shape[0], bq = b._shape[1];
            int outN = am * bp * an * bq;
            using var bufA = GetOrAllocateBuffer(backend, ca.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, cb.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, outN);
            backend.Kron2D(bufA.Buffer, bufB.Buffer, bufOut.Buffer, am, an, bp, bq);
            var arr = FinishGpuOp<T>(backend, bufOut, outN);
            return new Tensor<T>(arr, new[] { am * bp, an * bq });
        }
        catch (Exception) { return base.TensorKron(a, b); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorHistc<T>(Tensor<T> input, int bins, T min, T max)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (typeof(T) != typeof(float) || bins <= 0 || !TryGetBackend(out var backend))
            return base.TensorHistc(input, bins, min, max);
        float mn = ToFloatScalar(min), mx = ToFloatScalar(max);
        // min==max means "use the data's own range" — that needs a reduction first, so defer to base.
        if (!(mn < mx))
            return base.TensorHistc(input, bins, min, max);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int n = input.Length;
            using var bufIn = GetOrAllocateBuffer(backend, c.GetDataArray());
            var bufHist = AllocateOutputBuffer(backend, bins);
            backend.Fill(bufHist.Buffer, 0f, bins);
            backend.Histc(bufIn.Buffer, bufHist.Buffer, n, bins, mn, mx);
            var arr = FinishGpuOp<T>(backend, bufHist, bins);
            return new Tensor<T>(arr, new[] { bins });
        }
        catch (Exception) { return base.TensorHistc(input, bins, min, max); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorPDist<T>(Tensor<T> input, double p = 2.0)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (typeof(T) != typeof(float) || input.Rank != 2 || input._shape[0] < 2 || !TryGetBackend(out var backend))
            return base.TensorPDist(input, p);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int n = input._shape[0], d = input._shape[1];
            int outN = n * (n - 1) / 2;
            using var bufIn = GetOrAllocateBuffer(backend, c.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, outN);
            backend.PDist(bufIn.Buffer, bufOut.Buffer, n, d, (float)p);
            var arr = FinishGpuOp<T>(backend, bufOut, outN);
            return new Tensor<T>(arr, new[] { outN });
        }
        catch (Exception) { return base.TensorPDist(input, p); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorCDist<T>(Tensor<T> x1, Tensor<T> x2, double p = 2.0)
    {
        if (x1 is null) throw new ArgumentNullException(nameof(x1));
        if (x2 is null) throw new ArgumentNullException(nameof(x2));
        if (typeof(T) != typeof(float) || x1.Rank != 2 || x2.Rank != 2
            || x1._shape[1] != x2._shape[1] || !TryGetBackend(out var backend))
            return base.TensorCDist(x1, x2, p);
        try
        {
            var c1 = x1.IsContiguous ? x1 : (Tensor<T>)x1.Contiguous();
            var c2 = x2.IsContiguous ? x2 : (Tensor<T>)x2.Contiguous();
            int m = x1._shape[0], n = x2._shape[0], d = x1._shape[1];
            int outN = m * n;
            using var buf1 = GetOrAllocateBuffer(backend, c1.GetDataArray());
            using var buf2 = GetOrAllocateBuffer(backend, c2.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, outN);
            backend.CDist(buf1.Buffer, buf2.Buffer, bufOut.Buffer, m, n, d, (float)p);
            var arr = FinishGpuOp<T>(backend, bufOut, outN);
            return new Tensor<T>(arr, new[] { m, n });
        }
        catch (Exception) { return base.TensorCDist(x1, x2, p); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSliceScatter<T>(Tensor<T> tensor, Tensor<T> source, int dim, int start, int length)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (source is null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        int d = dim < 0 ? dim + rank : dim;
        if (typeof(T) != typeof(float) || d < 0 || d >= rank || source.Rank != rank
            || source._shape[d] != length || start < 0 || start + length > tensor._shape[d]
            || !TryGetBatchBackend(out var backend))
            return base.TensorSliceScatter(tensor, source, dim, start, length);
        for (int k = 0; k < rank; k++)
            if (k != d && source._shape[k] != tensor._shape[k])
                return base.TensorSliceScatter(tensor, source, dim, start, length);
        try
        {
            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            var cs = source.IsContiguous ? source : (Tensor<T>)source.Contiguous();
            int dstAxis = tensor._shape[d], n = tensor.Length;
            int outerSize = 1; for (int k = 0; k < d; k++) outerSize *= tensor._shape[k];
            int innerSize = 1; for (int k = d + 1; k < rank; k++) innerSize *= tensor._shape[k];
            var idxData = new int[length]; for (int i = 0; i < length; i++) idxData[i] = start + i;

            using var bufIn = GetOrAllocateBuffer(backend, ct.GetDataArray());
            using var bufSrc = GetOrAllocateBuffer(backend, cs.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(idxData);
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Copy(bufIn.Buffer, bufOut.Buffer, n);
            backend.IndexWrite(bufOut.Buffer, bufIdx, bufSrc.Buffer, 0f, 0, outerSize, length, innerSize, dstAxis);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorSliceScatter(tensor, source, dim, start, length); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorSelectScatter<T>(Tensor<T> tensor, Tensor<T> source, int dim, int index)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (source is null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        int d = dim < 0 ? dim + rank : dim;
        int ix = index < 0 ? index + (d >= 0 && d < rank ? tensor._shape[d] : 0) : index;
        if (typeof(T) != typeof(float) || d < 0 || d >= rank || source.Rank != rank - 1
            || ix < 0 || ix >= tensor._shape[d] || !TryGetBatchBackend(out _))
            return base.TensorSelectScatter(tensor, source, dim, index);
        try
        {
            // Insert a size-1 axis at d so the slice-scatter path (length 1) applies.
            var srcShape = (int[])tensor._shape.Clone();
            srcShape[d] = 1;
            var reshaped = source.Reshape(srcShape);
            return TensorSliceScatter(tensor, reshaped, d, ix, 1);
        }
        catch (Exception) { return base.TensorSelectScatter(tensor, source, dim, index); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIndexCopy<T>(Tensor<T> tensor, int axis, Tensor<int> indices, Tensor<T> source)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (source is null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        int ax = axis < 0 ? axis + rank : axis;
        if (typeof(T) != typeof(float) || ax < 0 || ax >= rank || indices.Rank != 1 || !TryGetBatchBackend(out var backend))
            return base.TensorIndexCopy(tensor, axis, indices, source);
        try
        {
            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            var cs = source.IsContiguous ? source : (Tensor<T>)source.Contiguous();
            var ci = indices.IsContiguous ? indices : (Tensor<int>)indices.Contiguous();
            int dstAxis = tensor._shape[ax], idxAxis = indices.Length, n = tensor.Length;
            int outerSize = 1; for (int k = 0; k < ax; k++) outerSize *= tensor._shape[k];
            int innerSize = 1; for (int k = ax + 1; k < rank; k++) innerSize *= tensor._shape[k];

            using var bufIn = GetOrAllocateBuffer(backend, ct.GetDataArray());
            using var bufSrc = GetOrAllocateBuffer(backend, cs.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(ci.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Copy(bufIn.Buffer, bufOut.Buffer, n);   // seed with the original tensor
            backend.IndexWrite(bufOut.Buffer, bufIdx, bufSrc.Buffer, 0f, /*mode copy*/0, outerSize, idxAxis, innerSize, dstAxis);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorIndexCopy(tensor, axis, indices, source); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorIndexFill<T>(Tensor<T> tensor, int axis, Tensor<int> indices, T value)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        int rank = tensor.Rank;
        int ax = axis < 0 ? axis + rank : axis;
        if (typeof(T) != typeof(float) || ax < 0 || ax >= rank || indices.Rank != 1 || !TryGetBatchBackend(out var backend))
            return base.TensorIndexFill(tensor, axis, indices, value);
        try
        {
            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            var ci = indices.IsContiguous ? indices : (Tensor<int>)indices.Contiguous();
            int dstAxis = tensor._shape[ax], idxAxis = indices.Length, n = tensor.Length;
            int outerSize = 1; for (int k = 0; k < ax; k++) outerSize *= tensor._shape[k];
            int innerSize = 1; for (int k = ax + 1; k < rank; k++) innerSize *= tensor._shape[k];
            float fill = ToFloatScalar(value);

            using var bufIn = GetOrAllocateBuffer(backend, ct.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(ci.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Copy(bufIn.Buffer, bufOut.Buffer, n);
            // mode 1 = fill; source unused, pass bufIn as a non-null placeholder.
            backend.IndexWrite(bufOut.Buffer, bufIdx, bufIn.Buffer, fill, /*mode fill*/1, outerSize, idxAxis, innerSize, dstAxis);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorIndexFill(tensor, axis, indices, value); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorNextAfter<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (typeof(T) != typeof(float) || !ShapesEqual(a._shape, b._shape) || !TryGetBackend(out var backend))
            return base.TensorNextAfter(a, b);
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<T>)a.Contiguous();
            var cb = b.IsContiguous ? b : (Tensor<T>)b.Contiguous();
            int n = a.Length;
            using var bufA = GetOrAllocateBuffer(backend, ca.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, cb.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.NextAfter(bufA.Buffer, bufB.Buffer, bufOut.Buffer, n);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])a._shape.Clone());
        }
        catch (Exception) { return base.TensorNextAfter(a, b); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorCross<T>(Tensor<T> a, Tensor<T> b, int dim = -1)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        int rank = a.Rank;
        int d = dim < 0 ? dim + rank : dim;
        if (typeof(T) != typeof(float) || !ShapesEqual(a._shape, b._shape)
            || d < 0 || d >= rank || a._shape[d] != 3 || !TryGetBackend(out var backend))
            return base.TensorCross(a, b, dim);
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<T>)a.Contiguous();
            var cb = b.IsContiguous ? b : (Tensor<T>)b.Contiguous();
            int outerSize = 1; for (int k = 0; k < d; k++) outerSize *= a._shape[k];
            int innerSize = 1; for (int k = d + 1; k < rank; k++) innerSize *= a._shape[k];
            int n = a.Length;
            using var bufA = GetOrAllocateBuffer(backend, ca.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, cb.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Cross3(bufA.Buffer, bufB.Buffer, bufOut.Buffer, outerSize, innerSize);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])a._shape.Clone());
        }
        catch (Exception) { return base.TensorCross(a, b, dim); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorLdexp<T>(Tensor<T> x, Tensor<int> exp)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (exp is null) throw new ArgumentNullException(nameof(exp));
        if (typeof(T) != typeof(float) || !ShapesEqual(x._shape, exp._shape) || !TryGetBackend(out var backend))
            return base.TensorLdexp(x, exp);
        try
        {
            var cx = x.IsContiguous ? x : (Tensor<T>)x.Contiguous();
            var ce = exp.IsContiguous ? exp : (Tensor<int>)exp.Contiguous();
            int n = x.Length;
            using var bufX = GetOrAllocateBuffer(backend, cx.GetDataArray());
            using var bufE = backend.AllocateIntBuffer(ce.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Ldexp(bufX.Buffer, bufE, bufOut.Buffer, n);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])x._shape.Clone());
        }
        catch (Exception) { return base.TensorLdexp(x, exp); }
    }

    // ----- Category D: Tensor<Bit> comparison predicates -----
    // The backend comparison kernels write a FLOAT mask (1.0/0.0). The result is a Tensor<Bit> (one
    // byte/elem), which leaves the device — but computing the compare on the GPU keeps the (often
    // resident) input tensors on the device instead of forcing the base to download them to scan on
    // the host. We download only the mask and pack it into Bit. `invert` flips the predicate (used for
    // Eq-via-NotEqual). Float-only. Caller disposes maskBuf.
    private static Tensor<Bit> PackMask(IDirectGpuBackend backend, OwnedBuffer maskBuf, int[] shape, int n, bool invert)
    {
        var span = backend.DownloadBuffer(maskBuf.Buffer);
        var result = new Tensor<Bit>((int[])shape.Clone());
        var dst = result.AsWritableSpan();
        for (int i = 0; i < n; i++)
        {
            bool set = span[i] != 0f;
            dst[i] = (set ^ invert) ? Bit.True : Bit.False;
        }
        return result;
    }

    /// <inheritdoc/>
    public override Tensor<Bit> TensorEq<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (typeof(T) != typeof(float) || !ShapesEqual(a._shape, b._shape) || !TryGetBackend(out var backend))
            return base.TensorEq(a, b);
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<T>)a.Contiguous();
            var cb = b.IsContiguous ? b : (Tensor<T>)b.Contiguous();
            int n = a.Length;
            using var bufA = GetOrAllocateBuffer(backend, ca.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, cb.GetDataArray());
            using var bufOut = AllocateOutputBuffer(backend, n);
            backend.Equal(bufA.Buffer, bufB.Buffer, bufOut.Buffer, n);
            backend.Synchronize();
            return PackMask(backend, bufOut, a._shape, n, invert: false);
        }
        catch (Exception) { return base.TensorEq(a, b); }
    }

    /// <inheritdoc/>
    public override Tensor<Bit> TensorEqScalar<T>(Tensor<T> a, T scalar)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.TensorEqScalar(a, scalar);
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<T>)a.Contiguous();
            int n = a.Length;
            float s = ToFloatScalar(scalar);
            using var bufA = GetOrAllocateBuffer(backend, ca.GetDataArray());
            using var bufOut = AllocateOutputBuffer(backend, n);
            // NotEqualScalar gives (a != s); invert to get equality.
            backend.NotEqualScalar(bufA.Buffer, bufOut.Buffer, s, n);
            backend.Synchronize();
            return PackMask(backend, bufOut, a._shape, n, invert: true);
        }
        catch (Exception) { return base.TensorEqScalar(a, scalar); }
    }

    // IEEE classification predicates via the new ClassifyFloat backend kernel (bit-pattern based, so it
    // survives the GPU compilers' fast/finite-math that folds isnan/x!=x away). Backends without the
    // kernel yet throw NotSupportedException, which the try/catch turns into the correct CPU fallback.
    /// <inheritdoc/>
    public override Tensor<Bit> TensorIsNan<T>(Tensor<T> tensor) => ClassifyToBit(tensor, 0, base.TensorIsNan);
    /// <inheritdoc/>
    public override Tensor<Bit> TensorIsInf<T>(Tensor<T> tensor) => ClassifyToBit(tensor, 1, base.TensorIsInf);
    /// <inheritdoc/>
    public override Tensor<Bit> TensorIsFinite<T>(Tensor<T> tensor) => ClassifyToBit(tensor, 2, base.TensorIsFinite);

    /// <inheritdoc/>
    public override Tensor<T> TensorClampTensor<T>(Tensor<T> tensor, Tensor<T>? min, Tensor<T>? max)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (min is null && max is null)
            throw new ArgumentException("At least one of min / max must be supplied");

        // clamp(x, min, max) = min(max(x, min), max), elementwise via the GPU binary Max/Min. Only the
        // same-shape (no-broadcast) bound case maps cleanly; broadcast bounds / dtype / no-GPU defer.
        if (typeof(T) != typeof(float) || !TryGetBackend(out _)
            || (min is not null && !ShapesEqual(min._shape, tensor._shape))
            || (max is not null && !ShapesEqual(max._shape, tensor._shape)))
            return base.TensorClampTensor(tensor, min, max);

        try
        {
            var cur = tensor;
            if (min is not null) cur = TensorMax(cur, min);   // lower bound: max(x, min)
            if (max is not null) cur = TensorMin(cur, max);   // upper bound: min(x, max)
            return cur;
        }
        catch (Exception) { return base.TensorClampTensor(tensor, min, max); }
    }

    /// <inheritdoc/>
    public override bool TensorEqual<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (typeof(T) != typeof(float) || !ShapesEqual(a._shape, b._shape) || !TryGetBackend(out var backend))
            return base.TensorEqual(a, b);

        // all-equal iff min over the equality mask == 1. Compute the mask on the GPU (Equal), then
        // min = -max(-mask) via Negate+MaxAxis; only the scalar comes back (the base downloaded both
        // tensors to compare on the host).
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<T>)a.Contiguous();
            var cb = b.IsContiguous ? b : (Tensor<T>)b.Contiguous();
            int n = a.Length;
            using var bufA = GetOrAllocateBuffer(backend, ca.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, cb.GetDataArray());
            using var bufMask = AllocateOutputBuffer(backend, n);
            using var bufNeg = AllocateOutputBuffer(backend, n);
            using var bufMin = AllocateOutputBuffer(backend, 1);
            backend.Equal(bufA.Buffer, bufB.Buffer, bufMask.Buffer, n);   // 1.0 where equal
            backend.Negate(bufMask.Buffer, bufNeg.Buffer, n);
            backend.MaxAxis(bufNeg.Buffer, bufMin.Buffer, 1, n);          // = -min(mask)
            backend.Synchronize();
            float minMask = -backend.DownloadBuffer(bufMin.Buffer)[0];
            return minMask > 0.5f;   // mask is 0/1; min==1 iff every element matched
        }
        catch (Exception) { return base.TensorEqual(a, b); }
    }

    /// <inheritdoc/>
    public override Tensor<Bit> TensorIsClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        // equalNan handling needs an extra isnan path; defer it (and non-float / shape-mismatch) to base.
        if (typeof(T) != typeof(float) || equalNan || !ShapesEqual(a._shape, b._shape) || !TryGetBackend(out var backend))
            return base.TensorIsClose(a, b, rtol, atol, equalNan);
        try
        {
            using var maskBuf = AllocCloseMask(backend, a, b, rtol, atol, out int n);
            return PackMask(backend, maskBuf, a._shape, n, invert: false);
        }
        catch (Exception) { return base.TensorIsClose(a, b, rtol, atol, equalNan); }
    }

    /// <inheritdoc/>
    public override bool TensorAllClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (typeof(T) != typeof(float) || equalNan || !ShapesEqual(a._shape, b._shape) || !TryGetBackend(out var backend))
            return base.TensorAllClose(a, b, rtol, atol, equalNan);
        try
        {
            using var maskBuf = AllocCloseMask(backend, a, b, rtol, atol, out int n);
            // allclose iff EVERY element is close (mask all 1) iff min(mask) == 1.  min = -max(-mask).
            using var bufNeg = AllocateOutputBuffer(backend, n);
            using var bufMin = AllocateOutputBuffer(backend, 1);
            backend.Negate(maskBuf.Buffer, bufNeg.Buffer, n);
            backend.MaxAxis(bufNeg.Buffer, bufMin.Buffer, 1, n);
            backend.Synchronize();
            float minMask = -backend.DownloadBuffer(bufMin.Buffer)[0];
            return minMask > 0.5f;
        }
        catch (Exception) { return base.TensorAllClose(a, b, rtol, atol, equalNan); }
    }

    // Builds the "close" mask on the GPU: 1.0 where |a-b| < atol + rtol*|b|, else 0.0. Uses LessThan
    // (strict) on purpose — it correctly yields "not close" for NaN (NaN < x is false), whereas the
    // GreaterThan-then-invert form would wrongly call NaN close. The < vs <= boundary (|a-b| EXACTLY
    // == threshold) is measure-zero for floats and benign. Caller disposes the returned buffer.
    private OwnedBuffer AllocCloseMask<T>(IDirectGpuBackend backend, Tensor<T> a, Tensor<T> b, T rtol, T atol, out int n)
    {
        n = a.Length;
        var diff = TensorAbs(TensorSubtract(a, b));                                  // |a-b|, GPU
        var thr = TensorAddScalar(TensorMultiplyScalar(TensorAbs(b), rtol), atol);   // atol + rtol*|b|, GPU
        using var diffBuf = GetOrAllocateBuffer(backend, diff);
        using var thrBuf = GetOrAllocateBuffer(backend, thr);
        var maskBuf = AllocateOutputBuffer(backend, n);
        backend.LessThan(diffBuf.Buffer, thrBuf.Buffer, maskBuf.Buffer, n);
        backend.Synchronize();
        return maskBuf;
    }

    private Tensor<Bit> ClassifyToBit<T>(Tensor<T> tensor, int mode, System.Func<Tensor<T>, Tensor<Bit>> fallback)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return fallback(tensor);
        try
        {
            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            int n = tensor.Length;
            using var bufA = GetOrAllocateBuffer(backend, ct.GetDataArray());
            using var bufOut = AllocateOutputBuffer(backend, n);
            backend.ClassifyFloat(bufA.Buffer, bufOut.Buffer, mode, n);
            backend.Synchronize();
            return PackMask(backend, bufOut, tensor._shape, n, invert: false);
        }
        catch (Exception) { return fallback(tensor); }
    }

    // ----- Category B: composite forwards -----

    /// <inheritdoc/>
    public override Tensor<T> MlpForward<T>(
        Tensor<T> input,
        System.Collections.Generic.IReadOnlyList<Tensor<T>> weights,
        System.Collections.Generic.IReadOnlyList<Tensor<T>?> biases,
        FusedActivationType hiddenActivation,
        FusedActivationType outputActivation = FusedActivationType.None,
        FusedActivationParams? hiddenActivationParams = null,
        FusedActivationParams? outputActivationParams = null)
    {
        // The CPU base runs a native-BLAS GEMM chain on the host (downloads input, the whole MLP
        // forward computes on CPU). Decompose into per-layer GPU FusedLinear instead so every layer's
        // GEMM+bias+activation runs on the device and the intermediates stay resident between layers
        // (safe now that strided/chained buffers are handled). Inference-only contract is unchanged:
        // GraphMode disables the GPU backend, so we fall back to base (which throws for tape use).
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (biases is null) throw new ArgumentNullException(nameof(biases));
        if (weights.Count == 0 || biases.Count != weights.Count || !TryGetBackend(out _))
        {
            return base.MlpForward(input, weights, biases, hiddenActivation, outputActivation,
                hiddenActivationParams, outputActivationParams);
        }

        try
        {
            int last = weights.Count - 1;
            var cur = input;
            for (int i = 0; i < weights.Count; i++)
            {
                var act = (i == last) ? outputActivation : hiddenActivation;
                var ap = (i == last) ? outputActivationParams : hiddenActivationParams;
                cur = FusedLinear(cur, weights[i], biases[i], act, ap);   // GPU-resident per layer
            }
            return cur;
        }
        catch (Exception)
        {
            return base.MlpForward(input, weights, biases, hiddenActivation, outputActivation,
                hiddenActivationParams, outputActivationParams);
        }
    }

    // Materializing GPU permute: unlike the public TensorPermute (a metadata-only STRIDED VIEW that
    // shares the input's buffer — whose cached contiguous data does NOT reflect the permuted strides,
    // the source of the deep interleaved-chain divergence), this runs the backend Permute kernel into a
    // fresh CONTIGUOUS device buffer and returns a deferred GPU-resident result. Use this for multi-step
    // GPU compositions so each stage hands the next a correctly-laid-out resident buffer.
    /// <inheritdoc/>
    public override Tensor<T> TensorBroadcastTo<T>(Tensor<T> input, int[] targetShape)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (targetShape is null) throw new ArgumentNullException(nameof(targetShape));

        // Tape-active / no-GPU / rank-shrink defer to base (which records BroadcastToBackward and
        // validates). The base also keeps Tier 1 (identity) and Tier 2 (pure size-1 prepend == Reshape)
        // already GPU-resident, so we only need the genuine Tier-3 size-1→N expansion here.
        if (IsTapeActive<T>() || targetShape.Length < input.Rank || !TryGetBatchBackend(out var backend))
            return base.TensorBroadcastTo(input, targetShape);

        try
        {
            // Align to the target rank by prepending size-1 axes (a residency-preserving reshape).
            Tensor<T> cur = input;
            if (targetShape.Length > input.Rank)
            {
                var aligned = new int[targetShape.Length];
                int rd = targetShape.Length - input.Rank;
                for (int i = 0; i < rd; i++) aligned[i] = 1;
                for (int i = 0; i < input.Rank; i++) aligned[rd + i] = input._shape[i];
                cur = cur.Reshape(aligned);
            }

            // Expand each broadcast axis (size 1 → target) with the GPU TileAxis kernel. With the
            // strided-view materialize fix in GetOrAllocateBuffer, this multi-step chain stays correct
            // and device-resident. A genuinely incompatible axis falls back to base (which throws).
            for (int d = 0; d < targetShape.Length; d++)
            {
                if (cur._shape[d] == targetShape[d]) continue;
                if (cur._shape[d] != 1) return base.TensorBroadcastTo(input, targetShape);

                int repeats = targetShape[d];
                int outerSize = 1; for (int i = 0; i < d; i++) outerSize *= cur._shape[i];
                int innerSize = 1; for (int i = d + 1; i < cur.Rank; i++) innerSize *= cur._shape[i];

                var src = cur.IsContiguous ? cur : (Tensor<T>)cur.Contiguous();
                using var bufIn = GetOrAllocateBuffer(backend, src.GetDataArray());
                var bufOut = AllocateOutputBuffer(backend, src.Length * repeats);
                backend.TileAxis(bufIn.Buffer, bufOut.Buffer, outerSize, /*axisSize:*/ 1, innerSize, repeats);
                var arr = FinishGpuOp<T>(backend, bufOut, src.Length * repeats);
                var newShape = (int[])cur._shape.Clone();
                newShape[d] = repeats;
                cur = new Tensor<T>(arr, newShape);
            }
            return cur;
        }
        catch (Exception) { return base.TensorBroadcastTo(input, targetShape); }
    }

    // ----- Category E: full min/max reduction -----

    /// <inheritdoc/>
    public override (T Min, T Max) TensorAminmax<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        // float-only (the backend reduce kernels are float); other dtypes / no-GPU defer to base.
        if (typeof(T) != typeof(float) || tensor.Length == 0 || !TryGetBackend(out var backend))
            return base.TensorAminmax(tensor);

        // max via MaxAxis over the whole tensor (outerSize=1); min via -max(-x) using Negate.
        // The O(n) reduction runs on the GPU; only the two scalars come back, instead of the base
        // downloading the entire tensor to scan on the host.
        try
        {
            var src = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            int n = tensor.Length;
            using var bufIn = GetOrAllocateBuffer(backend, src.GetDataArray());
            using var bufMax = AllocateOutputBuffer(backend, 1);
            using var bufNeg = AllocateOutputBuffer(backend, n);
            using var bufNegMax = AllocateOutputBuffer(backend, 1);

            backend.MaxAxis(bufIn.Buffer, bufMax.Buffer, 1, n);
            backend.Negate(bufIn.Buffer, bufNeg.Buffer, n);
            backend.MaxAxis(bufNeg.Buffer, bufNegMax.Buffer, 1, n);
            backend.Synchronize();

            float maxV = backend.DownloadBuffer(bufMax.Buffer)[0];
            float minV = -backend.DownloadBuffer(bufNegMax.Buffer)[0];
            return ((T)(object)minV, (T)(object)maxV);
        }
        catch (Exception) { return base.TensorAminmax(tensor); }
    }

    /// <inheritdoc/>
    public override T TensorTrace<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (typeof(T) != typeof(float) || tensor.Rank != 2 || !TryGetBatchBackend(out var backend))
            return base.TensorTrace(tensor);

        // trace = sum of the diagonal. Extract the diagonal with the GPU kernel, then GPU reduce-sum;
        // only the scalar comes back instead of the host scanning the whole matrix.
        try
        {
            int rows = tensor._shape[0], cols = tensor._shape[1];
            int n = System.Math.Min(rows, cols);
            var src = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            using var bufIn = GetOrAllocateBuffer(backend, src.GetDataArray());
            var bufDiag = AllocateOutputBuffer(backend, n);
            backend.ExtractDiagKernel(bufIn.Buffer, bufDiag.Buffer, n, cols);
            var diagArr = FinishGpuOp<T>(backend, bufDiag, n);
            var diag = new Tensor<T>(diagArr, new[] { n });
            var sum = ReduceSum(diag, null, false);
            return sum.AsSpan()[0];
        }
        catch (Exception) { return base.TensorTrace(tensor); }
    }

    private Tensor<T> PermuteResidentGpu<T>(IDirectGpuBackend backend, Tensor<T> tensor, int[] axes)
    {
        var src = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
        using var bufIn = GetOrAllocateBuffer(backend, src.GetDataArray());
        var bufOut = AllocateOutputBuffer(backend, tensor.Length);
        backend.Permute(bufIn.Buffer, bufOut.Buffer, src.Shape._dims, axes);
        var result = FinishGpuOp<T>(backend, bufOut, tensor.Length);
        var outShape = new int[tensor.Rank];
        for (int i = 0; i < tensor.Rank; i++) outShape[i] = src.Shape._dims[axes[i]];
        return new Tensor<T>(result, outShape);
    }
}
