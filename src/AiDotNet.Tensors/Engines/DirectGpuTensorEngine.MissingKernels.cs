using System.Collections.Generic;
using AiDotNet.Tensors;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Helpers;
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

        // index_add: result = copy(destination); result[index[i]] += updates[i]. The backend
        // ScatterAdd kernel is FLAT (featureSize=1), so only the 1-D / axis-0 case maps;
        // higher-rank / featured / non-zero-axis fall back to the (correct) base. Tape-active
        // also defers (the scatter backward stays on the base's recording).
        //
        // The wrapper SEEDS the output buffer with the base destination via backend.Copy
        // (scatter is ADD onto existing values), then the kernel atomic-adds the indexed
        // updates. CudaBackend.ScatterAdd is hard-routed to the atomic embedding_backward
        // kernel (NOT the deterministic-overwrite variant) because the latter overwrites
        // the seed — see the routing comment in CudaBackend.ScatterAdd.
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

    /// <inheritdoc/>
    public override Tensor<Bit> TensorIsIn<T>(Tensor<T> elements, Tensor<T> testElements, bool invert = false)
    {
        if (elements is null) throw new ArgumentNullException(nameof(elements));
        if (testElements is null) throw new ArgumentNullException(nameof(testElements));
        if (typeof(T) != typeof(float) || elements.Length == 0 || testElements.Length == 0 || !TryGetBackend(out var backend))
            return base.TensorIsIn(elements, testElements, invert);
        try
        {
            var ce = elements.IsContiguous ? elements : (Tensor<T>)elements.Contiguous();
            var ct = testElements.IsContiguous ? testElements : (Tensor<T>)testElements.Contiguous();
            int testLen = testElements.Length, n = elements.Length;
            var (sortedTest, _) = BitonicSortRows(backend, (float[])(object)ct.GetDataArray(), 1, testLen, false);
            using var bufElem = GetOrAllocateBuffer(backend, ce.GetDataArray());
            using var bufTest = GetOrAllocateBuffer(backend, (T[])(object)sortedTest);
            var bufMask = AllocateOutputBuffer(backend, n);
            backend.IsIn(bufElem.Buffer, bufTest.Buffer, bufMask.Buffer, n, testLen);
            return PackMaskResident(backend, bufMask, elements._shape, n, invert);
        }
        catch (Exception) { return base.TensorIsIn(elements, testElements, invert); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorUnfold<T>(Tensor<T> tensor, int dim, int size, int step)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "size must be positive");
        if (step <= 0) throw new ArgumentOutOfRangeException(nameof(step), "step must be positive");
        int rank = tensor.Rank;
        int d = dim < 0 ? dim + rank : dim;
        if (typeof(T) != typeof(float) || d < 0 || d >= rank || size > tensor._shape[d] || !TryGetBackend(out var backend))
            return base.TensorUnfold(tensor, dim, size, step);
        try
        {
            int dimSize = tensor._shape[d];
            int nWindows = (dimSize - size) / step + 1;
            int outerSize = 1; for (int k = 0; k < d; k++) outerSize *= tensor._shape[k];
            int innerSize = 1; for (int k = d + 1; k < rank; k++) innerSize *= tensor._shape[k];
            int outN = outerSize * nWindows * innerSize * size;
            var c = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            using var bufSrc = GetOrAllocateBuffer(backend, c.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, outN);
            backend.Unfold(bufSrc.Buffer, bufOut.Buffer, outerSize, dimSize, innerSize, nWindows, size, step);
            var arr = FinishGpuOp<T>(backend, bufOut, outN);
            var outShape = new int[rank + 1];
            for (int i = 0; i < rank; i++) outShape[i] = tensor._shape[i];
            outShape[d] = nWindows; outShape[rank] = size;
            return new Tensor<T>(arr, outShape);
        }
        catch (Exception) { return base.TensorUnfold(tensor, dim, size, step); }
    }

    // ----- Misc reductions / construction -----

    /// <inheritdoc/>
    public override int TensorCountNonzero<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (typeof(T) != typeof(float) || tensor.Length == 0 || !TryGetBackend(out var backend))
            return base.TensorCountNonzero(tensor);
        try
        {
            var c = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            int n = tensor.Length;
            using var bufX = GetOrAllocateBuffer(backend, c.GetDataArray());
            var bufMask = AllocateOutputBuffer(backend, n);
            backend.NotEqualScalar(bufX.Buffer, bufMask.Buffer, 0f, n);   // 1.0 where x != 0
            var maskArr = FinishGpuOp<T>(backend, bufMask, n);
            var sum = ReduceSum(new Tensor<T>(maskArr, new[] { n }), null, false);
            var sArr = (float[])(object)sum.GetDataArray();
            return (int)System.Math.Round(sArr[0]);
        }
        catch (Exception) { return base.TensorCountNonzero(tensor); }
    }

    /// <inheritdoc/>
    public override T TensorNanMedian<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0) throw new ArgumentException("NanMedian requires a non-empty tensor");
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.TensorNanMedian(input);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int L = input.Length;
            var (vals, _) = BitonicSortRows(backend, (float[])(object)c.GetDataArray(), 1, L, false);
            // Ascending with NaN sorted to the end: finite values fill [0, numFinite).
            int numFinite = 0; while (numFinite < L && !float.IsNaN(vals[numFinite])) numFinite++;
            if (numFinite == 0) return (T)(object)float.NaN;
            int k = (numFinite + 1) / 2;   // lower median of the finite values
            return (T)(object)vals[k - 1];
        }
        catch (Exception) { return base.TensorNanMedian(input); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorBlockDiag<T>(Tensor<T>[] matrices)
    {
        if (matrices is null) throw new ArgumentNullException(nameof(matrices));
        if (matrices.Length == 0) throw new ArgumentException("BlockDiag requires at least one matrix");
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.TensorBlockDiag(matrices);
        foreach (var m in matrices)
        {
            if (m is null) throw new ArgumentNullException(nameof(matrices));
            if (m.Rank != 2) return base.TensorBlockDiag(matrices);
        }
        int totalRows = 0, totalCols = 0;
        foreach (var m in matrices) { totalRows += m._shape[0]; totalCols += m._shape[1]; }
        try
        {
            int outN = totalRows * totalCols;
            var bufOut = AllocateOutputBuffer(backend, outN);
            backend.Fill(bufOut.Buffer, 0f, outN);
            var blockBufs = new List<OwnedBuffer>();
            int rowOff = 0, colOff = 0;
            foreach (var m in matrices)
            {
                int r = m._shape[0], c = m._shape[1];
                var cm = m.IsContiguous ? m : (Tensor<T>)m.Contiguous();
                var bufBlock = GetOrAllocateBuffer(backend, cm.GetDataArray());
                blockBufs.Add(bufBlock);
                backend.CopyBlock2D(bufBlock.Buffer, bufOut.Buffer, r, c, totalCols, rowOff, colOff);
                rowOff += r; colOff += c;
            }
            var arr = FinishGpuOp<T>(backend, bufOut, outN);
            foreach (var b in blockBufs) b.Dispose();
            return new Tensor<T>(arr, new[] { totalRows, totalCols });
        }
        catch (Exception) { return base.TensorBlockDiag(matrices); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorCartesianProd<T>(Tensor<T>[] tensors)
    {
        if (tensors is null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length < 2 || typeof(T) != typeof(float) || !TryGetBackend(out _))
            return base.TensorCartesianProd(tensors);
        foreach (var t in tensors)
        {
            if (t is null) throw new ArgumentNullException(nameof(tensors));
            if (t.Rank != 1) return base.TensorCartesianProd(tensors);
        }
        try
        {
            // ij-meshgrid, give each grid a trailing size-1 axis, concatenate along it, then flatten to
            // [prod, numInputs] — all GPU-resident (Meshgrid + Reshape + Concatenate).
            int m = tensors.Length;
            var grids = TensorMeshgrid(tensors, "ij");
            var gshape = grids[0]._shape;
            var withOne = new int[gshape.Length + 1];
            for (int i = 0; i < gshape.Length; i++) withOne[i] = gshape[i];
            withOne[gshape.Length] = 1;
            var reshaped = new Tensor<T>[m];
            for (int k = 0; k < m; k++) reshaped[k] = grids[k].Reshape(withOne);
            var stacked = TensorConcatenate(reshaped, gshape.Length);
            int prod = 1; foreach (var t in tensors) prod *= t._shape[0];
            return stacked.Reshape(new[] { prod, m });
        }
        catch (Exception) { return base.TensorCartesianProd(tensors); }
    }

    /// <inheritdoc/>
    public override Tensor<int> TensorHistogram<T>(Tensor<T> input, int bins, T min, T max)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (typeof(T) != typeof(float) || bins <= 0 || !TryGetBackend(out var backend))
            return base.TensorHistogram(input, bins, min, max);
        float mn = ToFloatScalar(min), mx = ToFloatScalar(max);
        if (!(mn < mx)) return base.TensorHistogram(input, bins, min, max);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int n = input.Length;
            using var bufIn = GetOrAllocateBuffer(backend, c.GetDataArray());
            using var bufHist = AllocateOutputBuffer(backend, bins);
            backend.Fill(bufHist.Buffer, 0f, bins);
            backend.Histc(bufIn.Buffer, bufHist.Buffer, n, bins, mn, mx);
            backend.Synchronize();
            var f = backend.DownloadBuffer(bufHist.Buffer);
            var res = new Tensor<int>(new[] { bins });
            var dst = res.AsWritableSpan();
            for (int i = 0; i < bins; i++) dst[i] = (int)f[i];
            return res;
        }
        catch (Exception) { return base.TensorHistogram(input, bins, min, max); }
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
    // Sorts numRows rows of length L on the GPU and returns the un-padded sorted VALUE + INDEX buffers
    // (index buffer holds the original positions as float; caller owns both — NOT downloaded). This is
    // the resident core; download wrappers and FinishGpuOp callers build on it.
    private (OwnedBuffer outVal, OwnedBuffer outIdx) BitonicSortRowsToBuffers(IDirectGpuBackend backend, float[] data, int numRows, int L, bool descending)
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
        var outVal = AllocateOutputBuffer(backend, numRows * L);
        var outIdx = AllocateOutputBuffer(backend, numRows * L);
        backend.CopyRows(valBuf.Buffer, outVal.Buffer, P, L, numRows, L);   // un-pad P -> L (drop tail)
        backend.CopyRows(idxBuf.Buffer, outIdx.Buffer, P, L, numRows, L);
        return (outVal, outIdx);
    }

    private (float[] vals, int[] idx) BitonicSortRows(IDirectGpuBackend backend, float[] data, int numRows, int L, bool descending)
    {
        var (outVal, outIdx) = BitonicSortRowsToBuffers(backend, data, numRows, L, descending);
        using (outVal)
        using (outIdx)
        {
            backend.Synchronize();
            var vals = backend.DownloadBuffer(outVal.Buffer);
            var idxF = backend.DownloadBuffer(outIdx.Buffer);
            var idx = new int[numRows * L];
            for (int i = 0; i < idx.Length; i++) idx[i] = (int)idxF[i];
            return (vals, idx);
        }
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
            // Resident: keep the sorted values AND the int indices on the GPU (FinishGpuOp). The index
            // tensor rides as a float buffer (int-as-float); downstream TakeAlongDim/gather reuse it in
            // place (no re-upload) — completing the argsort -> gather int-residency chain.
            var (outVal, outIdx) = BitonicSortRowsToBuffers(backend, data, numRows, L, descending);
            var valuesArr = FinishGpuOp<T>(backend, outVal, numRows * L);
            var indicesArr = FinishGpuOp<int>(backend, outIdx, numRows * L);
            var valuesT = new Tensor<T>(valuesArr, (int[])input._shape.Clone());
            var indicesT = new Tensor<int>(indicesArr, (int[])input._shape.Clone());
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
            // Indices ride as a FLOAT buffer (int values, cast in-kernel) so a RESIDENT int-index tensor
            // (e.g. from Argsort) is reused in place instead of re-uploaded — the int-residency chain.
            using var bufIdx = GetOrAllocateBuffer(backend, ci);
            var bufOut = AllocateOutputBuffer(backend, outN);
            backend.TakeAlongDim(bufIn.Buffer, bufIdx.Buffer, bufOut.Buffer, outerSize, axisOut, innerSize, axisIn);
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
            // CPU reference (CpuEngine.Parity210.PNormCross) accumulates in `double`
            // and only narrows on the final cast. The CUDA `parity210_cdist` kernel
            // does the same (sum is `double`, narrowed on output) so this override
            // is GPU/CPU bit-equivalent within fp32 sqrt rounding.
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
    public override Tensor<T> TensorPut<T>(Tensor<T> tensor, Tensor<int> indices, Tensor<T> source)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (indices.Length != source.Length)
            throw new ArgumentException("indices and source must have the same element count");
        if (typeof(T) != typeof(float) || !TryGetBatchBackend(out var backend))
            return base.TensorPut(tensor, indices, source);
        try
        {
            // Flat scatter: output[indices[k]] = source[k] (IndexWrite over the flattened tensor).
            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            var cs = source.IsContiguous ? source : (Tensor<T>)source.Contiguous();
            var ci = indices.IsContiguous ? indices : (Tensor<int>)indices.Contiguous();
            int n = tensor.Length, count = indices.Length;
            using var bufIn = GetOrAllocateBuffer(backend, ct.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Copy(bufIn.Buffer, bufOut.Buffer, n);
            if (count > 0)
            {
                using var bufSrc = GetOrAllocateBuffer(backend, cs.GetDataArray());
                using var bufIdx = backend.AllocateIntBuffer(ci.GetDataArray());
                backend.IndexWrite(bufOut.Buffer, bufIdx, bufSrc.Buffer, 0f, 0, 1, count, 1, n);
            }
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorPut(tensor, indices, source); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorMaskedScatter<T>(Tensor<T> tensor, Tensor<Bit> mask, Tensor<T> source)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (mask is null) throw new ArgumentNullException(nameof(mask));
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (typeof(T) != typeof(float) || !ShapesEqual(tensor._shape, mask._shape) || !TryGetBatchBackend(out var backend))
            return base.TensorMaskedScatter(tensor, mask, source);
        try
        {
            // Sequentially fill the masked (flat) positions from source[0..count) — IndexWrite over the
            // flattened tensor with the CPU-built masked index list.
            var cm = mask.IsContiguous ? mask : (Tensor<Bit>)mask.Contiguous();
            var mspan = cm.GetDataArray();
            int n = tensor.Length;
            var masked = new List<int>();
            for (int i = 0; i < n; i++) if (mspan[i]) masked.Add(i);
            int count = masked.Count;
            if (count > source.Length) return base.TensorMaskedScatter(tensor, mask, source);

            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            using var bufIn = GetOrAllocateBuffer(backend, ct.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Copy(bufIn.Buffer, bufOut.Buffer, n);
            if (count > 0)
            {
                var cs = source.IsContiguous ? source : (Tensor<T>)source.Contiguous();
                using var bufSrc = GetOrAllocateBuffer(backend, cs.GetDataArray());
                using var bufIdx = backend.AllocateIntBuffer(masked.ToArray());
                backend.IndexWrite(bufOut.Buffer, bufIdx, bufSrc.Buffer, 0f, 0, 1, count, 1, n);
            }
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorMaskedScatter(tensor, mask, source); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorScatterReduce<T>(
        Tensor<T> tensor, int dim, Tensor<int> indices, Tensor<T> source, ScatterReduceMode mode, bool includeSelf = true)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (source is null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        int d = dim < 0 ? dim + rank : dim;
        // kernel modes: 0=sum,1=prod,2=amax,3=amin. Mean / includeSelf=false defer to base.
        int kmode = mode switch
        {
            ScatterReduceMode.Sum => 0,
            ScatterReduceMode.Prod => 1,
            ScatterReduceMode.AMax => 2,
            ScatterReduceMode.AMin => 3,
            _ => -1,
        };
        if (typeof(T) != typeof(float) || !includeSelf || kmode < 0 || d < 0 || d >= rank
            || indices.Rank != rank || !ShapesEqual(indices._shape, source._shape) || !TryGetBatchBackend(out var backend))
            return base.TensorScatterReduce(tensor, dim, indices, source, mode, includeSelf);
        for (int k = 0; k < rank; k++)
            if (k != d && source._shape[k] != tensor._shape[k])
                return base.TensorScatterReduce(tensor, dim, indices, source, mode, includeSelf);
        try
        {
            int dstDim = tensor._shape[d], srcDim = source._shape[d], n = tensor.Length;
            int outerSize = 1; for (int k = 0; k < d; k++) outerSize *= tensor._shape[k];
            int innerSize = 1; for (int k = d + 1; k < rank; k++) innerSize *= tensor._shape[k];
            var ct = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            var cs = source.IsContiguous ? source : (Tensor<T>)source.Contiguous();
            var ci = indices.IsContiguous ? indices : (Tensor<int>)indices.Contiguous();
            using var bufIn = GetOrAllocateBuffer(backend, ct.GetDataArray());
            using var bufSrc = GetOrAllocateBuffer(backend, cs.GetDataArray());
            using var bufIdx = backend.AllocateIntBuffer(ci.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Copy(bufIn.Buffer, bufOut.Buffer, n);   // seed with the original tensor (includeSelf)
            backend.ScatterReduce(bufOut.Buffer, bufSrc.Buffer, bufIdx, outerSize, srcDim, dstDim, innerSize, kmode);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorScatterReduce(tensor, dim, indices, source, mode, includeSelf); }
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
    public override Tensor<int> Nms<T>(Tensor<T> boxes, Tensor<T> scores, double iouThreshold)
    {
        if (boxes is null) throw new ArgumentNullException(nameof(boxes));
        if (scores is null) throw new ArgumentNullException(nameof(scores));
        if (scores.Rank != 1 || scores._shape[0] != boxes._shape[0])
            throw new ArgumentException("scores must be rank-1 with length N.");
        int N = boxes._shape[0];
        if (typeof(T) != typeof(float) || boxes.Rank != 2 || boxes._shape[1] != 4 || N == 0 || !TryGetBackend(out var backend))
            return base.Nms(boxes, scores, iouThreshold);
        try
        {
            // Pairwise IoU on the GPU (boxes stay resident), then the greedy score-ordered suppression on
            // the host (inherently sequential) — matching the CPU exactly.
            var cb = boxes.IsContiguous ? boxes : (Tensor<T>)boxes.Contiguous();
            using var bufBoxes = GetOrAllocateBuffer(backend, cb.GetDataArray());
            using var bufIou = AllocateOutputBuffer(backend, N * N);
            backend.PairwiseIou(bufBoxes.Buffer, bufIou.Buffer, N);
            backend.Synchronize();
            var iou = backend.DownloadBuffer(bufIou.Buffer);
            var s = (float[])(object)(scores.IsContiguous ? scores : (Tensor<T>)scores.Contiguous()).GetDataArray();
            var order = new int[N];
            for (int i = 0; i < N; i++) order[i] = i;
            System.Array.Sort(order, (x, y) => s[y].CompareTo(s[x]));
            var keep = new List<int>(N);
            var suppressed = new bool[N];
            float thresh = (float)iouThreshold;
            for (int oi = 0; oi < N; oi++)
            {
                int i = order[oi];
                if (suppressed[i]) continue;
                keep.Add(i);
                for (int oj = oi + 1; oj < N; oj++)
                {
                    int j = order[oj];
                    if (!suppressed[j] && iou[i * N + j] > thresh) suppressed[j] = true;
                }
            }
            return new Tensor<int>(keep.ToArray(), new[] { keep.Count });
        }
        catch (Exception) { return base.Nms(boxes, scores, iouThreshold); }
    }

    /// <inheritdoc/>
    public override Tensor<int> MasksToBoxes<T>(Tensor<T> masks)
    {
        if (masks is null) throw new ArgumentNullException(nameof(masks));
        if (masks.Rank != 3) throw new ArgumentException("MasksToBoxes requires rank-3 masks [N, H, W].");
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.MasksToBoxes(masks);
        int N = masks._shape[0], H = masks._shape[1], W = masks._shape[2];
        if (N == 0) return base.MasksToBoxes(masks);
        try
        {
            var c = masks.IsContiguous ? masks : (Tensor<T>)masks.Contiguous();
            using var bufIn = GetOrAllocateBuffer(backend, c.GetDataArray());
            using var bufOut = AllocateOutputBuffer(backend, N * 4);
            backend.MasksToBoxes(bufIn.Buffer, bufOut.Buffer, N, H, W);
            backend.Synchronize();
            var f = backend.DownloadBuffer(bufOut.Buffer);
            var res = new Tensor<int>(new[] { N, 4 });
            var dst = res.AsWritableSpan();
            for (int i = 0; i < N * 4; i++) dst[i] = (int)f[i];
            return res;
        }
        catch (Exception) { return base.MasksToBoxes(masks); }
    }

    /// <inheritdoc/>
    public override Tensor<int> TensorHistogramDD<T>(Tensor<T> samples, int[] bins, T[] mins, T[] maxs)
    {
        if (samples is null) throw new ArgumentNullException(nameof(samples));
        if (bins is null) throw new ArgumentNullException(nameof(bins));
        if (mins is null) throw new ArgumentNullException(nameof(mins));
        if (maxs is null) throw new ArgumentNullException(nameof(maxs));
        if (typeof(T) != typeof(float) || samples.Rank != 2 || bins.Length != samples._shape[1]
            || mins.Length != samples._shape[1] || maxs.Length != samples._shape[1] || !TryGetBatchBackend(out var backend))
            return base.TensorHistogramDD(samples, bins, mins, maxs);
        int n = samples._shape[0], d = samples._shape[1];
        int total = 1; foreach (var bb in bins) total *= bb;
        if (total <= 0) return base.TensorHistogramDD(samples, bins, mins, maxs);
        try
        {
            var cs = samples.IsContiguous ? samples : (Tensor<T>)samples.Contiguous();
            using var bufSamp = GetOrAllocateBuffer(backend, cs.GetDataArray());
            using var bufBins = new OwnedBuffer(backend.AllocateIntBuffer(bins), true);
            using var bufMins = GetOrAllocateBuffer(backend, (T[])(object)((float[])(object)mins).Clone());
            using var bufMaxs = GetOrAllocateBuffer(backend, (T[])(object)((float[])(object)maxs).Clone());
            using var bufHist = AllocateOutputBuffer(backend, total);
            backend.Fill(bufHist.Buffer, 0f, total);
            backend.HistogramDD(bufSamp.Buffer, bufHist.Buffer, bufBins.Buffer, bufMins.Buffer, bufMaxs.Buffer, n, d);
            backend.Synchronize();
            var f = backend.DownloadBuffer(bufHist.Buffer);
            var res = new Tensor<int>((int[])bins.Clone());
            var dst = res.AsWritableSpan();
            for (int i = 0; i < total; i++) dst[i] = (int)f[i];
            return res;
        }
        catch (Exception) { return base.TensorHistogramDD(samples, bins, mins, maxs); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TimeStretch<T>(Tensor<T> waveform, double rate, int nFft = 512, int hopLength = 128)
    {
        if (waveform is null) throw new ArgumentNullException(nameof(waveform));
        if (rate <= 0) throw new ArgumentException("rate must be positive.");
        if (nFft < 2) throw new ArgumentException("nFft must be at least 2 (Hann window divides by nFft-1).");
        if (hopLength <= 0) throw new ArgumentException("hopLength must be positive.");
        if (System.Math.Abs(rate - 1.0) < 1e-12) return (Tensor<T>)waveform.Clone();
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.TimeStretch(waveform, rate, nFft, hopLength);
        try
        {
            int rank0 = waveform.Rank;
            int L = waveform._shape[rank0 - 1];
            int pad = nFft / 2; int Lp = L + 2 * pad;
            int batch = L == 0 ? 0 : waveform.Length / L;
            int numFrames = (Lp - nFft) / hopLength + 1;
            int numFreqs = nFft / 2 + 1;
            if (batch == 0 || numFrames <= 0 || pad >= L)
                return base.TimeStretch(waveform, rate, nFft, hopLength);
            // STFT output is [.., numFreqs, numFrames]; the CPU phase vocoder reads it with its own
            // (axis-flipped) convention: nFramesV = mag._shape[^2] = numFreqs, nFreqV = mag._shape[^1] = numFrames.
            int nFreqV = numFrames, nFramesV = numFreqs, leading = batch;
            int outFrames = (int)System.Math.Floor(nFramesV / rate);
            int targetLen = (int)System.Math.Round((double)L / rate);
            int outputLength = targetLen - nFft;   // ISTFT center subtracts nFft even when length is given
            if (outFrames <= 0 || outputLength <= 0)
                return base.TimeStretch(waveform, rate, nFft, hopLength);

            var warr = new float[nFft];
            for (int i = 0; i < nFft; i++) warr[i] = (float)(0.5 - 0.5 * System.Math.Cos(2.0 * System.Math.PI * i / (nFft - 1)));
            var cw = waveform.IsContiguous ? waveform : (Tensor<T>)waveform.Contiguous();
            using var bufWave = GetOrAllocateBuffer(backend, cw.GetDataArray());
            using var bufWin = GetOrAllocateBuffer(backend, (T[])(object)warr);
            using var bufPadded = AllocateOutputBuffer(backend, batch * Lp);
            backend.ReflectPad1d(bufWave.Buffer, bufPadded.Buffer, batch, L, Lp, pad);
            int stftLen = batch * numFreqs * numFrames;
            using var bufMag = AllocateOutputBuffer(backend, stftLen);
            using var bufPhase = AllocateOutputBuffer(backend, stftLen);
            backend.StftMagPhase(bufPadded.Buffer, bufWin.Buffer, bufMag.Buffer, bufPhase.Buffer, batch, Lp, nFft, hopLength, numFrames, numFreqs);
            int newLen = leading * outFrames * nFreqV;
            using var bufNewMag = AllocateOutputBuffer(backend, newLen);
            using var bufNewPhase = AllocateOutputBuffer(backend, newLen);
            backend.PhaseVocoder(bufMag.Buffer, bufPhase.Buffer, bufNewMag.Buffer, bufNewPhase.Buffer, leading, nFramesV, nFreqV, outFrames, (float)rate);
            int outTotal = batch * outputLength;
            // ISTFT sees numFreqs=outFrames, numFrames=nFreqV (the vocoder output axes). Build the full
            // conj-symmetric spectrum first (replays the CPU two passes in order), then inverse-DFT it.
            using var bufSpecRe = AllocateOutputBuffer(backend, batch * nFreqV * nFft);
            using var bufSpecIm = AllocateOutputBuffer(backend, batch * nFreqV * nFft);
            backend.BuildSpectrum(bufNewMag.Buffer, bufNewPhase.Buffer, bufSpecRe.Buffer, bufSpecIm.Buffer, batch, outFrames, nFreqV, nFft);
            using var bufWinSum = AllocateOutputBuffer(backend, outTotal);
            var bufResult = AllocateOutputBuffer(backend, outTotal);   // NOT using — FinishGpuOp owns it
            backend.Fill(bufResult.Buffer, 0f, outTotal);
            backend.Fill(bufWinSum.Buffer, 0f, outTotal);
            backend.IstftFromSpectrum(bufSpecRe.Buffer, bufSpecIm.Buffer, bufWin.Buffer, bufResult.Buffer, bufWinSum.Buffer, batch, nFreqV, nFft, hopLength, outputLength, 1);
            backend.IstftNormalize(bufResult.Buffer, bufWinSum.Buffer, outTotal);
            backend.Synchronize();   // all `using` intermediates feed the deferred result — force compute first
            var outArr = FinishGpuOp<T>(backend, bufResult, outTotal);
            var outShape = new int[rank0];
            for (int i = 0; i < rank0 - 1; i++) outShape[i] = waveform._shape[i];
            outShape[rank0 - 1] = outputLength;
            return new Tensor<T>(outArr, outShape);
        }
        catch (Exception) { return base.TimeStretch(waveform, rate, nFft, hopLength); }
    }

    /// <inheritdoc/>
    public override Tensor<T> Spectrogram<T>(Tensor<T> waveform, int nFft, int hopLength, int winLength, Tensor<T>? window = null)
    {
        if (waveform is null) throw new ArgumentNullException(nameof(waveform));
        if (typeof(T) != typeof(float) || nFft <= 0 || hopLength <= 0 || !TryGetBackend(out var backend))
            return base.Spectrogram(waveform, nFft, hopLength, winLength, window);
        // Build the window (must be length nFft for STFT). Match CpuEngine.HannWindow exactly.
        Tensor<T> win;
        if (window is not null) win = window;
        else
        {
            var warr = new T[winLength];
            var wf = (float[])(object)warr;
            for (int i = 0; i < winLength; i++)
                wf[i] = (float)(0.5 - 0.5 * System.Math.Cos(2.0 * System.Math.PI * i / System.Math.Max(1, winLength - 1)));
            win = new Tensor<T>(warr, new[] { winLength });
        }
        if (win.Length != nFft)
            return base.Spectrogram(waveform, nFft, hopLength, winLength, window);
        try
        {
            int rank = waveform.Rank;
            int L = waveform._shape[rank - 1];
            int pad = nFft / 2;                 // Spectrogram always uses STFT center:true
            int Lp = L + 2 * pad;
            int batch = L == 0 ? 0 : waveform.Length / L;
            int numFrames = (Lp - nFft) / hopLength + 1;
            int numFreqs = nFft / 2 + 1;
            if (batch == 0 || numFrames <= 0 || pad >= L)   // reflect needs pad < L
                return base.Spectrogram(waveform, nFft, hopLength, winLength, window);
            var newShape = new int[rank + 1];
            for (int i = 0; i < rank - 1; i++) newShape[i] = waveform._shape[i];
            newShape[rank - 1] = numFreqs;
            newShape[rank] = numFrames;
            int outLen = batch * numFreqs * numFrames;

            var cw = waveform.IsContiguous ? waveform : (Tensor<T>)waveform.Contiguous();
            var cwin = win.IsContiguous ? win : (Tensor<T>)win.Contiguous();
            using var bufWave = GetOrAllocateBuffer(backend, cw.GetDataArray());     // waveform stays resident
            using var bufWin = GetOrAllocateBuffer(backend, cwin.GetDataArray());
            using var bufPadded = AllocateOutputBuffer(backend, batch * Lp);
            backend.ReflectPad1d(bufWave.Buffer, bufPadded.Buffer, batch, L, Lp, pad);
            var bufMag = AllocateOutputBuffer(backend, outLen);
            var bufPhase = AllocateOutputBuffer(backend, outLen);
            backend.StftMagPhase(bufPadded.Buffer, bufWin.Buffer, bufMag.Buffer, bufPhase.Buffer, batch, Lp, nFft, hopLength, numFrames, numFreqs);
            // Force the stft compute to finish while the intermediate `bufPadded` is still alive (it is
            // `using` and feeds the kernel). FinishGpuOp only defers the DOWNLOAD, not the compute, so
            // without this the padded buffer could be freed before a deferred materialization runs the
            // kernel — a use-after-free. mag/phase still stay GPU-resident (deferred download).
            backend.Synchronize();
            var magArr = FinishGpuOp<T>(backend, bufMag, outLen);
            var phaseArr = FinishGpuOp<T>(backend, bufPhase, outLen);
            var mag = new Tensor<T>(magArr, newShape);
            var phase = new Tensor<T>(phaseArr, (int[])newShape.Clone());
            int origLength = waveform._shape[rank - 1];
            // Same backward contract as the base (phase piped through ISTFT); no-op when no tape is active.
            DifferentiableOps.RecordUnary("Spectrogram", mag, waveform,
                BackwardFunctions<T>.SpectrogramBackward, new object[] { nFft, hopLength, win, phase, origLength });
            return mag;
        }
        catch (Exception) { return base.Spectrogram(waveform, nFft, hopLength, winLength, window); }
    }

    /// <inheritdoc/>
    public override Tensor<T> GridSampleBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> grid, int[] inputShape,
        GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (grid is null) throw new ArgumentNullException(nameof(grid));
        if (inputShape is null) throw new ArgumentNullException(nameof(inputShape));
        // NCHW (PyTorch): inputShape [N,C,H,W], gradOutput [N,C,outH,outW], grid [N,outH,outW,2].
        if (typeof(T) != typeof(float) || mode != GridSampleMode.Bilinear || padding != GridSamplePadding.Zeros
            || alignCorners || inputShape.Length != 4 || gradOutput.Rank != 4 || grid.Rank != 4 || !TryGetBackend(out var backend))
            return base.GridSampleBackwardInput(gradOutput, grid, inputShape, mode, padding, alignCorners);
        int N = inputShape[0], C = inputShape[1], H = inputShape[2], W = inputShape[3];
        int outH = gradOutput._shape[2], outW = gradOutput._shape[3];
        try
        {
            var cgo = gradOutput.IsContiguous ? gradOutput : (Tensor<T>)gradOutput.Contiguous();
            var cg = grid.IsContiguous ? grid : (Tensor<T>)grid.Contiguous();
            int inSize = N * H * W * C;
            using var bufGO = GetOrAllocateBuffer(backend, cgo.GetDataArray());
            using var bufGrid = GetOrAllocateBuffer(backend, cg.GetDataArray());
            var bufGradIn = AllocateOutputBuffer(backend, inSize);
            backend.Fill(bufGradIn.Buffer, 0f, inSize);
            backend.GridSampleBackwardInputNhwc(bufGO.Buffer, bufGrid.Buffer, bufGradIn.Buffer, N, H, W, C, outH, outW);
            var arr = FinishGpuOp<T>(backend, bufGradIn, inSize);
            return new Tensor<T>(arr, (int[])inputShape.Clone());
        }
        catch (Exception) { return base.GridSampleBackwardInput(gradOutput, grid, inputShape, mode, padding, alignCorners); }
    }

    /// <inheritdoc/>
    public override Tensor<T> GridSampleBackwardGrid<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> grid,
        GridSampleMode mode, GridSamplePadding padding, bool alignCorners)
    {
        if (gradOutput is null) throw new ArgumentNullException(nameof(gradOutput));
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (grid is null) throw new ArgumentNullException(nameof(grid));
        if (typeof(T) != typeof(float) || mode != GridSampleMode.Bilinear || padding != GridSamplePadding.Zeros
            || alignCorners || input.Rank != 4 || gradOutput.Rank != 4 || grid.Rank != 4 || !TryGetBackend(out var backend))
            return base.GridSampleBackwardGrid(gradOutput, input, grid, mode, padding, alignCorners);
        int N = input._shape[0], C = input._shape[1], H = input._shape[2], W = input._shape[3];
        int outH = grid._shape[1], outW = grid._shape[2];
        try
        {
            var cgo = gradOutput.IsContiguous ? gradOutput : (Tensor<T>)gradOutput.Contiguous();
            var ci = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            var cg = grid.IsContiguous ? grid : (Tensor<T>)grid.Contiguous();
            int gridSize = N * outH * outW * 2;
            using var bufGO = GetOrAllocateBuffer(backend, cgo.GetDataArray());
            using var bufIn = GetOrAllocateBuffer(backend, ci.GetDataArray());
            using var bufGrid = GetOrAllocateBuffer(backend, cg.GetDataArray());
            var bufGradGrid = AllocateOutputBuffer(backend, gridSize);
            backend.GridSampleBackwardGridNhwc(bufGO.Buffer, bufIn.Buffer, bufGrid.Buffer, bufGradGrid.Buffer, N, H, W, C, outH, outW);
            var arr = FinishGpuOp<T>(backend, bufGradGrid, gridSize);
            return new Tensor<T>(arr, (int[])grid._shape.Clone());
        }
        catch (Exception) { return base.GridSampleBackwardGrid(gradOutput, input, grid, mode, padding, alignCorners); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorZeta<T>(Tensor<T> x, Tensor<T> q)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (q is null) throw new ArgumentNullException(nameof(q));
        if (typeof(T) != typeof(float) || !ShapesEqual(x._shape, q._shape) || !TryGetBackend(out var backend))
            return base.TensorZeta(x, q);
        try
        {
            var cx = x.IsContiguous ? x : (Tensor<T>)x.Contiguous();
            var cq = q.IsContiguous ? q : (Tensor<T>)q.Contiguous();
            int n = x.Length;
            using var bufX = GetOrAllocateBuffer(backend, cx.GetDataArray());
            using var bufQ = GetOrAllocateBuffer(backend, cq.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Zeta(bufX.Buffer, bufQ.Buffer, bufOut.Buffer, n);
            var arr = FinishGpuOp<T>(backend, bufOut, n);
            return new Tensor<T>(arr, (int[])x._shape.Clone());
        }
        catch (Exception) { return base.TensorZeta(x, q); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorPolygamma<T>(int n, Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (typeof(T) != typeof(float) || n < 1 || !TryGetBackend(out var backend))
            return base.TensorPolygamma(n, tensor);
        try
        {
            var c = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            int len = tensor.Length;
            using var bufX = GetOrAllocateBuffer(backend, c.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, len);
            backend.Polygamma(bufX.Buffer, bufOut.Buffer, n, len);
            var arr = FinishGpuOp<T>(backend, bufOut, len);
            return new Tensor<T>(arr, (int[])tensor._shape.Clone());
        }
        catch (Exception) { return base.TensorPolygamma(n, tensor); }
    }

    /// <inheritdoc/>
    public override Tensor<int> TensorNonzero<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (typeof(T) != typeof(float) || tensor.Length == 0 || !TryGetBackend(out var backend))
            return base.TensorNonzero(tensor);
        try
        {
            var c = tensor.IsContiguous ? tensor : (Tensor<T>)tensor.Contiguous();
            int len = tensor.Length, rank = tensor.Rank;
            using var bufIn = GetOrAllocateBuffer(backend, c.GetDataArray());
            using var bufMask = AllocateOutputBuffer(backend, len);
            backend.NotEqualScalar(bufIn.Buffer, bufMask.Buffer, 0f, len);   // nonzero mask on GPU (input stays resident)
            backend.Synchronize();
            var mask = backend.DownloadBuffer(bufMask.Buffer);
            int nnz = 0; for (int i = 0; i < len; i++) if (mask[i] != 0f) nnz++;
            var strides = new int[rank];
            strides[rank - 1] = 1;
            for (int d = rank - 2; d >= 0; d--) strides[d] = strides[d + 1] * tensor._shape[d + 1];
            var result = new Tensor<int>(new[] { nnz, rank });
            var dst = result.AsWritableSpan();
            int w = 0;
            for (int i = 0; i < len; i++)
                if (mask[i] != 0f)
                {
                    int rem = i;
                    for (int d = 0; d < rank; d++) { dst[w * rank + d] = rem / strides[d]; rem %= strides[d]; }
                    w++;
                }
            return result;
        }
        catch (Exception) { return base.TensorNonzero(tensor); }
    }

    /// <inheritdoc/>
    public override Tensor<T> TensorUniqueConsecutive<T>(Tensor<T> input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (typeof(T) != typeof(float) || input.Length == 0 || !TryGetBatchBackend(out var backend))
            return base.TensorUniqueConsecutive(input);
        try
        {
            var c = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            int n = input.Length;
            using var bufIn = GetOrAllocateBuffer(backend, c.GetDataArray());
            using var bufMask = AllocateOutputBuffer(backend, n);
            backend.ShiftedDiff(bufIn.Buffer, bufMask.Buffer, n);   // keep mask on GPU
            backend.Synchronize();
            var mask = backend.DownloadBuffer(bufMask.Buffer);
            var kept = new List<int>();
            for (int i = 0; i < n; i++) if (mask[i] != 0f) kept.Add(i);
            int count = kept.Count;
            using var bufIdx = new OwnedBuffer(backend.AllocateIntBuffer(kept.ToArray()), true);
            var bufOut = AllocateOutputBuffer(backend, count);
            backend.Gather(bufIn.Buffer, bufIdx.Buffer, bufOut.Buffer, count, 1);   // gather kept values (resident)
            var arr = FinishGpuOp<T>(backend, bufOut, count);
            return new Tensor<T>(arr, new[] { count });
        }
        catch (Exception) { return base.TensorUniqueConsecutive(input); }
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

    // ----- Logical ops on Tensor<Bit> (GPU-RESIDENT boolean expressions) -----
    // Bit masks ride the GPU as float-0/1 buffers (BitOperations converts Bit<->float). Inputs use the
    // tensor overload of GetOrAllocateBuffer (resident masks stay on-device, no re-download), and the
    // result is a DEFERRED Tensor<Bit> via FinishGpuOp<Bit> — so chains like (a==b) & (c<d) | !e keep
    // every intermediate on the GPU and only materialize at the final CPU read.
    private Tensor<Bit> LogicalBinaryGpu(Tensor<Bit> a, Tensor<Bit> b, int mode, System.Func<Tensor<Bit>, Tensor<Bit>, Tensor<Bit>> fallback)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (!ShapesEqual(a._shape, b._shape) || !TryGetBackend(out var backend))
            return fallback(a, b);
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<Bit>)a.Contiguous();
            var cb = b.IsContiguous ? b : (Tensor<Bit>)b.Contiguous();
            int n = a.Length;
            using var bufA = GetOrAllocateBuffer(backend, ca);
            using var bufB = GetOrAllocateBuffer(backend, cb);
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.LogicalOp(bufA.Buffer, bufB.Buffer, bufOut.Buffer, mode, n);
            var arr = FinishGpuOp<Bit>(backend, bufOut, n);
            return new Tensor<Bit>(arr, (int[])a._shape.Clone());
        }
        catch (Exception) { return fallback(a, b); }
    }

    /// <inheritdoc/>
    public override Tensor<Bit> TensorLogicalAnd(Tensor<Bit> a, Tensor<Bit> b) => LogicalBinaryGpu(a, b, 0, base.TensorLogicalAnd);
    /// <inheritdoc/>
    public override Tensor<Bit> TensorLogicalOr(Tensor<Bit> a, Tensor<Bit> b) => LogicalBinaryGpu(a, b, 1, base.TensorLogicalOr);
    /// <inheritdoc/>
    public override Tensor<Bit> TensorLogicalXor(Tensor<Bit> a, Tensor<Bit> b) => LogicalBinaryGpu(a, b, 2, base.TensorLogicalXor);

    /// <inheritdoc/>
    public override Tensor<Bit> TensorLogicalNot(Tensor<Bit> a)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (!TryGetBackend(out var backend)) return base.TensorLogicalNot(a);
        try
        {
            var ca = a.IsContiguous ? a : (Tensor<Bit>)a.Contiguous();
            int n = a.Length;
            using var bufA = GetOrAllocateBuffer(backend, ca);
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.LogicalNot(bufA.Buffer, bufOut.Buffer, n);
            var arr = FinishGpuOp<Bit>(backend, bufOut, n);
            return new Tensor<Bit>(arr, (int[])a._shape.Clone());
        }
        catch (Exception) { return base.TensorLogicalNot(a); }
    }

    // Deferred (GPU-RESIDENT) variant of PackMask: keeps the float-0/1 mask on the device as a
    // Tensor<Bit> (materialized on first CPU read via FinishGpuOp<Bit> + BitOperations), so
    // predicate -> logical chains stay resident. `invert` flips 0<->1 in-place on the GPU first.
    // Takes ownership of maskBuf (do NOT `using` it at the call site).
    private Tensor<Bit> PackMaskResident(IDirectGpuBackend backend, OwnedBuffer maskBuf, int[] shape, int n, bool invert)
    {
        if (invert) backend.LogicalNot(maskBuf.Buffer, maskBuf.Buffer, n);
        var arr = FinishGpuOp<Bit>(backend, maskBuf, n);
        return new Tensor<Bit>(arr, (int[])shape.Clone());
    }

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
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.Equal(bufA.Buffer, bufB.Buffer, bufOut.Buffer, n);
            return PackMaskResident(backend, bufOut, a._shape, n, invert: false);
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
            var bufOut = AllocateOutputBuffer(backend, n);
            // NotEqualScalar gives (a != s); invert to get equality.
            backend.NotEqualScalar(bufA.Buffer, bufOut.Buffer, s, n);
            return PackMaskResident(backend, bufOut, a._shape, n, invert: true);
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
            var maskBuf = AllocCloseMask(backend, a, b, rtol, atol, out int n);
            return PackMaskResident(backend, maskBuf, a._shape, n, invert: false);
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
            var bufOut = AllocateOutputBuffer(backend, n);
            backend.ClassifyFloat(bufA.Buffer, bufOut.Buffer, mode, n);
            return PackMaskResident(backend, bufOut, tensor._shape, n, invert: false);
        }
        catch (Exception) { return fallback(tensor); }
    }

    // ----- Category B: composite forwards -----

    // Returns the contiguous float data of an optional tensor, or a zero array of `size` when null.
    private float[] ZerosOrData<T>(Tensor<T>? t, int size)
    {
        if (t is null) return new float[size];
        var c = t.IsContiguous ? t : (Tensor<T>)t.Contiguous();
        return (float[])(object)c.GetDataArray();
    }

    /// <inheritdoc/>
    public override Tensor<T> FusedHierarchicalSoftmax<T>(Tensor<T> input, Tensor<T> nodeWeights, int numClasses)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (nodeWeights is null) throw new ArgumentNullException(nameof(nodeWeights));
        if (numClasses < 2) throw new ArgumentException("numClasses must be >= 2.", nameof(numClasses));
        if (typeof(T) != typeof(float) || nodeWeights.Rank != 2 || !TryGetBackend(out var backend))
            return base.FusedHierarchicalSoftmax(input, nodeWeights, numClasses);
        int d = input._shape[input.Rank - 1];
        int treeDepth = nodeWeights._shape[0];
        int minDepth = 0; while ((1 << minDepth) < numClasses) minDepth++;
        if (nodeWeights._shape[1] != d || treeDepth < minDepth)
            return base.FusedHierarchicalSoftmax(input, nodeWeights, numClasses);
        try
        {
            // Per-level node activations on the GPU (input · nodeWeightsᵀ), then the leaf-probability
            // path products (sigmoid folded into the kernel). The kernel was previously diverging
            // from CPU at treeDepth > 32 because `1 << (treeDepth - level - 1)` is undefined for
            // shift counts ≥ 32 — C# masks to 5 bits (returning 1) while CUDA PTX returns 0.
            // Both reference paths now clamp `sh < 32` so any bit past int's width is treated as
            // 0 (semantic: a tree deeper than the integer's bit-width can't address those bits).
            int rows = input.Length / d;
            var input2d = (input.IsContiguous ? input : (Tensor<T>)input.Contiguous()).Reshape(new[] { rows, d });
            var acts = TensorMatMulTransposed(input2d, nodeWeights);   // [rows, treeDepth]
            var actsC = acts.IsContiguous ? acts : (Tensor<T>)acts.Contiguous();
            using var bufActs = GetOrAllocateBuffer(backend, actsC.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, rows * numClasses);
            backend.HierarchicalSoftmaxPaths(bufActs.Buffer, bufOut.Buffer, rows, treeDepth, numClasses);
            var arr = FinishGpuOp<T>(backend, bufOut, rows * numClasses);
            var outShape = (int[])input._shape.Clone();
            outShape[outShape.Length - 1] = numClasses;
            return new Tensor<T>(arr, outShape);
        }
        catch (Exception) { return base.FusedHierarchicalSoftmax(input, nodeWeights, numClasses); }
    }

    /// <inheritdoc/>
    public override Tensor<T> FusedLinearMaxout<T>(Tensor<T> input, Tensor<T> weights, Tensor<T>? bias, int numPieces)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (weights is null) throw new ArgumentNullException(nameof(weights));
        if (numPieces < 2) throw new ArgumentException("numPieces must be >= 2.", nameof(numPieces));
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend))
            return base.FusedLinearMaxout(input, weights, bias, numPieces);
        try
        {
            // GEMM+bias on the GPU, then maxout = max over each unit's numPieces consecutive features
            // (MaxAxis with reduceSize = numPieces).
            var pre = FusedLinear(input, weights, bias, FusedActivationType.None);
            int n = pre._shape[pre.Rank - 1];
            if (n % numPieces != 0) return base.FusedLinearMaxout(input, weights, bias, numPieces);
            int units = n / numPieces;
            int rows = pre.Length / n;
            var preC = pre.IsContiguous ? pre : (Tensor<T>)pre.Contiguous();
            using var bufPre = GetOrAllocateBuffer(backend, preC.GetDataArray());
            var bufOut = AllocateOutputBuffer(backend, rows * units);
            backend.MaxAxis(bufPre.Buffer, bufOut.Buffer, rows * units, numPieces);
            var arr = FinishGpuOp<T>(backend, bufOut, rows * units);
            var outShape = (int[])pre._shape.Clone();
            outShape[outShape.Length - 1] = units;
            return new Tensor<T>(arr, outShape);
        }
        catch (Exception) { return base.FusedLinearMaxout(input, weights, bias, numPieces); }
    }

    /// <inheritdoc/>
    public override Tensor<T> Rwkv7SequenceForward<T>(
        Tensor<T> rProj, Tensor<T> kProj, Tensor<T> vProj, Tensor<T> aProj, Tensor<T> bProj, int numHeads)
    {
        if (rProj is null) throw new ArgumentNullException(nameof(rProj));
        if (kProj is null) throw new ArgumentNullException(nameof(kProj));
        if (vProj is null) throw new ArgumentNullException(nameof(vProj));
        if (aProj is null) throw new ArgumentNullException(nameof(aProj));
        if (bProj is null) throw new ArgumentNullException(nameof(bProj));
        if (typeof(T) != typeof(float) || rProj.Rank != 3 || numHeads < 1 || !TryGetBackend(out var backend))
            return base.Rwkv7SequenceForward(rProj, kProj, vProj, aProj, bProj, numHeads);
        int batch = rProj._shape[0], seqLen = rProj._shape[1], modelDim = rProj._shape[2];
        if (modelDim % numHeads != 0
            || !ShapesEqual(kProj._shape, rProj._shape) || !ShapesEqual(vProj._shape, rProj._shape)
            || !ShapesEqual(aProj._shape, rProj._shape) || !ShapesEqual(bProj._shape, rProj._shape))
            return base.Rwkv7SequenceForward(rProj, kProj, vProj, aProj, bProj, numHeads);
        int headDim = modelDim / numHeads;
        try
        {
            var cr = rProj.IsContiguous ? rProj : (Tensor<T>)rProj.Contiguous();
            var ck = kProj.IsContiguous ? kProj : (Tensor<T>)kProj.Contiguous();
            var cv = vProj.IsContiguous ? vProj : (Tensor<T>)vProj.Contiguous();
            var ca = aProj.IsContiguous ? aProj : (Tensor<T>)aProj.Contiguous();
            var cb = bProj.IsContiguous ? bProj : (Tensor<T>)bProj.Contiguous();
            int total = batch * seqLen * modelDim;
            using var bufR = GetOrAllocateBuffer(backend, cr.GetDataArray());
            using var bufK = GetOrAllocateBuffer(backend, ck.GetDataArray());
            using var bufV = GetOrAllocateBuffer(backend, cv.GetDataArray());
            using var bufA = GetOrAllocateBuffer(backend, ca.GetDataArray());
            using var bufB = GetOrAllocateBuffer(backend, cb.GetDataArray());
            using var bufS = AllocateOutputBuffer(backend, batch * numHeads * headDim * headDim);
            var bufOut = AllocateOutputBuffer(backend, total);
            backend.Rwkv7Forward(bufR.Buffer, bufK.Buffer, bufV.Buffer, bufA.Buffer, bufB.Buffer, bufOut.Buffer,
                bufS.Buffer, batch, seqLen, modelDim, numHeads, headDim);
            var arr = FinishGpuOp<T>(backend, bufOut, total);
            return new Tensor<T>(arr, new[] { batch, seqLen, modelDim });
        }
        catch (Exception)
        {
            return base.Rwkv7SequenceForward(rProj, kProj, vProj, aProj, bProj, numHeads);
        }
    }

    /// <inheritdoc/>
    public override Tensor<T> LstmSequenceForward<T>(
        Tensor<T> input, Tensor<T>? h0, Tensor<T>? c0, Tensor<T> wIh, Tensor<T> wHh,
        Tensor<T>? bIh, Tensor<T>? bHh, bool returnSequences = false)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (wIh is null) throw new ArgumentNullException(nameof(wIh));
        if (wHh is null) throw new ArgumentNullException(nameof(wHh));
        if (typeof(T) != typeof(float) || input.Rank != 3 || !TryGetBatchBackend(out var backend))
            return base.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);

        int B = input._shape[0], S = input._shape[1], In = input._shape[2];
        int gateRows = wIh._shape[0];
        if (gateRows % 4 != 0 || wIh._shape[1] != In || wHh._shape[0] != gateRows || wHh._shape[1] != gateRows / 4)
            return base.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
        int Hd = gateRows / 4;
        if (B == 0 || S == 0)
            return base.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);

        // Engine weights/bias/gates (PyTorch i,f,g,o; [4*hidden, *]) match the kernel exactly. Only the
        // sequence layout differs: the kernel wants [seq, batch, *], the engine uses [batch, seq, *], so
        // transpose in and (for returnSequences) out.
        try
        {
            var inBSI = input.IsContiguous ? input : (Tensor<T>)input.Contiguous();
            var inSBI = PermuteResidentGpu(backend, inBSI, new[] { 1, 0, 2 });   // [S, B, In]

            using var bufInput = GetOrAllocateBuffer(backend, inSBI.GetDataArray());
            using var bufWih = GetOrAllocateBuffer(backend, (wIh.IsContiguous ? wIh : (Tensor<T>)wIh.Contiguous()).GetDataArray());
            using var bufWhh = GetOrAllocateBuffer(backend, (wHh.IsContiguous ? wHh : (Tensor<T>)wHh.Contiguous()).GetDataArray());
            using var bufH0 = GetOrAllocateBuffer(backend, (T[])(object)ZerosOrData(h0, B * Hd));
            using var bufC0 = GetOrAllocateBuffer(backend, (T[])(object)ZerosOrData(c0, B * Hd));
            using var bufBih = GetOrAllocateBuffer(backend, (T[])(object)ZerosOrData(bIh, gateRows));
            using var bufBhh = GetOrAllocateBuffer(backend, (T[])(object)ZerosOrData(bHh, gateRows));

            using var bufHf = AllocateOutputBuffer(backend, B * Hd);
            using var bufCf = AllocateOutputBuffer(backend, B * Hd);
            using var bufAllH = AllocateOutputBuffer(backend, (S + 1) * B * Hd);
            using var bufAllC = AllocateOutputBuffer(backend, (S + 1) * B * Hd);
            using var bufGates = AllocateOutputBuffer(backend, S * B * Hd * 4);
            var bufOut = AllocateOutputBuffer(backend, S * B * Hd);

            backend.LstmForwardSequence(
                bufInput.Buffer, bufH0.Buffer, bufC0.Buffer, bufWih.Buffer, bufWhh.Buffer, bufBih.Buffer, bufBhh.Buffer,
                bufOut.Buffer, bufHf.Buffer, bufCf.Buffer, bufAllH.Buffer, bufAllC.Buffer, bufGates.Buffer,
                S, B, In, Hd);

            if (returnSequences)
            {
                var outArr = FinishGpuOp<T>(backend, bufOut, S * B * Hd);
                var outSBH = new Tensor<T>(outArr, new[] { S, B, Hd });
                return PermuteResidentGpu(backend, outSBH, new[] { 1, 0, 2 });   // [B, S, Hd]
            }
            backend.Synchronize();
            var hf = backend.DownloadBuffer(bufHf.Buffer);
            bufOut.Dispose();
            return new Tensor<T>((T[])(object)hf, new[] { B, Hd });
        }
        catch (Exception)
        {
            return base.LstmSequenceForward(input, h0, c0, wIh, wHh, bIh, bHh, returnSequences);
        }
    }

    // Row-wise softmax over the last axis, GPU-resident.
    private Tensor<T> SoftmaxLastAxisGpu<T>(IDirectGpuBackend backend, Tensor<T> x)
    {
        int cols = x._shape[x.Rank - 1];
        int rows = cols == 0 ? 0 : x.Length / cols;
        var c = x.IsContiguous ? x : (Tensor<T>)x.Contiguous();
        using var inBuf = GetOrAllocateBuffer(backend, c.GetDataArray());
        var outBuf = AllocateOutputBuffer(backend, x.Length);
        backend.SoftmaxRows(inBuf.Buffer, outBuf.Buffer, rows, cols);
        var arr = FinishGpuOp<T>(backend, outBuf, x.Length);
        return new Tensor<T>(arr, (int[])x._shape.Clone());
    }

    // [B*S, D] -> [B,S,H,d] -> (materialized) [B,H,S,d] -> [B*H, S, d].
    private Tensor<T> ReshapePermuteHeads<T>(IDirectGpuBackend backend, Tensor<T> x, int B, int S, int H, int d)
    {
        var bshd = x.Reshape(new[] { B, S, H, d });
        var bhsd = PermuteResidentGpu(backend, bshd, new[] { 0, 2, 1, 3 });   // [B,H,S,d]
        return bhsd.Reshape(new[] { B * H, S, d });
    }

    // PR #638 diag: pin the residual MHA leak source — count tape-active vs tapeless MHA-forward calls (the
    // tapeless ones get DropTransientActivationsCreatedAfter; tape-active are skipped). AIDOTNET_MHA_DIAG=1.
    private static readonly bool s_mhaDiag = System.Environment.GetEnvironmentVariable("AIDOTNET_MHA_DIAG") == "1";
    private static long s_mhaTotal, s_mhaTapeActive;
    private const int MhaDiagDumpInterval = 500;                    // dump every N MHA calls
    private const string MhaDiagFileName = "aidotnet_mha_diag.txt"; // under Path.GetTempPath()
    private static void MhaDiag(bool tapeActive)
    {
        long tot = System.Threading.Interlocked.Increment(ref s_mhaTotal);
        if (tapeActive) System.Threading.Interlocked.Increment(ref s_mhaTapeActive);
        if (tot % MhaDiagDumpInterval == 0)
            try { System.IO.File.AppendAllText(System.IO.Path.Combine(System.IO.Path.GetTempPath(), MhaDiagFileName),
                $"MHA calls={tot} tapeActive={System.Threading.Interlocked.Read(ref s_mhaTapeActive)} dropFreedTotal={System.Threading.Interlocked.Read(ref s_dropFreedTotal)}" + System.Environment.NewLine); } catch { }
    }

    /// <inheritdoc/>
    public override Tensor<T> MultiHeadAttentionForward<T>(
        Tensor<T> input, Tensor<T> qWeight, Tensor<T> kWeight, Tensor<T> vWeight, Tensor<T> outWeight,
        int numHeads, Tensor<bool>? mask = null)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (qWeight is null) throw new ArgumentNullException(nameof(qWeight));
        if (kWeight is null) throw new ArgumentNullException(nameof(kWeight));
        if (vWeight is null) throw new ArgumentNullException(nameof(vWeight));
        if (outWeight is null) throw new ArgumentNullException(nameof(outWeight));

        // GPU path: float, rank-3 input, divisible heads, no mask (masked attention defers to base for
        // now), inference only. Anything else uses the validated CPU base.
        if (typeof(T) != typeof(float) || input.Rank != 3 || numHeads <= 0 || mask is not null
            || input._shape[2] % numHeads != 0 || !TryGetBackend(out var backend))
            return base.MultiHeadAttentionForward(input, qWeight, kWeight, vWeight, outWeight, numHeads, mask);

        int B = input._shape[0], S = input._shape[1], D = input._shape[2];
        int H = numHeads, d = D / H;
        if (qWeight.Rank != 2 || qWeight._shape[0] != D || qWeight._shape[1] != D ||
            kWeight.Rank != 2 || kWeight._shape[0] != D || kWeight._shape[1] != D ||
            vWeight.Rank != 2 || vWeight._shape[0] != D || vWeight._shape[1] != D ||
            outWeight.Rank != 2 || outWeight._shape[0] != D || outWeight._shape[1] != D)
            return base.MultiHeadAttentionForward(input, qWeight, kWeight, vWeight, outWeight, numHeads, mask);

        try
        {
            // Snapshot the activation cache so the dead intermediates this fused op caches (qh/kh/vh, the per-head
            // scores/attn/headOuts, oBHSD/oBSHD) can be released on the way out. This is a TAPELESS inference op
            // (reached via MultiHeadAttentionLayer.TryFusedAttentionInference) — no GradientTape brackets it, so
            // NOTHING else releases those entries, and with the compiled step's eviction suspended they accumulate
            // every step → the d128/L1 async-pool-OOM CUDA-700 that aborts whole-step capture. (Leak localized via
            // GpuMemoryTracker: x536-540 MHA-forward intermediates/step.) We free them WITHOUT a download (they are
            // never read) so it is capture-safe; the resident result is preserved via its keep-key.
            long actSnapshot = ActivationCacheTimestampSnapshot();
            float scale = (float)(1.0 / System.Math.Sqrt(d));
            T scaleT = (T)(object)scale;
            var input2d = (input.IsContiguous ? input : (Tensor<T>)input.Contiguous()).Reshape(new[] { B * S, D });
            // Q/K/V projections on the GPU (2-D GEMM), then split into [B*H, S, d].
            var qh = ReshapePermuteHeads(backend, TensorMatMul(input2d, qWeight), B, S, H, d);
            var kh = ReshapePermuteHeads(backend, TensorMatMul(input2d, kWeight), B, S, H, d);
            var vh = ReshapePermuteHeads(backend, TensorMatMul(input2d, vWeight), B, S, H, d);

            var headOuts = new Tensor<T>[B * H];
            for (int bh = 0; bh < B * H; bh++)
            {
                var qbh = TensorSlice(qh, new[] { bh, 0, 0 }, new[] { 1, S, d }).Reshape(new[] { S, d });
                var kbh = TensorSlice(kh, new[] { bh, 0, 0 }, new[] { 1, S, d }).Reshape(new[] { S, d });
                var vbh = TensorSlice(vh, new[] { bh, 0, 0 }, new[] { 1, S, d }).Reshape(new[] { S, d });
                var scores = TensorMultiplyScalar(TensorMatMulTransposed(qbh, kbh), scaleT);  // [S,S] = (q·kᵀ)·scale
                var attn = SoftmaxLastAxisGpu(backend, scores);                                // softmax over keys
                headOuts[bh] = TensorMatMul(attn, vbh).Reshape(new[] { 1, S, d });            // [1,S,d]
            }

            var oBHSD = TensorConcatenate(headOuts, 0).Reshape(new[] { B, H, S, d });
            var oBSHD = PermuteResidentGpu(backend, oBHSD, new[] { 0, 2, 1, 3 }).Reshape(new[] { B * S, D });
            var outProj = TensorMatMul(oBSHD, outWeight);                    // [B*S, D] — the live result buffer
            var result = outProj.Reshape(new[] { B, S, D });
            // Release the dead intermediates (keep the result's resident buffer). Only when no tape depends on
            // them; if a tape is recording (the sub-ops registered backwards), their lifetime is the tape's.
            bool tapeActive = IsTapeActive<T>();
            if (!tapeActive)
                // Tapeless MHA calls: drop the dead intermediates (caller-thread; measured: they are NOT cached on
                // pool threads, so the allThreads variant gives no gain here). The tape-ACTIVE MHA calls are
                // released via the tape-dispose path instead — see GradientTape.
                DropTransientActivationsCreatedAfter(actSnapshot, outProj.DataVector.GetBackingArrayUnsafe());
            if (s_mhaDiag) MhaDiag(tapeActive);   // PR #638 diag: count tape-active vs tapeless MHA calls
            return result;
        }
        catch (Exception)
        {
            return base.MultiHeadAttentionForward(input, qWeight, kWeight, vWeight, outWeight, numHeads, mask);
        }
    }

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

    // CpuEngine.MaxPool3D is non-virtual, so these are explicit IEngine re-implementations. The scalar
    // overload resolves its defaults then routes through the interface so the int[] GPU path is hit.
    Tensor<T> IEngine.MaxPool3D<T>(Tensor<T> input, int poolSize, int stride, int padding)
    {
        if (stride == 0) stride = poolSize;
        return ((IEngine)this).MaxPool3D(input, new[] { poolSize, poolSize, poolSize },
            new[] { stride, stride, stride }, new[] { padding, padding, padding });
    }

    /// <inheritdoc/>
    Tensor<T> IEngine.MaxPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        // GPU maxpool3d kernel handles the no-padding float case; defer everything else (tape/graph,
        // non-float, padded, wrong rank) to the base, which also records the backward.
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || input.Rank != 5 || poolSize is not { Length: 3 } || stride is not { Length: 3 }
            || padding is not { Length: 3 } || padding[0] != 0 || padding[1] != 0 || padding[2] != 0
            || !TryGetBackend(out var backend))
            return base.MaxPool3D(input, poolSize, stride, padding);
        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inD = input.Shape._dims[2], inH = input.Shape._dims[3], inW = input.Shape._dims[4];
            int kD = poolSize[0], kH = poolSize[1], kW = poolSize[2];
            int sD = stride[0], sH = stride[1], sW = stride[2];
            if (sD <= 0 || sH <= 0 || sW <= 0) return base.MaxPool3D(input, poolSize, stride, padding);
            int outD = (inD - kD) / sD + 1, outH = (inH - kH) / sH + 1, outW = (inW - kW) / sW + 1;
            if (outD <= 0 || outH <= 0 || outW <= 0) return base.MaxPool3D(input, poolSize, stride, padding);
            int outLen = batch * channels * outD * outH * outW;
            using var inBuf = GetOrAllocateBuffer(backend, input);
            var outBuf = AllocateOutputBuffer(backend, outLen);
            try
            {
                backend.MaxPool3D(inBuf.Buffer, outBuf.Buffer, null, batch, channels,
                    inD, inH, inW, outD, outH, outW, kD, kH, kW, sD, sH, sW);
                var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                return new Tensor<T>(arr, new[] { batch, channels, outD, outH, outW });
            }
            catch { outBuf.Dispose(); throw; }
        }
        catch { return base.MaxPool3D(input, poolSize, stride, padding); }
    }

    /// <inheritdoc/>
    Tensor<T> IEngine.ScaledDotProductAttention<T>(Tensor<T> query, Tensor<T> key, Tensor<T> value,
        Tensor<bool>? mask, double? scale, out Tensor<T> attentionWeights)
    {
        // GPU SDPA kernel: self-attention [batch, numHeads, seqLen, headDim], no mask. Anything else
        // (tape/graph, non-float, masked, cross-attention with differing seqLen) defers to the base.
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || query.Rank != 4 || key.Rank != 4 || value.Rank != 4 || mask is not null || !TryGetBackend(out var backend))
            return base.ScaledDotProductAttention(query, key, value, mask, scale, out attentionWeights);
        try
        {
            int batch = query.Shape._dims[0], numHeads = query.Shape._dims[1], seqLen = query.Shape._dims[2], headDim = query.Shape._dims[3];
            if (key.Shape._dims[0] != batch || key.Shape._dims[1] != numHeads || key.Shape._dims[2] != seqLen || key.Shape._dims[3] != headDim
                || value.Shape._dims[0] != batch || value.Shape._dims[1] != numHeads || value.Shape._dims[2] != seqLen || value.Shape._dims[3] != headDim)
                return base.ScaledDotProductAttention(query, key, value, mask, scale, out attentionWeights);
            float sc = (float)(scale ?? (1.0 / System.Math.Sqrt(headDim)));
            int outLen = batch * numHeads * seqLen * headDim;
            int awLen = batch * numHeads * seqLen * seqLen;
            using var qB = GetOrAllocateBuffer(backend, query);
            using var kB = GetOrAllocateBuffer(backend, key);
            using var vB = GetOrAllocateBuffer(backend, value);
            using var outB = AllocateOutputBuffer(backend, outLen);
            using var awB = AllocateOutputBuffer(backend, awLen);
            backend.ScaledDotProductAttention(qB.Buffer, kB.Buffer, vB.Buffer, outB.Buffer, awB.Buffer, null,
                batch, numHeads, seqLen, headDim, sc, false);
            float[] awArr = backend.DownloadBuffer(awB.Buffer);
            float[] outArr = backend.DownloadBuffer(outB.Buffer);
            attentionWeights = new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(awArr), new[] { batch, numHeads, seqLen, seqLen });
            return new Tensor<T>(DirectGpuEngine.FromFloatArray<T>(outArr), new[] { batch, numHeads, seqLen, headDim });
        }
        catch { return base.ScaledDotProductAttention(query, key, value, mask, scale, out attentionWeights); }
    }

    /// <inheritdoc/>
    Tensor<T> IEngine.Gather<T>(Tensor<T> input, Tensor<int> indices, int axis)
    {
        // GPU gather_kernel is an axis-0 row-gather (output row i = source row indices[i]); other axes /
        // n-D index tensors / tape / graph / non-float defer to the base.
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || axis != 0 || indices.Rank != 1 || input.Rank < 1 || input.Shape._dims[0] <= 0 || !TryGetBackend(out var backend))
            return base.Gather(input, indices, axis);
        try
        {
            int n = input.Shape._dims[0];
            int featureSize = input.Length / n;
            int numIndices = indices.Length;
            var idxData = indices.GetDataArray();
            // gather_kernel reads INT indices (aligned with the 8 other backend.Gather callers, which all
            // upload via AllocateIntBuffer).
            using var srcB = GetOrAllocateBuffer(backend, input);
            using var idxB = new OwnedBuffer(backend.AllocateIntBuffer(idxData), ownsBuffer: true);
            var outB = AllocateOutputBuffer(backend, numIndices * featureSize);
            try
            {
                backend.Gather(srcB.Buffer, idxB.Buffer, outB.Buffer, numIndices, featureSize);
                var outShape = new int[input.Rank];
                outShape[0] = numIndices;
                for (int i = 1; i < input.Rank; i++) outShape[i] = input.Shape._dims[i];
                var arr = FinishGpuOp<T>(backend, outB, numIndices * featureSize);
                return new Tensor<T>(arr, outShape);
            }
            catch { outB.Dispose(); throw; }
        }
        catch { return base.Gather(input, indices, axis); }
    }

    // #775: GPU compositions for genuinely CPU-bound element-wise ops. Each CpuEngine body runs a raw
    // NumOps span loop (it reaches no virtual primitive), so without these it executes on the CPU even
    // on a GPU engine — a silent CPU fallback the parity harness would score as a false CPU-vs-CPU pass.
    // Composing from the GPU-resident public overrides (TensorMultiply/Add/Negate/AddScalar) moves them
    // on-device for EVERY backend at once (each backend supplies its own primitives), and mirrors the
    // CpuEngine op ORDER so the fp32 result is bit-identical. Defer to base under a live tape / GraphMode
    // so the recorded backward graph is unchanged.

    Tensor<T> IEngine.ComplexMagnitudeSquared<T>(Tensor<T> real, Tensor<T> imag)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.ComplexMagnitudeSquared(real, imag);
        // re*re + im*im — same two-multiply-then-add order as CpuEngine.
        return TensorAdd(TensorMultiply(real, real), TensorMultiply(imag, imag));
    }

    Tensor<T> IEngine.SigmoidDerivative<T>(Tensor<T> sigmoidOutput)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.SigmoidDerivative(sigmoidOutput);
        // y * (1 - y): negate, +1, multiply — mirrors CpuEngine's op sequence exactly.
        var numOps = MathHelper.GetNumericOperations<T>();
        var oneMinusY = TensorAddScalar(TensorNegate(sigmoidOutput), numOps.One);
        return TensorMultiply(sigmoidOutput, oneMinusY);
    }

    Tensor<T> IEngine.TanhDerivative<T>(Tensor<T> tanhOutput)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TanhDerivative(tanhOutput);
        // 1 - y*y: square, negate, +1 — mirrors CpuEngine's op sequence exactly.
        var numOps = MathHelper.GetNumericOperations<T>();
        return TensorAddScalar(TensorNegate(TensorMultiply(tanhOutput, tanhOutput)), numOps.One);
    }

    Tensor<T> IEngine.ScalarMinusTensor<T>(T scalar, Tensor<T> tensor)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.ScalarMinusTensor(scalar, tensor);
        // scalar - x = (-x) + scalar — negate then AddScalar, mirroring CpuEngine exactly.
        return TensorAddScalar(TensorNegate(tensor), scalar);
    }

    Tensor<T> IEngine.TensorOuterProduct<T>(Tensor<T> a, Tensor<T> b)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TensorOuterProduct(a, b);
        // outer(a,b)[i,j] = a_i * b_j == (a as [n,1]) @ (b as [1,m]). The contraction dim is 1, so it
        // is a single multiply per element (no accumulation) — bit-identical to the CpuEngine loop and
        // fully GPU-resident via TensorMatMul. Flatten to 1-D first (OuterProduct treats both as vectors).
        int n = a.Length, m = b.Length;
        var af = a.IsContiguous ? a : a.Contiguous();
        var bf = b.IsContiguous ? b : b.Contiguous();
        return TensorMatMul(af.Reshape(new[] { n, 1 }), bf.Reshape(new[] { 1, m }));
    }

    // #775: BatchNormAffine is inference-mode affine batch-norm — y = gamma*(x-mean)/sqrt(var+eps)+beta
    // using the PROVIDED running mean/variance. That is exactly BatchNormInference, which has a dedicated
    // on-device kernel. CpuEngine.BatchNormAffine instead routes through BatchNormInferenceInto, whose
    // allocate-free Into variant has no GPU override, so it runs on the host even on a GPU engine. Route
    // to the resident BatchNormInference override (identical computation → same result). Defer under
    // tape/GraphMode/non-rank-4 to base, which BatchNormInference itself also does.
    public override Tensor<T> BatchNormAffine<T>(Tensor<T> x, Tensor<T> gamma, Tensor<T> beta, Tensor<T> mean, Tensor<T> variance, double epsilon)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || x.Rank != 4)
            return base.BatchNormAffine(x, gamma, beta, mean, variance, epsilon);
        return BatchNormInference(x, gamma, beta, mean, variance, epsilon);
    }

    // #775: population variance over the given axes, composed to run on-device. Mirrors CpuEngine's
    // TWO-PASS algorithm E[(x-mean)^2] (not the one-pass E[x^2]-E[x]^2, which cancels), so it tracks
    // the CPU result tightly: reduce the mean keeping dims, broadcast-subtract, square, reduce again.
    // ReduceMean and the elementwise ops are GPU-resident; the per-outer-group broadcast-subtract runs
    // on the host but stays deterministic. Defer to base under a live tape/GraphMode (which also records
    // the dedicated ReduceVariance backward node).
    Tensor<T> IEngine.ReduceVariance<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.ReduceVariance(input, axes, keepDims);
        var mean = ReduceMean(input, axes, keepDims: true);
        var centered = TensorBroadcastSubtract(input, mean);
        var sq = TensorMultiply(centered, centered);
        return ReduceMean(sq, axes, keepDims);
    }

    // #775: log-variance = log(variance + eps). Reuse the on-device ReduceVariance compose (via the
    // interface so the explicit override is hit, not the inherited CpuEngine method), then AddScalar +
    // TensorLog — same op order as CpuEngine.ReduceLogVariance. Defer to base under tape/GraphMode.
    Tensor<T> IEngine.ReduceLogVariance<T>(Tensor<T> input, int[] axes, bool keepDims, double epsilon)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.ReduceLogVariance(input, axes, keepDims, epsilon);
        var numOps = MathHelper.GetNumericOperations<T>();
        var variance = ((IEngine)this).ReduceVariance(input, axes, keepDims);
        return TensorLog(TensorAddScalar(variance, numOps.FromDouble(epsilon)));
    }

    // #775: reduce-std = sqrt(reduce-variance), exactly as CpuEngine. On a GPU engine CpuEngine.ReduceStd's
    // `this.ReduceVariance` binds to the CPU method (ReduceVariance is an explicit-interface override), so
    // ReduceStd ran on the host; route ReduceVariance through the interface to hit the on-device compose,
    // then TensorSqrt. Defer to base under tape/GraphMode.
    Tensor<T> IEngine.ReduceStd<T>(Tensor<T> input, int[] axes, bool keepDims)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.ReduceStd(input, axes, keepDims);
        return TensorSqrt(((IEngine)this).ReduceVariance(input, axes, keepDims));
    }

    // #775: whole-tensor population variance/std (output [1]). Reduce over ALL axes via the on-device
    // ReduceVariance compose (through the interface), reshape to [1]; std = sqrt(var). Same two-pass
    // population formula as CpuEngine. Defer to base under tape/GraphMode.
    Tensor<T> IEngine.TensorVar<T>(Tensor<T> tensor)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TensorVar(tensor);
        var allAxes = new int[tensor.Rank];
        for (int i = 0; i < tensor.Rank; i++) allAxes[i] = i;
        return ((IEngine)this).ReduceVariance(tensor, allAxes, keepDims: false).Reshape(new[] { 1 });
    }

    Tensor<T> IEngine.TensorStd<T>(Tensor<T> tensor)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TensorStd(tensor);
        return TensorSqrt(((IEngine)this).TensorVar(tensor));
    }

    // #775: MSE loss = mean((pred - target)^2), output [1]. Subtract -> square -> reduce-mean over all
    // axes (all GPU-resident), reshape to [1]. Defer to base under tape/GraphMode.
    Tensor<T> IEngine.TensorMSELoss<T>(Tensor<T> predictions, Tensor<T> targets)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TensorMSELoss(predictions, targets);
        var diff = TensorSubtract(predictions, targets);
        var sq = TensorMultiply(diff, diff);
        var allAxes = new int[sq.Rank];
        for (int i = 0; i < sq.Rank; i++) allAxes[i] = i;
        return ReduceMean(sq, allAxes, keepDims: false).Reshape(new[] { 1 });
    }

    // #775: order-2 Taylor softmax = normalize(1 + s + s^2/2) along `axis`, s = x - max(x). Only order 2
    // is accelerated: its polynomial 0.5*(s+1)^2 + 0.5 >= 0.5 > 0, so the axis-sum is strictly positive
    // and CpuEngine's sumExp==0 fallback is provably dead — matching the CPU value under Accum(1e-3).
    // Other orders (polynomial may vanish/negate) and tape/GraphMode defer to base. The per-row
    // max/sum broadcast runs on the host (deterministic); the poly + reductions are GPU-resident.
    Tensor<T> IEngine.TaylorSoftmax<T>(Tensor<T> input, int order, int axis)
    {
        int ax = axis < 0 ? input.Rank + axis : axis;
        if (order != 2 || IsTapeActive<T>() || Compilation.GraphMode.IsActive || ax < 0 || ax >= input.Rank)
            return base.TaylorSoftmax(input, order, axis);
        var axes = new[] { ax };
        var numOps = MathHelper.GetNumericOperations<T>();
        var max = ReduceMax(input, axes, keepDims: true, out _);
        var shifted = TensorBroadcastSubtract(input, max);                 // s = x - max
        var sq = TensorMultiply(shifted, shifted);                         // s^2
        // 1 + s + s^2/2 (0.5 is exact, so *0.5 == CpuEngine's /2!).
        var poly = TensorAddScalar(TensorAdd(shifted, TensorMultiplyScalar(sq, numOps.FromDouble(0.5))), numOps.One);
        var sum = ReduceSum(poly, axes, keepDims: true);
        return TensorBroadcastDivide(poly, sum);
    }

    // #775: spherical softmax = Softmax(x / ||x||_2) along `axis`. L2-normalize on-device (square ->
    // axis-sum -> sqrt -> broadcast-divide) then the GPU-resident TensorSoftmax. For Rand inputs the
    // norm is strictly positive so CpuEngine's norm==0 guard is dead; defer to base under tape/GraphMode.
    Tensor<T> IEngine.SphericalSoftmax<T>(Tensor<T> input, int axis)
    {
        int ax = axis < 0 ? input.Rank + axis : axis;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || ax < 0 || ax >= input.Rank)
            return base.SphericalSoftmax(input, axis);
        var axes = new[] { ax };
        var sumSq = ReduceSum(TensorMultiply(input, input), axes, keepDims: true);
        var normalized = TensorBroadcastDivide(input, TensorSqrt(sumSq));
        return ((IEngine)this).TensorSoftmax(normalized, ax);
    }

    // #775: element-wise binary cross-entropy -(t*log(pc) + (1-t)*log(1-pc)), pc = clamp(p, eps, 1-eps).
    // Composes from GPU-resident TensorClamp/Log/Multiply/Add/Negate (1-x via Negate+AddScalar), matching
    // CpuEngine's clamp bounds and term order under Accum(1e-3). Defer to base under tape/GraphMode.
    Tensor<T> IEngine.TensorBinaryCrossEntropy<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TensorBinaryCrossEntropy(predictions, targets, epsilon);
        var numOps = MathHelper.GetNumericOperations<T>();
        var upper = numOps.Subtract(numOps.One, epsilon);
        var pc = TensorClamp(predictions, epsilon, upper);                                 // clamp(p, eps, 1-eps)
        var logP = TensorLog(pc);
        var log1mP = TensorLog(TensorAddScalar(TensorNegate(pc), numOps.One));             // log(1 - pc)
        var oneMinusT = TensorAddScalar(TensorNegate(targets), numOps.One);                // 1 - t
        var loss = TensorAdd(TensorMultiply(targets, logP), TensorMultiply(oneMinusT, log1mP));
        return TensorNegate(loss);
    }

    // #775: BCE gradient d/dp = (1-t)/(1-pc) - t/pc, pc = clamp(p, eps, 1-eps). Element-wise compose
    // from GPU-resident TensorClamp/Divide/Subtract (1-x via Negate+AddScalar), same clamp + term order
    // as CpuEngine. Defer to base under tape/GraphMode.
    Tensor<T> IEngine.TensorBinaryCrossEntropyBackward<T>(Tensor<T> predictions, Tensor<T> targets, T epsilon)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TensorBinaryCrossEntropyBackward(predictions, targets, epsilon);
        var numOps = MathHelper.GetNumericOperations<T>();
        var upper = numOps.Subtract(numOps.One, epsilon);
        var pc = TensorClamp(predictions, epsilon, upper);
        var oneMinusPc = TensorAddScalar(TensorNegate(pc), numOps.One);                    // 1 - pc
        var oneMinusT = TensorAddScalar(TensorNegate(targets), numOps.One);                // 1 - t
        return TensorSubtract(TensorDivide(oneMinusT, oneMinusPc), TensorDivide(targets, pc));
    }

    // #775: GLU backward. GLU(input) = a * sigmoid(b) where a,b are the halves of `input` split along
    // `dim`. So d/da = sigmoid(b)*gradOut and d/db = a*sigmoid(b)*(1-sigmoid(b))*gradOut, concatenated
    // back. Fully on-device: TensorNarrow (split) + TensorSigmoid + elementwise + TensorConcatenate are
    // all GPU-resident. Same multiply order as CpuEngine. Defer to base under tape/GraphMode/odd split.
    Tensor<T> IEngine.GLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim)
    {
        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || actualDim < 0 || actualDim >= input.Rank
            || input.Shape._dims[actualDim] % 2 != 0)
            return base.GLUBackward(gradOutput, input, dim);
        int half = input.Shape._dims[actualDim] / 2;
        var numOps = MathHelper.GetNumericOperations<T>();
        var a = TensorNarrow(input, actualDim, 0, half);           // value half
        var b = TensorNarrow(input, actualDim, half, half);        // gate half
        var sigB = TensorSigmoid(b);
        var gradA = TensorMultiply(sigB, gradOutput);              // d/da = sigmoid(b)*gradOut
        var oneMinusSig = TensorAddScalar(TensorNegate(sigB), numOps.One);
        var gradB = TensorMultiply(TensorMultiply(TensorMultiply(a, sigB), oneMinusSig), gradOutput);
        return TensorConcatenate(new[] { gradA, gradB }, actualDim);
    }

    // #775: SwiGLU backward. SwiGLU(input) = a * swish(b), swish(b) = b*sigmoid(b). d/da = swish(b)*gradOut;
    // d/db = a * swish'(b) * gradOut, swish'(b) = sigmoid(b) + b*sigmoid(b)*(1-sigmoid(b)). Fully on-device,
    // same op order as CpuEngine. Defer to base under tape/GraphMode/odd split.
    Tensor<T> IEngine.SwiGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim)
    {
        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || actualDim < 0 || actualDim >= input.Rank
            || input.Shape._dims[actualDim] % 2 != 0)
            return base.SwiGLUBackward(gradOutput, input, dim);
        int half = input.Shape._dims[actualDim] / 2;
        var numOps = MathHelper.GetNumericOperations<T>();
        var a = TensorNarrow(input, actualDim, 0, half);
        var b = TensorNarrow(input, actualDim, half, half);
        var sigB = TensorSigmoid(b);
        var swishB = TensorMultiply(b, sigB);                                        // b*sigmoid(b)
        var oneMinusSig = TensorAddScalar(TensorNegate(sigB), numOps.One);
        // swish'(b) = sigmoid(b) + (b*sigmoid(b))*(1-sigmoid(b))
        var swishDeriv = TensorAdd(sigB, TensorMultiply(swishB, oneMinusSig));
        var gradA = TensorMultiply(swishB, gradOutput);
        var gradB = TensorMultiply(TensorMultiply(a, swishDeriv), gradOutput);
        return TensorConcatenate(new[] { gradA, gradB }, actualDim);
    }

    // #775: ReGLU backward. ReGLU(input) = a * relu(b). d/da = relu(b)*gradOut = (b*gradOut) masked by
    // (b>0); d/db = a*(b>0)*gradOut = (a*gradOut) masked by (b>0). Both maskings ARE ReluBackward(g,b) =
    // g*(b>0), which is GPU-resident — so the whole op stays on-device. Defer to base under tape/GraphMode.
    Tensor<T> IEngine.ReGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim)
    {
        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || actualDim < 0 || actualDim >= input.Rank
            || input.Shape._dims[actualDim] % 2 != 0)
            return base.ReGLUBackward(gradOutput, input, dim);
        int half = input.Shape._dims[actualDim] / 2;
        var a = TensorNarrow(input, actualDim, 0, half);
        var b = TensorNarrow(input, actualDim, half, half);
        var gradA = ReluBackward(TensorMultiply(b, gradOutput), b);   // relu(b)*gradOut
        var gradB = ReluBackward(TensorMultiply(a, gradOutput), b);   // a*(b>0)*gradOut
        return TensorConcatenate(new[] { gradA, gradB }, actualDim);
    }

    // #775: GeGLU backward. GeGLU(input) = a * gelu(b). d/da = gelu(b)*gradOut; d/db = a*gelu'(b)*gradOut,
    // and gelu'(b)*(a*gradOut) IS GeluBackward(a*gradOut, b), which is GPU-resident (same gelu the GPU
    // GELU forward uses). Fully on-device. Defer to base under tape/GraphMode/odd split.
    Tensor<T> IEngine.GeGLUBackward<T>(Tensor<T> gradOutput, Tensor<T> input, int dim)
    {
        int actualDim = dim < 0 ? input.Rank + dim : dim;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || actualDim < 0 || actualDim >= input.Rank
            || input.Shape._dims[actualDim] % 2 != 0)
            return base.GeGLUBackward(gradOutput, input, dim);
        int half = input.Shape._dims[actualDim] / 2;
        var a = TensorNarrow(input, actualDim, 0, half);
        var b = TensorNarrow(input, actualDim, half, half);
        var gradA = TensorMultiply(GELU(b), gradOutput);               // gelu(b)*gradOut
        var gradB = GeluBackward(TensorMultiply(a, gradOutput), b);    // a*gelu'(b)*gradOut
        return TensorConcatenate(new[] { gradA, gradB }, actualDim);
    }

    // #775: Gumbel-softmax backward = softmax backward scaled by 1/temperature (the gradient flows
    // through the softmax, divided by the temperature). Reuse the GPU-resident SoftmaxBackward (via the
    // interface so the override is hit) then TensorMultiplyScalar. Defer to base under tape/GraphMode.
    Tensor<T> IEngine.GumbelSoftmaxBackward<T>(Tensor<T> gradOutput, Tensor<T> output, double temperature, int axis)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || temperature <= 0)
            return base.GumbelSoftmaxBackward(gradOutput, output, temperature, axis);
        var numOps = MathHelper.GetNumericOperations<T>();
        var softmaxGrad = ((IEngine)this).SoftmaxBackward(gradOutput, output, axis);
        return TensorMultiplyScalar(softmaxGrad, numOps.FromDouble(1.0 / temperature));
    }

    // #775: complex conjugate of an interleaved [.., re, im, re, im, ...] tensor negates every imaginary
    // (odd) element. When the last axis is even the pairs live within it, so split it into (pairs, 2),
    // negate the imaginary column, and re-interleave — all exact, GPU-resident data moves + TensorNegate.
    // Odd last-axis (pairs straddle rows) and tape/GraphMode defer to base.
    Tensor<T> IEngine.TensorComplexConjugate<T>(Tensor<T> a)
    {
        int rank = a.Rank;
        int lastDim = rank > 0 ? a.Shape._dims[rank - 1] : 0;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || rank == 0 || lastDim % 2 != 0)
            return base.TensorComplexConjugate(a);
        var pairShape = new int[rank + 1];
        for (int i = 0; i < rank - 1; i++) pairShape[i] = a.Shape._dims[i];
        pairShape[rank - 1] = lastDim / 2;
        pairShape[rank] = 2;
        var reshaped = (a.IsContiguous ? a : a.Contiguous()).Reshape(pairShape);
        var re = TensorNarrow(reshaped, rank, 0, 1);                 // real column
        var im = TensorNarrow(reshaped, rank, 1, 1);                 // imaginary column
        var conj = TensorConcatenate(new[] { re, TensorNegate(im) }, rank);
        return conj.Reshape(a.Shape._dims);
    }

    // #775: complex magnitude sqrt(re^2 + im^2) per interleaved pair. Same even-last-axis de-interleave
    // (split into (pairs, 2), square + add the two columns, sqrt), then flatten to the CpuEngine's 1-D
    // [pairs] output. Fully GPU-resident (Narrow/Multiply/Add/Sqrt). Defers to base otherwise.
    Tensor<T> IEngine.TensorComplexMagnitude<T>(Tensor<T> a)
    {
        int rank = a.Rank;
        int lastDim = rank > 0 ? a.Shape._dims[rank - 1] : 0;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || rank == 0 || lastDim % 2 != 0)
            return base.TensorComplexMagnitude(a);
        var pairShape = new int[rank + 1];
        for (int i = 0; i < rank - 1; i++) pairShape[i] = a.Shape._dims[i];
        pairShape[rank - 1] = lastDim / 2;
        pairShape[rank] = 2;
        var reshaped = (a.IsContiguous ? a : a.Contiguous()).Reshape(pairShape);
        var re = TensorNarrow(reshaped, rank, 0, 1);
        var im = TensorNarrow(reshaped, rank, 1, 1);
        var mag = TensorSqrt(TensorAdd(TensorMultiply(re, re), TensorMultiply(im, im)));
        return mag.Reshape(new[] { a.Length / 2 });   // CpuEngine returns a flat [pairs] vector
    }

    // #775: complex multiply (aRe+aIm i)(bRe+bIm i) = (aRe*bRe - aIm*bIm) + (aRe*bIm + aIm*bRe) i, both
    // operands interleaved. De-interleave the even last axis of each (metadata reshape + TensorNarrow),
    // form the real/imag results with GPU-resident Multiply/Subtract/Add in CpuEngine's order, then
    // re-interleave. Defers to base on odd last-axis / length-mismatch / tape / GraphMode.
    Tensor<T> IEngine.TensorComplexMultiply<T>(Tensor<T> a, Tensor<T> b)
    {
        int rank = a.Rank;
        int lastDim = rank > 0 ? a.Shape._dims[rank - 1] : 0;
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || rank == 0 || lastDim % 2 != 0 || a.Length != b.Length)
            return base.TensorComplexMultiply(a, b);
        var pairShape = new int[rank + 1];
        for (int i = 0; i < rank - 1; i++) pairShape[i] = a.Shape._dims[i];
        pairShape[rank - 1] = lastDim / 2;
        pairShape[rank] = 2;
        var ra = (a.IsContiguous ? a : a.Contiguous()).Reshape(pairShape);
        var rb = (b.IsContiguous ? b : b.Contiguous()).Reshape(pairShape);
        var aRe = TensorNarrow(ra, rank, 0, 1); var aIm = TensorNarrow(ra, rank, 1, 1);
        var bRe = TensorNarrow(rb, rank, 0, 1); var bIm = TensorNarrow(rb, rank, 1, 1);
        var outRe = TensorSubtract(TensorMultiply(aRe, bRe), TensorMultiply(aIm, bIm));
        var outIm = TensorAdd(TensorMultiply(aRe, bIm), TensorMultiply(aIm, bRe));
        return TensorConcatenate(new[] { outRe, outIm }, rank).Reshape(a.Shape._dims);
    }

    // #775: fused linear backward for the no-activation case. With FusedActivationType.None the activation
    // gradient is a passthrough (gradActivation = gradOutput), so gradInput = gradOut @ Wᵀ, gradWeights =
    // inputᵀ @ gradOut, gradBias = sum over batch — all GPU-resident (TensorMatMul/TensorTranspose/ReduceSum),
    // matching CpuEngine's steps. Fused activations and non-2D defer to base (which runs its activation-backward).
    Tensor<T> IEngine.FusedLinearBackward<T>(Tensor<T> gradOutput, Tensor<T> input, Tensor<T> weights,
        Tensor<T> preActivation, FusedActivationType activation, out Tensor<T> gradWeights, out Tensor<T>? gradBias)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || activation != FusedActivationType.None
            || gradOutput.Rank != 2)
            return base.FusedLinearBackward(gradOutput, input, weights, preActivation, activation, out gradWeights, out gradBias);
        gradBias = ReduceSum(gradOutput, new[] { 0 }, false);                       // sum over batch
        gradWeights = TensorMatMul(TensorTranspose(input), gradOutput);            // inputᵀ @ gradOut
        return TensorMatMul(gradOutput, TensorTranspose(weights));                 // gradOut @ Wᵀ
    }

    // #775: MaxPool3D backward on the existing maxpool3d_backward kernel. The kernel scatters each output
    // gradient into gradInput at a precomputed FLAT input index; the engine gets a 6D maxIndices[b,c,od,oh,
    // ow,{0,1,2}] = the (d,h,w) of the max, so flatten it to that per-output flat index (host, small) then
    // scatter on-device. gradInput is zeroed first (scatter-add). Non-float / tape / GraphMode defer to base.
    Tensor<T> IEngine.MaxPool3DBackward<T>(Tensor<T> gradOutput, int[,,,,,] maxIndices,
        int[] inputShape, int[] poolSize, int[] stride)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || inputShape is not { Length: 5 } || !TryGetBackend(out var backend))
            return base.MaxPool3DBackward(gradOutput, maxIndices, inputShape, poolSize, stride);
        try
        {
            int batch = inputShape[0], channels = inputShape[1], depth = inputShape[2], height = inputShape[3], width = inputShape[4];
            int outD = maxIndices.GetLength(2), outH = maxIndices.GetLength(3), outW = maxIndices.GetLength(4);
            int outLen = batch * channels * outD * outH * outW;
            int inLen = batch * channels * depth * height * width;
            var flat = new int[outLen];
            int p = 0;
            for (int b = 0; b < batch; b++)
                for (int c = 0; c < channels; c++)
                    for (int od = 0; od < outD; od++)
                        for (int oh = 0; oh < outH; oh++)
                            for (int ow = 0; ow < outW; ow++)
                            {
                                int id = maxIndices[b, c, od, oh, ow, 0];
                                int ih = maxIndices[b, c, od, oh, ow, 1];
                                int iw = maxIndices[b, c, od, oh, ow, 2];
                                // maxpool3d_backward decodes indices[] as a SPATIAL-only flat index
                                // (id*inH*inW + ih*inW + iw) and adds the b,c offset itself.
                                flat[p++] = (id * height + ih) * width + iw;
                            }
            using var gradOutBuf = GetOrAllocateBuffer(backend, gradOutput);
            using var idxBuf = new OwnedBuffer(backend.AllocateIntBuffer(flat), ownsBuffer: true);
            var gradInBuf = AllocateOutputBuffer(backend, inLen);
            try
            {
                backend.Fill(gradInBuf.Buffer, 0f, inLen);
                backend.MaxPool3DBackward(gradOutBuf.Buffer, idxBuf.Buffer, gradInBuf.Buffer,
                    batch, channels, depth, height, width, outD, outH, outW);
                var arr = FinishGpuOp<T>(backend, gradInBuf, inLen);
                return new Tensor<T>(arr, (int[])inputShape.Clone());
            }
            catch { gradInBuf.Dispose(); throw; }
        }
        catch { return base.MaxPool3DBackward(gradOutput, maxIndices, inputShape, poolSize, stride); }
    }

    // #775: AvgPool3D (NCDHW, no-padding). Scalar overload routes to the int[] one through the interface
    // (a direct this.AvgPool3D call would bypass the explicit impl). Padded / non-float / wrong-rank /
    // tape / GraphMode defer to base.
    Tensor<T> IEngine.AvgPool3D<T>(Tensor<T> input, int poolSize, int stride, int padding)
    {
        if (stride == 0) stride = poolSize;
        return ((IEngine)this).AvgPool3D(input, new[] { poolSize, poolSize, poolSize },
            new[] { stride, stride, stride }, new[] { padding, padding, padding });
    }

    Tensor<T> IEngine.AvgPool3D<T>(Tensor<T> input, int[] poolSize, int[] stride, int[] padding)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || input.Rank != 5 || poolSize is not { Length: 3 } || stride is not { Length: 3 }
            || padding is not { Length: 3 } || padding[0] != 0 || padding[1] != 0 || padding[2] != 0
            || !TryGetBackend(out var backend))
            return base.AvgPool3D(input, poolSize, stride, padding);
        try
        {
            int batch = input.Shape._dims[0], channels = input.Shape._dims[1];
            int inD = input.Shape._dims[2], inH = input.Shape._dims[3], inW = input.Shape._dims[4];
            int kD = poolSize[0], kH = poolSize[1], kW = poolSize[2];
            int sD = stride[0], sH = stride[1], sW = stride[2];
            if (sD <= 0 || sH <= 0 || sW <= 0) return base.AvgPool3D(input, poolSize, stride, padding);
            int outD = (inD - kD) / sD + 1, outH = (inH - kH) / sH + 1, outW = (inW - kW) / sW + 1;
            if (outD <= 0 || outH <= 0 || outW <= 0) return base.AvgPool3D(input, poolSize, stride, padding);
            int outLen = batch * channels * outD * outH * outW;
            using var inBuf = GetOrAllocateBuffer(backend, input);
            var outBuf = AllocateOutputBuffer(backend, outLen);
            try
            {
                // No padding, so every window is full -> countIncludePad is irrelevant; pass 1.
                backend.AvgPool3D(inBuf.Buffer, outBuf.Buffer, batch, channels, inD, inH, inW,
                    outD, outH, outW, kD, kH, kW, sD, sH, sW, countIncludePad: 1);
                var arr = FinishGpuOp<T>(backend, outBuf, outLen);
                return new Tensor<T>(arr, new[] { batch, channels, outD, outH, outW });
            }
            catch { outBuf.Dispose(); throw; }
        }
        catch { return base.AvgPool3D(input, poolSize, stride, padding); }
    }

    // #775: AvgPool3D backward (NCDHW, no-padding). The avgpool3d_backward kernel parallelizes over input
    // elements and WRITES each gradInput cell (gather, not scatter) so no pre-zeroing. Defer padded /
    // non-float / wrong-rank / tape / GraphMode to base.
    Tensor<T> IEngine.AvgPool3DBackward<T>(Tensor<T> gradOutput, int[] inputShape, int[] poolSize, int[] stride, int[] padding)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || inputShape is not { Length: 5 } || poolSize is not { Length: 3 } || stride is not { Length: 3 }
            || padding is not { Length: 3 } || padding[0] != 0 || padding[1] != 0 || padding[2] != 0
            || gradOutput.Rank != 5 || !TryGetBackend(out var backend))
            return base.AvgPool3DBackward(gradOutput, inputShape, poolSize, stride, padding);
        try
        {
            int batch = inputShape[0], channels = inputShape[1], inD = inputShape[2], inH = inputShape[3], inW = inputShape[4];
            int outD = gradOutput.Shape._dims[2], outH = gradOutput.Shape._dims[3], outW = gradOutput.Shape._dims[4];
            int kD = poolSize[0], kH = poolSize[1], kW = poolSize[2];
            int sD = stride[0], sH = stride[1], sW = stride[2];
            int inLen = batch * channels * inD * inH * inW;
            using var gradOutBuf = GetOrAllocateBuffer(backend, gradOutput);
            var gradInBuf = AllocateOutputBuffer(backend, inLen);
            try
            {
                backend.AvgPool3DBackward(gradOutBuf.Buffer, gradInBuf.Buffer, batch, channels, inD, inH, inW,
                    outD, outH, outW, kD, kH, kW, sD, sH, sW, countIncludePad: 1);
                var arr = FinishGpuOp<T>(backend, gradInBuf, inLen);
                return new Tensor<T>(arr, (int[])inputShape.Clone());
            }
            catch { gradInBuf.Dispose(); throw; }
        }
        catch { return base.AvgPool3DBackward(gradOutput, inputShape, poolSize, stride, padding); }
    }

    // #775: ConvTranspose2D backward w.r.t. input. Backend kernel gathers per gradInput element (1D over
    // input, no zeroing). inputShape=[N,inC,inH,inW]; gradOutput=[N,outC,outH,outW]; kernel=[inC,outC,kH,kW].
    // The engine op carries no output-padding, so pass 0. Non-float / wrong-rank / tape / GraphMode -> base.
    Tensor<T> IEngine.ConvTranspose2DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || inputShape is not { Length: 4 } || gradOutput.Rank != 4 || kernel.Rank != 4
            || stride is not { Length: 2 } || padding is not { Length: 2 } || !TryGetBackend(out var backend))
            return base.ConvTranspose2DBackwardInput(gradOutput, kernel, inputShape, stride, padding);
        try
        {
            int batch = inputShape[0], inChannels = inputShape[1], inH = inputShape[2], inW = inputShape[3];
            int outChannels = gradOutput.Shape._dims[1], outH = gradOutput.Shape._dims[2], outW = gradOutput.Shape._dims[3];
            int kH = kernel.Shape._dims[2], kW = kernel.Shape._dims[3];
            int inLen = batch * inChannels * inH * inW;
            using var gradOutBuf = GetOrAllocateBuffer(backend, gradOutput);
            using var kBuf = GetOrAllocateBuffer(backend, kernel);
            var gradInBuf = AllocateOutputBuffer(backend, inLen);
            try
            {
                backend.ConvTranspose2DBackwardInput(gradOutBuf.Buffer, kBuf.Buffer, gradInBuf.Buffer,
                    batch, inChannels, inH, inW, outChannels, outH, outW, kH, kW,
                    stride[0], stride[1], padding[0], padding[1], 0, 0);
                var arr = FinishGpuOp<T>(backend, gradInBuf, inLen);
                return new Tensor<T>(arr, (int[])inputShape.Clone());
            }
            catch { gradInBuf.Dispose(); throw; }
        }
        catch { return base.ConvTranspose2DBackwardInput(gradOutput, kernel, inputShape, stride, padding); }
    }

    // #775: ConvTranspose2D backward w.r.t. the kernel. Backend zeroes gradKernel + atomic-accumulates.
    // kernelShape=[inC,outC,kH,kW]; input=[N,inC,inH,inW]; gradOutput=[N,outC,outH,outW]. Backend arg order
    // is (input, gradOutput, ...). No output-padding on the engine op -> pass 0.
    Tensor<T> IEngine.ConvTranspose2DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || kernelShape is not { Length: 4 } || input.Rank != 4 || gradOutput.Rank != 4
            || stride is not { Length: 2 } || padding is not { Length: 2 } || !TryGetBackend(out var backend))
            return base.ConvTranspose2DBackwardKernel(gradOutput, input, kernelShape, stride, padding);
        try
        {
            int batch = input.Shape._dims[0], inChannels = input.Shape._dims[1], inH = input.Shape._dims[2], inW = input.Shape._dims[3];
            int outChannels = gradOutput.Shape._dims[1], outH = gradOutput.Shape._dims[2], outW = gradOutput.Shape._dims[3];
            int kH = kernelShape[2], kW = kernelShape[3];
            int kLen = kernelShape[0] * kernelShape[1] * kernelShape[2] * kernelShape[3];
            using var inBuf = GetOrAllocateBuffer(backend, input);
            using var gradOutBuf = GetOrAllocateBuffer(backend, gradOutput);
            var gradKBuf = AllocateOutputBuffer(backend, kLen);
            try
            {
                backend.ConvTranspose2DBackwardKernel(inBuf.Buffer, gradOutBuf.Buffer, gradKBuf.Buffer,
                    batch, inChannels, inH, inW, outChannels, outH, outW, kH, kW,
                    stride[0], stride[1], padding[0], padding[1], 0, 0);
                var arr = FinishGpuOp<T>(backend, gradKBuf, kLen);
                return new Tensor<T>(arr, (int[])kernelShape.Clone());
            }
            catch { gradKBuf.Dispose(); throw; }
        }
        catch { return base.ConvTranspose2DBackwardKernel(gradOutput, input, kernelShape, stride, padding); }
    }

    // #775: Conv3D backward w.r.t. input (no dilation). weights=[outC,inC,kD,kH,kW]; inputShape=[N,inC,D,H,W];
    // gradOutput=[N,outC,outD,outH,outW]. Kernel gathers per gradInput element. Dilated/non-float/wrong-rank -> base.
    Tensor<T> IEngine.Conv3DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int[] stride, int[] padding, int[] dilation)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || inputShape is not { Length: 5 } || gradOutput.Rank != 5 || kernel.Rank != 5
            || stride is not { Length: 3 } || padding is not { Length: 3 }
            || dilation is not { Length: 3 } || dilation[0] != 1 || dilation[1] != 1 || dilation[2] != 1
            || !TryGetBackend(out var backend))
            return base.Conv3DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        try
        {
            int n = inputShape[0], inC = inputShape[1], inD = inputShape[2], inH = inputShape[3], inW = inputShape[4];
            int outC = gradOutput.Shape._dims[1], outD = gradOutput.Shape._dims[2], outH = gradOutput.Shape._dims[3], outW = gradOutput.Shape._dims[4];
            int kD = kernel.Shape._dims[2], kH = kernel.Shape._dims[3], kW = kernel.Shape._dims[4];
            int inLen = n * inC * inD * inH * inW;
            using var gradOutBuf = GetOrAllocateBuffer(backend, gradOutput);
            using var kBuf = GetOrAllocateBuffer(backend, kernel);
            var gradInBuf = AllocateOutputBuffer(backend, inLen);
            try
            {
                backend.Conv3DBackwardInput(gradOutBuf.Buffer, kBuf.Buffer, gradInBuf.Buffer,
                    n, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW,
                    stride[0], stride[1], stride[2], padding[0], padding[1], padding[2]);
                var arr = FinishGpuOp<T>(backend, gradInBuf, inLen);
                return new Tensor<T>(arr, (int[])inputShape.Clone());
            }
            catch { gradInBuf.Dispose(); throw; }
        }
        catch { return base.Conv3DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation); }
    }

    // #775: Conv3D backward w.r.t. weights (no dilation). kernelShape=[outC,inC,kD,kH,kW]; input=[N,inC,D,H,W].
    // Kernel gathers per gradKernel element (backend arg order: gradOutput, input, gradKernel).
    Tensor<T> IEngine.Conv3DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int[] stride, int[] padding, int[] dilation)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive || typeof(T) != typeof(float)
            || kernelShape is not { Length: 5 } || input.Rank != 5 || gradOutput.Rank != 5
            || stride is not { Length: 3 } || padding is not { Length: 3 }
            || dilation is not { Length: 3 } || dilation[0] != 1 || dilation[1] != 1 || dilation[2] != 1
            || !TryGetBackend(out var backend))
            return base.Conv3DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        try
        {
            int n = input.Shape._dims[0], inC = input.Shape._dims[1], inD = input.Shape._dims[2], inH = input.Shape._dims[3], inW = input.Shape._dims[4];
            int outC = gradOutput.Shape._dims[1], outD = gradOutput.Shape._dims[2], outH = gradOutput.Shape._dims[3], outW = gradOutput.Shape._dims[4];
            int kD = kernelShape[2], kH = kernelShape[3], kW = kernelShape[4];
            int kLen = kernelShape[0] * kernelShape[1] * kD * kH * kW;
            using var gradOutBuf = GetOrAllocateBuffer(backend, gradOutput);
            using var inBuf = GetOrAllocateBuffer(backend, input);
            var gradKBuf = AllocateOutputBuffer(backend, kLen);
            try
            {
                backend.Conv3DBackwardKernel(gradOutBuf.Buffer, inBuf.Buffer, gradKBuf.Buffer,
                    n, inC, inD, inH, inW, outC, outD, outH, outW, kD, kH, kW,
                    stride[0], stride[1], stride[2], padding[0], padding[1], padding[2]);
                var arr = FinishGpuOp<T>(backend, gradKBuf, kLen);
                return new Tensor<T>(arr, (int[])kernelShape.Clone());
            }
            catch { gradKBuf.Dispose(); throw; }
        }
        catch { return base.Conv3DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation); }
    }

    // #775: Conv1D backward == Conv2D backward with a singleton height (exactly how CpuEngine implements
    // it), so route to the on-device Conv2DBackwardInput/Kernel (the correct, validated conv2d_backward_*
    // kernels) instead of backend.Conv1DBackwardInput (which wrongly launches the im2col kernel and
    // corrupts GPU state). Reshape the 1D tensors to [N,C,1,L], go through the interface to hit the GPU
    // override, reshape the result back. Defer to base under tape/GraphMode/wrong-rank.
    Tensor<T> IEngine.Conv1DBackwardInput<T>(Tensor<T> gradOutput, Tensor<T> kernel, int[] inputShape, int stride, int padding, int dilation)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive
            || inputShape is not { Length: 3 } || gradOutput.Rank != 3 || kernel.Rank != 3)
            return base.Conv1DBackwardInput(gradOutput, kernel, inputShape, stride, padding, dilation);
        var g = gradOutput.Shape._dims; var kd = kernel.Shape._dims;
        var grad4D = gradOutput.Reshape(new[] { g[0], g[1], 1, g[2] });
        var kernel4D = kernel.Reshape(new[] { kd[0], kd[1], 1, kd[2] });
        var result4D = ((IEngine)this).Conv2DBackwardInput(grad4D, kernel4D,
            new[] { inputShape[0], inputShape[1], 1, inputShape[2] },
            new[] { 1, stride }, new[] { 0, padding }, new[] { 1, dilation });
        return result4D.Reshape((int[])inputShape.Clone());
    }

    Tensor<T> IEngine.Conv1DBackwardKernel<T>(Tensor<T> gradOutput, Tensor<T> input, int[] kernelShape, int stride, int padding, int dilation)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive
            || kernelShape is not { Length: 3 } || input.Rank != 3 || gradOutput.Rank != 3)
            return base.Conv1DBackwardKernel(gradOutput, input, kernelShape, stride, padding, dilation);
        var g = gradOutput.Shape._dims; var ind = input.Shape._dims;
        var grad4D = gradOutput.Reshape(new[] { g[0], g[1], 1, g[2] });
        var input4D = input.Reshape(new[] { ind[0], ind[1], 1, ind[2] });
        var result4D = ((IEngine)this).Conv2DBackwardKernel(grad4D, input4D,
            new[] { kernelShape[0], kernelShape[1], 1, kernelShape[2] },
            new[] { 1, stride }, new[] { 0, padding }, new[] { 1, dilation });
        return result4D.Reshape((int[])kernelShape.Clone());
    }

    // #775: whole-tensor mean (output [1]) via the GPU-resident ReduceMean over all axes.
    Tensor<T> IEngine.TensorMeanDiff<T>(Tensor<T> tensor)
    {
        if (IsTapeActive<T>() || Compilation.GraphMode.IsActive)
            return base.TensorMeanDiff(tensor);
        var allAxes = new int[tensor.Rank];
        for (int i = 0; i < tensor.Rank; i++) allAxes[i] = i;
        return ReduceMean(tensor, allAxes, keepDims: false).Reshape(new[] { 1 });
    }

    // #775 NOTE: TensorRepeatElements is NOT wired here. Routing it to TensorRepeatInterleave (last-axis
    // GPU only) or a reshape->TensorTile->reshape identity both leave it on the CPU for a non-last axis
    // (the GpuLaunchProbe measured zero GPU work) — TensorTile/RepeatInterleave/Gather all fall back to
    // the host for these shapes. It needs a real non-last-axis repeat kernel; until then it stays an
    // honest entry on BOTH the reflection worklist and the gpu-cpu-fallback worklist.
}
