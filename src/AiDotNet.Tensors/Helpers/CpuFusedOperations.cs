using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Engines.Simd;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#if NET5_0_OR_GREATER
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Helpers;

/// <summary>
/// Provides SIMD-optimized fused CPU operations that combine multiple operations
/// into single-pass algorithms for better cache efficiency and reduced memory allocation.
/// </summary>
/// <remarks>
/// <para>
/// Unlike GPU fused operations which primarily eliminate kernel launch overhead, CPU fused
/// operations provide performance benefits through:
/// <list type="bullet">
/// <item>Cache locality: Processing data in a single pass keeps it in L1/L2 cache</item>
/// <item>Reduced allocations: No intermediate tensors needed between operations</item>
/// <item>SIMD vectorization: Uses TensorPrimitives for hardware-accelerated operations</item>
/// <item>Tiling: Cache-efficient blocking for large matrix operations</item>
/// </list>
/// </para>
/// <para>
/// <b>Performance Characteristics:</b>
/// <list type="bullet">
/// <item>FusedGemmBiasActivation: 20-40% faster than separate operations</item>
/// <item>FusedLayerNorm: 30-50% faster due to single-pass mean/variance calculation</item>
/// <item>FusedResidualLayerNorm: 40-60% faster for Transformer blocks</item>
/// </list>
/// </para>
/// </remarks>
public static class CpuFusedOperations
{
    // Threshold for parallelization (number of elements)
    private const int PARALLEL_THRESHOLD = 4096;

    #region Fused GEMM + Bias + Activation (Float Arrays)

    /// <summary>
    /// Performs fused GEMM + Bias + ReLU: output = ReLU(A * B + bias).
    /// Single-pass operation with cache-efficient processing.
    /// </summary>
    /// <param name="A">Input matrix (M x K), row-major.</param>
    /// <param name="B">Weight matrix (K x N), row-major.</param>
    /// <param name="bias">Bias vector (N elements), or null.</param>
    /// <param name="output">Output matrix (M x N), row-major.</param>
    /// <param name="M">Number of rows in A.</param>
    /// <param name="N">Number of columns in B.</param>
    /// <param name="K">Shared dimension (columns of A, rows of B).</param>
    public static void FusedGemmBiasRelu(
        float[] A,
        float[] B,
        float[]? bias,
        float[] output,
        int M, int N, int K)
    {
        FusedGemmBiasActivation(A, B, bias, output, M, N, K, FusedActivationType.ReLU);
    }

    /// <summary>
    /// Performs fused GEMM + Bias + GELU: output = GELU(A * B + bias).
    /// Uses the fast tanh approximation for GELU.
    /// </summary>
    public static void FusedGemmBiasGelu(
        float[] A,
        float[] B,
        float[]? bias,
        float[] output,
        int M, int N, int K)
    {
        FusedGemmBiasActivation(A, B, bias, output, M, N, K, FusedActivationType.GELU);
    }

    /// <summary>
    /// Performs fused GEMM + Bias + Sigmoid: output = Sigmoid(A * B + bias).
    /// </summary>
    public static void FusedGemmBiasSigmoid(
        float[] A,
        float[] B,
        float[]? bias,
        float[] output,
        int M, int N, int K)
    {
        FusedGemmBiasActivation(A, B, bias, output, M, N, K, FusedActivationType.Sigmoid);
    }

    /// <summary>
    /// Performs fused GEMM + Bias + Tanh: output = Tanh(A * B + bias).
    /// </summary>
    public static void FusedGemmBiasTanh(
        float[] A,
        float[] B,
        float[]? bias,
        float[] output,
        int M, int N, int K)
    {
        FusedGemmBiasActivation(A, B, bias, output, M, N, K, FusedActivationType.Tanh);
    }

    /// <summary>
    /// Performs fused GEMM + Bias + Activation with configurable activation function.
    /// Uses cache-efficient tiling for large matrices.
    /// </summary>
    /// <param name="A">Input matrix (M x K), row-major.</param>
    /// <param name="B">Weight matrix (K x N), row-major.</param>
    /// <param name="bias">Bias vector (N elements), or null for no bias.</param>
    /// <param name="output">Output matrix (M x N), row-major.</param>
    /// <param name="M">Number of rows in A (batch size).</param>
    /// <param name="N">Number of columns in B (output features).</param>
    /// <param name="K">Shared dimension (input features).</param>
    /// <param name="activation">Activation function to apply.</param>
    public static void FusedGemmBiasActivation(
        float[] A,
        float[] B,
        float[]? bias,
        float[] output,
        int M, int N, int K,
        FusedActivationType activation)
    {
        if (A.Length < M * K)
            throw new ArgumentException($"A must have at least {M * K} elements", nameof(A));
        if (B.Length < K * N)
            throw new ArgumentException($"B must have at least {K * N} elements", nameof(B));
        if (output.Length < M * N)
            throw new ArgumentException($"output must have at least {M * N} elements", nameof(output));
        if (bias != null && bias.Length < N)
            throw new ArgumentException($"bias must have at least {N} elements", nameof(bias));

        // Use BLAS for the O(MNK) GEMM, then fuse bias+activation in a cheap O(MN) second pass.
        if (BlasProvider.TryGemm(M, N, K, A, 0, K, B, 0, N, output, 0, N))
        {
            ApplyBiasActivationInPlace(output, bias, M, N, activation);
            return;
        }

        // BLAS unavailable: Path A's SgemmWithCachedB pre-packs B (the weight)
        // once and amortises over every forward call. Falls through to the
        // standard Sgemm inside SgemmWithCachedB for shapes where caching
        // isn't applicable (n > Nc=4096 or AVX-512 eligible).
        SimdGemm.SgemmWithCachedB(
            A.AsSpan(0, M * K), B, output.AsSpan(0, M * N), M, K, N);
        ApplyBiasActivationInPlace(output, bias, M, N, activation);
    }

    /// <summary>
    /// Applies bias addition and activation function in-place over the GEMM output.
    /// Single O(MN) pass — cheap compared to the O(MNK) GEMM.
    /// </summary>
    /// <summary>
    /// Channel-wise bias + activation for a NCHW Conv2D output. Bias is [C]
    /// — scalar per channel. Output is [N, C, H, W]. Broadcast: each
    /// output[n, c, :, :] gets bias[c] added then activation applied.
    /// Parallelises across (N × C) "planes"; each worker processes HW
    /// elements with a scalar bias broadcast via Vector256.Create + SIMD add.
    /// </summary>
    [MethodImpl(Hot)]
    internal static void ApplyBiasActivationNCHWInPlace(
        float[] output, float[]? bias,
        int N, int C, int H, int W, FusedActivationType activation)
    {
        bool hasBias = bias != null;
        bool hasActivation = activation != FusedActivationType.None;
        if (!hasBias && !hasActivation) return;

        int hw = H * W;
        int totalPlanes = N * C;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && hw >= 8 &&
            (activation == FusedActivationType.None
             || activation == FusedActivationType.ReLU))
        {
            // Per-plane work is `hw` elements ≈ hw/8 AVX256 vector ops.
            // For ResNet-style shapes (C=256, hw=14×14=196) per-plane
            // work is ~50 ns — the Parallel.For dispatch + sync cost
            // (~1 µs per task in a contested thread pool) is >>
            // the work it dispatches, and the fused path becomes
            // slower than the unfused (Conv + BroadcastAdd) reference.
            // CI runners with 2-4 vCPUs see this as a 5× regression.
            // Gate parallelism on TOTAL element count: only spin up
            // workers when the work amortises the dispatch overhead.
            // 65536 ≈ 8 µs of SIMD work — a comfortable floor.
            const long ParallelElementThreshold = 65536L;
            long totalElements = (long)totalPlanes * hw;
            if (totalPlanes >= 4 && totalElements >= ParallelElementThreshold)
            {
                Parallel.For(0, totalPlanes, p =>
                    ApplyNchwPlaneSimd(output, bias, p, C, hw, activation == FusedActivationType.ReLU));
            }
            else
            {
                for (int p = 0; p < totalPlanes; p++)
                    ApplyNchwPlaneSimd(output, bias, p, C, hw, activation == FusedActivationType.ReLU);
            }
            return;
        }
#endif

        Func<float, float>? activationFn = hasActivation ? GetFloatActivation(activation) : null;
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                float b = hasBias ? bias![c] : 0f;
                int offset = (n * C + c) * hw;
                for (int i = 0; i < hw; i++)
                {
                    float val = output[offset + i];
                    if (hasBias) val += b;
                    if (activationFn != null) val = activationFn(val);
                    output[offset + i] = val;
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    // ──────────────────────────────────────────────────────────────────────
    // NaN-preserving ReLU primitives.
    //
    // x86 MAXPS / MAXPD / VMAXPS / VMAXPD all return SRC2 when SRC1 is NaN
    // (Intel SDM Vol.2, MAXPD pseudocode). So `Avx.Max(v, vZero)` silently
    // turns every NaN lane into +0.0 — divergent from the scalar fallback
    // `val < 0 ? 0 : val`, where `NaN < 0` is false (ordered) so NaN is
    // preserved. That divergence makes numerical results depend on which
    // CPU SIMD level is active and silently masks invalid activations
    // during training.
    //
    // The compare-and-AndNot form below mirrors the scalar semantics on
    // every SIMD path: build a mask of the strict-negative lanes (NaN
    // gives ordered-false → mask 0), then `(NOT mask) AND v` keeps the
    // bit pattern of v where v >= 0 OR v is NaN, and zeroes lanes where
    // v < 0. NaN bit patterns survive identically.
    // ──────────────────────────────────────────────────────────────────────
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ReluNaNSafeAvx(Vector256<float> v, Vector256<float> vZero)
    {
        var ltZero = Avx.Compare(v, vZero, FloatComparisonMode.OrderedLessThanNonSignaling);
        return Avx.AndNot(ltZero, v);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<double> ReluNaNSafeAvx(Vector256<double> v, Vector256<double> vZero)
    {
        var ltZero = Avx.Compare(v, vZero, FloatComparisonMode.OrderedLessThanNonSignaling);
        return Avx.AndNot(ltZero, v);
    }

#if NET8_0_OR_GREATER
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> ReluNaNSafeAvx512(Vector512<float> v, Vector512<float> vZero)
    {
        var ltZero = Avx512F.CompareLessThan(v, vZero);
        return Avx512F.AndNot(ltZero.AsUInt32(), v.AsUInt32()).AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<double> ReluNaNSafeAvx512(Vector512<double> v, Vector512<double> vZero)
    {
        var ltZero = Avx512F.CompareLessThan(v, vZero);
        return Avx512F.AndNot(ltZero.AsUInt64(), v.AsUInt64()).AsDouble();
    }
#endif

    /// <summary>
    /// SIMD per-(n, c) plane kernel: scalar bias[c] broadcast to vector,
    /// added to each HW position, optional ReLU, in-place store. 32-float
    /// unroll for load-store throughput. Used by ApplyBiasActivationNCHWInPlace.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void ApplyNchwPlaneSimd(
        float[] output, float[]? bias, int planeIdx, int C, int hw, bool applyRelu)
    {
        int c = planeIdx % C;
        float b = bias is not null ? bias[c] : 0f;
        int offset = planeIdx * hw;
        fixed (float* pOut = &output[offset])
        {
            int j = 0;
            var vZero = Vector256<float>.Zero;
            var vBias = Vector256.Create(b);

#if NET8_0_OR_GREATER
            if (CpuFeatures.HasAVX512F)
            {
                var vZero512 = Vector512<float>.Zero;
                var vBias512 = Vector512.Create(b);
                int simdLen512 = hw & ~31;
                for (; j < simdLen512; j += 32)
                {
                    var v0 = Avx512F.Add(Avx512F.LoadVector512(pOut + j), vBias512);
                    var v1 = Avx512F.Add(Avx512F.LoadVector512(pOut + j + 16), vBias512);
                    if (applyRelu)
                    {
                        v0 = ReluNaNSafeAvx512(v0, vZero512);
                        v1 = ReluNaNSafeAvx512(v1, vZero512);
                    }
                    Avx512F.Store(pOut + j, v0);
                    Avx512F.Store(pOut + j + 16, v1);
                }
                for (; j + 16 <= hw; j += 16)
                {
                    var v = Avx512F.Add(Avx512F.LoadVector512(pOut + j), vBias512);
                    if (applyRelu) v = ReluNaNSafeAvx512(v, vZero512);
                    Avx512F.Store(pOut + j, v);
                }
                for (; j < hw; j++)
                {
                    float val = pOut[j] + b;
                    pOut[j] = applyRelu && val < 0f ? 0f : val;
                }
                return;
            }
#endif

            int simdLen = hw & ~31;
            for (; j < simdLen; j += 32)
            {
                var v0 = Avx.Add(Avx.LoadVector256(pOut + j),      vBias);
                var v1 = Avx.Add(Avx.LoadVector256(pOut + j + 8),  vBias);
                var v2 = Avx.Add(Avx.LoadVector256(pOut + j + 16), vBias);
                var v3 = Avx.Add(Avx.LoadVector256(pOut + j + 24), vBias);
                if (applyRelu)
                {
                    v0 = ReluNaNSafeAvx(v0, vZero);
                    v1 = ReluNaNSafeAvx(v1, vZero);
                    v2 = ReluNaNSafeAvx(v2, vZero);
                    v3 = ReluNaNSafeAvx(v3, vZero);
                }
                Avx.Store(pOut + j,      v0);
                Avx.Store(pOut + j + 8,  v1);
                Avx.Store(pOut + j + 16, v2);
                Avx.Store(pOut + j + 24, v3);
            }
            for (; j + 8 <= hw; j += 8)
            {
                var v = Avx.Add(Avx.LoadVector256(pOut + j), vBias);
                if (applyRelu) v = ReluNaNSafeAvx(v, vZero);
                Avx.Store(pOut + j, v);
            }
            for (; j < hw; j++)
            {
                float val = pOut[j] + b;
                pOut[j] = applyRelu && val < 0f ? 0f : val;
            }
        }
    }
#endif

    /// <summary>
    /// Double-precision counterpart of <see cref="ApplyBiasActivationNCHWInPlace(float[], float[], int, int, int, int, FusedActivationType)"/>.
    /// Same tiling + parallelisation strategy, but SIMD registers hold 4
    /// doubles (AVX2) or 8 doubles (AVX-512) so the per-lane unroll is
    /// halved. Closes the issue #251 hot path: ResNet50 double-precision
    /// training decomposed Conv+Bias into three engine calls + two
    /// intermediate tensor allocations per conv layer; with this kernel
    /// the bias broadcast folds into a single in-place SIMD pass over
    /// the Conv2D output.
    /// </summary>
    /// <remarks>
    /// The NCHW plane layout makes bias broadcast trivial: for each
    /// (n, c) plane the bias value <c>bias[c]</c> is a scalar that we
    /// broadcast into a SIMD register once, then add against successive
    /// vector loads from the <c>H·W</c> output positions. The
    /// per-plane SIMD helper mirrors <see cref="ApplyNchwPlaneSimd"/>
    /// but with double register widths and no GELU branch (double GELU
    /// isn't currently a bottleneck for the workloads that hit this
    /// path — add it when a concrete request surfaces).
    /// </remarks>
    [MethodImpl(Hot)]
    internal static void ApplyBiasActivationNCHWInPlace(
        double[] output, double[]? bias,
        int N, int C, int H, int W, FusedActivationType activation)
    {
        bool hasBias = bias != null;
        bool hasActivation = activation != FusedActivationType.None;
        if (!hasBias && !hasActivation) return;

        int hw = H * W;
        int totalPlanes = N * C;

#if NET5_0_OR_GREATER
        if (Avx.IsSupported && hw >= 4 &&
            (activation == FusedActivationType.None
             || activation == FusedActivationType.ReLU))
        {
            // Same parallel-dispatch threshold rationale as the float
            // overload above. Doubles process half the lanes per AVX
            // register but the dispatch overhead is identical, so the
            // crossover point in element count is roughly the same.
            // Without this gate, ResNet-style [1, 64, 14, 14] inputs
            // (12544 elements) trigger a Parallel.For across 64 tiny
            // tasks and the fused path slows down 5× on CI runners
            // with limited vCPUs vs the unfused (Conv + BroadcastAdd)
            // reference — the exact regression issue #251 introduced
            // this fast path to AVOID.
            const long ParallelElementThreshold = 65536L;
            long totalElements = (long)totalPlanes * hw;
            if (totalPlanes >= 4 && totalElements >= ParallelElementThreshold)
            {
                Parallel.For(0, totalPlanes, p =>
                    ApplyNchwPlaneSimdDouble(output, bias, p, C, hw, activation == FusedActivationType.ReLU));
            }
            else
            {
                for (int p = 0; p < totalPlanes; p++)
                    ApplyNchwPlaneSimdDouble(output, bias, p, C, hw, activation == FusedActivationType.ReLU);
            }
            return;
        }
#endif

        // Scalar fallback — every runtime that doesn't hit the SIMD
        // branch (pre-AVX x86, ARM without vectorisation, net471).
        // Resolve the activation delegate once outside the loop so
        // every supported FusedActivationType value works here, not
        // only ReLU. Mirrors the float overload's GetFloatActivation
        // path — same surface, same semantics.
        Func<double, double>? activationFn = hasActivation ? GetDoubleActivation(activation) : null;
        for (int n = 0; n < N; n++)
        {
            for (int c = 0; c < C; c++)
            {
                double b = hasBias ? bias![c] : 0.0;
                int offset = (n * C + c) * hw;
                for (int i = 0; i < hw; i++)
                {
                    double val = output[offset + i];
                    if (hasBias) val += b;
                    if (activationFn != null) val = activationFn(val);
                    output[offset + i] = val;
                }
            }
        }
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Per-(n, c) plane kernel for doubles — broadcast bias[c] once,
    /// add to each HW position via AVX/AVX-512, optional ReLU, in-place
    /// store. Mirrors <see cref="ApplyNchwPlaneSimd"/> at half the lane
    /// count (4 doubles per AVX-256 register, 8 per AVX-512).
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void ApplyNchwPlaneSimdDouble(
        double[] output, double[]? bias, int planeIdx, int C, int hw, bool applyRelu)
    {
        int c = planeIdx % C;
        double b = bias is not null ? bias[c] : 0.0;
        int offset = planeIdx * hw;
        fixed (double* pOut = &output[offset])
        {
            int j = 0;
            var vZero = Vector256<double>.Zero;
            var vBias = Vector256.Create(b);

#if NET8_0_OR_GREATER
            if (CpuFeatures.HasAVX512F)
            {
                var vZero512 = Vector512<double>.Zero;
                var vBias512 = Vector512.Create(b);
                int simdLen512 = hw & ~15;
                for (; j < simdLen512; j += 16)
                {
                    var v0 = Avx512F.Add(Avx512F.LoadVector512(pOut + j), vBias512);
                    var v1 = Avx512F.Add(Avx512F.LoadVector512(pOut + j + 8), vBias512);
                    if (applyRelu)
                    {
                        v0 = ReluNaNSafeAvx512(v0, vZero512);
                        v1 = ReluNaNSafeAvx512(v1, vZero512);
                    }
                    Avx512F.Store(pOut + j, v0);
                    Avx512F.Store(pOut + j + 8, v1);
                }
                for (; j + 8 <= hw; j += 8)
                {
                    var v = Avx512F.Add(Avx512F.LoadVector512(pOut + j), vBias512);
                    if (applyRelu) v = ReluNaNSafeAvx512(v, vZero512);
                    Avx512F.Store(pOut + j, v);
                }
                for (; j < hw; j++)
                {
                    double val = pOut[j] + b;
                    pOut[j] = applyRelu && val < 0.0 ? 0.0 : val;
                }
                return;
            }
#endif

            // AVX path — 4 doubles per register, 4× unroll = 16 per block.
            int simdLen = hw & ~15;
            for (; j < simdLen; j += 16)
            {
                var v0 = Avx.Add(Avx.LoadVector256(pOut + j),      vBias);
                var v1 = Avx.Add(Avx.LoadVector256(pOut + j + 4),  vBias);
                var v2 = Avx.Add(Avx.LoadVector256(pOut + j + 8),  vBias);
                var v3 = Avx.Add(Avx.LoadVector256(pOut + j + 12), vBias);
                if (applyRelu)
                {
                    v0 = ReluNaNSafeAvx(v0, vZero);
                    v1 = ReluNaNSafeAvx(v1, vZero);
                    v2 = ReluNaNSafeAvx(v2, vZero);
                    v3 = ReluNaNSafeAvx(v3, vZero);
                }
                Avx.Store(pOut + j,      v0);
                Avx.Store(pOut + j + 4,  v1);
                Avx.Store(pOut + j + 8,  v2);
                Avx.Store(pOut + j + 12, v3);
            }
            for (; j + 4 <= hw; j += 4)
            {
                var v = Avx.Add(Avx.LoadVector256(pOut + j), vBias);
                if (applyRelu) v = ReluNaNSafeAvx(v, vZero);
                Avx.Store(pOut + j, v);
            }
            for (; j < hw; j++)
            {
                double val = pOut[j] + b;
                pOut[j] = applyRelu && val < 0.0 ? 0.0 : val;
            }
        }
    }
#endif

    [MethodImpl(Hot)]
    internal static void ApplyBiasActivationInPlace(float[] output, float[]? bias, int M, int N, FusedActivationType activation)
    {
        bool hasBias = bias != null;
        bool hasActivation = activation != FusedActivationType.None;
        if (!hasBias && !hasActivation) return;

#if NET5_0_OR_GREATER
        // Path B: SIMD-vectorised bias + activation epilogue. The prior scalar
        // loop with per-element delegate dispatch took ~8 ms on BERT FFN
        // up output [256, 3072] (786 KB, memory-bound budget is ~200 µs).
        //
        // None / ReLU: fused single-pass kernel writes bias + activation in
        //   one SIMD loop per row.
        // GELU: bias add via the SIMD path, then SimdKernels.GELUUnsafe in
        //   a second pass — already vectorised + 4×-unrolled. Two passes
        //   double the memory traffic (~300 µs vs 150 µs for a true fused
        //   kernel) but still 25× faster than the scalar baseline.
        if (Avx.IsSupported && (M * N) >= 4096 &&
            (activation == FusedActivationType.None
             || activation == FusedActivationType.ReLU
             || activation == FusedActivationType.GELU))
        {
            bool simdRelu = activation == FusedActivationType.ReLU;
            bool simdGelu = activation == FusedActivationType.GELU;
            // 64K elems → parallelise. Below this the Parallel.For dispatch
            // cost (~5–10 µs + ~3 KB allocation per call for closure +
            // partitioner state) outweighs the SIMD pass it's amortising.
            // For a 32K-element bias+activation pass (256×128 — the
            // ChainAsync BS=128 case in issue #296) the SIMD-only loop
            // measures ~5 µs vs ~10 µs with Parallel.For dispatch. The
            // 64K threshold matches the (unrelated but related)
            // ApplyBiasActivationNCHWInPlace floor in this same file.
            // Closes the per-call allocation gap to PyTorch flagged by
            // issue #296 acceptance criterion #5 at BS=128 (3.7 KB →
            // ≈400 B once Parallel.For is bypassed).
            int parallelThreshold = 64 * 1024;

            if (simdGelu && hasBias)
            {
                // Truly fused bias+GELU path — one SIMD pass over output
                // adds bias and applies GELU. Saves the ~786 KB memory
                // round-trip the two-pass form needed (load+bias+store,
                // then load+gelu+store). For BERT FFN up this cuts
                // ~200 µs per call; 12 calls per BERT = ~2.4 ms.
                if (M * N >= parallelThreshold && M >= 4)
                    Parallel.For(0, M, i => ApplyBiasGeluRowSimd(output, bias!, i, N));
                else
                    for (int i = 0; i < M; i++)
                        ApplyBiasGeluRowSimd(output, bias!, i, N);
                return;
            }

            if (hasBias)
            {
                if (M * N >= parallelThreshold && M >= 4)
                {
                    Parallel.For(0, M, i => ApplyBiasReluRowSimd(output, bias, i, N, simdRelu));
                }
                else
                {
                    for (int i = 0; i < M; i++)
                        ApplyBiasReluRowSimd(output, bias, i, N, simdRelu);
                }
            }
            else if (simdRelu)
            {
                // Pure ReLU, no bias
                if (M * N >= parallelThreshold && M >= 4)
                    Parallel.For(0, M, i => ApplyBiasReluRowSimd(output, null, i, N, true));
                else
                    for (int i = 0; i < M; i++)
                        ApplyBiasReluRowSimd(output, null, i, N, true);
            }

            // GELU without bias: still needs the separate GELU pass.
            if (simdGelu && !hasBias)
            {
                unsafe
                {
                    fixed (float* pOut = output)
                        SimdKernels.GELUUnsafe(pOut, pOut, M * N);
                }
            }
            return;
        }
#endif

        // Hoist the delegate lookup outside the hot loop to avoid per-element dictionary access
        Func<float, float>? activationFn = hasActivation ? GetFloatActivation(activation) : null;

        for (int i = 0; i < M; i++)
        {
            int rowOffset = i * N;
            for (int j = 0; j < N; j++)
            {
                float val = output[rowOffset + j];
                if (hasBias) val += bias![j];
                if (activationFn != null) val = activationFn(val);
                output[rowOffset + j] = val;
            }
        }
    }

#if NET5_0_OR_GREATER
    /// <summary>
    /// Fused bias + GELU(tanh approximation) row kernel — single SIMD pass
    /// over output. GELU(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715·x³))).
    /// Inputs: output row (post-MatMul), bias row, N columns. Output is
    /// written back in-place with bias added and GELU applied.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void ApplyBiasGeluRowSimd(float[] output, float[] bias, int row, int N)
    {
        int rowOff = row * N;
        fixed (float* pOut = &output[rowOff])
        fixed (float* pBias = bias)
        {
            int j = 0;

#if NET8_0_OR_GREATER
            // AVX-512 path: 16 floats per lane with Vector512. Uses the
            // AVX-512 Padé sigmoid variant for the tanh-via-sigmoid
            // reformulation. ~2× throughput vs the Vector256 path on
            // Zen 5 / Intel Ice Lake+ / Sapphire Rapids.
            if (CpuFeatures.HasAVX512F && N >= 16)
            {
                var vSqrt2OverPi512 = Vector512.Create(0.7978845608028654f);
                var vCoeff512       = Vector512.Create(0.044715f);
                var vHalf512        = Vector512.Create(0.5f);
                var vOne512         = Vector512.Create(1.0f);
                var vTwo512         = Vector512.Create(2.0f);
                int simdLen512 = N & ~15;
                for (; j < simdLen512; j += 16)
                {
                    var biased = Avx512F.Add(Avx512F.LoadVector512(pOut + j), Avx512F.LoadVector512(pBias + j));
                    var biasedSq = Avx512F.Multiply(biased, biased);
                    var biasedCu = Avx512F.Multiply(biasedSq, biased);
                    var inner    = Avx512F.FusedMultiplyAdd(vCoeff512, biasedCu, biased);
                    var tanhArg  = Avx512F.Multiply(vSqrt2OverPi512, inner);
                    var sigArg   = Avx512F.Multiply(vTwo512, tanhArg);
                    var sig      = PadeSigmoid.Sigmoid16(sigArg);
                    var tanh     = Avx512F.Subtract(Avx512F.Multiply(vTwo512, sig), vOne512);
                    var onePlusTanh = Avx512F.Add(vOne512, tanh);
                    var result   = Avx512F.Multiply(vHalf512, Avx512F.Multiply(biased, onePlusTanh));
                    Avx512F.Store(pOut + j, result);
                }
                for (; j < N; j++)
                {
                    float b = pOut[j] + pBias[j];
                    float inner = 0.7978845608028654f * (b + 0.044715f * b * b * b);
                    float tanh = MathF.Tanh(inner);
                    pOut[j] = 0.5f * b * (1.0f + tanh);
                }
                return;
            }
#endif

            // GELU constants (AVX2 8-wide)
            var vSqrt2OverPi = Vector256.Create(0.7978845608028654f);
            var vCoeff       = Vector256.Create(0.044715f);
            var vHalf        = Vector256.Create(0.5f);
            var vOne         = Vector256.Create(1.0f);
            var vTwo         = Vector256.Create(2.0f);

            int simdLen = N & ~7;
            for (; j < simdLen; j += 8)
            {
                // Fused: biased = output[j..j+8] + bias[j..j+8]
                var biased = Avx.Add(Avx.LoadVector256(pOut + j), Avx.LoadVector256(pBias + j));

                // GELU(biased) = 0.5 * biased * (1 + tanh(sqrt(2/pi) * (biased + 0.044715 * biased^3)))
                //             = 0.5 * biased * (1 + (2*sigmoid(2z) - 1))
                //             = biased * sigmoid(2z), where z = sqrt(2/pi) * (biased + 0.044715*biased³)
                var biasedSq = Avx.Multiply(biased, biased);
                var biasedCu = Avx.Multiply(biasedSq, biased);
                var inner    = Fma.MultiplyAdd(vCoeff, biasedCu, biased);
                var tanhArg  = Avx.Multiply(vSqrt2OverPi, inner);
                // tanh via 2*sigmoid(2x) - 1
                var sigArg   = Avx.Multiply(vTwo, tanhArg);
                var sig      = SimdKernels.FastSigmoid256(sigArg);
                var tanh     = Avx.Subtract(Avx.Multiply(vTwo, sig), vOne);
                var onePlusTanh = Avx.Add(vOne, tanh);
                var result   = Avx.Multiply(vHalf, Avx.Multiply(biased, onePlusTanh));
                Avx.Store(pOut + j, result);
            }
            for (; j < N; j++)
            {
                float b = pOut[j] + pBias[j];
                float inner = 0.7978845608028654f * (b + 0.044715f * b * b * b);
                float tanh = MathF.Tanh(inner);
                pOut[j] = 0.5f * b * (1.0f + tanh);
            }
        }
    }

    /// <summary>
    /// SIMD row kernel: output[row, :] = output[row, :] + bias; optionally ReLU.
    /// 32-float unroll (4× Vector256) for load-store throughput. Memory-bound
    /// on most CPUs — the compute is trivial relative to the read+write bandwidth.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void ApplyBiasReluRowSimd(float[] output, float[]? bias, int row, int N, bool applyRelu)
    {
        int rowOff = row * N;
        fixed (float* pOut = &output[rowOff])
        fixed (float* pBias = bias)  // null-safe: becomes null ptr when bias is null
        {
            int j = 0;

#if NET8_0_OR_GREATER
            // AVX-512 path: 16-wide Vector512, still 32-float unroll per
            // iter (2× Vector512) so register pressure stays balanced vs
            // the AVX2 4×Vector256 form. ~2× throughput on Zen 5 / Ice Lake+.
            if (CpuFeatures.HasAVX512F && bias is not null)
            {
                var vZero512 = Vector512<float>.Zero;
                int simdLen512 = N & ~31;
                for (; j < simdLen512; j += 32)
                {
                    var v0 = Avx512F.Add(Avx512F.LoadVector512(pOut + j),      Avx512F.LoadVector512(pBias + j));
                    var v1 = Avx512F.Add(Avx512F.LoadVector512(pOut + j + 16), Avx512F.LoadVector512(pBias + j + 16));
                    if (applyRelu)
                    {
                        v0 = ReluNaNSafeAvx512(v0, vZero512);
                        v1 = ReluNaNSafeAvx512(v1, vZero512);
                    }
                    Avx512F.Store(pOut + j,      v0);
                    Avx512F.Store(pOut + j + 16, v1);
                }
                for (; j + 16 <= N; j += 16)
                {
                    var v = Avx512F.Add(Avx512F.LoadVector512(pOut + j), Avx512F.LoadVector512(pBias + j));
                    if (applyRelu) v = ReluNaNSafeAvx512(v, vZero512);
                    Avx512F.Store(pOut + j, v);
                }
                for (; j < N; j++)
                {
                    float val = pOut[j] + pBias[j];
                    pOut[j] = applyRelu && val < 0f ? 0f : val;
                }
                return;
            }
#endif

            var vZero = Vector256<float>.Zero;

            if (bias is not null)
            {
                int simdLen = N & ~31;
                for (; j < simdLen; j += 32)
                {
                    var v0 = Avx.Add(Avx.LoadVector256(pOut + j),      Avx.LoadVector256(pBias + j));
                    var v1 = Avx.Add(Avx.LoadVector256(pOut + j + 8),  Avx.LoadVector256(pBias + j + 8));
                    var v2 = Avx.Add(Avx.LoadVector256(pOut + j + 16), Avx.LoadVector256(pBias + j + 16));
                    var v3 = Avx.Add(Avx.LoadVector256(pOut + j + 24), Avx.LoadVector256(pBias + j + 24));
                    if (applyRelu)
                    {
                        v0 = ReluNaNSafeAvx(v0, vZero);
                        v1 = ReluNaNSafeAvx(v1, vZero);
                        v2 = ReluNaNSafeAvx(v2, vZero);
                        v3 = ReluNaNSafeAvx(v3, vZero);
                    }
                    Avx.Store(pOut + j,      v0);
                    Avx.Store(pOut + j + 8,  v1);
                    Avx.Store(pOut + j + 16, v2);
                    Avx.Store(pOut + j + 24, v3);
                }
                for (; j + 8 <= N; j += 8)
                {
                    var v = Avx.Add(Avx.LoadVector256(pOut + j), Avx.LoadVector256(pBias + j));
                    if (applyRelu) v = ReluNaNSafeAvx(v, vZero);
                    Avx.Store(pOut + j, v);
                }
                for (; j < N; j++)
                {
                    float val = pOut[j] + pBias[j];
                    pOut[j] = applyRelu && val < 0f ? 0f : val;
                }
            }
            else  // Relu only
            {
                int simdLen = N & ~31;
                for (; j < simdLen; j += 32)
                {
                    Avx.Store(pOut + j,      ReluNaNSafeAvx(Avx.LoadVector256(pOut + j),      vZero));
                    Avx.Store(pOut + j + 8,  ReluNaNSafeAvx(Avx.LoadVector256(pOut + j + 8),  vZero));
                    Avx.Store(pOut + j + 16, ReluNaNSafeAvx(Avx.LoadVector256(pOut + j + 16), vZero));
                    Avx.Store(pOut + j + 24, ReluNaNSafeAvx(Avx.LoadVector256(pOut + j + 24), vZero));
                }
                for (; j + 8 <= N; j += 8)
                    Avx.Store(pOut + j, ReluNaNSafeAvx(Avx.LoadVector256(pOut + j), vZero));
                for (; j < N; j++)
                    if (pOut[j] < 0f) pOut[j] = 0f;
            }
        }
    }
#endif

    #endregion

    #region Fused LayerNorm + Activation

    /// <summary>
    /// Performs fused LayerNorm + Activation in a single pass.
    /// Computes mean and variance in first pass, then normalizes and applies activation.
    /// </summary>
    /// <param name="input">Input tensor data (flattened).</param>
    /// <param name="gamma">Scale parameter (per feature).</param>
    /// <param name="beta">Shift parameter (per feature).</param>
    /// <param name="output">Output tensor data (flattened).</param>
    /// <param name="batchSize">Number of samples in batch.</param>
    /// <param name="featureSize">Number of features per sample.</param>
    /// <param name="epsilon">Small constant for numerical stability.</param>
    /// <param name="activation">Activation function to apply after normalization.</param>
    public static void FusedLayerNormActivation(
        float[] input,
        float[] gamma,
        float[] beta,
        float[] output,
        int batchSize,
        int featureSize,
        float epsilon,
        FusedActivationType activation)
    {
        if (input.Length < batchSize * featureSize)
            throw new ArgumentException("Input size mismatch", nameof(input));
        if (gamma.Length < featureSize)
            throw new ArgumentException("Gamma size mismatch", nameof(gamma));
        if (beta.Length < featureSize)
            throw new ArgumentException("Beta size mismatch", nameof(beta));
        if (output.Length < batchSize * featureSize)
            throw new ArgumentException("Output size mismatch", nameof(output));

        if (batchSize >= 4)
        {
            Parallel.For(0, batchSize, b =>
            {
                FusedLayerNormSingleSample(input, gamma, beta, output, b, featureSize, epsilon, activation);
            });
        }
        else
        {
            for (int b = 0; b < batchSize; b++)
            {
                FusedLayerNormSingleSample(input, gamma, beta, output, b, featureSize, epsilon, activation);
            }
        }
    }

    /// <summary>
    /// Performs LayerNorm + Activation for a single sample in one pass.
    /// Uses Welford's online algorithm for numerically stable mean/variance.
    /// </summary>
    private static void FusedLayerNormSingleSample(
        float[] input,
        float[] gamma,
        float[] beta,
        float[] output,
        int batchIndex,
        int featureSize,
        float epsilon,
        FusedActivationType activation)
    {
        int offset = batchIndex * featureSize;

        // Pass 1: Compute mean
        float sum = 0f;
        for (int i = 0; i < featureSize; i++)
        {
            sum += input[offset + i];
        }
        float mean = sum / featureSize;

        // Pass 1.5: Compute variance
        float variance = 0f;
        for (int i = 0; i < featureSize; i++)
        {
            float diff = input[offset + i] - mean;
            variance += diff * diff;
        }
        variance /= featureSize;

        // Compute inverse standard deviation (1 / sqrt(variance + epsilon))
        float invStd = 1f / MathF.Sqrt(variance + epsilon);

        // Pass 2: Normalize, scale, shift, and apply activation (fused)
        // Hoist delegate lookup outside the loop
        var activationFn = activation != FusedActivationType.None ? GetFloatActivation(activation) : null;
        for (int i = 0; i < featureSize; i++)
        {
            float normalized = (input[offset + i] - mean) * invStd;
            float scaled = normalized * gamma[i] + beta[i];
            output[offset + i] = activationFn != null ? activationFn(scaled) : scaled;
        }
    }

    #endregion

    #region Fused Residual + LayerNorm (Transformer blocks)

    /// <summary>
    /// Performs fused Residual + LayerNorm: output = LayerNorm(input + residual).
    /// Critical optimization for Transformer blocks.
    /// </summary>
    /// <param name="input">Input tensor data.</param>
    /// <param name="residual">Residual connection data.</param>
    /// <param name="gamma">LayerNorm scale parameter.</param>
    /// <param name="beta">LayerNorm shift parameter.</param>
    /// <param name="output">Output tensor data.</param>
    /// <param name="batchSize">Number of samples.</param>
    /// <param name="featureSize">Features per sample.</param>
    /// <param name="epsilon">Numerical stability constant.</param>
    public static void FusedResidualLayerNorm(
        float[] input,
        float[] residual,
        float[] gamma,
        float[] beta,
        float[] output,
        int batchSize,
        int featureSize,
        float epsilon)
    {
        if (batchSize >= 4)
        {
            Parallel.For(0, batchSize, b =>
            {
                FusedResidualLayerNormSingle(input, residual, gamma, beta, output, b, featureSize, epsilon);
            });
        }
        else
        {
            for (int b = 0; b < batchSize; b++)
            {
                FusedResidualLayerNormSingle(input, residual, gamma, beta, output, b, featureSize, epsilon);
            }
        }
    }

    private static void FusedResidualLayerNormSingle(
        float[] input,
        float[] residual,
        float[] gamma,
        float[] beta,
        float[] output,
        int batchIndex,
        int featureSize,
        float epsilon)
    {
        int offset = batchIndex * featureSize;

        // Compute residual sum and mean in single pass
        float sum = 0f;
        for (int i = 0; i < featureSize; i++)
        {
            float val = input[offset + i] + residual[offset + i];
            output[offset + i] = val; // Store intermediate for variance calculation
            sum += val;
        }
        float mean = sum / featureSize;

        // Compute variance
        float variance = 0f;
        for (int i = 0; i < featureSize; i++)
        {
            float diff = output[offset + i] - mean;
            variance += diff * diff;
        }
        variance /= featureSize;

        float invStd = 1f / MathF.Sqrt(variance + epsilon);

        // Final pass: normalize, scale, shift (fused)
        for (int i = 0; i < featureSize; i++)
        {
            float normalized = (output[offset + i] - mean) * invStd;
            output[offset + i] = normalized * gamma[i] + beta[i];
        }
    }

    #endregion

    #region Fused Softmax (Attention)

    /// <summary>
    /// Performs fused scaled softmax for attention: softmax(x / sqrt(d)).
    /// Single pass with numerical stability (max subtraction).
    /// </summary>
    /// <param name="input">Input logits (batch_size x seq_len).</param>
    /// <param name="output">Output probabilities.</param>
    /// <param name="batchSize">Number of rows.</param>
    /// <param name="seqLen">Sequence length (number of columns).</param>
    /// <param name="scale">Scale factor (typically 1/sqrt(d_k)).</param>
    public static void FusedScaledSoftmax(
        float[] input,
        float[] output,
        int batchSize,
        int seqLen,
        float scale)
    {
        if (batchSize >= 4)
        {
            Parallel.For(0, batchSize, b =>
            {
                FusedScaledSoftmaxRow(input, output, b, seqLen, scale);
            });
        }
        else
        {
            for (int b = 0; b < batchSize; b++)
            {
                FusedScaledSoftmaxRow(input, output, b, seqLen, scale);
            }
        }
    }

    private static void FusedScaledSoftmaxRow(
        float[] input,
        float[] output,
        int batchIndex,
        int seqLen,
        float scale)
    {
        int offset = batchIndex * seqLen;

        // Find max for numerical stability
        float maxVal = float.NegativeInfinity;
        for (int i = 0; i < seqLen; i++)
        {
            float val = input[offset + i] * scale;
            if (val > maxVal) maxVal = val;
        }

        // Compute exp(x * scale - max) and sum
        float sum = 0f;
        for (int i = 0; i < seqLen; i++)
        {
            float val = MathF.Exp(input[offset + i] * scale - maxVal);
            output[offset + i] = val;
            sum += val;
        }

        // Normalize
        float invSum = 1f / sum;
        for (int i = 0; i < seqLen; i++)
        {
            output[offset + i] *= invSum;
        }
    }

    #endregion

    #region Fused Bias + Dropout (Training)

    /// <summary>
    /// Performs fused Bias + Dropout for training.
    /// Combines bias addition and dropout in single pass.
    /// </summary>
    /// <param name="input">Input tensor.</param>
    /// <param name="bias">Bias vector (per feature).</param>
    /// <param name="output">Output tensor.</param>
    /// <param name="dropoutMask">Pre-generated dropout mask (0 or 1).</param>
    /// <param name="batchSize">Batch size.</param>
    /// <param name="featureSize">Feature size.</param>
    /// <param name="dropoutScale">Scale factor (1 / (1 - dropout_rate)).</param>
    public static void FusedBiasDropout(
        float[] input,
        float[] bias,
        float[] output,
        float[] dropoutMask,
        int batchSize,
        int featureSize,
        float dropoutScale)
    {
        int totalElements = batchSize * featureSize;

        // #294 Phase 4 wiring: this kernel is a textbook
        // LoopOptimizer.Tile2D consumer — a 2D iteration over
        // (batch, feature) with a fused per-element body. Using the
        // helper makes the iteration shape uniform with the rest of
        // the codebase's tiled patterns; the inner per-tile loop is a
        // tight scalar fused-op stream that the JIT can vectorize.
        // Tile size of 64 covers the cache-line stride for floats
        // (16 floats per 64B cache line × 4 lines) and matches the
        // typical bias-row reuse pattern.
        const int FusedTileSize = 64;
        if (totalElements >= PARALLEL_THRESHOLD)
        {
            // Large tensors: tile in 2D and dispatch the tiles through
            // LoopOptimizer.ParallelTile2D, which routes onto the repo's
            // CpuParallelSettings.LightweightParallel persistent worker
            // pool. Replaces the previous raw Parallel.For so we honour
            // the configured MaxDegreeOfParallelism cap and share the
            // pool with SimdConvHelper / Im2ColHelper instead of
            // spinning up a fresh ThreadPool slice per call.
            LoopOptimizer.ParallelTile2D(batchSize, featureSize, FusedTileSize,
                new FusedBiasDropoutTile(input, bias, output, dropoutMask, featureSize, dropoutScale));
        }
        else
        {
            // Sequential path uses Tile2D with one row per tile so
            // the fused body sees a contiguous feature span.
            LoopOptimizer.Tile2D(batchSize, featureSize, FusedTileSize,
                new FusedBiasDropoutTile(input, bias, output, dropoutMask, featureSize, dropoutScale));
        }
    }

    /// <summary>
    /// Inner per-row body for the fused bias+dropout fused op.
    /// Hoisted into a helper so both the parallel and the tiled
    /// sequential paths can share the same scalar/vectorizable body
    /// without delegate allocation.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static void FusedBiasDropoutRow(
        float[] input, float[] bias, float[] output, float[] dropoutMask,
        int rowOffset, int featureSize, float dropoutScale)
    {
        for (int i = 0; i < featureSize; i++)
        {
            int idx = rowOffset + i;
            output[idx] = (input[idx] + bias[i]) * dropoutMask[idx] * dropoutScale;
        }
    }

    /// <summary>
    /// Struct callback for <see cref="LoopOptimizer.Tile2D{TAction}"/>
    /// that runs <see cref="FusedBiasDropoutRow"/> over each tile's
    /// row range. Struct (not delegate) so RyuJIT specializes the
    /// helper per concrete callee, devirtualizes Invoke, and inlines
    /// the body into the Tile2D loop — same shape as
    /// <see cref="System.Numerics.Vector{T}"/>'s callbacks.
    /// </summary>
    private readonly struct FusedBiasDropoutTile : LoopOptimizer.ITile2DAction
    {
        private readonly float[] _input, _bias, _output, _dropoutMask;
        private readonly int _featureSize;
        private readonly float _dropoutScale;
        public FusedBiasDropoutTile(float[] input, float[] bias, float[] output, float[] dropoutMask,
            int featureSize, float dropoutScale)
        {
            _input = input; _bias = bias; _output = output; _dropoutMask = dropoutMask;
            _featureSize = featureSize; _dropoutScale = dropoutScale;
        }
        public void Invoke(int i0, int i1, int j0, int j1)
        {
            for (int b = i0; b < i1; b++)
            {
                int rowOffset = b * _featureSize;
                for (int j = j0; j < j1; j++)
                {
                    int idx = rowOffset + j;
                    _output[idx] = (_input[idx] + _bias[j]) * _dropoutMask[idx] * _dropoutScale;
                }
            }
        }
    }

    #endregion

    #region Activation Functions

    /// <summary>
    /// OCP-compliant activation dispatch table for pointwise float activations.
    /// Resolve once before a loop, then call the returned delegate per element with zero lookup overhead.
    /// </summary>
    private static readonly Dictionary<FusedActivationType, Func<float, float>> _floatActivations = new()
    {
        { FusedActivationType.None, x => x },
        // NaN-preserving: `x < 0 ? 0 : x` — when x is NaN, `x < 0` is false (ordered),
        // so NaN survives. The seemingly equivalent `x > 0 ? x : 0` would convert
        // NaN to 0, masking invalid activations during training.
        { FusedActivationType.ReLU, x => x < 0f ? 0f : x },
        { FusedActivationType.GELU, ApplyGelu },
        { FusedActivationType.Sigmoid, x => 1f / (1f + MathF.Exp(-x)) },
        { FusedActivationType.Tanh, MathF.Tanh },
        { FusedActivationType.LeakyReLU, x => x > 0f ? x : 0.01f * x },
        { FusedActivationType.Swish, x => x / (1f + MathF.Exp(-x)) },
        // Softmax is NOT pointwise (depends on entire row) — must not appear here.
        // Fused paths that include Softmax should apply it separately after the GEMM loop.
    };

    /// <summary>Gets the float activation function delegate for use in tight loops.
    /// Resolve once outside the loop, then call the returned delegate per element.</summary>
    internal static Func<float, float> GetFloatActivation(FusedActivationType activation)
    {
        if (_floatActivations.TryGetValue(activation, out var fn))
            return fn;
        throw new ArgumentException($"No float activation registered for type: {activation}");
    }

    /// <summary>
    /// GELU activation using fast tanh approximation.
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static float ApplyGelu(float x)
    {
        const float sqrt2OverPi = 0.7978845608028654f;
        const float coeff = 0.044715f;

        float xCubed = x * x * x;
        float inner = sqrt2OverPi * (x + coeff * xCubed);
        return 0.5f * x * (1f + MathF.Tanh(inner));
    }

    #endregion

    #region Double Precision Overloads

    /// <summary>
    /// Performs fused GEMM + Bias + Activation for double precision.
    /// </summary>
    public static void FusedGemmBiasActivation(
        double[] A,
        double[] B,
        double[]? bias,
        double[] output,
        int M, int N, int K,
        FusedActivationType activation)
    {
        if (A.Length < M * K)
            throw new ArgumentException($"A must have at least {M * K} elements", nameof(A));
        if (B.Length < K * N)
            throw new ArgumentException($"B must have at least {K * N} elements", nameof(B));
        if (output.Length < M * N)
            throw new ArgumentException($"output must have at least {M * N} elements", nameof(output));
        if (bias != null && bias.Length < N)
            throw new ArgumentException($"bias must have at least {N} elements", nameof(bias));

        // Use BLAS for the O(MNK) GEMM, then fuse bias+activation in a cheap O(MN) second pass.
        if (BlasProvider.TryGemm(M, N, K, A, 0, K, B, 0, N, output, 0, N))
        {
            ApplyBiasActivationInPlaceDouble(output, bias, M, N, activation);
            return;
        }

        // BLAS unavailable: route through MatrixMultiplyHelper.MultiplyBlocked,
        // whose inner kernel is SimdKernels.ScalarMultiplyAdd — AVX2/FMA
        // Vector256<double> on modern x64. ~50-100x faster than the previous
        // naive parallel scalar row loop. Same structural fix as TensorMatMul's
        // fallback (see CpuEngine.TensorMatMul2D). Without this, every DiT
        // block's MLP dense layers (the 80% of DiT wall-clock for double
        // diffusion models per Pika21 probe) runs at ~1-2 GFLOPS scalar
        // instead of ~40 GFLOPS SIMD.
        System.Array.Clear(output, 0, M * N);
        MatrixMultiplyHelper.MultiplyBlocked(
            MathHelper.GetNumericOperations<double>(),
            A.AsMemory(0, M * K), B.AsMemory(0, K * N), output.AsMemory(0, M * N),
            M, K, N, K, N, N);
        ApplyBiasActivationInPlaceDouble(output, bias, M, N, activation);
    }

    /// <summary>
    /// Applies bias addition and activation function in-place over the double GEMM output.
    /// </summary>
    [MethodImpl(Hot)]
    internal static void ApplyBiasActivationInPlaceDouble(double[] output, double[]? bias, int M, int N, FusedActivationType activation)
    {
        bool hasBias = bias != null;
        bool hasActivation = activation != FusedActivationType.None;
        if (!hasBias && !hasActivation) return;

        // Hoist the delegate lookup outside the hot loop
        Func<double, double>? activationFn = hasActivation ? GetDoubleActivation(activation) : null;

        for (int i = 0; i < M; i++)
        {
            int rowOffset = i * N;
            for (int j = 0; j < N; j++)
            {
                double val = output[rowOffset + j];
                if (hasBias) val += bias![j];
                if (activationFn != null) val = activationFn(val);
                output[rowOffset + j] = val;
            }
        }
    }

    /// <summary>OCP-compliant double activation dispatch table.</summary>
    private static readonly Dictionary<FusedActivationType, Func<double, double>> _doubleActivations = new()
    {
        { FusedActivationType.None, x => x },
        // NaN-preserving: see _floatActivations comment above.
        { FusedActivationType.ReLU, x => x < 0.0 ? 0.0 : x },
        { FusedActivationType.GELU, ApplyGeluDouble },
        { FusedActivationType.Sigmoid, x => 1.0 / (1.0 + Math.Exp(-x)) },
        { FusedActivationType.Tanh, Math.Tanh },
        { FusedActivationType.LeakyReLU, x => x > 0.0 ? x : 0.01 * x },
        { FusedActivationType.Swish, x => x / (1.0 + Math.Exp(-x)) },
        // Softmax is NOT pointwise — must not appear here.
    };

    /// <summary>Gets the double activation function delegate for use in tight loops.
    /// Resolve once outside the loop, then call the returned delegate per element.</summary>
    internal static Func<double, double> GetDoubleActivation(FusedActivationType activation)
    {
        if (_doubleActivations.TryGetValue(activation, out var fn))
            return fn;
        throw new ArgumentException($"No double activation registered for type: {activation}");
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static double ApplyGeluDouble(double x)
    {
        const double sqrt2OverPi = 0.7978845608028654;
        const double coeff = 0.044715;

        double xCubed = x * x * x;
        double inner = sqrt2OverPi * (x + coeff * xCubed);
        return 0.5 * x * (1.0 + Math.Tanh(inner));
    }

    #endregion
}
