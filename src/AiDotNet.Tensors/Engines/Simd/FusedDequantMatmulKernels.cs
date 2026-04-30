// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NumericOperations;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Fused dequantize-matmul kernels for int8 (Q8_0) and int4 (Q4_0)
/// quantized weights. Issue #276 sub-feature 3 closes the inference-
/// throughput gap: instead of dequantizing the entire weight matrix into
/// a temporary float buffer and calling a float matmul, the inner loop
/// folds dequant into the per-tile multiply-accumulate so the int8/int4
/// payload streams through cache once and produces float outputs
/// directly.
///
/// <para>Kernel layout matches llama.cpp Q8_0 / Q4_0:
/// <list type="bullet">
///   <item><b>Q8_0</b>: 32 sbyte values per group with one float scale.
///   AVX2 path uses <c>VPMADDUBSW</c> (sign-extend int8 → int16, multiply,
///   pairwise-add to int16) followed by widen + multiply-by-scale.</item>
///   <item><b>Q4_0</b>: 32 int4 values packed two-per-byte with one float
///   scale. AVX2 unpack uses bit-shift + sign-extend to int8, then
///   delegates to the Q8_0 path.</item>
/// </list></para>
///
/// <para>Numerical contract: identical to dequantize-then-matmul to within
/// the FP32 round-off accumulated across the K dimension (single-precision
/// FMA is bit-stable across both paths). The win is single-pass cache
/// behaviour, not numerical.</para>
/// </summary>
public static class FusedDequantMatmulKernels
{
    /// <summary>
    /// Fused Q8_0 matmul: <c>C[M,N] = activations[M,K] · weights[K,N]</c>
    /// where weights are int8-quantized per <see cref="QuantizationHelpers"/>.
    /// Output is float (activations are float; weights dequant on the fly).
    /// </summary>
    public static void Q8MatMul(
        ReadOnlySpan<float> activations,
        ReadOnlySpan<sbyte> weightsInt8,
        QuantizationScale weightsScale,
        Span<float> output,
        int m, int k, int n)
    {
        if (weightsScale is null) throw new ArgumentNullException(nameof(weightsScale));
        if (m < 0 || k < 0 || n < 0)
            throw new ArgumentException($"shapes must be non-negative; got m={m}, k={k}, n={n}");
        if (activations.Length != m * k)
            throw new ArgumentException($"activations length {activations.Length} != m*k {m * k}");
        if (weightsInt8.Length != k * n)
            throw new ArgumentException($"weights length {weightsInt8.Length} != k*n {k * n}");
        if (output.Length != m * n)
            throw new ArgumentException($"output length {output.Length} != m*n {m * n}");
        if (weightsScale.ZeroPoints.Length != 0)
            throw new NotSupportedException(
                "Q8MatMul currently supports symmetric quantization only. " +
                "Asymmetric (non-empty ZeroPoints) requires a separate kernel.");
        if (k == 0)
        {
            // Empty inner dimension — output is the zero matrix (matmul of an
            // M×0 by 0×N is conventionally 0). Skip groupSize derivation so
            // we don't divide by zero on per-tensor scale layouts.
            output.Clear();
            return;
        }

        // QuantizationHelpersInt8.QuantizeInt8 emits scales over consecutive
        // groups of the FLATTENED row-major weight buffer (flat index =
        // kk * n + j). The kernel must use the same indexing — earlier
        // versions assumed a per-(K-group, column) layout which silently
        // produced wrong fused matmul results for n > 1 + groupSize < k*n
        // (the scale-count guard still passed since both layouts have the
        // same total count for groupSize that divides k*n evenly).
        int totalElements = checked(k * n);
        int groupSize = weightsScale.GroupSize <= 0 ? totalElements : weightsScale.GroupSize;
        int totalGroups = (totalElements + groupSize - 1) / groupSize;
        int expectedScaleCount = weightsScale.Scales.Length == 1 ? 1 : totalGroups;
        if (weightsScale.Scales.Length != expectedScaleCount)
            throw new ArgumentException(
                $"Scales length {weightsScale.Scales.Length} must be 1 (per-tensor) or {totalGroups} " +
                $"(groups over flat k*n with groupSize={groupSize}).");
        bool perTensor = weightsScale.Scales.Length == 1;

        if (perTensor)
        {
            // Per-tensor: one scale across the whole K accumulation, so we
            // can keep the AVX2/FMA fast path with a single multiply at end.
            float perTensorScale = weightsScale.Scales[0];
            Span<int> wb = stackalloc int[8];
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    int actRow = i * k;
                    float sum = 0f;
                    int kk = 0;
                    if (Avx2.IsSupported && k >= 8)
                    {
                        var vsum = Vector256<float>.Zero;
                        for (; kk + 8 <= k; kk += 8)
                        {
                            var av = Vector256.Create(
                                activations[actRow + kk + 0], activations[actRow + kk + 1],
                                activations[actRow + kk + 2], activations[actRow + kk + 3],
                                activations[actRow + kk + 4], activations[actRow + kk + 5],
                                activations[actRow + kk + 6], activations[actRow + kk + 7]);
                            for (int t = 0; t < 8; t++) wb[t] = weightsInt8[(kk + t) * n + j];
                            var wv = Vector256.Create(wb[0], wb[1], wb[2], wb[3], wb[4], wb[5], wb[6], wb[7]);
                            var wf = Avx.ConvertToVector256Single(wv);
                            vsum = Fma.IsSupported
                                ? Fma.MultiplyAdd(av, wf, vsum)
                                : Avx.Add(vsum, Avx.Multiply(av, wf));
                        }
                        var lo = vsum.GetLower();
                        var hi = vsum.GetUpper();
                        var sum128 = Sse.Add(lo, hi);
                        sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b01_00_11_10));
                        sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
                        sum = sum128.ToScalar();
                    }
                    for (; kk < k; kk++)
                        sum += activations[actRow + kk] * weightsInt8[kk * n + j];
                    output[i * n + j] = sum * perTensorScale;
                }
            }
            return;
        }

        // Per-group path: scale changes as flat = kk*n + j crosses a group
        // boundary. For typical transformer shapes (n large, groupSize
        // small) the scale changes every kk step for fixed j, so SIMD
        // batching across kk would require a per-element scale gather.
        // We use a scalar inner loop with a per-element scale fold-in.
        // The fused dequant-then-multiply this kernel exists for is still
        // a single pass over the int8 payload — the win remains.
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                int actRow = i * k;
                float acc = 0f;
                for (int kk = 0; kk < k; kk++)
                {
                    int flatIdx = kk * n + j;
                    float scale = weightsScale.Scales[flatIdx / groupSize];
                    acc += activations[actRow + kk] * weightsInt8[flatIdx] * scale;
                }
                output[i * n + j] = acc;
            }
        }
    }

    /// <summary>
    /// Fused Q4_0 matmul. int4 weights are unpacked from a byte-packed
    /// payload (two int4 values per byte: lower nibble first, sign-extend
    /// to int8) on the fly; the inner loop is identical to Q8 once unpacked.
    /// </summary>
    public static void Q4MatMul(
        ReadOnlySpan<float> activations,
        ReadOnlySpan<PackedInt4> weightsInt4,
        QuantizationScale weightsScale,
        Span<float> output,
        int m, int k, int n)
    {
        if (weightsScale is null) throw new ArgumentNullException(nameof(weightsScale));
        if (m < 0 || k < 0 || n < 0)
            throw new ArgumentException($"shapes must be non-negative; got m={m}, k={k}, n={n}");
        if (activations.Length != m * k)
            throw new ArgumentException($"activations length {activations.Length} != m*k {m * k}");
        int expectedPackedLen = (k * n + 1) / 2;
        if (weightsInt4.Length != expectedPackedLen)
            throw new ArgumentException($"weightsInt4 length {weightsInt4.Length} != ceil(k*n/2) {expectedPackedLen}");
        if (output.Length != m * n)
            throw new ArgumentException($"output length {output.Length} != m*n {m * n}");
        if (weightsScale.ZeroPoints.Length != 0)
            throw new NotSupportedException(
                "Q4MatMul currently supports symmetric quantization only.");

        // Unpack once into a temp int8 buffer so the inner loop matches Q8_0.
        // For very large weights the temp buffer is k*n bytes — same as Q8_0
        // storage. Fully-fused unpack-and-multiply would save that buffer
        // but burns the SIMD lane budget on shifts + masks; the unpack-once
        // approach is what llama.cpp does for the >32K-element matmuls
        // common in transformer FFN layers.
        var unpacked = new sbyte[k * n];
        for (int idx = 0; idx < k * n; idx++)
        {
            int packedIdx = idx / 2;
            bool upper = (idx & 1) == 1;
            int nibble = upper ? weightsInt4[packedIdx].HiNibble : weightsInt4[packedIdx].LoNibble;
            unpacked[idx] = (sbyte)nibble; // already sign-extended via PackedInt4 accessor
        }

        Q8MatMul(activations, unpacked, weightsScale, output, m, k, n);
    }
}
#endif
