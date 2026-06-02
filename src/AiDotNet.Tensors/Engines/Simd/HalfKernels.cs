// Copyright (c) AiDotNet. All rights reserved.

#if NET5_0_OR_GREATER
using System;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// #378: SIMD-accelerated FP16 (<see cref="System.Half"/>) GEMM microkernel for the
/// BlasManaged dispatch — the FP16 counterpart of <see cref="BFloat16Kernels"/>.Matmul.
///
/// <para>
/// Like the issue's "AVX2 emulation via FP32 upcast" option, each FP16 input is widened
/// to <see cref="float"/> and the inner product is accumulated in a <b>float</b> register
/// (not FP16) so deep transformer K-dimensions (4096+) don't lose precision; the result is
/// rounded back to FP16 only at the very end. The 8-lane FMA accumulation is vectorized
/// (AVX2 + FMA); the widening is per-element because FP16→FP32 has no cheap bulk SIMD path
/// on AVX2 (F16C aside), so the expensive part — the K-reduction FMAs — is what gets the
/// vector treatment, matching the proven BF16 kernel's structure.
/// </para>
///
/// <para>Compiles only on net5.0+ (System.Half + Intrinsics). On net471 the BlasManaged
/// dispatch routes FP16 GEMM through the generic blocked scalar path instead.</para>
/// </summary>
public static class HalfKernels
{
    /// <summary>
    /// FP16 matmul with float accumulation: <c>C[m,n] = A[m,k] · B[k,n]</c>. <paramref name="c"/>
    /// is written as FP16 (round-to-nearest via the <c>(Half)float</c> cast); the inner
    /// accumulator is float. Non-transposed, row-major; row strides are caller-supplied so an
    /// ldc-padded / offset slice works.
    /// </summary>
    public static void Matmul(
        ReadOnlySpan<Half> a, int aRowStride,
        ReadOnlySpan<Half> b, int bRowStride,
        Span<Half> c, int cRowStride,
        int m, int k, int n)
    {
        if (m < 0 || k < 0 || n < 0)
            throw new ArgumentException($"Matmul shapes must be non-negative; got m={m}, k={k}, n={n}.");
        if (aRowStride < k)
            throw new ArgumentException($"aRowStride {aRowStride} must be >= k {k}.");
        if (bRowStride < n)
            throw new ArgumentException($"bRowStride {bRowStride} must be >= n {n}.");
        if (cRowStride < n)
            throw new ArgumentException($"cRowStride {cRowStride} must be >= n {n}.");
        if (m > 0 && a.Length < (m - 1) * aRowStride + k)
            throw new ArgumentException($"a span ({a.Length}) too short for m={m}, k={k}, stride={aRowStride}.");
        if (k > 0 && b.Length < (k - 1) * bRowStride + n)
            throw new ArgumentException($"b span ({b.Length}) too short for k={k}, n={n}, stride={bRowStride}.");
        if (m > 0 && c.Length < (m - 1) * cRowStride + n)
            throw new ArgumentException($"c span ({c.Length}) too short for m={m}, n={n}, stride={cRowStride}.");

        Span<float> tmpA = stackalloc float[8];
        Span<float> tmpB = stackalloc float[8];
        for (int i = 0; i < m; i++)
        {
            int aRow = i * aRowStride;
            for (int j = 0; j < n; j++)
            {
                float acc = 0f;
                int kk = 0;
                if (Avx2.IsSupported && k >= 8)
                {
                    var vsum = Vector256<float>.Zero;
                    for (; kk + 8 <= k; kk += 8)
                    {
                        for (int t = 0; t < 8; t++)
                        {
                            tmpA[t] = (float)a[aRow + kk + t];
                            tmpB[t] = (float)b[(kk + t) * bRowStride + j];
                        }
                        var av = Vector256.Create(tmpA[0], tmpA[1], tmpA[2], tmpA[3], tmpA[4], tmpA[5], tmpA[6], tmpA[7]);
                        var bv = Vector256.Create(tmpB[0], tmpB[1], tmpB[2], tmpB[3], tmpB[4], tmpB[5], tmpB[6], tmpB[7]);
                        vsum = Fma.IsSupported ? Fma.MultiplyAdd(av, bv, vsum) : Avx.Add(vsum, Avx.Multiply(av, bv));
                    }
                    var lo = vsum.GetLower();
                    var hi = vsum.GetUpper();
                    var sum128 = Sse.Add(lo, hi);
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b01_00_11_10));
                    sum128 = Sse.Add(sum128, Sse.Shuffle(sum128, sum128, 0b10_11_00_01));
                    acc = sum128.ToScalar();
                }
                for (; kk < k; kk++)
                    acc += (float)a[aRow + kk] * (float)b[kk * bRowStride + j];

                c[i * cRowStride + j] = (Half)acc;
            }
        }
    }
}
#endif
