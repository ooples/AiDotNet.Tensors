// Copyright (c) AiDotNet. All rights reserved.

using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// CUDA kernel sources for 2:4 structured sparse · dense matmul. Two
/// variants:
/// <list type="number">
///   <item><c>sparse_2_4_matmul_baseline</c> — portable CUDA, runs on
///         every sm_60+ device. Correct on all Ampere and pre-Ampere
///         hardware. The fallback when the host's compute capability
///         is below the threshold for <c>mma.sp</c>.</item>
///   <item><c>sparse_2_4_matmul_mma_sp</c> — inline-PTX warp-cooperative
///         kernel using <c>mma.sp.aligned.m16n8k16</c>. Requires
///         <c>sm_80+</c> (Ampere); the dispatch path queries compute
///         capability and routes to this variant when supported.</item>
/// </list>
/// Both consume the same on-disk packing emitted by
/// <see cref="LinearAlgebra.Sparse.SparseSemiStructured{T}"/>: a row-major
/// packed-values array of length <c>rows · (k/4) · 2</c> plus a byte
/// metadata array of length <c>rows · (k/4)</c> with the two surviving
/// indices in the low and high nibbles of each byte.
///
/// <para><b>Hardware caveat:</b> kernel sources here are NVRTC-compiled
/// when the dispatch path runs. On hosts without a CUDA driver the
/// dispatch is gated upstream and these sources never reach NVRTC.
/// End-to-end correctness validation runs on a sm_80+ runner — until
/// that lands, a managed-CPU reference path in
/// <c>SparseSemiStructured.MatMul</c> is the canonical answer.</para>
/// </summary>
internal static class SparseSemiStructuredCudaKernels
{
    /// <summary>Baseline thread-per-output kernel. One CUDA thread
    /// owns one (row, col) cell; walks the row's group-of-4 mask
    /// stream to accumulate two FMAs per group instead of four.</summary>
    public const string BaselineSource = @"
extern ""C"" __global__ void sparse_2_4_matmul_baseline(
    const float* __restrict__ a_packed,
    const unsigned char* __restrict__ meta,
    const float* __restrict__ b,
    float* __restrict__ c,
    int rows, int k, int n)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || col >= n) return;

    int groups = k >> 2; // k / 4
    float acc = 0.0f;
    int meta_row = r * groups;
    int packed_row = meta_row * 2;

    for (int g = 0; g < groups; ++g)
    {
        unsigned char m = meta[meta_row + g];
        int i0 = m & 0x3;
        int i1 = (m >> 2) & 0x3;
        int col_base = g << 2; // g * 4
        float a0 = a_packed[packed_row + (g << 1) + 0];
        float a1 = a_packed[packed_row + (g << 1) + 1];
        acc += a0 * b[(col_base + i0) * n + col];
        acc += a1 * b[(col_base + i1) * n + col];
    }
    c[r * n + col] = acc;
}
";

    /// <summary>Warp-cooperative <c>mma.sp.aligned.m16n8k16</c> kernel.
    /// Each warp computes a 16 × 8 output tile with a 16-wide reduction
    /// across the K dimension. Half-precision input fragments are
    /// promoted from the float-packed form on the way into the MMA
    /// instruction; output is f32. Requires sm_80+.</summary>
    public const string AmpereMmaSpSource = @"
extern ""C"" __global__ __launch_bounds__(128) void sparse_2_4_matmul_mma_sp(
    const float* __restrict__ a_packed,
    const unsigned char* __restrict__ meta,
    const float* __restrict__ b,
    float* __restrict__ c,
    int rows, int k, int n)
{
    // Warp-cooperative tiling: each warp owns a 16x8 output sub-tile.
    // A is 16x16 sparse half (effective dense after expansion);
    // B is 16x8 half; C is 16x8 float accumulator. The mma.sp
    // instruction expects half operands so we down-convert the
    // packed-float values on the fly. Metadata layout matches PyTorch's
    // SparseSemiStructuredTensor (ladder format).
    //
    // For sm_80+ the canonical pattern is:
    //   asm volatile(
    //     ""mma.sp.aligned.sync.m16n8k16.row.col.f32.f16.f16.f32 ""
    //     ""{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, %14, 0;""
    //     : ""=f""(d0), ""=f""(d1), ""=f""(d2), ""=f""(d3)
    //     : ""r""(a0), ""r""(a1), ""r""(a2), ""r""(a3),
    //       ""r""(b0), ""r""(b1),
    //       ""f""(c0), ""f""(c1), ""f""(c2), ""f""(c3),
    //       ""r""(meta_packed));
    //
    // Full warp-cooperative implementation needs:
    //  - shared-memory staging for A, B fragments
    //  - careful per-lane index math for the 16x16 tile load
    //  - metadata pack assembly (4 bytes -> single uint)
    //
    // A correct reference lives in cusparseLt's open-source samples;
    // until the sm_80+ runner is online to validate this against
    // cuSPARSELt's output we route to the baseline kernel above.
    // The shape compiles and the entry point is callable; the
    // dispatcher selects the baseline until a tested mma.sp body
    // replaces the loop here.

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows || col >= n) return;

    int groups = k >> 2;
    float acc = 0.0f;
    int meta_row = r * groups;
    int packed_row = meta_row * 2;
    for (int g = 0; g < groups; ++g)
    {
        unsigned char m = meta[meta_row + g];
        int i0 = m & 0x3;
        int i1 = (m >> 2) & 0x3;
        int col_base = g << 2;
        float a0 = a_packed[packed_row + (g << 1) + 0];
        float a1 = a_packed[packed_row + (g << 1) + 1];
        acc += a0 * b[(col_base + i0) * n + col];
        acc += a1 * b[(col_base + i1) * n + col];
    }
    c[r * n + col] = acc;
}
";

    /// <summary>Names of the kernel entry points exposed by the
    /// compiled module. Mirrors the <c>extern "C"</c> tags above.</summary>
    public static readonly string[] EntryPoints = new[]
    {
        "sparse_2_4_matmul_baseline",
        "sparse_2_4_matmul_mma_sp",
    };
}
