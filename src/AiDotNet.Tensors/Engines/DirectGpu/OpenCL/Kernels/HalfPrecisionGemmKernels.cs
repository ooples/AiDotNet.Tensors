// Copyright (c) AiDotNet. All rights reserved.
// OpenCL half-precision (FP16) GEMM kernels for the IGpuHalfPrecisionBackend
// capability interface (issue #560: route MatrixMultiply<Half> to an FP16
// kernel instead of the scalar CPU fallback).

namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL.Kernels;

/// <summary>
/// FP16 GEMM kernels: <c>C = A · B</c> with FP16 (half) inputs.
/// <para>
/// Storage convention matches the rest of the OpenCL backend: FP16 values are
/// held in <c>ushort</c>-sized buffers (2 bytes each), the same layout
/// <c>Fp16Kernels.convert_fp32_to_fp16</c> / <see cref="MixedPrecisionKernels"/>
/// produce. Reads use the CORE OpenCL <c>vload_half</c> built-in (NOT the
/// optional <c>cl_khr_fp16</c> extension), so these kernels compile and run on
/// every OpenCL 1.0+ device regardless of whether it advertises native half
/// arithmetic — the extension is only required to declare <c>half</c> scalar
/// variables and do half-precision math, neither of which these kernels do.
/// </para>
/// <para>
/// Two variants:
/// <list type="bullet">
/// <item><c>gemm_fp16in_fp32out</c> — FP16 inputs, FP32 accumulator and output.
/// The industry-standard mixed-precision matmul (matches cuBLAS
/// <c>cublasGemmEx(CUDA_R_16F in, CUBLAS_COMPUTE_32F)</c> and the AMP forward
/// pass): half the memory bandwidth on activations, full FP32 accumulation so
/// long matmul chains don't lose precision.</item>
/// <item><c>gemm_fp16in_fp16out</c> — FP16 inputs and output, still accumulated
/// in FP32 internally then rounded to half on store (matches <c>cublasHgemm</c>
/// behaviour closely while avoiding the worst FP16-accumulation drift). For
/// forward-only inference where a half result is wanted.</item>
/// </list>
/// </para>
/// <para>
/// Both are tiled (16×16 work-group, FP32 local-memory staging) so the per-tile
/// reuse keeps RDNA/GCN/Intel/NVIDIA OpenCL devices compute-bound rather than
/// bandwidth-bound. RDNA1 and earlier lack matrix cores, so this packed-load +
/// FP32-accumulate path is the fastest correct option there; on hardware with
/// cooperative-matrix support a future variant can swap the inner product for a
/// matrix-core intrinsic without changing this contract.
/// </para>
/// </summary>
internal static class HalfPrecisionGemmKernels
{
    /// <summary>Kernel name: FP16 inputs, FP32 accumulator + output.</summary>
    public const string Fp16In32fOutKernelName = "gemm_fp16in_fp32out";

    /// <summary>Kernel name: FP16 inputs, FP32 accumulator, FP16 output.</summary>
    public const string Fp16In16fOutKernelName = "gemm_fp16in_fp16out";

    /// <summary>Work-group tile size; global sizes are rounded up to a multiple of this.</summary>
    public const int TileSize = 16;

    /// <summary>The kernel names this source compiles, for cache registration.</summary>
    public static string[] GetKernelNames() => new[]
    {
        Fp16In32fOutKernelName,
        Fp16In16fOutKernelName,
    };

    /// <summary>OpenCL C source for the FP16 GEMM kernels.</summary>
    public static string GetSource() => @"
// FP16 GEMM kernels — C = A * B, row-major.
//   A: M*K half values (stored as ushort), A[row*K + col]
//   B: K*N half values (stored as ushort), B[row*N + col]
//   C: M*N float (fp32 variant) or M*N half/ushort (fp16 variant)
// Uses core OpenCL vload_half / vstore_half (NO cl_khr_fp16 dependency).

#define HGEMM_TS 16

__kernel void gemm_fp16in_fp32out(
    __global const ushort* A,
    __global const ushort* B,
    __global float* C,
    const int M, const int N, const int K)
{
    const int lx = get_local_id(0);              // column within tile (N axis)
    const int ly = get_local_id(1);              // row within tile (M axis)
    const int col = get_group_id(0) * HGEMM_TS + lx;  // global N index
    const int row = get_group_id(1) * HGEMM_TS + ly;  // global M index

    __local float Asub[HGEMM_TS][HGEMM_TS];
    __local float Bsub[HGEMM_TS][HGEMM_TS];

    float acc = 0.0f;
    const int numTiles = (K + HGEMM_TS - 1) / HGEMM_TS;
    for (int t = 0; t < numTiles; ++t)
    {
        const int aCol = t * HGEMM_TS + lx;      // K index for A
        const int bRow = t * HGEMM_TS + ly;      // K index for B
        Asub[ly][lx] = (row < M && aCol < K)
            ? vload_half(row * K + aCol, (__global const half*)A) : 0.0f;
        Bsub[ly][lx] = (bRow < K && col < N)
            ? vload_half(bRow * N + col, (__global const half*)B) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int kk = 0; kk < HGEMM_TS; ++kk)
            acc += Asub[ly][kk] * Bsub[kk][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

__kernel void gemm_fp16in_fp16out(
    __global const ushort* A,
    __global const ushort* B,
    __global ushort* C,
    const int M, const int N, const int K)
{
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int col = get_group_id(0) * HGEMM_TS + lx;
    const int row = get_group_id(1) * HGEMM_TS + ly;

    __local float Asub[HGEMM_TS][HGEMM_TS];
    __local float Bsub[HGEMM_TS][HGEMM_TS];

    float acc = 0.0f;
    const int numTiles = (K + HGEMM_TS - 1) / HGEMM_TS;
    for (int t = 0; t < numTiles; ++t)
    {
        const int aCol = t * HGEMM_TS + lx;
        const int bRow = t * HGEMM_TS + ly;
        Asub[ly][lx] = (row < M && aCol < K)
            ? vload_half(row * K + aCol, (__global const half*)A) : 0.0f;
        Bsub[ly][lx] = (bRow < K && col < N)
            ? vload_half(bRow * N + col, (__global const half*)B) : 0.0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int kk = 0; kk < HGEMM_TS; ++kk)
            acc += Asub[ly][kk] * Bsub[kk][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < M && col < N)
        vstore_half(acc, row * N + col, (__global half*)C);
}
";
}
