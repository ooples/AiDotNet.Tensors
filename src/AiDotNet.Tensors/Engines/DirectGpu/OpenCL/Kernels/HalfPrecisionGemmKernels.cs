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

    /// <summary>Kernel name: generalized transposed FP16 GEMM for the fused matmul backward
    /// (<c>C = op(A)·op(B)</c> with per-operand transpose + selectable FP32/FP16 output).</summary>
    public const string Fp16BackwardKernelName = "gemm_fp16_backward";

    /// <summary>Work-group tile size; global sizes are rounded up to a multiple of this.</summary>
    public const int TileSize = 16;

    /// <summary>The kernel names this source compiles, for cache registration.</summary>
    public static string[] GetKernelNames() => new[]
    {
        Fp16In32fOutKernelName,
        Fp16In16fOutKernelName,
        Fp16BackwardKernelName,
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

// Generalized transposed FP16 GEMM for the fused matmul BACKWARD:
//   C[Mo,No] = op(A) . op(B),  FP16 inputs, FP32 accumulate.
//   transA/transB (0/1) select how the logical operand is read from storage WITHOUT a materialized transpose:
//     op(A) logical [Mo,Kc]:  transA==0 -> A stored [Mo,Kc], A[i,p]=A[i*Kc+p]
//                             transA==1 -> A stored [Kc,Mo], A[i,p]=A[p*Mo+i]
//     op(B) logical [Kc,No]:  transB==0 -> B stored [Kc,No], B[p,j]=B[p*No+j]
//                             transB==1 -> B stored [No,Kc], B[p,j]=B[j*Kc+p]
//   gradOutHalf (0/1) selects the stored output dtype (FP32 float / FP16 half); accumulate is FP32 either way.
//   C is a raw byte buffer (uchar*) cast inside per gradOutHalf. With transA=0,transB=0,gradOutHalf=0 this is
//   exactly gemm_fp16in_fp32out, so the tiled staging matches the verified forward kernel.
//   The two backward GEMMs map as:
//     gradA[M,K] = gradC[M,N].B^T : Mo=M,No=K,Kc=N,  A=gradC transA=0,  B=Bfwd transB=1
//     gradB[K,N] = A^T.gradC[M,N] : Mo=K,No=N,Kc=M,  A=Afwd transA=1,   B=gradC transB=0
__kernel void gemm_fp16_backward(
    __global const ushort* A,
    __global const ushort* B,
    __global uchar* C,
    const int Mo, const int No, const int Kc,
    const int transA, const int transB, const int gradOutHalf)
{
    const int lx = get_local_id(0);                  // column within tile (No axis)
    const int ly = get_local_id(1);                  // row within tile (Mo axis)
    const int col = get_group_id(0) * HGEMM_TS + lx; // global No index (j)
    const int row = get_group_id(1) * HGEMM_TS + ly; // global Mo index (i)

    __local float Asub[HGEMM_TS][HGEMM_TS];
    __local float Bsub[HGEMM_TS][HGEMM_TS];

    float acc = 0.0f;
    const int numTiles = (Kc + HGEMM_TS - 1) / HGEMM_TS;
    for (int t = 0; t < numTiles; ++t)
    {
        const int aK = t * HGEMM_TS + lx;            // Kc index paired with row for A
        const int bK = t * HGEMM_TS + ly;            // Kc index paired with col for B
        float av = 0.0f;
        if (row < Mo && aK < Kc)
        {
            const int idxA = (transA == 0) ? (row * Kc + aK) : (aK * Mo + row);
            av = vload_half(idxA, (__global const half*)A);
        }
        Asub[ly][lx] = av;
        float bv = 0.0f;
        if (bK < Kc && col < No)
        {
            const int idxB = (transB == 0) ? (bK * No + col) : (col * Kc + bK);
            bv = vload_half(idxB, (__global const half*)B);
        }
        Bsub[ly][lx] = bv;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int kk = 0; kk < HGEMM_TS; ++kk)
            acc += Asub[ly][kk] * Bsub[kk][lx];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (row < Mo && col < No)
    {
        const int o = row * No + col;
        if (gradOutHalf == 0)
            ((__global float*)C)[o] = acc;
        else
            vstore_half(acc, o, (__global half*)C);
    }
}
";
}
