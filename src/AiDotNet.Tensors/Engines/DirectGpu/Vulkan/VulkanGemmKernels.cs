// Copyright (c) AiDotNet. All rights reserved.
// Vulkan GLSL compute shaders for GEMM (C = A·B), compiled to SPIR-V at runtime
// (libshaderc). These give the Vulkan backend a REAL on-device matmul — the
// previous Gemm downloaded to the CPU, looped, and re-uploaded. Issue #560 (FP16)
// builds on the FP16 variant here.

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

/// <summary>
/// GEMM compute kernels. Flat 1-D dispatch: one invocation per output element
/// (the backend dispatches <c>M*N</c> threads in <c>local_size_x = 256</c>
/// workgroups via <c>CalculateWorkgroupCount</c>). Row-major throughout:
/// A is M×K (<c>A[row*K + k]</c>), B is K×N (<c>B[k*N + col]</c>), C is M×N
/// (<c>C[row*N + col]</c>). Push constants carry {M, N, K}.
/// <para>
/// Correctness-first (no shared-memory tiling): a simple one-thread-per-output
/// kernel that is unambiguously correct and already replaces the CPU fallback
/// with genuine GPU execution. A tiled / cooperative-matrix variant is a later
/// performance pass and does not change this contract.
/// </para>
/// </summary>
internal static class VulkanGemmKernels
{
    private const string Header = @"#version 450
layout(local_size_x = 256) in;
";

    /// <summary>FP32 GEMM: float A, float B, float C. Replaces the CPU-fallback matmul.</summary>
    public static string GemmFp32 => Header + @"
layout(set=0,binding=0) readonly buffer Ab { float A[]; };
layout(set=0,binding=1) readonly buffer Bb { float B[]; };
layout(set=0,binding=2) writeonly buffer Cb { float C[]; };
layout(push_constant) uniform PC { uint M; uint N; uint K; };
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= M * N) return;
    uint row = gid / N;
    uint col = gid % N;
    uint aBase = row * K;
    float acc = 0.0;
    for (uint kk = 0u; kk < K; ++kk)
        acc += A[aBase + kk] * B[kk * N + col];
    C[gid] = acc;
}";

    /// <summary>
    /// Mixed-precision GEMM: FP16 inputs (packed two halves per 32-bit word, the
    /// layout VulkanBackend.ConvertToFp16 produces), FP32 accumulator + output.
    /// Reads halves with the core <c>unpackHalf2x16</c> built-in (no extension).
    /// The AMP standard — matches cuBLAS cublasGemmEx(CUDA_R_16F, COMPUTE_32F).
    /// </summary>
    public static string GemmFp16In32fOut => Header + @"
layout(set=0,binding=0) readonly buffer Ab { uint Apacked[]; };
layout(set=0,binding=1) readonly buffer Bb { uint Bpacked[]; };
layout(set=0,binding=2) writeonly buffer Cb { float C[]; };
layout(push_constant) uniform PC { uint M; uint N; uint K; };

// Halves are packed two per 32-bit word (low 16 bits = even element, high 16 =
// odd — little-endian, matching ConvertToFp16's byte layout). unpackHalf2x16 is
// a core GLSL built-in (no extension needed).
void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= M * N) return;
    uint row = gid / N;
    uint col = gid % N;
    uint aBase = row * K;
    float acc = 0.0;
    for (uint kk = 0u; kk < K; ++kk) {
        uint ae = aBase + kk;
        uint be = kk * N + col;
        vec2 ap = unpackHalf2x16(Apacked[ae >> 1]);
        vec2 bp = unpackHalf2x16(Bpacked[be >> 1]);
        float av = ((ae & 1u) == 0u) ? ap.x : ap.y;
        float bv = ((be & 1u) == 0u) ? bp.x : bp.y;
        acc += av * bv;
    }
    C[gid] = acc;
}";

    /// <summary>
    /// Generalized transposed FP16 GEMM for the fused matmul BACKWARD: <c>C[Mo,No] = op(A)·op(B)</c>, FP16
    /// packed inputs, FP32 accumulator + output. <c>transA</c>/<c>transB</c> (0/1, via push constants) select how
    /// each logical operand is read from storage with no materialized transpose — op(A) logical [Mo,Kc]:
    /// transA==0 → A[i*Kc+p], transA==1 → A[p*Mo+i]; op(B) logical [Kc,No]: transB==0 → B[p*No+j],
    /// transB==1 → B[j*Kc+p]. With transA=0,transB=0 this is exactly <see cref="GemmFp16In32fOut"/>. The two
    /// backward GEMMs are gradA[M,K]=gradC·Bᵀ (transA=0,transB=1) and gradB[K,N]=Aᵀ·gradC (transA=1,transB=0).
    /// FP16-output grads are produced by running this into an FP32 scratch then <c>ConvertToFp16</c> (mirrors
    /// <see cref="VulkanBackend.Hgemm"/>), so the shader only ever writes FP32.
    /// </summary>
    public static string GemmFp16BackwardTransposed => Header + @"
layout(set=0,binding=0) readonly buffer Ab { uint Apacked[]; };
layout(set=0,binding=1) readonly buffer Bb { uint Bpacked[]; };
layout(set=0,binding=2) writeonly buffer Cb { float C[]; };
layout(push_constant) uniform PC { uint Mo; uint No; uint Kc; uint transA; uint transB; };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    if (gid >= Mo * No) return;
    uint row = gid / No;   // Mo index (i)
    uint col = gid % No;   // No index (j)
    float acc = 0.0;
    for (uint p = 0u; p < Kc; ++p) {
        uint ae = (transA == 0u) ? (row * Kc + p) : (p * Mo + row);
        uint be = (transB == 0u) ? (p * No + col) : (col * Kc + p);
        vec2 ap = unpackHalf2x16(Apacked[ae >> 1]);
        vec2 bp = unpackHalf2x16(Bpacked[be >> 1]);
        float av = ((ae & 1u) == 0u) ? ap.x : ap.y;
        float bv = ((be & 1u) == 0u) ? bp.x : bp.y;
        acc += av * bv;
    }
    C[gid] = acc;
}";
}
