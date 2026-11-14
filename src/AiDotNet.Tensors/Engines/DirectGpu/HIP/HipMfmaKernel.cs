// Copyright (c) CrowdTrainer LLC. All rights reserved.
// AMD GEMM kernels: MFMA (CDNA matrix cores) and portable LDS-tiled kernels for every other AMD GPU.

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// AMD GEMM kernel sources, in two modules: the portable kernels (every AMD GPU) and the MFMA kernels (CDNA only).
/// </summary>
/// <remarks>
/// <para><b>MFMA targets:</b> MI100 (gfx908), MI200 (gfx90a), MI300 (gfx940/gfx941/gfx942), MI350 (gfx950).</para>
/// <para>The MFMA kernels are a correct baseline (one 64 x 64 tile per workgroup, single-buffered LDS), not a tuned
/// GEMM; large shapes go to the vendor BLAS (see <c>ShouldUseVendorGemm</c>).</para>
/// </remarks>
internal static class HipMfmaKernel
{
    /// <summary>Portable FP32 GEMM kernels (no matrix-core instructions); compile on every AMD target.</summary>
    public static string GetPortableSource()
    {
        // hiprtc provides the device intrinsics; no includes needed.
        return @"
#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

// ===========================================================================
// Scalar GEMM Kernel for GPUs without MFMA/WMMA
// Works on: RDNA1 (RX 5000), RDNA2 (RX 6000), older GCN
// Optimized with LDS tiling and coalesced memory access
// ===========================================================================
extern ""C"" __global__ void scalar_gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    // Tile dimensions for scalar kernel - 16x16 tiles
    const int TILE_SIZE = 16;

    // Block and thread indices
    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;

    // Global row and column
    const int row = blockIdx.x * TILE_SIZE + tx;
    const int col = blockIdx.y * TILE_SIZE + ty;

    // Shared memory for tiles with padding to avoid bank conflicts
    __shared__ float As[16][17];
    __shared__ float Bs[16][17];

    // Accumulator
    float acc = 0.0f;

    // Number of tiles in K dimension
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        const int kBase = t * TILE_SIZE;

        // Load A tile collaboratively
        const int aRow = blockIdx.x * TILE_SIZE + tx;
        const int aCol = kBase + ty;
        if (aRow < M && aCol < K) {
            As[tx][ty] = A[aRow * K + aCol];
        } else {
            As[tx][ty] = 0.0f;
        }

        // Load B tile collaboratively
        const int bRow = kBase + tx;
        const int bCol = blockIdx.y * TILE_SIZE + ty;
        if (bRow < K && bCol < N) {
            Bs[tx][ty] = B[bRow * N + bCol];
        } else {
            Bs[tx][ty] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product
        #pragma unroll
        for (int k = 0; k < 16; k++) {
            acc += As[tx][k] * Bs[k][ty];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        const int idx = row * N + col;
        C[idx] = beta == 0.0f ? alpha * acc : alpha * acc + beta * C[idx];
    }
}

// ===========================================================================
// Wave32 Optimized Kernel for RDNA GPUs
// RDNA3 (RX 7000): Can use WMMA when available
// RDNA1/RDNA2: Uses optimized scalar with wave32 occupancy hints
// ===========================================================================
extern ""C"" __global__ void rdna_gemm_wave32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
    __attribute__((amdgpu_waves_per_eu(8)))
    __attribute__((amdgpu_flat_work_group_size(256, 256)))
{
    // Optimized for RDNA wave32 architecture
    // Uses 2D thread block for better occupancy

    const int TILE_M = 32;
    const int TILE_N = 32;
    const int TILE_K = 16;

    // Each thread computes a 2x2 block of output
    const int tx = threadIdx.x % 16;  // 0-15
    const int ty = threadIdx.x / 16;  // 0-15

    const int row0 = blockIdx.x * TILE_M + tx * 2;
    const int col0 = blockIdx.y * TILE_N + ty * 2;

    // Shared memory with padding
    __shared__ float As[TILE_K][TILE_M + 4];
    __shared__ float Bs[TILE_K][TILE_N + 4];

    // 2x2 accumulators per thread
    float acc00 = 0.0f, acc01 = 0.0f;
    float acc10 = 0.0f, acc11 = 0.0f;

    const int numKTiles = (K + TILE_K - 1) / TILE_K;

    for (int kt = 0; kt < numKTiles; kt++) {
        const int kBase = kt * TILE_K;

        // Collaborative load - each thread loads 2 elements from A and B
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            const int loadIdx = threadIdx.x * 2 + i;
            const int aLoadRow = loadIdx % TILE_M;
            const int aLoadCol = loadIdx / TILE_M;

            if (aLoadCol < TILE_K) {
                const int gRow = blockIdx.x * TILE_M + aLoadRow;
                const int gCol = kBase + aLoadCol;
                As[aLoadCol][aLoadRow] = (gRow < M && gCol < K) ? A[gRow * K + gCol] : 0.0f;
            }

            const int bLoadRow = loadIdx / TILE_N;
            const int bLoadCol = loadIdx % TILE_N;

            if (bLoadRow < TILE_K) {
                const int gRow = kBase + bLoadRow;
                const int gCol = blockIdx.y * TILE_N + bLoadCol;
                Bs[bLoadRow][bLoadCol] = (gRow < K && gCol < N) ? B[gRow * N + gCol] : 0.0f;
            }
        }

        __syncthreads();

        // Compute 2x2 output per thread
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            const float a0 = As[k][tx * 2];
            const float a1 = As[k][tx * 2 + 1];
            const float b0 = Bs[k][ty * 2];
            const float b1 = Bs[k][ty * 2 + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write 2x2 results
    if (row0 < M && col0 < N)
        C[row0 * N + col0] = beta == 0.0f ? alpha * acc00 : alpha * acc00 + beta * C[row0 * N + col0];
    if (row0 < M && col0 + 1 < N)
        C[row0 * N + col0 + 1] = beta == 0.0f ? alpha * acc01 : alpha * acc01 + beta * C[row0 * N + col0 + 1];
    if (row0 + 1 < M && col0 < N)
        C[(row0 + 1) * N + col0] = beta == 0.0f ? alpha * acc10 : alpha * acc10 + beta * C[(row0 + 1) * N + col0];
    if (row0 + 1 < M && col0 + 1 < N)
        C[(row0 + 1) * N + col0 + 1] = beta == 0.0f ? alpha * acc11 : alpha * acc11 + beta * C[(row0 + 1) * N + col0 + 1];
}
";
    }

    /// <summary>MFMA GEMM kernels for CDNA; on other targets they compile to a scalar body that is never selected.</summary>
    public static string GetMfmaSource()
    {
        return @"
// ===========================================================================
// AMD MFMA GEMM kernels (CDNA: gfx908, gfx90a, gfx940-942, gfx950).
// Compiled as their own module so a failure here cannot take the portable GEMM kernels down with it.
//
// Register layouts are the ones AMD documents for the 32x32 single-block instructions (AMD Matrix Instruction
// Calculator; CDNA3 ISA section 7). For lane l of the 64-lane wave:
//   v_mfma_f32_32x32x2f32 : a = A[l % 32][l / 32],             b = B[l / 32][l % 32]
//   v_mfma_f32_32x32x8f16 : a[j] = A[l % 32][4 * (l / 32) + j], b[j] = B[4 * (l / 32) + j][l % 32], j = 0..3
//   accumulator item i (0..15) holds D[8 * (i / 4) + 4 * (l / 32) + (i % 4)][l % 32]
// ===========================================================================

#ifndef INFINITY
#define INFINITY __builtin_huge_valf()
#endif

#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
    defined(__gfx942__) || defined(__gfx950__)
#define AIDN_HAS_MFMA 1
#endif

// One 256-thread workgroup = 4 waves of 64 lanes in a 2 x 2 grid of 32 x 32 tiles: a 64 x 64 output tile.
#define MFMA_WG_M 64
#define MFMA_WG_N 64
#define MFMA_WG_K 32
#define MFMA_BLOCK 256
#define MFMA_WAVE 64

typedef float aidn_v16f __attribute__((ext_vector_type(16)));
typedef _Float16 aidn_v4h __attribute__((ext_vector_type(4)));

// Row of the output element that accumulator item i of lane l holds within its wave's 32 x 32 tile.
__device__ __forceinline__ int aidn_mfma_row(int i, int lane)
{
    return 8 * (i / 4) + 4 * (lane / 32) + (i % 4);
}

// C = alpha * A * B + beta * C, A [M x K], B [K x N], C [M x N], all row-major FP32.
// beta == 0 never reads C (BLAS semantics): a fresh output buffer can hold NaN, and 0 * NaN is NaN.
extern ""C"" __global__ void __launch_bounds__(MFMA_BLOCK) mfma_gemm_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    const int wave = threadIdx.x / MFMA_WAVE;
    const int lane = threadIdx.x % MFMA_WAVE;
    const int blockRow = blockIdx.x * MFMA_WG_M;
    const int blockCol = blockIdx.y * MFMA_WG_N;
    const int waveRow = (wave / 2) * 32;
    const int waveCol = (wave % 2) * 32;

    // +1 column of padding: the 32 lanes of a half-wave read 32 different rows of As without bank conflicts.
    __shared__ float As[MFMA_WG_M][MFMA_WG_K + 1];
    __shared__ float Bs[MFMA_WG_K][MFMA_WG_N + 1];

    aidn_v16f acc;
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int kBase = 0; kBase < K; kBase += MFMA_WG_K) {
        #pragma unroll
        for (int i = 0; i < (MFMA_WG_M * MFMA_WG_K) / MFMA_BLOCK; i++) {
            const int idx = threadIdx.x + i * MFMA_BLOCK;
            const int r = idx / MFMA_WG_K;
            const int c = idx % MFMA_WG_K;
            const int gr = blockRow + r;
            const int gc = kBase + c;
            As[r][c] = (gr < M && gc < K) ? A[(long long)gr * K + gc] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (MFMA_WG_K * MFMA_WG_N) / MFMA_BLOCK; i++) {
            const int idx = threadIdx.x + i * MFMA_BLOCK;
            const int r = idx / MFMA_WG_N;
            const int c = idx % MFMA_WG_N;
            const int gr = kBase + r;
            const int gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? B[(long long)gr * N + gc] : 0.0f;
        }
        __syncthreads();

#ifdef AIDN_HAS_MFMA
        const int m = lane % 32;
        const int kOff = lane / 32;
        #pragma unroll
        for (int k = 0; k < MFMA_WG_K; k += 2) {
            const float a = As[waveRow + m][k + kOff];
            const float b = Bs[k + kOff][waveCol + m];
            acc = __builtin_amdgcn_mfma_f32_32x32x2f32(a, b, acc, 0, 0, 0);
        }
#else
        // Same output elements per lane as the MFMA layout, computed with scalar FMAs.
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const int r = waveRow + aidn_mfma_row(i, lane);
            const int c = waveCol + lane % 32;
            float s = 0.0f;
            for (int k = 0; k < MFMA_WG_K; k++) s += As[r][k] * Bs[k][c];
            acc[i] += s;
        }
#endif
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int gr = blockRow + waveRow + aidn_mfma_row(i, lane);
        const int gc = blockCol + waveCol + lane % 32;
        if (gr < M && gc < N) {
            const long long o = (long long)gr * N + gc;
            C[o] = beta == 0.0f ? alpha * acc[i] : alpha * acc[i] + beta * C[o];
        }
    }
}

// FP16 inputs, FP32 accumulate and output; same tiling and layout rules as mfma_gemm_f32.
extern ""C"" __global__ void __launch_bounds__(MFMA_BLOCK) mfma_gemm_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K,
    const float alpha,
    const float beta)
{
    const int wave = threadIdx.x / MFMA_WAVE;
    const int lane = threadIdx.x % MFMA_WAVE;
    const int blockRow = blockIdx.x * MFMA_WG_M;
    const int blockCol = blockIdx.y * MFMA_WG_N;
    const int waveRow = (wave / 2) * 32;
    const int waveCol = (wave % 2) * 32;

    // +4 halves of padding keeps each row's 4-element reads 8-byte aligned and spreads rows over banks.
    __shared__ _Float16 As[MFMA_WG_M][MFMA_WG_K + 4];
    __shared__ _Float16 Bs[MFMA_WG_K][MFMA_WG_N + 4];

    aidn_v16f acc;
    #pragma unroll
    for (int i = 0; i < 16; i++) acc[i] = 0.0f;

    for (int kBase = 0; kBase < K; kBase += MFMA_WG_K) {
        #pragma unroll
        for (int i = 0; i < (MFMA_WG_M * MFMA_WG_K) / MFMA_BLOCK; i++) {
            const int idx = threadIdx.x + i * MFMA_BLOCK;
            const int r = idx / MFMA_WG_K;
            const int c = idx % MFMA_WG_K;
            const int gr = blockRow + r;
            const int gc = kBase + c;
            As[r][c] = (gr < M && gc < K) ? (_Float16)__half2float(A[(long long)gr * K + gc]) : (_Float16)0.0f;
        }
        #pragma unroll
        for (int i = 0; i < (MFMA_WG_K * MFMA_WG_N) / MFMA_BLOCK; i++) {
            const int idx = threadIdx.x + i * MFMA_BLOCK;
            const int r = idx / MFMA_WG_N;
            const int c = idx % MFMA_WG_N;
            const int gr = kBase + r;
            const int gc = blockCol + c;
            Bs[r][c] = (gr < K && gc < N) ? (_Float16)__half2float(B[(long long)gr * N + gc]) : (_Float16)0.0f;
        }
        __syncthreads();

#ifdef AIDN_HAS_MFMA
        const int m = lane % 32;
        const int kOff = 4 * (lane / 32);
        #pragma unroll
        for (int k = 0; k < MFMA_WG_K; k += 8) {
            aidn_v4h a, b;
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                a[j] = As[waveRow + m][k + kOff + j];
                b[j] = Bs[k + kOff + j][waveCol + m];
            }
            acc = __builtin_amdgcn_mfma_f32_32x32x8f16(a, b, acc, 0, 0, 0);
        }
#else
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            const int r = waveRow + aidn_mfma_row(i, lane);
            const int c = waveCol + lane % 32;
            float s = 0.0f;
            for (int k = 0; k < MFMA_WG_K; k++) s += (float)As[r][k] * (float)Bs[k][c];
            acc[i] += s;
        }
#endif
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        const int gr = blockRow + waveRow + aidn_mfma_row(i, lane);
        const int gc = blockCol + waveCol + lane % 32;
        if (gr < M && gc < N) {
            const long long o = (long long)gr * N + gc;
            C[o] = beta == 0.0f ? alpha * acc[i] : alpha * acc[i] + beta * C[o];
        }
    }
}
";
    }

    /// <summary>Kernel names in <see cref="GetPortableSource"/>.</summary>
    public static string[] GetPortableKernelNames() => new[] { "scalar_gemm_f32", "rdna_gemm_wave32" };

    /// <summary>Kernel names in <see cref="GetMfmaSource"/>.</summary>
    public static string[] GetMfmaKernelNames() => new[] { "mfma_gemm_f32", "mfma_gemm_f16" };

    /// <summary>Output tile edge of one MFMA workgroup (rows and columns); the launch grid divides M and N by it.</summary>
    public const int MfmaTile = 64;

    /// <summary>
    /// Gets the HIP compilation flags for different AMD architectures.
    /// </summary>
    public static string GetCompileFlags(AmdGpuArchitecture arch)
    {
        return arch switch
        {
            AmdGpuArchitecture.MI100 => "--offload-arch=gfx908",
            AmdGpuArchitecture.MI200 => "--offload-arch=gfx90a",
            AmdGpuArchitecture.MI300 => "--offload-arch=gfx940 --offload-arch=gfx941 --offload-arch=gfx942",
            AmdGpuArchitecture.MI350 => "--offload-arch=gfx950",
            AmdGpuArchitecture.RDNA3 => "--offload-arch=gfx1100",
            AmdGpuArchitecture.RDNA2 => "--offload-arch=gfx1030",  // No MFMA, uses scalar kernel
            AmdGpuArchitecture.RDNA => "--offload-arch=gfx1012",   // RX 5500 XT, no MFMA, uses scalar kernel
            _ => "--offload-arch=gfx900"  // Generic GCN
        };
    }

    /// <summary>
    /// Gets the HIP compilation flag for the exact target reported by the selected device.
    /// </summary>
    internal static string GetCompileFlags(string architectureTarget)
    {
        if (string.IsNullOrWhiteSpace(architectureTarget))
        {
            throw new ArgumentException("A HIP architecture target is required.", nameof(architectureTarget));
        }

        return $"--offload-arch={architectureTarget}";
    }

    /// <summary>
    /// Returns true if the architecture supports MFMA instructions.
    /// </summary>
    public static bool SupportsMfma(AmdGpuArchitecture arch)
    {
        return arch switch
        {
            AmdGpuArchitecture.MI100 => true,
            AmdGpuArchitecture.MI200 => true,
            AmdGpuArchitecture.MI300 => true,
            AmdGpuArchitecture.MI350 => true,
            _ => false  // RDNA1, RDNA2, RDNA3, GCN do not support MFMA
        };
    }

    /// <summary>
    /// Returns true if the architecture supports WMMA instructions (RDNA3 only).
    /// </summary>
    public static bool SupportsWmma(AmdGpuArchitecture arch)
    {
        return arch == AmdGpuArchitecture.RDNA3;
    }

    /// <summary>
    /// Gets the recommended kernel for the given architecture.
    /// </summary>
    public static string GetRecommendedKernel(AmdGpuArchitecture arch)
    {
        return arch switch
        {
            AmdGpuArchitecture.MI100 => "mfma_gemm_f32",
            AmdGpuArchitecture.MI200 => "mfma_gemm_f32",
            AmdGpuArchitecture.MI300 => "mfma_gemm_f32",
            AmdGpuArchitecture.MI350 => "mfma_gemm_f32",
            AmdGpuArchitecture.RDNA3 => "rdna_gemm_wave32",  // RDNA3 can use optimized wave32 kernel
            AmdGpuArchitecture.RDNA2 => "scalar_gemm_f32",   // No MFMA/WMMA, use scalar
            AmdGpuArchitecture.RDNA => "scalar_gemm_f32",    // No MFMA/WMMA, use scalar
            _ => "scalar_gemm_f32"  // Fallback to scalar for unknown/GCN
        };
    }
}

/// <summary>
/// AMD GPU architecture enumeration.
/// </summary>
public enum AmdGpuArchitecture
{
    Unknown,
    GCN,      // Pre-CDNA (Vega, etc.)
    RDNA,     // RX 5000 series
    RDNA2,    // RX 6000 series (no MFMA)
    RDNA3,    // RX 7000 series (WMMA)
    MI100,    // CDNA gfx908
    MI200,    // CDNA2 gfx90a
    MI300,    // CDNA3 gfx94x
    MI350     // CDNA4 gfx950 (MI350X / MI355X)
}
