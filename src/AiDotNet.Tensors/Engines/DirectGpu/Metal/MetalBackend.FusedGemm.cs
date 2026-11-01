// Copyright (c) CrowdTrainer LLC. All rights reserved.
// NOTE: validated only on a Metal runner -- the dev box has no Apple GPU.
using System;
using static AiDotNet.Tensors.Engines.DirectGpu.Metal.MetalNativeBindings;

namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

// One kernel each for GEMM members that were compositions of other backend members (TENS0008, #1638): the GemmBias
// family (a GEMM, a bias pass and an activation pass, each re-reading the product; swish and leaky ReLU also through a
// temporary), Gemm's alpha and beta (alpha was a second scaling pass and beta was ignored) and the FP16 GEMM (an FP32
// GEMM into a temporary, then a rounding pass). Compiled as its own library on first use.
public sealed partial class MetalBackend
{
    private const string FusedGemmLibraryName = "FusedGemm";

    /// <summary>The activation gemm_epilogue applies to scale * A * B (+ bias) (+ beta * C).</summary>
    private enum GemmActivation : uint { None = 0, Relu = 1, Gelu = 2, Sigmoid = 3, Tanh = 4, Swish = 5, LeakyRelu = 6 }

    // 16 x 16 output tiles staged through threadgroup memory (declared in the kernels, so no setThreadgroupMemoryLength),
    // accumulated in FP32. The activations are those of MetalKernels' gelu, sigmoid, silu and leaky_relu.
    internal const string FusedGemmSource = @"
#include <metal_stdlib>
using namespace metal;

#define FUSED_GEMM_TILE 16
constant float FUSED_GEMM_SQRT_2_OVER_PI = 0.7978845608028654f;
constant float FUSED_GEMM_GELU_COEF = 0.044715f;

kernel void gemm_epilogue(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* C [[buffer(3)]],
    constant uint& M [[buffer(4)]],
    constant uint& N [[buffer(5)]],
    constant uint& K [[buffer(6)]],
    constant uint& activation [[buffer(7)]],
    constant uint& hasBias [[buffer(8)]],
    constant float& scale [[buffer(9)]],
    constant float& beta [[buffer(10)]],
    constant float& slope [[buffer(11)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]])
{
    threadgroup float tileA[FUSED_GEMM_TILE][FUSED_GEMM_TILE];
    threadgroup float tileB[FUSED_GEMM_TILE][FUSED_GEMM_TILE];
    uint row = group_id.y * FUSED_GEMM_TILE + lid.y;
    uint col = group_id.x * FUSED_GEMM_TILE + lid.x;
    float acc = 0.0f;
    uint steps = (K + FUSED_GEMM_TILE - 1) / FUSED_GEMM_TILE;
    for (uint t = 0; t < steps; t++) {
        uint ak = t * FUSED_GEMM_TILE + lid.x;
        uint bk = t * FUSED_GEMM_TILE + lid.y;
        tileA[lid.y][lid.x] = (row < M && ak < K) ? A[row * K + ak] : 0.0f;
        tileB[lid.y][lid.x] = (bk < K && col < N) ? B[bk * N + col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < FUSED_GEMM_TILE; i++) acc += tileA[lid.y][i] * tileB[i][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row >= M || col >= N) return;
    uint idx = row * N + col;
    float x = scale * acc;
    if (hasBias != 0) x += bias[col];
    if (beta != 0.0f) x += beta * C[idx];
    switch (activation) {
        case 1: x = max(x, 0.0f); break;
        case 2: {
            // GELU is saturated at these tails in FP32. Avoid overflowing the cubic,
            // and keep Metal's fast tanh implementation within its finite range.
            if (x >= 10.0f) break;
            if (x <= -10.0f) { x = 0.0f; break; }
            float inner = FUSED_GEMM_SQRT_2_OVER_PI * (x + FUSED_GEMM_GELU_COEF * x * x * x);
            x = 0.5f * x * (1.0f + tanh(clamp(inner, -20.0f, 20.0f)));
            break;
        }
        case 3: x = 1.0f / (1.0f + exp(-x)); break;
        case 4: x = tanh(clamp(x, -20.0f, 20.0f)); break;
        case 5: x = x * (1.0f / (1.0f + exp(-x))); break;
        case 6: x = x > 0.0f ? x : slope * x; break;
        default: break;
    }
    C[idx] = x;
}

kernel void hgemm_tiled(
    device const half* A [[buffer(0)]],
    device const half* B [[buffer(1)]],
    device half* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint2 lid [[thread_position_in_threadgroup]],
    uint2 group_id [[threadgroup_position_in_grid]])
{
    threadgroup float tileA[FUSED_GEMM_TILE][FUSED_GEMM_TILE];
    threadgroup float tileB[FUSED_GEMM_TILE][FUSED_GEMM_TILE];
    uint row = group_id.y * FUSED_GEMM_TILE + lid.y;
    uint col = group_id.x * FUSED_GEMM_TILE + lid.x;
    float acc = 0.0f;
    uint steps = (K + FUSED_GEMM_TILE - 1) / FUSED_GEMM_TILE;
    for (uint t = 0; t < steps; t++) {
        uint ak = t * FUSED_GEMM_TILE + lid.x;
        uint bk = t * FUSED_GEMM_TILE + lid.y;
        tileA[lid.y][lid.x] = (row < M && ak < K) ? float(A[row * K + ak]) : 0.0f;
        tileB[lid.y][lid.x] = (bk < K && col < N) ? float(B[bk * N + col]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = 0; i < FUSED_GEMM_TILE; i++) acc += tileA[lid.y][i] * tileB[i][lid.x];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < M && col < N) C[row * N + col] = half(acc);
}
";

    private const int FusedGemmTile = 16;

    private IntPtr _fusedGemmLibrary;

    /// <summary>Fused GEMM + Bias + ReLU: output = ReLU(A * B + bias)</summary>
    public IGpuBuffer GemmBiasRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmEpilogue(A, B, bias, M, N, K, GemmActivation.Relu, 0f);

    /// <summary>Fused GEMM + Bias + GELU: output = GELU(A * B + bias)</summary>
    public IGpuBuffer GemmBiasGelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmEpilogue(A, B, bias, M, N, K, GemmActivation.Gelu, 0f);

    /// <summary>Fused GEMM + Bias + Sigmoid: output = Sigmoid(A * B + bias)</summary>
    public IGpuBuffer GemmBiasSigmoid(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmEpilogue(A, B, bias, M, N, K, GemmActivation.Sigmoid, 0f);

    /// <summary>Fused GEMM + Bias + Tanh: output = Tanh(A * B + bias)</summary>
    public IGpuBuffer GemmBiasTanh(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmEpilogue(A, B, bias, M, N, K, GemmActivation.Tanh, 0f);

    /// <summary>Fused GEMM + Bias (no activation): output = A * B + bias</summary>
    public IGpuBuffer GemmBias(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmEpilogue(A, B, bias, M, N, K, GemmActivation.None, 0f);

    public IGpuBuffer GemmBiasSwish(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K)
        => GemmEpilogue(A, B, bias, M, N, K, GemmActivation.Swish, 0f);

    public IGpuBuffer GemmBiasLeakyRelu(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K, float alpha = 0.01f)
        => GemmEpilogue(A, B, bias, M, N, K, GemmActivation.LeakyRelu, alpha);

    private IGpuBuffer GemmEpilogue(IGpuBuffer A, IGpuBuffer B, IGpuBuffer bias, int M, int N, int K,
        GemmActivation activation, float slope)
    {
        ThrowIfDisposed();
        if (A is not MetalGpuBuffer aBuffer || B is not MetalGpuBuffer bBuffer || bias is not MetalGpuBuffer biasBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var C = AllocateBuffer(checked(M * N));
        DispatchGemmEpilogue(aBuffer, bBuffer, biasBuffer, (MetalGpuBuffer)C, M, N, K, activation, 1f, 0f, slope);
        return C;
    }

    /// <summary>C = scale * A * B (+ bias[col]) (+ beta * C), then the activation; C is not read when beta is zero.</summary>
    private void DispatchGemmEpilogue(MetalGpuBuffer a, MetalGpuBuffer b, MetalGpuBuffer? bias, MetalGpuBuffer c,
        int M, int N, int K, GemmActivation activation, float scale, float beta, float slope)
    {
        if (M <= 0 || N <= 0) return;
        var pipeline = FusedGemmPipeline("gemm_epilogue");
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(a, 0);
        encoder.SetBuffer(b, 1);
        encoder.SetBuffer(bias ?? a, 2);   // unread when hasBias is 0
        encoder.SetBuffer(c, 3);
        encoder.SetBytes((uint)M, 4);
        encoder.SetBytes((uint)N, 5);
        encoder.SetBytes((uint)K, 6);
        encoder.SetBytes((uint)activation, 7);
        encoder.SetBytes(bias is null ? 0u : 1u, 8);
        encoder.SetBytes(scale, 9);
        encoder.SetBytes(beta, 10);
        encoder.SetBytes(slope, 11);
        encoder.DispatchThreadgroups(
            new MTLSize((ulong)((N + FusedGemmTile - 1) / FusedGemmTile), (ulong)((M + FusedGemmTile - 1) / FusedGemmTile), 1),
            new MTLSize(FusedGemmTile, FusedGemmTile, 1));
    }

    /// <inheritdoc/>
    public void Hgemm(IGpuBuffer aFp16, IGpuBuffer bFp16, IGpuBuffer cFp16, int m, int n, int k)
    {
        ThrowIfDisposed();
        ValidateHalfGemmArgs(aFp16, bFp16, cFp16, m, n, k);
        if (aFp16 is not MetalGpuBuffer a || bFp16 is not MetalGpuBuffer b || cFp16 is not MetalGpuBuffer c)
            throw new ArgumentException("Buffers must be MetalGpuBuffer");
        var pipeline = FusedGemmPipeline("hgemm_tiled");
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(a, 0);
        encoder.SetBuffer(b, 1);
        encoder.SetBuffer(c, 2);
        encoder.SetBytes((uint)m, 3);
        encoder.SetBytes((uint)n, 4);
        encoder.SetBytes((uint)k, 5);
        encoder.DispatchThreadgroups(
            new MTLSize((ulong)((n + FusedGemmTile - 1) / FusedGemmTile), (ulong)((m + FusedGemmTile - 1) / FusedGemmTile), 1),
            new MTLSize(FusedGemmTile, FusedGemmTile, 1));
    }

    private MetalPipelineState FusedGemmPipeline(string kernelName)
    {
        if (_fusedGemmLibrary == IntPtr.Zero)
        {
            lock (_tunedSlotsLock)
            {
                if (_fusedGemmLibrary == IntPtr.Zero)
                    _fusedGemmLibrary = _shaderLibrary.CompileLibrary(FusedGemmLibraryName, FusedGemmSource);
            }
        }
        var pipeline = GetPipeline(FusedGemmLibraryName, _fusedGemmLibrary, kernelName);
        if (pipeline.MaxTotalThreadsPerThreadgroup < FusedGemmTile * FusedGemmTile)
            throw new GpuCapabilityUnavailableException(
                $"Metal {kernelName} needs {FusedGemmTile * FusedGemmTile} threads per threadgroup; this pipeline allows {pipeline.MaxTotalThreadsPerThreadgroup}.");
        return pipeline;
    }
}
