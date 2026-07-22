using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxQkvRopeCacheCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxQkvRopeCacheCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxQkvRopeCacheCoverageStatus Status,
    string DirectPtxAssignment);

/// <summary>
/// Executable issue-#835 inventory. A focused test owns the exact public API
/// list so additions cannot silently escape a PTX assignment or explicit plan.
/// </summary>
internal static class DirectPtxQkvRopeCacheCoverageManifest
{
    internal static IReadOnlyList<DirectPtxQkvRopeCacheCoverageCell> All { get; } =
    [
        new("CudaBackend.QkvProjectionRoPECacheD64",
            "cuBLAS MatMulTransposed + NVRTC BiasAdd/RopeInterleaved + three D2D copies",
            "decode-token packed QKV projection, bias, interleaved Q/K RoPE, dense KV update",
            "input [HD], weights [3,H,D,HD], bias [3,H,D], cache [S,H,D]",
            "FP32",
            DirectPtxQkvRopeCacheCoverageStatus.ExperimentalDirectPtx,
            "v1 one-launch D64 decode for H=4/8/16, capacity=16/32/64/128, baked position"),
        new("IDirectGpuBackend.MatMulTransposed",
            "backend GEMM/GEMV implementation (cuBLAS on CUDA)",
            "dense and quantized projection building block",
            "row-major A[M,K], output-major B[N,K], C[M,N]",
            "FP32 plus backend-specific modes",
            DirectPtxQkvRopeCacheCoverageStatus.ExperimentalDirectPtx,
            "the admitted FP32 decode M=1 packed-QKV case is fused; prefill and quantized families planned"),
        new("DirectGpuTensorEngine.ApplyRoPEInterleavedGpu",
            "rope_interleaved (NVRTC)",
            "interleaved adjacent-pair rotary embedding with absolute position",
            "canonical contiguous rows plus [maxSequence,D/2] cosine/sine",
            "FP32",
            DirectPtxQkvRopeCacheCoverageStatus.ExperimentalDirectPtx,
            "Q and K rotation is fused into v1 decode projection; standalone/general rows remain baseline"),
        new("IEngine.ApplyRoPEInterleaved",
            "CPU scalar/SIMD or DirectGpuTensorEngine backend route",
            "public differentiable interleaved RoPE",
            "logical tensor shape ending in even D",
            "generic public dtype; GPU FP32",
            DirectPtxQkvRopeCacheCoverageStatus.ExperimentalDirectPtx,
            "inference Q/K decode subgraph is fused; standalone and backward routes remain baseline"),
        new("KVCache<T>.Append",
            "host tensor slice/copy",
            "dense host KV append with batch and sequence bookkeeping",
            "[batch,maxSequence,H,D]",
            "generic",
            DirectPtxQkvRopeCacheCoverageStatus.PlannedDirectPtx,
            "device-resident dense decoder flow must use the new fused backend primitive; host cache remains baseline"),
        new("DevicePagedKVCache.Append",
            "host block manager plus device-to-device Copy",
            "paged KV allocation, copy-on-write, append, and block-table bookkeeping",
            "[physicalBlock,blockPosition,H,D] plus host block table",
            "FP32",
            DirectPtxQkvRopeCacheCoverageStatus.PlannedDirectPtx,
            "paged-address fused QKV/RoPE write and metadata handoff remain a separate specialization"),
        new("Tensor reshape/transpose/slice QKV handoff",
            "generic tensor views, kernels, or D2D copies",
            "split packed QKV and form attention/cache physical layouts",
            "logical packed [row,3*HD] to BHSD or sequence-head-D",
            "generic",
            DirectPtxQkvRopeCacheCoverageStatus.ExperimentalDirectPtx,
            "v1 decode never materializes packed QKV or rotated K/V; prefill and training graph rewrites planned"),
        new("RecordingGpuBackend.RopeInterleaved",
            "recorded standalone backend operation",
            "record/replay decoder RoPE node",
            "same contract as IDirectGpuBackend.RopeInterleaved",
            "FP32",
            DirectPtxQkvRopeCacheCoverageStatus.PlannedDirectPtx,
            "v1 kernel is CUDA-graph capturable after prewarm; recordable fused-node substitution remains planned"),
        new("Attention ABI handoff",
            "separate Q/K/V layouts consumed by FlashDecode/PagedAttention",
            "feed rotated Q and updated K/V directly to decode attention",
            "Q [H,64], dense cache [S,H,64]",
            "FP32",
            DirectPtxQkvRopeCacheCoverageStatus.ExperimentalDirectPtx,
            "v1 writes the exact dense FlashDecode ABI with no transpose/copy; same-launch attention and paged handoff planned")
    ];

    internal static DirectPtxQkvRopeCacheCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxQkvRopeCacheCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException(
            $"QKV/RoPE/cache API '{api}' is not assigned in the #835 coverage manifest.");
    }
}
