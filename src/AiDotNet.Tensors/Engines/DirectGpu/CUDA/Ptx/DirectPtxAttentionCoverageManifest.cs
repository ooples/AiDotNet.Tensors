using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxAttentionCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

/// <summary>
/// One auditable assignment in issue #834. This inventory is code rather than a
/// markdown snapshot so tests can reject duplicate, missing, or accidentally
/// unassigned attention entry points as the backend surface evolves.
/// </summary>
internal sealed record DirectPtxAttentionCoverageCell(
    string Api,
    string ExistingCudaImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxAttentionCoverageStatus Status,
    string DirectPtxAssignment);

internal static class DirectPtxAttentionCoverageManifest
{
    internal static IReadOnlyList<DirectPtxAttentionCoverageCell> All { get; } =
    [
        new("ScaledDotProductAttention", "scaled_dot_product_attention (NVRTC)",
            "forward; optional weights/mask; causal; softcap; MHA/GQA/MQA",
            "dense BHSD", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v3 online D64 FP16 family for weight-free no-mask/no-softcap inference; remaining semantics planned"),
        new("ScaledDotProductAttentionBackward", "BatchedGemm + SoftmaxBackward composition",
            "backward; MHA; optional causal contract", "dense BHSD + materialized probabilities", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 deterministic materialized-probability D64 MHA for Sq/Skv=16/32/64/128; recomputation family planned"),
        new("FlashAttention", "flash_attention_v2 or scaled_dot_product_attention (NVRTC)",
            "compatibility forward; optional mask; causal; MHA", "dense BHSD", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v3 online D64 FP16 family for mask-free FP32 API inputs"),
        new("FlashAttentionV2", "flash_attention_v2 (NVRTC)",
            "forward; LSE; causal; optional additive attention bias", "dense BHSD", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v3 online D64 FP16 family for no-bias FP32 API inputs; additive bias planned"),
        new("FlashAttentionBackward", "flash_attention_backward plus deterministic gradq/gradkv (NVRTC)",
            "recomputation backward; LSE; causal; additive bias; deterministic mode",
            "dense BHSD", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 deterministic recomputation D64 MHA for Sq/Skv=16/32/64/128, optional broadcast/per-batch additive bias, unmasked/causal; other dimensions/dtypes and GQA planned"),
        new("GroupedQueryAttention", "grouped_query_attention (NVRTC)",
            "forward; optional materialized weights; causal; GQA/MQA", "dense BHSD", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v3 online D64 FP16 family for weight-free inference; weights-producing mode planned"),
        new("GroupedQueryAttentionBackward", "grouped_query_attention_backward plus deterministic gradq/gradkv (NVRTC)",
            "backward; explicit queries-per-KV; deterministic mode",
            "dense BHSD + materialized probabilities", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 deterministic materialized-probability D64 GQA/MQA for Sq/Skv=16/32/64/128; recomputation family planned"),
        new("FlashDecode", "flash_decode_partial + flash_decode_reduce (NVRTC)",
            "single-token split-K decode; MHA/GQA/MQA", "contiguous [S,Hkv,D] KV", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 single-pass register-online D64 FP32 MHA/GQA/MQA for S=16/32/64/128; explicit split hint falls back"),
        new("PagedAttentionDecode", "paged_attention_decode (NVRTC)",
            "single-token paged decode; MHA", "block-table paged KV", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 block-table D64 FP32 MHA decode for S=16/32/64/128 and block size 16/32"),
        new("PagedAttentionPrefill", "paged_attention_prefill (NVRTC)",
            "causal prefill with absolute start position; MHA", "block-table paged KV", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 register-online D64 FP32 MHA prefill for Q=2/4/8/16/32, max key length <=128, block size 16/32"),
        new("PagedAttentionDecodeGqa", "paged_attention_decode_gqa (NVRTC)",
            "single-token paged decode; GQA/MQA", "block-table paged KV", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 block-table D64 FP32 GQA/MQA decode for S=16/32/64/128 and block size 16/32"),
        new("PagedAttentionPrefillGqa", "paged_attention_prefill_gqa (NVRTC)",
            "causal prefill with absolute start position; GQA/MQA", "block-table paged KV", "FP32",
            DirectPtxAttentionCoverageStatus.ExperimentalDirectPtx,
            "v1 register-online D64 FP32 GQA/MQA prefill for Q=2/4/8/16/32, max key length <=128, block size 16/32")
    ];

    internal static DirectPtxAttentionCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxAttentionCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException($"Attention API '{api}' is not assigned in the #834 coverage manifest.");
    }
}
