#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxLossCoverageStatus
{
    ExistingBackend,
    ExperimentalDirectPtx,
    PromotedDirectPtx,
    PlannedDirectPtx
}

internal sealed record DirectPtxLossCoverageCell(
    string Api,
    string ExistingImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    string DirectPtxAssignment,
    DirectPtxLossCoverageStatus Status);

/// <summary>
/// Executable issue-#847 inventory. Every CUDA loss, objective, logits-to-loss,
/// and loss-backward boundary has one explicit direct-PTX lane. Shape and layout
/// families remain closed sets: a planned row must be split into exact physical
/// ABI cells before it can be promoted.
/// </summary>
internal static class DirectPtxLossCoverageManifest
{
    private const string Rows = "canonical contiguous row-major [rows,features]";
    private const string Vector = "canonical contiguous vector";

    internal static IReadOnlyList<DirectPtxLossCoverageCell> All { get; } =
    [
        Experimental("CudaBackend.MseLoss", "NVRTC mse_loss", "loss[i] = mean_j (pred-target)^2", Rows, "FP32", "v1 Ampere warp-row C64/C128 exact-shape cells"),
        Planned("CudaBackend.MaeLoss", "NVRTC mae_loss", "loss[i] = mean_j |pred-target|", Rows, "FP32", "warp-row-mae-families"),
        Planned("CudaBackend.HuberLoss", "NVRTC huber_loss", "smooth-L1 with baked delta", Rows, "FP32", "baked-delta-warp-row-families"),
        Planned("CudaBackend.BceLoss", "NVRTC bce_loss", "binary cross-entropy from probabilities", Vector, "FP32", "warp-row-bce-families"),
        Planned("CudaBackend.BceWithLogitsLoss", "NVRTC bce_with_logits_loss", "stable BCE from logits", Vector, "FP32", "stable-logsigmoid-warp-row-families"),
        Planned("CudaBackend.CrossEntropyLoss", "NVRTC cross_entropy_loss", "softmax + NLL over classes", Rows, "FP32", "fused-logsoftmax-nll-row-families"),
        Planned("CudaBackend.NllLoss", "NVRTC nll_loss", "negative log-likelihood from log-probs", Rows, "FP32 log-probs; INT32 targets", "gathered-nll-row-families"),
        Planned("CudaBackend.KlDivLoss", "NVRTC kl_div_loss", "KL divergence input vs target", Vector, "FP32", "warp-row-kldiv-families"),
        Planned("CudaBackend.FusedLinearCrossEntropyDense", "NVRTC fused_linear_cross_entropy", "logits = x@W then cross-entropy without materialized logits", "canonical contiguous [rows,hidden] and [hidden,classes]", "FP32", "fused-gemm-logsoftmax-nll-families"),
        Planned("CudaBackend.FusedLinearCrossEntropyIndex", "NVRTC fused_linear_cross_entropy_index", "indexed fused linear cross-entropy", "canonical contiguous with INT32 targets", "FP32; INT32 targets", "fused-gemm-indexed-nll-families"),
        Planned("CudaBackend.CosineEmbeddingLoss", "NVRTC cosine_embedding_loss", "1 - cos similarity margin loss", Rows, "FP32", "warp-row-cosine-families"),
        Planned("CudaBackend.TripletLoss", "NVRTC triplet_loss", "max(0, d(a,p)-d(a,n)+margin)", Rows, "FP32", "baked-margin-triplet-families"),
        Planned("CudaBackend.MseLossBackward", "NVRTC mse_loss_backward", "grad = 2/N * (pred-target) * gradOut", Vector, "FP32", "elementwise-mse-backward-families"),
        Planned("CudaBackend.MaeBackward", "NVRTC mae_backward", "grad = sign(pred-target)/N * gradOut", Vector, "FP32", "elementwise-mae-backward-families"),
        Planned("CudaBackend.HuberLossBackward", "NVRTC huber_loss_backward", "clamped smooth-L1 gradient", Vector, "FP32", "baked-delta-backward-families"),
        Planned("CudaBackend.CrossEntropyBackward", "NVRTC cross_entropy_backward", "softmax(pred) - onehot(target)", Rows, "FP32", "fused-softmax-minus-onehot-families"),
        Planned("CudaBackend.BinaryCrossEntropyBackward", "NVRTC binary_cross_entropy_backward", "BCE input gradient", Vector, "FP32", "elementwise-bce-backward-families"),
        Planned("DirectGpuEngine.MseLoss", "array upload + CudaBackend.MseLoss + download", "public array MSE loss", Rows, "generic public; CUDA FP32", "public-array-mse-routing"),
        Planned("DirectGpuTensorEngine.TensorMseLoss", "resident CudaBackend.MseLoss or CPU fallback", "public tensor MSE loss with reduction", "logical strided tensors; canonical contiguous fast path", "generic public; CUDA FP32", "public-tensor-mse-routing-and-materialization"),
        Planned("DirectGpuTensorEngine.TensorCrossEntropy", "CudaBackend.CrossEntropyLoss or CPU fallback", "public tensor cross-entropy", "logical [rows,classes]; canonical contiguous admitted view", "generic public; CUDA FP32", "public-cross-entropy-routing"),
    ];

    private static readonly IReadOnlyDictionary<string, DirectPtxLossCoverageCell> ByApi =
        All.ToDictionary(cell => cell.Api, StringComparer.Ordinal);

    internal static DirectPtxLossCoverageCell Get(string api) =>
        ByApi.TryGetValue(api, out DirectPtxLossCoverageCell? cell)
            ? cell
            : throw new KeyNotFoundException($"No direct-PTX loss coverage cell for '{api}'.");

    private static DirectPtxLossCoverageCell Experimental(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxLossCoverageStatus.ExperimentalDirectPtx);

    private static DirectPtxLossCoverageCell Planned(
        string api, string existing, string semantics, string layout, string dtypes, string assignment) =>
        new(api, existing, semantics, layout, dtypes, assignment, DirectPtxLossCoverageStatus.PlannedDirectPtx);
}
#endif
