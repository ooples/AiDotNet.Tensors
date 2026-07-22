using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSoftmaxCoverageStatus
{
    BaselineOnly,
    ExperimentalDirectPtx,
    PlannedDirectPtx
}

/// <summary>
/// One auditable assignment in issue #840 (softmax, log-softmax, masking, log-sum-exp and
/// backward fusions). The inventory is code rather than a markdown snapshot so tests reject
/// duplicate, missing, or accidentally unassigned normalization-of-logits entry points as
/// the backend surface evolves.
/// </summary>
internal sealed record DirectPtxSoftmaxCoverageCell(
    string Api,
    string ExistingCudaImplementation,
    string Semantics,
    string PhysicalLayout,
    string DTypes,
    DirectPtxSoftmaxCoverageStatus Status,
    string DirectPtxAssignment);

internal static class DirectPtxSoftmaxCoverageManifest
{
    private const string RowMajor = "contiguous row-major [M,N], softmax over the last axis";
    private const string Planned =
        "shape/dtype-specific PTX row-normalization kernel; no generic runtime-stride kernel";
    private const string RowSoftmax =
        "one-block-per-row shared-resident PTX kernel (PtxSoftmaxKernel) with in-block tree " +
        "reductions for the max and exp-sum; fails closed until GPU-validated and promoted";

    internal static IReadOnlyList<DirectPtxSoftmaxCoverageCell> All { get; } =
    [
        new("CudaBackend.Softmax", "cuDNN/NVRTC row softmax", "row-wise stable softmax over the last axis",
            RowMajor, "FP32", DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx, RowSoftmax),
        new("CudaBackend.SoftmaxRows", "NVRTC row softmax", "row-wise stable softmax, explicit rows API",
            RowMajor, "FP32", DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx, RowSoftmax),
        new("CudaBackend.SoftmaxBackward", "NVRTC softmax backward", "dX = softmax * (dY - sum(dY*softmax))",
            RowMajor + ", plus upstream grad", "FP32", DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.LogSoftmax", "NVRTC log-softmax", "x - logsumexp over the last axis",
            RowMajor, "FP32", DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            "one-block-per-row shared-resident PTX kernel (PtxLogSoftmaxKernel) subtracting the tree-reduced log-partition; fails closed until GPU-validated and promoted"),
        new("CudaBackend.LogSumExpAxis", "NVRTC log-sum-exp", "log(sum(exp(x))) over an axis with max shift",
            RowMajor, "FP32", DirectPtxSoftmaxCoverageStatus.ExperimentalDirectPtx,
            "one-block-per-row shared-resident PTX reduction (PtxLogSumExpKernel) writing the [M] log-partition vector; fails closed until GPU-validated and promoted"),
        new("CudaBackend.LogSumExpBackward", "NVRTC log-sum-exp backward", "dX = softmax(x) * dY",
            RowMajor + ", plus upstream grad", "FP32", DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.MaskedFillKernel", "NVRTC masked fill", "conditional fill by mask before softmax",
            RowMajor + ", plus mask", "FP32", DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.MaskedFillBackward", "NVRTC masked-fill backward", "zero the gradient at masked positions",
            RowMajor + ", plus mask", "FP32", DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.Sparsemax", "NVRTC sparsemax", "Euclidean projection of logits onto the simplex",
            RowMajor, "FP32", DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx, Planned),
        new("CudaBackend.TaylorSoftmax", "NVRTC Taylor softmax", "second-order Taylor approximation of softmax",
            RowMajor, "FP32", DirectPtxSoftmaxCoverageStatus.PlannedDirectPtx, Planned)
    ];

    internal static DirectPtxSoftmaxCoverageCell Get(string api)
    {
        PtxCompat.ThrowIfNullOrWhiteSpace(api, nameof(api));
        foreach (DirectPtxSoftmaxCoverageCell cell in All)
            if (string.Equals(cell.Api, api, StringComparison.Ordinal)) return cell;
        throw new KeyNotFoundException($"Softmax API '{api}' is not assigned in the #840 coverage manifest.");
    }
}
