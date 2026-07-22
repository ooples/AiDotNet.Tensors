// Copyright (c) AiDotNet. All rights reserved.
// PTX transfer tracking. Which CUDA kernels have a PTX replacement, and how far
// along it is — the ledger that answers "when can we delete the CUDA kernels?".
#if !NETFRAMEWORK

using System;
using System.Collections.Generic;
using System.Linq;

namespace AiDotNet.Tensors.Tests.Engines.PtxParity;

public enum PtxTransferStatus
{
    /// <summary>
    /// A PTX kernel targeting this CUDA kernel exists but is not yet promoted
    /// (fails closed behind the release gate / awaiting benchmark evidence).
    /// The CUDA kernel must stay.
    /// </summary>
    PtxInProgress,

    /// <summary>
    /// A PTX kernel has cleared parity + the release gate and is promoted as the
    /// default path. The CUDA kernel is now redundant and may be deleted.
    /// </summary>
    PtxPromotedReplaced,

    /// <summary>
    /// Deliberately excluded from the transfer denominator: an infra/utility
    /// kernel that is not a compute op, a genuine gap not covered by any epic
    /// #833 child, or a kernel a category issue will subsume without a dedicated
    /// port entry. Excluded kernels do not block <c>FullTransferComplete</c>; the
    /// <c>Note</c> states the disposition.
    /// </summary>
    NotPlanned
}

/// <summary>One CUDA kernel's PTX-replacement record.</summary>
public sealed record CudaToPtxEntry(
    string CudaKernel,
    string PtxKernel,
    PtxTransferStatus Status,
    string Note);

/// <summary>
/// The explicit record of every CUDA kernel that has a PTX replacement effort.
/// Kernels absent from this ledger have not been started (they are counted as
/// remaining by <see cref="CudaToPtxTransferTests"/>). Full transfer is complete
/// — and the CUDA kernels are deletable — only when every kernel in
/// <see cref="CudaKernelCensus"/> appears here as
/// <see cref="PtxTransferStatus.PtxPromotedReplaced"/>.
/// </summary>
public static class CudaToPtxTransferLedger
{
    public static IReadOnlyList<CudaToPtxEntry> Entries { get; } = new[]
    {
        new CudaToPtxEntry("sum_axis", "PtxFusedRowReduceF32Kernel", PtxTransferStatus.PtxInProgress,
            "row-sum reduction (#843); PTX kernel on agent/direct-ptx-reduction-843, fails closed until 3 clean >=1.10x runs."),
        new CudaToPtxEntry("reduce_sum", "PtxFusedRowReduceF32Kernel", PtxTransferStatus.PtxInProgress,
            "same reduction family (#843); shares the row-sum PTX kernel."),
        new CudaToPtxEntry("softmax", "PtxFusedSoftmaxF32Kernel", PtxTransferStatus.PtxInProgress,
            "row-softmax (#840); PTX kernel on agent/direct-ptx-softmax-840, fails closed until promotion evidence."),
        new CudaToPtxEntry("softmax_rows", "PtxFusedSoftmaxF32Kernel", PtxTransferStatus.PtxInProgress,
            "same softmax family (#840); shares the row-softmax PTX kernel."),
        new CudaToPtxEntry("rmsnorm_forward", "PtxFusedResidualRmsNormD64Kernel", PtxTransferStatus.PtxInProgress,
            "fused residual RMSNorm; PTX kernel exists but has no public route wired yet (see parity registry)."),
        new CudaToPtxEntry("dropout_mask", "PtxFusedPhiloxDropoutF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 fuses versioned Philox mask production into dropout forward; disabled until three clean benchmark/Nsight runs."),
        new CudaToPtxEntry("dropout_forward", "PtxFusedPhiloxDropoutF32Kernel", PtxTransferStatus.PtxInProgress,
            "issue #849 exact-shape FP32 fused dropout forward; established CUDA kernels remain the fail-closed fallback."),

        // --- Triaged anomalies: the 4 census kernels that mapped to no epic
        // #833 child during the full 888-kernel cross-reference. Recorded so the
        // tracker's disposition of every kernel is explicit.
        new CudaToPtxEntry("resident_mode", "(none)", PtxTransferStatus.NotPlanned,
            "GPU-residency infra/utility kernel, not a compute op; excluded from the PTX transfer."),
        new CudaToPtxEntry("fused_ddim_step", "(none)", PtxTransferStatus.NotPlanned,
            "DDIM diffusion sampler step is now assigned to the #849 stochastic manifest; no direct specialization has started yet."),
        new CudaToPtxEntry("squash", "(none)", PtxTransferStatus.NotPlanned,
            "capsule-network squash activation; reassigned to #839 (pointwise/activation) scope — no standalone transfer entry."),
        new CudaToPtxEntry("squash_backward", "(none)", PtxTransferStatus.NotPlanned,
            "squash activation backward; reassigned to #839 (pointwise/activation) scope."),
    };

    public static IEnumerable<CudaToPtxEntry> NotPlanned =>
        Entries.Where(e => e.Status == PtxTransferStatus.NotPlanned);

    public static IEnumerable<CudaToPtxEntry> Replaced =>
        Entries.Where(e => e.Status == PtxTransferStatus.PtxPromotedReplaced);

    public static IEnumerable<CudaToPtxEntry> InProgress =>
        Entries.Where(e => e.Status == PtxTransferStatus.PtxInProgress);
}
#endif
