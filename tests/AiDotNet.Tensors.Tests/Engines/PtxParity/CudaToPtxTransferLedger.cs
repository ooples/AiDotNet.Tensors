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
    PtxPromotedReplaced
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
    };

    public static IEnumerable<CudaToPtxEntry> Replaced =>
        Entries.Where(e => e.Status == PtxTransferStatus.PtxPromotedReplaced);

    public static IEnumerable<CudaToPtxEntry> InProgress =>
        Entries.Where(e => e.Status == PtxTransferStatus.PtxInProgress);
}
#endif
