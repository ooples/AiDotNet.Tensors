namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The gated delta rule's decode step behind <see cref="IEngine.GatedDeltaNetStep{T}"/> as one kernel: a warp per state
/// row reads the row once, updates it in place and reduces the output. A backend without it composes the step from the
/// engine's own ops through GatedDeltaNetStepHost: on a GPU engine, the backend's batched GEMMs and elementwise ops.
/// </summary>
[BackendSpecificRoute("CUDA one-kernel gated delta rule decode step (state read once and updated in place); every backend composes it natively from batched GEMMs and elementwise ops",
    nameof(IDirectGpuBackend.BatchedGemm), nameof(IDirectGpuBackend.Multiply), nameof(IDirectGpuBackend.Subtract), nameof(IDirectGpuBackend.Add))]
internal interface IGatedDeltaNetStepBackend
{
    /// <summary>
    /// With <c>k' = keyScale k</c> and <c>d = alpha</c> when <paramref name="decayFirst"/> (else 1):
    /// <c>S &lt;- alpha S + beta (v - d S k') k'^T</c> in place and <c>o = S q</c>, for <paramref name="rows"/> rows of a
    /// [rows, dv, dk] state; q and k are [rows, dk], v and o [rows, dv], alpha and beta [rows].
    /// </summary>
    void GatedDeltaNetStep(IGpuBuffer state, IGpuBuffer query, IGpuBuffer key, IGpuBuffer value, IGpuBuffer alpha,
        IGpuBuffer beta, IGpuBuffer output, int rows, int dv, int dk, float keyScale, bool decayFirst);
}
