namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxAttentionMaskKind
{
    None,
    CausalTopLeft,
    CausalBottomRight,
    AdditiveBias,
    Padding,
    SlidingWindow,
    BlockSparse
}

internal enum DirectPtxAttentionPhase
{
    Inference,
    TrainingForward,
    Backward
}

internal readonly record struct DirectPtxAttentionRequest(
    DirectPtxArchitectureFamily Architecture,
    int ComputeCapabilityMajor,
    int ComputeCapabilityMinor,
    DirectPtxPhysicalType InputType,
    DirectPtxPhysicalLayout Layout,
    int Batch,
    int QueryHeads,
    int KeyValueHeads,
    int QuerySequence,
    int KeyValueSequence,
    int HeadDimension,
    DirectPtxAttentionMaskKind Mask,
    DirectPtxAttentionPhase Phase,
    float DropoutProbability,
    bool IsRagged,
    bool UsesPagedKv);

internal readonly record struct DirectPtxEligibilityResult(bool IsEligible, string Reason)
{
    internal static DirectPtxEligibilityResult Eligible => new(true, "eligible");
    internal static DirectPtxEligibilityResult Reject(string reason) => new(false, reason);
}

/// <summary>
/// Single source of truth for the checked-in specialization's semantic domain.
/// Unsupported requests are explicit fallback decisions, never latent stride
/// or shape branches in PTX.
/// </summary>
internal static class DirectPtxAttentionEligibility
{
    internal static DirectPtxEligibilityResult Evaluate(in DirectPtxAttentionRequest request)
    {
        if (!DirectPtxArchitecture.HasValidatedOnlineAttention(
            request.ComputeCapabilityMajor, request.ComputeCapabilityMinor))
            return DirectPtxEligibilityResult.Reject("sm-version-not-validated");
        if (request.InputType != DirectPtxPhysicalType.Float16)
            return DirectPtxEligibilityResult.Reject("dtype-not-fp16");
        if (request.Layout != DirectPtxPhysicalLayout.Bhsd)
            return DirectPtxEligibilityResult.Reject("layout-not-dense-bhsd");
        if (request.Batch <= 0 || request.QueryHeads <= 0 || request.KeyValueHeads <= 0)
            return DirectPtxEligibilityResult.Reject("invalid-batch-or-head-count");
        if (request.QueryHeads % request.KeyValueHeads != 0)
            return DirectPtxEligibilityResult.Reject("query-heads-not-divisible-by-kv-heads");
        if (!PtxOnlineFusedAttention128x64Kernel.IsSupportedSequenceLength(request.QuerySequence))
            return DirectPtxEligibilityResult.Reject("query-sequence-bucket-not-implemented");
        if (!PtxOnlineFusedAttention128x64Kernel.IsSupportedSequenceLength(request.KeyValueSequence))
            return DirectPtxEligibilityResult.Reject("kv-sequence-bucket-not-implemented");
        if (request.HeadDimension != PtxOnlineFusedAttention128x64Kernel.HeadDimension)
            return DirectPtxEligibilityResult.Reject("head-dimension-not-64");
        if (request.Mask is not (DirectPtxAttentionMaskKind.None or
            DirectPtxAttentionMaskKind.CausalTopLeft or
            DirectPtxAttentionMaskKind.CausalBottomRight))
            return DirectPtxEligibilityResult.Reject("mask-kind-not-implemented");
        if (request.Phase != DirectPtxAttentionPhase.Inference)
            return DirectPtxEligibilityResult.Reject("training-or-backward-not-implemented");
        if (request.DropoutProbability != 0)
            return DirectPtxEligibilityResult.Reject("dropout-not-implemented");
        if (request.IsRagged)
            return DirectPtxEligibilityResult.Reject("ragged-layout-not-implemented");
        if (request.UsesPagedKv)
            return DirectPtxEligibilityResult.Reject("paged-kv-not-implemented");
        return DirectPtxEligibilityResult.Eligible;
    }
}
