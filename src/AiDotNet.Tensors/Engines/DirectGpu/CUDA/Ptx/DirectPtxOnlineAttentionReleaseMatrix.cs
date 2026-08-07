namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal readonly record struct DirectPtxOnlineAttentionBasicReleaseCase(
    int Sequence,
    bool Causal,
    bool Epilogue,
    int Warps);

internal readonly record struct DirectPtxOnlineAttentionFamilyReleaseCase(
    int Batch,
    int QueryHeads,
    int KeyValueHeads,
    int QuerySequence,
    int KeyValueSequence,
    bool Causal,
    int CausalQueryOffset,
    bool Epilogue);

/// <summary>
/// Canonical, ordered SM86 online-attention release matrix shared by offline
/// cubin generation and the embedded-artifact/driver tests.
/// </summary>
internal static class DirectPtxOnlineAttentionReleaseMatrix
{
    internal static readonly DirectPtxOnlineAttentionBasicReleaseCase[] BasicCases =
    [
        new(16, false, false, 1),
        new(16, true, true, 1),
        new(32, false, false, 1),
        new(32, true, true, 2),
        new(64, false, false, 2),
        new(64, true, true, 4),
        new(128, false, false, 4),
        new(128, false, true, 8),
        new(128, true, false, 4),
        new(128, true, true, 8)
    ];

    internal static readonly DirectPtxOnlineAttentionFamilyReleaseCase[] FamilyCases =
    [
        new(2, 4, 2, 32, 64, false, 0, false),
        new(2, 8, 1, 32, 64, true, 0, false),
        new(2, 8, 2, 32, 64, true, 32, false),
        new(1, 4, 4, 128, 32, true, 0, false),
        new(1, 4, 2, 128, 64, true, -64, false),
        new(2, 4, 2, 32, 64, false, 0, true)
    ];
}
