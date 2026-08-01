using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Canonical text encodings shared by direct PTX emitters.</summary>
internal static class DirectPtxPtxText
{
    internal static void AppendModuleHeader(
        StringBuilder text,
        int computeCapabilityMajor,
        int computeCapabilityMinor,
        bool disableLoopUnrolling = false)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (computeCapabilityMajor <= 0) throw new ArgumentOutOfRangeException(nameof(computeCapabilityMajor));
        if (computeCapabilityMinor < 0) throw new ArgumentOutOfRangeException(nameof(computeCapabilityMinor));

        text.AppendLine(".version 7.1");
        text.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        text.AppendLine(".address_size 64");
        if (disableLoopUnrolling)
            text.AppendLine(".pragma \"nounroll\";");
    }

    internal static string Hex(float value)
        => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    /// <summary>Appends a full-warp FP32 sum reduction using five shuffle-down stages.</summary>
    /// <remarks>
    /// Callers own register declarations so the primitive composes with hand-written emitters.
    /// Every lane must execute the emitted sequence with a full active mask; lanes outside a
    /// ragged logical dimension must contribute zero.
    /// </remarks>
    internal static void AppendWarpSum(
        StringBuilder text,
        string partialFloat,
        string sourceBits,
        string shuffledBits,
        string shuffledFloat)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (string.IsNullOrWhiteSpace(partialFloat)) throw new ArgumentException("A partial register is required.", nameof(partialFloat));
        if (string.IsNullOrWhiteSpace(sourceBits)) throw new ArgumentException("A source-bits register is required.", nameof(sourceBits));
        if (string.IsNullOrWhiteSpace(shuffledBits)) throw new ArgumentException("A shuffled-bits register is required.", nameof(shuffledBits));
        if (string.IsNullOrWhiteSpace(shuffledFloat)) throw new ArgumentException("A shuffled-float register is required.", nameof(shuffledFloat));

        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            text.AppendLine($"    mov.b32 {sourceBits}, {partialFloat};");
            text.AppendLine($"    shfl.sync.down.b32 {shuffledBits}, {sourceBits}, {offset}, 31, 0xffffffff;");
            text.AppendLine($"    mov.b32 {shuffledFloat}, {shuffledBits};");
            text.AppendLine($"    add.rn.f32 {partialFloat}, {partialFloat}, {shuffledFloat};");
        }
    }
}
