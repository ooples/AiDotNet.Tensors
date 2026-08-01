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

    /// <summary>Appends a full-warp FP32 maximum reduction using five shuffle-down stages.</summary>
    internal static void AppendWarpMax(
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
            text.AppendLine($"    max.f32 {partialFloat}, {partialFloat}, {shuffledFloat};");
        }
    }

    /// <summary>Appends a scalar/v2/v4 pair reduction while one thread retains one output.</summary>
    /// <remarks>
    /// The emitted contract uses <c>%f0</c> as the accumulator, <c>%rd6/%rd7</c> as input walkers,
    /// <c>%r9</c> as the loop counter, <c>%p0</c> as the loop predicate, and <c>%f1..%f9</c> as
    /// scratch. Aligned vector loads reduce memory transactions for short contiguous reductions
    /// without increasing the launched warp count.
    /// </remarks>
    internal static void AppendVectorizedPairReduction(
        StringBuilder text,
        int length,
        int vectorWidth,
        bool squaredDifference,
        string loopLabel)
    {
        if (text is null) throw new ArgumentNullException(nameof(text));
        if (string.IsNullOrWhiteSpace(loopLabel)) throw new ArgumentException("A loop label is required.", nameof(loopLabel));
        if (vectorWidth is not (1 or 2 or 4) || length <= 0 || length % vectorWidth != 0)
            throw new ArgumentOutOfRangeException(nameof(vectorWidth));

        text.AppendLine("    mov.f32 %f0, 0f00000000;");
        text.AppendLine("    mov.u32 %r9, 0;");
        text.AppendLine($"{loopLabel}:");
        if (vectorWidth == 4)
        {
            text.AppendLine("    ld.global.nc.v4.f32 {%f1, %f2, %f3, %f4}, [%rd6];");
            text.AppendLine("    ld.global.nc.v4.f32 {%f5, %f6, %f7, %f8}, [%rd7];");
            for (int lane = 0; lane < 4; lane++)
                AppendPairTerm(text, squaredDifference, $"%f{lane + 1}", $"%f{lane + 5}", "%f9");
        }
        else if (vectorWidth == 2)
        {
            text.AppendLine("    ld.global.nc.v2.f32 {%f1, %f2}, [%rd6];");
            text.AppendLine("    ld.global.nc.v2.f32 {%f3, %f4}, [%rd7];");
            for (int lane = 0; lane < 2; lane++)
                AppendPairTerm(text, squaredDifference, $"%f{lane + 1}", $"%f{lane + 3}", "%f5");
        }
        else
        {
            text.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
            text.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
            AppendPairTerm(text, squaredDifference, "%f1", "%f2", "%f3");
        }
        int byteStride = vectorWidth * sizeof(float);
        text.AppendLine($"    add.u64 %rd6, %rd6, {byteStride};");
        text.AppendLine($"    add.u64 %rd7, %rd7, {byteStride};");
        text.AppendLine("    add.u32 %r9, %r9, 1;");
        text.AppendLine($"    setp.lt.u32 %p0, %r9, {length / vectorWidth};");
        text.AppendLine($"    @%p0 bra {loopLabel};");
    }

    private static void AppendPairTerm(
        StringBuilder text, bool squaredDifference, string left, string right, string difference)
    {
        if (squaredDifference)
        {
            text.AppendLine($"    sub.rn.f32 {difference}, {left}, {right};");
            text.AppendLine($"    fma.rn.f32 %f0, {difference}, {difference}, %f0;");
        }
        else
        {
            text.AppendLine($"    fma.rn.f32 %f0, {left}, {right}, %f0;");
        }
    }
}
