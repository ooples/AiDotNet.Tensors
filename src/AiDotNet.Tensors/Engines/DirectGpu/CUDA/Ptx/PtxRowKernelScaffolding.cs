using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shared hierarchical warp reduction used by row-normalization PTX kernels.
/// </summary>
internal static class PtxRowReduce
{
    internal const int WarpCount = 8;
    internal const int SharedBytes = WarpCount * sizeof(float);

    internal static void Emit(StringBuilder ptx, string operation, string accumulator)
    {
        PtxCompat.ThrowIfNull(ptx, nameof(ptx));
        if (operation != "max.f32" && operation != "add.rn.f32")
            throw new ArgumentOutOfRangeException(nameof(operation), operation,
                "Only the audited max and add row reductions are supported.");
        if (accumulator != "%f0")
            throw new ArgumentOutOfRangeException(nameof(accumulator), accumulator,
                "The audited row kernels reduce their per-lane accumulator in %f0.");

        // Reduce the caller's register accumulator within each warp, publish only the eight
        // warp leaders, then let the first warp finish. The previous shared-memory ABI first
        // staged and reloaded all 256 lane partials; shuffles make that round trip redundant.
        ptx.AppendLine($"    mov.f32 %f10, {accumulator};");
        EmitWarpShuffle(ptx, operation, predicate: "");
        ptx.AppendLine("    and.b32 %r10, %r0, 31;");
        ptx.AppendLine("    setp.eq.u32 %p3, %r10, 0;");
        ptx.AppendLine("    shr.u32 %r11, %r0, 5;");
        ptx.AppendLine("    mul.wide.u32 %rd19, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd19, %rd5, %rd19;");
        ptx.AppendLine("    @%p3 st.shared.f32 [%rd19], %f10;");
        ptx.AppendLine("    bar.sync 0;");

        ptx.AppendLine(operation == "max.f32"
            ? "    mov.f32 %f10, 0fFF800000;"
            : "    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    setp.lt.u32 %p3, %r0, 8;");
        ptx.AppendLine("    @%p3 ld.shared.f32 %f10, [%rd10];");
        ptx.AppendLine("    setp.lt.u32 %p3, %r0, 32;");
        EmitWarpShuffle(ptx, operation, predicate: "@%p3 ");
        ptx.AppendLine("    setp.eq.u32 %p3, %r0, 0;");
        ptx.AppendLine("    @%p3 st.shared.f32 [%rd5], %f10;");
        ptx.AppendLine("    bar.sync 0;");
    }

    private static void EmitWarpShuffle(StringBuilder ptx, string operation, string predicate)
    {
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            ptx.AppendLine($"    {predicate}mov.b32 %r10, %f10;");
            ptx.AppendLine($"    {predicate}shfl.sync.down.b32 %r11, %r10, {offset}, 31, 0xffffffff;");
            ptx.AppendLine($"    {predicate}mov.b32 %f11, %r11;");
            ptx.AppendLine($"    {predicate}{operation} %f10, %f10, %f11;");
        }
    }
}

/// <summary>Single source of truth for the audited softmax-family row shapes.</summary>
internal static class PtxRowShape
{
    internal const int BlockThreads = 256;

    internal static bool IsSupported(int m, int n) =>
        m > 0 && m % 64 == 0 &&
        n > 0 && n % BlockThreads == 0 &&
        m is 64 or 128 or 256 or 512 or 1024 or 2048 &&
        n is 256 or 512 or 1024 or 2048 or 4096;

    internal static bool IsPromoted(int m, int n) => false;

    internal static void Validate(int m, int n, string operation)
    {
        if (!IsSupported(m, n))
            throw new ArgumentOutOfRangeException(
                nameof(m),
                $"{operation} supports M in {{64,128,256,512,1024,2048}}, " +
                "N in {256,512,1024,2048,4096}.");
    }
}

/// <summary>Shared launch bounds for flat softmax-family elementwise kernels.</summary>
internal static class PtxElementwiseShape
{
    internal const int BlockThreads = PtxRowShape.BlockThreads;
    // Each thread owns two float4 packs striped across the tensor halves. Consecutive
    // lanes therefore access consecutive 16-byte vectors in both transactions; making
    // the packs adjacent per thread instead put lanes 32 bytes apart and doubled L1
    // sector demand.
    internal const int VectorWidth = 8;
    internal const int MaxCount = 2048 * 4096;

    internal static bool IsSupported(int count) =>
        count > 0 && count % BlockThreads == 0 && count <= MaxCount;

    internal static bool IsPromoted(int count) => false;

    internal static int VectorGridBlocks(int count, int blockThreads = BlockThreads)
    {
        Validate(count, "Vectorized elementwise launch");
        if (blockThreads <= 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads));
        int elementsPerBlock = checked(blockThreads * VectorWidth);
        return checked((count + elementsPerBlock - 1) / elementsPerBlock);
    }

    internal static bool RequiresBoundsGuard(int count, int blockThreads = BlockThreads)
    {
        Validate(count, "Vectorized elementwise launch");
        if (blockThreads <= 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads));
        int vectorCount = count / VectorWidth;
        return vectorCount % blockThreads != 0;
    }

    internal static void Validate(int count, string operation)
    {
        if (!IsSupported(count))
            throw new ArgumentOutOfRangeException(
                nameof(count),
                $"{operation} supports a positive element count that is a multiple of " +
                $"{BlockThreads} up to {MaxCount}.");
    }
}

/// <summary>Shared exact tensor-view ABI validation for direct-PTX launches.</summary>
internal static class PtxAbiGuard
{
    internal static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
