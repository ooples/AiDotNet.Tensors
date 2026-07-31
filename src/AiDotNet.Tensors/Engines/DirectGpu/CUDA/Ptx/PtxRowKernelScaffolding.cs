using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Shared 256-thread tree reduction used by row-normalization PTX kernels.</summary>
internal static class PtxRowReduce
{
    internal static void Emit(StringBuilder ptx, string operation)
    {
        PtxCompat.ThrowIfNull(ptx, nameof(ptx));
        if (operation != "max.f32" && operation != "add.rn.f32")
            throw new ArgumentOutOfRangeException(nameof(operation), operation,
                "Only the audited max and add row reductions are supported.");

        for (int stride = PtxRowShape.BlockThreads / 2; stride > 0; stride >>= 1)
        {
            ptx.AppendLine($"    setp.lt.u32 %p3, %r0, {stride};");
            ptx.AppendLine("    @%p3 ld.shared.f32 %f10, [%rd10];");
            ptx.AppendLine($"    @%p3 ld.shared.f32 %f11, [%rd10+{stride * sizeof(float)}];");
            ptx.AppendLine($"    @%p3 {operation} %f10, %f10, %f11;");
            ptx.AppendLine("    @%p3 st.shared.f32 [%rd10], %f10;");
            ptx.AppendLine("    bar.sync 0;");
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
    internal const int MaxCount = 2048 * 4096;

    internal static bool IsSupported(int count) =>
        count > 0 && count % BlockThreads == 0 && count <= MaxCount;

    internal static bool IsPromoted(int count) => false;

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
