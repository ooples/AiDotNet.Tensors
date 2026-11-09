using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Launch-time checks for kernels whose element count comes from the caller: a count past a buffer's capacity would
/// make the kernel read or write outside that buffer.
/// </summary>
internal static class GpuBufferCapacity
{
    /// <summary>
    /// True when a launch of <paramref name="count"/> elements has nothing to do; a negative count throws rather than
    /// returning with the output unwritten.
    /// </summary>
    internal static bool IsEmptyLaunch(int count)
    {
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), $"The element count {count} is negative.");
        return count == 0;
    }

    /// <summary>Throws when <paramref name="buffer"/> holds fewer than <paramref name="count"/> 4-byte elements.</summary>
    internal static void Require(IGpuBuffer buffer, int count, string name)
    {
        if (buffer is null) throw new ArgumentNullException(name);
        if (count < 0) throw new ArgumentOutOfRangeException(nameof(count), $"The element count {count} is negative.");
        if (buffer.SizeInBytes < (long)count * sizeof(float))
            throw new ArgumentException(
                $"{name} holds {buffer.SizeInBytes / sizeof(float)} 4-byte elements; the launch needs {count}.", name);
    }

    /// <summary>The <see cref="IDirectGpuBackend.ScatterPositions"/> arguments a launch can trust.</summary>
    internal static void RequirePositions(IGpuBuffer output, int srcDim, int inner, int count)
    {
        if (srcDim <= 0) throw new ArgumentOutOfRangeException(nameof(srcDim), $"The scatter axis length {srcDim} must be positive.");
        if (inner <= 0) throw new ArgumentOutOfRangeException(nameof(inner), $"The inner size {inner} must be positive.");
        Require(output, count, nameof(output));
    }
}
