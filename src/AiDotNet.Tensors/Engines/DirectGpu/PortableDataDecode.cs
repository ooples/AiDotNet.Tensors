// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The host side of <see cref="IDataDecodeBackend"/>, shared by every backend: it validates the sizes, packs the int
/// parameters in each kernel's order and launches the per-element kernels in pieces of at most 65535 groups (every
/// API's guaranteed 1-D grid). The backend supplies only the launch.
/// </summary>
internal static class PortableDataDecode
{
    /// <summary>
    /// Whether a failure while building the kernels leaves the backend usable, so the op reports itself unavailable and
    /// the engine falls back (logged). Only runtime-fatal exceptions propagate.
    /// </summary>
    internal static bool IsBuildFailure(Exception ex) =>
        ex is not (OutOfMemoryException or StackOverflowException or AccessViolationException
            or System.Threading.ThreadAbortException);

    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="DataDecodeKernelTemplate.GroupSize"/>.</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    private const int MaxGroupsPerLaunch = 65535;

    /// <summary>Launches <paramref name="items"/> threads in chunks of at most 65535 groups, the chunk's first item in the
    /// last int.</summary>
    private static void LaunchItems(Launcher launch, string kernel, int items, IGpuBuffer[] buffers, int[] ints)
    {
        int firstSlot = ints.Length - 1;
        const long perLaunch = (long)MaxGroupsPerLaunch * DataDecodeKernelTemplate.GroupSize;
        for (long start = 0; start < items; start += perLaunch)
        {
            long count = Math.Min(perLaunch, items - start);
            var slice = (int[])ints.Clone();
            slice[firstSlot] = (int)start;
            launch(kernel, (int)((count + DataDecodeKernelTemplate.GroupSize - 1) / DataDecodeKernelTemplate.GroupSize),
                buffers, slice);
        }
    }

    private static int Segments(int length) => (length + DataDecodeKernelTemplate.SegmentBytes - 1) / DataDecodeKernelTemplate.SegmentBytes;

    internal static bool TryDecode(Launcher launch, IGpuBuffer words, IGpuBuffer output, int count,
        NumericStorageType sourceType, bool littleEndian)
    {
        int width;
        try { width = sourceType.GetByteWidth(); }
        catch (ArgumentOutOfRangeException) { return false; }
        if (count <= 0 || (long)count * width > int.MaxValue - 3) return false;
        LaunchItems(launch, DataDecodeKernelTemplate.Decode, count, new[] { words, output },
            new[] { count, (int)sourceType, width, littleEndian ? 1 : 0, 0 });
        return true;
    }

    internal static bool TryCountSeparators(Launcher launch, IGpuBuffer words, IGpuBuffer counts, int length, byte delimiter)
    {
        if (length <= 0) return false;
        int segments = Segments(length);
        LaunchItems(launch, DataDecodeKernelTemplate.CountSeparators, segments, new[] { words, counts },
            new[] { length, delimiter, segments, 0 });
        return true;
    }

    internal static bool TryScan(Launcher launch, IGpuBuffer values, IGpuBuffer total, int count)
    {
        if (count <= 0) return false;
        launch(DataDecodeKernelTemplate.Scan, 1, new[] { values, total }, new[] { count });
        return true;
    }

    internal static bool TryScatterBounds(Launcher launch, IGpuBuffer words, IGpuBuffer offsets, IGpuBuffer bounds,
        int length, byte delimiter, bool trailingNewline)
    {
        if (length <= 0) return false;
        int segments = Segments(length);
        LaunchItems(launch, DataDecodeKernelTemplate.ScatterBounds, segments, new[] { words, offsets, bounds },
            new[] { length, delimiter, segments, trailingNewline ? 1 : 0, 0 });
        return true;
    }

    internal static bool TryParse(Launcher launch, IGpuBuffer words, IGpuBuffer bounds, IGpuBuffer output,
        IGpuBuffer status, int fields, int styleMask, float missingValue)
    {
        if (fields <= 0 || fields > int.MaxValue / 2) return false;
        LaunchItems(launch, DataDecodeKernelTemplate.ParseFields, fields, new[] { words, bounds, output, status },
            new[] { fields, styleMask, SingleBits(missingValue), 0 });
        return true;
    }

    /// <summary>The bits of a float, net471-safe.</summary>
    internal static int SingleBits(float value)
    {
        var one = new[] { value };
        var bits = new int[1];
        Buffer.BlockCopy(one, 0, bits, 0, sizeof(float));
        return bits[0];
    }
}
