// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The host side of <see cref="IDataTransformBackend"/> on every backend: it checks the sizes fit the kernels' 32-bit
/// indices, packs the ints (float parameters as their bits) in each kernel's order and launches the grid in pieces of at
/// most 65535 groups (every API's guaranteed 1-D grid), passing each piece's first group as the kernels' last int.
/// </summary>
internal static class PortableDataTransform
{
    /// <summary>
    /// Whether a failure while building the kernels leaves the backend usable, so the ops report themselves unavailable
    /// and the engine falls back (logged). Only runtime-fatal exceptions propagate.
    /// </summary>
    internal static bool IsBuildFailure(Exception ex) =>
        ex is not (OutOfMemoryException or StackOverflowException or AccessViolationException
            or System.Threading.ThreadAbortException);

    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="DataTransformKernelTemplate.GroupSize"/>.</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    private const int MaxGroupsPerLaunch = 65535;

    private static long Groups(long threads) => (threads + DataTransformKernelTemplate.GroupSize - 1) / DataTransformKernelTemplate.GroupSize;

    private static void LaunchChunked(Launcher launch, string kernel, long groups, IGpuBuffer[] buffers, int[] ints)
    {
        int firstSlot = ints.Length - 1;   // "first" is every kernel's last int
        for (long start = 0; start < groups; start += MaxGroupsPerLaunch)
        {
            var slice = (int[])ints.Clone();
            slice[firstSlot] = (int)start;
            launch(kernel, (int)Math.Min(MaxGroupsPerLaunch, groups - start), buffers, slice);
        }
    }

    // A grid whose first-group offset times 256 plus the thread index stays inside int.
    private static bool Fits(long threads) => threads > 0 && threads <= int.MaxValue - DataTransformKernelTemplate.GroupSize;

    internal static bool TryBoxMuller(Launcher launch, IGpuBuffer uniforms, IGpuBuffer output, int count, BoxMullerBranch branch,
        float mean, float std, bool clamp, float low, float high)
    {
        if (!Fits(2L * count) || (branch != BoxMullerBranch.Cosine && branch != BoxMullerBranch.Sine)) return false;
        LaunchChunked(launch, DataTransformKernelTemplate.BoxMuller, Groups(count), new[] { uniforms, output },
            new[] { count, (int)branch, PortableDataDecode.SingleBits(mean), PortableDataDecode.SingleBits(std), clamp ? 1 : 0,
                PortableDataDecode.SingleBits(low), PortableDataDecode.SingleBits(high), 0 });
        return true;
    }

    internal static bool TryUniformThreshold(Launcher launch, IGpuBuffer uniforms, IGpuBuffer? thresholds, IGpuBuffer output,
        int count, bool above, float threshold, float onSuccess, float otherwise)
    {
        if (!Fits(2L * count)) return false;
        LaunchChunked(launch, DataTransformKernelTemplate.Threshold, Groups(count), new[] { uniforms, output, thresholds ?? uniforms },
            new[] { count, above ? 1 : 0, thresholds is null ? 0 : 1, PortableDataDecode.SingleBits(threshold),
                PortableDataDecode.SingleBits(onSuccess), PortableDataDecode.SingleBits(otherwise), 0 });
        return true;
    }

    internal static bool TryLinearFilter(Launcher launch, IGpuBuffer x, IGpuBuffer b, IGpuBuffer a, IGpuBuffer? zi,
        IGpuBuffer y, IGpuBuffer workspace, int signals, int length, int nb, int na, bool guard, float maxAbs)
    {
        int order = Math.Max(nb, na);
        if (signals <= 0 || length <= 0 || nb <= 0 || na <= 0 || !Fits(signals)
            || (long)signals * length > int.MaxValue || (long)signals * order > int.MaxValue)
            return false;
        LaunchChunked(launch, DataTransformKernelTemplate.LinearFilter, Groups(signals),
            new[] { x, y, b, a, zi ?? x, workspace },
            new[] { signals, length, nb, na, zi is null ? 0 : 1, guard ? 1 : 0, PortableDataDecode.SingleBits(maxAbs), 0 });
        return true;
    }

    internal static bool TryGatherRows(Launcher launch, IGpuBuffer table, IGpuBuffer ids, IGpuBuffer output, int rows, int rowSize)
    {
        long threads = (long)rows * rowSize;
        if (rows <= 0 || rowSize <= 0 || !Fits(threads)) return false;
        LaunchChunked(launch, DataTransformKernelTemplate.GatherRows, Groups(threads), new[] { table, output, ids },
            new[] { rows, rowSize, 0 });
        return true;
    }

    internal static bool TryScatterRows(Launcher launch, IGpuBuffer source, IGpuBuffer ids, IGpuBuffer table, int rows, int rowSize)
    {
        long threads = (long)rows * rowSize;
        if (rows <= 0 || rowSize <= 0 || !Fits(threads)) return false;
        LaunchChunked(launch, DataTransformKernelTemplate.ScatterRows, Groups(threads), new[] { source, table, ids },
            new[] { rows, rowSize, 0 });
        return true;
    }

    internal static bool TryRoundIntegral(Launcher launch, IGpuBuffer input, IGpuBuffer output, int count, RoundingMode mode,
        float low, float high)
    {
        if (!Fits(count) || mode < RoundingMode.TowardZero || mode > RoundingMode.TowardPositiveInfinity) return false;
        LaunchChunked(launch, DataTransformKernelTemplate.RoundIntegral, Groups(count), new[] { input, output },
            new[] { count, (int)mode, PortableDataDecode.SingleBits(low), PortableDataDecode.SingleBits(high), 0 });
        return true;
    }
}
