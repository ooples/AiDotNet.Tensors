// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The host side of <see cref="INumericRecurrenceBackend"/> on every backend: it checks the shape fits the kernels'
/// 32-bit indices and float-encoded integers, sizes the grid and launches it in pieces of at
/// most 65535 groups (every API's guaranteed 1-D grid), passing each piece's first group as the kernels' last int.
/// </summary>
internal static class PortableNumericRecurrence
{
    /// <summary>
    /// Whether a failure while building the kernels leaves the backend usable, so the ops report themselves unavailable
    /// and the engine falls back (logged). Only runtime-fatal exceptions propagate.
    /// </summary>
    internal static bool IsBuildFailure(Exception ex) =>
        ex is not (OutOfMemoryException or StackOverflowException or AccessViolationException
            or System.Threading.ThreadAbortException);

    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="NumericRecurrenceKernelTemplate.GroupSize"/>.</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    private const int MaxGroupsPerLaunch = 65535;

    // Integers ride in float buffers: exact only below 2^24.
    private const int MaxExactInteger = 1 << 24;

    private static int Groups(long threads) => (int)((threads + NumericRecurrenceKernelTemplate.GroupSize - 1) / NumericRecurrenceKernelTemplate.GroupSize);

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

    internal static bool TryTridiagonalSolve(Launcher launch, IGpuBuffer lower, IGpuBuffer diagonal, IGpuBuffer upper,
        IGpuBuffer rhs, IGpuBuffer x, IGpuBuffer workspace, int batch, int n)
    {
        if (batch <= 0 || n <= 0 || (long)batch * n > int.MaxValue) return false;
        LaunchChunked(launch, NumericRecurrenceKernelTemplate.Tridiagonal, Groups(batch),
            new[] { lower, x, diagonal, upper, rhs, workspace }, new[] { batch, n, 0 });
        return true;
    }

    internal static bool TryFritschCarlsonLimit(Launcher launch, IGpuBuffer slopes, IGpuBuffer secants, IGpuBuffer limited,
        int batch, int n)
    {
        if (batch <= 0 || n <= 0 || (long)batch * n > int.MaxValue) return false;
        LaunchChunked(launch, NumericRecurrenceKernelTemplate.FritschCarlson, Groups(batch),
            new[] { slopes, limited, secants }, new[] { batch, n, 0 });
        return true;
    }

    private static bool CtcFits(int maxTime, int batch, int classes, int maxStates, int blank) =>
        maxTime > 0 && batch > 0 && classes > 0 && maxStates > 0 && maxStates < MaxExactInteger
        && blank >= 0 && blank < classes && classes < MaxExactInteger && maxTime < MaxExactInteger && batch < MaxExactInteger
        && (long)maxTime * batch * classes <= int.MaxValue && (long)batch * maxTime * maxStates <= int.MaxValue;

    internal static bool TryCtcForwardBackward(Launcher launch, IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer lengths,
        IGpuBuffer logAlpha, IGpuBuffer logBeta, IGpuBuffer logLikelihood, int maxTime, int batch, int classes, int maxStates,
        int blank)
    {
        if (!CtcFits(maxTime, batch, classes, maxStates, blank)) return false;
        LaunchChunked(launch, NumericRecurrenceKernelTemplate.CtcForwardBackward, batch,
            new[] { logProbs, logAlpha, targets, lengths, logBeta, logLikelihood }, new[] { maxTime, batch, classes, maxStates, blank, 0 });
        return true;
    }

    internal static bool TryCtcLabelPosteriors(Launcher launch, IGpuBuffer logAlpha, IGpuBuffer logBeta, IGpuBuffer targets,
        IGpuBuffer lengths, IGpuBuffer logLikelihood, IGpuBuffer gamma, int maxTime, int batch, int classes, int maxStates,
        int blank)
    {
        if (!CtcFits(maxTime, batch, classes, maxStates, blank)) return false;
        LaunchChunked(launch, NumericRecurrenceKernelTemplate.CtcPosteriors, Groups((long)maxTime * batch),
            new[] { logAlpha, gamma, logBeta, targets, lengths, logLikelihood }, new[] { maxTime, batch, classes, maxStates, blank, 0 });
        return true;
    }

    internal static bool TryLinearSumAssignment(Launcher launch, IGpuBuffer cost, IGpuBuffer assignment, IGpuBuffer workspace,
        int batch, int size)
    {
        if (batch <= 0 || size <= 0 || size > NumericRecurrenceKernelTemplate.AssignmentMaxSize
            || (long)batch * size * size > int.MaxValue || (long)batch * 8 * size > int.MaxValue)
            return false;
        LaunchChunked(launch, NumericRecurrenceKernelTemplate.LinearSumAssignment, batch,
            new[] { cost, assignment, workspace }, new[] { batch, size, 0 });
        return true;
    }
}
