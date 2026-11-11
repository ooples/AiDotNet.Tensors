// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The host side of <see cref="IGroupedGemmBackend"/> on the backends that run <see cref="GroupedGemmKernelTemplate"/>'s
/// kernels (OpenCL, Vulkan, Metal, WebGPU): it uploads the offsets as floats, sizes the tile grid and launches it in
/// pieces of at most 65535 groups (every API's guaranteed 1-D grid).
/// </summary>
internal static class PortableGroupedGemm
{
    /// <summary>
    /// Whether a failure while building the kernels leaves the backend usable, so the op reports itself unavailable and
    /// the engine falls back (logged). Driver and compiler errors surface as many exception types across the four
    /// backends; only runtime-fatal ones propagate.
    /// </summary>
    internal static bool IsBuildFailure(Exception ex) =>
        ex is not (OutOfMemoryException or StackOverflowException or AccessViolationException
            or System.Threading.ThreadAbortException);

    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="GroupedGemmKernelTemplate.GroupSize"/>.</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    private const int MaxGroupsPerLaunch = 65535;

    // The offsets are floats on the device (the template's buffers are float arrays): exact only below 2^24.
    private const int MaxExactRows = 1 << 24;

    /// <summary>Offsets non-decreasing, within [0, m], and m below 2^24 so the float-encoded offsets stay exact.</summary>
    private static bool Valid(int[] ends, int m)
    {
        if (ends.Length == 0 || m >= MaxExactRows) return false;
        int prev = 0;
        foreach (int e in ends)
        {
            if (e < prev) return false;
            prev = e;
        }
        return prev <= m;
    }

    private static float[] EndsAsFloats(int[] ends)
    {
        var f = new float[ends.Length];
        for (int i = 0; i < ends.Length; i++) f[i] = ends[i];
        return f;
    }

    private static int Tiles(int extent) => (extent + GroupedGemmKernelTemplate.Tile - 1) / GroupedGemmKernelTemplate.Tile;

    /// <summary>Launches total workgroups in chunks of at most 65535, passing each chunk's first tile as tileBase.</summary>
    private static void LaunchTiled(Launcher launch, string kernel, long total, IGpuBuffer[] buffers, int[] ints)
    {
        int baseSlot = ints.Length - 1;   // tileBase is the last int of both kernels
        for (long start = 0; start < total; start += MaxGroupsPerLaunch)
        {
            int count = (int)Math.Min(MaxGroupsPerLaunch, total - start);
            var slice = (int[])ints.Clone();
            slice[baseSlot] = (int)start;
            launch(kernel, count, buffers, slice);
        }
    }

    /// <summary>Runs the forward kernel over every group in one (chunked) launch; false if the shape is refused.</summary>
    internal static bool TryForward(IDirectGpuBackend backend, Launcher launch, IGpuBuffer a, IGpuBuffer b, IGpuBuffer y,
        int[] ends, int m, int n, int k, bool transposeB, IGpuBuffer? bias, bool biasPerGroup)
    {
        int g = ends.Length;
        if (m <= 0 || n <= 0 || k <= 0 || !Valid(ends, m)
            || (long)m * Tiles(n) > int.MaxValue / 2 || (long)m * k > int.MaxValue || (long)m * n > int.MaxValue
            || (long)g * k * n > int.MaxValue)
            return false;
        long tiles = (long)Tiles(m) * Tiles(n);
        using var endsBuffer = backend.AllocateBuffer(EndsAsFloats(ends));
        int biasMode = bias is null ? 0 : biasPerGroup ? 2 : 1;
        LaunchTiled(launch, GroupedGemmKernelTemplate.Forward, tiles,
            new[] { a, y, b, endsBuffer, bias ?? endsBuffer },
            new[] { m, n, k, g, Tiles(n), transposeB ? 1 : 0, biasMode, 0 });
        return true;
    }

    /// <summary>Runs the per-group weight-gradient kernel (w_g = a_g^T c_g, zero for empty groups); false if refused.</summary>
    internal static bool TryWeightGrad(IDirectGpuBackend backend, Launcher launch, IGpuBuffer a, IGpuBuffer c, IGpuBuffer w,
        int[] ends, int m, int k, int n)
    {
        int g = ends.Length;
        if (k <= 0 || n <= 0 || m < 0 || !Valid(ends, m) || (long)g * k * n > int.MaxValue
            || (long)m * k > int.MaxValue || (long)m * n > int.MaxValue)
            return false;
        long tiles = (long)g * Tiles(k) * Tiles(n);
        using var endsBuffer = backend.AllocateBuffer(EndsAsFloats(ends));
        LaunchTiled(launch, GroupedGemmKernelTemplate.WeightGrad, tiles,
            new[] { a, w, c, endsBuffer },
            new[] { m, k, n, g, Tiles(k), Tiles(n), 0 });
        return true;
    }
}
