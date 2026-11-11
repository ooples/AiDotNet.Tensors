// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The capability every DirectGpu backend implements for <see cref="MoeKernelTemplate"/>: whether its module built, and
/// one launch of a kernel over <c>groups</c> 256-thread groups with the kernel's buffers (binding order) and ints.
/// </summary>
internal interface IMoeKernelBackend
{
    bool MoeKernelsReady { get; }

    void LaunchMoeKernel(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);
}

/// <summary>Shapes of one fused MoE call: N tokens of width D, G experts, top-K, expert hidden width F, B sequences.</summary>
internal readonly struct MoeShape
{
    internal MoeShape(int n, int d, int g, int k, int f, int b)
    {
        N = n; D = d; G = g; K = k; F = f; B = b;
    }

    internal int N { get; }
    internal int D { get; }
    internal int G { get; }
    internal int K { get; }
    internal int F { get; }
    internal int B { get; }
    internal int NK => N * K;

    /// <summary>The kernels' limits: top-k in registers, float-exact indices, sequences that split the tokens evenly.</summary>
    internal bool Admitted =>
        N > 0 && D > 0 && G > 0 && F > 0 && K > 0 && K <= MoeKernelTemplate.MaxTopK && K <= G && B > 0 && N % B == 0
        && (long)N * K < (1 << 24) && (long)N * K * Math.Max(2L * F, D) <= int.MaxValue && (long)N * G <= int.MaxValue
        && (long)G * D * F <= int.MaxValue;
}

/// <summary>
/// The launch sequence of the fused MoE layer over device buffers, shared by every backend (see
/// <see cref="MoeKernelTemplate"/> for what each kernel computes). Nothing here reads device memory on the host.
/// </summary>
internal static class PortableMoe
{
    private const int MaxGroupsPerLaunch = 65535;
    private const int MaxElementsPerLaunch = MaxGroupsPerLaunch * MoeKernelTemplate.GroupSize;
    internal const int MaxSortSlotsPerSlice = 65536;
    private const int T = MoeKernelTemplate.Tile;

    private static int Tiles(int extent) => (extent + T - 1) / T;

    private static int Groups(long threads) => (int)((threads + MoeKernelTemplate.GroupSize - 1) / MoeKernelTemplate.GroupSize);

    private static void Tiled(IMoeKernelBackend be, string kernel, long total, IGpuBuffer[] buffers, int[] ints)
    {
        // tileBase is ints[5] in both GEMM int lists.
        for (long first = 0; first < total; first += MaxGroupsPerLaunch)
        {
            var slice = (int[])ints.Clone();
            slice[5] = (int)first;
            be.LaunchMoeKernel(kernel, (int)Math.Min(MaxGroupsPerLaunch, total - first), buffers, slice);
        }
    }

    private static void Elementwise(IMoeKernelBackend be, string kernel, int total, IGpuBuffer[] buffers, int[] ints)
    {
        // baseOffset is the last int of Route, Combine, and DScores.
        int baseSlot = ints.Length - 1;
        for (long baseOffset = 0; baseOffset < total; baseOffset += MaxElementsPerLaunch)
        {
            int chunk = (int)Math.Min(MaxElementsPerLaunch, total - baseOffset);
            var slice = (int[])ints.Clone();
            slice[baseSlot] = (int)baseOffset;
            be.LaunchMoeKernel(kernel, Groups(chunk), buffers, slice);
        }
    }

    private static void ChunkedGroups(IMoeKernelBackend be, string kernel, int totalGroups, IGpuBuffer[] buffers, int[] ints)
    {
        int baseSlot = ints.Length - 1;
        for (long first = 0; first < totalGroups; first += MaxGroupsPerLaunch)
        {
            int count = (int)Math.Min(MaxGroupsPerLaunch, totalGroups - first);
            var slice = (int[])ints.Clone();
            slice[baseSlot] = (int)first;
            be.LaunchMoeKernel(kernel, count, buffers, slice);
        }
    }

    internal static void LaunchSort(IMoeKernelBackend be, in MoeShape s, IGpuBuffer ids, IGpuBuffer perm,
        IGpuBuffer pos, IGpuBuffer ends, IGpuBuffer loads)
    {
        int sps = s.NK / s.B;
        var buffers = new[] { ids, perm, pos, ends, loads };
        if (s.NK <= MaxSortSlotsPerSlice)
        {
            ChunkedGroups(be, MoeKernelTemplate.Sort, s.G, buffers,
                new[] { s.NK, s.G, s.B, sps, 0, s.NK, 3, 0 });
            return;
        }

        int sliceSlots = sps <= MaxSortSlotsPerSlice
            ? (MaxSortSlotsPerSlice / sps) * sps
            : MaxSortSlotsPerSlice;

        for (long slotStart = 0; slotStart < s.NK; slotStart += sliceSlots)
        {
            int slotEnd = (int)Math.Min(s.NK, slotStart + sliceSlots);
            int phase = slotStart == 0 ? 0 : 1;
            ChunkedGroups(be, MoeKernelTemplate.Sort, s.G, buffers,
                new[] { s.NK, s.G, s.B, sps, (int)slotStart, slotEnd, phase, 0 });
        }

        for (long slotStart = 0; slotStart < s.NK; slotStart += sliceSlots)
        {
            int slotEnd = (int)Math.Min(s.NK, slotStart + sliceSlots);
            ChunkedGroups(be, MoeKernelTemplate.Sort, s.G, buffers,
                new[] { s.NK, s.G, s.B, sps, (int)slotStart, slotEnd, 2, 0 });
        }
    }

    private static int[] Gemm(int m, int n, int kd, in MoeShape s) =>
        new[] { m, n, kd, s.G, Tiles(n), 0, s.K, s.F };

    private static int[] Wgrad(int p, int nn, in MoeShape s) =>
        new[] { p, nn, s.G, Tiles(p), Tiles(nn), 0, s.K, s.F };

    internal static void LaunchRoute(IMoeKernelBackend be, int n, int g, int k, IGpuBuffer scores, IGpuBuffer? bias,
        IGpuBuffer ids, IGpuBuffer gates, IGpuBuffer ssum) =>
        Elementwise(be, MoeKernelTemplate.Route, n, new[] { scores, bias ?? scores, ids, gates, ssum },
            new[] { n, g, k, bias is null ? 0 : 1, 0 });

    internal static void LaunchDScores(IMoeKernelBackend be, int n, int d, int g, int k, IGpuBuffer o, IGpuBuffer dy,
        IGpuBuffer pos, IGpuBuffer gates, IGpuBuffer ssum, IGpuBuffer ids, IGpuBuffer ds) =>
        Elementwise(be, MoeKernelTemplate.DScores, n, new[] { o, dy, pos, gates, ssum, ids, ds },
            new[] { n, d, g, k, 0 });

    /// <summary>
    /// The forward: route, sort, up-projection (h13 [NK, 2F], kept for the backward), down-projection (o [NK, D], kept),
    /// and the gated combine into y [N, D]. Writes ids/gates [NK], ssum [N], perm/pos [NK], ends [G], loads [B, G].
    /// </summary>
    internal static void Forward(IMoeKernelBackend be, in MoeShape s, IGpuBuffer x, IGpuBuffer scores, IGpuBuffer? bias,
        IGpuBuffer w1, IGpuBuffer w3, IGpuBuffer w2, IGpuBuffer ids, IGpuBuffer gates, IGpuBuffer ssum, IGpuBuffer perm,
        IGpuBuffer pos, IGpuBuffer ends, IGpuBuffer loads, IGpuBuffer h13, IGpuBuffer o, IGpuBuffer y)
    {
        LaunchRoute(be, s.N, s.G, s.K, scores, bias, ids, gates, ssum);
        LaunchSort(be, s, ids, perm, pos, ends, loads);
        Tiled(be, MoeKernelTemplate.Up, (long)Tiles(s.NK) * Tiles(2 * s.F), new[] { x, perm, w1, w3, ends, h13 },
            Gemm(s.NK, 2 * s.F, s.D, s));
        Tiled(be, MoeKernelTemplate.Down, (long)Tiles(s.NK) * Tiles(s.D), new[] { h13, w2, ends, o },
            Gemm(s.NK, s.D, s.F, s));
        Elementwise(be, MoeKernelTemplate.Combine, s.N * s.D, new[] { o, pos, gates, y },
            new[] { s.N, s.D, s.K, 1, 0 });
    }

    /// <summary>
    /// The backward from dy [N, D]: the scores' gradient ds [N, G]; dh13 [NK, 2F] and dxs [NK, D] as scratch; dx [N, D];
    /// the weight gradients dw1, dw3 [G, D, F] and dw2 [G, F, D] (every element written, empty experts zero).
    /// </summary>
    internal static void Backward(IMoeKernelBackend be, in MoeShape s, IGpuBuffer dy, IGpuBuffer x, IGpuBuffer w1,
        IGpuBuffer w3, IGpuBuffer w2, IGpuBuffer ids, IGpuBuffer gates, IGpuBuffer ssum, IGpuBuffer perm, IGpuBuffer pos,
        IGpuBuffer ends, IGpuBuffer h13, IGpuBuffer o, IGpuBuffer ds, IGpuBuffer dh13, IGpuBuffer dxs, IGpuBuffer dx,
        IGpuBuffer dw1, IGpuBuffer dw3, IGpuBuffer dw2)
    {
        LaunchDScores(be, s.N, s.D, s.G, s.K, o, dy, pos, gates, ssum, ids, ds);
        Tiled(be, MoeKernelTemplate.DAct, (long)Tiles(s.NK) * Tiles(s.F), new[] { dy, perm, gates, w2, ends, h13, dh13 },
            Gemm(s.NK, s.F, s.D, s));
        Tiled(be, MoeKernelTemplate.DX, (long)Tiles(s.NK) * Tiles(s.D), new[] { dh13, w1, w3, ends, dxs },
            Gemm(s.NK, s.D, 2 * s.F, s));
        Elementwise(be, MoeKernelTemplate.Combine, s.N * s.D, new[] { dxs, pos, gates, dx },
            new[] { s.N, s.D, s.K, 0, 0 });
        Tiled(be, MoeKernelTemplate.WUp, (long)s.G * Tiles(s.D) * Tiles(2 * s.F), new[] { x, perm, dh13, ends, dw1, dw3 },
            Wgrad(s.D, 2 * s.F, s));
        Tiled(be, MoeKernelTemplate.WDown, (long)s.G * Tiles(s.F) * Tiles(s.D), new[] { h13, perm, gates, dy, ends, dw2 },
            Wgrad(s.F, s.D, s));
    }
}
