// Copyright (c) AiDotNet. All rights reserved.
using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// The stride-2 input gradients of <see cref="ConvDgradS2KernelTemplate"/> on HIP, Metal, OpenCL, Vulkan and WebGPU
/// (#1479-#1483): the host side the five share. It admits the shapes CUDA's kernels take (the same limits, so a shape
/// served on one backend is served on all), sizes the output-channel split to the device, launches into dX or into
/// partial slices and sums the slices in split order. Each backend supplies a <see cref="Launcher"/> for its own
/// dispatch.
/// </summary>
internal static class PortableConvDgradS2
{
    /// <summary>One launch: the kernel's buffers in binding order and its ints, over <paramref name="groups"/> groups of
    /// <see cref="ConvDgradS2KernelTemplate.GroupSize"/> (the reduce: <paramref name="groups"/> x 256 threads, one per
    /// output).</summary>
    internal delegate void Launcher(string kernelName, int groups, IGpuBuffer[] buffers, int[] ints);

    // Operator opt-outs, shared with CUDA's candidates of the same kernels.
    internal static readonly bool K3Enabled = Environment.GetEnvironmentVariable("AIDOTNET_CONV_S2K3_DGRAD") != "0";
    internal static readonly bool K1Enabled = Environment.GetEnvironmentVariable("AIDOTNET_CONV_S2K1_DGRAD") != "0";

    // Every API's guaranteed 1-D grid; past it the backend's other input-gradient routes serve the shape.
    private const int MaxGroups = 65535;

    private const int CoTile = ConvDgradS2KernelTemplate.CoTile;

    /// <summary>The 3 x 3 kernel's shapes: stride 2, padding 1, undilated, an input exactly twice the output, input
    /// channels a multiple of four, an output width that divides the 32-cell group (4, 8, 16 or 32), every index within
    /// int range.</summary>
    internal static bool AdmitsK3(in TunedConv2DBackwardArgs a)
        => !a.WeightGradient && a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 2 && a.StrideW == 2
           && a.PadH == 1 && a.PadW == 1 && a.DilationH == 1 && a.DilationW == 1
           && Common(a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels, a.OutHeight, a.OutWidth, 9)
           && ConvDgradS2KernelTemplate.CellTile % a.OutWidth == 0;

    /// <summary><see cref="AdmitsK3"/> on a tuned-slot shape (laid out as <see cref="TunedConv2DBackwardArgs.Shape"/>).</summary>
    internal static bool AdmitsK3Shape(TunedShape s)
        => s[5] == 3 && s[6] == 3 && s[7] == 2 && s[8] == 2 && s[9] == 1 && s[10] == 1 && s[11] == 1 && s[12] == 1
           && Common(s[0], s[1], s[2], s[3], s[4], s[13], s[14], 9) && ConvDgradS2KernelTemplate.CellTile % s[14] == 0;

    /// <summary>The 1 x 1 kernel's shapes: stride 2, padding 0, undilated, an input exactly twice the output, input
    /// channels and output width multiples of four, every index within int range.</summary>
    internal static bool AdmitsK1(in TunedConv2DBackwardArgs a)
        => !a.WeightGradient && a.KernelH == 1 && a.KernelW == 1 && a.StrideH == 2 && a.StrideW == 2
           && a.PadH == 0 && a.PadW == 0 && a.DilationH == 1 && a.DilationW == 1
           && Common(a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels, a.OutHeight, a.OutWidth, 1);

    /// <summary><see cref="AdmitsK1"/> on a tuned-slot shape.</summary>
    internal static bool AdmitsK1Shape(TunedShape s)
        => s[5] == 1 && s[6] == 1 && s[7] == 2 && s[8] == 2 && s[9] == 0 && s[10] == 0 && s[11] == 1 && s[12] == 1
           && Common(s[0], s[1], s[2], s[3], s[4], s[13], s[14], 1);

    /// <summary>The pairs the pair kernel takes: a 3 x 3 stride-2 padding-1 convolution and a 1 x 1 stride-2 padding-0
    /// projection of one input, with <see cref="AdmitsK3"/>'s limits over both output-channel counts.</summary>
    internal static bool AdmitsPair(int batch, int inChannels, int inHeight, int inWidth, int outChannels,
        int shortcutChannels, int outHeight, int outWidth)
        => shortcutChannels > 0
           && Common(batch, inChannels, inHeight, inWidth, Math.Max(outChannels, shortcutChannels), outHeight, outWidth, 9)
           && outChannels > 0 && ConvDgradS2KernelTemplate.CellTile % outWidth == 0;

    private static bool Common(int n, int c, int h, int w, int k, int oh, int ow, int taps)
        => n > 0 && c > 0 && k > 0 && oh > 0 && ow > 0 && c % 4 == 0 && ow % 4 == 0 && h == 2 * oh && w == 2 * ow
           && (long)n * c * h * w <= int.MaxValue && (long)n * k * oh * ow <= int.MaxValue
           && (long)k * c * taps <= int.MaxValue;

    /// <summary>
    /// The output-channel splits for <paramref name="tiles"/> groups: doubled while the grid stays within about one
    /// group per compute unit (<paramref name="groupTarget"/>) and every split keeps at least four eight-channel steps,
    /// as CUDA's grid choice.
    /// </summary>
    internal static int ChooseSplits(long tiles, int outChannels, int groupTarget)
    {
        int maxSplits = Math.Max(1, (outChannels + CoTile - 1) / CoTile / 4);
        int splits = 1;
        while (splits * 2 <= maxSplits && tiles * splits * 2 <= Math.Max(1, groupTarget)) splits *= 2;
        return splits;
    }

    /// <summary>dX of a 3 x 3 stride-2 convolution (a.First = dY, a.Second = its weights), split
    /// <paramref name="splits"/> ways (0: <see cref="ChooseSplits"/>). Overwrites dX. False: nothing launched.</summary>
    internal static bool TryK3(IDirectGpuBackend backend, Launcher launch, in TunedConv2DBackwardArgs a, int groupTarget,
        int splits = 0)
    {
        if (!AdmitsK3(a)) return false;
        int rowsPerTile = ConvDgradS2KernelTemplate.CellTile / a.OutWidth;
        long tiles = (long)((a.Batch * a.OutHeight + rowsPerTile - 1) / rowsPerTile) * ChannelTiles(a.InChannels);
        return Run(backend, launch, ConvDgradS2KernelTemplate.K3, a, tiles, groupTarget, splits, null, null, 0);
    }

    /// <summary>dX of a 1 x 1 stride-2 convolution, as <see cref="TryK3"/>.</summary>
    internal static bool TryK1(IDirectGpuBackend backend, Launcher launch, in TunedConv2DBackwardArgs a, int groupTarget,
        int splits = 0)
    {
        if (!AdmitsK1(a)) return false;
        long cells = (long)a.Batch * a.OutHeight * a.OutWidth;
        long tiles = (cells + ConvDgradS2KernelTemplate.CellTile - 1) / ConvDgradS2KernelTemplate.CellTile
                     * ChannelTiles(a.InChannels);
        return Run(backend, launch, ConvDgradS2KernelTemplate.K1, a, tiles, groupTarget, splits, null, null, 0);
    }

    /// <summary>dX of a 3 x 3 stride-2 convolution plus that of its 1 x 1 stride-2 projection reading the same input
    /// (<paramref name="dyShortcut"/> [N][K2][OH][OW], <paramref name="wShortcut"/> [K2][C]), in one launch: both
    /// reductions split the same way. Overwrites dX. False: nothing launched.</summary>
    internal static bool TryPair(IDirectGpuBackend backend, Launcher launch, in TunedConv2DBackwardArgs a,
        IGpuBuffer dyShortcut, IGpuBuffer wShortcut, int shortcutChannels, int groupTarget, int splits = 0)
    {
        if (!AdmitsK3(a) || !AdmitsPair(a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels, shortcutChannels,
                a.OutHeight, a.OutWidth))
            return false;
        int rowsPerTile = ConvDgradS2KernelTemplate.CellTile / a.OutWidth;
        long tiles = (long)((a.Batch * a.OutHeight + rowsPerTile - 1) / rowsPerTile) * ChannelTiles(a.InChannels);
        return Run(backend, launch, ConvDgradS2KernelTemplate.K3Pair, a, tiles, groupTarget, splits, dyShortcut,
            wShortcut, shortcutChannels);
    }

    private static long ChannelTiles(int channels)
        => (channels + ConvDgradS2KernelTemplate.ChannelTile - 1) / ConvDgradS2KernelTemplate.ChannelTile;

    // Rounds each reduction's split to whole steps (the pair splits both the same number of ways), launches into dX or
    // into partial slices, and sums the slices in split order.
    private static bool Run(IDirectGpuBackend backend, Launcher launch, string kernel, in TunedConv2DBackwardArgs a,
        long tiles, int groupTarget, int splits, IGpuBuffer? dy2, IGpuBuffer? ws, int k2)
    {
        if (backend is null) throw new ArgumentNullException(nameof(backend));
        if (launch is null) throw new ArgumentNullException(nameof(launch));
        int k = a.OutChannels;
        if (splits <= 0) splits = ChooseSplits(tiles, k + k2, groupTarget);
        // Both reductions split the same number of ways; the count is that of whichever needs more after rounding to
        // whole steps, so a slice of the other may be empty (it adds nothing).
        int coPerSplit = ((k + CoTile - 1) / CoTile + splits - 1) / splits * CoTile;
        int co2PerSplit = dy2 is null ? 0 : ((k2 + CoTile - 1) / CoTile + splits - 1) / splits * CoTile;
        splits = Math.Max((k + coPerSplit - 1) / coPerSplit, dy2 is null ? 0 : (k2 + co2PerSplit - 1) / co2PerSplit);
        long outputs = (long)a.Batch * a.InChannels * a.InHeight * a.InWidth;
        if (tiles * splits > MaxGroups || outputs * splits > int.MaxValue) return false;
        int n = a.Batch, c = a.InChannels, oh = a.OutHeight, ow = a.OutWidth;
        int[] ints = dy2 is null
            ? new[] { n, c, k, oh, ow, coPerSplit }
            : new[] { n, c, k, oh, ow, coPerSplit, k2, co2PerSplit };
        if (splits == 1)
        {
            launch(kernel, (int)tiles, Buffers(a.First, a.Output, a.Second, dy2, ws), ints);
            return true;
        }
        using var partials = backend.AllocateBuffer((int)(outputs * splits));
        launch(kernel, (int)(tiles * splits), Buffers(a.First, partials, a.Second, dy2, ws), ints);
        launch(ConvDgradS2KernelTemplate.Reduce, (int)((outputs + ConvDgradS2KernelTemplate.GroupSize - 1)
            / ConvDgradS2KernelTemplate.GroupSize), new[] { partials, a.Output }, new[] { (int)outputs, splits });
        return true;
    }

    private static IGpuBuffer[] Buffers(IGpuBuffer dy, IGpuBuffer dx, IGpuBuffer w, IGpuBuffer? dy2, IGpuBuffer? ws)
        => dy2 is null || ws is null ? new[] { dy, dx, w } : new[] { dy, dx, w, dy2, ws };
}
