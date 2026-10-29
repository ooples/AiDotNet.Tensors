// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend : IConvShortcutPairBackend
{
    private readonly object _dgradS2ModuleLock = new();
    private bool _dgradS2ModuleTried;
    private bool _dgradS2ModuleReady;
    private IGpuBuffer? _dgradS2Partials;

    // Operator opt-out for the four-phase stride-2 input-gradient candidate alone.
    private static readonly bool s_convS2K3DgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_S2K3_DGRAD") != "0";

    // Operator opt-out for the 1x1 stride-2 input-gradient candidate alone.
    private static readonly bool s_convS2K1DgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_S2K1_DGRAD") != "0";

    /// <summary>Compiles the four-phase stride-2 input-gradient kernel (<see cref="Kernels.CudaConvDgradS2Kernels"/>) once.</summary>
    internal bool EnsureDgradS2Module()
    {
        lock (_dgradS2ModuleLock)
        {
            if (_dgradS2ModuleTried) return _dgradS2ModuleReady;
            _dgradS2ModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaConvDgradS2Kernels.GetSource(), "conv_dgrad_s2_kernels",
                    Kernels.CudaConvDgradS2Kernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvDgradS2Kernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _dgradS2ModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Stride-2 input-gradient kernel unavailable: {ex.GetType().Name}: {ex.Message}");
            }
            return _dgradS2ModuleReady;
        }
    }

    /// <summary>True for the input gradients conv2d_dgrad_s2k3 computes: a 3 x 3, stride-2, padding-1, undilated
    /// convolution whose input is exactly twice its output, input channels a multiple of four (16-byte weight rows), an output width that divides the 32-cell block (4, 8, 16 or
    /// 32: whole rows per block, 16-byte rows), and every index within int range.</summary>
    internal static bool DgradS2Admits(in TunedConv2DBackwardArgs a)
        => !a.WeightGradient && a.InChannels % 4 == 0 && a.KernelH == 3 && a.KernelW == 3 && a.StrideH == 2 && a.StrideW == 2
           && a.PadH == 1 && a.PadW == 1 && a.DilationH == 1 && a.DilationW == 1
           && a.InHeight == 2 * a.OutHeight && a.InWidth == 2 * a.OutWidth
           && a.OutWidth % 4 == 0 && Kernels.CudaConvDgradS2Kernels.CellTile % a.OutWidth == 0
           && a.Batch > 0 && a.InChannels > 0 && a.OutChannels > 0 && a.OutHeight > 0
           && (long)a.Batch * a.InChannels * a.InHeight * a.InWidth <= int.MaxValue
           && (long)a.Batch * a.OutChannels * a.OutHeight * a.OutWidth <= int.MaxValue
           && (long)a.OutChannels * a.InChannels * 9 <= int.MaxValue;

    /// <summary><see cref="DgradS2Admits"/> on a tuned-slot shape (laid out as <see cref="TunedConv2DBackwardArgs.Shape"/>).</summary>
    internal static bool DgradS2AdmitsShape(AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedShape s)
        => s[1] % 4 == 0 && s[5] == 3 && s[6] == 3 && s[7] == 2 && s[8] == 2 && s[9] == 1 && s[10] == 1 && s[11] == 1
           && s[12] == 1 && s[2] == 2 * s[13] && s[3] == 2 * s[14] && s[14] % 4 == 0
           && Kernels.CudaConvDgradS2Kernels.CellTile % s[14] == 0 && s[0] > 0 && s[1] > 0 && s[4] > 0 && s[13] > 0;

    /// <summary>
    /// The (groups, splits) conv2d_dgrad_s2k3 runs a grid of <paramref name="blocks"/> (cell x channel) tiles with:
    /// about four 64-thread units per SM, first as in-block output-channel groups (1, 2 or 4: no partials), then as
    /// split slices (at least four eight-channel steps each). Measured per call on an RTX 3080 at batch 8 against every
    /// (groups, splits) pair it picks the fastest: 64->128 at 32x32 (128 tiles) g2 s1, 128->256 at 16x16 (64) g4 s1,
    /// 256->512 at 8x8 (32) g4 s2.
    /// </summary>
    internal static void ChooseDgradS2Grid(long blocks, int outChannels, int sms, out int groups, out int splits)
    {
        long units = 4L * sms;
        groups = 1;
        while (groups < 4 && blocks * groups * 2 <= units) groups *= 2;
        int maxSplits = Math.Max(1, (outChannels + Kernels.CudaConvDgradS2Kernels.CoTile - 1) / Kernels.CudaConvDgradS2Kernels.CoTile / 4);
        splits = 1;
        while (splits * 2 <= maxSplits && blocks * groups * splits * 2 <= units) splits *= 2;
    }

    /// <summary>
    /// dX = the input gradient of a stride-2 3 x 3 convolution (a.First = dY, a.Second = the forward weights) through
    /// conv2d_dgrad_s2k3_g<paramref name="groups"/>, with the output-channel reduction split <paramref name="splits"/>
    /// ways (0 for both: <see cref="ChooseDgradS2Grid"/>). Overwrites dX. False: nothing launched.
    /// </summary>
    internal bool TryConv2DBackwardInputS2K3(in TunedConv2DBackwardArgs a, int groups = 0, int splits = 0)
    {
        if (!DgradS2Admits(a) || !EnsureDgradS2Module()) return false;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, oh = a.OutHeight, ow = a.OutWidth;
        int rowsPerBlock = Kernels.CudaConvDgradS2Kernels.CellTile / ow;
        int gx = (n * oh + rowsPerBlock - 1) / rowsPerBlock;
        int gy = (c + Kernels.CudaConvDgradS2Kernels.ChannelTile - 1) / Kernels.CudaConvDgradS2Kernels.ChannelTile;
        long blocks = (long)gx * gy;
        if (groups <= 0 || splits <= 0)
        {
            ChooseDgradS2Grid(blocks, k, Math.Max(1, _multiProcessorCount), out int g, out int s);
            if (groups <= 0) groups = g;
            if (splits <= 0) splits = s;
        }
        return LaunchDgradS2(a, Kernels.CudaConvDgradS2Kernels.KernelName(groups), gx, gy, groups, splits);
    }

    /// <summary>True for the input gradients conv2d_dgrad_s2k1 computes: a 1 x 1, stride-2, padding-0, undilated
    /// convolution whose input is exactly twice its output, input channels a multiple of four (16-byte weight rows), an
    /// output width that is a multiple of four (a thread's four cells share a row), and every index within int range.</summary>
    internal static bool DgradS2K1Admits(in TunedConv2DBackwardArgs a)
        => !a.WeightGradient && a.InChannels % 4 == 0 && a.KernelH == 1 && a.KernelW == 1 && a.StrideH == 2 && a.StrideW == 2
           && a.PadH == 0 && a.PadW == 0 && a.DilationH == 1 && a.DilationW == 1
           && a.InHeight == 2 * a.OutHeight && a.InWidth == 2 * a.OutWidth && a.OutWidth % 4 == 0
           && a.Batch > 0 && a.InChannels > 0 && a.OutChannels > 0 && a.OutHeight > 0
           && (long)a.Batch * a.InChannels * a.InHeight * a.InWidth <= int.MaxValue
           && (long)a.Batch * a.OutChannels * a.OutHeight * a.OutWidth <= int.MaxValue
           && (long)a.OutChannels * a.InChannels <= int.MaxValue;

    /// <summary><see cref="DgradS2K1Admits"/> on a tuned-slot shape (laid out as <see cref="TunedConv2DBackwardArgs.Shape"/>).</summary>
    internal static bool DgradS2K1AdmitsShape(AiDotNet.Tensors.Helpers.Autotune.TunedKernels.TunedShape s)
        => s[1] % 4 == 0 && s[5] == 1 && s[6] == 1 && s[7] == 2 && s[8] == 2 && s[9] == 0 && s[10] == 0 && s[11] == 1
           && s[12] == 1 && s[2] == 2 * s[13] && s[3] == 2 * s[14] && s[14] % 4 == 0 && s[0] > 0 && s[1] > 0 && s[4] > 0
           && s[13] > 0;

    /// <summary>
    /// dX = the input gradient of a 1 x 1 stride-2 convolution (a.First = dY, a.Second = the forward weights) through
    /// conv2d_dgrad_s2k1_g<paramref name="groups"/>, with the output-channel reduction split <paramref name="splits"/>
    /// ways (0 for both: <see cref="ChooseDgradS2Grid"/>). Writes every element of dX, the zero phases included.
    /// False: nothing launched.
    /// </summary>
    internal bool TryConv2DBackwardInputS2K1(in TunedConv2DBackwardArgs a, int groups = 0, int splits = 0)
    {
        if (!DgradS2K1Admits(a) || !EnsureDgradS2Module()) return false;
        long cells = (long)a.Batch * a.OutHeight * a.OutWidth;
        int gx = (int)((cells + Kernels.CudaConvDgradS2Kernels.CellTile - 1) / Kernels.CudaConvDgradS2Kernels.CellTile);
        int gy = (a.InChannels + Kernels.CudaConvDgradS2Kernels.ChannelTile - 1) / Kernels.CudaConvDgradS2Kernels.ChannelTile;
        if (groups <= 0 || splits <= 0)
        {
            ChooseDgradS2Grid((long)gx * gy, a.OutChannels, Math.Max(1, _multiProcessorCount), out int g, out int s);
            if (groups <= 0) groups = g;
            if (splits <= 0) splits = s;
        }
        return LaunchDgradS2(a, Kernels.CudaConvDgradS2Kernels.KernelNameK1(groups), gx, gy, groups, splits);
    }

    /// <summary>True for the shortcut pairs conv2d_dgrad_s2k3sc computes, from shapes alone: a 3 x 3 stride-2 padding-1
    /// convolution (<paramref name="outChannels"/>) and a 1 x 1 stride-2 padding-0 projection
    /// (<paramref name="shortcutChannels"/>) of one input exactly twice their output, with <see cref="DgradS2Admits"/>'s
    /// channel, width and range limits.</summary>
    internal static bool DgradS2PairAdmitsShape(int batch, int inChannels, int inHeight, int inWidth, int outChannels,
        int shortcutChannels, int outHeight, int outWidth)
        => batch > 0 && inChannels > 0 && outChannels > 0 && shortcutChannels > 0 && outHeight > 0
           && inChannels % 4 == 0 && inHeight == 2 * outHeight && inWidth == 2 * outWidth
           && outWidth % 4 == 0 && Kernels.CudaConvDgradS2Kernels.CellTile % outWidth == 0
           && (long)batch * inChannels * inHeight * inWidth <= int.MaxValue
           && (long)batch * Math.Max(outChannels, shortcutChannels) * outHeight * outWidth <= int.MaxValue
           && (long)Math.Max(outChannels, shortcutChannels) * inChannels * 9 <= int.MaxValue;

    /// <summary>
    /// dX = the input gradient of a 3 x 3 stride-2 convolution (<paramref name="a"/>: dY, its weights) plus that of the
    /// projection shortcut reading the same input (a 1 x 1, stride-2, padding-0 convolution: <paramref name="dyShortcut"/>
    /// [N][<paramref name="shortcutChannels"/>][OH][OW], <paramref name="wShortcut"/> [K2][C]) in one launch,
    /// conv2d_dgrad_s2k3sc: the shortcut's gradient is the centre tap's phase. Both reductions split the same
    /// <paramref name="splits"/> ways (0: <see cref="ChooseDgradS2Grid"/> over K + K2). Overwrites dX. False: nothing
    /// launched.
    /// </summary>
    internal unsafe bool TryConv2DBackwardInputS2K3Pair(in TunedConv2DBackwardArgs a, IGpuBuffer dyShortcut, IGpuBuffer wShortcut,
        int shortcutChannels, int groups = 0, int splits = 0)
    {
        if (!DgradS2Admits(a) || shortcutChannels <= 0 || !EnsureDgradS2Module()) return false;
        const int CoTile = Kernels.CudaConvDgradS2Kernels.CoTile;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, k2 = shortcutChannels, oh = a.OutHeight, ow = a.OutWidth;
        if ((long)n * k2 * oh * ow > int.MaxValue || (long)k2 * c > int.MaxValue) return false;
        int rowsPerBlock = Kernels.CudaConvDgradS2Kernels.CellTile / ow;
        int gx = (n * oh + rowsPerBlock - 1) / rowsPerBlock;
        int gy = (c + Kernels.CudaConvDgradS2Kernels.ChannelTile - 1) / Kernels.CudaConvDgradS2Kernels.ChannelTile;
        if (groups <= 0 || splits <= 0)
        {
            ChooseDgradS2Grid((long)gx * gy, k + k2, Math.Max(1, _multiProcessorCount), out int g, out int s);
            if (groups <= 0) groups = g;
            if (splits <= 0) splits = s;
        }
        if (!_kernelCache.TryGetValue(Kernels.CudaConvDgradS2Kernels.KernelNamePair(groups), out var kernel)) return false;
        int coPerSplit = ((k + CoTile - 1) / CoTile + splits - 1) / splits * CoTile;
        int co2PerSplit = ((k2 + CoTile - 1) / CoTile + splits - 1) / splits * CoTile;
        splits = Math.Max((k + coPerSplit - 1) / coPerSplit, (k2 + co2PerSplit - 1) / co2PerSplit);
        long outputs = (long)n * c * a.InHeight * a.InWidth;
        IntPtr reduce = IntPtr.Zero;
        if (splits > 1 && (!_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce) || outputs * splits > int.MaxValue))
            return false;
        using var context = PushContext();
        IGpuBuffer destination = a.Output;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _dgradS2Partials, splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }
        IntPtr dyPtr = a.First.Handle, wPtr = a.Second.Handle, dxPtr = destination.Handle, dy2Ptr = dyShortcut.Handle,
            ws2Ptr = wShortcut.Handle;
        int cps = coPerSplit, c2ps = co2PerSplit;
        void** args = stackalloc void*[13];
        args[0] = &dyPtr; args[1] = &wPtr; args[2] = &dxPtr; args[3] = &n; args[4] = &c; args[5] = &k;
        args[6] = &oh; args[7] = &ow; args[8] = &cps; args[9] = &dy2Ptr; args[10] = &ws2Ptr; args[11] = &k2; args[12] = &c2ps;
        LaunchKernel3D(kernel, (uint)gx, (uint)gy, (uint)splits,
            (uint)(Kernels.CudaConvDgradS2Kernels.GroupThreads * groups), 1, 1, args, 0);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, a.Output, (int)outputs, splits);
        return true;
    }

    bool IConvShortcutPairBackend.SupportsConv2DBackwardInputShortcutPair(int batch, int inChannels, int inHeight,
        int inWidth, int outChannels, int shortcutChannels, int outHeight, int outWidth)
        => DgradS2PairAdmitsShape(batch, inChannels, inHeight, inWidth, outChannels, shortcutChannels, outHeight, outWidth)
           && EnsureDgradS2Module();

    bool IConvShortcutPairBackend.TryConv2DBackwardInputShortcutPair(IGpuBuffer gradOutput, IGpuBuffer kernel,
        IGpuBuffer gradShortcut, IGpuBuffer kernelShortcut, IGpuBuffer gradInput, int batch, int inChannels, int inHeight,
        int inWidth, int outChannels, int shortcutChannels, int outHeight, int outWidth)
        => TryConv2DBackwardInputS2K3Pair(new TunedConv2DBackwardArgs(gradOutput, kernel, gradInput, false, batch, inChannels,
            inHeight, inWidth, outChannels, outHeight, outWidth, 3, 3, 2, 2, 1, 1, 1, 1), gradShortcut, kernelShortcut,
            shortcutChannels);

    // The shared tail of both stride-2 input gradients: round the output-channel split to whole steps, launch into dX or
    // into partial slices, and sum the slices in split order (conv2d_wgrad_splitk_reduce).
    private unsafe bool LaunchDgradS2(in TunedConv2DBackwardArgs a, string kernelName, int gx, int gy, int groups, int splits)
    {
        if (!_kernelCache.TryGetValue(kernelName, out var kernel)) return false;
        const int CoTile = Kernels.CudaConvDgradS2Kernels.CoTile;
        int n = a.Batch, c = a.InChannels, k = a.OutChannels, oh = a.OutHeight, ow = a.OutWidth;
        int coSteps = (k + CoTile - 1) / CoTile;
        int coPerSplit = (coSteps + splits - 1) / splits * CoTile;
        splits = (k + coPerSplit - 1) / coPerSplit;
        long outputs = (long)n * c * a.InHeight * a.InWidth;
        IntPtr reduce = IntPtr.Zero;
        if (splits > 1 && (!_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce) || outputs * splits > int.MaxValue))
            return false;
        using var context = PushContext();
        IGpuBuffer destination = a.Output;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _dgradS2Partials, splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }
        IntPtr dyPtr = a.First.Handle, wPtr = a.Second.Handle, dxPtr = destination.Handle;
        int cps = coPerSplit;
        void** args = stackalloc void*[9];
        args[0] = &dyPtr; args[1] = &wPtr; args[2] = &dxPtr; args[3] = &n; args[4] = &c; args[5] = &k;
        args[6] = &oh; args[7] = &ow; args[8] = &cps;
        LaunchKernel3D(kernel, (uint)gx, (uint)gy, (uint)splits,
            (uint)(Kernels.CudaConvDgradS2Kernels.GroupThreads * groups), 1, 1, args, 0);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, a.Output, (int)outputs, splits);
        return true;
    }
}
