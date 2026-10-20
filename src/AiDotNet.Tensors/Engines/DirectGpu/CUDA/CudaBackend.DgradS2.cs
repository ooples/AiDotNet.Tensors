// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly object _dgradS2ModuleLock = new();
    private bool _dgradS2ModuleTried;
    private bool _dgradS2ModuleReady;
    private IGpuBuffer? _dgradS2Partials;

    // Operator opt-out for the four-phase stride-2 input-gradient candidate alone.
    private static readonly bool s_convS2K3DgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_S2K3_DGRAD") != "0";

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
    internal unsafe bool TryConv2DBackwardInputS2K3(in TunedConv2DBackwardArgs a, int groups = 0, int splits = 0)
    {
        if (!DgradS2Admits(a) || !EnsureDgradS2Module()) return false;
        const int CoTile = Kernels.CudaConvDgradS2Kernels.CoTile;
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
        if (!_kernelCache.TryGetValue("conv2d_dgrad_s2k3_g" + groups, out var kernel)) return false;
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
        LaunchKernel3D(kernel, (uint)gx, (uint)gy, (uint)splits, (uint)(64 * groups), 1, 1, args, 0);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, a.Output, (int)outputs, splits);
        return true;
    }
}
