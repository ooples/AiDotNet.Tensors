// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly object _fpropR8ModuleLock = new();
    private bool _fpropR8ModuleTried;
    private bool _fpropR8ModuleReady;
    private IGpuBuffer? _fpropR8Partials;

    // Operator opt-out for the r8 forward candidates alone.
    private static readonly bool s_convR8FpropEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_R8_FPROP") != "0";

    /// <summary>Compiles the 8x8-micro-tile forward kernels (<see cref="Kernels.CudaConvFpropR8Kernels"/>) once.</summary>
    internal bool EnsureFpropR8Module()
    {
        lock (_fpropR8ModuleLock)
        {
            if (_fpropR8ModuleTried) return _fpropR8ModuleReady;
            _fpropR8ModuleTried = true;
            try
            {
                using var _ = PushContext();
                if (CudaNativeBindings.cuCtxGetDevice(out int device) != CudaResult.Success) return false;
                CompileKernelModule(device, Kernels.CudaConvFpropR8Kernels.GetSource(), "conv_fprop_r8_kernels",
                    Kernels.CudaConvFpropR8Kernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvFpropR8Kernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name)) return false;
                _fpropR8ModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Forward r8 convolution kernels unavailable: {ex.GetType().Name}: {ex.Message}");
            }
            return _fpropR8ModuleReady;
        }
    }

    /// <summary>True for the shapes the r8 forward computes: input channels a multiple of four (16-byte weight rows) and
    /// every index within int range.</summary>
    internal static bool FpropR8Admits(in CudaConv2DArgs a)
        => a.InChannels % 4 == 0 && a.Batch > 0 && a.OutChannels > 0 && a.OutHeight > 0 && a.OutWidth > 0
           && (long)a.Batch * a.InChannels * a.InHeight * a.InWidth <= int.MaxValue
           && (long)a.Batch * a.OutChannels * a.OutHeight * a.OutWidth <= int.MaxValue
           && (long)a.OutChannels * a.InChannels * a.KernelH * a.KernelW <= int.MaxValue;

    /// <summary>
    /// y = conv(x, w) through conv2d_fprop_r8_{tileM}x{tileN} with a split-K count (0: enough splits for about two blocks
    /// per SM). Overwrites y. False: nothing launched.
    /// </summary>
    internal unsafe bool TryConv2DForwardR8(in CudaConv2DArgs a, int tileM, int tileN, int splits = 0)
    {
        if (!FpropR8Admits(a) || !EnsureFpropR8Module()
            || !_kernelCache.TryGetValue(Kernels.CudaConvFpropR8Kernels.KernelName(tileM, tileN), out var gemm))
            return false;
        const int Bk = Kernels.CudaConvFpropR8Kernels.KTile;
        long kTotal = (long)a.InChannels * a.KernelH * a.KernelW;
        long columns = (long)a.Batch * a.OutHeight * a.OutWidth;
        long outputs = columns * a.OutChannels;
        int threads = (tileM / 8) * (tileN / 8);
        int tilesN = (int)((columns + tileN - 1) / tileN), tilesM = (a.OutChannels + tileM - 1) / tileM;
        long tiles = (long)tilesN * tilesM;
        int sms = Math.Max(1, _multiProcessorCount);
        if (splits <= 0)
            splits = tiles >= 2L * sms ? 1 : (int)Math.Min(Math.Max(1, kTotal / (8L * Bk)), (2L * sms + tiles - 1) / tiles);
        int kPerSplit = (int)(((kTotal + splits - 1) / splits + Bk - 1) / Bk * Bk);
        splits = (int)((kTotal + kPerSplit - 1) / kPerSplit);
        IntPtr reduce = IntPtr.Zero;
        if (splits > 1 && (!_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce) || outputs * splits > int.MaxValue))
            return false;
        using var context = PushContext();
        IGpuBuffer destination = a.Output;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _fpropR8Partials, splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }
        IntPtr x = a.Input.Handle, wPtr = a.Kernel.Handle, o = destination.Handle;
        int batch = a.Batch, inC = a.InChannels, h = a.InHeight, w = a.InWidth, outC = a.OutChannels, outH = a.OutHeight,
            outW = a.OutWidth, kh = a.KernelH, kw = a.KernelW, sh = a.StrideH, sw = a.StrideW, ph = a.PadH, pw = a.PadW,
            dh = a.DilationH, dw = a.DilationW, kps = kPerSplit;
        void** args = stackalloc void*[19];
        args[0] = &x; args[1] = &wPtr; args[2] = &o;
        args[3] = &batch; args[4] = &inC; args[5] = &h; args[6] = &w;
        args[7] = &outC; args[8] = &outH; args[9] = &outW;
        args[10] = &kh; args[11] = &kw; args[12] = &sh; args[13] = &sw;
        args[14] = &ph; args[15] = &pw; args[16] = &dh; args[17] = &dw; args[18] = &kps;
        LaunchKernel3D(gemm, (uint)tilesN, (uint)tilesM, (uint)splits, (uint)threads, 1, 1, args, 0);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, a.Output, (int)outputs, splits);
        return true;
    }
}
