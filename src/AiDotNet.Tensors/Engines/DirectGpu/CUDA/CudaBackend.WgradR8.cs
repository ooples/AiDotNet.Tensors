// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    private readonly object _wgradR8ModuleLock = new();
    private bool _wgradR8ModuleTried;
    private bool _wgradR8ModuleReady;

    /// <summary>Compiles the 8x8-micro-tile weight-gradient kernels (<see cref="Kernels.CudaConvWgradKernels"/>) once.</summary>
    internal bool EnsureWgradR8Module()
    {
        lock (_wgradR8ModuleLock)
        {
            if (_wgradR8ModuleTried) return _wgradR8ModuleReady;
            _wgradR8ModuleTried = true;
            try
            {
                using var _ = PushContext();
                var deviceResult = CudaNativeBindings.cuCtxGetDevice(out int device);
                if (deviceResult != CudaResult.Success)
                {
                    System.Diagnostics.Trace.TraceWarning($"Weight-gradient r8 kernels unavailable: cuCtxGetDevice returned {deviceResult}.");
                    return false;
                }
                CompileKernelModule(device, Kernels.CudaConvWgradKernels.GetSource(), "conv_wgrad_r8_kernels",
                    Kernels.CudaConvWgradKernels.GetKernelNames());
                foreach (var name in Kernels.CudaConvWgradKernels.GetKernelNames())
                    if (!_kernelCache.ContainsKey(name))
                    {
                        System.Diagnostics.Trace.TraceWarning($"Weight-gradient r8 kernels unavailable: '{name}' missing after compile.");
                        return false;
                    }
                _wgradR8ModuleReady = true;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Trace.TraceWarning($"Weight-gradient r8 kernels unavailable: {ex.GetType().Name}: {ex.Message}");
            }
            return _wgradR8ModuleReady;
        }
    }

    /// <summary>
    /// The weight gradient through conv2d_wgrad_r8_{tileM}x{tileN} (8x8 micro-tile per thread) with a split-K count
    /// (0: enough splits for about two blocks per SM). Overwrites dW. False: nothing launched.
    /// </summary>
    internal unsafe bool TryConv2DBackwardKernelR8(in TunedConv2DBackwardArgs a, int tileM, int tileN, int splits)
    {
        if (!a.WeightGradient || a.Batch <= 0 || a.OutChannels <= 0 || !EnsureWgradR8Module()) return false;
        long p = (long)a.InChannels * a.KernelH * a.KernelW;
        long kTotal = (long)a.Batch * a.OutHeight * a.OutWidth;
        long outputs = a.OutChannels * p;
        if (p <= 0 || kTotal <= 0 || kTotal > int.MaxValue || outputs > int.MaxValue
            || (long)a.Batch * a.InChannels * a.InHeight * a.InWidth > int.MaxValue
            || (long)a.Batch * a.OutChannels * a.OutHeight * a.OutWidth > int.MaxValue
            || !_kernelCache.TryGetValue($"conv2d_wgrad_r8_{tileM}x{tileN}", out var gemm))
            return false;
        const int Bk = 16;   // the kernels' K tile (WR_KERNEL ..., 16)
        int threads = (tileM / 8) * (tileN / 8);
        int tilesN = (int)((p + tileN - 1) / tileN), tilesM = (a.OutChannels + tileM - 1) / tileM;
        long tiles = (long)tilesN * tilesM;
        int sms = Math.Max(1, _multiProcessorCount);
        if (splits <= 0)
            splits = tiles >= 2L * sms ? 1 : (int)Math.Min(Math.Max(1, kTotal / (8L * Bk)), (2L * sms + tiles - 1) / tiles);
        int kPerSplit = (int)(((kTotal + splits - 1) / splits + Bk - 1) / Bk * Bk);
        splits = (int)((kTotal + kPerSplit - 1) / kPerSplit);
        IntPtr reduce = IntPtr.Zero;
        if (splits > 1 && !_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce)) return false;
        using var context = PushContext();
        IGpuBuffer destination = a.Output;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _convEngineWgradPartials, splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }
        IntPtr x = a.First.Handle, dy = a.Second.Handle, o = destination.Handle;
        int batch = a.Batch, inC = a.InChannels, h = a.InHeight, w = a.InWidth, outC = a.OutChannels, outH = a.OutHeight,
            outW = a.OutWidth, kh = a.KernelH, kw = a.KernelW, sh = a.StrideH, sw = a.StrideW, ph = a.PadH, pw = a.PadW,
            dh = a.DilationH, dw = a.DilationW, kps = kPerSplit;
        void** args = stackalloc void*[19];
        args[0] = &x; args[1] = &dy; args[2] = &o;
        args[3] = &batch; args[4] = &inC; args[5] = &h; args[6] = &w;
        args[7] = &outC; args[8] = &outH; args[9] = &outW;
        args[10] = &kh; args[11] = &kw; args[12] = &sh; args[13] = &sw;
        args[14] = &ph; args[15] = &pw; args[16] = &dh; args[17] = &dw; args[18] = &kps;
        LaunchKernel3D(gemm, (uint)tilesN, (uint)tilesM, (uint)splits, (uint)threads, 1, 1, args, 0);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, a.Output, (int)outputs, splits);
        return true;
    }
}
