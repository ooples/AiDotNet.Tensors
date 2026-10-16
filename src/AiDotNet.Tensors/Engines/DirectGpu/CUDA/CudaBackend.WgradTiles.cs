// Copyright (c) AiDotNet. All rights reserved.
using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

public sealed partial class CudaBackend
{
    /// <summary>
    /// The weight gradient on the engine body with an explicit tile (rows x columns of dW = Cout x Cin*kH*kW: 64x64,
    /// 128x128, 64x128 or 128x64) and split-K count (0: enough splits for about two blocks per SM). Overwrites dW.
    /// False: nothing launched.
    /// </summary>
    internal bool TryConv2DBackwardKernelEngineTiled(in CudaConv2DBackwardArgs a, int tileM, int tileN, int splits)
    {
        if (!a.WeightGradient || a.Batch <= 0 || a.OutChannels <= 0) return false;
        string? name = (tileM, tileN) switch
        {
            (64, 64) => "conv2d_wgrad_ce64",
            (128, 128) => "conv2d_wgrad_ce128",
            (64, 128) => "conv2d_wgrad_ce64x128",
            (128, 64) => "conv2d_wgrad_ce128x64",
            _ => null,
        };
        long p = (long)a.InChannels * a.KernelH * a.KernelW;
        long kTotal = (long)a.Batch * a.OutHeight * a.OutWidth;
        long outputs = a.OutChannels * p;
        if (name is null || p <= 0 || kTotal <= 0 || kTotal > int.MaxValue || outputs > int.MaxValue
            || !_kernelCache.TryGetValue(name, out var gemm))
            return false;
        int tilesN = (int)((p + tileN - 1) / tileN), tilesM = (a.OutChannels + tileM - 1) / tileM;
        long tiles = (long)tilesN * tilesM;
        int sms = Math.Max(1, _multiProcessorCount);
        if (splits <= 0)
            splits = tiles >= 2L * sms ? 1 : (int)Math.Min(Math.Max(1, kTotal / (8L * ConvEngineTileK)), (2L * sms + tiles - 1) / tiles);
        int kPerSplit = (int)(((kTotal + splits - 1) / splits + ConvEngineTileK - 1) / ConvEngineTileK * ConvEngineTileK);
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
        LaunchConvEngine(gemm, a.First, a.Second, destination, tilesN, tilesM, splits, kPerSplit, a.Batch, a.InChannels,
            a.InHeight, a.InWidth, a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW,
            a.PadH, a.PadW, a.DilationH, a.DilationW);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, a.Output, (int)outputs, splits);
        return true;
    }
}
