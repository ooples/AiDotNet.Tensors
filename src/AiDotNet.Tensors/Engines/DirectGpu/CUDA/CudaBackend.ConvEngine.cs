using System;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

// #1325: Conv2D forward through the implicit-GEMM conv engine (CudaConvolutionKernels conv2d_fprop_ig*): a 128 x 128 or
// 64 x 64 output tile per 256-thread block, the input gathered into shared memory with the convolution's own index
// arithmetic, register prefetch over double-buffered K tiles, and a deterministic split-K when the tiles alone cannot
// fill the device. It competes in the forward tuned-kernel slot like every other conv kernel.
public sealed partial class CudaBackend
{
    /// <summary>AIDOTNET_CONV_ENGINE_FPROP=0 keeps the engine's forward out of the tuned-kernel slot.</summary>
    private static readonly bool s_convEngineFprop =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_ENGINE_FPROP") != "0";

    // Engine K tile (CE_BK in CudaConvolutionKernels).
    private const int ConvEngineTileK = 8;

    private IGpuBuffer? _convFpropPartials;

    /// <summary>
    /// Tile size and split-K of an engine launch: 128 x 128 tiles when they already give every SM a block, else 64 x 64;
    /// when even those leave SMs idle, K is split so there are about two blocks per SM, never with a split shorter
    /// than eight K tiles. kPerSplit is a multiple of the K tile.
    /// </summary>
    private void ConvEnginePlan(long columns, int rows, long kTotal,
        out int tile, out int tilesN, out int tilesM, out int splits, out int kPerSplit)
    {
        int sms = Math.Max(1, _multiProcessorCount);
        tile = 128;
        tilesN = (int)((columns + tile - 1) / tile);
        tilesM = (rows + tile - 1) / tile;
        if ((long)tilesN * tilesM < sms)
        {
            tile = 64;
            tilesN = (int)((columns + tile - 1) / tile);
            tilesM = (rows + tile - 1) / tile;
        }
        long tiles = (long)tilesN * tilesM;
        splits = 1;
        if (tiles < sms)
        {
            long maxSplits = Math.Max(1, kTotal / (8L * ConvEngineTileK));
            splits = (int)Math.Min(maxSplits, (2L * sms + tiles - 1) / tiles);
        }
        kPerSplit = (int)(((kTotal + splits - 1) / splits + ConvEngineTileK - 1) / ConvEngineTileK * ConvEngineTileK);
        splits = (int)((kTotal + kPerSplit - 1) / kPerSplit);
    }

    /// <summary>
    /// Y = conv(X, W) through conv2d_fprop_ig*. Overwrites the output. False (nothing launched) when the kernels are
    /// missing, the sizes overflow a 32-bit index, or split-K partials cannot be sized (during a capture).
    /// </summary>
    internal unsafe bool TryConv2DForwardEngine(in CudaConv2DArgs a)
    {
        if (a.Batch <= 0 || a.OutHeight <= 0 || a.OutWidth <= 0 || a.OutChannels <= 0) return false;
        long columns = (long)a.Batch * a.OutHeight * a.OutWidth;
        long kTotal = (long)a.InChannels * a.KernelH * a.KernelW;
        long outputs = columns * a.OutChannels;
        if (kTotal <= 0 || outputs > int.MaxValue || kTotal > int.MaxValue
            || (long)a.Batch * a.InChannels * a.InHeight * a.InWidth > int.MaxValue
            || (long)a.OutChannels * kTotal > int.MaxValue)
            return false;

        ConvEnginePlan(columns, a.OutChannels, kTotal, out int tile, out int tilesN, out int tilesM,
            out int splits, out int kPerSplit);
        // Cin % 8 == 0 keeps every K tile on one kernel tap (the _tap variant); otherwise per-element index math.
        bool tapUniform = a.InChannels % ConvEngineTileK == 0;
        string name = (tile == 128 ? "conv2d_fprop_ig128_" : "conv2d_fprop_ig64_") + (tapUniform ? "tap" : "gen");
        if (!_kernelCache.TryGetValue(name, out var gemm)) return false;
        IntPtr reduce = IntPtr.Zero;
        if (splits > 1 && !_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce)) return false;

        using var _ = PushContext();
        IGpuBuffer destination = a.Output;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _convFpropPartials, splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }

        IntPtr xPtr = a.Input.Handle, wPtr = a.Kernel.Handle, outPtr = destination.Handle;
        int batch = a.Batch, inC = a.InChannels, inH = a.InHeight, inW = a.InWidth, outC = a.OutChannels;
        int outH = a.OutHeight, outW = a.OutWidth, kH = a.KernelH, kW = a.KernelW, sH = a.StrideH, sW = a.StrideW;
        int pH = a.PadH, pW = a.PadW, dH = a.DilationH, dW = a.DilationW;
        void** args = stackalloc void*[19];
        args[0] = &xPtr; args[1] = &wPtr; args[2] = &outPtr;
        args[3] = &batch; args[4] = &inC; args[5] = &inH; args[6] = &inW;
        args[7] = &outC; args[8] = &outH; args[9] = &outW;
        args[10] = &kH; args[11] = &kW; args[12] = &sH; args[13] = &sW;
        args[14] = &pH; args[15] = &pW; args[16] = &dH; args[17] = &dW; args[18] = &kPerSplit;
        LaunchKernel3D(gemm, (uint)tilesN, (uint)tilesM, (uint)splits, (uint)DefaultBlockSize, 1, 1, args, 0);

        if (splits > 1)
        {
            IntPtr partPtr = destination.Handle, yPtr = a.Output.Handle;
            int count = (int)outputs;
            void** rargs = stackalloc void*[4];
            rargs[0] = &partPtr; rargs[1] = &yPtr; rargs[2] = &count; rargs[3] = &splits;
            LaunchKernel(reduce, (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, rargs);
        }
        return true;
    }

    // Entry points for the per-path forward benchmark (Bench_ResNet18ConvForward): each runs one route regardless of
    // what the tuned-kernel slot would pick. False when that route declines the shape.
    internal bool Conv2DForwardViaEngine(in CudaConv2DArgs a) => TryConv2DForwardEngine(a);

    internal bool Conv2DForwardViaNarrowImplicit(in CudaConv2DArgs a) =>
        TryConv2DImplicitGemm(a.Input, a.Kernel, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW,
            a.PadH, a.PadW, a.DilationH, a.DilationW);

    internal bool Conv2DForwardViaIm2colGemm(in CudaConv2DArgs a) => TryConv2DForwardGemm(a);

    internal bool Conv2DForwardViaGeneric(in CudaConv2DArgs a)
    {
        LaunchConv2DGeneric(a.Input, a.Kernel, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW,
            a.PadH, a.PadW, a.DilationH, a.DilationW);
        return true;
    }
}
