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

    // #1335: the engine's inner product. FMA = SIMT fp32; X3 = 3xTF32 on the Tensor Cores (fp32-level accuracy);
    // TF32 = plain TF32 (explicit reduced-precision opt-in, CudaDispatchPolicy.ConvReducedPrecisionTF32).
    internal const int ConvMainloopFma = 0, ConvMainloop3xTf32 = 1, ConvMainloopTf32 = 2;

    private static string ConvMainloopSuffix(int mainloop) => mainloop switch
    {
        ConvMainloop3xTf32 => "_x3",
        ConvMainloopTf32 => "_tf32",
        _ => "",
    };

    /// <summary>mma.sync TF32 needs compute capability 8.0+.</summary>
    internal bool ConvTensorCoresAvailable => _ccMajor >= 8;

    private int? _convEngineMainloopOverride;

    /// <summary>
    /// The mainloop the engine runs outside the tuned slots (the fixed fallback chain and the per-route entry points):
    /// plain TF32 under the explicit opt-in, else 3xTF32 on Ampere+, else FMA. Settable for tests and the benchmark.
    /// </summary>
    internal int ConvEngineMainloop
    {
        get => _convEngineMainloopOverride ?? (!ConvTensorCoresAvailable ? ConvMainloopFma
            : CudaDispatchPolicy.ConvReducedPrecisionTF32 ? ConvMainloopTf32 : ConvMainloop3xTf32);
        set => _convEngineMainloopOverride = value;
    }

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
    internal unsafe bool TryConv2DForwardEngine(in CudaConv2DArgs a) => TryConv2DForwardEngine(a, ConvEngineMainloop);

    internal unsafe bool TryConv2DForwardEngine(in CudaConv2DArgs a, int mainloop)
    {
        if (mainloop != ConvMainloopFma && !ConvTensorCoresAvailable) return false;
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
        string name = (tile == 128 ? "conv2d_fprop_ig128_" : "conv2d_fprop_ig64_") + (tapUniform ? "tap" : "gen")
                      + ConvMainloopSuffix(mainloop);
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

    /// <summary>
    /// #1325: conv backward through the engine (conv2d_wgrad_ce*, conv2d_dgrad_ce*) before #1323's kernels;
    /// AIDOTNET_CONV_ENGINE_BACKWARD=0 starts it off. Settable so tests and the benchmark can interleave both.
    /// </summary>
    internal bool UseConvEngineBackward { get; set; } =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_ENGINE_BACKWARD") != "0";

    private IGpuBuffer? _convEngineWgradPartials, _convEngineDgradPartials;

    /// <summary>gradKernel[Cout, Cin*kH*kW] through conv2d_wgrad_ce*. Overwrites gradKernel. False: nothing launched.</summary>
    private bool TryConv2DBackwardKernelEngine(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int mainloop)
    {
        if (!s_convGemmBackward || !UseConvEngineBackward || batch <= 0 || outChannels <= 0) return false;
        if (mainloop != ConvMainloopFma && !ConvTensorCoresAvailable) return false;
        long p = (long)inChannels * kernelH * kernelW;
        long kTotal = (long)batch * outHeight * outWidth;
        long outputs = outChannels * p;
        if (p <= 0 || kTotal <= 0 || kTotal > int.MaxValue || outputs > int.MaxValue
            || (long)batch * inChannels * inHeight * inWidth > int.MaxValue
            || (long)batch * outChannels * outHeight * outWidth > int.MaxValue)
            return false;
        ConvEnginePlan(p, outChannels, kTotal, out int tile, out int tilesN, out int tilesM, out int splits, out int kPerSplit);
        if (!_kernelCache.TryGetValue((tile == 128 ? "conv2d_wgrad_ce128" : "conv2d_wgrad_ce64") + ConvMainloopSuffix(mainloop),
                out var gemm)) return false;
        IntPtr reduce = IntPtr.Zero;
        if (splits > 1 && !_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce)) return false;

        using var context = PushContext();
        IGpuBuffer destination = gradKernel;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _convEngineWgradPartials, splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }
        LaunchConvEngine(gemm, input, gradOutput, destination, tilesN, tilesM, splits, kPerSplit, batch, inChannels, inHeight,
            inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, gradKernel, (int)outputs, splits);
        return true;
    }

    /// <summary>
    /// gradInput through conv2d_dgrad_ce* for Cout % 8 == 0: the sub-pixel phase kernel for strided, undilated
    /// convolutions whose kernel covers the stride, else the tap-uniform kernel with split-K. Overwrites gradInput.
    /// False: nothing launched.
    /// </summary>
    private bool TryConv2DBackwardInputEngine(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int mainloop)
    {
        if (mainloop != ConvMainloopFma && !ConvTensorCoresAvailable) return false;
        if (!s_convGemmBackward || !UseConvEngineBackward || batch <= 0 || inChannels <= 0
            || outChannels % ConvEngineTileK != 0) return false;
        long outputs = (long)batch * inChannels * inHeight * inWidth;
        long kTotal = (long)outChannels * kernelH * kernelW;
        if (outputs <= 0 || kTotal <= 0 || outputs > int.MaxValue || kTotal > int.MaxValue
            || (long)batch * outChannels * outHeight * outWidth > int.MaxValue)
            return false;
        bool phase = (strideH > 1 || strideW > 1) && dilationH == 1 && dilationW == 1
                     && kernelH >= strideH && kernelW >= strideW;
        using var context = PushContext();
        if (phase)
        {
            // The largest phase has ceil(H / sH) x ceil(W / sW) pixels per image; one grid layer per phase, no split.
            long columns = (long)batch * ((inHeight + strideH - 1) / strideH) * ((inWidth + strideW - 1) / strideW);
            ConvEnginePlan(columns, inChannels, kTotal, out int ptile, out int ptilesN, out int ptilesM, out _, out _);
            if (!_kernelCache.TryGetValue((ptile == 128 ? "conv2d_dgrad_ce128_phase" : "conv2d_dgrad_ce64_phase")
                    + ConvMainloopSuffix(mainloop), out var pgemm))
                return false;
            LaunchConvEngine(pgemm, gradOutput, kernel, gradInput, ptilesN, ptilesM, strideH * strideW, 0, batch, inChannels,
                inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
                dilationH, dilationW);
            return true;
        }

        ConvEnginePlan((long)batch * inHeight * inWidth, inChannels, kTotal, out int tile, out int tilesN, out int tilesM,
            out int splits, out int kPerSplit);
        if (!_kernelCache.TryGetValue((tile == 128 ? "conv2d_dgrad_ce128" : "conv2d_dgrad_ce64") + ConvMainloopSuffix(mainloop),
                out var gemm)) return false;
        IntPtr reduce = IntPtr.Zero;
        if (splits > 1 && !_kernelCache.TryGetValue("conv2d_wgrad_splitk_reduce", out reduce)) return false;
        IGpuBuffer destination = gradInput;
        if (splits > 1)
        {
            var partials = TryConvScratch(ref _convEngineDgradPartials, splits * outputs);
            if (partials is null) return false;
            destination = partials;
        }
        LaunchConvEngine(gemm, gradOutput, kernel, destination, tilesN, tilesM, splits, kPerSplit, batch, inChannels, inHeight,
            inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW, dilationH, dilationW);
        if (splits > 1) LaunchSplitKReduce(reduce, destination, gradInput, (int)outputs, splits);
        return true;
    }

    // Every engine kernel shares the 19-parameter ABI (two operands, output, the 15 conv sizes, kPerSplit).
    private unsafe void LaunchConvEngine(IntPtr function, IGpuBuffer first, IGpuBuffer second, IGpuBuffer output,
        int tilesN, int tilesM, int gridZ, int kPerSplit, int batch, int inC, int inH, int inW, int outC, int outH, int outW,
        int kH, int kW, int sH, int sW, int pH, int pW, int dH, int dW)
    {
        IntPtr aPtr = first.Handle, bPtr = second.Handle, outPtr = output.Handle;
        void** args = stackalloc void*[19];
        args[0] = &aPtr; args[1] = &bPtr; args[2] = &outPtr;
        args[3] = &batch; args[4] = &inC; args[5] = &inH; args[6] = &inW;
        args[7] = &outC; args[8] = &outH; args[9] = &outW;
        args[10] = &kH; args[11] = &kW; args[12] = &sH; args[13] = &sW;
        args[14] = &pH; args[15] = &pW; args[16] = &dH; args[17] = &dW; args[18] = &kPerSplit;
        LaunchKernel3D(function, (uint)tilesN, (uint)tilesM, (uint)gridZ, (uint)DefaultBlockSize, 1, 1, args, 0);
    }

    // dst[i] = sum over z of partials[z][i], in z order (conv2d_wgrad_splitk_reduce).
    private unsafe void LaunchSplitKReduce(IntPtr reduce, IGpuBuffer partials, IGpuBuffer destination, int count, int splits)
    {
        IntPtr partPtr = partials.Handle, dstPtr = destination.Handle;
        void** rargs = stackalloc void*[4];
        rargs[0] = &partPtr; rargs[1] = &dstPtr; rargs[2] = &count; rargs[3] = &splits;
        LaunchKernel(reduce, (uint)((count + DefaultBlockSize - 1) / DefaultBlockSize), DefaultBlockSize, rargs);
    }

    // Entry points for the per-path forward benchmark (Bench_ResNet18ConvForward): each runs one route regardless of
    // what the tuned-kernel slot would pick. False when that route declines the shape.
    internal bool Conv2DForwardViaEngine(in CudaConv2DArgs a) => TryConv2DForwardEngine(a);

    internal bool Conv2DForwardViaEngine(in CudaConv2DArgs a, int mainloop) => TryConv2DForwardEngine(a, mainloop);

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
