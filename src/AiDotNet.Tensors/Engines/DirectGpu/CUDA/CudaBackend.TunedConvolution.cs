using System;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>2-D convolution arguments (NCHW input, [Cout, Cin, kH, kW] filter).</summary>
internal readonly struct CudaConv2DArgs
{
    internal CudaConv2DArgs(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        Input = input; Kernel = kernel; Output = output;
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth;
        OutChannels = outChannels; OutHeight = outHeight; OutWidth = outWidth;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW;
        PadH = padH; PadW = padW; DilationH = dilationH; DilationW = dilationW;
    }

    internal IGpuBuffer Input { get; }
    internal IGpuBuffer Kernel { get; }
    internal IGpuBuffer Output { get; }
    internal int Batch { get; }
    internal int InChannels { get; }
    internal int InHeight { get; }
    internal int InWidth { get; }
    internal int OutChannels { get; }
    internal int OutHeight { get; }
    internal int OutWidth { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int StrideH { get; }
    internal int StrideW { get; }
    internal int PadH { get; }
    internal int PadW { get; }
    internal int DilationH { get; }
    internal int DilationW { get; }

    internal int OutputCount => Batch * OutChannels * OutHeight * OutWidth;

    /// <summary>Shape class: every geometric parameter (output extent follows from them).</summary>
    internal unsafe TunedShape Shape()
    {
        int* d = stackalloc int[15];
        d[0] = Batch; d[1] = InChannels; d[2] = InHeight; d[3] = InWidth; d[4] = OutChannels;
        d[5] = KernelH; d[6] = KernelW; d[7] = StrideH; d[8] = StrideW; d[9] = PadH; d[10] = PadW;
        d[11] = DilationH; d[12] = DilationW; d[13] = OutHeight; d[14] = OutWidth;
        return TunedShape.Create(TunedKernelDType.Float32, new ReadOnlySpan<int>(d, 15));
    }
}

/// <summary>
/// Convolution forward through the tuned-kernel registry. The reference is the established generic dispatch
/// (Winograd F(2x2,3x3) for 3x3 stride 1, else tiled, else direct); the candidates are the explicit tiled and
/// direct kernels and im2col + one strided-batched cuBLAS GEMM (the implicit-GEMM formulation the backward pass
/// already uses), each selected per shape class only after the correctness and paired-timing gate.
/// </summary>
public sealed partial class CudaBackend
{
    // Convolution reassociates long K = Cin*kH*kW dot products; Winograd and GEMM differ from the direct sum by
    // a few ulps of the output scale.
    private const double ConvolutionTolerance = 1e-4;
    private TunedKernelSlot<CudaConv2DArgs>? _conv2DForwardSlot;

    internal TunedKernelSlot<CudaConv2DArgs> Conv2DForwardSlot
    {
        get
        {
            if (_conv2DForwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_conv2DForwardSlot is null)
                {
                    var candidates = new ITunedKernelCandidate<CudaConv2DArgs>[]
                    {
                        new CudaTunedKernelCandidate<CudaConv2DArgs>("nvrtc.conv2d.generic", TunedKernelOrigin.Builtin,
                            true, _ => true, (in CudaConv2DArgs a) => LaunchConv2DGeneric(a.Input, a.Kernel, a.Output,
                                a.Batch, a.InChannels, a.InHeight, a.InWidth, a.OutChannels, a.OutHeight, a.OutWidth,
                                a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW)),
                        new CudaTunedKernelCandidate<CudaConv2DArgs>("cublas.conv2d.im2col_gemm", TunedKernelOrigin.Vendor,
                            true, shape => ConvGemmForwardFits(shape), (in CudaConv2DArgs a) => ExecuteConv2DForwardGemm(a)),
                        new CudaTunedKernelCandidate<CudaConv2DArgs>("nvrtc.conv2d.tiled", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_tiled"), (in CudaConv2DArgs a) => LaunchConv2DTiled(a)),
                        new CudaTunedKernelCandidate<CudaConv2DArgs>("nvrtc.conv2d.direct", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_direct"), (in CudaConv2DArgs a) => LaunchConv2DDirect(a)),
                    };
                    _conv2DForwardSlot = new TunedKernelSlot<CudaConv2DArgs>(TunedKernelOp.Conv2DForward,
                        TunedDeviceKey,
                        new CudaTunedKernelHarness<CudaConv2DArgs>(this, ConvolutionTolerance,
                            a => new[] { (a.Output, a.OutputCount) },
                            a => a.Output.Handle == a.Input.Handle || a.Output.Handle == a.Kernel.Handle),
                        () => GpuDeterminism.IsActive, candidates);
                }
                return Created(_conv2DForwardSlot);
            }
        }
    }

    // The im2col scratch is batch * Cin*kH*kW * outH*outW floats; keep it under 256 MiB.
    private static bool ConvGemmForwardFits(TunedShape s)
    {
        long outPixels = (long)s[13] * s[14];
        long floats = (long)s[0] * s[1] * s[5] * s[6] * outPixels;
        return floats > 0 && floats <= 64L * 1024 * 1024;
    }

    private void ExecuteConv2DForwardGemm(in CudaConv2DArgs a)
    {
        // During capture without a cuBLAS workspace, or before the scratch is sized, the GEMM route is unusable;
        // the generic kernel computes the same convolution.
        if (!TryConv2DForwardGemm(a))
            LaunchConv2DGeneric(a.Input, a.Kernel, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW,
                a.PadH, a.PadW, a.DilationH, a.DilationW);
    }

    /// <summary>out_b[Cout, L] = W[Cout, P] * col_b[P, L] for every batch item, as one strided-batched GEMM.</summary>
    private bool TryConv2DForwardGemm(in CudaConv2DArgs a)
    {
        if (!ConvGemmUsable() || a.Batch <= 0) return false;
        int L = a.OutHeight * a.OutWidth, P = a.InChannels * a.KernelH * a.KernelW;
        long needed = (long)a.Batch * P * L;
        using var _ = PushContext();
        // Null when the scratch cannot be sized (too large, or undersized during a capture): the slot's next
        // candidate runs instead.
        var col = TryConvScratch(ref _convColScratch, needed);
        if (col is null) return false;
        LaunchIm2Col(a.Input, col, a.Batch, a.InChannels, a.InHeight, a.InWidth, a.KernelH, a.KernelW,
            a.StrideH, a.StrideW, a.PadH, a.PadW, a.DilationH, a.DilationW, a.OutHeight, a.OutWidth);
        ApplyDeterministicGemmMathMode();
        float one = 1f, zero = 0f;
        // Column-major: out_b^T[L x Cout] = col_b^T-view[L x P] * W^T-view[P x Cout]; W is shared (stride 0).
        CuBlasNative.CheckCublasStatus(CuBlasNative.cublasSgemmStridedBatched(_cublasHandle,
            CublasOperation.None, CublasOperation.None, L, a.OutChannels, P,
            ref one, col.Handle, L, (long)P * L, a.Kernel.Handle, P, 0L,
            ref zero, a.Output.Handle, L, (long)a.OutChannels * L, a.Batch), "cublasSgemmStridedBatched(conv fwd)");
        return true;
    }

    private unsafe void LaunchConv2DTiled(in CudaConv2DArgs a)
    {
        IntPtr tiledKernel = _kernelCache["conv2d_tiled"];
        using var _ = PushContext();
        const int TILE_OUT = 16;
        int effKH = (a.KernelH - 1) * a.DilationH + 1;
        int effKW = (a.KernelW - 1) * a.DilationW + 1;
        int tileInH = TILE_OUT * a.StrideH + effKH - a.StrideH;
        int tileInW = TILE_OUT * a.StrideW + effKW - a.StrideW;
        uint sharedMem = (uint)(tileInH * tileInW * sizeof(float));
        LaunchConv2DGrid(tiledKernel, a, (uint)((a.OutWidth + TILE_OUT - 1) / TILE_OUT),
            (uint)((a.OutHeight + TILE_OUT - 1) / TILE_OUT), TILE_OUT, sharedMem);
    }

    private unsafe void LaunchConv2DDirect(in CudaConv2DArgs a)
    {
        IntPtr directKernel = _kernelCache["conv2d_direct"];
        using var _ = PushContext();
        const int BLOCK = 16;
        LaunchConv2DGrid(directKernel, a, (uint)((a.OutWidth + BLOCK - 1) / BLOCK),
            (uint)((a.OutHeight + BLOCK - 1) / BLOCK), BLOCK, 0);
    }

    // conv2d_tiled and conv2d_direct share the 18-parameter ABI (in, w, out, B, Cin, H, W, Cout, OH, OW, kH, kW,
    // sH, sW, pH, pW, dH, dW) and a (OW tiles, OH tiles, B*Cout) grid.
    private unsafe void LaunchConv2DGrid(IntPtr function, in CudaConv2DArgs a, uint gx, uint gy, int block, uint sharedMem)
    {
        IntPtr inputPtr = a.Input.Handle, kernelPtr = a.Kernel.Handle, outputPtr = a.Output.Handle;
        int batch = a.Batch, inC = a.InChannels, inH = a.InHeight, inW = a.InWidth, outC = a.OutChannels;
        int outH = a.OutHeight, outW = a.OutWidth, kH = a.KernelH, kW = a.KernelW, sH = a.StrideH, sW = a.StrideW;
        int pH = a.PadH, pW = a.PadW, dH = a.DilationH, dW = a.DilationW;
        void** args = stackalloc void*[18];
        args[0] = &inputPtr; args[1] = &kernelPtr; args[2] = &outputPtr; args[3] = &batch; args[4] = &inC;
        args[5] = &inH; args[6] = &inW; args[7] = &outC; args[8] = &outH; args[9] = &outW; args[10] = &kH;
        args[11] = &kW; args[12] = &sH; args[13] = &sW; args[14] = &pH; args[15] = &pW; args[16] = &dH;
        args[17] = &dW;
        LaunchKernel3D(function, gx, gy, (uint)(batch * outC), (uint)block, (uint)block, 1, args, sharedMem);
    }
}
