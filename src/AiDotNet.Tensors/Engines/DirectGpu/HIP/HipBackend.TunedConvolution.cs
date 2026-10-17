// Copyright (c) AiDotNet. All rights reserved.
// #1113: HIP's Conv2D forward through the tuned-kernel registry. The reference is the dispatch HIP shipped (Winograd
// F(2x2,3x3) for 3x3 stride-1 undilated filters, else the shared-memory tiled kernel, else the direct kernel); each of
// those kernels and an im2col + one-GEMM formulation are offered on their own, and the registry measures per shape.
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend
{
    private const double ConvolutionTolerance = 1e-4;
    private TunedKernelSlot<TunedConv2DArgs>? _conv2DForwardSlot;

    internal TunedKernelSlot<TunedConv2DArgs> Conv2DForwardSlot
    {
        get
        {
            if (_conv2DForwardSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_conv2DForwardSlot is null)
                {
                    var candidates = new ITunedKernelCandidate<TunedConv2DArgs>[]
                    {
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("hip.conv2d.generic", TunedKernelOrigin.Builtin, true,
                            _ => true, (in TunedConv2DArgs a) => Conv2DGeneric(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("hip.conv2d.im2col_gemm", TunedKernelOrigin.Generated, true,
                            shape => TunedConv2DArgs.GemmScratchFits(shape) && HasTunedKernel(HipTunedRowKernels.Im2ColName),
                            (in TunedConv2DArgs a) => Conv2DIm2ColGemm(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("hip.conv2d.winograd_f2x2_3x3", TunedKernelOrigin.Builtin, true,
                            shape => IsWinogradShape(shape) && HasTunedKernel("conv2d_winograd_f2x2_3x3"),
                            (in TunedConv2DArgs a) => Conv2DWinograd(a)),
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("hip.conv2d.tiled", TunedKernelOrigin.Builtin, true,
                            _ => HasTunedKernel("conv2d_tiled"), (in TunedConv2DArgs a) => Conv2DTiledOrDirect(a, tiled: true)),
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("hip.conv2d.direct", TunedKernelOrigin.Builtin, true,
                            _ => HasTunedKernel("conv2d_direct"), (in TunedConv2DArgs a) => Conv2DTiledOrDirect(a, tiled: false)),
                        // #1368: Winograd as transforms around one batched GEMM, where conv2d_winograd_f2x2_3x3 computes
                        // each output tile's whole channel sum in one thread.
                        new DelegateTunedKernelCandidate<TunedConv2DArgs>("hip.conv2d.winograd_gemm", TunedKernelOrigin.Builtin, true,
                            shape => WinogradF23Pipeline.AdmitsShape(shape) && WinogradKernelsReady,
                            (in TunedConv2DArgs a) => { if (!TryConv2DForwardWinograd(a)) Conv2DGeneric(a); }),
                    };
                    _conv2DForwardSlot = new TunedKernelSlot<TunedConv2DArgs>(TunedKernelOp.Conv2DForward, TunedDeviceKey,
                        new BackendTunedKernelHarness<TunedConv2DArgs>(this, ConvolutionTolerance, () => IsAvailable,
                            MeasureTunedKernelMilliseconds, a => new[] { (a.Output, a.OutputCount) },
                            a => a.Output.Handle == a.Input.Handle || a.Output.Handle == a.Kernel.Handle),
                        () => GpuDeterminism.IsActive, candidates);
                }
                return _conv2DForwardSlot;
            }
        }
    }

    // Shape class order (TunedConv2DArgs.Shape): kernelH 5, kernelW 6, strideH 7, strideW 8, dilationH 11, dilationW 12.
    private static bool IsWinogradShape(TunedShape s)
        => s[5] == 3 && s[6] == 3 && s[7] == 1 && s[8] == 1 && s[11] == 1 && s[12] == 1;

    /// <summary>The dispatch HIP's Conv2D ran before the registry, kept as the reference.</summary>
    private void Conv2DGeneric(in TunedConv2DArgs a)
    {
        if (IsWinogradShape(a.Shape()) && HasTunedKernel("conv2d_winograd_f2x2_3x3")) Conv2DWinograd(a);
        else Conv2DTiledOrDirect(a, tiled: HasTunedKernel("conv2d_tiled"));
    }

    private unsafe void Conv2DWinograd(in TunedConv2DArgs a)
    {
        IntPtr kernel = TunedKernel("conv2d_winograd_f2x2_3x3");
        IntPtr x = a.Input.Handle, w = a.Kernel.Handle, y = a.Output.Handle;
        int batch = a.Batch, cin = a.InChannels, h = a.InHeight, wd = a.InWidth, cout = a.OutChannels;
        int oh = a.OutHeight, ow = a.OutWidth, padH = a.PadH, padW = a.PadW;
        int tiles = batch * cout * ((oh + 1) / 2) * ((ow + 1) / 2);
        void** args = stackalloc void*[12];
        args[0] = &x; args[1] = &w; args[2] = &y; args[3] = &batch; args[4] = &cin; args[5] = &h; args[6] = &wd;
        args[7] = &cout; args[8] = &oh; args[9] = &ow; args[10] = &padH; args[11] = &padW;
        LaunchKernel(kernel, (uint)((tiles + 255) / 256), 256, args);
    }

    // conv2d_tiled and conv2d_direct take the same 18 arguments; they differ in block shape and shared memory.
    private unsafe void Conv2DTiledOrDirect(in TunedConv2DArgs a, bool tiled)
    {
        IntPtr kernel = TunedKernel(tiled ? "conv2d_tiled" : "conv2d_direct");
        IntPtr x = a.Input.Handle, w = a.Kernel.Handle, y = a.Output.Handle;
        int batch = a.Batch, cin = a.InChannels, h = a.InHeight, wd = a.InWidth, cout = a.OutChannels;
        int oh = a.OutHeight, ow = a.OutWidth, kh = a.KernelH, kw = a.KernelW, sh = a.StrideH, sw = a.StrideW;
        int ph = a.PadH, pw = a.PadW, dh = a.DilationH, dw = a.DilationW;
        void** args = stackalloc void*[18];
        args[0] = &x; args[1] = &w; args[2] = &y; args[3] = &batch; args[4] = &cin; args[5] = &h; args[6] = &wd;
        args[7] = &cout; args[8] = &oh; args[9] = &ow; args[10] = &kh; args[11] = &kw; args[12] = &sh; args[13] = &sw;
        args[14] = &ph; args[15] = &pw; args[16] = &dh; args[17] = &dw;
        const int Tile = 16;
        uint gx = (uint)((ow + Tile - 1) / Tile), gy = (uint)((oh + Tile - 1) / Tile), gz = (uint)(batch * cout);
        if (!tiled)
        {
            LaunchKernel3D(kernel, gx, gy, gz, Tile, Tile, 1, args);
            return;
        }
        int effKH = (kh - 1) * dh + 1, effKW = (kw - 1) * dw + 1;
        int tileInH = Tile * sh + effKH - sh, tileInW = Tile * sw + effKW - sw;
        LaunchKernel3D(kernel, gx, gy, gz, Tile, Tile, 1, args, (uint)(tileInH * tileInW * sizeof(float)));
    }

    private unsafe void Conv2DIm2ColGemm(in TunedConv2DArgs a)
    {
        int l = a.OutHeight * a.OutWidth, p = a.InChannels * a.KernelH * a.KernelW;
        using var col = AllocateBuffer(checked(a.Batch * p * l));
        IntPtr kernel = TunedKernel(HipTunedRowKernels.Im2ColName);
        IntPtr x = a.Input.Handle, c = col.Handle;
        int batch = a.Batch, cin = a.InChannels, h = a.InHeight, w = a.InWidth, kh = a.KernelH, kw = a.KernelW;
        int sh = a.StrideH, sw = a.StrideW, ph = a.PadH, pw = a.PadW, dh = a.DilationH, dw = a.DilationW;
        int oh = a.OutHeight, ow = a.OutWidth;
        void** args = stackalloc void*[16];
        args[0] = &x; args[1] = &c; args[2] = &batch; args[3] = &cin; args[4] = &h; args[5] = &w;
        args[6] = &kh; args[7] = &kw; args[8] = &sh; args[9] = &sw; args[10] = &ph; args[11] = &pw;
        args[12] = &dh; args[13] = &dw; args[14] = &oh; args[15] = &ow;
        LaunchKernel(kernel, (uint)((batch * l + 63) / 64), 64, args);
        if (a.Batch == 1)
        {
            Gemm(a.Kernel, col, a.Output, a.OutChannels, l, p);
            return;
        }
        using var tmp = AllocateBuffer(checked(a.OutChannels * a.Batch * l));
        Gemm(a.Kernel, col, tmp, a.OutChannels, a.Batch * l, p);
        Permute(tmp, a.Output, new[] { a.OutChannels, a.Batch, l }, new[] { 1, 0, 2 });
    }
}