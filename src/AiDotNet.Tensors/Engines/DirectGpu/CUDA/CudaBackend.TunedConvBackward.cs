using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// Conv2D backward arguments. Weight gradient: First = input, Second = gradOutput, Output = gradKernel. Input gradient:
/// First = gradOutput, Second = kernel, Output = gradInput. Sizes as in <see cref="CudaConv2DArgs"/>.
/// </summary>
internal readonly struct CudaConv2DBackwardArgs
{
    internal CudaConv2DBackwardArgs(IGpuBuffer first, IGpuBuffer second, IGpuBuffer output, bool weightGradient,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
    {
        First = first; Second = second; Output = output; WeightGradient = weightGradient;
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth;
        OutChannels = outChannels; OutHeight = outHeight; OutWidth = outWidth;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW;
        PadH = padH; PadW = padW; DilationH = dilationH; DilationW = dilationW;
    }

    internal IGpuBuffer First { get; }
    internal IGpuBuffer Second { get; }
    internal IGpuBuffer Output { get; }
    internal bool WeightGradient { get; }
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

    internal int OutputCount => WeightGradient
        ? OutChannels * InChannels * KernelH * KernelW
        : Batch * InChannels * InHeight * InWidth;

    /// <summary>Shape class: every geometric parameter, laid out as <see cref="CudaConv2DArgs.Shape"/>.</summary>
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
/// #1325: Conv2D backward (weight and input gradients) through the tuned-kernel registry, as the forward pass already is.
/// No one route wins every shape: on ResNet-18 #1323's implicit GEMM has the fastest weight gradient on most layers,
/// the #1325 engine the fastest stride-1 input gradient, and SGEMM + col2im the fastest strided 3x3 input gradient. Each
/// route competes per shape class after the correctness and paired-timing gate. The slot's reference (its first
/// applicable candidate, served under capture and once the process's online tuning budget is spent) is the strongest
/// general route; a route that declines at dispatch (partials that cannot be sized during a capture) runs the direct
/// kernel instead.
/// A route switched off for the process is not registered, so its two states never share persisted decisions.
/// </summary>
public sealed partial class CudaBackend
{
    private TunedKernelSlot<CudaConv2DBackwardArgs>? _conv2DBackwardFilterSlot, _conv2DBackwardDataSlot;

    private static readonly bool s_convImplicitWgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_IMPLICIT_WGRAD") != "0";
    private static readonly bool s_convImplicitDgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_IMPLICIT_DGRAD") != "0";
    private static readonly bool s_convEngineBackwardEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_ENGINE_BACKWARD") != "0";

    internal TunedKernelSlot<CudaConv2DBackwardArgs> Conv2DBackwardFilterSlot
    {
        get
        {
            if (_conv2DBackwardFilterSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_conv2DBackwardFilterSlot is null)
                {
                    // The first applicable candidate is the slot's reference: it is served under capture, after the
                    // process's online tuning budget is spent, and as the correctness baseline. So the strongest general
                    // route goes first (#1323's implicit GEMM: fastest weight gradient on most ResNet-18 layers), and the
                    // direct kernel, which never wins, is registered only when nothing else is.
                    var candidates = new List<ITunedKernelCandidate<CudaConv2DBackwardArgs>>();
                    if (s_convGemmBackward && s_convImplicitWgradEnv)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dw.implicit",
                            TunedKernelOrigin.Builtin, true, _ => HasTunedKernel("conv2d_wgrad_implicit_gemm"),
                            (in CudaConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaImplicit(a)) LaunchDwDirect(a); }));
                    if (s_convGemmBackward && s_convEngineBackwardEnv)
                        foreach (int mainloop in ConvEngineMainloopsInOrder())
                        {
                            int ml = mainloop;
                            candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>(
                                "nvrtc.conv2d_dw.engine" + ConvMainloopSuffix(ml), TunedKernelOrigin.Builtin, true,
                                _ => HasTunedKernel("conv2d_wgrad_ce128" + ConvMainloopSuffix(ml)),
                                (in CudaConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaEngine(a, ml)) LaunchDwDirect(a); }));
                        }
                    if (s_convGemmBackward && ImplicitConvEnabled)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dw.implicit_narrow",
                            TunedKernelOrigin.Builtin, true, shape => HasTunedKernel("conv2d_weight_grad_implicit") && NarrowWgradAdmits(shape),
                            (in CudaConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaNarrow(a)) LaunchDwDirect(a); }));
                    if (s_convGemmBackward)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("cublas.conv2d_dw.im2col_gemm",
                            TunedKernelOrigin.Vendor, true, _ => true,
                            (in CudaConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaIm2col(a)) LaunchDwDirect(a); }));
                    // Winograd F(3x3,2x2) weight gradient (16 GEMMs over 2x2 dY tiles, split along the tiles when the
                    // K x C grid is small). Registered after the references: it wins ResNet's 128- and 256-channel
                    // layers and loses where the tile count or the channel count is small.
                    if (candidates.Count > 0)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dw.winograd_f32",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradModule(),
                            (in CudaConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradF32(a)) LaunchDwDirect(a); }));
                    // Winograd F(3x3,4x4): opt-in only (owner decision, #1335), like the F(4x4) input gradient.
                    if (candidates.Count > 0 && CudaDispatchPolicy.ConvWinogradF4)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dw.winograd_f34",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradF43Module(),
                            (in CudaConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradF34(a)) LaunchDwDirect(a); }));
                    if (candidates.Count == 0)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_backward_kernel"), (in CudaConv2DBackwardArgs a) => LaunchDwDirect(a)));
                    _conv2DBackwardFilterSlot = new TunedKernelSlot<CudaConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardFilter,
                        TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_conv2DBackwardFilterSlot);
            }
        }
    }

    internal TunedKernelSlot<CudaConv2DBackwardArgs> Conv2DBackwardDataSlot
    {
        get
        {
            if (_conv2DBackwardDataSlot is { } s) return s;
            lock (_tunedSlotsLock)
            {
                if (_conv2DBackwardDataSlot is null)
                {
                    // Reference first (see the weight-gradient slot): the engine (fastest stride-1 input gradient, but
                    // Cout % 8 == 0 only), then #1323's implicit GEMM, which takes every shape; direct only alone.
                    var candidates = new List<ITunedKernelCandidate<CudaConv2DBackwardArgs>>();
                    if (s_convGemmBackward && s_convEngineBackwardEnv)
                        foreach (int mainloop in ConvEngineMainloopsInOrder())
                        {
                            int ml = mainloop;
                            candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>(
                                "nvrtc.conv2d_dx.engine" + ConvMainloopSuffix(ml), TunedKernelOrigin.Builtin, true,
                                shape => HasTunedKernel("conv2d_dgrad_ce128" + ConvMainloopSuffix(ml)) && shape[4] % ConvEngineTileK == 0,
                                (in CudaConv2DBackwardArgs a) => { if (!Conv2DBackwardInputViaEngine(a, ml)) LaunchDxDirect(a); }));
                        }
                    if (s_convGemmBackward && s_convImplicitDgradEnv)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dx.implicit",
                            TunedKernelOrigin.Builtin, true, _ => HasTunedKernel("conv2d_dgrad_implicit_gemm"),
                            (in CudaConv2DBackwardArgs a) => { if (!Conv2DBackwardInputViaImplicit(a)) LaunchDxDirect(a); }));
                    if (s_convGemmBackward)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("cublas.conv2d_dx.gemm_col2im",
                            TunedKernelOrigin.Vendor, true, _ => true,
                            (in CudaConv2DBackwardArgs a) => { if (!Conv2DBackwardInputViaCol2im(a)) LaunchDxDirect(a); }));
                    // Winograd F(2x2,3x3) on the flipped, channel-transposed filter (cuDNN's input-gradient algorithm for
                    // ResNet's mid layers). Registered after the references: it loses where the tile count is small.
                    if (candidates.Count > 0)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dx.winograd_f23",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradModule(),
                            (in CudaConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinogradF23(a)) LaunchDxDirect(a); }));
                    // The same F(2x2) arithmetic fused into one kernel after the filter transform (cuDNN's input gradient for
                    // ResNet's 32x32 and 16x16 layers): no V/M round trips through global memory.
                    if (candidates.Count > 0)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dx.winograd_f23_fused",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradFusedModule(),
                            (in CudaConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinogradF23Fused(a)) LaunchDxDirect(a); }));
                    // Winograd F(4x4,3x3): registered ONLY under the explicit reduced-precision opt-in (owner decision,
                    // #1335): ~20x the fp32 rounding error of F(2x2). Unregistered otherwise, so the persisted-decision key
                    // differs between the two settings.
                    if (candidates.Count > 0 && CudaDispatchPolicy.ConvWinogradF4)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dx.winograd_f43",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradF43Module(),
                            (in CudaConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinogradF43(a)) LaunchDxDirect(a); }));
                    if (candidates.Count == 0)
                        candidates.Add(new CudaTunedKernelCandidate<CudaConv2DBackwardArgs>("nvrtc.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_backward_input"), (in CudaConv2DBackwardArgs a) => LaunchDxDirect(a)));
                    _conv2DBackwardDataSlot = new TunedKernelSlot<CudaConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardData,
                        TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_conv2DBackwardDataSlot);
            }
        }
    }

    private CudaTunedKernelHarness<CudaConv2DBackwardArgs> BackwardHarness() =>
        new(this, ConvolutionTolerance,
            a => new[] { (a.Output, a.OutputCount) },
            a => a.Output.Handle == a.First.Handle || a.Output.Handle == a.Second.Handle);

    // The shapes #1274's conv2d_weight_grad_implicit takes (it also needs the thread's cuBLAS workspace at dispatch).
    private static bool NarrowWgradAdmits(TunedShape s)
    {
        long rows = (long)s[0] * s[13] * s[14];
        long reduction = (long)s[1] * s[5] * s[6];
        return ImplicitConvEnabled && s[0] > 0 && s[13] > 0 && s[14] > 0 && s[4] > 0 && s[4] <= 256
            && reduction >= 32 && reduction <= 4096 && rows >= 4096 && rows <= int.MaxValue
            && s[7] >= 1 && s[8] >= 1 && s[11] >= 1 && s[12] >= 1;
    }

    // Per-route entry points: the slot candidates above, and the tests and benchmark, which run one route regardless of
    // what the slot would pick. False when that route declines the shape; nothing is launched then.
    internal bool Conv2DBackwardKernelViaEngine(in CudaConv2DBackwardArgs a) => Conv2DBackwardKernelViaEngine(a, ConvEngineMainloop);

    internal bool Conv2DBackwardKernelViaEngine(in CudaConv2DBackwardArgs a, int mainloop) =>
        TryConv2DBackwardKernelEngine(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW, mainloop);

    internal bool Conv2DBackwardKernelViaImplicit(in CudaConv2DBackwardArgs a) =>
        TryConv2DBackwardKernelImplicitGemm(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardKernelViaNarrow(in CudaConv2DBackwardArgs a) =>
        TryConv2DBackwardKernelImplicitNarrow(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardKernelViaIm2col(in CudaConv2DBackwardArgs a) =>
        TryConv2DBackwardKernelIm2col(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardInputViaEngine(in CudaConv2DBackwardArgs a) => Conv2DBackwardInputViaEngine(a, ConvEngineMainloop);

    internal bool Conv2DBackwardInputViaEngine(in CudaConv2DBackwardArgs a, int mainloop) =>
        TryConv2DBackwardInputEngine(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW, mainloop);

    internal bool Conv2DBackwardInputViaImplicit(in CudaConv2DBackwardArgs a) =>
        TryConv2DBackwardInputImplicitGemm(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardInputViaCol2im(in CudaConv2DBackwardArgs a) =>
        TryConv2DBackwardInputCol2im(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    private void LaunchDwDirect(in CudaConv2DBackwardArgs a) =>
        LaunchConv2DBackwardKernelDirect(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    private void LaunchDxDirect(in CudaConv2DBackwardArgs a) =>
        LaunchConv2DBackwardInputDirect(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);
}
