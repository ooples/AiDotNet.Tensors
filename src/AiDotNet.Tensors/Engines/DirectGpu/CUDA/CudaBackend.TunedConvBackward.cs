using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

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
    private TunedKernelSlot<TunedConv2DBackwardArgs>? _conv2DBackwardFilterSlot, _conv2DBackwardDataSlot;

    private static readonly bool s_convImplicitWgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_IMPLICIT_WGRAD") != "0";
    private static readonly bool s_convImplicitDgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_IMPLICIT_DGRAD") != "0";
    private static readonly bool s_convEngineBackwardEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_ENGINE_BACKWARD") != "0";
    // Operator opt-out for the r8 weight-gradient candidates alone, leaving the other GEMM backward routes registered.
    private static readonly bool s_convR8WgradEnv =
        Environment.GetEnvironmentVariable("AIDOTNET_CONV_R8_WGRAD") != "0";

    internal TunedKernelSlot<TunedConv2DBackwardArgs> Conv2DBackwardFilterSlot
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
                    var candidates = new List<ITunedKernelCandidate<TunedConv2DBackwardArgs>>();
                    if (s_convGemmBackward && s_convImplicitWgradEnv)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dw.implicit",
                            TunedKernelOrigin.Builtin, true, _ => HasTunedKernel("conv2d_wgrad_implicit_gemm"),
                            (in TunedConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaImplicit(a)) LaunchDwDirect(a); }));
                    if (s_convGemmBackward && s_convEngineBackwardEnv)
                        foreach (int mainloop in ConvEngineMainloopsInOrder())
                        {
                            int ml = mainloop;
                            candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>(
                                "nvrtc.conv2d_dw.engine" + ConvMainloopSuffix(ml), TunedKernelOrigin.Builtin, true,
                                _ => HasTunedKernel("conv2d_wgrad_ce128" + ConvMainloopSuffix(ml)),
                                (in TunedConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaEngine(a, ml)) LaunchDwDirect(a); }));
                        }
                    if (s_convGemmBackward && ImplicitConvEnabled)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dw.implicit_narrow",
                            TunedKernelOrigin.Builtin, true, shape => HasTunedKernel("conv2d_weight_grad_implicit") && NarrowWgradAdmits(shape),
                            (in TunedConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaNarrow(a)) LaunchDwDirect(a); }));
                    if (s_convGemmBackward)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("cublas.conv2d_dw.im2col_gemm",
                            TunedKernelOrigin.Vendor, true, _ => true,
                            (in TunedConv2DBackwardArgs a) => { if (!Conv2DBackwardKernelViaIm2col(a)) LaunchDwDirect(a); }));
                    // 8x8-micro-tile weight gradient (CudaConvWgradKernels): the engine tiles spent 54-60% of their issued
                    // instructions on per-element index arithmetic (Nsight Compute). Per call on an RTX 3080 (batch 8) vs the
                    // implicit kernel: 64@32 81.0 -> 59.1 us (64x128), 64->128 s2 48.2 -> 36.9, 128->256 s2 45.8 -> 37.9,
                    // 256->512 s2 42.7 -> 32.7, 512@4 69.3 -> 58.2 us; 1x1 and stem layers stay on the existing kernels.
                    if (s_convGemmBackward && s_convR8WgradEnv)
                        foreach (var (tileM, tileN) in Kernels.CudaConvWgradKernels.TileShapes)
                        {
                            int tm = tileM, tn = tileN;
                            candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>(
                                $"nvrtc.conv2d_dw.r8_{tm}x{tn}", TunedKernelOrigin.Builtin, true,
                                _ => EnsureWgradR8Module(),
                                (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardKernelR8(a, tm, tn, 0)) LaunchDwDirect(a); }));
                        }
                    // Winograd F(3x3,2x2) weight gradient (16 GEMMs over 2x2 dY tiles, split along the tiles when the
                    // K x C grid is small). Registered after the references: it wins ResNet's 128- and 256-channel
                    // layers and loses where the tile count or the channel count is small.
                    if (candidates.Count > 0)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dw.winograd_f32",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradModule(),
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradF32(a)) LaunchDwDirect(a); }));
                    // The same Winograd weight gradient fused into one launch (transforms and output transform in the
                    // GEMM's block, no U/V/M planes in memory). Faster than the four-launch pipeline on every ResNet-18
                    // 3x3 layer measured, and than r8v on the 64-channel 32x32 layers.
                    if (candidates.Count > 0)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dw.winograd_fused",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradModule(),
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradFused(a)) LaunchDwDirect(a); }));
                    // Winograd F(3x3,4x4): opt-in only (owner decision, #1335), like the F(4x4) input gradient.
                    if (candidates.Count > 0 && CudaDispatchPolicy.ConvWinogradF4)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dw.winograd_f34",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradF43Module(),
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardFilterWinogradF34(a)) LaunchDwDirect(a); }));
                    if (candidates.Count == 0)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dw.direct", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_backward_kernel"), (in TunedConv2DBackwardArgs a) => LaunchDwDirect(a)));
                    _conv2DBackwardFilterSlot = new TunedKernelSlot<TunedConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardFilter,
                        TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_conv2DBackwardFilterSlot);
            }
        }
    }

    internal TunedKernelSlot<TunedConv2DBackwardArgs> Conv2DBackwardDataSlot
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
                    var candidates = new List<ITunedKernelCandidate<TunedConv2DBackwardArgs>>();
                    if (s_convGemmBackward && s_convEngineBackwardEnv)
                        foreach (int mainloop in ConvEngineMainloopsInOrder())
                        {
                            int ml = mainloop;
                            candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>(
                                "nvrtc.conv2d_dx.engine" + ConvMainloopSuffix(ml), TunedKernelOrigin.Builtin, true,
                                shape => HasTunedKernel("conv2d_dgrad_ce128" + ConvMainloopSuffix(ml)) && shape[4] % ConvEngineTileK == 0,
                                (in TunedConv2DBackwardArgs a) => { if (!Conv2DBackwardInputViaEngine(a, ml)) LaunchDxDirect(a); }));
                        }
                    if (s_convGemmBackward && s_convImplicitDgradEnv)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dx.implicit",
                            TunedKernelOrigin.Builtin, true, _ => HasTunedKernel("conv2d_dgrad_implicit_gemm"),
                            (in TunedConv2DBackwardArgs a) => { if (!Conv2DBackwardInputViaImplicit(a)) LaunchDxDirect(a); }));
                    if (s_convGemmBackward)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("cublas.conv2d_dx.gemm_col2im",
                            TunedKernelOrigin.Vendor, true, _ => true,
                            (in TunedConv2DBackwardArgs a) => { if (!Conv2DBackwardInputViaCol2im(a)) LaunchDxDirect(a); }));
                    // Winograd F(2x2,3x3) on the flipped, channel-transposed filter (cuDNN's input-gradient algorithm for
                    // ResNet's mid layers). Registered after the references: it loses where the tile count is small.
                    if (candidates.Count > 0)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dx.winograd_f23",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradModule(),
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinogradF23(a)) LaunchDxDirect(a); }));
                    // The same F(2x2) arithmetic fused into one kernel after the filter transform (cuDNN's input gradient for
                    // ResNet's 32x32 and 16x16 layers): no V/M round trips through global memory.
                    if (candidates.Count > 0)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dx.winograd_f23_fused",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradFusedModule(),
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinogradF23Fused(a)) LaunchDxDirect(a); }));
                    // Winograd F(4x4,3x3): registered ONLY under the explicit reduced-precision opt-in (owner decision,
                    // #1335): ~20x the fp32 rounding error of F(2x2). Unregistered otherwise, so the persisted-decision key
                    // differs between the two settings.
                    if (candidates.Count > 0 && CudaDispatchPolicy.ConvWinogradF4)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dx.winograd_f43",
                            TunedKernelOrigin.Builtin, true, shape => WinogradF23AdmitsShape(shape) && EnsureWinogradF43Module(),
                            (in TunedConv2DBackwardArgs a) => { if (!TryConv2DBackwardDataWinogradF43(a)) LaunchDxDirect(a); }));
                    if (candidates.Count == 0)
                        candidates.Add(new CudaTunedKernelCandidate<TunedConv2DBackwardArgs>("nvrtc.conv2d_dx.direct", TunedKernelOrigin.Builtin,
                            true, _ => HasTunedKernel("conv2d_backward_input"), (in TunedConv2DBackwardArgs a) => LaunchDxDirect(a)));
                    _conv2DBackwardDataSlot = new TunedKernelSlot<TunedConv2DBackwardArgs>(TunedKernelOp.Conv2DBackwardData,
                        TunedDeviceKey, BackwardHarness(), () => GpuDeterminism.IsActive, candidates.ToArray());
                }
                return Created(_conv2DBackwardDataSlot);
            }
        }
    }

    private CudaTunedKernelHarness<TunedConv2DBackwardArgs> BackwardHarness() =>
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
    internal bool Conv2DBackwardKernelViaEngine(in TunedConv2DBackwardArgs a) => Conv2DBackwardKernelViaEngine(a, ConvEngineMainloop);

    internal bool Conv2DBackwardKernelViaEngine(in TunedConv2DBackwardArgs a, int mainloop) =>
        TryConv2DBackwardKernelEngine(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW, mainloop);

    internal bool Conv2DBackwardKernelViaImplicit(in TunedConv2DBackwardArgs a) =>
        TryConv2DBackwardKernelImplicitGemm(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardKernelViaNarrow(in TunedConv2DBackwardArgs a) =>
        TryConv2DBackwardKernelImplicitNarrow(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardKernelViaIm2col(in TunedConv2DBackwardArgs a) =>
        TryConv2DBackwardKernelIm2col(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardInputViaEngine(in TunedConv2DBackwardArgs a) => Conv2DBackwardInputViaEngine(a, ConvEngineMainloop);

    internal bool Conv2DBackwardInputViaEngine(in TunedConv2DBackwardArgs a, int mainloop) =>
        TryConv2DBackwardInputEngine(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW, mainloop);

    internal bool Conv2DBackwardInputViaImplicit(in TunedConv2DBackwardArgs a) =>
        TryConv2DBackwardInputImplicitGemm(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    internal bool Conv2DBackwardInputViaCol2im(in TunedConv2DBackwardArgs a) =>
        TryConv2DBackwardInputCol2im(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    private void LaunchDwDirect(in TunedConv2DBackwardArgs a) =>
        LaunchConv2DBackwardKernelDirect(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);

    private void LaunchDxDirect(in TunedConv2DBackwardArgs a) =>
        LaunchConv2DBackwardInputDirect(a.First, a.Second, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
            a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
            a.DilationH, a.DilationW);
}
