// Copyright (c) CrowdTrainer LLC. All rights reserved.
// Depthwise and transposed convolutions on declared programs (#890 phase 4b). Both are the dense convolution's spec
// under another reading, so they lower through the same builder (BuildDeclaredConv) and share its program cache:
//
//   - depthwise is the grouped convolution with one group per input channel (CodegenKernelCatalog.Conv2D, groups =
//     inC): the hand-written kernels' layouts are the grouped spec's -- output channel ic * M + m, weights
//     [inC, M, kh, kw] -- so the same buffers bind unchanged;
//   - a transposed convolution with weights [inC, outC, kh, kw] is the adjoint of the convolution from outC to inC
//     channels over the transposed output's extent, whose weights have the same bytes: its forward is that
//     convolution's data gradient, its data gradient that convolution's forward, and its weight gradient that
//     convolution's weight gradient with the operands' roles exchanged.
//
// Each of the six operators dispatches through its own tuned slot whose reference is the hand-written kernel the entry
// point always ran (the *Incumbent methods); a declared schedule serves a shape only where the slot measured it
// correct and faster, and the decision persists.

using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune.TunedKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>A depthwise or transposed convolution operator.</summary>
internal enum DeclaredConvVariant : byte
{
    DepthwiseForward,
    DepthwiseBackwardData,
    DepthwiseBackwardWeights,
    TransposeForward,
    TransposeBackwardData,
    TransposeBackwardWeights,
}

/// <summary>
/// A depthwise or transposed convolution as its entry point received it: A, B and Output are the entry point's three
/// buffers in its parameter order; the geometry is the entry point's (for depthwise, OutChannels = InChannels *
/// Multiplier).
/// </summary>
internal readonly struct CudaConvVariantArgs
{
    internal CudaConvVariantArgs(DeclaredConvVariant variant, IGpuBuffer a, IGpuBuffer b, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int outputPadH = 0, int outputPadW = 0,
        int multiplier = 1)
    {
        Variant = variant; A = a; B = b; Output = output;
        Batch = batch; InChannels = inChannels; InHeight = inHeight; InWidth = inWidth;
        OutChannels = outChannels; OutHeight = outHeight; OutWidth = outWidth;
        KernelH = kernelH; KernelW = kernelW; StrideH = strideH; StrideW = strideW; PadH = padH; PadW = padW;
        OutputPadH = outputPadH; OutputPadW = outputPadW; Multiplier = multiplier;
    }

    internal DeclaredConvVariant Variant { get; }
    internal IGpuBuffer A { get; }
    internal IGpuBuffer B { get; }
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
    internal int OutputPadH { get; }
    internal int OutputPadW { get; }
    internal int Multiplier { get; }

    /// <summary>Elements the operator writes.</summary>
    internal int OutputCount => Variant switch
    {
        DeclaredConvVariant.DepthwiseForward or DeclaredConvVariant.TransposeForward =>
            Batch * OutChannels * OutHeight * OutWidth,
        DeclaredConvVariant.DepthwiseBackwardData or DeclaredConvVariant.TransposeBackwardData =>
            Batch * InChannels * InHeight * InWidth,
        _ => InChannels * (Variant == DeclaredConvVariant.DepthwiseBackwardWeights ? Multiplier : OutChannels) *
             KernelH * KernelW,
    };

    /// <summary>Shape class: every geometric parameter.</summary>
    internal unsafe TunedShape Shape()
    {
        int* d = stackalloc int[16];
        d[0] = Batch; d[1] = InChannels; d[2] = InHeight; d[3] = InWidth; d[4] = OutChannels; d[5] = OutHeight;
        d[6] = OutWidth; d[7] = KernelH; d[8] = KernelW; d[9] = StrideH; d[10] = StrideW; d[11] = PadH; d[12] = PadW;
        d[13] = OutputPadH; d[14] = OutputPadW; d[15] = Multiplier;
        return TunedShape.Create(TunedKernelDType.Float32, new ReadOnlySpan<int>(d, 16));
    }
}

/// <summary>The convolution a variant is computed as: its pass and geometry (dilation 1).</summary>
internal readonly record struct DeclaredConvEquivalent(DeclaredConvPass Pass, int Batch, int InChannels,
    int OutChannels, int Height, int Width, int Kernel, int Stride, int Pad, int OutHeight, int OutWidth, int Groups);

public sealed partial class CudaBackend
{
    // The variants' schedules: only those that were fastest for some measured shape (`--kernel-routes --family
    // conv-variants`, GTX 1660 Ti), since every registered candidate costs online tuning time. Depthwise takes the
    // affine emitter's own lowering -- its kh*kw-long reduction leaves no GEMM worth tiling (the implicit-GEMM tiles ran
    // 0.07-0.26x of the hand-written kernels; the affine lowering 1.01-3.65x) -- split for the weight gradient's long
    // reduction (8x144 at 56: 1818 against 2494 us unsplit). The transposed passes take the implicit-GEMM tiles alone:
    // the parity split and the reduction splits never beat them (forward 6472 us at m128n64 against 8195 parity).
    private static readonly DeclaredConvSchedule[] s_depthwiseSchedules = { new(0, 0, 0) };

    private static readonly DeclaredConvSchedule[] s_depthwiseWeightSchedules = { new(0, 0, 0), new(0, 0, 4) };

    private static readonly DeclaredConvSchedule[] s_transposeSchedules = { new(64, 64, 0), new(128, 64, 0) };

    internal static IReadOnlyList<DeclaredConvSchedule> DeclaredConvVariantSchedules(DeclaredConvVariant variant) =>
        variant switch
        {
            DeclaredConvVariant.DepthwiseForward or DeclaredConvVariant.DepthwiseBackwardData => s_depthwiseSchedules,
            DeclaredConvVariant.DepthwiseBackwardWeights => s_depthwiseWeightSchedules,
            _ => s_transposeSchedules,
        };

    private static string VariantName(DeclaredConvVariant variant) => variant switch
    {
        DeclaredConvVariant.DepthwiseForward => "depthwise_conv2d",
        DeclaredConvVariant.DepthwiseBackwardData => "depthwise_conv2d_bwd_data",
        DeclaredConvVariant.DepthwiseBackwardWeights => "depthwise_conv2d_bwd_weights",
        DeclaredConvVariant.TransposeForward => "conv_transpose2d",
        DeclaredConvVariant.TransposeBackwardData => "conv_transpose2d_bwd_data",
        _ => "conv_transpose2d_bwd_weights",
    };

    private static TunedKernelOp VariantOp(DeclaredConvVariant variant) => variant switch
    {
        DeclaredConvVariant.DepthwiseForward => TunedKernelOp.DepthwiseConv2DForward,
        DeclaredConvVariant.DepthwiseBackwardData => TunedKernelOp.DepthwiseConv2DBackwardData,
        DeclaredConvVariant.DepthwiseBackwardWeights => TunedKernelOp.DepthwiseConv2DBackwardFilter,
        DeclaredConvVariant.TransposeForward => TunedKernelOp.ConvTranspose2DForward,
        DeclaredConvVariant.TransposeBackwardData => TunedKernelOp.ConvTranspose2DBackwardData,
        _ => TunedKernelOp.ConvTranspose2DBackwardFilter,
    };

    /// <summary>
    /// The convolution pass and geometry a variant of shape class <paramref name="s"/> (CudaConvVariantArgs.Shape) is
    /// computed as, or null when it has none (a non-square window, stride or padding).
    /// </summary>
    internal static DeclaredConvEquivalent? ConvVariantEquivalent(DeclaredConvVariant variant, in TunedShape s)
    {
        int batch = s[0], inC = s[1], inH = s[2], inW = s[3], outC = s[4], outH = s[5], outW = s[6];
        int kernelH = s[7], kernelW = s[8], strideH = s[9], strideW = s[10], padH = s[11], padW = s[12];
        int multiplier = s[15];
        if (kernelH != kernelW || strideH != strideW || padH != padW) return null;
        switch (variant)
        {
            case DeclaredConvVariant.DepthwiseForward:
            case DeclaredConvVariant.DepthwiseBackwardData:
            case DeclaredConvVariant.DepthwiseBackwardWeights:
                var pass = variant == DeclaredConvVariant.DepthwiseForward ? DeclaredConvPass.Forward
                    : variant == DeclaredConvVariant.DepthwiseBackwardData ? DeclaredConvPass.BackwardData
                    : DeclaredConvPass.BackwardWeights;
                return new DeclaredConvEquivalent(pass, batch, inC, inC * multiplier, inH, inW, kernelH, strideH, padH,
                    outH, outW, inC);
            default:
                // The convolution from the transposed output (outC channels, outH x outW) to its input.
                var tpass = variant == DeclaredConvVariant.TransposeForward ? DeclaredConvPass.BackwardData
                    : variant == DeclaredConvVariant.TransposeBackwardData ? DeclaredConvPass.Forward
                    : DeclaredConvPass.BackwardWeights;
                return new DeclaredConvEquivalent(tpass, batch, outC, inC, outH, outW, kernelH, strideH, padH, inH, inW, 1);
        }
    }

    // Whether a schedule can lower a variant's shape class (the slot asks before it measures a candidate).
    internal bool DeclaredConvVariantAdmits(DeclaredConvVariant variant, in TunedShape shape,
        DeclaredConvSchedule schedule)
        => DeclaredKernelsEnabled && DeclaredConvVariantLowers(variant, shape, schedule);

    // Whether the schedule lowers the shape at all, on any device: the admission test without the release gate.
    internal static bool DeclaredConvVariantLowers(DeclaredConvVariant variant, in TunedShape shape,
        DeclaredConvSchedule schedule)
    {
        if (ConvVariantEquivalent(variant, shape) is not { } e) return false;
        var spec = DeclaredConvSpec(e.Pass, e.Batch, e.InChannels, e.OutChannels, e.Height, e.Width, e.Kernel,
            e.Kernel, e.Stride, e.Stride, e.Pad, e.Pad, 1, 1, e.OutHeight, e.OutWidth, biasRelu: false, e.Groups);
        return spec is not null && DeclaredConvSpecAdmits(spec, schedule);
    }

    /// <summary>Runs a depthwise or transposed convolution as a declared program; false when the shape declines.</summary>
    internal bool TryDeclaredConvVariant(in CudaConvVariantArgs a, DeclaredConvSchedule schedule)
    {
        if (ConvVariantEquivalent(a.Variant, a.Shape()) is not { } e) return false;
        if (!DeclaredConvGeometryFits(e.Batch, e.InChannels, e.OutChannels, e.Height, e.Width, e.Kernel, e.Kernel,
                e.Stride, e.Stride, e.Pad, e.Pad, 1, 1, e.OutHeight, e.OutWidth))
            return false;
        var key = new DeclaredConvKey(e.Pass, e.Batch, e.InChannels, e.OutChannels, e.Height, e.Width, e.Kernel,
            e.Stride, e.Pad, 1, BiasRelu: false, schedule, e.Groups);
        // The convolution's operands in its own roles: a transposed forward's input is its data gradient's dOut, and
        // a transposed weight gradient's input and output gradient trade places.
        return e.Pass switch
        {
            DeclaredConvPass.Forward =>
                TryDeclaredConv(key, s_convForwardNames, new DeclaredBindings(a.A, a.B, a.Output)),
            DeclaredConvPass.BackwardData =>
                TryDeclaredConv(key, s_convBackwardDataNames, new DeclaredBindings(a.A, a.B, a.Output)),
            _ => TryDeclaredConv(key, s_convBackwardWeightNames, new DeclaredBindings(a.A, a.B, a.Output)),
        };
    }

    private readonly TunedKernelSlot<CudaConvVariantArgs>?[] _convVariantSlots = new TunedKernelSlot<CudaConvVariantArgs>?[6];

    internal TunedKernelSlot<CudaConvVariantArgs> ConvVariantSlot(DeclaredConvVariant variant)
    {
        if (_convVariantSlots[(int)variant] is { } s) return s;
        lock (_tunedSlotsLock)
        {
            if (_convVariantSlots[(int)variant] is null)
            {
                string name = VariantName(variant);
                var candidates = new List<ITunedKernelCandidate<CudaConvVariantArgs>>
                {
                    new CudaTunedKernelCandidate<CudaConvVariantArgs>("backend." + name + ".incumbent",
                        TunedKernelOrigin.Builtin, true, _ => true, (in CudaConvVariantArgs a) => RunConvVariantIncumbent(a)),
                };
                foreach (var schedule in DeclaredConvVariantSchedules(variant))
                {
                    var sc = schedule;
                    candidates.Add(new CudaTunedKernelCandidate<CudaConvVariantArgs>("ptx." + name + "." + sc.Suffix,
                        TunedKernelOrigin.Generated, true, shape => DeclaredConvVariantAdmits(variant, shape, sc),
                        (in CudaConvVariantArgs a) =>
                        {
                            // Not yet built under a capture (a persisted decision met first there): the incumbent.
                            if (!TryDeclaredConvVariant(a, sc)) RunConvVariantIncumbent(a);
                        }));
                }
                _convVariantSlots[(int)variant] = new TunedKernelSlot<CudaConvVariantArgs>(VariantOp(variant),
                    TunedDeviceKey,
                    new CudaTunedKernelHarness<CudaConvVariantArgs>(this, ConvolutionTolerance,
                        a => new[] { (a.Output, a.OutputCount) },
                        a => a.Output.Handle == a.A.Handle || a.Output.Handle == a.B.Handle),
                    () => GpuDeterminism.IsActive, candidates.ToArray());
            }
            return Created(_convVariantSlots[(int)variant] ?? throw new InvalidOperationException("Slot not created."));
        }
    }

    // The slot when the declared route is on; false (nothing launched) otherwise.
    private bool TryConvVariantSlot(in CudaConvVariantArgs a)
    {
        if (!s_declaredKernels || !DeclaredKernelsEnabled || a.Batch <= 0 || a.OutputCount <= 0) return false;
        return ConvVariantSlot(a.Variant).TryExecute(a.Shape(), a);
    }

    private void RunConvVariantIncumbent(in CudaConvVariantArgs a)
    {
        switch (a.Variant)
        {
            case DeclaredConvVariant.DepthwiseForward:
                DepthwiseConv2DIncumbent(a.A, a.B, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                    a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW);
                break;
            case DeclaredConvVariant.DepthwiseBackwardData:
                DepthwiseConv2DBackwardInputIncumbent(a.A, a.B, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                    a.Multiplier, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW);
                break;
            case DeclaredConvVariant.DepthwiseBackwardWeights:
                DepthwiseConv2DBackwardKernelIncumbent(a.A, a.B, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                    a.Multiplier, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW);
                break;
            case DeclaredConvVariant.TransposeForward:
                ConvTranspose2DIncumbent(a.A, a.B, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                    a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
                    a.OutputPadH, a.OutputPadW);
                break;
            case DeclaredConvVariant.TransposeBackwardData:
                ConvTranspose2DBackwardInputIncumbent(a.A, a.B, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                    a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
                    a.OutputPadH, a.OutputPadW);
                break;
            default:
                ConvTranspose2DBackwardKernelIncumbent(a.A, a.B, a.Output, a.Batch, a.InChannels, a.InHeight, a.InWidth,
                    a.OutChannels, a.OutHeight, a.OutWidth, a.KernelH, a.KernelW, a.StrideH, a.StrideW, a.PadH, a.PadW,
                    a.OutputPadH, a.OutputPadW);
                break;
        }
    }

    public void DepthwiseConv2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int channels, int inHeight, int inWidth, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW)
    {
        var args = new CudaConvVariantArgs(DeclaredConvVariant.DepthwiseForward, input, kernel, output, batch, channels,
            inHeight, inWidth, channels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW);
        if (!TryConvVariantSlot(args)) RunConvVariantIncumbent(args);
    }

    public void DepthwiseConv2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW)
    {
        var args = new CudaConvVariantArgs(DeclaredConvVariant.DepthwiseBackwardData, gradOutput, kernel, gradInput, n,
            inC, h, w, inC * m, outH, outW, kH, kW, strideH, strideW, padH, padW, multiplier: m);
        if (!TryConvVariantSlot(args)) RunConvVariantIncumbent(args);
    }

    public void DepthwiseConv2DBackwardKernel(IGpuBuffer gradOutput, IGpuBuffer input, IGpuBuffer gradKernel,
        int n, int inC, int h, int w, int m, int outH, int outW, int kH, int kW,
        int strideH, int strideW, int padH, int padW)
    {
        var args = new CudaConvVariantArgs(DeclaredConvVariant.DepthwiseBackwardWeights, gradOutput, input, gradKernel,
            n, inC, h, w, inC * m, outH, outW, kH, kW, strideH, strideW, padH, padW, multiplier: m);
        if (!TryConvVariantSlot(args)) RunConvVariantIncumbent(args);
    }

    public void ConvTranspose2D(IGpuBuffer input, IGpuBuffer kernel, IGpuBuffer output,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int outputPadH, int outputPadW)
    {
        var args = new CudaConvVariantArgs(DeclaredConvVariant.TransposeForward, input, kernel, output, batch, inChannels,
            inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW, padH, padW,
            outputPadH, outputPadW);
        if (!TryConvVariantSlot(args)) RunConvVariantIncumbent(args);
    }

    public void ConvTranspose2DBackwardInput(IGpuBuffer gradOutput, IGpuBuffer kernel, IGpuBuffer gradInput,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int outputPadH, int outputPadW)
    {
        var args = new CudaConvVariantArgs(DeclaredConvVariant.TransposeBackwardData, gradOutput, kernel, gradInput,
            batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW,
            padH, padW, outputPadH, outputPadW);
        if (!TryConvVariantSlot(args)) RunConvVariantIncumbent(args);
    }

    public void ConvTranspose2DBackwardKernel(IGpuBuffer input, IGpuBuffer gradOutput, IGpuBuffer gradKernel,
        int batch, int inChannels, int inHeight, int inWidth, int outChannels, int outHeight, int outWidth,
        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW, int outputPadH, int outputPadW)
    {
        var args = new CudaConvVariantArgs(DeclaredConvVariant.TransposeBackwardWeights, input, gradOutput, gradKernel,
            batch, inChannels, inHeight, inWidth, outChannels, outHeight, outWidth, kernelH, kernelW, strideH, strideW,
            padH, padW, outputPadH, outputPadW);
        if (!TryConvVariantSlot(args)) RunConvVariantIncumbent(args);
    }
}
