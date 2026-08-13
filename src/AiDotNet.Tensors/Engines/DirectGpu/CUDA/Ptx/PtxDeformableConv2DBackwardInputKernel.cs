using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX Deformable Conv2D backward-input (DCNv2 input gradient). Each output
/// position/tap scatters its contribution to the four bilinear input corners:
/// dInput[n,c,yy,xx] += (sum_k gradOut[n,k,oh,ow]*W[k,c,pos]) * mask[n,pos,oh,ow] *
/// corner_weight(yy,xx) at the sample point (py,px). One thread per (n,c,oh,ow) loops
/// over the taps and uses red.global.add.f32 to accumulate into gradInput (which must be
/// zero-initialized), since many (oh,ow,pos) map onto the same input pixel. Bounds-guarded
/// ceil-div grid.
/// </summary>
internal sealed class PtxDeformableConv2DBackwardInputKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int OutputChannels { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal int Taps => KernelH * KernelW;
    internal int TotalThreads => Batch * InputChannels * OutH * OutW;
    internal long WeightBytes => (long)OutputChannels * InputChannels * Taps * sizeof(float);
    internal long OffsetBytes => (long)Batch * 2 * Taps * OutH * OutW * sizeof(float);
    internal long MaskBytes => (long)Batch * Taps * OutH * OutW * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);

    internal DeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding);
    internal static string EntryFor(DeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deformable_conv2d_bwd_input_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Deformable backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");

        DeformableConv2DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, DeformableConv2DShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Height = shape.Height, Width = shape.Width, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, OutH = shape.OutH, OutW = shape.OutW, Taps = shape.Taps;
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        var offset = new DirectPtxExtent(Batch, 2 * Taps, OutH, OutW);
        var mask = new DirectPtxExtent(Batch, Taps, OutH, OutW);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var dx = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dcnv2-fp32"),
            Tensors:
            [
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, dx, dx, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 96, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dInput[n,c,yy,xx] += (sum_k gradOut*W)*mask*corner_weight(yy,xx) scattered to 4 bilinear corners",
                ["accumulate"] = "red.global.add.f32 (gradInput zero-initialized)", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView weights, DirectPtxTensorView offset, DirectPtxTensorView mask, DirectPtxTensorView gradOutput, DirectPtxTensorView gradInput)
    {
        DirectPtxAbiGuard.Require(weights, Blueprint.Tensors[0], nameof(weights));
        DirectPtxAbiGuard.Require(offset, Blueprint.Tensors[1], nameof(offset));
        DirectPtxAbiGuard.Require(mask, Blueprint.Tensors[2], nameof(mask));
        DirectPtxAbiGuard.Require(gradOutput, Blueprint.Tensors[3], nameof(gradOutput));
        DirectPtxAbiGuard.Require(gradInput, Blueprint.Tensors[4], nameof(gradInput));
        IntPtr wPtr = weights.Pointer, offPtr = offset.Pointer, mPtr = mask.Pointer, gPtr = gradOutput.Pointer, xPtr = gradInput.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &wPtr; arguments[1] = &offPtr; arguments[2] = &mPtr; arguments[3] = &gPtr; arguments[4] = &xPtr;
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, DeformableConv2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 Deformable backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * taps;
        int offN = 2 * taps * ohow, maskN = taps * ohow, total = shape.Batch * c * ohow;
        string entry = EntryFor(shape);

        var s = new StringBuilder(40960);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 offset_ptr,");
        s.AppendLine("    .param .u64 mask_ptr,");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 dx_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    .reg .f32 %f<40>;");
        s.AppendLine("    ld.param.u64 %rd0, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [offset_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [mask_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd4, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*C*OHOW + c*OHOW + sp
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(c * ohow)};");  // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(c * ohow)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");      // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");      // sp
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");       // ow
        s.AppendLine($"    mad.lo.u32 %r9, %r3, {I(c)}, %r5;");
        s.AppendLine($"    mul.lo.u32 %r9, %r9, {I(hw)};");     // input channel base (n,c)
        s.AppendLine($"    mad.lo.u32 %r10, %r3, {I(kohow)}, %r6;");  // gradOut[n][0][sp]
        s.AppendLine($"    mad.lo.u32 %r11, %r3, {I(offN)}, %r6;");   // offset[n][0][sp]
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(maskN)}, %r6;");  // mask[n][0][sp]
        s.AppendLine($"    mul.lo.u32 %r13, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");  // oh_base
        s.AppendLine($"    mul.lo.u32 %r14, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r14, %r14, {I(Padding)};");  // ow_base
        s.AppendLine("    mov.u32 %r15, 0;");                   // pos
        s.AppendLine("LOOP_TAP:");
        s.AppendLine($"    div.u32 %r16, %r15, {I(kw)};");      // kh
        s.AppendLine($"    rem.u32 %r17, %r15, {I(kw)};");      // kw
        // offY/offX
        s.AppendLine($"    mad.lo.u32 %r18, %r15, {I(2 * ohow)}, %r11;");
        s.AppendLine("    mul.wide.u32 %rd5, %r18, 4;");
        s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");     // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd5+{I(ohow * 4)}];");  // offX
        // mask
        s.AppendLine($"    mad.lo.u32 %r19, %r15, {I(ohow)}, %r12;");
        s.AppendLine("    mul.wide.u32 %rd6, %r19, 4;");
        s.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");     // mask
        // topgrad = sum_k gradOut[n][k][sp]*W[k][c][pos]
        s.AppendLine("    mov.f32 %f4, 0f00000000;");
        s.AppendLine($"    mad.lo.u32 %r20, %r5, {I(taps)}, %r15;");  // c*taps + pos
        s.AppendLine("    mov.u32 %r21, 0;");                  // kk
        s.AppendLine("LOOP_K:");
        s.AppendLine($"    mad.lo.u32 %r22, %r21, {I(ohow)}, %r10;");
        s.AppendLine("    mul.wide.u32 %rd7, %r22, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    ld.global.nc.f32 %f5, [%rd7];");    // gradOut
        s.AppendLine($"    mad.lo.u32 %r23, %r21, {I(ckk)}, %r20;");
        s.AppendLine("    mul.wide.u32 %rd8, %r23, 4;");
        s.AppendLine("    add.u64 %rd8, %rd0, %rd8;");
        s.AppendLine("    ld.global.nc.f32 %f6, [%rd8];");    // W
        s.AppendLine("    fma.rn.f32 %f4, %f5, %f6, %f4;");
        s.AppendLine("    add.u32 %r21, %r21, 1;");
        s.AppendLine($"    setp.lt.u32 %p1, %r21, {I(k)};");
        s.AppendLine("    @%p1 bra LOOP_K;");
        s.AppendLine("    mul.rn.f32 %f7, %f4, %f3;");        // contrib = topgrad*mask
        // py/px, corner weights
        s.AppendLine($"    mul.lo.u32 %r24, %r7, {I(Stride)};");  // reuse oh -> already r13 is oh_base
        s.AppendLine("    add.s32 %r24, %r13, %r16;");        // oh_base + kh
        s.AppendLine("    cvt.rn.f32.s32 %f8, %r24;");
        s.AppendLine("    add.rn.f32 %f8, %f8, %f1;");        // py
        s.AppendLine("    add.s32 %r25, %r14, %r17;");        // ow_base + kw
        s.AppendLine("    cvt.rn.f32.s32 %f9, %r25;");
        s.AppendLine("    add.rn.f32 %f9, %f9, %f2;");        // px
        s.AppendLine("    cvt.rmi.f32.f32 %f10, %f8;");
        s.AppendLine("    cvt.rmi.f32.f32 %f11, %f9;");
        s.AppendLine("    cvt.rmi.s32.f32 %r26, %f8;");       // y0
        s.AppendLine("    cvt.rmi.s32.f32 %r27, %f9;");       // x0
        s.AppendLine("    sub.rn.f32 %f12, %f8, %f10;");      // wy1
        s.AppendLine("    sub.rn.f32 %f13, %f9, %f11;");      // wx1
        s.AppendLine("    sub.rn.f32 %f14, 0f3F800000, %f12;");// wy0
        s.AppendLine("    sub.rn.f32 %f15, 0f3F800000, %f13;");// wx0
        s.AppendLine("    mul.rn.f32 %f16, %f14, %f15;");     // w00
        s.AppendLine("    mul.rn.f32 %f17, %f14, %f13;");     // w01
        s.AppendLine("    mul.rn.f32 %f18, %f12, %f15;");     // w10
        s.AppendLine("    mul.rn.f32 %f19, %f12, %f13;");     // w11
        s.AppendLine("    add.s32 %r28, %r26, 1;");           // y1
        s.AppendLine("    add.s32 %r29, %r27, 1;");           // x1
        void Scatter(string yy, string xx, string wReg)
        {
            s.AppendLine($"    setp.ge.s32 %p2, {yy}, 0;");
            s.AppendLine($"    setp.lt.s32 %p3, {yy}, {I(h)};");
            s.AppendLine($"    setp.ge.s32 %p4, {xx}, 0;");
            s.AppendLine($"    setp.lt.s32 %p5, {xx}, {I(w)};");
            s.AppendLine("    and.pred %p2, %p2, %p3;");
            s.AppendLine("    and.pred %p4, %p4, %p5;");
            s.AppendLine("    and.pred %p2, %p2, %p4;");
            s.AppendLine($"    @!%p2 bra SKIP_{wReg.Substring(2)};");
            s.AppendLine($"    mul.rn.f32 %f20, %f7, {wReg};");   // contrib * corner weight
            s.AppendLine($"    mad.lo.u32 %r30, {yy}, {I(w)}, %r9;");
            s.AppendLine($"    add.u32 %r30, %r30, {xx};");
            s.AppendLine("    mul.wide.u32 %rd9, %r30, 4;");
            s.AppendLine("    add.u64 %rd9, %rd4, %rd9;");
            s.AppendLine("    red.global.add.f32 [%rd9], %f20;");
            s.AppendLine($"SKIP_{wReg.Substring(2)}:");
        }
        Scatter("%r26", "%r27", "%f16");
        Scatter("%r26", "%r29", "%f17");
        Scatter("%r28", "%r27", "%f18");
        Scatter("%r28", "%r29", "%f19");
        s.AppendLine("    add.u32 %r15, %r15, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r15, {I(taps)};");
        s.AppendLine("    @%p6 bra LOOP_TAP;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
