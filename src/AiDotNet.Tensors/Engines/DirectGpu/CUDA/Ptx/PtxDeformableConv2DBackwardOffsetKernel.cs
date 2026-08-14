using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX Deformable Conv2D backward-offset (DCNv2 offset gradient). For each
/// (n, pos, oh, ow) it writes both the y- and x-offset gradients:
/// dOff[n,2*pos,oh,ow]   = mask * sum_{c,k} gradOut*W * (dval_c/dpy),
/// dOff[n,2*pos+1,oh,ow] = mask * sum_{c,k} gradOut*W * (dval_c/dpx),
/// where the bilinear derivatives are dval/dpy = wx0*(v10-v00)+wx1*(v11-v01) and
/// dval/dpx = wy0*(v01-v00)+wy1*(v11-v10) over the four zero-padded input corners v**.
/// One thread per (n,pos,oh,ow); consecutive threads own consecutive ow -> coalesced.
/// </summary>
internal sealed class PtxDeformableConv2DBackwardOffsetKernel : IDisposable
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
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * Taps * sizeof(float);
    internal long OffsetBytes => (long)Batch * 2 * Taps * OutH * OutW * sizeof(float);
    internal long MaskBytes => (long)Batch * Taps * OutH * OutW * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);

    internal DeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding);
    internal static string EntryFor(DeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deformable_conv2d_bwd_offset_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DBackwardOffsetKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Deformable backward-offset has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * Taps * OutH * OutW % BlockThreads != 0)
            throw new ArgumentException($"N*(KH*KW)*OH*OW must be a multiple of {BlockThreads}.");

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
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        var offset = new DirectPtxExtent(Batch, 2 * Taps, OutH, OutW);
        var mask = new DirectPtxExtent(Batch, Taps, OutH, OutW);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var doff = new DirectPtxExtent(Batch, 2 * Taps, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-backward-offset", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dcnv2-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOffset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, doff, doff, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 112, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dOff_y = mask*sum gradOut*W*(wx0*(v10-v00)+wx1*(v11-v01)); dOff_x = mask*sum gradOut*W*(wy0*(v01-v00)+wy1*(v11-v10))",
                ["sampling"] = "bilinear derivative", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView offset, DirectPtxTensorView mask, DirectPtxTensorView gradOutput, DirectPtxTensorView gradOffset)
    {
        DirectPtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbiGuard.Require(weights, Blueprint.Tensors[1], nameof(weights));
        DirectPtxAbiGuard.Require(offset, Blueprint.Tensors[2], nameof(offset));
        DirectPtxAbiGuard.Require(mask, Blueprint.Tensors[3], nameof(mask));
        DirectPtxAbiGuard.Require(gradOutput, Blueprint.Tensors[4], nameof(gradOutput));
        DirectPtxAbiGuard.Require(gradOffset, Blueprint.Tensors[5], nameof(gradOffset));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, offPtr = offset.Pointer, mPtr = mask.Pointer, gPtr = gradOutput.Pointer, dPtr = gradOffset.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &offPtr; arguments[3] = &mPtr; arguments[4] = &gPtr; arguments[5] = &dPtr;
        int total = Batch * Taps * OutH * OutW;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, DeformableConv2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 Deformable backward-offset emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * taps;
        int offN = 2 * taps * ohow, maskN = taps * ohow;
        string entry = EntryFor(shape);

        var s = new StringBuilder(40960);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 offset_ptr,");
        s.AppendLine("    .param .u64 mask_ptr,");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 doff_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    .reg .f32 %f<48>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [offset_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [mask_ptr];");
        s.AppendLine("    ld.param.u64 %rd4, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd5, [doff_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*taps*OHOW + pos*OHOW + sp
        s.AppendLine($"    div.u32 %r3, %r2, {I(maskN)};");     // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(maskN)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");      // pos
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");      // sp
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");       // ow
        s.AppendLine($"    div.u32 %r9, %r5, {I(kw)};");        // kh
        s.AppendLine($"    rem.u32 %r10, %r5, {I(kw)};");       // kw
        // offY/offX at [n][2pos or +1][sp] ; offset channel base = n*offN + 2*pos*OHOW
        s.AppendLine($"    mad.lo.u32 %r11, %r3, {I(offN)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r11, %r5, {I(2 * ohow)}, %r11;");  // + 2*pos*OHOW
        s.AppendLine("    mul.wide.u32 %rd6, %r11, 4;");
        s.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");     // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd6+{I(ohow * 4)}];");  // offX
        s.AppendLine($"    mul.lo.u32 %r12, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r12, %r12, {I(Padding)};");
        s.AppendLine("    add.s32 %r12, %r12, %r9;");
        s.AppendLine("    cvt.rn.f32.s32 %f3, %r12;");
        s.AppendLine("    add.rn.f32 %f3, %f3, %f1;");         // py
        s.AppendLine($"    mul.lo.u32 %r13, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");
        s.AppendLine("    add.s32 %r13, %r13, %r10;");
        s.AppendLine("    cvt.rn.f32.s32 %f4, %r13;");
        s.AppendLine("    add.rn.f32 %f4, %f4, %f2;");         // px
        s.AppendLine("    cvt.rmi.f32.f32 %f5, %f3;");
        s.AppendLine("    cvt.rmi.f32.f32 %f6, %f4;");
        s.AppendLine("    cvt.rmi.s32.f32 %r14, %f3;");        // y0
        s.AppendLine("    cvt.rmi.s32.f32 %r15, %f4;");        // x0
        s.AppendLine("    sub.rn.f32 %f7, %f3, %f5;");         // wy1
        s.AppendLine("    sub.rn.f32 %f8, %f4, %f6;");         // wx1
        s.AppendLine("    sub.rn.f32 %f9, 0f3F800000, %f7;");  // wy0
        s.AppendLine("    sub.rn.f32 %f10, 0f3F800000, %f8;"); // wx0
        s.AppendLine("    add.s32 %r16, %r14, 1;");            // y1
        s.AppendLine("    add.s32 %r17, %r15, 1;");            // x1
        // gradOut base (n) and input batch base
        s.AppendLine($"    mad.lo.u32 %r18, %r3, {I(kohow)}, %r6;");
        s.AppendLine($"    mul.lo.u32 %r19, %r3, {I(chw)};");
        s.AppendLine("    mov.f32 %f0, 0f00000000;");          // accY
        s.AppendLine("    mov.f32 %f30, 0f00000000;");         // accX
        s.AppendLine("    mov.u32 %r20, 0;");                  // cc
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r21, %r20, {I(hw)}, %r19;");  // input channel base
        // load 4 corners v00 %f11? reuse regs: v00=%f20, v01=%f21, v10=%f22, v11=%f23
        void CornerVal(string yy, string xx, string vReg)
        {
            s.AppendLine($"    setp.ge.s32 %p0, {yy}, 0;");
            s.AppendLine($"    setp.lt.s32 %p1, {yy}, {I(h)};");
            s.AppendLine($"    setp.ge.s32 %p2, {xx}, 0;");
            s.AppendLine($"    setp.lt.s32 %p3, {xx}, {I(w)};");
            s.AppendLine("    and.pred %p0, %p0, %p1;");
            s.AppendLine("    and.pred %p2, %p2, %p3;");
            s.AppendLine("    and.pred %p0, %p0, %p2;");
            s.AppendLine($"    mad.lo.u32 %r22, {yy}, {I(w)}, %r21;");
            s.AppendLine($"    add.u32 %r22, %r22, {xx};");
            s.AppendLine("    mul.wide.u32 %rd7, %r22, 4;");
            s.AppendLine("    add.u64 %rd7, %rd0, %rd7;");
            s.AppendLine($"    mov.f32 {vReg}, 0f00000000;");
            s.AppendLine($"    @%p0 ld.global.nc.f32 {vReg}, [%rd7];");
        }
        CornerVal("%r14", "%r15", "%f20");   // v00
        CornerVal("%r14", "%r17", "%f21");   // v01
        CornerVal("%r16", "%r15", "%f22");   // v10
        CornerVal("%r16", "%r17", "%f23");   // v11
        // dvaly = wx0*(v10-v00) + wx1*(v11-v01)
        s.AppendLine("    sub.rn.f32 %f24, %f22, %f20;");
        s.AppendLine("    sub.rn.f32 %f25, %f23, %f21;");
        s.AppendLine("    mul.rn.f32 %f26, %f10, %f24;");
        s.AppendLine("    fma.rn.f32 %f26, %f8, %f25, %f26;");   // dvaly
        // dvalx = wy0*(v01-v00) + wy1*(v11-v10)
        s.AppendLine("    sub.rn.f32 %f27, %f21, %f20;");
        s.AppendLine("    sub.rn.f32 %f28, %f23, %f22;");
        s.AppendLine("    mul.rn.f32 %f29, %f9, %f27;");
        s.AppendLine("    fma.rn.f32 %f29, %f7, %f28, %f29;");   // dvalx
        // gk = sum_k gradOut[n][k][sp]*W[k][cc][pos]
        s.AppendLine("    mov.f32 %f31, 0f00000000;");         // gk
        s.AppendLine($"    mad.lo.u32 %r23, %r20, {I(taps)}, %r5;");  // cc*taps + pos
        s.AppendLine("    mov.u32 %r24, 0;");                  // kk
        s.AppendLine("LOOP_K:");
        s.AppendLine($"    mad.lo.u32 %r25, %r24, {I(ohow)}, %r18;");
        s.AppendLine("    mul.wide.u32 %rd8, %r25, 4;");
        s.AppendLine("    add.u64 %rd8, %rd4, %rd8;");
        s.AppendLine("    ld.global.nc.f32 %f32, [%rd8];");   // gradOut
        s.AppendLine($"    mad.lo.u32 %r26, %r24, {I(ckk)}, %r23;");
        s.AppendLine("    mul.wide.u32 %rd9, %r26, 4;");
        s.AppendLine("    add.u64 %rd9, %rd1, %rd9;");
        s.AppendLine("    ld.global.nc.f32 %f33, [%rd9];");   // W
        s.AppendLine("    fma.rn.f32 %f31, %f32, %f33, %f31;");
        s.AppendLine("    add.u32 %r24, %r24, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r24, {I(k)};");
        s.AppendLine("    @%p4 bra LOOP_K;");
        s.AppendLine("    fma.rn.f32 %f0, %f31, %f26, %f0;");   // accY += gk*dvaly
        s.AppendLine("    fma.rn.f32 %f30, %f31, %f29, %f30;"); // accX += gk*dvalx
        s.AppendLine("    add.u32 %r20, %r20, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r20, {I(c)};");
        s.AppendLine("    @%p4 bra LOOP_C;");
        // mask
        s.AppendLine($"    mad.lo.u32 %r27, %r3, {I(maskN)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r27, %r5, {I(ohow)}, %r27;");
        s.AppendLine("    mul.wide.u32 %rd10, %r27, 4;");
        s.AppendLine("    add.u64 %rd10, %rd3, %rd10;");
        s.AppendLine("    ld.global.nc.f32 %f34, [%rd10];");  // mask
        s.AppendLine("    mul.rn.f32 %f0, %f0, %f34;");        // dOffY
        s.AppendLine("    mul.rn.f32 %f30, %f30, %f34;");      // dOffX
        // write dOff[n][2pos][sp] = accY, dOff[n][2pos+1][sp] = accX ; base = r11 (offset channel base, same layout)
        s.AppendLine("    mul.wide.u32 %rd11, %r11, 4;");
        s.AppendLine("    add.u64 %rd11, %rd5, %rd11;");
        s.AppendLine("    st.global.f32 [%rd11], %f0;");
        s.AppendLine($"    st.global.f32 [%rd11+{I(ohow * 4)}], %f30;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
