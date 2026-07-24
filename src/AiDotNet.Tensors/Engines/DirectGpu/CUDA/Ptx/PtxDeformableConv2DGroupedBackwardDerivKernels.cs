using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX grouped Deformable Conv2D backward-offset (deform_groups > 1). One thread per
/// (n, g, pos, oh, ow) writes both offset gradients into dOff[n, g*2*taps + 2*pos (+1), oh, ow]:
/// dOff_y = mask_g * sum_{c in group g, k} gradOut*W * (wx0*(v10-v00)+wx1*(v11-v01)),
/// dOff_x = mask_g * sum_{c in group g, k} gradOut*W * (wy0*(v01-v00)+wy1*(v11-v10)),
/// where the four corners v** are the zero-padded samples of input[n,c] and the reduction runs
/// only over the deformable group's channels (c in [g*C/dg, (g+1)*C/dg)). Bounds-guarded grid.
/// </summary>
internal sealed class PtxDeformableConv2DGroupedBackwardOffsetKernel : IDisposable
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
    internal int DeformGroups { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal int Taps => KernelH * KernelW;
    internal int ChannelsPerGroup => InputChannels / DeformGroups;
    internal int TotalThreads => Batch * DeformGroups * Taps * OutH * OutW;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * Taps * sizeof(float);
    internal long OffsetBytes => (long)Batch * DeformGroups * 2 * Taps * OutH * OutW * sizeof(float);
    internal long MaskBytes => (long)Batch * DeformGroups * Taps * OutH * OutW * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradOffsetBytes => OffsetBytes;

    internal GroupedDeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, DeformGroups);
    internal static string EntryFor(GroupedDeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deform_grouped_bwd_offset_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}_dg{s.DeformGroups}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DGroupedBackwardOffsetKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int deformGroups)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Grouped Deformable backward-offset has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || deformGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (inputChannels % deformGroups != 0) throw new ArgumentException("InputChannels must be divisible by deformGroups.");
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; DeformGroups = deformGroups;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");

        GroupedDeformableConv2DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, GroupedDeformableConv2DShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Height = shape.Height, Width = shape.Width, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, DeformGroups = shape.DeformGroups;
        int OutH = shape.OutH, OutW = shape.OutW, Taps = shape.Taps, ChannelsPerGroup = shape.ChannelsPerGroup;
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        var offset = new DirectPtxExtent(Batch, DeformGroups * 2 * Taps, OutH, OutW);
        var mask = new DirectPtxExtent(Batch, DeformGroups * Taps, OutH, OutW);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var doff = new DirectPtxExtent(Batch, DeformGroups * 2 * Taps, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped-backward-offset", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dg{DeformGroups}-dcnv2-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOffset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, doff, doff, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 120, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dOff_y/x[n,g,pos,oh,ow] = mask_g*sum_{c in g,k} gradOut*W*bilinear-deriv; g selects offset/mask field",
                ["sampling"] = "bilinear derivative, group-restricted channel reduction", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView offset, DirectPtxTensorView mask, DirectPtxTensorView gradOutput, DirectPtxTensorView gradOffset)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(offset, Blueprint.Tensors[2], nameof(offset));
        Require(mask, Blueprint.Tensors[3], nameof(mask));
        Require(gradOutput, Blueprint.Tensors[4], nameof(gradOutput));
        Require(gradOffset, Blueprint.Tensors[5], nameof(gradOffset));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, offPtr = offset.Pointer, mPtr = mask.Pointer, gPtr = gradOutput.Pointer, dPtr = gradOffset.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &offPtr; arguments[3] = &mPtr; arguments[4] = &gPtr; arguments[5] = &dPtr;
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, GroupedDeformableConv2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 grouped Deformable backward-offset emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH, DeformGroups = shape.DeformGroups;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * taps;
        int maskNg = DeformGroups * taps * ohow, offNg = DeformGroups * 2 * taps * ohow;
        int offGroup = 2 * taps * ohow, maskGroup = taps * ohow, cpg = shape.ChannelsPerGroup, total = shape.Batch * taps * ohow;
        string entry = EntryFor(shape);

        var s = new StringBuilder(45056);
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
        s.AppendLine("    .reg .b32 %r<56>;");
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
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*(dg*taps*OHOW) + gp*OHOW + sp
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(maskNg)};");    // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(maskNg)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");      // gp = g*taps + pos
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");      // sp
        s.AppendLine($"    div.u32 %r30, %r5, {I(taps)};");     // g
        s.AppendLine($"    rem.u32 %r31, %r5, {I(taps)};");     // pos
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");       // ow
        s.AppendLine($"    div.u32 %r9, %r31, {I(kw)};");       // kh
        s.AppendLine($"    rem.u32 %r10, %r31, {I(kw)};");      // kw
        // offset element base r11 = n*offNg + sp + g*offGroup + 2*pos*ohow
        s.AppendLine($"    mad.lo.u32 %r11, %r3, {I(offNg)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r11, %r30, {I(offGroup)}, %r11;");
        s.AppendLine($"    mad.lo.u32 %r11, %r31, {I(2 * ohow)}, %r11;");
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
        s.AppendLine($"    mad.lo.u32 %r18, %r3, {I(kohow)}, %r6;");  // gradOut[n][0][sp]
        s.AppendLine($"    mul.lo.u32 %r19, %r3, {I(chw)};");   // input batch base
        s.AppendLine("    mov.f32 %f0, 0f00000000;");          // accY
        s.AppendLine("    mov.f32 %f30, 0f00000000;");         // accX
        s.AppendLine($"    mul.lo.u32 %r20, %r30, {I(cpg)};"); // cstart = g*cpg
        s.AppendLine($"    add.u32 %r32, %r20, {I(cpg)};");    // cend = cstart + cpg
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r21, %r20, {I(hw)}, %r19;");  // input channel base
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
        s.AppendLine("    sub.rn.f32 %f24, %f22, %f20;");
        s.AppendLine("    sub.rn.f32 %f25, %f23, %f21;");
        s.AppendLine("    mul.rn.f32 %f26, %f10, %f24;");
        s.AppendLine("    fma.rn.f32 %f26, %f8, %f25, %f26;");   // dvaly
        s.AppendLine("    sub.rn.f32 %f27, %f21, %f20;");
        s.AppendLine("    sub.rn.f32 %f28, %f23, %f22;");
        s.AppendLine("    mul.rn.f32 %f29, %f9, %f27;");
        s.AppendLine("    fma.rn.f32 %f29, %f7, %f28, %f29;");   // dvalx
        s.AppendLine("    mov.f32 %f31, 0f00000000;");          // gk
        s.AppendLine($"    mad.lo.u32 %r23, %r20, {I(taps)}, %r31;");  // cc*taps + pos
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
        s.AppendLine("    setp.lt.u32 %p4, %r20, %r32;");
        s.AppendLine("    @%p4 bra LOOP_C;");
        // mask at [n][g][pos][sp] = n*maskNg + g*maskGroup + pos*ohow + sp
        s.AppendLine($"    mad.lo.u32 %r27, %r3, {I(maskNg)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r27, %r30, {I(maskGroup)}, %r27;");
        s.AppendLine($"    mad.lo.u32 %r27, %r31, {I(ohow)}, %r27;");
        s.AppendLine("    mul.wide.u32 %rd10, %r27, 4;");
        s.AppendLine("    add.u64 %rd10, %rd3, %rd10;");
        s.AppendLine("    ld.global.nc.f32 %f34, [%rd10];");  // mask
        s.AppendLine("    mul.rn.f32 %f0, %f0, %f34;");        // dOffY
        s.AppendLine("    mul.rn.f32 %f30, %f30, %f34;");      // dOffX
        s.AppendLine("    mul.wide.u32 %rd11, %r11, 4;");
        s.AppendLine("    add.u64 %rd11, %rd5, %rd11;");
        s.AppendLine("    st.global.f32 [%rd11], %f0;");
        s.AppendLine($"    st.global.f32 [%rd11+{I(ohow * 4)}], %f30;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Direct-PTX grouped Deformable Conv2D backward-mask (deform_groups > 1):
/// dMask[n,g,pos,oh,ow] = sum_{c in group g, k} gradOut[n,k,oh,ow]*W[k,c,pos]*bilinear(input[n,c]; py_g, px_g),
/// with the channel reduction restricted to the deformable group's channels. One thread per grouped
/// mask element; consecutive ow -> coalesced. Zero-padded 4-corner bilinear. Bounds-guarded grid.
/// </summary>
internal sealed class PtxDeformableConv2DGroupedBackwardMaskKernel : IDisposable
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
    internal int DeformGroups { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal int Taps => KernelH * KernelW;
    internal int ChannelsPerGroup => InputChannels / DeformGroups;
    internal int TotalThreads => Batch * DeformGroups * Taps * OutH * OutW;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * Taps * sizeof(float);
    internal long OffsetBytes => (long)Batch * DeformGroups * 2 * Taps * OutH * OutW * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradMaskBytes => (long)Batch * DeformGroups * Taps * OutH * OutW * sizeof(float);

    internal GroupedDeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, DeformGroups);
    internal static string EntryFor(GroupedDeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deform_grouped_bwd_mask_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}_dg{s.DeformGroups}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DGroupedBackwardMaskKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int deformGroups)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Grouped Deformable backward-mask has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || deformGroups <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (inputChannels % deformGroups != 0) throw new ArgumentException("InputChannels must be divisible by deformGroups.");
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; DeformGroups = deformGroups;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");

        GroupedDeformableConv2DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, GroupedDeformableConv2DShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Height = shape.Height, Width = shape.Width, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, DeformGroups = shape.DeformGroups;
        int OutH = shape.OutH, OutW = shape.OutW, Taps = shape.Taps, ChannelsPerGroup = shape.ChannelsPerGroup;
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        var offset = new DirectPtxExtent(Batch, DeformGroups * 2 * Taps, OutH, OutW);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var dmask = new DirectPtxExtent(Batch, DeformGroups * Taps, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped-backward-mask", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dg{DeformGroups}-dcnv2-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradMask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, dmask, dmask, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 96, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dMask[n,g,pos,oh,ow] = sum_{c in g,k} gradOut*W*bilinear(input[n,c]; py_g, px_g)",
                ["sampling"] = "zero-padded 4-corner bilinear, group-restricted", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView offset, DirectPtxTensorView gradOutput, DirectPtxTensorView gradMask)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(offset, Blueprint.Tensors[2], nameof(offset));
        Require(gradOutput, Blueprint.Tensors[3], nameof(gradOutput));
        Require(gradMask, Blueprint.Tensors[4], nameof(gradMask));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, offPtr = offset.Pointer, gPtr = gradOutput.Pointer, mPtr = gradMask.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &offPtr; arguments[3] = &gPtr; arguments[4] = &mPtr;
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, GroupedDeformableConv2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 grouped Deformable backward-mask emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH, DeformGroups = shape.DeformGroups;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * taps;
        int maskNg = DeformGroups * taps * ohow, offNg = DeformGroups * 2 * taps * ohow;
        int offGroup = 2 * taps * ohow, cpg = shape.ChannelsPerGroup, total = shape.Batch * taps * ohow;
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
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 dmask_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<52>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    .reg .f32 %f<40>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [offset_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd4, [dmask_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(maskNg)};");    // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(maskNg)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");      // gp = g*taps+pos
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");      // sp
        s.AppendLine($"    div.u32 %r28, %r5, {I(taps)};");     // g
        s.AppendLine($"    rem.u32 %r29, %r5, {I(taps)};");     // pos
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");       // ow
        s.AppendLine($"    div.u32 %r9, %r29, {I(kw)};");       // kh
        s.AppendLine($"    rem.u32 %r10, %r29, {I(kw)};");      // kw
        // offset base r11 = n*offNg + sp + g*offGroup + 2*pos*ohow
        s.AppendLine($"    mad.lo.u32 %r11, %r3, {I(offNg)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r11, %r28, {I(offGroup)}, %r11;");
        s.AppendLine($"    mad.lo.u32 %r11, %r29, {I(2 * ohow)}, %r11;");
        s.AppendLine("    mul.wide.u32 %rd5, %r11, 4;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");     // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd5+{I(ohow * 4)}];");  // offX
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
        s.AppendLine("    mul.rn.f32 %f11, %f9, %f10;");       // w00
        s.AppendLine("    mul.rn.f32 %f12, %f9, %f8;");        // w01
        s.AppendLine("    mul.rn.f32 %f13, %f7, %f10;");       // w10
        s.AppendLine("    mul.rn.f32 %f14, %f7, %f8;");        // w11
        s.AppendLine("    add.s32 %r16, %r14, 1;");            // y1
        s.AppendLine("    add.s32 %r17, %r15, 1;");            // x1
        s.AppendLine($"    mad.lo.u32 %r18, %r3, {I(kohow)}, %r6;");
        s.AppendLine($"    mul.lo.u32 %r19, %r3, {I(chw)};");
        s.AppendLine("    mov.f32 %f0, 0f00000000;");          // acc
        s.AppendLine($"    mul.lo.u32 %r20, %r28, {I(cpg)};"); // cstart
        s.AppendLine($"    add.u32 %r30, %r20, {I(cpg)};");    // cend
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r21, %r20, {I(hw)}, %r19;");
        s.AppendLine("    mov.f32 %f20, 0f00000000;");
        void Corner(string yy, string xx, string wReg)
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
            s.AppendLine("    mul.wide.u32 %rd6, %r22, 4;");
            s.AppendLine("    add.u64 %rd6, %rd0, %rd6;");
            s.AppendLine("    mov.f32 %f21, 0f00000000;");
            s.AppendLine("    @%p0 ld.global.nc.f32 %f21, [%rd6];");
            s.AppendLine($"    fma.rn.f32 %f20, %f21, {wReg}, %f20;");
        }
        Corner("%r14", "%r15", "%f11");
        Corner("%r14", "%r17", "%f12");
        Corner("%r16", "%r15", "%f13");
        Corner("%r16", "%r17", "%f14");
        s.AppendLine("    mov.f32 %f22, 0f00000000;");        // gk
        s.AppendLine($"    mad.lo.u32 %r23, %r20, {I(taps)}, %r29;");
        s.AppendLine("    mov.u32 %r24, 0;");
        s.AppendLine("LOOP_K:");
        s.AppendLine($"    mad.lo.u32 %r25, %r24, {I(ohow)}, %r18;");
        s.AppendLine("    mul.wide.u32 %rd7, %r25, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    ld.global.nc.f32 %f23, [%rd7];");   // gradOut
        s.AppendLine($"    mad.lo.u32 %r26, %r24, {I(ckk)}, %r23;");
        s.AppendLine("    mul.wide.u32 %rd8, %r26, 4;");
        s.AppendLine("    add.u64 %rd8, %rd1, %rd8;");
        s.AppendLine("    ld.global.nc.f32 %f24, [%rd8];");   // W
        s.AppendLine("    fma.rn.f32 %f22, %f23, %f24, %f22;");
        s.AppendLine("    add.u32 %r24, %r24, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r24, {I(k)};");
        s.AppendLine("    @%p4 bra LOOP_K;");
        s.AppendLine("    fma.rn.f32 %f0, %f22, %f20, %f0;");
        s.AppendLine("    add.u32 %r20, %r20, 1;");
        s.AppendLine("    setp.lt.u32 %p4, %r20, %r30;");
        s.AppendLine("    @%p4 bra LOOP_C;");
        s.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        s.AppendLine("    add.u64 %rd9, %rd4, %rd9;");
        s.AppendLine("    st.global.f32 [%rd9], %f0;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
