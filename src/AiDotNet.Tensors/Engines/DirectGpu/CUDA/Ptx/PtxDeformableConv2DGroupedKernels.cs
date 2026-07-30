using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX grouped Deformable Conv2D forward (DCNv2 with deform_groups > 1). Input
/// channels are partitioned into <c>dg</c> deformable groups; each group carries its own
/// offset/mask field. For input channel c the group is g = c/(C/dg), and the sample point
/// uses offset[n, g*2*taps + 2*pos (+1), oh, ow] and mask[n, g*taps + pos, oh, ow]:
/// out[n,k,oh,ow] = bias[k] + sum_{c,pos} W[k,c,pos]*mask_g*bilinear(input[n,c]; py_g, px_g).
/// Zero-padded 4-corner bilinear. One thread per output element; consecutive ow -> coalesced
/// offset/mask reads. Bounds-guarded ceil-div grid. dg=1 reproduces the single-group kernel.
/// </summary>
/// <summary>Shared geometry for the grouped DCNv2 (deform_groups) kernels (device-free re-emit).</summary>
internal readonly record struct GroupedDeformableConv2DShape(
    int Batch, int InputChannels, int OutputChannels, int Height, int Width,
    int KernelH, int KernelW, int Stride, int Padding, int DeformGroups)
{
    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal int Taps => KernelH * KernelW;
    internal int ChannelsPerGroup => InputChannels / DeformGroups;
}

internal sealed class PtxDeformableConv2DGroupedForwardKernel : IDisposable
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
    internal int TotalThreads => Batch * OutputChannels * OutH * OutW;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * Taps * sizeof(float);
    internal long OffsetBytes => (long)Batch * DeformGroups * 2 * Taps * OutH * OutW * sizeof(float);
    internal long MaskBytes => (long)Batch * DeformGroups * Taps * OutH * OutW * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);

    internal GroupedDeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, DeformGroups);
    internal static string EntryFor(GroupedDeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deform_grouped_fwd_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}_dg{s.DeformGroups}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DGroupedForwardKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int deformGroups)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Grouped Deformable forward has no experimental non-SM86 specialization.");
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
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dg{DeformGroups}-dcnv2-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 112, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,oh,ow] = bias[k] + sum W[k,c,pos]*mask[g]*bilinear(input[n,c]; py_g, px_g), g=c/(C/dg)",
                ["sampling"] = "zero-padded 4-corner bilinear", ["variant"] = "DCNv2 deform_groups>1",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView offset, DirectPtxTensorView mask, DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(offset, Blueprint.Tensors[2], nameof(offset));
        Require(mask, Blueprint.Tensors[3], nameof(mask));
        Require(bias, Blueprint.Tensors[4], nameof(bias));
        Require(output, Blueprint.Tensors[5], nameof(output));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, offPtr = offset.Pointer, mPtr = mask.Pointer, bPtr = bias.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &offPtr; arguments[3] = &mPtr; arguments[4] = &bPtr; arguments[5] = &oPtr;
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
            throw new NotSupportedException("Only the experimental SM86 grouped Deformable forward emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH, DeformGroups = shape.DeformGroups;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * taps;
        int offN = DeformGroups * 2 * taps * ohow, maskN = DeformGroups * taps * ohow;
        int offGroup = 2 * taps * ohow, maskGroup = taps * ohow, cpg = shape.ChannelsPerGroup, total = shape.Batch * k * ohow;
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
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<56>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    .reg .f32 %f<32>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [offset_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [mask_ptr];");
        s.AppendLine("    ld.param.u64 %rd4, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd5, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(kohow)};");     // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(kohow)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");      // k
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");      // sp
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");       // ow
        s.AppendLine("    mul.wide.u32 %rd6, %r5, 4;");
        s.AppendLine("    add.u64 %rd6, %rd4, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd6];");     // bias[k]
        s.AppendLine($"    mul.lo.u32 %r9, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r9, %r9, {I(Padding)};");  // oh_base
        s.AppendLine($"    mul.lo.u32 %r10, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r10, %r10, {I(Padding)};");// ow_base
        s.AppendLine($"    mad.lo.u32 %r11, %r3, {I(offN)}, %r6;");   // offset n base + sp
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(maskN)}, %r6;");  // mask n base + sp
        s.AppendLine($"    mul.lo.u32 %r13, %r3, {I(chw)};");   // input batch base
        s.AppendLine($"    mul.lo.u32 %r14, %r5, {I(ckk)};");   // weight out-channel base
        s.AppendLine("    mov.u32 %r15, 0;");                   // cc
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r16, %r15, {I(hw)}, %r13;");   // input channel base
        s.AppendLine($"    mad.lo.u32 %r17, %r15, {I(taps)}, %r14;"); // weight (k,cc) base
        s.AppendLine($"    div.u32 %r33, %r15, {I(cpg)};");     // g = cc / cpg
        s.AppendLine($"    mad.lo.u32 %r34, %r33, {I(offGroup)}, %r11;");  // offset group+n+sp base
        s.AppendLine($"    mad.lo.u32 %r35, %r33, {I(maskGroup)}, %r12;"); // mask group+n+sp base
        s.AppendLine("    mov.u32 %r18, 0;");                   // pos
        s.AppendLine("LOOP_TAP:");
        s.AppendLine($"    div.u32 %r19, %r18, {I(kw)};");      // kh
        s.AppendLine($"    rem.u32 %r20, %r18, {I(kw)};");      // kw
        s.AppendLine($"    mad.lo.u32 %r21, %r18, {I(2 * ohow)}, %r34;");
        s.AppendLine("    mul.wide.u32 %rd7, %r21, 4;");
        s.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd7];");     // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd7+{I(ohow * 4)}];"); // offX
        s.AppendLine($"    mad.lo.u32 %r22, %r18, {I(ohow)}, %r35;");
        s.AppendLine("    mul.wide.u32 %rd8, %r22, 4;");
        s.AppendLine("    add.u64 %rd8, %rd3, %rd8;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd8];");     // mask
        s.AppendLine("    add.s32 %r23, %r9, %r19;");
        s.AppendLine("    cvt.rn.f32.s32 %f4, %r23;");
        s.AppendLine("    add.rn.f32 %f4, %f4, %f1;");         // py
        s.AppendLine("    add.s32 %r24, %r10, %r20;");
        s.AppendLine("    cvt.rn.f32.s32 %f5, %r24;");
        s.AppendLine("    add.rn.f32 %f5, %f5, %f2;");         // px
        s.AppendLine("    cvt.rmi.f32.f32 %f6, %f4;");
        s.AppendLine("    cvt.rmi.f32.f32 %f7, %f5;");
        s.AppendLine("    cvt.rmi.s32.f32 %r25, %f4;");        // y0
        s.AppendLine("    cvt.rmi.s32.f32 %r26, %f5;");        // x0
        s.AppendLine("    sub.rn.f32 %f8, %f4, %f6;");         // wy1
        s.AppendLine("    sub.rn.f32 %f9, %f5, %f7;");         // wx1
        s.AppendLine("    sub.rn.f32 %f10, 0f3F800000, %f8;"); // wy0
        s.AppendLine("    sub.rn.f32 %f11, 0f3F800000, %f9;"); // wx0
        s.AppendLine("    mul.rn.f32 %f12, %f10, %f11;");      // w00
        s.AppendLine("    mul.rn.f32 %f13, %f10, %f9;");       // w01
        s.AppendLine("    mul.rn.f32 %f14, %f8, %f11;");       // w10
        s.AppendLine("    mul.rn.f32 %f15, %f8, %f9;");        // w11
        s.AppendLine("    add.s32 %r27, %r25, 1;");            // y1
        s.AppendLine("    add.s32 %r28, %r26, 1;");            // x1
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
            s.AppendLine($"    mad.lo.u32 %r29, {yy}, {I(w)}, %r16;");
            s.AppendLine($"    add.u32 %r29, %r29, {xx};");
            s.AppendLine("    mul.wide.u32 %rd9, %r29, 4;");
            s.AppendLine("    add.u64 %rd9, %rd0, %rd9;");
            s.AppendLine("    mov.f32 %f21, 0f00000000;");
            s.AppendLine("    @%p0 ld.global.nc.f32 %f21, [%rd9];");
            s.AppendLine($"    fma.rn.f32 %f20, %f21, {wReg}, %f20;");
        }
        Corner("%r25", "%r26", "%f12");
        Corner("%r25", "%r28", "%f13");
        Corner("%r27", "%r26", "%f14");
        Corner("%r27", "%r28", "%f15");
        s.AppendLine("    add.u32 %r30, %r17, %r18;");         // weight index
        s.AppendLine("    mul.wide.u32 %rd10, %r30, 4;");
        s.AppendLine("    add.u64 %rd10, %rd1, %rd10;");
        s.AppendLine("    ld.global.nc.f32 %f22, [%rd10];");  // W
        s.AppendLine("    mul.rn.f32 %f23, %f22, %f3;");      // W*mask
        s.AppendLine("    fma.rn.f32 %f0, %f23, %f20, %f0;");
        s.AppendLine("    add.u32 %r18, %r18, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r18, {I(taps)};");
        s.AppendLine("    @%p4 bra LOOP_TAP;");
        s.AppendLine("    add.u32 %r15, %r15, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r15, {I(c)};");
        s.AppendLine("    @%p4 bra LOOP_C;");
        s.AppendLine("    mul.wide.u32 %rd11, %r2, 4;");
        s.AppendLine("    add.u64 %rd11, %rd5, %rd11;");
        s.AppendLine("    st.global.f32 [%rd11], %f0;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Direct-PTX grouped Deformable Conv2D backward-input (deform_groups > 1): atomic scatter of
/// dInput[n,c,yy,xx] += (sum_k gradOut[n,k,oh,ow]*W[k,c,pos])*mask_g*corner_weight to the four
/// bilinear corners, with the deformable group g = c/(C/dg) selecting the offset/mask field.
/// One thread per (n,c,oh,ow); gradInput zero-initialized; red.global.add.f32. Bounds-guarded.
/// </summary>
internal sealed class PtxDeformableConv2DGroupedBackwardInputKernel : IDisposable
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
    internal int TotalThreads => Batch * InputChannels * OutH * OutW;
    internal long WeightBytes => (long)OutputChannels * InputChannels * Taps * sizeof(float);
    internal long OffsetBytes => (long)Batch * DeformGroups * 2 * Taps * OutH * OutW * sizeof(float);
    internal long MaskBytes => (long)Batch * DeformGroups * Taps * OutH * OutW * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);

    internal GroupedDeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, DeformGroups);
    internal static string EntryFor(GroupedDeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deform_grouped_bwd_input_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}_dg{s.DeformGroups}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DGroupedBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int deformGroups)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Grouped Deformable backward-input has no experimental non-SM86 specialization.");
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
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        var offset = new DirectPtxExtent(Batch, DeformGroups * 2 * Taps, OutH, OutW);
        var mask = new DirectPtxExtent(Batch, DeformGroups * Taps, OutH, OutW);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var dx = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dg{DeformGroups}-dcnv2-fp32"),
            Tensors:
            [
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, dx, dx, 16, DirectPtxTensorAccess.ReadWrite, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 112, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dInput[n,c,yy,xx] += (sum_k gradOut*W)*mask[g]*corner_weight scattered; g=c/(C/dg)",
                ["accumulate"] = "red.global.add.f32 (gradInput zero-initialized)", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView weights, DirectPtxTensorView offset, DirectPtxTensorView mask, DirectPtxTensorView gradOutput, DirectPtxTensorView gradInput)
    {
        Require(weights, Blueprint.Tensors[0], nameof(weights));
        Require(offset, Blueprint.Tensors[1], nameof(offset));
        Require(mask, Blueprint.Tensors[2], nameof(mask));
        Require(gradOutput, Blueprint.Tensors[3], nameof(gradOutput));
        Require(gradInput, Blueprint.Tensors[4], nameof(gradInput));
        IntPtr wPtr = weights.Pointer, offPtr = offset.Pointer, mPtr = mask.Pointer, gPtr = gradOutput.Pointer, xPtr = gradInput.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &wPtr; arguments[1] = &offPtr; arguments[2] = &mPtr; arguments[3] = &gPtr; arguments[4] = &xPtr;
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
            throw new NotSupportedException("Only the experimental SM86 grouped Deformable backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH, DeformGroups = shape.DeformGroups;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * taps;
        int offN = DeformGroups * 2 * taps * ohow, maskN = DeformGroups * taps * ohow;
        int offGroup = 2 * taps * ohow, maskGroup = taps * ohow, cpg = shape.ChannelsPerGroup, total = shape.Batch * c * ohow;
        string entry = EntryFor(shape);

        var s = new StringBuilder(45056);
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
        s.AppendLine("    .reg .b32 %r<52>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    .reg .f32 %f<40>;");
        s.AppendLine("    ld.param.u64 %rd0, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [offset_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [mask_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd4, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");
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
        s.AppendLine($"    div.u32 %r36, %r5, {I(cpg)};");      // g = c / cpg
        s.AppendLine($"    mad.lo.u32 %r11, %r3, {I(offN)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r11, %r36, {I(offGroup)}, %r11;");  // offset[n][g,0][sp]
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(maskN)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r12, %r36, {I(maskGroup)}, %r12;"); // mask[n][g,0][sp]
        s.AppendLine($"    mul.lo.u32 %r13, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");
        s.AppendLine($"    mul.lo.u32 %r14, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r14, %r14, {I(Padding)};");
        s.AppendLine("    mov.u32 %r15, 0;");                   // pos
        s.AppendLine("LOOP_TAP:");
        s.AppendLine($"    div.u32 %r16, %r15, {I(kw)};");      // kh
        s.AppendLine($"    rem.u32 %r17, %r15, {I(kw)};");      // kw
        s.AppendLine($"    mad.lo.u32 %r18, %r15, {I(2 * ohow)}, %r11;");
        s.AppendLine("    mul.wide.u32 %rd5, %r18, 4;");
        s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");     // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd5+{I(ohow * 4)}];");  // offX
        s.AppendLine($"    mad.lo.u32 %r19, %r15, {I(ohow)}, %r12;");
        s.AppendLine("    mul.wide.u32 %rd6, %r19, 4;");
        s.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");     // mask
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
            s.AppendLine($"    mul.rn.f32 %f20, %f7, {wReg};");
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

/// <summary>
/// Direct-PTX grouped Deformable Conv2D backward-weight (deform_groups > 1):
/// dW[k,c,pos] = sum_{n,oh,ow} gradOut[n,k,oh,ow]*mask_g*bilinear(input[n,c]; py_g, px_g), with
/// group g = c/(C/dg) selecting the offset/mask field. One thread per weight element loops the
/// batch+output-spatial axis (bounds-guarded grid). Reuses the forward 4-corner bilinear.
/// </summary>
internal sealed class PtxDeformableConv2DGroupedBackwardWeightKernel : IDisposable
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
    internal int TotalWeights => OutputChannels * InputChannels * Taps;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long OffsetBytes => (long)Batch * DeformGroups * 2 * Taps * OutH * OutW * sizeof(float);
    internal long MaskBytes => (long)Batch * DeformGroups * Taps * OutH * OutW * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradWeightBytes => (long)TotalWeights * sizeof(float);

    internal GroupedDeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, DeformGroups);
    internal static string EntryFor(GroupedDeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deform_grouped_bwd_weight_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}_dg{s.DeformGroups}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DGroupedBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int deformGroups)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Grouped Deformable backward-weight has no experimental non-SM86 specialization.");
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
        var offset = new DirectPtxExtent(Batch, DeformGroups * 2 * Taps, OutH, OutW);
        var mask = new DirectPtxExtent(Batch, DeformGroups * Taps, OutH, OutW);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var dw = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dg{DeformGroups}-dcnv2-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 96, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[k,c,pos] = sum_{n,oh,ow} gradOut*mask[g]*bilinear(input[n,c]; py_g, px_g); g=c/(C/dg)",
                ["sampling"] = "zero-padded 4-corner bilinear; bounds-guarded grid", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView offset, DirectPtxTensorView mask, DirectPtxTensorView gradOutput, DirectPtxTensorView gradWeight)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(offset, Blueprint.Tensors[1], nameof(offset));
        Require(mask, Blueprint.Tensors[2], nameof(mask));
        Require(gradOutput, Blueprint.Tensors[3], nameof(gradOutput));
        Require(gradWeight, Blueprint.Tensors[4], nameof(gradWeight));
        IntPtr iPtr = input.Pointer, offPtr = offset.Pointer, mPtr = mask.Pointer, gPtr = gradOutput.Pointer, wPtr = gradWeight.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &iPtr; arguments[1] = &offPtr; arguments[2] = &mPtr; arguments[3] = &gPtr; arguments[4] = &wPtr;
        uint blocks = (uint)((TotalWeights + BlockThreads - 1) / BlockThreads);
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
            throw new NotSupportedException("Only the experimental SM86 grouped Deformable backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH, DeformGroups = shape.DeformGroups, Batch = shape.Batch;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, ohow = ohh * oww;
        int offN = DeformGroups * 2 * taps * ohow, maskN = DeformGroups * taps * ohow;
        int offGroup = 2 * taps * ohow, maskGroup = taps * ohow, cpg = shape.ChannelsPerGroup, total = k * c * taps;
        string entry = EntryFor(shape);

        var s = new StringBuilder(36864);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 offset_ptr,");
        s.AppendLine("    .param .u64 mask_ptr,");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 dw_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<52>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    .reg .f32 %f<40>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [offset_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [mask_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd4, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    rem.u32 %r3, %r2, {I(taps)};");        // pos
        s.AppendLine($"    div.u32 %r4, %r2, {I(taps)};");
        s.AppendLine($"    rem.u32 %r5, %r4, {I(c)};");           // c
        s.AppendLine($"    div.u32 %r6, %r4, {I(c)};");           // k
        s.AppendLine($"    div.u32 %r7, %r3, {I(kw)};");          // kh
        s.AppendLine($"    rem.u32 %r8, %r3, {I(kw)};");          // kw
        s.AppendLine($"    div.u32 %r31, %r5, {I(cpg)};");        // g = c / cpg
        s.AppendLine($"    mul.lo.u32 %r32, %r31, {I(offGroup)};");  // g*offGroup
        s.AppendLine($"    mul.lo.u32 %r33, %r31, {I(maskGroup)};"); // g*maskGroup
        s.AppendLine("    mov.f32 %f0, 0f00000000;");            // acc
        s.AppendLine("    mov.u32 %r9, 0;");                     // n
        s.AppendLine("LOOP_N:");
        s.AppendLine($"    mad.lo.u32 %r10, %r9, {I(c)}, %r5;");
        s.AppendLine($"    mul.lo.u32 %r10, %r10, {I(hw)};");         // input (n,c) base
        s.AppendLine($"    mad.lo.u32 %r11, %r9, {I(k)}, %r6;");
        s.AppendLine($"    mul.lo.u32 %r11, %r11, {I(ohow)};");       // gradOut (n,k) base
        s.AppendLine($"    mul.lo.u32 %r12, %r9, {I(offN)};");
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(2 * ohow)}, %r12;");
        s.AppendLine("    add.u32 %r12, %r12, %r32;");                // + g*offGroup (offY channel base)
        s.AppendLine($"    mul.lo.u32 %r13, %r9, {I(maskN)};");
        s.AppendLine($"    mad.lo.u32 %r13, %r3, {I(ohow)}, %r13;");
        s.AppendLine("    add.u32 %r13, %r13, %r33;");                // + g*maskGroup
        s.AppendLine("    mov.u32 %r14, 0;");                    // oh
        s.AppendLine("LOOP_OH:");
        s.AppendLine("    mov.u32 %r15, 0;");                    // ow
        s.AppendLine("LOOP_OW:");
        s.AppendLine($"    mad.lo.u32 %r16, %r14, {I(oww)}, %r15;");  // sp
        s.AppendLine("    add.u32 %r17, %r12, %r16;");
        s.AppendLine("    mul.wide.u32 %rd5, %r17, 4;");
        s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");      // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd5+{I(ohow * 4)}];"); // offX
        s.AppendLine("    add.u32 %r18, %r13, %r16;");
        s.AppendLine("    mul.wide.u32 %rd6, %r18, 4;");
        s.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");      // mask
        s.AppendLine("    add.u32 %r19, %r11, %r16;");
        s.AppendLine("    mul.wide.u32 %rd7, %r19, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    ld.global.nc.f32 %f4, [%rd7];");      // gradOut
        s.AppendLine($"    mul.lo.u32 %r20, %r14, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r20, %r20, {I(Padding)};");
        s.AppendLine("    add.s32 %r20, %r20, %r7;");
        s.AppendLine("    cvt.rn.f32.s32 %f5, %r20;");
        s.AppendLine("    add.rn.f32 %f5, %f5, %f1;");          // py
        s.AppendLine($"    mul.lo.u32 %r21, %r15, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r21, %r21, {I(Padding)};");
        s.AppendLine("    add.s32 %r21, %r21, %r8;");
        s.AppendLine("    cvt.rn.f32.s32 %f6, %r21;");
        s.AppendLine("    add.rn.f32 %f6, %f6, %f2;");          // px
        s.AppendLine("    cvt.rmi.f32.f32 %f7, %f5;");
        s.AppendLine("    cvt.rmi.f32.f32 %f8, %f6;");
        s.AppendLine("    cvt.rmi.s32.f32 %r22, %f5;");        // y0
        s.AppendLine("    cvt.rmi.s32.f32 %r23, %f6;");        // x0
        s.AppendLine("    sub.rn.f32 %f9, %f5, %f7;");         // wy1
        s.AppendLine("    sub.rn.f32 %f10, %f6, %f8;");        // wx1
        s.AppendLine("    sub.rn.f32 %f11, 0f3F800000, %f9;"); // wy0
        s.AppendLine("    sub.rn.f32 %f12, 0f3F800000, %f10;");// wx0
        s.AppendLine("    mul.rn.f32 %f13, %f11, %f12;");      // w00
        s.AppendLine("    mul.rn.f32 %f14, %f11, %f10;");      // w01
        s.AppendLine("    mul.rn.f32 %f15, %f9, %f12;");       // w10
        s.AppendLine("    mul.rn.f32 %f16, %f9, %f10;");       // w11
        s.AppendLine("    add.s32 %r24, %r22, 1;");            // y1
        s.AppendLine("    add.s32 %r25, %r23, 1;");            // x1
        s.AppendLine("    mov.f32 %f20, 0f00000000;");
        void Corner(string yy, string xx, string wReg)
        {
            s.AppendLine($"    setp.ge.s32 %p1, {yy}, 0;");
            s.AppendLine($"    setp.lt.s32 %p2, {yy}, {I(h)};");
            s.AppendLine($"    setp.ge.s32 %p3, {xx}, 0;");
            s.AppendLine($"    setp.lt.s32 %p4, {xx}, {I(w)};");
            s.AppendLine("    and.pred %p1, %p1, %p2;");
            s.AppendLine("    and.pred %p3, %p3, %p4;");
            s.AppendLine("    and.pred %p1, %p1, %p3;");
            s.AppendLine($"    mad.lo.u32 %r26, {yy}, {I(w)}, %r10;");
            s.AppendLine($"    add.u32 %r26, %r26, {xx};");
            s.AppendLine("    mul.wide.u32 %rd8, %r26, 4;");
            s.AppendLine("    add.u64 %rd8, %rd0, %rd8;");
            s.AppendLine("    mov.f32 %f21, 0f00000000;");
            s.AppendLine("    @%p1 ld.global.nc.f32 %f21, [%rd8];");
            s.AppendLine($"    fma.rn.f32 %f20, %f21, {wReg}, %f20;");
        }
        Corner("%r22", "%r23", "%f13");
        Corner("%r22", "%r25", "%f14");
        Corner("%r24", "%r23", "%f15");
        Corner("%r24", "%r25", "%f16");
        s.AppendLine("    mul.rn.f32 %f22, %f4, %f3;");        // gradOut*mask
        s.AppendLine("    fma.rn.f32 %f0, %f22, %f20, %f0;");
        s.AppendLine("    add.u32 %r15, %r15, 1;");
        s.AppendLine($"    setp.lt.u32 %p5, %r15, {I(oww)};");
        s.AppendLine("    @%p5 bra LOOP_OW;");
        s.AppendLine("    add.u32 %r14, %r14, 1;");
        s.AppendLine($"    setp.lt.u32 %p5, %r14, {I(ohh)};");
        s.AppendLine("    @%p5 bra LOOP_OH;");
        s.AppendLine("    add.u32 %r9, %r9, 1;");
        s.AppendLine($"    setp.lt.u32 %p5, %r9, {I(Batch)};");
        s.AppendLine("    @%p5 bra LOOP_N;");
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
