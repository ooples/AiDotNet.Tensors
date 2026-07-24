using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX Deformable Conv2D forward (DCNv2, single deformable group) with per-kernel
/// learned offsets and modulation masks, plus per-output-channel bias:
/// out[n,k,oh,ow] = bias[k] + sum_{c,kh,kw} W[k,c,kh,kw] * mask[n,kh*KW+kw,oh,ow] *
/// bilinear(input[n,c]; py, px), with py = oh*stride+kh-pad + offY, px = ow*stride+kw-pad + offX
/// (offY/offX from offset[N,2*KH*KW,OH,OW], mask from mask[N,KH*KW,OH,OW]). Sampling uses
/// zero-padded 4-corner bilinear interpolation. One thread per output element; consecutive
/// threads own consecutive ow so the offset/mask reads (contiguous OH*OW) coalesce.
/// </summary>
/// <summary>Shared geometry for the DCNv2 (single deform-group) kernels (device-free re-emit).</summary>
internal readonly record struct DeformableConv2DShape(
    int Batch, int InputChannels, int OutputChannels, int Height, int Width,
    int KernelH, int KernelW, int Stride, int Padding)
{
    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal int Taps => KernelH * KernelW;
}

internal sealed class PtxDeformableConv2DKernel : IDisposable
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
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);

    internal DeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding);
    internal static string EntryFor(DeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deformable_conv2d_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("DeformableConv2D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * outputChannels * OutH * OutW % BlockThreads != 0)
            throw new ArgumentException($"N*K*OH*OW must be a multiple of {BlockThreads}.");

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
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dcnv2-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, offset, offset, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 96, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,oh,ow] = bias[k] + sum W[k,c,kh,kw]*mask[..]*bilinear(input[n,c]; oh*s+kh-pad+offY, ow*s+kw-pad+offX)",
                ["sampling"] = "zero-padded 4-corner bilinear", ["variant"] = "DCNv2 single deformable group",
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
        int total = Batch * OutputChannels * OutH * OutW;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, DeformableConv2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 DeformableConv2D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kh = shape.KernelH, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = kh * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * taps;
        int offN = 2 * taps * ohow;   // offset per-batch stride
        int maskN = taps * ohow;      // mask per-batch stride
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
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<48>;");
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
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*K*OH*OW + k*OH*OW + oh*OW + ow
        s.AppendLine($"    div.u32 %r3, %r2, {I(kohow)};");     // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(kohow)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");      // k
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");      // ohow = oh*OW+ow
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");       // ow
        s.AppendLine("    mul.wide.u32 %rd6, %r5, 4;");
        s.AppendLine("    add.u64 %rd6, %rd4, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd6];");     // acc = bias[k]
        // oh_base = oh*stride - pad ; ow_base = ow*stride - pad
        s.AppendLine($"    mul.lo.u32 %r9, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r9, %r9, {I(Padding)};");   // oh_base
        s.AppendLine($"    mul.lo.u32 %r10, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r10, %r10, {I(Padding)};"); // ow_base
        // offset batch base = n*offN + (oh*OW+ow)  [channel stride = OH*OW]; mask batch base = n*maskN + r6
        s.AppendLine($"    mad.lo.u32 %r11, %r3, {I(offN)}, %r6;");   // offset[n][0][oh][ow] elem index
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(maskN)}, %r6;");  // mask[n][0][oh][ow] elem index
        s.AppendLine($"    mul.lo.u32 %r13, %r3, {I(chw)};");    // input batch base
        s.AppendLine($"    mul.lo.u32 %r14, %r5, {I(ckk)};");    // weight out-channel base
        s.AppendLine("    mov.u32 %r15, 0;");                    // cc
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r16, %r15, {I(hw)}, %r13;");   // input channel base
        s.AppendLine($"    mad.lo.u32 %r17, %r15, {I(taps)}, %r14;"); // weight (k,cc) base
        s.AppendLine("    mov.u32 %r18, 0;");                    // pos = kh*KW+kw
        s.AppendLine("LOOP_TAP:");
        // kh = pos / KW ; kw = pos % KW
        s.AppendLine($"    div.u32 %r19, %r18, {I(kw)};");      // kh
        s.AppendLine($"    rem.u32 %r20, %r18, {I(kw)};");      // kw
        // offY = offset[n][2*pos][oh][ow] ; offX = offset[n][2*pos+1][oh][ow]
        s.AppendLine($"    mad.lo.u32 %r21, %r18, {I(2 * ohow)}, %r11;");  // + 2*pos*OHOW
        s.AppendLine("    mul.wide.u32 %rd7, %r21, 4;");
        s.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd7];");      // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd7+{I(ohow * 4)}];");  // offX (next channel)
        // mask = mask[n][pos][oh][ow]
        s.AppendLine($"    mad.lo.u32 %r22, %r18, {I(ohow)}, %r12;");
        s.AppendLine("    mul.wide.u32 %rd8, %r22, 4;");
        s.AppendLine("    add.u64 %rd8, %rd3, %rd8;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd8];");      // mask
        // py = (oh_base + kh) + offY ; px = (ow_base + kw) + offX
        s.AppendLine("    add.s32 %r23, %r9, %r19;");
        s.AppendLine("    cvt.rn.f32.s32 %f4, %r23;");
        s.AppendLine("    add.rn.f32 %f4, %f4, %f1;");          // py
        s.AppendLine("    add.s32 %r24, %r10, %r20;");
        s.AppendLine("    cvt.rn.f32.s32 %f5, %r24;");
        s.AppendLine("    add.rn.f32 %f5, %f5, %f2;");          // px
        // y0 = floor(py), x0 = floor(px)
        s.AppendLine("    cvt.rmi.f32.f32 %f6, %f4;");          // floor(py) as float
        s.AppendLine("    cvt.rmi.f32.f32 %f7, %f5;");          // floor(px)
        s.AppendLine("    cvt.rmi.s32.f32 %r25, %f4;");         // y0 int
        s.AppendLine("    cvt.rmi.s32.f32 %r26, %f5;");         // x0 int
        s.AppendLine("    sub.rn.f32 %f8, %f4, %f6;");          // wy1 = py - y0
        s.AppendLine("    sub.rn.f32 %f9, %f5, %f7;");          // wx1 = px - x0
        s.AppendLine("    sub.rn.f32 %f10, 0f3F800000, %f8;");  // wy0 = 1 - wy1
        s.AppendLine("    sub.rn.f32 %f11, 0f3F800000, %f9;");  // wx0 = 1 - wx1
        // corner weights: w00=wy0*wx0, w01=wy0*wx1, w10=wy1*wx0, w11=wy1*wx1
        s.AppendLine("    mul.rn.f32 %f12, %f10, %f11;");       // w00
        s.AppendLine("    mul.rn.f32 %f13, %f10, %f9;");        // w01
        s.AppendLine("    mul.rn.f32 %f14, %f8, %f11;");        // w10
        s.AppendLine("    mul.rn.f32 %f15, %f8, %f9;");         // w11
        s.AppendLine("    add.s32 %r27, %r25, 1;");             // y1
        s.AppendLine("    add.s32 %r28, %r26, 1;");             // x1
        s.AppendLine("    mov.f32 %f20, 0f00000000;");          // val accumulator
        // helper: for each corner (yy in {r25,r27}, xx in {r26,r28}, weight fW) load + fma
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
        Corner("%r25", "%r26", "%f12");   // (y0,x0)
        Corner("%r25", "%r28", "%f13");   // (y0,x1)
        Corner("%r27", "%r26", "%f14");   // (y1,x0)
        Corner("%r27", "%r28", "%f15");   // (y1,x1)
        // weight
        s.AppendLine($"    add.u32 %r30, %r17, %r18;");        // weight index = base + pos
        s.AppendLine("    mul.wide.u32 %rd10, %r30, 4;");
        s.AppendLine("    add.u64 %rd10, %rd1, %rd10;");
        s.AppendLine("    ld.global.nc.f32 %f22, [%rd10];");   // W
        // acc += W * mask * val
        s.AppendLine("    mul.rn.f32 %f23, %f22, %f3;");       // W*mask
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
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
