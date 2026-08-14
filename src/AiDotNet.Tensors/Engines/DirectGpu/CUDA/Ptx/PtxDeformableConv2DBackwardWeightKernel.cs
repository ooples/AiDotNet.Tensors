using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX Deformable Conv2D backward-weight (DCNv2 weight gradient):
/// dW[k,c,pos] = sum_{n,oh,ow} gradOut[n,k,oh,ow] * mask[n,pos,oh,ow] * bilinear(input[n,c]; py, px)
/// with py = oh*stride+kh-pad+offY, px = ow*stride+kw-pad+offX. One thread per weight element
/// loops over batch + output spatial (bounds-guarded grid since K*C*KH*KW is small), reusing the
/// forward zero-padded 4-corner bilinear sampling.
/// </summary>
internal sealed class PtxDeformableConv2DBackwardWeightKernel : IDisposable
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
    internal int TotalWeights => OutputChannels * InputChannels * Taps;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long OffsetBytes => (long)Batch * 2 * Taps * OutH * OutW * sizeof(float);
    internal long MaskBytes => (long)Batch * Taps * OutH * OutW * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradWeightBytes => (long)TotalWeights * sizeof(float);

    internal DeformableConv2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding);
    internal static string EntryFor(DeformableConv2DShape s) => FormattableString.Invariant(
        $"aidotnet_deformable_conv2d_bwd_weight_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxDeformableConv2DBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Deformable backward-weight has no experimental non-SM86 specialization.");
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
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var offset = new DirectPtxExtent(Batch, 2 * Taps, OutH, OutW);
        var mask = new DirectPtxExtent(Batch, Taps, OutH, OutW);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var dw = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-dcnv2-fp32"),
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
                ["equation"] = "dW[k,c,pos] = sum_{n,oh,ow} gradOut[n,k,oh,ow]*mask[n,pos,oh,ow]*bilinear(input[n,c]; py, px)",
                ["sampling"] = "zero-padded 4-corner bilinear; bounds-guarded grid", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView offset, DirectPtxTensorView mask, DirectPtxTensorView gradOutput, DirectPtxTensorView gradWeight)
    {
        DirectPtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbiGuard.Require(offset, Blueprint.Tensors[1], nameof(offset));
        DirectPtxAbiGuard.Require(mask, Blueprint.Tensors[2], nameof(mask));
        DirectPtxAbiGuard.Require(gradOutput, Blueprint.Tensors[3], nameof(gradOutput));
        DirectPtxAbiGuard.Require(gradWeight, Blueprint.Tensors[4], nameof(gradWeight));
        IntPtr iPtr = input.Pointer, offPtr = offset.Pointer, mPtr = mask.Pointer, gPtr = gradOutput.Pointer, wPtr = gradWeight.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &iPtr; arguments[1] = &offPtr; arguments[2] = &mPtr; arguments[3] = &gPtr; arguments[4] = &wPtr;
        uint blocks = (uint)((TotalWeights + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, DeformableConv2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 Deformable backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, KernelH = shape.KernelH, Batch = shape.Batch;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int taps = KernelH * kw, hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow;
        int offN = 2 * taps * ohow, maskN = taps * ohow, total = k * c * taps;
        string entry = EntryFor(shape);

        var s = new StringBuilder(32768);
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
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<24>;");
        s.AppendLine("    .reg .f32 %f<40>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [offset_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [mask_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd4, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // widx
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    rem.u32 %r3, %r2, {I(taps)};");        // pos
        s.AppendLine($"    div.u32 %r4, %r2, {I(taps)};");
        s.AppendLine($"    rem.u32 %r5, %r4, {I(c)};");           // c
        s.AppendLine($"    div.u32 %r6, %r4, {I(c)};");           // k
        s.AppendLine($"    div.u32 %r7, %r3, {I(kw)};");          // kh
        s.AppendLine($"    rem.u32 %r8, %r3, {I(kw)};");          // kw
        s.AppendLine("    mov.f32 %f0, 0f00000000;");            // acc
        s.AppendLine("    mov.u32 %r9, 0;");                     // n
        s.AppendLine("LOOP_N:");
        // per-batch bases
        s.AppendLine($"    mad.lo.u32 %r10, %r9, {I(c)}, %r5;");       // n*C + c
        s.AppendLine($"    mul.lo.u32 %r10, %r10, {I(hw)};");          // input (n,c) base
        s.AppendLine($"    mad.lo.u32 %r11, %r9, {I(k)}, %r6;");       // n*K + k
        s.AppendLine($"    mul.lo.u32 %r11, %r11, {I(ohow)};");        // gradOut (n,k) base
        s.AppendLine($"    mad.lo.u32 %r12, %r9, {I(offN)}, 0;");      // offset n base
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(2 * ohow)}, %r12;"); // + 2*pos*OHOW (offY channel base)
        s.AppendLine($"    mad.lo.u32 %r13, %r9, {I(maskN)}, 0;");     // mask n base
        s.AppendLine($"    mad.lo.u32 %r13, %r3, {I(ohow)}, %r13;");   // + pos*OHOW
        s.AppendLine("    mov.u32 %r14, 0;");                    // oh
        s.AppendLine("LOOP_OH:");
        s.AppendLine("    mov.u32 %r15, 0;");                    // ow
        s.AppendLine("LOOP_OW:");
        s.AppendLine($"    mad.lo.u32 %r16, %r14, {I(oww)}, %r15;");   // sp = oh*OW+ow
        // offY/offX
        s.AppendLine("    add.u32 %r17, %r12, %r16;");
        s.AppendLine("    mul.wide.u32 %rd5, %r17, 4;");
        s.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");      // offY
        s.AppendLine($"    ld.global.nc.f32 %f2, [%rd5+{I(ohow * 4)}];");  // offX
        // mask, gradOut
        s.AppendLine("    add.u32 %r18, %r13, %r16;");
        s.AppendLine("    mul.wide.u32 %rd6, %r18, 4;");
        s.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd6];");      // mask
        s.AppendLine("    add.u32 %r19, %r11, %r16;");
        s.AppendLine("    mul.wide.u32 %rd7, %r19, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    ld.global.nc.f32 %f4, [%rd7];");      // gradOut
        // py = (oh*stride-pad+kh)+offY ; px = (ow*stride-pad+kw)+offX
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
        // bilinear sample input[n][c] (base r10) -> %f20
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
        // acc += gradOut * mask * sample
        s.AppendLine("    mul.rn.f32 %f22, %f4, %f3;");        // g*mask
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
