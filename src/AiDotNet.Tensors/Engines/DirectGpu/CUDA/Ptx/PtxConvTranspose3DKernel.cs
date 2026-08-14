using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX ConvTranspose3D forward with per-output-channel bias and optional ReLU.
/// Weights are IODHW [Cin,Cout,KD,KH,KW]; out[n,co,od,oh,ow] = relu(bias[co] + sum over
/// (ci,kd,kh,kw) with id=(od+pad-kd)/stride, ih=(oh+pad-kh)/stride, iw=(ow+pad-kw)/stride
/// all valid, of input[n,ci,id,ih,iw] * W[ci,co,kd,kh,kw]). The 3D transpose-gather run as
/// a forward op. One thread per output element; consecutive threads own consecutive ow so
/// at stride 1 the input reads and output stores coalesce (the contiguous NCDHW axis).
/// </summary>
/// <summary>Shape identity for <see cref="PtxConvTranspose3DKernel"/> for device-free re-emit.</summary>
internal readonly record struct ConvTranspose3DShape(
    int Batch, int InputChannels, int OutputChannels, int Depth, int Height, int Width,
    int KernelD, int KernelH, int KernelW, int Stride, int Padding, int OutputPadding, bool Relu)
{
    internal int OutD => (Depth - 1) * Stride - 2 * Padding + KernelD + OutputPadding;
    internal int OutH => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutW => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal string Entry => FormattableString.Invariant(
        $"aidotnet_convtranspose3d_n{Batch}_ci{InputChannels}_co{OutputChannels}_d{Depth}_h{Height}_w{Width}_kd{KernelD}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}_op{OutputPadding}{(Relu ? "_relu" : "")}");
}

internal sealed class PtxConvTranspose3DKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int OutputChannels { get; }
    internal int Depth { get; }
    internal int Height { get; }
    internal int Width { get; }
    internal int KernelD { get; }
    internal int KernelH { get; }
    internal int KernelW { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal int OutputPadding { get; }
    internal bool Relu { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutD => (Depth - 1) * Stride - 2 * Padding + KernelD + OutputPadding;
    internal int OutH => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutW => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal long InputBytes => (long)Batch * InputChannels * Depth * Height * Width * sizeof(float);
    internal long WeightBytes => (long)InputChannels * OutputChannels * KernelD * KernelH * KernelW * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutD * OutH * OutW * sizeof(float);

    internal ConvTranspose3DShape Shape => new(Batch, InputChannels, OutputChannels, Depth, Height, Width, KernelD, KernelH, KernelW, Stride, Padding, OutputPadding, Relu);
    internal string EntryPoint => Shape.Entry;

    internal PtxConvTranspose3DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int depth, int height, int width, int kernelD, int kernelH, int kernelW, int stride, int padding, int outputPadding, bool relu = true)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("ConvTranspose3D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Depth = depth; Height = height; Width = width; KernelD = kernelD; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; OutputPadding = outputPadding; Relu = relu;
        if (OutD <= 0 || OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * outputChannels * OutD * OutH * OutW % BlockThreads != 0)
            throw new ArgumentException($"N*Cout*OD*OH*OW must be a multiple of {BlockThreads}.");

        ConvTranspose3DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, ConvTranspose3DShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Depth = shape.Depth, Height = shape.Height, Width = shape.Width;
        int KernelD = shape.KernelD, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, OutputPadding = shape.OutputPadding;
        bool Relu = shape.Relu;
        int OutD = shape.OutD, OutH = shape.OutH, OutW = shape.OutW;
        var input = new DirectPtxExtent(Batch, InputChannels * Depth, Height, Width);
        var weight = new DirectPtxExtent(InputChannels, OutputChannels * KernelD, KernelH, KernelW);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels * OutD, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "convtranspose3d-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-ci{InputChannels}-co{OutputChannels}-d{Depth}-h{Height}-w{Width}-kd{KernelD}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-op{OutputPadding}{(Relu ? "-relu" : "")}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 64, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,co,od,oh,ow] = " + (Relu ? "relu(" : "(") + "bias[co] + sum input[n,ci,(od+pad-kd)/s,(oh+pad-kh)/s,(ow+pad-kw)/s]*W[ci,co,kd,kh,kw])",
                ["weights"] = "IODHW", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        DirectPtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbiGuard.Require(weights, Blueprint.Tensors[1], nameof(weights));
        DirectPtxAbiGuard.Require(bias, Blueprint.Tensors[2], nameof(bias));
        DirectPtxAbiGuard.Require(output, Blueprint.Tensors[3], nameof(output));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, bPtr = bias.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &bPtr; arguments[3] = &oPtr;
        int total = Batch * OutputChannels * OutD * OutH * OutW;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, ConvTranspose3DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 ConvTranspose3D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding;
        bool Relu = shape.Relu;
        int ci = shape.InputChannels, co = shape.OutputChannels, d = shape.Depth, h = shape.Height, w = shape.Width;
        int kd = shape.KernelD, khh = shape.KernelH, kw = shape.KernelW, od = shape.OutD, oh = shape.OutH, ow = shape.OutW;
        int dhw = d * h * w, hw = h * w, cidhw = ci * dhw;
        int odohow = od * oh * ow, ohow = oh * ow, coodohow = co * odohow;
        int cokdkhkw = co * kd * khh * kw, kdkhkw = kd * khh * kw, khkw = khh * kw;
        string entry = shape.Entry;

        var s = new StringBuilder(32768);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<12>;");
        s.AppendLine("    .reg .b32 %r<52>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx
        s.AppendLine($"    div.u32 %r3, %r2, {I(coodohow)};");   // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(coodohow)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(odohow)};");     // co
        s.AppendLine($"    rem.u32 %r6, %r4, {I(odohow)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(ohow)};");       // od
        s.AppendLine($"    rem.u32 %r8, %r6, {I(ohow)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(ow)};");         // oh
        s.AppendLine($"    rem.u32 %r10, %r8, {I(ow)};");        // ow
        s.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");       // acc = bias[co]
        s.AppendLine($"    mul.lo.u32 %r11, %r3, {I(cidhw)};");  // input batch base
        s.AppendLine("    mov.u32 %r12, 0;");                    // cc (input channel)
        s.AppendLine("LOOP_CI:");
        s.AppendLine($"    mad.lo.u32 %r13, %r12, {I(dhw)}, %r11;");   // input channel base
        s.AppendLine($"    mad.lo.u32 %r14, %r12, {I(cokdkhkw)}, 0;"); // weight (cc,.) base
        s.AppendLine($"    mad.lo.u32 %r14, %r5, {I(kdkhkw)}, %r14;"); // + co*KDKHKW
        s.AppendLine("    mov.u32 %r15, 0;");                    // kdc
        s.AppendLine("LOOP_KD:");
        // numD = od + pad - kd
        s.AppendLine($"    add.s32 %r16, %r7, {I(Padding)};");
        s.AppendLine("    sub.s32 %r16, %r16, %r15;");
        s.AppendLine("    setp.ge.s32 %p0, %r16, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r17, %r16;");
        else { s.AppendLine($"    rem.s32 %r18, %r16, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p1, %r18, 0;"); s.AppendLine("    and.pred %p0, %p0, %p1;"); s.AppendLine($"    div.s32 %r17, %r16, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p2, %r17, {I(d)};");
        s.AppendLine("    and.pred %p0, %p0, %p2;");             // id valid
        s.AppendLine("    mov.u32 %r19, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine($"    add.s32 %r20, %r9, {I(Padding)};");
        s.AppendLine("    sub.s32 %r20, %r20, %r19;");
        s.AppendLine("    setp.ge.s32 %p3, %r20, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r21, %r20;");
        else { s.AppendLine($"    rem.s32 %r22, %r20, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p4, %r22, 0;"); s.AppendLine("    and.pred %p3, %p3, %p4;"); s.AppendLine($"    div.s32 %r21, %r20, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p5, %r21, {I(h)};");
        s.AppendLine("    and.pred %p3, %p3, %p5;");
        s.AppendLine("    and.pred %p3, %p3, %p0;");
        s.AppendLine("    mov.u32 %r23, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine($"    add.s32 %r24, %r10, {I(Padding)};");
        s.AppendLine("    sub.s32 %r24, %r24, %r23;");
        s.AppendLine("    setp.ge.s32 %p6, %r24, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r25, %r24;");
        else { s.AppendLine($"    rem.s32 %r26, %r24, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p7, %r26, 0;"); s.AppendLine("    and.pred %p6, %p6, %p7;"); s.AppendLine($"    div.s32 %r25, %r24, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p8, %r25, {I(w)};");
        s.AppendLine("    and.pred %p6, %p6, %p8;");
        s.AppendLine("    and.pred %p6, %p6, %p3;");
        // input index = r13 + id*H*W + ih*W + iw
        s.AppendLine($"    mad.lo.u32 %r27, %r17, {I(hw)}, %r13;");
        s.AppendLine($"    mad.lo.u32 %r27, %r21, {I(w)}, %r27;");
        s.AppendLine("    add.u32 %r27, %r27, %r25;");
        s.AppendLine("    mul.wide.u32 %rd5, %r27, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p6 ld.global.nc.f32 %f1, [%rd5];");
        // weight index = r14 + kd*KH*KW + kh*KW + kw
        s.AppendLine($"    mad.lo.u32 %r28, %r15, {I(khkw)}, %r14;");
        s.AppendLine($"    mad.lo.u32 %r28, %r19, {I(kw)}, %r28;");
        s.AppendLine("    add.u32 %r28, %r28, %r23;");
        s.AppendLine("    mul.wide.u32 %rd6, %r28, 4;");
        s.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r23, %r23, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r23, {I(kw)};");
        s.AppendLine("    @%p9 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r19, %r19, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r19, {I(khh)};");
        s.AppendLine("    @%p9 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r15, %r15, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r15, {I(kd)};");
        s.AppendLine("    @%p9 bra LOOP_KD;");
        s.AppendLine("    add.u32 %r12, %r12, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r12, {I(ci)};");
        s.AppendLine("    @%p9 bra LOOP_CI;");
        if (Relu) s.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        s.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    st.global.f32 [%rd7], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
