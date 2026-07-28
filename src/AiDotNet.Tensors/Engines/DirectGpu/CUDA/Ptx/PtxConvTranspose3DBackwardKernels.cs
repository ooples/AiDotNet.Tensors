using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX ConvTranspose3D backward-input: dInput[n,ci,id,ih,iw] = sum over (co,kd,kh,kw)
/// with od=id*s-pad+kd in [0,OD), oh=ih*s-pad+kh in [0,OH), ow=iw*s-pad+kw in [0,OW) of
/// gradOut[n,co,od,oh,ow] * W[ci,co,kd,kh,kw] (IODHW) -- a regular-3D-conv-style correlation
/// of gradOut with the transposed weights. One thread per input-gradient element; consecutive
/// threads own consecutive iw so gradOut reads and dInput stores coalesce at stride 1.
/// </summary>
/// <summary>Shared geometry for the ConvTranspose3D backward kernels (device-free re-emit).</summary>
internal readonly record struct ConvTranspose3DBackwardShape(
    int Batch, int InputChannels, int OutputChannels, int Depth, int Height, int Width,
    int KernelD, int KernelH, int KernelW, int Stride, int Padding, int OutputPadding)
{
    internal int OutD => (Depth - 1) * Stride - 2 * Padding + KernelD + OutputPadding;
    internal int OutH => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutW => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
}

internal sealed class PtxConvTranspose3DBackwardInputKernel : IDisposable
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
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutD => (Depth - 1) * Stride - 2 * Padding + KernelD + OutputPadding;
    internal int OutH => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutW => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutD * OutH * OutW * sizeof(float);
    internal long WeightBytes => (long)InputChannels * OutputChannels * KernelD * KernelH * KernelW * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Depth * Height * Width * sizeof(float);

    internal ConvTranspose3DBackwardShape Shape => new(Batch, InputChannels, OutputChannels, Depth, Height, Width, KernelD, KernelH, KernelW, Stride, Padding, OutputPadding);
    internal static string EntryFor(ConvTranspose3DBackwardShape s) => FormattableString.Invariant(
        $"aidotnet_convtranspose3d_bwd_input_n{s.Batch}_ci{s.InputChannels}_co{s.OutputChannels}_d{s.Depth}_h{s.Height}_w{s.Width}_kd{s.KernelD}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}_op{s.OutputPadding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxConvTranspose3DBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int depth, int height, int width, int kernelD, int kernelH, int kernelW, int stride, int padding, int outputPadding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("ConvTranspose3D backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Depth = depth; Height = height; Width = width; KernelD = kernelD; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; OutputPadding = outputPadding;
        if (OutD <= 0 || OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * inputChannels * depth * height * width % BlockThreads != 0)
            throw new ArgumentException($"N*Ci*D*H*W must be a multiple of {BlockThreads}.");

        ConvTranspose3DBackwardShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, ConvTranspose3DBackwardShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Depth = shape.Depth, Height = shape.Height, Width = shape.Width;
        int KernelD = shape.KernelD, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, OutputPadding = shape.OutputPadding;
        int OutD = shape.OutD, OutH = shape.OutH, OutW = shape.OutW;
        var grad = new DirectPtxExtent(Batch, OutputChannels * OutD, OutH, OutW);
        var weight = new DirectPtxExtent(InputChannels, OutputChannels * KernelD, KernelH, KernelW);
        var dx = new DirectPtxExtent(Batch, InputChannels * Depth, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "convtranspose3d-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-ci{InputChannels}-co{OutputChannels}-d{Depth}-h{Height}-w{Width}-kd{KernelD}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-op{OutputPadding}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dx, dx, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 64, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dInput[n,ci,id,ih,iw] = sum_{co,kd,kh,kw} gradOut[n,co,id*s-pad+kd,ih*s-pad+kh,iw*s-pad+kw]*W[ci,co,kd,kh,kw]",
                ["weights"] = "IODHW", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView gradOutput, DirectPtxTensorView weights, DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        IntPtr gPtr = gradOutput.Pointer, wPtr = weights.Pointer, xPtr = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gPtr; arguments[1] = &wPtr; arguments[2] = &xPtr;
        int total = Batch * InputChannels * Depth * Height * Width;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, ConvTranspose3DBackwardShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 ConvTranspose3D backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, OutputPadding = shape.OutputPadding, Batch = shape.Batch;
        int KernelD = shape.KernelD, KernelH = shape.KernelH, KernelW = shape.KernelW, OutD = shape.OutD, OutH = shape.OutH, OutW = shape.OutW;
        int ci = shape.InputChannels, co = shape.OutputChannels, d = shape.Depth, h = shape.Height, w = shape.Width;
        int kd = KernelD, khh = KernelH, kw = KernelW, od = OutD, oh = OutH, ow = OutW;
        int dhw = d * h * w, hw = h * w, cidhw = ci * dhw;
        int odohow = od * oh * ow, ohow = oh * ow, coodohow = co * odohow;
        int cokdkhkw = co * kd * khh * kw, kdkhkw = kd * khh * kw, khkw = khh * kw;
        string entry = EntryFor(shape);

        var s = new StringBuilder(32768);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 dx_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<12>;");
        s.AppendLine("    .reg .b32 %r<52>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*Ci*D*H*W + ci*D*H*W + id*H*W + ih*W + iw
        s.AppendLine($"    div.u32 %r3, %r2, {I(cidhw)};");     // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(cidhw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(dhw)};");       // ci
        s.AppendLine($"    rem.u32 %r6, %r4, {I(dhw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(hw)};");        // id
        s.AppendLine($"    rem.u32 %r8, %r6, {I(hw)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(w)};");         // ih
        s.AppendLine($"    rem.u32 %r10, %r8, {I(w)};");        // iw
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine($"    mul.lo.u32 %r11, %r3, {I(coodohow)};");  // gradOut batch base
        // od0/oh0/ow0 = i*stride - pad
        s.AppendLine($"    mul.lo.u32 %r12, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r12, %r12, {I(Padding)};");
        s.AppendLine($"    mul.lo.u32 %r13, %r9, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");
        s.AppendLine($"    mul.lo.u32 %r14, %r10, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r14, %r14, {I(Padding)};");
        // weight base for ci = ci*Co*KD*KH*KW
        s.AppendLine($"    mul.lo.u32 %r15, %r5, {I(cokdkhkw)};");
        s.AppendLine("    mov.u32 %r16, 0;");                   // cc (output channel)
        s.AppendLine("LOOP_CO:");
        s.AppendLine($"    mad.lo.u32 %r17, %r16, {I(odohow)}, %r11;");  // gradOut[n][cc] base
        s.AppendLine($"    mad.lo.u32 %r18, %r16, {I(kdkhkw)}, %r15;");  // weight[ci][cc] base
        s.AppendLine("    mov.u32 %r19, 0;");                   // kdc
        s.AppendLine("LOOP_KD:");
        s.AppendLine("    add.s32 %r20, %r12, %r19;");          // od = od0 + kd
        s.AppendLine("    setp.ge.s32 %p0, %r20, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r20, {I(od)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    mov.u32 %r21, 0;");                   // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine("    add.s32 %r22, %r13, %r21;");          // oh = oh0 + kh
        s.AppendLine("    setp.ge.s32 %p2, %r22, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r22, {I(oh)};");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p2, %p2, %p0;");
        s.AppendLine("    mov.u32 %r23, 0;");                   // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine("    add.s32 %r24, %r14, %r23;");          // ow = ow0 + kw
        s.AppendLine("    setp.ge.s32 %p4, %r24, 0;");
        s.AppendLine($"    setp.lt.s32 %p5, %r24, {I(ow)};");
        s.AppendLine("    and.pred %p4, %p4, %p5;");
        s.AppendLine("    and.pred %p4, %p4, %p2;");
        // gradOut index = r17 + od*OH*OW + oh*OW + ow
        s.AppendLine($"    mad.lo.u32 %r25, %r20, {I(ohow)}, %r17;");
        s.AppendLine($"    mad.lo.u32 %r25, %r22, {I(ow)}, %r25;");
        s.AppendLine("    add.u32 %r25, %r25, %r24;");
        s.AppendLine("    mul.wide.u32 %rd3, %r25, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p4 ld.global.nc.f32 %f1, [%rd3];");
        // weight index = r18 + kd*KH*KW + kh*KW + kw
        s.AppendLine($"    mad.lo.u32 %r26, %r19, {I(khkw)}, %r18;");
        s.AppendLine($"    mad.lo.u32 %r26, %r21, {I(kw)}, %r26;");
        s.AppendLine("    add.u32 %r26, %r26, %r23;");
        s.AppendLine("    mul.wide.u32 %rd4, %r26, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r23, %r23, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r23, {I(kw)};");
        s.AppendLine("    @%p6 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r21, %r21, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r21, {I(khh)};");
        s.AppendLine("    @%p6 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r19, %r19, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r19, {I(kd)};");
        s.AppendLine("    @%p6 bra LOOP_KD;");
        s.AppendLine("    add.u32 %r16, %r16, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r16, {I(co)};");
        s.AppendLine("    @%p6 bra LOOP_CO;");
        s.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        s.AppendLine("    st.global.f32 [%rd5], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Direct-PTX ConvTranspose3D backward-weight: dW[ci,co,kd,kh,kw] = sum over (n,id,ih,iw)
/// with od=id*s-pad+kd in [0,OD), oh=ih*s-pad+kh in [0,OH), ow=iw*s-pad+kw in [0,OW) of
/// input[n,ci,id,ih,iw] * gradOut[n,co,od,oh,ow] (IODHW weights). One block per (ci,co)
/// reduces the N x D x H x W contraction into the KD*KH*KW taps with coalesced input reads
/// (reused across taps) and a shared tree reduction per tap. KD*KH*KW &lt;= 27.
/// </summary>
internal sealed class PtxConvTranspose3DBackwardWeightKernel : IDisposable
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
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutD => (Depth - 1) * Stride - 2 * Padding + KernelD + OutputPadding;
    internal int OutH => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutW => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal long InputBytes => (long)Batch * InputChannels * Depth * Height * Width * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutD * OutH * OutW * sizeof(float);
    internal long GradWeightBytes => (long)InputChannels * OutputChannels * KernelD * KernelH * KernelW * sizeof(float);

    internal ConvTranspose3DBackwardShape Shape => new(Batch, InputChannels, OutputChannels, Depth, Height, Width, KernelD, KernelH, KernelW, Stride, Padding, OutputPadding);
    internal static string EntryFor(ConvTranspose3DBackwardShape s) => FormattableString.Invariant(
        $"aidotnet_convtranspose3d_bwd_weight_n{s.Batch}_ci{s.InputChannels}_co{s.OutputChannels}_d{s.Depth}_h{s.Height}_w{s.Width}_kd{s.KernelD}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}_op{s.OutputPadding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxConvTranspose3DBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int depth, int height, int width, int kernelD, int kernelH, int kernelW, int stride, int padding, int outputPadding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("ConvTranspose3D backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (kernelD * kernelH * kernelW > 27) throw new ArgumentOutOfRangeException(nameof(kernelD), "KD*KH*KW <= 27 (per-tap accumulators).");
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Depth = depth; Height = height; Width = width; KernelD = kernelD; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; OutputPadding = outputPadding;
        if (OutD <= 0 || OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");

        ConvTranspose3DBackwardShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, ConvTranspose3DBackwardShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Depth = shape.Depth, Height = shape.Height, Width = shape.Width;
        int KernelD = shape.KernelD, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, OutputPadding = shape.OutputPadding;
        int OutD = shape.OutD, OutH = shape.OutH, OutW = shape.OutW;
        var input = new DirectPtxExtent(Batch, InputChannels * Depth, Height, Width);
        var grad = new DirectPtxExtent(Batch, OutputChannels * OutD, OutH, OutW);
        var dw = new DirectPtxExtent(InputChannels, OutputChannels * KernelD, KernelH, KernelW);
        return new DirectPtxKernelBlueprint(
            Operation: "convtranspose3d-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-ci{InputChannels}-co{OutputChannels}-d{Depth}-h{Height}-w{Width}-kd{KernelD}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-op{OutputPadding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 96, MaxStaticSharedBytes: BlockThreads * sizeof(float), MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[ci,co,kd,kh,kw] = sum_{n,id,ih,iw} input[n,ci,id,ih,iw]*gradOut[n,co,id*s-pad+kd,ih*s-pad+kh,iw*s-pad+kw]",
                ["weights"] = "IODHW", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView gradOutput, DirectPtxTensorView gradWeight)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(gradOutput, Blueprint.Tensors[1], nameof(gradOutput));
        Require(gradWeight, Blueprint.Tensors[2], nameof(gradWeight));
        IntPtr iPtr = input.Pointer, gPtr = gradOutput.Pointer, wPtr = gradWeight.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &iPtr; arguments[1] = &gPtr; arguments[2] = &wPtr;
        _module.Launch(_function, (uint)(InputChannels * OutputChannels), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int major, int minor, ConvTranspose3DBackwardShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 ConvTranspose3D backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, OutputPadding = shape.OutputPadding, Batch = shape.Batch;
        int KernelD = shape.KernelD, KernelH = shape.KernelH, KernelW = shape.KernelW, OutD = shape.OutD, OutH = shape.OutH, OutW = shape.OutW;
        int ci = shape.InputChannels, co = shape.OutputChannels, d = shape.Depth, h = shape.Height, w = shape.Width;
        int kd = KernelD, khh = KernelH, kw = KernelW, od = OutD, oh = OutH, ow = OutW, taps = kd * khh * kw;
        int dhw = d * h * w, hw = h * w, cidhw = ci * dhw, ndhw = Batch * dhw;
        int odohow = od * oh * ow, ohow = oh * ow, coodohow = co * odohow;
        string entry = EntryFor(shape);

        var s = new StringBuilder(40960);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 dw_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<10>;");
        s.AppendLine("    .reg .b32 %r<36>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine($"    .reg .f32 %f<{I(taps + 4)}>;");
        s.AppendLine($"    .shared .align 4 .b32 red[{I(BlockThreads)}];");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");             // block = ci*Co + co
        s.AppendLine($"    div.u32 %r2, %r1, {I(co)};");        // ci
        s.AppendLine($"    rem.u32 %r3, %r1, {I(co)};");        // co
        for (int t = 0; t < taps; t++)
            s.AppendLine($"    mov.f32 %f{t}, 0f00000000;");
        s.AppendLine("    mov.u32 %r4, %r0;");                  // i over N*D*H*W
        s.AppendLine("LOOP:");
        s.AppendLine($"    setp.ge.u32 %p0, %r4, {I(ndhw)};");
        s.AppendLine("    @%p0 bra REDUCE;");
        s.AppendLine($"    div.u32 %r5, %r4, {I(dhw)};");       // nn
        s.AppendLine($"    rem.u32 %r6, %r4, {I(dhw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(hw)};");        // id
        s.AppendLine($"    rem.u32 %r8, %r6, {I(hw)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(w)};");         // ih
        s.AppendLine($"    rem.u32 %r10, %r8, {I(w)};");        // iw
        // input[nn][ci][id][ih][iw] index = nn*Ci*DHW + ci*DHW + r6
        s.AppendLine($"    mad.lo.u32 %r11, %r5, {I(cidhw)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r11, %r2, {I(dhw)}, %r11;");
        s.AppendLine("    mul.wide.u32 %rd3, %r11, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine($"    ld.global.nc.f32 %f{I(taps)}, [%rd3];");   // input value
        // gradOut[nn][co] base = nn*Co*ODOHOW + co*ODOHOW
        s.AppendLine($"    mad.lo.u32 %r12, %r5, {I(coodohow)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(odohow)}, %r12;");
        // od0/oh0/ow0 = i*stride - pad
        s.AppendLine($"    mul.lo.u32 %r13, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");
        s.AppendLine($"    mul.lo.u32 %r14, %r9, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r14, %r14, {I(Padding)};");
        s.AppendLine($"    mul.lo.u32 %r15, %r10, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r15, %r15, {I(Padding)};");
        for (int a = 0; a < kd; a++)
            for (int rr = 0; rr < khh; rr++)
                for (int t = 0; t < kw; t++)
                {
                    int tap = (a * khh + rr) * kw + t;
                    s.AppendLine($"    add.s32 %r16, %r13, {I(a)};");
                    s.AppendLine($"    add.s32 %r17, %r14, {I(rr)};");
                    s.AppendLine($"    add.s32 %r18, %r15, {I(t)};");
                    s.AppendLine("    setp.ge.s32 %p1, %r16, 0;");
                    s.AppendLine($"    setp.lt.s32 %p2, %r16, {I(od)};");
                    s.AppendLine("    setp.ge.s32 %p3, %r17, 0;");
                    s.AppendLine($"    setp.lt.s32 %p4, %r17, {I(oh)};");
                    s.AppendLine("    setp.ge.s32 %p5, %r18, 0;");
                    s.AppendLine($"    setp.lt.s32 %p6, %r18, {I(ow)};");
                    s.AppendLine("    and.pred %p1, %p1, %p2;");
                    s.AppendLine("    and.pred %p3, %p3, %p4;");
                    s.AppendLine("    and.pred %p5, %p5, %p6;");
                    s.AppendLine("    and.pred %p1, %p1, %p3;");
                    s.AppendLine("    and.pred %p1, %p1, %p5;");
                    s.AppendLine($"    mad.lo.u32 %r19, %r16, {I(ohow)}, %r12;");
                    s.AppendLine($"    mad.lo.u32 %r19, %r17, {I(ow)}, %r19;");
                    s.AppendLine("    add.u32 %r19, %r19, %r18;");
                    s.AppendLine("    mul.wide.u32 %rd4, %r19, 4;");
                    s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
                    s.AppendLine($"    mov.f32 %f{I(taps + 1)}, 0f00000000;");
                    s.AppendLine($"    @%p1 ld.global.nc.f32 %f{I(taps + 1)}, [%rd4];");
                    s.AppendLine($"    fma.rn.f32 %f{tap}, %f{I(taps + 1)}, %f{I(taps)}, %f{tap};");
                }
        s.AppendLine($"    add.u32 %r4, %r4, {I(BlockThreads)};");
        s.AppendLine("    bra LOOP;");
        s.AppendLine("REDUCE:");
        s.AppendLine("    mov.u64 %rd5, red;");
        s.AppendLine("    mul.wide.u32 %rd6, %r0, 4;");
        s.AppendLine("    add.u64 %rd6, %rd5, %rd6;");
        // dW base = (ci*Co + co)*KD*KH*KW
        s.AppendLine($"    mad.lo.u32 %r20, %r2, {I(co)}, %r3;");
        s.AppendLine($"    mul.lo.u32 %r20, %r20, {I(taps)};");
        s.AppendLine("    mul.wide.u32 %rd7, %r20, 4;");
        s.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        for (int t = 0; t < taps; t++)
        {
            s.AppendLine("    bar.sync 0;");
            s.AppendLine($"    st.shared.f32 [%rd6], %f{t};");
            s.AppendLine("    bar.sync 0;");
            for (int offset = BlockThreads / 2; offset > 0; offset >>= 1)
            {
                string skip = $"S_{t}_{offset}";
                s.AppendLine($"    setp.lt.u32 %p7, %r0, {I(offset)};");
                s.AppendLine($"    @!%p7 bra {skip};");
                s.AppendLine($"    ld.shared.f32 %f{I(taps + 2)}, [%rd6];");
                s.AppendLine($"    ld.shared.f32 %f{I(taps + 3)}, [%rd6+{I(offset * 4)}];");
                s.AppendLine($"    add.rn.f32 %f{I(taps + 2)}, %f{I(taps + 2)}, %f{I(taps + 3)};");
                s.AppendLine($"    st.shared.f32 [%rd6], %f{I(taps + 2)};");
                s.AppendLine($"{skip}:");
                s.AppendLine("    bar.sync 0;");
            }
            s.AppendLine("    setp.ne.u32 %p7, %r0, 0;");
            s.AppendLine($"    @%p7 bra AFTER_{t};");
            s.AppendLine($"    ld.shared.f32 %f{I(taps + 2)}, [%rd5];");
            s.AppendLine($"    st.global.f32 [%rd7+{I(t * 4)}], %f{I(taps + 2)};");
            s.AppendLine($"AFTER_{t}:");
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
