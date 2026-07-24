using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX native Conv3D backward-input: dInput[n,c,id,ih,iw] = sum over (k,kd,kh,kw)
/// with od=(id+pad-kd)/stride, oh=(ih+pad-kh)/stride, ow=(iw+pad-kw)/stride all valid
/// (non-negative, divisible, in-range) of W[k,c,kd,kh,kw] * gradOut[n,k,od,oh,ow]. One
/// thread per input-gradient element; consecutive threads own consecutive iw so gradOut
/// reads and dInput stores coalesce at stride 1 and the weights broadcast across the warp.
/// </summary>
internal sealed class PtxConv3DBackwardInputKernel : IDisposable
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
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutD => (Depth + 2 * Padding - KernelD) / Stride + 1;
    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutD * OutH * OutW * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * KernelD * KernelH * KernelW * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Depth * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv3d_bwd_input_n{Batch}_c{InputChannels}_k{OutputChannels}_d{Depth}_h{Height}_w{Width}_kd{KernelD}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}");

    internal PtxConv3DBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int depth, int height, int width, int kernelD, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv3D backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Depth = depth; Height = height; Width = width; KernelD = kernelD; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutD <= 0 || OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * inputChannels * depth * height * width % BlockThreads != 0)
            throw new ArgumentException($"N*C*D*H*W must be a multiple of {BlockThreads}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var grad = new DirectPtxExtent(Batch, OutputChannels * OutD, OutH, OutW);
        var weight = new DirectPtxExtent(OutputChannels, InputChannels * KernelD, KernelH, KernelW);
        var dx = new DirectPtxExtent(Batch, InputChannels * Depth, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "conv3d-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-d{Depth}-h{Height}-w{Width}-kd{KernelD}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dx, dx, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 56, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dInput[n,c,id,ih,iw] = sum_{k,kd,kh,kw} W[k,c,kd,kh,kw]*gradOut[n,k,(id+pad-kd)/s,(ih+pad-kh)/s,(iw+pad-kw)/s]",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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

    internal string EmitPtx(int major, int minor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 Conv3D backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, k = OutputChannels, d = Depth, h = Height, w = Width;
        int kd = KernelD, khh = KernelH, kw = KernelW, od = OutD, oh = OutH, ow = OutW;
        int dhw = d * h * w, hw = h * w, cdhw = c * dhw;
        int odohow = od * oh * ow, ohow = oh * ow, kodohow = k * odohow;
        int ckdkhkw = c * kd * khh * kw, kdkhkw = kd * khh * kw, khkw = khh * kw;
        string entry = EntryPoint;

        var s = new StringBuilder(28672);
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
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*C*D*H*W + c*D*H*W + id*H*W + ih*W + iw
        s.AppendLine($"    div.u32 %r3, %r2, {I(cdhw)};");        // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(cdhw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(dhw)};");         // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(dhw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(hw)};");          // id
        s.AppendLine($"    rem.u32 %r8, %r6, {I(hw)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(w)};");           // ih
        s.AppendLine($"    rem.u32 %r10, %r8, {I(w)};");          // iw
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine($"    mul.lo.u32 %r11, %r3, {I(kodohow)};"); // gradOut batch base
        s.AppendLine($"    mul.lo.u32 %r12, %r5, {I(kdkhkw)};");  // weight channel offset (c*KD*KH*KW)
        s.AppendLine("    mov.u32 %r13, 0;");                     // kk (output channel)
        s.AppendLine("LOOP_K:");
        s.AppendLine($"    mad.lo.u32 %r14, %r13, {I(odohow)}, %r11;"); // gradOut[n][kk] base
        s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(ckdkhkw)}, %r12;"); // weight[kk][c] base
        s.AppendLine("    mov.u32 %r16, 0;");                     // kdc
        s.AppendLine("LOOP_KD:");
        // numD = id + pad - kd
        s.AppendLine($"    add.s32 %r17, %r7, {I(Padding)};");
        s.AppendLine("    sub.s32 %r17, %r17, %r16;");
        s.AppendLine("    setp.ge.s32 %p0, %r17, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r18, %r17;");
        else { s.AppendLine($"    rem.s32 %r19, %r17, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p1, %r19, 0;"); s.AppendLine("    and.pred %p0, %p0, %p1;"); s.AppendLine($"    div.s32 %r18, %r17, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p2, %r18, {I(od)};");
        s.AppendLine("    and.pred %p0, %p0, %p2;");              // od valid
        s.AppendLine("    mov.u32 %r20, 0;");                     // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine($"    add.s32 %r21, %r9, {I(Padding)};");
        s.AppendLine("    sub.s32 %r21, %r21, %r20;");
        s.AppendLine("    setp.ge.s32 %p3, %r21, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r22, %r21;");
        else { s.AppendLine($"    rem.s32 %r23, %r21, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p4, %r23, 0;"); s.AppendLine("    and.pred %p3, %p3, %p4;"); s.AppendLine($"    div.s32 %r22, %r21, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p5, %r22, {I(oh)};");
        s.AppendLine("    and.pred %p3, %p3, %p5;");
        s.AppendLine("    and.pred %p3, %p3, %p0;");
        s.AppendLine("    mov.u32 %r24, 0;");                     // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine($"    add.s32 %r25, %r10, {I(Padding)};");
        s.AppendLine("    sub.s32 %r25, %r25, %r24;");
        s.AppendLine("    setp.ge.s32 %p6, %r25, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r26, %r25;");
        else { s.AppendLine($"    rem.s32 %r27, %r25, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p7, %r27, 0;"); s.AppendLine("    and.pred %p6, %p6, %p7;"); s.AppendLine($"    div.s32 %r26, %r25, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p8, %r26, {I(ow)};");
        s.AppendLine("    and.pred %p6, %p6, %p8;");
        s.AppendLine("    and.pred %p6, %p6, %p3;");
        // gradOut index = r14 + od*OH*OW + oh*OW + ow
        s.AppendLine($"    mad.lo.u32 %r28, %r18, {I(ohow)}, %r14;");
        s.AppendLine($"    mad.lo.u32 %r28, %r22, {I(ow)}, %r28;");
        s.AppendLine("    add.u32 %r28, %r28, %r26;");
        s.AppendLine("    mul.wide.u32 %rd3, %r28, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p6 ld.global.nc.f32 %f1, [%rd3];");
        // weight index = r15 + kd*KH*KW + kh*KW + kw
        s.AppendLine($"    mad.lo.u32 %r29, %r16, {I(khkw)}, %r15;");
        s.AppendLine($"    mad.lo.u32 %r29, %r20, {I(kw)}, %r29;");
        s.AppendLine("    add.u32 %r29, %r29, %r24;");
        s.AppendLine("    mul.wide.u32 %rd4, %r29, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r24, %r24, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r24, {I(kw)};");
        s.AppendLine("    @%p9 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r20, %r20, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r20, {I(khh)};");
        s.AppendLine("    @%p9 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r16, %r16, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r16, {I(kd)};");
        s.AppendLine("    @%p9 bra LOOP_KD;");
        s.AppendLine("    add.u32 %r13, %r13, 1;");
        s.AppendLine($"    setp.lt.u32 %p9, %r13, {I(k)};");
        s.AppendLine("    @%p9 bra LOOP_K;");
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
/// Direct-PTX native Conv3D backward-weight: dW[k,c,kd,kh,kw] = sum over (n,od,oh,ow) of
/// input[n,c,od*s+kd-pad, oh*s+kh-pad, ow*s+kw-pad] * gradOut[n,k,od,oh,ow]. One block per
/// (k,c) reduces the N x OD x OH x OW contraction into the KD*KH*KW taps with coalesced
/// gradOut reads (reused across taps) and a shared tree reduction per tap. KD*KH*KW &lt;= 27.
/// </summary>
internal sealed class PtxConv3DBackwardWeightKernel : IDisposable
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
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutD => (Depth + 2 * Padding - KernelD) / Stride + 1;
    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal long InputBytes => (long)Batch * InputChannels * Depth * Height * Width * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutD * OutH * OutW * sizeof(float);
    internal long GradWeightBytes => (long)OutputChannels * InputChannels * KernelD * KernelH * KernelW * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv3d_bwd_weight_n{Batch}_c{InputChannels}_k{OutputChannels}_d{Depth}_h{Height}_w{Width}_kd{KernelD}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}");

    internal PtxConv3DBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int depth, int height, int width, int kernelD, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv3D backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (kernelD * kernelH * kernelW > 27) throw new ArgumentOutOfRangeException(nameof(kernelD), "KD*KH*KW <= 27 (per-tap accumulators).");
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Depth = depth; Height = height; Width = width; KernelD = kernelD; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutD <= 0 || OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var input = new DirectPtxExtent(Batch, InputChannels * Depth, Height, Width);
        var grad = new DirectPtxExtent(Batch, OutputChannels * OutD, OutH, OutW);
        var dw = new DirectPtxExtent(OutputChannels, InputChannels * KernelD, KernelH, KernelW);
        return new DirectPtxKernelBlueprint(
            Operation: "conv3d-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-d{Depth}-h{Height}-w{Width}-kd{KernelD}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 96, MaxStaticSharedBytes: BlockThreads * sizeof(float), MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[k,c,kd,kh,kw] = sum_{n,od,oh,ow} input[n,c,od*s+kd-pad,oh*s+kh-pad,ow*s+kw-pad]*gradOut[n,k,od,oh,ow]",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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
        _module.Launch(_function, (uint)(OutputChannels * InputChannels), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal string EmitPtx(int major, int minor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 Conv3D backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, k = OutputChannels, d = Depth, h = Height, w = Width;
        int kd = KernelD, khh = KernelH, kw = KernelW, od = OutD, oh = OutH, ow = OutW, taps = kd * khh * kw;
        int dhw = d * h * w, hw = h * w, cdhw = c * dhw;
        int odohow = od * oh * ow, ohow = oh * ow, kodohow = k * odohow, nodohow = Batch * odohow;
        string entry = EntryPoint;

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
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");               // block = k*C + c
        s.AppendLine($"    div.u32 %r2, %r1, {I(c)};");          // k
        s.AppendLine($"    rem.u32 %r3, %r1, {I(c)};");          // c
        for (int t = 0; t < taps; t++)
            s.AppendLine($"    mov.f32 %f{t}, 0f00000000;");
        s.AppendLine("    mov.u32 %r4, %r0;");                   // i over N*OD*OH*OW
        s.AppendLine("LOOP:");
        s.AppendLine($"    setp.ge.u32 %p0, %r4, {I(nodohow)};");
        s.AppendLine("    @%p0 bra REDUCE;");
        s.AppendLine($"    div.u32 %r5, %r4, {I(odohow)};");     // nn
        s.AppendLine($"    rem.u32 %r6, %r4, {I(odohow)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(ohow)};");       // od
        s.AppendLine($"    rem.u32 %r8, %r6, {I(ohow)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(ow)};");         // oh
        s.AppendLine($"    rem.u32 %r10, %r8, {I(ow)};");        // ow
        // gradOut[nn][k][od][oh][ow] index = nn*K*OD*OH*OW + k*OD*OH*OW + r6
        s.AppendLine($"    mad.lo.u32 %r11, %r5, {I(kodohow)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r11, %r2, {I(odohow)}, %r11;");
        s.AppendLine("    mul.wide.u32 %rd3, %r11, 4;");
        s.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        s.AppendLine($"    ld.global.nc.f32 %f{I(taps)}, [%rd3];");   // gradOut value
        // input channel base index = nn*C*D*H*W + c*D*H*W
        s.AppendLine($"    mad.lo.u32 %r12, %r5, {I(cdhw)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r12, %r3, {I(dhw)}, %r12;");
        // id0/ih0/iw0 = o*stride - pad
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
                    s.AppendLine($"    add.s32 %r16, %r13, {I(a)};");   // id
                    s.AppendLine($"    add.s32 %r17, %r14, {I(rr)};");  // ih
                    s.AppendLine($"    add.s32 %r18, %r15, {I(t)};");   // iw
                    s.AppendLine("    setp.ge.s32 %p1, %r16, 0;");
                    s.AppendLine($"    setp.lt.s32 %p2, %r16, {I(d)};");
                    s.AppendLine("    setp.ge.s32 %p3, %r17, 0;");
                    s.AppendLine($"    setp.lt.s32 %p4, %r17, {I(h)};");
                    s.AppendLine("    setp.ge.s32 %p5, %r18, 0;");
                    s.AppendLine($"    setp.lt.s32 %p6, %r18, {I(w)};");
                    s.AppendLine("    and.pred %p1, %p1, %p2;");
                    s.AppendLine("    and.pred %p3, %p3, %p4;");
                    s.AppendLine("    and.pred %p5, %p5, %p6;");
                    s.AppendLine("    and.pred %p1, %p1, %p3;");
                    s.AppendLine("    and.pred %p1, %p1, %p5;");
                    s.AppendLine($"    mad.lo.u32 %r19, %r16, {I(hw)}, %r12;");
                    s.AppendLine($"    mad.lo.u32 %r19, %r17, {I(w)}, %r19;");
                    s.AppendLine("    add.u32 %r19, %r19, %r18;");
                    s.AppendLine("    mul.wide.u32 %rd4, %r19, 4;");
                    s.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
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
        s.AppendLine($"    mad.lo.u32 %r20, %r2, {I(c)}, %r3;");
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
