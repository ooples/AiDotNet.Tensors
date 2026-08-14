using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX LocallyConnected2D backward-input: dInput[n,c,ih,iw] = sum over (k,kh,kw)
/// with oh=(ih+pad-kh)/stride, ow=(iw+pad-kw)/stride valid, of W[oh,ow,k,c,kh,kw] *
/// gradOut[n,k,oh,ow] (per-position weights [OH,OW,K,C,KH,KW]). One thread per input-
/// gradient element; consecutive threads own consecutive iw so gradOut reads and dInput
/// stores coalesce at stride 1.
/// </summary>
internal sealed class PtxLocallyConnected2DBackwardInputKernel : IDisposable
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
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long WeightBytes => (long)OutH * OutW * OutputChannels * InputChannels * KernelH * KernelW * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);

    internal LocallyConnected2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, false);
    internal static string EntryFor(LocallyConnected2DShape s) => FormattableString.Invariant(
        $"aidotnet_lc2d_bwd_input_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxLocallyConnected2DBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("LC2D backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * inputChannels * height * width % BlockThreads != 0)
            throw new ArgumentException($"N*C*H*W must be a multiple of {BlockThreads}.");

        LocallyConnected2DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, LocallyConnected2DShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Height = shape.Height, Width = shape.Width, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, OutH = shape.OutH, OutW = shape.OutW;
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var weight = new DirectPtxExtent(OutH * OutW, OutputChannels, InputChannels, KernelH * KernelW);
        var dx = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "locallyconnected2d-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, dx, dx, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dInput[n,c,ih,iw] = sum_{k,kh,kw} W[oh,ow,k,c,kh,kw]*gradOut[n,k,(ih+pad-kh)/s,(iw+pad-kw)/s]",
                ["weights"] = "per-position [OH,OW,K,C,KH,KW]", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView gradOutput, DirectPtxTensorView weights, DirectPtxTensorView gradInput)
    {
        DirectPtxAbiGuard.Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        DirectPtxAbiGuard.Require(weights, Blueprint.Tensors[1], nameof(weights));
        DirectPtxAbiGuard.Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));
        IntPtr gPtr = gradOutput.Pointer, wPtr = weights.Pointer, xPtr = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gPtr; arguments[1] = &wPtr; arguments[2] = &xPtr;
        int total = Batch * InputChannels * Height * Width;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, LocallyConnected2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 LC2D backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, Batch = shape.Batch;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kh = shape.KernelH, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow;
        int ckk = c * kh * kw, khkw = kh * kw, kckk = k * ckk;
        string entry = EntryFor(shape);

        var s = new StringBuilder(16384);
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
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<40>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*C*H*W + c*H*W + ih*W + iw
        s.AppendLine($"    div.u32 %r3, %r2, {I(chw)};");        // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(chw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");         // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");          // ih
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");          // iw
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine($"    mul.lo.u32 %r9, %r3, {I(kohow)};");   // gradOut batch base = n*K*OH*OW
        s.AppendLine("    mov.u32 %r10, 0;");                    // kk
        s.AppendLine("LOOP_K:");
        s.AppendLine("    mov.u32 %r11, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        // numH = ih + pad - kh
        s.AppendLine($"    add.s32 %r12, %r7, {I(Padding)};");
        s.AppendLine("    sub.s32 %r12, %r12, %r11;");
        s.AppendLine("    setp.ge.s32 %p0, %r12, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r13, %r12;");
        else { s.AppendLine($"    rem.s32 %r14, %r12, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p1, %r14, 0;"); s.AppendLine("    and.pred %p0, %p0, %p1;"); s.AppendLine($"    div.s32 %r13, %r12, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p2, %r13, {I(ohh)};");
        s.AppendLine("    and.pred %p0, %p0, %p2;");             // oh valid
        s.AppendLine("    mov.u32 %r15, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine($"    add.s32 %r16, %r8, {I(Padding)};");
        s.AppendLine("    sub.s32 %r16, %r16, %r15;");           // numW
        s.AppendLine("    setp.ge.s32 %p3, %r16, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r17, %r16;");
        else { s.AppendLine($"    rem.s32 %r18, %r16, {I(Stride)};"); s.AppendLine("    setp.eq.s32 %p4, %r18, 0;"); s.AppendLine("    and.pred %p3, %p3, %p4;"); s.AppendLine($"    div.s32 %r17, %r16, {I(Stride)};"); }
        s.AppendLine($"    setp.lt.s32 %p5, %r17, {I(oww)};");
        s.AppendLine("    and.pred %p3, %p3, %p5;");
        s.AppendLine("    and.pred %p3, %p3, %p0;");
        // gradOut[n][k][oh][ow] = r9 + k*OH*OW + oh*OW + ow
        s.AppendLine($"    mad.lo.u32 %r19, %r10, {I(ohow)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r19, %r13, {I(oww)}, %r19;");
        s.AppendLine("    add.u32 %r19, %r19, %r17;");
        s.AppendLine("    mul.wide.u32 %rd3, %r19, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p3 ld.global.nc.f32 %f1, [%rd3];");
        // weight[oh][ow][k][c][kh][kw] = (oh*OW+ow)*kckk + k*ckk + c*KH*KW + kh*KW + kw
        //   ohow index = oh*OW + ow = r13*OW + r17 (valid only when %p3; else weight read is masked out via %f1=0)
        s.AppendLine($"    mad.lo.u32 %r20, %r13, {I(oww)}, %r17;");
        s.AppendLine($"    mad.lo.u32 %r20, %r20, {I(kckk)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r20, %r10, {I(ckk)}, %r20;");
        s.AppendLine($"    mad.lo.u32 %r20, %r5, {I(khkw)}, %r20;");
        s.AppendLine($"    mad.lo.u32 %r20, %r11, {I(kw)}, %r20;");
        s.AppendLine("    add.u32 %r20, %r20, %r15;");
        s.AppendLine("    mul.wide.u32 %rd4, %r20, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    mov.f32 %f2, 0f00000000;");
        s.AppendLine("    @%p3 ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r15, %r15, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r15, {I(kw)};");
        s.AppendLine("    @%p6 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r11, %r11, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r11, {I(kh)};");
        s.AppendLine("    @%p6 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r10, %r10, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r10, {I(k)};");
        s.AppendLine("    @%p6 bra LOOP_K;");
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
/// Direct-PTX LocallyConnected2D backward-weight: each weight W[oh,ow,k,c,kh,kw] is used
/// by exactly one output position, so dW[oh,ow,k,c,kh,kw] = sum_n input[n,c,ih,iw] *
/// gradOut[n,k,oh,ow] with ih=oh*s+kh-pad, iw=ow*s+kw-pad. One thread per weight element
/// loops over the batch (no block reduction needed); if (ih,iw) is outside the padded
/// input the weight gradient is 0.
/// </summary>
internal sealed class PtxLocallyConnected2DBackwardWeightKernel : IDisposable
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
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradWeightBytes => (long)OutH * OutW * OutputChannels * InputChannels * KernelH * KernelW * sizeof(float);

    internal LocallyConnected2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, false);
    internal static string EntryFor(LocallyConnected2DShape s) => FormattableString.Invariant(
        $"aidotnet_lc2d_bwd_weight_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxLocallyConnected2DBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("LC2D backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)OutH * OutW * outputChannels * inputChannels * kernelH * kernelW % BlockThreads != 0)
            throw new ArgumentException($"OH*OW*K*C*KH*KW must be a multiple of {BlockThreads}.");

        LocallyConnected2DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, LocallyConnected2DShape shape)
    {
        int Batch = shape.Batch, InputChannels = shape.InputChannels, OutputChannels = shape.OutputChannels;
        int Height = shape.Height, Width = shape.Width, KernelH = shape.KernelH, KernelW = shape.KernelW;
        int Stride = shape.Stride, Padding = shape.Padding, OutH = shape.OutH, OutW = shape.OutW;
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var dw = new DirectPtxExtent(OutH * OutW, OutputChannels, InputChannels, KernelH * KernelW);
        return new DirectPtxKernelBlueprint(
            Operation: "locallyconnected2d-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[oh,ow,k,c,kh,kw] = sum_n input[n,c,oh*s+kh-pad,ow*s+kw-pad]*gradOut[n,k,oh,ow]",
                ["weights"] = "per-position [OH,OW,K,C,KH,KW]", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView gradOutput, DirectPtxTensorView gradWeight)
    {
        DirectPtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbiGuard.Require(gradOutput, Blueprint.Tensors[1], nameof(gradOutput));
        DirectPtxAbiGuard.Require(gradWeight, Blueprint.Tensors[2], nameof(gradWeight));
        IntPtr iPtr = input.Pointer, gPtr = gradOutput.Pointer, wPtr = gradWeight.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &iPtr; arguments[1] = &gPtr; arguments[2] = &wPtr;
        int total = OutH * OutW * OutputChannels * InputChannels * KernelH * KernelW;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, LocallyConnected2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 LC2D backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding, Batch = shape.Batch;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kh = shape.KernelH, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow;
        int ckk = c * kh * kw, khkw = kh * kw, kckk = k * ckk;
        string entry = EntryFor(shape);

        var s = new StringBuilder(12288);
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
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b32 %r<36>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<6>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // widx = (((ohow*K+k)*C+c)*KH+kh)*KW+kw
        s.AppendLine($"    rem.u32 %r3, %r2, {I(kw)};");         // kw
        s.AppendLine($"    div.u32 %r4, %r2, {I(kw)};");
        s.AppendLine($"    rem.u32 %r5, %r4, {I(kh)};");         // kh
        s.AppendLine($"    div.u32 %r6, %r4, {I(kh)};");
        s.AppendLine($"    rem.u32 %r7, %r6, {I(c)};");          // c
        s.AppendLine($"    div.u32 %r8, %r6, {I(c)};");
        s.AppendLine($"    rem.u32 %r9, %r8, {I(k)};");          // k
        s.AppendLine($"    div.u32 %r10, %r8, {I(k)};");         // ohow (oh*OW+ow)
        s.AppendLine($"    div.u32 %r11, %r10, {I(oww)};");      // oh
        s.AppendLine($"    rem.u32 %r12, %r10, {I(oww)};");      // ow
        // ih = oh*stride + kh - pad ; iw = ow*stride + kw - pad
        s.AppendLine($"    mad.lo.u32 %r13, %r11, {I(Stride)}, %r5;");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");  // ih
        s.AppendLine($"    mad.lo.u32 %r14, %r12, {I(Stride)}, %r3;");
        s.AppendLine($"    sub.s32 %r14, %r14, {I(Padding)};");  // iw
        s.AppendLine("    setp.ge.s32 %p0, %r13, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r13, {I(h)};");
        s.AppendLine("    setp.ge.s32 %p2, %r14, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r14, {I(w)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p0, %p0, %p2;");             // (ih,iw) in bounds
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine("    @!%p0 bra WRITE;");
        // input spatial offset within a (n,c) plane = ih*W + iw ; gradOut spatial = oh*OW+ow = r10
        s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(w)}, %r14;");   // ih*W + iw
        s.AppendLine("    mov.u32 %r16, 0;");                    // b
        s.AppendLine("LOOP_B:");
        // input[b][c][ih][iw] = (b*C + c)*HW + r15
        s.AppendLine($"    mad.lo.u32 %r17, %r16, {I(c)}, %r7;");
        s.AppendLine($"    mad.lo.u32 %r17, %r17, {I(hw)}, %r15;");
        s.AppendLine("    mul.wide.u32 %rd3, %r17, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd3];");
        // gradOut[b][k][oh][ow] = (b*K + k)*OHOW + r10
        s.AppendLine($"    mad.lo.u32 %r18, %r16, {I(k)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r18, %r18, {I(ohow)}, %r10;");
        s.AppendLine("    mul.wide.u32 %rd4, %r18, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r16, %r16, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r16, {I(Batch)};");
        s.AppendLine("    @%p4 bra LOOP_B;");
        s.AppendLine("WRITE:");
        s.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        s.AppendLine("    st.global.f32 [%rd5], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
