using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX ConvTranspose2D backward-input: dInput[n,ci,ih,iw] = sum over (co,kh,kw)
/// with oh = ih*stride - pad + kh in [0,OH) and ow = iw*stride - pad + kw in [0,OW) of
/// gradOut[n,co,oh,ow] * W[ci,co,kh,kw] (IOHW). This is a regular-conv-style correlation
/// of gradOut with the transposed weights. One thread per input-gradient element;
/// consecutive threads own consecutive iw so gradOut reads and dInput stores coalesce at
/// stride 1 and the weights broadcast across the warp.
/// </summary>
internal sealed class PtxConvTranspose2DBackwardInputKernel : IDisposable
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
    internal int OutputPadding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutHeight => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutWidth => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutHeight * OutWidth * sizeof(float);
    internal long WeightBytes => (long)InputChannels * OutputChannels * KernelH * KernelW * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_convtranspose2d_bwd_input_n{Batch}_ci{InputChannels}_co{OutputChannels}_h{Height}_w{Width}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}_op{OutputPadding}");

    internal PtxConvTranspose2DBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int outputPadding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("ConvTranspose2D backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; OutputPadding = outputPadding;
        if (OutHeight <= 0 || OutWidth <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * inputChannels * height * width % BlockThreads != 0)
            throw new ArgumentException($"N*Ci*H*W must be a multiple of {BlockThreads}.");

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
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        var weight = new DirectPtxExtent(InputChannels, OutputChannels, KernelH, KernelW);
        var dx = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "convtranspose2d-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-ci{InputChannels}-co{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-op{OutputPadding}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, dx, dx, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dInput[n,ci,ih,iw] = sum_{co,kh,kw} gradOut[n,co,ih*s-pad+kh,iw*s-pad+kw]*W[ci,co,kh,kw]",
                ["weights"] = "IOHW", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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
        int total = Batch * InputChannels * Height * Width;
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
            throw new NotSupportedException("Only the experimental SM86 ConvTranspose2D backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int ci = InputChannels, co = OutputChannels, h = Height, w = Width, kh = KernelH, kw = KernelW;
        int oh = OutHeight, ow = OutWidth;
        int hw = h * w, cihw = ci * hw, cohw = co * oh * ow, ohw = oh * ow, cokk = co * kh * kw, khkw = kh * kw;
        string entry = EntryPoint;

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
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*Ci*H*W + ci*H*W + ih*W + iw
        s.AppendLine($"    div.u32 %r3, %r2, {I(cihw)};");        // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(cihw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");          // ci
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");           // ih
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");           // iw
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine($"    mul.lo.u32 %r9, %r3, {I(cohw)};");     // gradOut batch base = n*Co*OH*OW
        // oh0 = ih*stride - pad ; ow0 = iw*stride - pad
        s.AppendLine($"    mul.lo.u32 %r10, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r10, %r10, {I(Padding)};");   // oh0
        s.AppendLine($"    mul.lo.u32 %r11, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r11, %r11, {I(Padding)};");   // ow0
        // weight (ci) base = ci*Co*KH*KW
        s.AppendLine($"    mul.lo.u32 %r12, %r5, {I(cokk)};");
        s.AppendLine("    mov.u32 %r13, 0;");                    // cc (output channel)
        s.AppendLine("LOOP_CO:");
        s.AppendLine($"    mad.lo.u32 %r14, %r13, {I(ohw)}, %r9;");  // gradOut[n][cc] base
        s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(khkw)}, %r12;"); // weight[ci][cc] base
        s.AppendLine("    mov.u32 %r16, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine("    add.s32 %r17, %r10, %r16;");           // oh = oh0 + kh
        s.AppendLine("    setp.ge.s32 %p0, %r17, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r17, {I(oh)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    mov.u32 %r18, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine("    add.s32 %r19, %r11, %r18;");           // ow = ow0 + kw
        s.AppendLine("    setp.ge.s32 %p2, %r19, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r19, {I(ow)};");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p2, %p2, %p0;");
        s.AppendLine($"    mad.lo.u32 %r20, %r17, {I(ow)}, %r14;");
        s.AppendLine("    add.u32 %r20, %r20, %r19;");           // gradOut index
        s.AppendLine("    mul.wide.u32 %rd3, %r20, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p2 ld.global.nc.f32 %f1, [%rd3];");
        s.AppendLine($"    mad.lo.u32 %r21, %r16, {I(kw)}, %r15;");
        s.AppendLine("    add.u32 %r21, %r21, %r18;");           // weight index
        s.AppendLine("    mul.wide.u32 %rd4, %r21, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r18, %r18, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r18, {I(kw)};");
        s.AppendLine("    @%p4 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r16, %r16, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r16, {I(kh)};");
        s.AppendLine("    @%p4 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r13, %r13, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r13, {I(co)};");
        s.AppendLine("    @%p4 bra LOOP_CO;");
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
/// Direct-PTX ConvTranspose2D backward-weight: dW[ci,co,kh,kw] = sum over (n,ih,iw) with
/// oh = ih*stride - pad + kh in [0,OH) and ow = iw*stride - pad + kw in [0,OW) of
/// input[n,ci,ih,iw] * gradOut[n,co,oh,ow] (IOHW weights). One block per (ci,co) reduces
/// the N x H x W contraction into the KH*KW taps: consecutive threads walk the flattened
/// (n,ih,iw) index (coalesced input reads, input reused across taps) and a shared tree
/// reduction combines each tap. KH*KW &lt;= 25.
/// </summary>
internal sealed class PtxConvTranspose2DBackwardWeightKernel : IDisposable
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
    internal int OutputPadding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutHeight => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutWidth => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutHeight * OutWidth * sizeof(float);
    internal long GradWeightBytes => (long)InputChannels * OutputChannels * KernelH * KernelW * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_convtranspose2d_bwd_weight_n{Batch}_ci{InputChannels}_co{OutputChannels}_h{Height}_w{Width}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}_op{OutputPadding}");

    internal PtxConvTranspose2DBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int outputPadding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("ConvTranspose2D backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (kernelH * kernelW > 25) throw new ArgumentOutOfRangeException(nameof(kernelH), "KH*KW <= 25 (per-tap accumulators).");
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; OutputPadding = outputPadding;
        if (OutHeight <= 0 || OutWidth <= 0) throw new ArgumentException("Non-positive output spatial.");

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
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        var dw = new DirectPtxExtent(InputChannels, OutputChannels, KernelH, KernelW);
        return new DirectPtxKernelBlueprint(
            Operation: "convtranspose2d-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-ci{InputChannels}-co{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-op{OutputPadding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 80, MaxStaticSharedBytes: BlockThreads * sizeof(float), MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[ci,co,kh,kw] = sum_{n,ih,iw} input[n,ci,ih,iw]*gradOut[n,co,ih*s-pad+kh,iw*s-pad+kw]",
                ["weights"] = "IOHW", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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

    internal string EmitPtx(int major, int minor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 ConvTranspose2D backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int ci = InputChannels, co = OutputChannels, h = Height, w = Width, kh = KernelH, kw = KernelW;
        int oh = OutHeight, ow = OutWidth, taps = kh * kw;
        int hw = h * w, nhw = Batch * hw, cihw = ci * hw, cohw = co * oh * ow, ohw = oh * ow;
        string entry = EntryPoint;

        var s = new StringBuilder(24576);
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
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<32>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine($"    .reg .f32 %f<{I(taps + 4)}>;");
        s.AppendLine($"    .shared .align 4 .b32 red[{I(BlockThreads)}];");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");               // block = ci*Co + co
        s.AppendLine($"    div.u32 %r2, %r1, {I(co)};");         // ci
        s.AppendLine($"    rem.u32 %r3, %r1, {I(co)};");         // co
        for (int t = 0; t < taps; t++)
            s.AppendLine($"    mov.f32 %f{t}, 0f00000000;");
        s.AppendLine("    mov.u32 %r4, %r0;");                   // i over N*H*W
        s.AppendLine("LOOP:");
        s.AppendLine($"    setp.ge.u32 %p0, %r4, {I(nhw)};");
        s.AppendLine("    @%p0 bra REDUCE;");
        s.AppendLine($"    div.u32 %r5, %r4, {I(hw)};");         // nn
        s.AppendLine($"    rem.u32 %r6, %r4, {I(hw)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(w)};");          // ih
        s.AppendLine($"    rem.u32 %r8, %r6, {I(w)};");          // iw
        // input[nn][ci][ih][iw] = nn*Ci*HW + ci*HW + ih*W + iw
        s.AppendLine($"    mad.lo.u32 %r9, %r5, {I(cihw)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r9, %r2, {I(hw)}, %r9;");
        s.AppendLine("    mul.wide.u32 %rd3, %r9, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine($"    ld.global.nc.f32 %f{I(taps)}, [%rd3];");   // input value (reused across taps)
        // gradOut[nn][co] base = nn*Co*OH*OW + co*OH*OW
        s.AppendLine($"    mad.lo.u32 %r10, %r5, {I(cohw)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r10, %r3, {I(ohw)}, %r10;");
        // oh0 = ih*stride - pad ; ow0 = iw*stride - pad
        s.AppendLine($"    mul.lo.u32 %r11, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r11, %r11, {I(Padding)};");
        s.AppendLine($"    mul.lo.u32 %r12, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r12, %r12, {I(Padding)};");
        for (int khc = 0; khc < kh; khc++)
            for (int kwc = 0; kwc < kw; kwc++)
            {
                int t = khc * kw + kwc;
                s.AppendLine($"    add.s32 %r13, %r11, {I(khc)};");   // ohh
                s.AppendLine($"    add.s32 %r14, %r12, {I(kwc)};");   // oww
                s.AppendLine("    setp.ge.s32 %p1, %r13, 0;");
                s.AppendLine($"    setp.lt.s32 %p2, %r13, {I(oh)};");
                s.AppendLine("    setp.ge.s32 %p3, %r14, 0;");
                s.AppendLine($"    setp.lt.s32 %p4, %r14, {I(ow)};");
                s.AppendLine("    and.pred %p1, %p1, %p2;");
                s.AppendLine("    and.pred %p3, %p3, %p4;");
                s.AppendLine("    and.pred %p1, %p1, %p3;");
                s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(ow)}, %r10;");
                s.AppendLine("    add.u32 %r15, %r15, %r14;");
                s.AppendLine("    mul.wide.u32 %rd4, %r15, 4;");
                s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
                s.AppendLine($"    mov.f32 %f{I(taps + 1)}, 0f00000000;");
                s.AppendLine($"    @%p1 ld.global.nc.f32 %f{I(taps + 1)}, [%rd4];");
                s.AppendLine($"    fma.rn.f32 %f{t}, %f{I(taps + 1)}, %f{I(taps)}, %f{t};");
            }
        s.AppendLine($"    add.u32 %r4, %r4, {I(BlockThreads)};");
        s.AppendLine("    bra LOOP;");
        s.AppendLine("REDUCE:");
        s.AppendLine("    mov.u64 %rd5, red;");
        s.AppendLine("    mul.wide.u32 %rd6, %r0, 4;");
        s.AppendLine("    add.u64 %rd6, %rd5, %rd6;");
        // dW base = (ci*Co + co)*KH*KW
        s.AppendLine($"    mad.lo.u32 %r16, %r2, {I(co)}, %r3;");
        s.AppendLine($"    mul.lo.u32 %r16, %r16, {I(taps)};");
        s.AppendLine("    mul.wide.u32 %rd7, %r16, 4;");
        s.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        for (int t = 0; t < taps; t++)
        {
            s.AppendLine("    bar.sync 0;");
            s.AppendLine($"    st.shared.f32 [%rd6], %f{t};");
            s.AppendLine("    bar.sync 0;");
            for (int offset = BlockThreads / 2; offset > 0; offset >>= 1)
            {
                string skip = $"S_{t}_{offset}";
                s.AppendLine($"    setp.lt.u32 %p5, %r0, {I(offset)};");
                s.AppendLine($"    @!%p5 bra {skip};");
                s.AppendLine($"    ld.shared.f32 %f{I(taps + 2)}, [%rd6];");
                s.AppendLine($"    ld.shared.f32 %f{I(taps + 3)}, [%rd6+{I(offset * 4)}];");
                s.AppendLine($"    add.rn.f32 %f{I(taps + 2)}, %f{I(taps + 2)}, %f{I(taps + 3)};");
                s.AppendLine($"    st.shared.f32 [%rd6], %f{I(taps + 2)};");
                s.AppendLine($"{skip}:");
                s.AppendLine("    bar.sync 0;");
            }
            s.AppendLine("    setp.ne.u32 %p5, %r0, 0;");
            s.AppendLine($"    @%p5 bra AFTER_{t};");
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
