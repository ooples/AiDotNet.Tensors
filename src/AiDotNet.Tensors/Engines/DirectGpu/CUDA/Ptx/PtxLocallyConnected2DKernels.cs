using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX LocallyConnected2D forward (unshared spatial weights): each output
/// position (oh,ow) has its own filter. Weights are laid out [OH,OW,K,C,KH,KW], bias
/// [K,OH,OW]; out[n,k,oh,ow] = relu(bias[k,oh,ow] + sum_{c,kh,kw} W[oh,ow,k,c,kh,kw] *
/// in[n,c,oh*s+kh-pad,ow*s+kw-pad]). One thread per output element; consecutive threads
/// own consecutive ow so at stride 1 the input reads and output stores coalesce.
/// </summary>
/// <summary>Shared geometry for the locally-connected 2D fwd/backward kernels (device-free re-emit).</summary>
internal readonly record struct LocallyConnected2DShape(
    int Batch, int InputChannels, int OutputChannels, int Height, int Width,
    int KernelH, int KernelW, int Stride, int Padding, bool Relu)
{
    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
}

internal sealed class PtxLocallyConnected2DKernel : IDisposable
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
    internal bool Relu { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutH * OutW * OutputChannels * InputChannels * KernelH * KernelW * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * OutH * OutW * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);

    internal LocallyConnected2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, Relu);
    internal static string EntryFor(LocallyConnected2DShape s) => FormattableString.Invariant(
        $"aidotnet_locallyconnected2d_n{s.Batch}_c{s.InputChannels}_k{s.OutputChannels}_h{s.Height}_w{s.Width}_kh{s.KernelH}_kw{s.KernelW}_s{s.Stride}_p{s.Padding}{(s.Relu ? "_relu" : "")}");
    internal string EntryPoint => EntryFor(Shape);

    internal PtxLocallyConnected2DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, bool relu = true)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("LocallyConnected2D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; Relu = relu;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * outputChannels * OutH * OutW % BlockThreads != 0)
            throw new ArgumentException($"N*K*OH*OW must be a multiple of {BlockThreads}.");

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
        bool Relu = shape.Relu;
        // Weights collapsed to rank-4 (OH*OW, K, C, KH*KW) for byte-length ABI.
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var weight = new DirectPtxExtent(OutH * OutW, OutputChannels, InputChannels, KernelH * KernelW);
        var bias = new DirectPtxExtent(OutputChannels, OutH, OutW);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "locallyconnected2d-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}{(Relu ? "-relu" : "")}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,oh,ow] = " + (Relu ? "relu(" : "(") + "bias[k,oh,ow] + sum W[oh,ow,k,c,kh,kw]*in[n,c,oh*s+kh-pad,ow*s+kw-pad])",
                ["weights"] = "per-position [OH,OW,K,C,KH,KW]", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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
        int total = Batch * OutputChannels * OutH * OutW;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, LocallyConnected2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 LocallyConnected2D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding; bool Relu = shape.Relu;
        int c = shape.InputChannels, k = shape.OutputChannels, h = shape.Height, w = shape.Width, kh = shape.KernelH, kw = shape.KernelW, ohh = shape.OutH, oww = shape.OutW;
        int hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow;
        int ckk = c * kh * kw, khkw = kh * kw, kckk = k * ckk;       // per-position weight block = K*C*KH*KW
        string entry = EntryFor(shape);

        var s = new StringBuilder(16384);
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
        s.AppendLine("    .reg .pred %p<8>;");
        s.AppendLine("    .reg .b32 %r<40>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*K*OH*OW + k*OH*OW + oh*OW + ow
        s.AppendLine($"    div.u32 %r3, %r2, {I(kohow)};");      // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(kohow)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");       // k
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");       // ohow (oh*OW+ow)
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");        // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");        // ow
        // bias[k][oh][ow] = k*OH*OW + oh*OW + ow = k*OHOW + r6
        s.AppendLine($"    mad.lo.u32 %r9, %r5, {I(ohow)}, %r6;");
        s.AppendLine("    mul.wide.u32 %rd4, %r9, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");       // acc = bias
        s.AppendLine($"    mul.lo.u32 %r10, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r10, %r10, {I(Padding)};");  // ih0
        s.AppendLine($"    mul.lo.u32 %r11, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r11, %r11, {I(Padding)};");  // iw0
        s.AppendLine($"    mul.lo.u32 %r12, %r3, {I(chw)};");    // input batch base
        // weight base for this (oh,ow,k): (r6*K + k)*C*KH*KW = r6*kckk + k*ckk
        s.AppendLine($"    mad.lo.u32 %r13, %r6, {I(kckk)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r13, %r5, {I(ckk)}, %r13;");
        s.AppendLine("    mov.u32 %r14, 0;");                    // cc
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r15, %r14, {I(hw)}, %r12;");   // input channel base
        s.AppendLine($"    mad.lo.u32 %r16, %r14, {I(khkw)}, %r13;"); // weight (.,c) base
        s.AppendLine("    mov.u32 %r17, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine("    add.s32 %r18, %r10, %r17;");           // ih
        s.AppendLine("    setp.ge.s32 %p0, %r18, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r18, {I(h)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    mov.u32 %r19, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine("    add.s32 %r20, %r11, %r19;");           // iw
        s.AppendLine("    setp.ge.s32 %p2, %r20, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r20, {I(w)};");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p2, %p2, %p0;");
        s.AppendLine($"    mad.lo.u32 %r21, %r18, {I(w)}, %r15;");
        s.AppendLine("    add.u32 %r21, %r21, %r20;");
        s.AppendLine("    mul.wide.u32 %rd5, %r21, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p2 ld.global.nc.f32 %f1, [%rd5];");
        s.AppendLine($"    mad.lo.u32 %r22, %r17, {I(kw)}, %r16;");
        s.AppendLine("    add.u32 %r22, %r22, %r19;");           // weight index
        s.AppendLine("    mul.wide.u32 %rd6, %r22, 4;");
        s.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r19, %r19, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r19, {I(kw)};");
        s.AppendLine("    @%p4 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r17, %r17, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r17, {I(kh)};");
        s.AppendLine("    @%p4 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r14, %r14, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r14, {I(c)};");
        s.AppendLine("    @%p4 bra LOOP_C;");
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

/// <summary>
/// Direct-PTX LocallyConnected2D backward-bias: each output position has its own bias,
/// so dBias[k,oh,ow] = sum over batch of gradOut[n,k,oh,ow]. One thread per (k,oh,ow)
/// output-bias element loops over N; consecutive threads own consecutive ow so the
/// gradOut reads coalesce.
/// </summary>
internal sealed class PtxLocallyConnected2DBackwardBiasKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int OutputChannels { get; }
    internal int OutH { get; }
    internal int OutW { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal long GradOutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);
    internal long GradBiasBytes => (long)OutputChannels * OutH * OutW * sizeof(float);

    internal static string EntryFor(int batch, int k, int oh, int ow) => FormattableString.Invariant(
        $"aidotnet_lc2d_bwd_bias_n{batch}_k{k}_oh{oh}_ow{ow}");
    internal string EntryPoint => EntryFor(Batch, OutputChannels, OutH, OutW);

    internal PtxLocallyConnected2DBackwardBiasKernel(DirectPtxRuntime runtime, int batch, int outputChannels, int outH, int outW)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("LC2D backward-bias has no experimental non-SM86 specialization.");
        if (batch <= 0 || outputChannels <= 0 || outH <= 0 || outW <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; OutputChannels = outputChannels; OutH = outH; OutW = outW;
        if ((long)outputChannels * outH * outW % BlockThreads != 0)
            throw new ArgumentException($"K*OH*OW must be a multiple of {BlockThreads}.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, Batch, OutputChannels, OutH, OutW);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, Batch, OutputChannels, OutH, OutW);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int Batch, int OutputChannels, int OutH, int OutW)
    {
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        var bias = new DirectPtxExtent(OutputChannels, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "locallyconnected2d-backward-bias", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-k{OutputChannels}-oh{OutH}-ow{OutW}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradBias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, bias, bias, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 24, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dBias[k,oh,ow] = sum_n gradOut[n,k,oh,ow]",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView gradOutput, DirectPtxTensorView gradBias)
    {
        DirectPtxAbiGuard.Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        DirectPtxAbiGuard.Require(gradBias, Blueprint.Tensors[1], nameof(gradBias));
        IntPtr gPtr = gradOutput.Pointer, bPtr = gradBias.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &gPtr; arguments[1] = &bPtr;
        int total = OutputChannels * OutH * OutW;
        _module.Launch(_function, (uint)(total / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, int Batch, int OutputChannels, int OutH, int OutW)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 LC2D backward-bias emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int khow = OutputChannels * OutH * OutW;   // per-batch gradOut stride and output size
        string entry = EntryFor(Batch, OutputChannels, OutH, OutW);

        var s = new StringBuilder(6144);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 bias_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<2>;");
        s.AppendLine("    .reg .b32 %r<8>;");
        s.AppendLine("    .reg .b64 %rd<10>;");
        s.AppendLine("    .reg .f32 %f<4>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [bias_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // elem = k*OH*OW + oh*OW + ow
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine("    mov.u32 %r3, 0;");                     // b
        s.AppendLine("LOOP_B:");
        s.AppendLine($"    mad.lo.u32 %r4, %r3, {I(khow)}, %r2;");
        s.AppendLine("    mul.wide.u32 %rd2, %r4, 4;");
        s.AppendLine("    add.u64 %rd2, %rd0, %rd2;");
        s.AppendLine("    ld.global.nc.f32 %f1, [%rd2];");
        s.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        s.AppendLine("    add.u32 %r3, %r3, 1;");
        s.AppendLine($"    setp.lt.u32 %p0, %r3, {I(Batch)};");
        s.AppendLine("    @%p0 bra LOOP_B;");
        s.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        s.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        s.AppendLine("    st.global.f32 [%rd3], %f0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
