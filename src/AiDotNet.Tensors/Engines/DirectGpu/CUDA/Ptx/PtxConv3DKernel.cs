using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX native Conv3D forward with per-output-channel bias and optional ReLU:
/// out[n,k,od,oh,ow] = relu(bias[k] + sum_{c,kd,kh,kw} W[k,c,kd,kh,kw] *
/// in[n,c,od*s+kd-pad, oh*s+kh-pad, ow*s+kw-pad]). General kernel extent / stride /
/// padding. One thread per output element; at stride 1 consecutive threads own
/// consecutive ow so the innermost input reads and the output stores coalesce (the
/// contiguous NCDHW axis) and the weights broadcast across the warp.
/// </summary>
internal sealed class PtxConv3DKernel : IDisposable
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
    internal bool Relu { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutD => (Depth + 2 * Padding - KernelD) / Stride + 1;
    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal long InputBytes => (long)Batch * InputChannels * Depth * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * KernelD * KernelH * KernelW * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutD * OutH * OutW * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv3d_n{Batch}_c{InputChannels}_k{OutputChannels}_d{Depth}_h{Height}_w{Width}_kd{KernelD}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}{(Relu ? "_relu" : "")}");

    internal PtxConv3DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int depth, int height, int width, int kernelD, int kernelH, int kernelW, int stride, int padding, bool relu = true)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv3D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Depth = depth; Height = height; Width = width; KernelD = kernelD; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; Relu = relu;
        if (OutD <= 0 || OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * outputChannels * OutD * OutH * OutW % BlockThreads != 0)
            throw new ArgumentException($"N*K*OD*OH*OW must be a multiple of {BlockThreads}.");

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
        // DirectPtxExtent is rank <= 4; collapse the leading dims (byte-length ABI only,
        // the emitter computes true 5D NCDHW offsets on the flat buffer).
        var input = new DirectPtxExtent(Batch, InputChannels * Depth, Height, Width);
        var weight = new DirectPtxExtent(OutputChannels, InputChannels * KernelD, KernelH, KernelW);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels * OutD, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "conv3d-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-d{Depth}-h{Height}-w{Width}-kd{KernelD}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}{(Relu ? "-relu" : "")}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 56, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,od,oh,ow] = " + (Relu ? "relu(" : "(") + "bias[k] + sum W[k,c,kd,kh,kw]*in[n,c,od*s+kd-pad,oh*s+kh-pad,ow*s+kw-pad])",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView bias, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, bPtr = bias.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &bPtr; arguments[3] = &oPtr;
        int total = Batch * OutputChannels * OutD * OutH * OutW;
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
            throw new NotSupportedException("Only the experimental SM86 Conv3D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, k = OutputChannels, d = Depth, h = Height, w = Width;
        int kd = KernelD, khh = KernelH, kw = KernelW;
        int od = OutD, oh = OutH, ow = OutW;
        int dhw = d * h * w, hw = h * w;
        int odohow = od * oh * ow, ohow = oh * ow;
        int cdhw = c * dhw;                              // input batch stride
        int kodohow = k * odohow;                        // output batch stride
        int ckdkhkw = c * kd * khh * kw;                 // weight output-channel stride
        int kdkhkw = kd * khh * kw;
        int khkw = khh * kw;
        string entry = EntryPoint;

        var s = new StringBuilder(24576);
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
        s.AppendLine("    .reg .pred %p<10>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx
        s.AppendLine($"    div.u32 %r3, %r2, {I(kodohow)};");    // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(kodohow)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(odohow)};");     // k
        s.AppendLine($"    rem.u32 %r6, %r4, {I(odohow)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(ohow)};");       // od
        s.AppendLine($"    rem.u32 %r8, %r6, {I(ohow)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(ow)};");         // oh
        s.AppendLine($"    rem.u32 %r10, %r8, {I(ow)};");        // ow
        s.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");       // acc = bias[k]
        // id0/ih0/iw0
        s.AppendLine($"    mul.lo.u32 %r11, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r11, %r11, {I(Padding)};");  // id0
        s.AppendLine($"    mul.lo.u32 %r12, %r9, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r12, %r12, {I(Padding)};");  // ih0
        s.AppendLine($"    mul.lo.u32 %r13, %r10, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");  // iw0
        s.AppendLine($"    mul.lo.u32 %r14, %r3, {I(cdhw)};");   // input batch base
        s.AppendLine($"    mul.lo.u32 %r15, %r5, {I(ckdkhkw)};");// weight out-channel base
        s.AppendLine("    mov.u32 %r16, 0;");                    // cc
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r17, %r16, {I(dhw)}, %r14;");   // input channel base
        s.AppendLine($"    mad.lo.u32 %r18, %r16, {I(kdkhkw)}, %r15;");// weight (k,cc) base
        s.AppendLine("    mov.u32 %r19, 0;");                    // kdc
        s.AppendLine("LOOP_KD:");
        s.AppendLine("    add.s32 %r20, %r11, %r19;");           // id
        s.AppendLine("    setp.ge.s32 %p0, %r20, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r20, {I(d)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    mov.u32 %r21, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine("    add.s32 %r22, %r12, %r21;");           // ih
        s.AppendLine("    setp.ge.s32 %p2, %r22, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r22, {I(h)};");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p2, %p2, %p0;");
        s.AppendLine("    mov.u32 %r23, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine("    add.s32 %r24, %r13, %r23;");           // iw
        s.AppendLine("    setp.ge.s32 %p4, %r24, 0;");
        s.AppendLine($"    setp.lt.s32 %p5, %r24, {I(w)};");
        s.AppendLine("    and.pred %p4, %p4, %p5;");
        s.AppendLine("    and.pred %p4, %p4, %p2;");
        // input index = ((cc-base) + id*H*W + ih*W + iw)  where r17 = channel base
        s.AppendLine($"    mad.lo.u32 %r25, %r20, {I(hw)}, %r17;");
        s.AppendLine($"    mad.lo.u32 %r25, %r22, {I(w)}, %r25;");
        s.AppendLine("    add.u32 %r25, %r25, %r24;");
        s.AppendLine("    mul.wide.u32 %rd5, %r25, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p4 ld.global.nc.f32 %f1, [%rd5];");
        // weight index = r18 + kd*KH*KW + kh*KW + kw
        s.AppendLine($"    mad.lo.u32 %r26, %r19, {I(khkw)}, %r18;");
        s.AppendLine($"    mad.lo.u32 %r26, %r21, {I(kw)}, %r26;");
        s.AppendLine("    add.u32 %r26, %r26, %r23;");
        s.AppendLine("    mul.wide.u32 %rd6, %r26, 4;");
        s.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
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
        s.AppendLine($"    setp.lt.u32 %p6, %r16, {I(c)};");
        s.AppendLine("    @%p6 bra LOOP_C;");
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
