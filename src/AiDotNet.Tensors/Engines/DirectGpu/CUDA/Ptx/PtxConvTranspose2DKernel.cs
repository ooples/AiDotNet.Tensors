using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX ConvTranspose2D (transposed convolution) forward with per-output-channel
/// bias and optional ReLU. Weights are IOHW [Cin, Cout, KH, KW]; output is
/// out[n,co,oh,ow] = relu(bias[co] + sum over (ci,kh,kw) with ih=(oh+pad-kh)/stride and
/// iw=(ow+pad-kw)/stride valid (non-negative, divisible, in-range) of
/// input[n,ci,ih,iw] * W[ci,co,kh,kw]). This is the transpose-gather pattern (same shape
/// as a regular conv's backward-input) run as a forward op. One thread per output element;
/// consecutive threads own consecutive ow so at stride 1 the input reads and output stores
/// coalesce (the contiguous NCHW axis) and the weights broadcast across the warp.
/// </summary>
internal sealed class PtxConvTranspose2DKernel : IDisposable
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
    internal bool Relu { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutHeight => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutWidth => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)InputChannels * OutputChannels * KernelH * KernelW * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutHeight * OutWidth * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_convtranspose2d_n{Batch}_ci{InputChannels}_co{OutputChannels}_h{Height}_w{Width}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}_op{OutputPadding}{(Relu ? "_relu" : "")}");

    internal PtxConvTranspose2DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int outputPadding, bool relu = true)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("ConvTranspose2D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; OutputPadding = outputPadding; Relu = relu;
        if (OutHeight <= 0 || OutWidth <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * outputChannels * OutHeight * OutWidth % BlockThreads != 0)
            throw new ArgumentException($"N*Cout*OH*OW must be a multiple of {BlockThreads}.");

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
        var weight = new DirectPtxExtent(InputChannels, OutputChannels, KernelH, KernelW);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "convtranspose2d-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-ci{InputChannels}-co{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-op{OutputPadding}{(Relu ? "-relu" : "")}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,co,oh,ow] = " + (Relu ? "relu(" : "(") + "bias[co] + sum input[n,ci,(oh+pad-kh)/s,(ow+pad-kw)/s]*W[ci,co,kh,kw])",
                ["weights"] = "IOHW", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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
        int total = Batch * OutputChannels * OutHeight * OutWidth;
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
            throw new NotSupportedException("Only the experimental SM86 ConvTranspose2D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int ci = InputChannels, co = OutputChannels, h = Height, w = Width, kh = KernelH, kw = KernelW;
        int oh = OutHeight, ow = OutWidth;
        int hw = h * w, cohw = co * oh * ow, cihw = ci * hw, cokk = co * kh * kw, khkw = kh * kw;
        string entry = EntryPoint;

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
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*Co*OH*OW + co*OH*OW + oh*OW + ow
        s.AppendLine($"    div.u32 %r3, %r2, {I(cohw)};");        // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(cohw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(oh * ow)};");     // co
        s.AppendLine($"    rem.u32 %r6, %r4, {I(oh * ow)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(ow)};");          // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(ow)};");          // ow
        s.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");       // acc = bias[co]
        s.AppendLine($"    mul.lo.u32 %r9, %r3, {I(cihw)};");    // input batch base = n*Ci*H*W
        s.AppendLine("    mov.u32 %r10, 0;");                    // cc = 0 (input channel)
        s.AppendLine("LOOP_CI:");
        // input channel base = n*Ci*HW + cc*HW ; weight (cc,co) base = cc*Co*KH*KW + co*KH*KW
        s.AppendLine($"    mad.lo.u32 %r11, %r10, {I(hw)}, %r9;");
        s.AppendLine($"    mad.lo.u32 %r12, %r10, {I(cokk)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r12, %r5, {I(khkw)}, %r12;");
        s.AppendLine("    mov.u32 %r13, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        // numH = oh + pad - kh
        s.AppendLine($"    add.s32 %r14, %r7, {I(Padding)};");
        s.AppendLine("    sub.s32 %r14, %r14, %r13;");
        s.AppendLine("    setp.ge.s32 %p0, %r14, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r15, %r14;");
        else
        {
            s.AppendLine($"    rem.s32 %r16, %r14, {I(Stride)};");
            s.AppendLine("    setp.eq.s32 %p1, %r16, 0;");
            s.AppendLine("    and.pred %p0, %p0, %p1;");
            s.AppendLine($"    div.s32 %r15, %r14, {I(Stride)};");   // ih
        }
        s.AppendLine($"    setp.lt.s32 %p2, %r15, {I(h)};");
        s.AppendLine("    and.pred %p0, %p0, %p2;");             // ih valid
        s.AppendLine("    mov.u32 %r17, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine($"    add.s32 %r18, %r8, {I(Padding)};");
        s.AppendLine("    sub.s32 %r18, %r18, %r17;");           // numW
        s.AppendLine("    setp.ge.s32 %p3, %r18, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r19, %r18;");
        else
        {
            s.AppendLine($"    rem.s32 %r20, %r18, {I(Stride)};");
            s.AppendLine("    setp.eq.s32 %p4, %r20, 0;");
            s.AppendLine("    and.pred %p3, %p3, %p4;");
            s.AppendLine($"    div.s32 %r19, %r18, {I(Stride)};");   // iw
        }
        s.AppendLine($"    setp.lt.s32 %p5, %r19, {I(w)};");
        s.AppendLine("    and.pred %p3, %p3, %p5;");
        s.AppendLine("    and.pred %p3, %p3, %p0;");             // both ih and iw valid
        // input[n][cc][ih][iw] = base r11 + ih*W + iw
        s.AppendLine($"    mad.lo.u32 %r21, %r15, {I(w)}, %r11;");
        s.AppendLine("    add.u32 %r21, %r21, %r19;");
        s.AppendLine("    mul.wide.u32 %rd5, %r21, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p3 ld.global.nc.f32 %f1, [%rd5];");
        // weight[cc][co][kh][kw] = r12 + kh*KW + kw
        s.AppendLine($"    mad.lo.u32 %r22, %r13, {I(kw)}, %r12;");
        s.AppendLine("    add.u32 %r22, %r22, %r17;");
        s.AppendLine("    mul.wide.u32 %rd6, %r22, 4;");
        s.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r17, %r17, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r17, {I(kw)};");
        s.AppendLine("    @%p6 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r13, %r13, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r13, {I(kh)};");
        s.AppendLine("    @%p6 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r10, %r10, 1;");
        s.AppendLine($"    setp.lt.u32 %p6, %r10, {I(ci)};");
        s.AppendLine("    @%p6 bra LOOP_CI;");
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
