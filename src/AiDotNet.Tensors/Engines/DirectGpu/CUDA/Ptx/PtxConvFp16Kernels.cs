using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX fused im2col + FP16 conversion (Im2colKNFp16): extracts sliding local
/// patches of input[N,C,H,W] and writes them as FP16 columns[N, C*KH*KW, OH*OW] for a
/// downstream Tensor-Core GEMM. columns[n, c*KH*KW+kh*KW+kw, oh*OW+ow] =
/// fp16(input[n,c,oh*s+kh-pad,ow*s+kw-pad]) (0 outside padded input). One thread per
/// output element; consecutive threads own consecutive output-spatial index so at
/// stride 1 the input reads and the FP16 column stores coalesce.
/// </summary>
internal sealed class PtxUnfold2DFp16Kernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
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
    internal int PatchRows => Channels * KernelH * KernelW;
    internal int Columns => OutH * OutW;
    internal long InputBytes => (long)Batch * Channels * Height * Width * sizeof(float);
    internal long OutputBytes => (long)Batch * PatchRows * Columns * sizeof(ushort);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_im2col_kn_fp16_n{Batch}_c{Channels}_h{Height}_w{Width}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}");

    internal PtxUnfold2DFp16Kernel(
        DirectPtxRuntime runtime, int batch, int channels, int height, int width, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Im2colKNFp16 has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * PatchRows * Columns % BlockThreads != 0)
            throw new ArgumentException($"N*(C*KH*KW)*(OH*OW) must be a multiple of {BlockThreads}.");

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
        var input = new DirectPtxExtent(Batch, Channels, Height, Width);
        var output = new DirectPtxExtent(Batch, PatchRows, Columns);
        return new DirectPtxKernelBlueprint(
            Operation: "im2col-kn-fp16", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{Channels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}-f32-to-f16"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("columns", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 32, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "columns_fp16[n,c*KH*KW+kh*KW+kw,oh*OW+ow] = fp16(input[n,c,oh*s+kh-pad,ow*s+kw-pad])",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView columns)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(columns, Blueprint.Tensors[1], nameof(columns));
        IntPtr iPtr = input.Pointer, oPtr = columns.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &iPtr; arguments[1] = &oPtr;
        int total = Batch * PatchRows * Columns;
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
            throw new NotSupportedException("Only the experimental SM86 Im2colKNFp16 emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = Channels, h = Height, w = Width, kw = KernelW, oww = OutW;
        int khkw = KernelH * kw, hw = h * w, cols = Columns, prc = PatchRows * cols;
        string entry = EntryPoint;

        var s = new StringBuilder(12288);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b16 %hf<2>;");
        s.AppendLine("    .reg .b32 %r<28>;");
        s.AppendLine("    .reg .b64 %rd<12>;");
        s.AppendLine("    .reg .f32 %f<4>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx
        s.AppendLine($"    div.u32 %r3, %r2, {I(prc)};");        // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(prc)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(cols)};");       // patchIdx
        s.AppendLine($"    rem.u32 %r6, %r4, {I(cols)};");       // spatialIdx
        s.AppendLine($"    div.u32 %r7, %r5, {I(khkw)};");       // c
        s.AppendLine($"    rem.u32 %r8, %r5, {I(khkw)};");
        s.AppendLine($"    div.u32 %r9, %r8, {I(kw)};");         // kh
        s.AppendLine($"    rem.u32 %r10, %r8, {I(kw)};");        // kw
        s.AppendLine($"    div.u32 %r11, %r6, {I(oww)};");       // oh
        s.AppendLine($"    rem.u32 %r12, %r6, {I(oww)};");       // ow
        s.AppendLine($"    mad.lo.u32 %r13, %r11, {I(Stride)}, %r9;");
        s.AppendLine($"    sub.s32 %r13, %r13, {I(Padding)};");  // ih
        s.AppendLine($"    mad.lo.u32 %r14, %r12, {I(Stride)}, %r10;");
        s.AppendLine($"    sub.s32 %r14, %r14, {I(Padding)};");  // iw
        s.AppendLine("    setp.ge.s32 %p0, %r13, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r13, {I(h)};");
        s.AppendLine("    setp.ge.s32 %p2, %r14, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r14, {I(w)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p0, %p0, %p2;");
        s.AppendLine($"    mad.lo.u32 %r15, %r3, {I(c)}, %r7;");
        s.AppendLine($"    mad.lo.u32 %r15, %r15, {I(hw)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(w)}, %r15;");
        s.AppendLine("    add.u32 %r15, %r15, %r14;");
        s.AppendLine("    mul.wide.u32 %rd2, %r15, 4;");
        s.AppendLine("    add.u64 %rd2, %rd0, %rd2;");
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine("    @%p0 ld.global.nc.f32 %f0, [%rd2];");
        s.AppendLine("    cvt.rn.f16.f32 %hf0, %f0;");
        s.AppendLine("    mul.wide.u32 %rd3, %r2, 2;");
        s.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        s.AppendLine("    st.global.b16 [%rd3], %hf0;");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Direct-PTX half-weight direct Conv2D (Conv2dDirectFp16Hw): FP32 input rounded to
/// FP16 and multiplied by FP16 weights with FP32 accumulation, per-channel bias +
/// optional ReLU. out[n,k,oh,ow] = relu(bias[k] + sum_{c,kh,kw} f16(W[k,c,kh,kw]) *
/// f16(in[n,c,oh*s+kh-pad,ow*s+kw-pad])). General kernel/stride/padding. One thread per
/// output element; consecutive threads own consecutive ow so the input reads and output
/// stores coalesce at stride 1, and the FP16 weights broadcast across the warp.
/// </summary>
internal sealed class PtxConv2DDirectFp16Kernel : IDisposable
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
    internal long WeightBytes => (long)OutputChannels * InputChannels * KernelH * KernelW * sizeof(ushort);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutH * OutW * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv2d_direct_fp16hw_n{Batch}_c{InputChannels}_k{OutputChannels}_h{Height}_w{Width}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}{(Relu ? "_relu" : "")}");

    internal PtxConv2DDirectFp16Kernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, bool relu = true)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv2dDirectFp16Hw has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; Relu = relu;
        if (OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");
        if ((long)batch * outputChannels * OutH * OutW % BlockThreads != 0)
            throw new ArgumentException($"N*K*OH*OW must be a multiple of {BlockThreads}.");

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
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelH, KernelW);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutH, OutW);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-direct-fp16hw", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-h{Height}-w{Width}-kh{KernelH}-kw{KernelW}-s{Stride}-p{Padding}{(Relu ? "-relu" : "")}-f16w-f32acc"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 48, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,oh,ow] = " + (Relu ? "relu(" : "(") + "bias[k] + sum f16(W[k,c,kh,kw])*f16(in[...]))",
                ["accumulate"] = "fp32", ["weights"] = "fp16 OIHW", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
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

    internal string EmitPtx(int major, int minor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 Conv2dDirectFp16Hw emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, k = OutputChannels, h = Height, w = Width, kh = KernelH, kw = KernelW, ohh = OutH, oww = OutW;
        int hw = h * w, chw = c * hw, ohow = ohh * oww, kohow = k * ohow, ckk = c * kh * kw, khkw = kh * kw;
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
        s.AppendLine("    .reg .b16 %hf<2>;");
        s.AppendLine("    .reg .b32 %r<40>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx
        s.AppendLine($"    div.u32 %r3, %r2, {I(kohow)};");      // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(kohow)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ohow)};");       // k
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ohow)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(oww)};");        // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(oww)};");        // ow
        s.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");       // acc = bias[k]
        s.AppendLine($"    mul.lo.u32 %r9, %r7, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r9, %r9, {I(Padding)};");    // ih0
        s.AppendLine($"    mul.lo.u32 %r10, %r8, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r10, %r10, {I(Padding)};");  // iw0
        s.AppendLine($"    mul.lo.u32 %r11, %r3, {I(chw)};");    // input batch base
        s.AppendLine($"    mul.lo.u32 %r12, %r5, {I(ckk)};");    // weight out-channel base
        s.AppendLine("    mov.u32 %r13, 0;");                    // cc
        s.AppendLine("LOOP_C:");
        s.AppendLine($"    mad.lo.u32 %r14, %r13, {I(hw)}, %r11;");
        s.AppendLine($"    mad.lo.u32 %r15, %r13, {I(khkw)}, %r12;");
        s.AppendLine("    mov.u32 %r16, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine("    add.s32 %r17, %r9, %r16;");            // ih
        s.AppendLine("    setp.ge.s32 %p0, %r17, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r17, {I(h)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    mov.u32 %r18, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine("    add.s32 %r19, %r10, %r18;");           // iw
        s.AppendLine("    setp.ge.s32 %p2, %r19, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r19, {I(w)};");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p2, %p2, %p0;");
        s.AppendLine($"    mad.lo.u32 %r20, %r17, {I(w)}, %r14;");
        s.AppendLine("    add.u32 %r20, %r20, %r19;");
        s.AppendLine("    mul.wide.u32 %rd5, %r20, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p2 ld.global.nc.f32 %f1, [%rd5];");
        s.AppendLine("    cvt.rn.f16.f32 %hf0, %f1;");           // round input to fp16
        s.AppendLine("    cvt.f32.f16 %f1, %hf0;");              // back to fp32 (fp16 precision)
        s.AppendLine($"    mad.lo.u32 %r21, %r16, {I(kw)}, %r15;");
        s.AppendLine("    add.u32 %r21, %r21, %r18;");           // weight index (fp16)
        s.AppendLine("    mul.wide.u32 %rd6, %r21, 2;");
        s.AppendLine("    add.u64 %rd6, %rd1, %rd6;");
        s.AppendLine("    ld.global.nc.b16 %hf1, [%rd6];");
        s.AppendLine("    cvt.f32.f16 %f2, %hf1;");              // weight fp16 -> fp32
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r18, %r18, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r18, {I(kw)};");
        s.AppendLine("    @%p4 bra LOOP_KW;");
        s.AppendLine("    add.u32 %r16, %r16, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r16, {I(kh)};");
        s.AppendLine("    @%p4 bra LOOP_KH;");
        s.AppendLine("    add.u32 %r13, %r13, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r13, {I(c)};");
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
