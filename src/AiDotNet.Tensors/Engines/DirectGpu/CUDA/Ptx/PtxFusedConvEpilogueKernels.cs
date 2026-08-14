using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Shape identity for <see cref="PtxFusedConv3DKernel"/>. Carries the full specialization so
/// PTX and the blueprint can be re-emitted device-free (for the compiled-cubin verify gate).
/// </summary>
internal readonly record struct FusedConv3DShape(
    int Batch, int InputChannels, int OutputChannels, int Depth, int Height, int Width,
    int KernelD, int KernelH, int KernelW, int Stride, int Padding)
{
    internal int OutD => (Depth + 2 * Padding - KernelD) / Stride + 1;
    internal int OutH => (Height + 2 * Padding - KernelH) / Stride + 1;
    internal int OutW => (Width + 2 * Padding - KernelW) / Stride + 1;
    internal int TotalThreads => Batch * OutputChannels * OutD * OutH * OutW;

    internal string Entry => FormattableString.Invariant(
        $"aidotnet_fused_conv3d_n{Batch}_c{InputChannels}_k{OutputChannels}_d{Depth}_h{Height}_w{Width}_kd{KernelD}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}");
}

/// <summary>
/// Direct-PTX fused Conv3D inference epilogue: convolution + per-output-channel bias +
/// per-output-channel scale + ReLU in one pass, no intermediate materialization:
/// out[n,k,od,oh,ow] = relu(scale[k] * (bias[k] + sum_{c,kd,kh,kw} W * in[...])).
/// One thread per output element; consecutive ow -> coalesced innermost NCDHW reads/stores.
/// Bounds-guarded ceil-div grid. Same conv arithmetic as PtxConv3DKernel with the scale+ReLU
/// epilogue folded into the accumulator before the store.
/// </summary>
internal sealed class PtxFusedConv3DKernel : IDisposable
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
    internal int TotalThreads => Batch * OutputChannels * OutD * OutH * OutW;
    internal long InputBytes => (long)Batch * InputChannels * Depth * Height * Width * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * KernelD * KernelH * KernelW * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long ScaleBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutD * OutH * OutW * sizeof(float);

    internal FusedConv3DShape Shape => new(Batch, InputChannels, OutputChannels, Depth, Height, Width, KernelD, KernelH, KernelW, Stride, Padding);
    internal string EntryPoint => Shape.Entry;

    internal PtxFusedConv3DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int depth, int height, int width, int kernelD, int kernelH, int kernelW, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Fused Conv3D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || depth <= 0 || height <= 0 || width <= 0 || kernelD <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Depth = depth; Height = height; Width = width; KernelD = kernelD; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding;
        if (OutD <= 0 || OutH <= 0 || OutW <= 0) throw new ArgumentException("Non-positive output spatial.");

        FusedConv3DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, FusedConv3DShape shape)
    {
        int batch = shape.Batch, inputChannels = shape.InputChannels, outputChannels = shape.OutputChannels;
        int depth = shape.Depth, height = shape.Height, width = shape.Width;
        int kernelD = shape.KernelD, kernelH = shape.KernelH, kernelW = shape.KernelW, stride = shape.Stride, padding = shape.Padding;
        int outD = shape.OutD, outH = shape.OutH, outW = shape.OutW;
        var input = new DirectPtxExtent(batch, inputChannels * depth, height, width);
        var weight = new DirectPtxExtent(outputChannels, inputChannels * kernelD, kernelH, kernelW);
        var bias = new DirectPtxExtent(outputChannels);
        var scale = new DirectPtxExtent(outputChannels);
        var output = new DirectPtxExtent(batch, outputChannels * outD, outH, outW);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-conv3d-bias-scale-relu", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{batch}-c{inputChannels}-k{outputChannels}-d{depth}-h{height}-w{width}-kd{kernelD}-kh{kernelH}-kw{kernelW}-s{stride}-p{padding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("scale", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, scale, scale, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 64, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,od,oh,ow] = relu(scale[k]*(bias[k] + sum W[k,c,kd,kh,kw]*in[n,c,od*s+kd-pad,oh*s+kh-pad,ow*s+kw-pad]))",
                ["epilogue"] = "bias + per-channel scale + ReLU fused", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView bias, DirectPtxTensorView scale, DirectPtxTensorView output)
    {
        DirectPtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbiGuard.Require(weights, Blueprint.Tensors[1], nameof(weights));
        DirectPtxAbiGuard.Require(bias, Blueprint.Tensors[2], nameof(bias));
        DirectPtxAbiGuard.Require(scale, Blueprint.Tensors[3], nameof(scale));
        DirectPtxAbiGuard.Require(output, Blueprint.Tensors[4], nameof(output));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, bPtr = bias.Pointer, sPtr = scale.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &bPtr; arguments[3] = &sPtr; arguments[4] = &oPtr;
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, FusedConv3DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 fused Conv3D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding;
        int c = shape.InputChannels, k = shape.OutputChannels, d = shape.Depth, h = shape.Height, w = shape.Width;
        int kd = shape.KernelD, khh = shape.KernelH, kw = shape.KernelW;
        int od = shape.OutD, oh = shape.OutH, ow = shape.OutW;
        int dhw = d * h * w, hw = h * w, odohow = od * oh * ow, ohow = oh * ow;
        int cdhw = c * dhw, kodohow = k * odohow, ckdkhkw = c * kd * khh * kw, kdkhkw = kd * khh * kw, khkw = khh * kw;
        int total = shape.TotalThreads;
        string entry = shape.Entry;

        var s = new StringBuilder(26624);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 scale_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<10>;");
        s.AppendLine("    .reg .b32 %r<48>;");
        s.AppendLine("    .reg .b64 %rd<20>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd8, [scale_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");
        s.AppendLine($"    setp.ge.u32 %p7, %r2, {I(total)};");
        s.AppendLine("    @%p7 bra END;");
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
        s.AppendLine($"    mad.lo.u32 %r17, %r16, {I(dhw)}, %r14;");
        s.AppendLine($"    mad.lo.u32 %r18, %r16, {I(kdkhkw)}, %r15;");
        s.AppendLine("    mov.u32 %r19, 0;");                    // kdc
        s.AppendLine("LOOP_KD:");
        s.AppendLine("    add.s32 %r20, %r11, %r19;");
        s.AppendLine("    setp.ge.s32 %p0, %r20, 0;");
        s.AppendLine($"    setp.lt.s32 %p1, %r20, {I(d)};");
        s.AppendLine("    and.pred %p0, %p0, %p1;");
        s.AppendLine("    mov.u32 %r21, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine("    add.s32 %r22, %r12, %r21;");
        s.AppendLine("    setp.ge.s32 %p2, %r22, 0;");
        s.AppendLine($"    setp.lt.s32 %p3, %r22, {I(h)};");
        s.AppendLine("    and.pred %p2, %p2, %p3;");
        s.AppendLine("    and.pred %p2, %p2, %p0;");
        s.AppendLine("    mov.u32 %r23, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine("    add.s32 %r24, %r13, %r23;");
        s.AppendLine("    setp.ge.s32 %p4, %r24, 0;");
        s.AppendLine($"    setp.lt.s32 %p5, %r24, {I(w)};");
        s.AppendLine("    and.pred %p4, %p4, %p5;");
        s.AppendLine("    and.pred %p4, %p4, %p2;");
        s.AppendLine($"    mad.lo.u32 %r25, %r20, {I(hw)}, %r17;");
        s.AppendLine($"    mad.lo.u32 %r25, %r22, {I(w)}, %r25;");
        s.AppendLine("    add.u32 %r25, %r25, %r24;");
        s.AppendLine("    mul.wide.u32 %rd5, %r25, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p4 ld.global.nc.f32 %f1, [%rd5];");
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
        // epilogue: acc = relu(scale[k] * acc)
        s.AppendLine("    mul.wide.u32 %rd9, %r5, 4;");
        s.AppendLine("    add.u64 %rd9, %rd8, %rd9;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd9];");      // scale[k]
        s.AppendLine("    mul.rn.f32 %f0, %f0, %f3;");
        s.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        s.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    st.global.f32 [%rd7], %f0;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Shape identity for <see cref="PtxFusedConvTranspose2DKernel"/> for device-free re-emit.
/// </summary>
internal readonly record struct FusedConvTranspose2DShape(
    int Batch, int InputChannels, int OutputChannels, int Height, int Width,
    int KernelH, int KernelW, int Stride, int Padding, int OutputPadding)
{
    internal int OutHeight => (Height - 1) * Stride - 2 * Padding + KernelH + OutputPadding;
    internal int OutWidth => (Width - 1) * Stride - 2 * Padding + KernelW + OutputPadding;
    internal int TotalThreads => Batch * OutputChannels * OutHeight * OutWidth;

    internal string Entry => FormattableString.Invariant(
        $"aidotnet_fused_convtranspose2d_n{Batch}_ci{InputChannels}_co{OutputChannels}_h{Height}_w{Width}_kh{KernelH}_kw{KernelW}_s{Stride}_p{Padding}_op{OutputPadding}");
}

/// <summary>
/// Direct-PTX fused ConvTranspose2D inference epilogue: transposed convolution +
/// per-output-channel bias + per-output-channel scale + ReLU in one pass:
/// out[n,co,oh,ow] = relu(scale[co] * (bias[co] + sum input[n,ci,(oh+pad-kh)/s,(ow+pad-kw)/s]*W[ci,co,kh,kw])).
/// Weights IOHW. Transpose-gather with valid-index checks; one thread per output element,
/// consecutive ow coalesced. Bounds-guarded ceil-div grid. Same arithmetic as
/// PtxConvTranspose2DKernel with the scale+ReLU epilogue folded in before the store.
/// </summary>
internal sealed class PtxFusedConvTranspose2DKernel : IDisposable
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
    internal int TotalThreads => Batch * OutputChannels * OutHeight * OutWidth;
    internal long InputBytes => (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal long WeightBytes => (long)InputChannels * OutputChannels * KernelH * KernelW * sizeof(float);
    internal long BiasBytes => (long)OutputChannels * sizeof(float);
    internal long ScaleBytes => (long)OutputChannels * sizeof(float);
    internal long OutputBytes => (long)Batch * OutputChannels * OutHeight * OutWidth * sizeof(float);

    internal FusedConvTranspose2DShape Shape => new(Batch, InputChannels, OutputChannels, Height, Width, KernelH, KernelW, Stride, Padding, OutputPadding);
    internal string EntryPoint => Shape.Entry;

    internal PtxFusedConvTranspose2DKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int height, int width, int kernelH, int kernelW, int stride, int padding, int outputPadding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Fused ConvTranspose2D has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || height <= 0 || width <= 0 || kernelH <= 0 || kernelW <= 0 || stride <= 0 || padding < 0 || outputPadding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Height = height; Width = width; KernelH = kernelH; KernelW = kernelW; Stride = stride; Padding = padding; OutputPadding = outputPadding;
        if (OutHeight <= 0 || OutWidth <= 0) throw new ArgumentException("Non-positive output spatial.");

        FusedConvTranspose2DShape shape = Shape;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, shape);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, shape);
        _module = runtime.LoadModule(Ptx, allowExperimentalJitFallback: DirectPtxFeatureGate.ConvolutionExperimentOverride);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo, BlockThreads, activeBlocks, _module);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, FusedConvTranspose2DShape shape)
    {
        int batch = shape.Batch, inputChannels = shape.InputChannels, outputChannels = shape.OutputChannels;
        int height = shape.Height, width = shape.Width, kernelH = shape.KernelH, kernelW = shape.KernelW;
        int stride = shape.Stride, padding = shape.Padding, outputPadding = shape.OutputPadding;
        int outHeight = shape.OutHeight, outWidth = shape.OutWidth;
        var input = new DirectPtxExtent(batch, inputChannels, height, width);
        var weight = new DirectPtxExtent(inputChannels, outputChannels, kernelH, kernelW);
        var bias = new DirectPtxExtent(outputChannels);
        var scale = new DirectPtxExtent(outputChannels);
        var output = new DirectPtxExtent(batch, outputChannels, outHeight, outWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "fused-convtranspose2d-bias-scale-relu", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{batch}-ci{inputChannels}-co{outputChannels}-h{height}-w{width}-kh{kernelH}-kw{kernelW}-s{stride}-p{padding}-op{outputPadding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("scale", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector, scale, scale, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 56, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,co,oh,ow] = relu(scale[co]*(bias[co] + sum input[n,ci,(oh+pad-kh)/s,(ow+pad-kw)/s]*W[ci,co,kh,kw]))",
                ["weights"] = "IOHW", ["epilogue"] = "bias + per-channel scale + ReLU fused", ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView bias, DirectPtxTensorView scale, DirectPtxTensorView output)
    {
        DirectPtxAbiGuard.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbiGuard.Require(weights, Blueprint.Tensors[1], nameof(weights));
        DirectPtxAbiGuard.Require(bias, Blueprint.Tensors[2], nameof(bias));
        DirectPtxAbiGuard.Require(scale, Blueprint.Tensors[3], nameof(scale));
        DirectPtxAbiGuard.Require(output, Blueprint.Tensors[4], nameof(output));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, bPtr = bias.Pointer, sPtr = scale.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &bPtr; arguments[3] = &sPtr; arguments[4] = &oPtr;
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }


    internal static string EmitPtx(int major, int minor, FusedConvTranspose2DShape shape)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(major, minor))
            throw new NotSupportedException("Only the experimental SM86 fused ConvTranspose2D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int Stride = shape.Stride, Padding = shape.Padding;
        int ci = shape.InputChannels, co = shape.OutputChannels, h = shape.Height, w = shape.Width, kh = shape.KernelH, kw = shape.KernelW;
        int oh = shape.OutHeight, ow = shape.OutWidth;
        int hw = h * w, cohw = co * oh * ow, cihw = ci * hw, cokk = co * kh * kw, khkw = kh * kw;
        int total = shape.TotalThreads;
        string entry = shape.Entry;

        var s = new StringBuilder(18432);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 bias_ptr,");
        s.AppendLine("    .param .u64 scale_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<10>;");
        s.AppendLine("    .reg .b32 %r<40>;");
        s.AppendLine("    .reg .b64 %rd<20>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        s.AppendLine("    ld.param.u64 %rd8, [scale_ptr];");
        s.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");
        s.AppendLine($"    setp.ge.u32 %p7, %r2, {I(total)};");
        s.AppendLine("    @%p7 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(cohw)};");        // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(cohw)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(oh * ow)};");     // co
        s.AppendLine($"    rem.u32 %r6, %r4, {I(oh * ow)};");
        s.AppendLine($"    div.u32 %r7, %r6, {I(ow)};");          // oh
        s.AppendLine($"    rem.u32 %r8, %r6, {I(ow)};");          // ow
        s.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        s.AppendLine("    add.u64 %rd4, %rd2, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");       // acc = bias[co]
        s.AppendLine($"    mul.lo.u32 %r9, %r3, {I(cihw)};");    // input batch base
        s.AppendLine("    mov.u32 %r10, 0;");                    // cc (input channel)
        s.AppendLine("LOOP_CI:");
        s.AppendLine($"    mad.lo.u32 %r11, %r10, {I(hw)}, %r9;");
        s.AppendLine($"    mul.lo.u32 %r12, %r10, {I(cokk)};");
        s.AppendLine($"    mad.lo.u32 %r12, %r5, {I(khkw)}, %r12;");
        s.AppendLine("    mov.u32 %r13, 0;");                    // khc
        s.AppendLine("LOOP_KH:");
        s.AppendLine($"    add.s32 %r14, %r7, {I(Padding)};");
        s.AppendLine("    sub.s32 %r14, %r14, %r13;");
        s.AppendLine("    setp.ge.s32 %p0, %r14, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r15, %r14;");
        else
        {
            s.AppendLine($"    rem.s32 %r16, %r14, {I(Stride)};");
            s.AppendLine("    setp.eq.s32 %p1, %r16, 0;");
            s.AppendLine("    and.pred %p0, %p0, %p1;");
            s.AppendLine($"    div.s32 %r15, %r14, {I(Stride)};");
        }
        s.AppendLine($"    setp.lt.s32 %p2, %r15, {I(h)};");
        s.AppendLine("    and.pred %p0, %p0, %p2;");
        s.AppendLine("    mov.u32 %r17, 0;");                    // kwc
        s.AppendLine("LOOP_KW:");
        s.AppendLine($"    add.s32 %r18, %r8, {I(Padding)};");
        s.AppendLine("    sub.s32 %r18, %r18, %r17;");
        s.AppendLine("    setp.ge.s32 %p3, %r18, 0;");
        if (Stride == 1) s.AppendLine("    mov.u32 %r19, %r18;");
        else
        {
            s.AppendLine($"    rem.s32 %r20, %r18, {I(Stride)};");
            s.AppendLine("    setp.eq.s32 %p4, %r20, 0;");
            s.AppendLine("    and.pred %p3, %p3, %p4;");
            s.AppendLine($"    div.s32 %r19, %r18, {I(Stride)};");
        }
        s.AppendLine($"    setp.lt.s32 %p5, %r19, {I(w)};");
        s.AppendLine("    and.pred %p3, %p3, %p5;");
        s.AppendLine("    and.pred %p3, %p3, %p0;");
        s.AppendLine($"    mad.lo.u32 %r21, %r15, {I(w)}, %r11;");
        s.AppendLine("    add.u32 %r21, %r21, %r19;");
        s.AppendLine("    mul.wide.u32 %rd5, %r21, 4;");
        s.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    @%p3 ld.global.nc.f32 %f1, [%rd5];");
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
        // epilogue: acc = relu(scale[co] * acc)
        s.AppendLine("    mul.wide.u32 %rd9, %r5, 4;");
        s.AppendLine("    add.u64 %rd9, %rd8, %rd9;");
        s.AppendLine("    ld.global.nc.f32 %f3, [%rd9];");      // scale[co]
        s.AppendLine("    mul.rn.f32 %f0, %f0, %f3;");
        s.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        s.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        s.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        s.AppendLine("    st.global.f32 [%rd7], %f0;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
