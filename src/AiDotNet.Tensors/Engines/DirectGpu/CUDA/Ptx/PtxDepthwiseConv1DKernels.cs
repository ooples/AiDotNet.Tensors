using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX native depthwise Conv1D forward (channel-multiplier 1):
/// out[n,c,ol] = sum_kl in[n,c,ol*stride+kl-pad] * W[c,kl]. Each channel convolves
/// with its own length-KL filter. One thread per output element; consecutive ol own
/// consecutive contiguous NCL positions so input reads (il = ol*stride+kl-pad) and
/// output stores coalesce at stride 1 -- the native 1D depthwise layout instead of
/// reshaping through Conv2D. Bounds-guarded ceil-div grid.
/// </summary>
internal sealed class PtxDepthwiseConv1DForwardKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
    internal int Length { get; }
    internal int KernelLength { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutLength => (Length + 2 * Padding - KernelLength) / Stride + 1;
    internal int TotalThreads => Batch * Channels * OutLength;
    internal long InputBytes => (long)Batch * Channels * Length * sizeof(float);
    internal long WeightBytes => (long)Channels * KernelLength * sizeof(float);
    internal long OutputBytes => (long)Batch * Channels * OutLength * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_dwconv1d_n{Batch}_c{Channels}_l{Length}_kl{KernelLength}_s{Stride}_p{Padding}");

    internal PtxDepthwiseConv1DForwardKernel(
        DirectPtxRuntime runtime, int batch, int channels, int length, int kernelLength, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Depthwise Conv1D has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || length <= 0 || kernelLength <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Length = length; KernelLength = kernelLength; Stride = stride; Padding = padding;
        if (OutLength <= 0) throw new ArgumentException("Non-positive output length.");

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
        var input = new DirectPtxExtent(Batch, Channels, Length);
        var weight = new DirectPtxExtent(Channels, KernelLength);
        var output = new DirectPtxExtent(Batch, Channels, OutLength);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv1d-forward", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{Channels}-l{Length}-kl{KernelLength}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 40, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,c,ol] = sum_kl in[n,c,ol*stride+kl-pad]*W[c,kl]",
                ["access"] = "thread-per-output, ol-contiguous -> coalesced at stride 1",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView weights, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(output, Blueprint.Tensors[2], nameof(output));
        IntPtr iPtr = input.Pointer, wPtr = weights.Pointer, oPtr = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &iPtr; arguments[1] = &wPtr; arguments[2] = &oPtr;
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
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
            throw new NotSupportedException("Only the experimental SM86 depthwise Conv1D emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int cch = Channels, kl = KernelLength, l = Length, ol = OutLength, col = cch * ol, total = TotalThreads;
        string entry = EntryPoint;

        var s = new StringBuilder(8192);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 weight_ptr,");
        s.AppendLine("    .param .u64 output_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<4>;");
        s.AppendLine("    .reg .b32 %r<24>;");
        s.AppendLine("    .reg .b64 %rd<12>;");
        s.AppendLine("    .reg .f32 %f<6>;");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");   // idx
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(col)};");     // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(col)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ol)};");      // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ol)};");      // ol
        s.AppendLine($"    mad.lo.u32 %r7, %r3, {I(cch)}, %r5;");
        s.AppendLine($"    mul.lo.u32 %r7, %r7, {I(l)};");    // input base (n,c)
        s.AppendLine($"    mul.lo.u32 %r8, %r5, {I(kl)};");   // weight base c*KL
        s.AppendLine($"    mul.lo.u32 %r9, %r6, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r9, %r9, {I(Padding)};"); // il0
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine("    mov.u32 %r10, 0;");                 // klc
        s.AppendLine("LOOP_KL:");
        s.AppendLine("    add.s32 %r11, %r9, %r10;");         // il
        s.AppendLine("    setp.ge.s32 %p1, %r11, 0;");
        s.AppendLine($"    setp.lt.s32 %p2, %r11, {I(l)};");
        s.AppendLine("    and.pred %p1, %p1, %p2;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    add.u32 %r12, %r7, %r11;");
        s.AppendLine("    mul.wide.u32 %rd3, %r12, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    @%p1 ld.global.nc.f32 %f1, [%rd3];");
        s.AppendLine("    add.u32 %r13, %r8, %r10;");
        s.AppendLine("    mul.wide.u32 %rd4, %r13, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r10, %r10, 1;");
        s.AppendLine($"    setp.lt.u32 %p1, %r10, {I(kl)};");
        s.AppendLine("    @%p1 bra LOOP_KL;");
        s.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        s.AppendLine("    st.global.f32 [%rd5], %f0;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Direct-PTX depthwise Conv1D backward-input:
/// dInput[n,c,il] = sum_kl gradOut[n,c,ol] * W[c,kl] where il = ol*stride+kl-pad, i.e.
/// ol = (il+pad-kl)/stride when that is a non-negative in-range multiple of stride
/// (the transpose-gather of the forward correlation). One thread per input element;
/// consecutive il own consecutive contiguous positions. Bounds-guarded ceil-div grid.
/// </summary>
internal sealed class PtxDepthwiseConv1DBackwardInputKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
    internal int Length { get; }
    internal int KernelLength { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutLength => (Length + 2 * Padding - KernelLength) / Stride + 1;
    internal int TotalThreads => Batch * Channels * Length;
    internal long GradOutputBytes => (long)Batch * Channels * OutLength * sizeof(float);
    internal long WeightBytes => (long)Channels * KernelLength * sizeof(float);
    internal long GradInputBytes => (long)Batch * Channels * Length * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_dwconv1d_bwd_input_n{Batch}_c{Channels}_l{Length}_kl{KernelLength}_s{Stride}_p{Padding}");

    internal PtxDepthwiseConv1DBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int channels, int length, int kernelLength, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Depthwise Conv1D backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || length <= 0 || kernelLength <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Length = length; KernelLength = kernelLength; Stride = stride; Padding = padding;
        if (OutLength <= 0) throw new ArgumentException("Non-positive output length.");

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
        var grad = new DirectPtxExtent(Batch, Channels, OutLength);
        var weight = new DirectPtxExtent(Channels, KernelLength);
        var dx = new DirectPtxExtent(Batch, Channels, Length);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv1d-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{Channels}-l{Length}-kl{KernelLength}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dx, dx, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 40, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dInput[n,c,il] = sum_kl gradOut[n,c,(il+pad-kl)/stride]*W[c,kl] over valid taps",
                ["access"] = "thread-per-input, il-contiguous; transpose-gather of forward",
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
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
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
            throw new NotSupportedException("Only the experimental SM86 depthwise Conv1D backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int cch = Channels, kl = KernelLength, l = Length, ol = OutLength, ccl = cch * l, total = TotalThreads;
        string entry = EntryPoint;

        var s = new StringBuilder(8192);
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
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b32 %r<28>;");
        s.AppendLine("    .reg .b64 %rd<12>;");
        s.AppendLine("    .reg .f32 %f<6>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");   // idx
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(ccl)};");    // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(ccl)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(l)};");      // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(l)};");      // il
        s.AppendLine($"    mad.lo.u32 %r7, %r3, {I(cch)}, %r5;");
        s.AppendLine($"    mul.lo.u32 %r7, %r7, {I(ol)};"); // gradOut base (n,c)
        s.AppendLine($"    mul.lo.u32 %r8, %r5, {I(kl)};");  // weight base c*KL
        s.AppendLine($"    add.s32 %r9, %r6, {I(Padding)};");// il+pad
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine("    mov.u32 %r10, 0;");                // klc
        s.AppendLine("LOOP_KL:");
        s.AppendLine("    sub.s32 %r11, %r9, %r10;");        // t = il+pad-kl
        s.AppendLine("    setp.ge.s32 %p1, %r11, 0;");
        s.AppendLine($"    rem.u32 %r12, %r11, {I(Stride)};"); // t % stride (r11>=0 when p1)
        s.AppendLine("    setp.eq.s32 %p2, %r12, 0;");
        s.AppendLine($"    div.u32 %r13, %r11, {I(Stride)};"); // ol = t/stride
        s.AppendLine($"    setp.lt.s32 %p3, %r13, {I(ol)};");
        s.AppendLine("    and.pred %p1, %p1, %p2;");
        s.AppendLine("    and.pred %p1, %p1, %p3;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    add.u32 %r14, %r7, %r13;");
        s.AppendLine("    mul.wide.u32 %rd3, %r14, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    @%p1 ld.global.nc.f32 %f1, [%rd3];");
        s.AppendLine("    add.u32 %r15, %r8, %r10;");
        s.AppendLine("    mul.wide.u32 %rd4, %r15, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    @%p1 fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r10, %r10, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r10, {I(kl)};");
        s.AppendLine("    @%p4 bra LOOP_KL;");
        s.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        s.AppendLine("    st.global.f32 [%rd5], %f0;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}

/// <summary>
/// Direct-PTX depthwise Conv1D backward-weight:
/// dW[c,kl] = sum_{n,ol} gradOut[n,c,ol] * in[n,c,ol*stride+kl-pad]. One thread per
/// weight element (C*KL is small) loops the batch and output-spatial axis with a
/// bounds-guarded ceil-div grid -- the reduction owner pattern without needing shared
/// memory since each thread owns a disjoint output.
/// </summary>
internal sealed class PtxDepthwiseConv1DBackwardWeightKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Channels { get; }
    internal int Length { get; }
    internal int KernelLength { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutLength => (Length + 2 * Padding - KernelLength) / Stride + 1;
    internal int TotalThreads => Channels * KernelLength;
    internal long GradOutputBytes => (long)Batch * Channels * OutLength * sizeof(float);
    internal long InputBytes => (long)Batch * Channels * Length * sizeof(float);
    internal long GradWeightBytes => (long)Channels * KernelLength * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_dwconv1d_bwd_weight_n{Batch}_c{Channels}_l{Length}_kl{KernelLength}_s{Stride}_p{Padding}");

    internal PtxDepthwiseConv1DBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int channels, int length, int kernelLength, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Depthwise Conv1D backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || channels <= 0 || length <= 0 || kernelLength <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; Channels = channels; Length = length; KernelLength = kernelLength; Stride = stride; Padding = padding;
        if (OutLength <= 0) throw new ArgumentException("Non-positive output length.");

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
        var grad = new DirectPtxExtent(Batch, Channels, OutLength);
        var input = new DirectPtxExtent(Batch, Channels, Length);
        var dw = new DirectPtxExtent(Channels, KernelLength);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv1d-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{Channels}-l{Length}-kl{KernelLength}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 80, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[c,kl] = sum_{n,ol} gradOut[n,c,ol]*in[n,c,ol*stride+kl-pad]",
                ["access"] = "thread-per-weight, batch+spatial reduction; bounds-guarded grid",
                ["shape-selection"] = "host-only-exact-contract", ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(DirectPtxTensorView gradOutput, DirectPtxTensorView input, DirectPtxTensorView gradWeights)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(gradWeights, Blueprint.Tensors[2], nameof(gradWeights));
        IntPtr gPtr = gradOutput.Pointer, iPtr = input.Pointer, wPtr = gradWeights.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gPtr; arguments[1] = &iPtr; arguments[2] = &wPtr;
        uint blocks = (uint)((TotalThreads + BlockThreads - 1) / BlockThreads);
        _module.Launch(_function, blocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
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
            throw new NotSupportedException("Only the experimental SM86 depthwise Conv1D backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int cch = Channels, kl = KernelLength, l = Length, ol = OutLength, total = TotalThreads;
        string entry = EntryPoint;

        var s = new StringBuilder(8192);
        s.AppendLine(".version 7.1");
        s.AppendLine($".target sm_{major}{minor}");
        s.AppendLine(".address_size 64");
        s.AppendLine();
        s.AppendLine($".visible .entry {entry}(");
        s.AppendLine("    .param .u64 grad_ptr,");
        s.AppendLine("    .param .u64 input_ptr,");
        s.AppendLine("    .param .u64 dw_ptr");
        s.AppendLine(")");
        s.AppendLine("{");
        s.AppendLine("    .reg .pred %p<6>;");
        s.AppendLine("    .reg .b32 %r<32>;");
        s.AppendLine("    .reg .b64 %rd<14>;");
        s.AppendLine("    .reg .f32 %f<6>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");   // idx = c*KL + kl
        s.AppendLine($"    setp.ge.u32 %p0, %r2, {I(total)};");
        s.AppendLine("    @%p0 bra END;");
        s.AppendLine($"    div.u32 %r3, %r2, {I(kl)};");     // c
        s.AppendLine($"    rem.u32 %r4, %r2, {I(kl)};");     // kl
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine("    mov.u32 %r5, 0;");                 // n
        s.AppendLine("LOOP_N:");
        s.AppendLine($"    mad.lo.u32 %r6, %r5, {I(cch)}, %r3;");
        s.AppendLine($"    mul.lo.u32 %r7, %r6, {I(ol)};");  // gradOut base (n,c)
        s.AppendLine($"    mul.lo.u32 %r8, %r6, {I(l)};");   // input base (n,c)
        s.AppendLine("    mov.u32 %r9, 0;");                 // ol
        s.AppendLine("LOOP_OL:");
        s.AppendLine($"    mul.lo.u32 %r10, %r9, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r10, %r10, {I(Padding)};");
        s.AppendLine("    add.s32 %r10, %r10, %r4;");        // il = ol*stride - pad + kl
        s.AppendLine("    setp.ge.s32 %p1, %r10, 0;");
        s.AppendLine($"    setp.lt.s32 %p2, %r10, {I(l)};");
        s.AppendLine("    and.pred %p1, %p1, %p2;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    mov.f32 %f2, 0f00000000;");
        s.AppendLine("    add.u32 %r11, %r7, %r9;");
        s.AppendLine("    mul.wide.u32 %rd3, %r11, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    @%p1 ld.global.nc.f32 %f1, [%rd3];");  // gradOut[n,c,ol]
        s.AppendLine("    add.u32 %r12, %r8, %r10;");
        s.AppendLine("    mul.wide.u32 %rd4, %r12, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    @%p1 ld.global.nc.f32 %f2, [%rd4];");  // input[n,c,il]
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r9, %r9, 1;");
        s.AppendLine($"    setp.lt.u32 %p3, %r9, {I(ol)};");
        s.AppendLine("    @%p3 bra LOOP_OL;");
        s.AppendLine("    add.u32 %r5, %r5, 1;");
        s.AppendLine($"    setp.lt.u32 %p4, %r5, {I(Batch)};");
        s.AppendLine("    @%p4 bra LOOP_N;");
        s.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        s.AppendLine("    add.u64 %rd5, %rd2, %rd5;");
        s.AppendLine("    st.global.f32 [%rd5], %f0;");
        s.AppendLine("END:");
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
