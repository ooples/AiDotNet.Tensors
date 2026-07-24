using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct-PTX native Conv1D backward-input: dX[n,c,il] = sum over (k, kl) with
/// (il + pad - kl) divisible by stride and ol = (il+pad-kl)/stride in range, of
/// W[k,c,kl] * gradOut[n,k,ol]. General kernel length / stride / padding. One thread
/// per input-gradient element; consecutive threads own consecutive il so gradOut
/// reads (at stride 1) and dX stores coalesce and the weights broadcast across the warp.
/// </summary>
internal sealed class PtxConv1DBackwardInputKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int OutputChannels { get; }
    internal int Length { get; }
    internal int KernelLength { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutLength => (Length + 2 * Padding - KernelLength) / Stride + 1;
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutLength * sizeof(float);
    internal long WeightBytes => (long)OutputChannels * InputChannels * KernelLength * sizeof(float);
    internal long GradInputBytes => (long)Batch * InputChannels * Length * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv1d_bwd_input_n{Batch}_c{InputChannels}_k{OutputChannels}_l{Length}_kl{KernelLength}_s{Stride}_p{Padding}");

    internal PtxConv1DBackwardInputKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int length, int kernelLength, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv1D backward-input has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || length <= 0 || kernelLength <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Length = length; KernelLength = kernelLength; Stride = stride; Padding = padding;
        if (OutLength <= 0) throw new ArgumentException("Non-positive output length.");
        if ((long)batch * inputChannels * length % BlockThreads != 0)
            throw new ArgumentException($"N*C*L must be a multiple of {BlockThreads}.");

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
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutLength);
        var weight = new DirectPtxExtent(OutputChannels, InputChannels, KernelLength);
        var dx = new DirectPtxExtent(Batch, InputChannels, Length);
        return new DirectPtxKernelBlueprint(
            Operation: "conv1d-backward-input", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-l{Length}-kl{KernelLength}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, weight, weight, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradInput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dx, dx, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 40, MaxStaticSharedBytes: 0, MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dX[n,c,il] = sum_{k,kl} W[k,c,kl]*gradOut[n,k,(il+pad-kl)/stride]",
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
        int total = Batch * InputChannels * Length;
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
            throw new NotSupportedException("Only the experimental SM86 Conv1D backward-input emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, kl = KernelLength, l = Length, ol = OutLength, kk = OutputChannels;
        int cl = c * l, kol = kk * ol, ckl = c * kl;
        string entry = EntryPoint;

        var s = new StringBuilder(12288);
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
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine("    .reg .f32 %f<8>;");
        s.AppendLine("    ld.param.u64 %rd0, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [weight_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dx_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");
        s.AppendLine($"    mad.lo.u32 %r2, %r1, {I(BlockThreads)}, %r0;");  // idx = n*C*L + c*L + il
        s.AppendLine($"    div.u32 %r3, %r2, {I(cl)};");         // n
        s.AppendLine($"    rem.u32 %r4, %r2, {I(cl)};");
        s.AppendLine($"    div.u32 %r5, %r4, {I(l)};");          // c
        s.AppendLine($"    rem.u32 %r6, %r4, {I(l)};");          // il
        s.AppendLine("    mov.f32 %f0, 0f00000000;");
        s.AppendLine($"    mul.lo.u32 %r7, %r3, {I(kol)};");     // gradOut batch base = n*K*OL
        s.AppendLine("    mov.u32 %r8, 0;");                     // kk = 0
        s.AppendLine("LOOP_K:");
        s.AppendLine($"    mad.lo.u32 %r9, %r8, {I(ol)}, %r7;"); // gradOut[n][kk] base = n*K*OL + kk*OL
        // weight (kk, c) base = kk*C*KL + c*KL
        s.AppendLine($"    mad.lo.u32 %r10, %r8, {I(ckl)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r10, %r5, {I(kl)}, %r10;");
        s.AppendLine("    mov.u32 %r11, 0;");                    // klc = 0
        s.AppendLine("LOOP_KL:");
        // num = il + pad - kl
        s.AppendLine($"    add.s32 %r12, %r6, {I(Padding)};");
        s.AppendLine("    sub.s32 %r12, %r12, %r11;");           // num
        s.AppendLine("    setp.ge.s32 %p0, %r12, 0;");
        if (Stride == 1)
        {
            s.AppendLine("    mov.u32 %r13, %r12;");             // ol = num
        }
        else
        {
            s.AppendLine($"    rem.s32 %r14, %r12, {I(Stride)};");
            s.AppendLine("    setp.eq.s32 %p1, %r14, 0;");
            s.AppendLine("    and.pred %p0, %p0, %p1;");
            s.AppendLine($"    div.s32 %r13, %r12, {I(Stride)};");   // ol (valid only if divisible & >=0)
        }
        s.AppendLine($"    setp.lt.s32 %p2, %r13, {I(ol)};");
        s.AppendLine("    and.pred %p0, %p0, %p2;");
        s.AppendLine("    mov.f32 %f1, 0f00000000;");
        s.AppendLine("    add.s32 %r15, %r9, %r13;");            // gradOut index
        s.AppendLine("    mul.wide.s32 %rd3, %r15, 4;");
        s.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        s.AppendLine("    @%p0 ld.global.nc.f32 %f1, [%rd3];");
        s.AppendLine("    add.u32 %r16, %r10, %r11;");           // weight index
        s.AppendLine("    mul.wide.u32 %rd4, %r16, 4;");
        s.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        s.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
        s.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        s.AppendLine("    add.u32 %r11, %r11, 1;");
        s.AppendLine($"    setp.lt.u32 %p3, %r11, {I(kl)};");
        s.AppendLine("    @%p3 bra LOOP_KL;");
        s.AppendLine("    add.u32 %r8, %r8, 1;");
        s.AppendLine($"    setp.lt.u32 %p3, %r8, {I(kk)};");
        s.AppendLine("    @%p3 bra LOOP_K;");
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
/// Direct-PTX native Conv1D backward-weight: dW[k,c,kl] = sum over (n, ol) of
/// input[n,c,ol*stride+kl-pad] * gradOut[n,k,ol]. One block per (k,c) pair reduces the
/// N x OL contraction into the KL taps with coalesced spatial reads (consecutive
/// threads walk the flattened (n,ol) index, gradOut reused across taps) and a shared
/// tree reduction per tap. General kernel length / stride / padding.
/// </summary>
internal sealed class PtxConv1DBackwardWeightKernel : IDisposable
{
    internal const int BlockThreads = 256;
    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int InputChannels { get; }
    internal int OutputChannels { get; }
    internal int Length { get; }
    internal int KernelLength { get; }
    internal int Stride { get; }
    internal int Padding { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal int OutLength => (Length + 2 * Padding - KernelLength) / Stride + 1;
    internal long InputBytes => (long)Batch * InputChannels * Length * sizeof(float);
    internal long GradOutputBytes => (long)Batch * OutputChannels * OutLength * sizeof(float);
    internal long GradWeightBytes => (long)OutputChannels * InputChannels * KernelLength * sizeof(float);

    internal string EntryPoint => FormattableString.Invariant(
        $"aidotnet_conv1d_bwd_weight_n{Batch}_c{InputChannels}_k{OutputChannels}_l{Length}_kl{KernelLength}_s{Stride}_p{Padding}");

    internal PtxConv1DBackwardWeightKernel(
        DirectPtxRuntime runtime, int batch, int inputChannels, int outputChannels,
        int length, int kernelLength, int stride, int padding)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException("Conv1D backward-weight has no experimental non-SM86 specialization.");
        if (batch <= 0 || inputChannels <= 0 || outputChannels <= 0 || length <= 0 || kernelLength <= 0 || stride <= 0 || padding < 0)
            throw new ArgumentOutOfRangeException(nameof(batch));
        if (kernelLength > 11) throw new ArgumentOutOfRangeException(nameof(kernelLength), "KL <= 11 (per-tap accumulators).");
        Batch = batch; InputChannels = inputChannels; OutputChannels = outputChannels;
        Length = length; KernelLength = kernelLength; Stride = stride; Padding = padding;
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
        var input = new DirectPtxExtent(Batch, InputChannels, Length);
        var grad = new DirectPtxExtent(Batch, OutputChannels, OutLength);
        var dw = new DirectPtxExtent(OutputChannels, InputChannels, KernelLength);
        return new DirectPtxKernelBlueprint(
            Operation: "conv1d-backward-weight", Version: 1, Architecture: architecture,
            Variant: FormattableString.Invariant($"n{Batch}-c{InputChannels}-k{OutputChannels}-l{Length}-kl{KernelLength}-s{Stride}-p{Padding}-fp32"),
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradOutput", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, grad, grad, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gradWeight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D, dw, dw, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(MaxRegistersPerThread: 64, MaxStaticSharedBytes: BlockThreads * sizeof(float), MaxLocalBytesPerThread: 0, MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[k,c,kl] = sum_{n,ol} input[n,c,ol*stride+kl-pad]*gradOut[n,k,ol]",
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
            throw new NotSupportedException("Only the experimental SM86 Conv1D backward-weight emitter exists.");
        string I(int v) => v.ToString(CultureInfo.InvariantCulture);
        int c = InputChannels, kl = KernelLength, l = Length, ol = OutLength, kk = OutputChannels;
        int nol = Batch * ol, cl = c * l, kol = kk * ol;
        string entry = EntryPoint;

        var s = new StringBuilder(16384);
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
        s.AppendLine("    .reg .b32 %r<28>;");
        s.AppendLine("    .reg .b64 %rd<16>;");
        s.AppendLine($"    .reg .f32 %f<{I(kl + 4)}>;");
        s.AppendLine($"    .shared .align 4 .b32 red[{I(BlockThreads)}];");
        s.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        s.AppendLine("    ld.param.u64 %rd1, [grad_ptr];");
        s.AppendLine("    ld.param.u64 %rd2, [dw_ptr];");
        s.AppendLine("    mov.u32 %r0, %tid.x;");
        s.AppendLine("    mov.u32 %r1, %ctaid.x;");               // block = k*C + c
        s.AppendLine($"    div.u32 %r2, %r1, {I(c)};");          // k
        s.AppendLine($"    rem.u32 %r3, %r1, {I(c)};");          // c
        for (int t = 0; t < kl; t++)
            s.AppendLine($"    mov.f32 %f{t}, 0f00000000;");
        s.AppendLine("    mov.u32 %r4, %r0;");                   // i over N*OL
        s.AppendLine("LOOP:");
        s.AppendLine($"    setp.ge.u32 %p0, %r4, {I(nol)};");
        s.AppendLine("    @%p0 bra REDUCE;");
        s.AppendLine($"    div.u32 %r5, %r4, {I(ol)};");         // nn
        s.AppendLine($"    rem.u32 %r6, %r4, {I(ol)};");         // ol
        // gradOut[nn][k][ol] index = nn*K*OL + k*OL + ol
        s.AppendLine($"    mad.lo.u32 %r7, %r5, {I(kol)}, %r6;");
        s.AppendLine($"    mad.lo.u32 %r7, %r2, {I(ol)}, %r7;");
        s.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
        s.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        s.AppendLine($"    ld.global.nc.f32 %f{I(kl)}, [%rd3];"); // gradOut value
        // input channel base index = nn*C*L + c*L
        s.AppendLine($"    mad.lo.u32 %r8, %r5, {I(cl)}, 0;");
        s.AppendLine($"    mad.lo.u32 %r8, %r3, {I(l)}, %r8;");
        // il0 = ol*stride - pad
        s.AppendLine($"    mul.lo.u32 %r9, %r6, {I(Stride)};");
        s.AppendLine($"    sub.s32 %r9, %r9, {I(Padding)};");
        for (int t = 0; t < kl; t++)
        {
            s.AppendLine($"    add.s32 %r10, %r9, {I(t)};");     // il = il0 + kl
            s.AppendLine("    setp.ge.s32 %p1, %r10, 0;");
            s.AppendLine($"    setp.lt.s32 %p2, %r10, {I(l)};");
            s.AppendLine("    and.pred %p1, %p1, %p2;");
            s.AppendLine("    add.u32 %r11, %r8, %r10;");
            s.AppendLine("    mul.wide.s32 %rd4, %r11, 4;");
            s.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
            s.AppendLine($"    mov.f32 %f{I(kl + 1)}, 0f00000000;");
            s.AppendLine($"    @%p1 ld.global.nc.f32 %f{I(kl + 1)}, [%rd4];");
            s.AppendLine($"    fma.rn.f32 %f{t}, %f{I(kl + 1)}, %f{I(kl)}, %f{t};");
        }
        s.AppendLine($"    add.u32 %r4, %r4, {I(BlockThreads)};");
        s.AppendLine("    bra LOOP;");
        s.AppendLine("REDUCE:");
        s.AppendLine("    mov.u64 %rd5, red;");
        s.AppendLine("    mul.wide.u32 %rd6, %r0, 4;");
        s.AppendLine("    add.u64 %rd6, %rd5, %rd6;");
        // dW base = (k*C + c)*KL
        s.AppendLine($"    mad.lo.u32 %r12, %r2, {I(c)}, %r3;");
        s.AppendLine($"    mul.lo.u32 %r12, %r12, {I(kl)};");
        s.AppendLine("    mul.wide.u32 %rd7, %r12, 4;");
        s.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        for (int t = 0; t < kl; t++)
        {
            s.AppendLine("    bar.sync 0;");
            s.AppendLine($"    st.shared.f32 [%rd6], %f{t};");
            s.AppendLine("    bar.sync 0;");
            for (int offset = BlockThreads / 2; offset > 0; offset >>= 1)
            {
                string skip = $"S_{t}_{offset}";
                s.AppendLine($"    setp.lt.u32 %p5, %r0, {I(offset)};");
                s.AppendLine($"    @!%p5 bra {skip};");
                s.AppendLine($"    ld.shared.f32 %f{I(kl + 2)}, [%rd6];");
                s.AppendLine($"    ld.shared.f32 %f{I(kl + 3)}, [%rd6+{I(offset * 4)}];");
                s.AppendLine($"    add.rn.f32 %f{I(kl + 2)}, %f{I(kl + 2)}, %f{I(kl + 3)};");
                s.AppendLine($"    st.shared.f32 [%rd6], %f{I(kl + 2)};");
                s.AppendLine($"{skip}:");
                s.AppendLine("    bar.sync 0;");
            }
            s.AppendLine("    setp.ne.u32 %p5, %r0, 0;");
            s.AppendLine($"    @%p5 bra AFTER_{t};");
            s.AppendLine($"    ld.shared.f32 %f{I(kl + 2)}, [%rd5];");
            s.AppendLine($"    st.global.f32 [%rd7+{I(t * 4)}], %f{I(kl + 2)};");
            s.AppendLine($"AFTER_{t}:");
        }
        s.AppendLine("    ret;");
        s.AppendLine("}");
        return s.ToString();
    }

    public void Dispose() => _module.Dispose();
}
