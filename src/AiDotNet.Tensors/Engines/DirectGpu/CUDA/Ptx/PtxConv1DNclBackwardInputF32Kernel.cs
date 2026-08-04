using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCL 1D convolution backward-input gradient (kernel 3, stride
/// 1, zero pad 1) — the transpose of <see cref="PtxConv1DNclForwardF32Kernel"/>. For
/// each input position <c>dIn[ci,l] = sum_co sum_k W[co,ci,k] * dOut[co, l-k+1]</c>.
/// The 1D tensors are addressed as H=1 NCHW so the shape record and layouts are the
/// forward's. One thread per input element runs an output-channel loop with the
/// three kernel taps unrolled and the length-axis halo predicated from l. Zero
/// global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConv1DNclBackwardInputF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv1d_bwd_input_n1_cin64_l256_cout64_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 64;
    internal const int Length = 256;
    internal const int OutputChannels = 64;
    internal const int KernelSize = 3;
    internal const int InputElements = Batch * InputChannels * Length;
    internal const long GradOutputBytes = (long)Batch * OutputChannels * Length * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * KernelSize * sizeof(float);
    internal const long GradInputBytes = (long)InputElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxConv1DNclForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv1DNclBackwardInputF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv1D backward-input has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture)
    {
        var gradOutput = new DirectPtxExtent(Batch, OutputChannels, 1, Length);
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, 1, KernelSize);
        var gradInput = new DirectPtxExtent(Batch, InputChannels, 1, Length);
        return new DirectPtxKernelBlueprint(
            Operation: "conv1d-bwd-input",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin64-l256-cout64-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradInput, gradInput, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // 0 = per-thread register ceiling derived from the device register
                // file at validation time; not pinned to a hardcoded literal.
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dIn[ci,l] = sum_co sum_k W[co,ci,k] * dOut[co, l-k+1]",
                ["grad_output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_input"] = "fp32",
                ["layout"] = "ncl-as-h1-nchw / oil-as-h1-oihw",
                ["padding"] = "zero-pad-1-halo-predicated",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView weights,
        DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &gradInputPointer;
        _module.Launch(
            _function, InputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    internal static string EmitPtx(int computeCapabilityMajor, int computeCapabilityMinor)
    {
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                computeCapabilityMajor, computeCapabilityMinor))
            throw new NotSupportedException(
                "Only the experimental SM86 Conv1D backward-input emitter exists.");

        const int gradOutChannelStrideBytes = Length * sizeof(float);
        const int weightOutStrideBytes = InputChannels * KernelSize * sizeof(float); // per co
        const int weightInBaseStrideBytes = KernelSize * sizeof(float);              // per ci (base)

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 gradout_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 gradin_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradin_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global input element id
        ptx.AppendLine("    shr.u32 %r3, %r2, 8;");                            // input channel ci (L = 256)
        ptx.AppendLine($"    and.b32 %r4, %r2, {Length - 1};");                 // l
        // Length-axis halo predicates (depend only on l).
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");                         // l >= 1  (k=2, dl'=-1)
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {Length - 1};");             // l <= L-2 (k=0, dl'=+1)
        // Weight base for input channel ci at output channel 0: weights_ptr + ci*(K*4).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {weightInBaseStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        // grad-output center base: gradout_ptr + l*4 (output channel 0).
        ptx.AppendLine("    mul.wide.u32 %rd4, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r5, 0;");                                  // output-channel counter
        ptx.AppendLine("CONV1D_BWD_INPUT:");
        // k=0 (dl'=+1): weight [rd3], dOut [rd4+4] guarded by p1.
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, 4;");
        ptx.AppendLine("    @%p1 ld.global.nc.f32 %f1, [%rd7];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        // k=1 (dl'=0): weight [rd3+4], dOut [rd4].
        ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{1 * sizeof(float)}];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        // k=2 (dl'=-1): weight [rd3+8], dOut [rd4-4] guarded by p0.
        ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{2 * sizeof(float)}];");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    sub.u64 %rd7, %rd4, 4;");
        ptx.AppendLine("    @%p0 ld.global.nc.f32 %f1, [%rd7];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        // Advance to next output channel.
        ptx.AppendLine($"    add.u64 %rd3, %rd3, {weightOutStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd4, %rd4, {gradOutChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r5, {OutputChannels};");
        ptx.AppendLine("    @%p2 bra CONV1D_BWD_INPUT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
