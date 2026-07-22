using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCL depthwise 1D convolution backward-input gradient (kernel 3,
/// stride 1, zero pad 1) — the transpose of
/// <see cref="PtxDepthwiseConv1DNcl3ForwardF32Kernel"/>. Per channel:
/// <c>dIn[c,l] = sum_k W[c,k] * dOut[c, l+1-k]</c>. One thread owns one input element
/// and evaluates the three taps directly (no channel loop) at the negated read
/// offset, with the length-axis halo predicated from l. Zero global intermediates,
/// zero local bytes, branchless.
/// </summary>
internal sealed class PtxDepthwiseConv1DNcl3BackwardInputF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_depthwise_conv1d_bwd_input_n1_c64_l256_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int Channels = 64;
    internal const int Length = 256;
    internal const int KernelSize = 3;
    internal const int OutputElements = Batch * Channels * Length;
    internal const long GradOutputBytes = (long)Batch * Channels * Length * sizeof(float);
    internal const long WeightBytes = (long)Channels * KernelSize * sizeof(float);
    internal const long GradInputBytes = (long)OutputElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxDepthwiseConv1DNcl3ForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDepthwiseConv1DNcl3BackwardInputF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"DepthwiseConv1D backward-input has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, Channels, 1, Length);
        var weights = new DirectPtxExtent(Channels, 1, 1, KernelSize);
        var gradInput = new DirectPtxExtent(Batch, Channels, 1, Length);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv1d-bwd-input",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-c64-l256-r3-s1-p1-fp32",
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
                ["equation"] = "dIn[c,l] = sum_k W[c,k] * dOut[c, l+1-k], per-channel",
                ["grad_output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_input"] = "fp32",
                ["layout"] = "ncl-as-h1-nchw / per-channel oil-as-h1-oihw",
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
            _function, OutputElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 DepthwiseConv1D backward-input emitter exists.");

        const int channelStrideBytes = Length * sizeof(float);
        const int weightChannelStrideBytes = KernelSize * sizeof(float);

        var ptx = new StringBuilder(4096);
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
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<6>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradin_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global input id
        ptx.AppendLine("    shr.u32 %r3, %r2, 8;");                            // channel c (L = 256)
        ptx.AppendLine($"    and.b32 %r4, %r2, {Length - 1};");                 // l
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");                         // l >= 1   (k=2, dl'=-1)
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {Length - 1};");            // l <= L-2 (k=0, dl'=+1)
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {weightChannelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");                        // weight base for c
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r3, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, %rd5;");                        // dOut center = doutbase + l*4
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        // k=0 (dl'=+1): weight [rd3], dOut [rd4+4] guarded by p1.
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    add.u64 %rd6, %rd4, 4;");
        ptx.AppendLine("    @%p1 ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        // k=1 (dl'=0): weight [rd3+4], dOut [rd4].
        ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{1 * sizeof(float)}];");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        // k=2 (dl'=-1): weight [rd3+8], dOut [rd4-4] guarded by p0.
        ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{2 * sizeof(float)}];");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    sub.u64 %rd6, %rd4, 4;");
        ptx.AppendLine("    @%p0 ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
