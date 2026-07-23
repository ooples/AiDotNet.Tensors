using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCL 1D convolution backward-weight gradient (kernel 3, stride
/// 1, zero pad 1). For each of the Cout*Cin*K weights the gradient is
/// <c>dW[co,ci,k] = sum_l dOut[co,l] * in[ci, l+k-1]</c>. One thread owns one weight
/// and evaluates its full length-axis dot product (halo predicated per position),
/// so there is no cross-thread reduction: the result is deterministic. The 1D
/// tensors are addressed as H=1 NCHW so the shape record and layouts match the
/// family. Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConv1DNclBackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv1d_bwd_weight_n1_cin64_l256_cout64_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 64;
    internal const int Length = 256;
    internal const int OutputChannels = 64;
    internal const int KernelSize = 3;
    internal const int WeightElements = OutputChannels * InputChannels * KernelSize;
    internal const long GradOutputBytes = (long)Batch * OutputChannels * Length * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * Length * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxConv1DNclForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv1DNclBackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv1D backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var input = new DirectPtxExtent(Batch, InputChannels, 1, Length);
        var gradWeight = new DirectPtxExtent(OutputChannels, InputChannels, 1, KernelSize);
        return new DirectPtxKernelBlueprint(
            Operation: "conv1d-bwd-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin64-l256-cout64-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_weight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    gradWeight, gradWeight, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "dW[co,ci,k] = sum_l dOut[co,l] * in[ci, l+k-1]",
                ["grad_output"] = "fp32",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_weight"] = "fp32",
                ["reduction"] = "thread-private-full-dot-deterministic",
                ["layout"] = "ncl-as-h1-nchw / oil-as-h1-oihw",
                ["padding"] = "zero-pad-1-halo-predicated",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView gradWeight)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(gradWeight, Blueprint.Tensors[2], nameof(gradWeight));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr gradWeightPointer = gradWeight.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &gradWeightPointer;
        // One thread per weight element (Cout*Cin*K).
        _module.Launch(
            _function, WeightElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 Conv1D backward-weight emitter exists.");

        const int channelStrideBytes = Length * sizeof(float);

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 gradout_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 gradweight_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradweight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // weight id = co*192 + ci*3 + k
        // Decompose: tmp = id/K (= co*Cin + ci), k = id - tmp*K, ci = tmp&63, co = tmp>>6.
        ptx.AppendLine($"    div.u32 %r3, %r2, {KernelSize};");                 // tmp
        ptx.AppendLine($"    mul.lo.u32 %r10, %r3, {KernelSize};");
        ptx.AppendLine("    sub.u32 %r4, %r2, %r10;");                          // k
        ptx.AppendLine("    and.b32 %r5, %r3, 63;");                            // ci
        ptx.AppendLine("    shr.u32 %r6, %r3, 6;");                             // co
        // dOut row base for co (advances +4 per l).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r6, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        // input row base for ci.
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r5, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        // dk = k - 1; running index il = l + dk starts at dk; in addr = in_base + il*4.
        ptx.AppendLine("    sub.s32 %r7, %r4, 1;");                             // dk
        ptx.AppendLine("    mul.wide.s32 %rd5, %r7, 4;");                       // dk*4 (signed)
        ptx.AppendLine("    add.u64 %rd6, %rd4, %rd5;");                        // in addr at l=0
        ptx.AppendLine("    mov.u32 %r8, %r7;");                                // il = dk
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r9, 0;");                                  // length counter
        ptx.AppendLine("CONV1D_BWD_WEIGHT:");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");                    // dOut[co,l]
        ptx.AppendLine("    setp.ge.s32 %p0, %r8, 0;");                         // il >= 0
        ptx.AppendLine($"    setp.lt.s32 %p1, %r8, {Length};");                 // il < L
        ptx.AppendLine("    and.pred %p2, %p0, %p1;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p2 ld.global.nc.f32 %f1, [%rd6];");               // in[ci, il]
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
        ptx.AppendLine("    add.u64 %rd3, %rd3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.s32 %r8, %r8, 1;");
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p3, %r9, {Length};");
        ptx.AppendLine("    @%p3 bra CONV1D_BWD_WEIGHT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
