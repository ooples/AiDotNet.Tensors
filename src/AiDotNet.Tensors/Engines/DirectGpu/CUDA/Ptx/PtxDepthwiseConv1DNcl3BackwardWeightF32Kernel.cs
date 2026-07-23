using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCL depthwise 1D convolution backward-weight gradient (kernel
/// 3, stride 1, zero pad 1). Per channel: <c>dW[c,k] = sum_l dOut[c,l] * in[c, l+k-1]</c>.
/// One thread owns one weight (Channels*3 threads) and evaluates its full length dot
/// with the k-offset halo predicated per position, so there is no cross-thread
/// reduction: the result is deterministic. Zero global intermediates, zero local
/// bytes.
/// </summary>
internal sealed class PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_depthwise_conv1d_bwd_weight_n1_c64_l256_r3_s1_p1";
    internal const int BlockThreads = 64;   // 192 weights / 64 = 3 blocks exactly.
    internal const int Batch = 1;
    internal const int Channels = 64;
    internal const int Length = 256;
    internal const int KernelSize = 3;
    internal const int WeightElements = Channels * KernelSize; // 192
    internal const long GradOutputBytes = (long)Batch * Channels * Length * sizeof(float);
    internal const long InputBytes = (long)Batch * Channels * Length * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxDepthwiseConv1DNcl3ForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDepthwiseConv1DNcl3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"DepthwiseConv1D backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var input = new DirectPtxExtent(Batch, Channels, 1, Length);
        var gradWeight = new DirectPtxExtent(Channels, 1, 1, KernelSize);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv1d-bwd-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-c64-l256-r3-s1-p1-fp32",
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
                ["equation"] = "dW[c,k] = sum_l dOut[c,l] * in[c, l+k-1], per-channel",
                ["grad_output"] = "fp32",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_weight"] = "fp32",
                ["reduction"] = "thread-private-full-dot-deterministic",
                ["layout"] = "ncl-as-h1-nchw / per-channel oil-as-h1-oihw",
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
                "Only the experimental SM86 DepthwiseConv1D backward-weight emitter exists.");

        const int channelStrideBytes = Length * sizeof(float);

        var ptx = new StringBuilder(4096);
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
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradweight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // weight id = c*3 + k
        ptx.AppendLine($"    div.u32 %r3, %r2, {KernelSize};");                 // c
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {KernelSize};");
        ptx.AppendLine("    sub.u32 %r4, %r2, %r4;");                           // k
        ptx.AppendLine("    sub.s32 %r5, %r4, 1;");                             // dk = k-1
        // dOut row base for channel c (advances +4 per position).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        // input base + dk offset: in_ptr + c*channelStride + dk*4 (advances +4).
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r3, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        ptx.AppendLine("    mul.lo.s32 %r6, %r5, 4;");                          // dk*4
        ptx.AppendLine("    cvt.s64.s32 %rd5, %r6;");
        ptx.AppendLine("    add.u64 %rd6, %rd4, %rd5;");                        // in addr at l=0
        ptx.AppendLine("    mov.u32 %r7, %r5;");                                // il = dk
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");                                  // length counter
        ptx.AppendLine("DWC1D_BWD_WEIGHT:");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");                    // dOut[c,l]
        ptx.AppendLine("    setp.ge.s32 %p0, %r7, 0;");                         // il >= 0
        ptx.AppendLine($"    setp.lt.s32 %p1, %r7, {Length};");                 // il < L
        ptx.AppendLine("    and.pred %p2, %p0, %p1;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p2 ld.global.nc.f32 %f1, [%rd6];");               // in[c, il]
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
        ptx.AppendLine("    add.u64 %rd3, %rd3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.s32 %r7, %r7, 1;");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p3, %r8, {Length};");
        ptx.AppendLine("    @%p3 bra DWC1D_BWD_WEIGHT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
