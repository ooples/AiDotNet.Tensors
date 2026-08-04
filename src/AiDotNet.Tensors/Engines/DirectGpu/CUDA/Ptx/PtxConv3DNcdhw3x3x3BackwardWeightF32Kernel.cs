using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCDHW 3x3x3 3D convolution backward-weight gradient (stride 1,
/// zero pad 1), OIDHW weights:
/// <c>dW[co,ci,kd,kh,kw] = sum_{d,h,w} dOut[co,d,h,w] * in[ci, d+kd-1, h+kh-1, w+kw-1]</c>
/// over the exact N1/Cin16/D8/H8/W8/Cout16 geometry. One thread owns one weight and
/// evaluates its full D*H*W dot with the 3D halo predicated per position, so there
/// is no cross-thread reduction: the result is deterministic. Byte-exact collapsed
/// rank-4 ABI extents. Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv3d_bwd_weight_n1_cin16_d8_h8_w8_cout16_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 16;
    internal const int Depth = 8;
    internal const int Height = 8;
    internal const int Width = 8;
    internal const int OutputChannels = 16;
    internal const int KernelSize = 3;
    internal const int TapsPerChannel = KernelSize * KernelSize * KernelSize; // 27
    internal const int SpatialElements = Depth * Height * Width;              // 512
    internal const int WeightElements = OutputChannels * InputChannels * TapsPerChannel; // 6912
    internal const long GradOutputBytes = (long)Batch * OutputChannels * SpatialElements * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * SpatialElements * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv3DNcdhw3x3x3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv3D backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, OutputChannels, Depth, Height * Width);
        var input = new DirectPtxExtent(Batch, InputChannels, Depth, Height * Width);
        var gradWeight = new DirectPtxExtent(OutputChannels, InputChannels, TapsPerChannel, 1);
        return new DirectPtxKernelBlueprint(
            Operation: "conv3d-bwd-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin16-d8-h8-w8-cout16-r3-s1-p1-fp32",
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
                ["equation"] = "dW[co,ci,kd,kh,kw] = sum_{d,h,w} dOut[co,d,h,w] * in[ci, d+kd-1, h+kh-1, w+kw-1]",
                ["grad_output"] = "fp32",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_weight"] = "fp32",
                ["reduction"] = "thread-private-full-volume-dot-deterministic",
                ["layout"] = "ncdhw/oidhw (abi extents collapsed to rank-4 byte-exact)",
                ["padding"] = "zero-pad-1 halo-predicated on d/h/w",
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
                "Only the experimental SM86 Conv3D backward-weight emitter exists.");

        const int channelStrideBytes = SpatialElements * sizeof(float);
        const int hw = Height * Width;

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
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradweight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // weight id (OIDHW)
        // Decompose: tmp = id/27 (= co*Cin + ci), m = id - tmp*27, ci = tmp&15, co = tmp>>4.
        ptx.AppendLine($"    div.u32 %r3, %r2, {TapsPerChannel};");            // tmp
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {TapsPerChannel};");
        ptx.AppendLine("    sub.u32 %r4, %r2, %r4;");                          // m
        ptx.AppendLine("    and.b32 %r5, %r3, 15;");                           // ci
        ptx.AppendLine("    shr.u32 %r6, %r3, 4;");                            // co
        // m -> kd, kh, kw.
        ptx.AppendLine($"    div.u32 %r7, %r4, {KernelSize * KernelSize};");   // kd
        ptx.AppendLine($"    mul.lo.u32 %r8, %r7, {KernelSize * KernelSize};");
        ptx.AppendLine("    sub.u32 %r8, %r4, %r8;");                          // rem9
        ptx.AppendLine($"    div.u32 %r9, %r8, {KernelSize};");               // kh
        ptx.AppendLine($"    mul.lo.u32 %r10, %r9, {KernelSize};");
        ptx.AppendLine("    sub.u32 %r10, %r8, %r10;");                        // kw
        ptx.AppendLine("    sub.s32 %r11, %r7, 1;");                           // dd = kd-1
        ptx.AppendLine("    sub.s32 %r12, %r9, 1;");                           // dh = kh-1
        ptx.AppendLine("    sub.s32 %r13, %r10, 1;");                          // dw = kw-1
        // dOut row base for output channel co (advances +4 per position).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r6, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        // input base + tap offset: in_ptr + ci*stride + (dd*H*W + dh*W + dw)*4.
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r5, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        ptx.AppendLine($"    mul.lo.s32 %r14, %r11, {hw};");                   // dd*H*W
        ptx.AppendLine($"    mad.lo.s32 %r14, %r12, {Width}, %r14;");          // + dh*W
        ptx.AppendLine("    add.s32 %r14, %r14, %r13;");                       // + dw
        ptx.AppendLine("    mul.lo.s32 %r14, %r14, 4;");
        ptx.AppendLine("    cvt.s64.s32 %rd6, %r14;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, %rd6;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r15, 0;");                                // spatial counter
        ptx.AppendLine("CONV3D_BWD_WEIGHT:");
        ptx.AppendLine("    shr.u32 %r16, %r15, 6;");                          // d
        ptx.AppendLine($"    and.b32 %r17, %r15, {hw - 1};");                  // rem
        ptx.AppendLine("    shr.u32 %r18, %r17, 3;");                          // h
        ptx.AppendLine($"    and.b32 %r19, %r17, {Width - 1};");               // w
        ptx.AppendLine("    add.s32 %r20, %r16, %r11;");                       // id_ = d + dd
        ptx.AppendLine("    add.s32 %r21, %r18, %r12;");                       // ih = h + dh
        ptx.AppendLine("    add.s32 %r22, %r19, %r13;");                       // iw = w + dw
        ptx.AppendLine("    setp.ge.s32 %p0, %r20, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r20, {Depth};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r21, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r21, {Height};");
        ptx.AppendLine("    setp.ge.s32 %p4, %r22, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p5, %r22, {Width};");
        ptx.AppendLine("    and.pred %p6, %p0, %p1;");
        ptx.AppendLine("    and.pred %p6, %p6, %p2;");
        ptx.AppendLine("    and.pred %p6, %p6, %p3;");
        ptx.AppendLine("    and.pred %p6, %p6, %p4;");
        ptx.AppendLine("    and.pred %p6, %p6, %p5;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");                   // dOut[co, s]
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p6 ld.global.nc.f32 %f1, [%rd4];");              // in[ci, id_, ih, iw]
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
        ptx.AppendLine("    add.u64 %rd3, %rd3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, 4;");
        ptx.AppendLine("    add.u32 %r15, %r15, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p7, %r15, {SpatialElements};");
        ptx.AppendLine("    @%p7 bra CONV3D_BWD_WEIGHT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
