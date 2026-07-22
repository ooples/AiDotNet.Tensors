using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCDHW 3x3x3 transposed 3D convolution backward-weight (stride 1,
/// zero pad 1). The weight gradient reduces the transposed sampling over all output
/// positions:
/// <c>dW[ci,co,kd,kh,kw] = sum_{od,oh,ow} dOut[co,od,oh,ow] * in[ci, od+1-kd, oh+1-kh, ow+1-kw]</c>
/// over the exact N1/Cin16/D8/H8/W8/Cout16 geometry, IODHW weights. Structurally the
/// Conv3D backward-weight (thread-per-weight, full-volume dot) but with the IODHW id
/// decompose (ci outer, co inner) and the transposed input read (dcr=1-kd). One thread
/// owns one weight and sweeps all 512 output positions with a 3D-halo-predicated input
/// load (thread-private, deterministic, no atomics). NCDHW/IODHW are rank-5, so the ABI
/// contract uses byte-exact collapsed rank-4 extents; the emitter owns the true strides.
/// </summary>
internal sealed class PtxConvTranspose3DNcdhw3x3x3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_convtranspose3d_n1_cin16_d8_h8_w8_cout16_r3_s1_p1_bwd_weight";
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
    internal const int WeightElements = InputChannels * OutputChannels * TapsPerChannel; // 6912
    internal const int OutputElements = Batch * OutputChannels * SpatialElements;
    internal const int InputElements = Batch * InputChannels * SpatialElements;
    internal const long GradOutputBytes = (long)OutputElements * sizeof(float);
    internal const long InputBytes = (long)InputElements * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConvTranspose3DNcdhw3x3x3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"ConvTranspose3D 3x3x3 backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradWeight = new DirectPtxExtent(InputChannels, OutputChannels, TapsPerChannel, 1);
        return new DirectPtxKernelBlueprint(
            Operation: "convtranspose3d-backward-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin16-d8-h8-w8-cout16-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_weight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Iohw,
                    gradWeight, gradWeight, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[ci,co,kd,kh,kw] = sum_{od,oh,ow} dOut[co,od,oh,ow] * in[ci, od+1-kd, oh+1-kh, ow+1-kw]",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad-weight"] = "fp32",
                ["reduction"] = "spatial (thread-private per weight, no atomics)",
                ["layout"] = "ncdhw/iodhw (abi extents collapsed to rank-4 byte-exact)",
                ["padding"] = "zero-pad-1 transpose read (in at od+1-kd)",
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
                "Only the experimental SM86 ConvTranspose3D 3x3x3 backward-weight emitter exists.");

        const int channelStrideBytes = SpatialElements * sizeof(float);   // 2048 per co (dOut) / ci (input)
        const int hw = Height * Width;

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 grad_weight_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<32>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_weight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // flat weight id
        // Decompose id (IODHW) = ci*Cout*27 + co*27 + tap.
        ptx.AppendLine("    div.u32 %r3, %r2, 27;");                           // rest = id / 27
        ptx.AppendLine("    mul.lo.u32 %r4, %r3, 27;");
        ptx.AppendLine("    sub.u32 %r5, %r2, %r4;");                          // tap
        ptx.AppendLine("    and.b32 %r6, %r3, 15;");                           // co = rest & 15
        ptx.AppendLine("    shr.u32 %r7, %r3, 4;");                            // ci = rest >> 4
        ptx.AppendLine("    div.u32 %r8, %r5, 9;");                            // kd
        ptx.AppendLine("    mul.lo.u32 %r9, %r8, 9;");
        ptx.AppendLine("    sub.u32 %r10, %r5, %r9;");                         // rem9
        ptx.AppendLine("    div.u32 %r11, %r10, 3;");                          // kh
        ptx.AppendLine("    mul.lo.u32 %r12, %r11, 3;");
        ptx.AppendLine("    sub.u32 %r13, %r10, %r12;");                       // kw
        ptx.AppendLine("    sub.s32 %r14, 1, %r8;");                           // dcr = 1 - kd
        ptx.AppendLine("    sub.s32 %r15, 1, %r11;");                          // dhr = 1 - kh
        ptx.AppendLine("    sub.s32 %r16, 1, %r13;");                          // dwr = 1 - kw
        // grad_output base for co (+4 per s); input base for ci (addr recomputed per s).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r6, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");                        // dout_base = grad_output + co*2048
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r7, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");                        // in_base = input + ci*2048
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r17, 0;");                                // spatial counter s
        ptx.AppendLine("CONVT3D_BWD_WEIGHT:");
        ptx.AppendLine("    shr.u32 %r18, %r17, 6;");                          // od = s / (H*W)
        ptx.AppendLine($"    and.b32 %r19, %r17, {hw - 1};");                  // rem = s % (H*W)
        ptx.AppendLine("    shr.u32 %r20, %r19, 3;");                          // oh
        ptx.AppendLine($"    and.b32 %r21, %r19, {Width - 1};");               // ow
        ptx.AppendLine("    add.s32 %r22, %r18, %r14;");                        // id_d = od + dcr
        ptx.AppendLine("    add.s32 %r23, %r20, %r15;");                        // id_h = oh + dhr
        ptx.AppendLine("    add.s32 %r24, %r21, %r16;");                        // id_w = ow + dwr
        ptx.AppendLine("    setp.ge.s32 %p0, %r22, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r22, {Depth};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r23, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r23, {Height};");
        ptx.AppendLine("    setp.ge.s32 %p4, %r24, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p5, %r24, {Width};");
        ptx.AppendLine("    and.pred %p6, %p0, %p1;");
        ptx.AppendLine("    and.pred %p6, %p6, %p2;");
        ptx.AppendLine("    and.pred %p6, %p6, %p3;");
        ptx.AppendLine("    and.pred %p6, %p6, %p4;");
        ptx.AppendLine("    and.pred %p6, %p6, %p5;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");                   // dOut[co,s]
        // input index = id_d*(H*W) + id_h*W + id_w.
        ptx.AppendLine($"    mad.lo.s32 %r25, %r23, {Width}, %r24;");
        ptx.AppendLine($"    mad.lo.s32 %r25, %r22, {hw}, %r25;");
        ptx.AppendLine("    mul.wide.s32 %rd5, %r25, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd4, %rd5;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p6 ld.global.nc.f32 %f1, [%rd5];");             // in[ci,id_d,id_h,id_w] or 0
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
        ptx.AppendLine("    add.u64 %rd3, %rd3, 4;");                          // next dOut spatial
        ptx.AppendLine("    add.u32 %r17, %r17, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p7, %r17, {SpatialElements};");
        ptx.AppendLine("    @%p7 bra CONVT3D_BWD_WEIGHT;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
