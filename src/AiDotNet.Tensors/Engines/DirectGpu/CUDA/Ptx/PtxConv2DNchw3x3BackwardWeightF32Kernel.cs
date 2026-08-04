using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW general multi-channel 3x3 convolution backward-weight
/// gradient (stride 1, zero pad 1). For each of the Cout*Cin*9 weights the gradient
/// is <c>dW[co,ci,ky,kx] = sum_{y,x} dOut[co,y,x] * in[ci, y+ky-1, x+kx-1]</c>. One
/// thread owns one weight and evaluates its full H*W dot (halo predicated per
/// position), so there is no cross-thread reduction: the result is deterministic.
/// Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConv2DNchw3x3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv2d_bwd_weight_n1_cin64_h16_w16_cout64_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int OutputChannels = 64;
    internal const int KernelSize = 3;
    internal const int SpatialElements = Height * Width;
    internal const int WeightElements = OutputChannels * InputChannels * KernelSize * KernelSize;
    internal const long GradOutputBytes = (long)Batch * OutputChannels * SpatialElements * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * SpatialElements * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxConv2DNchw3x3ForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv2DNchw3x3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv2D 3x3 backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, OutputChannels, Height, Width);
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var gradWeight = new DirectPtxExtent(OutputChannels, InputChannels, KernelSize, KernelSize);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-bwd-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin64-h16-w16-cout64-r3-s1-p1-fp32",
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
                ["equation"] = "dW[co,ci,ky,kx] = sum_{y,x} dOut[co,y,x] * in[ci, y+ky-1, x+kx-1]",
                ["grad_output"] = "fp32",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_weight"] = "fp32",
                ["reduction"] = "thread-private-full-dot-deterministic",
                ["layout"] = "nchw/oihw",
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
        // One thread per weight element (Cout*Cin*9).
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
                "Only the experimental SM86 Conv2D 3x3 backward-weight emitter exists.");

        const int channelStrideBytes = SpatialElements * sizeof(float);
        const int weightsPerOutBlock = KernelSize * KernelSize; // per (co,ci): 9

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
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradweight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // weight id
        // Decompose: tmp = id/9 (= co*Cin + ci), m = id - tmp*9, ci = tmp&63, co = tmp>>6.
        ptx.AppendLine($"    div.u32 %r3, %r2, {weightsPerOutBlock};");         // tmp
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {weightsPerOutBlock};");
        ptx.AppendLine("    sub.u32 %r4, %r2, %r4;");                           // m = ky*3 + kx
        ptx.AppendLine("    and.b32 %r5, %r3, 63;");                            // ci
        ptx.AppendLine("    shr.u32 %r6, %r3, 6;");                             // co
        ptx.AppendLine($"    div.u32 %r7, %r4, {KernelSize};");                 // ky
        ptx.AppendLine($"    mul.lo.u32 %r8, %r7, {KernelSize};");
        ptx.AppendLine("    sub.u32 %r8, %r4, %r8;");                           // kx
        ptx.AppendLine("    sub.s32 %r9, %r7, 1;");                             // dy = ky-1
        ptx.AppendLine("    sub.s32 %r10, %r8, 1;");                            // dx = kx-1
        // dOut row base for co (advances +4 per position).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r6, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        // input base + tapOffset: in_ptr + ci*channelStride + (dy*W+dx)*4 (advances +4).
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r5, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        ptx.AppendLine($"    mul.lo.s32 %r11, %r9, {Width};");                  // dy*W
        ptx.AppendLine("    add.s32 %r11, %r11, %r10;");                        // dy*W + dx
        ptx.AppendLine("    mul.lo.s32 %r11, %r11, 4;");                        // *4 (signed byte offset)
        ptx.AppendLine("    cvt.s64.s32 %rd6, %r11;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, %rd6;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r11, 0;");                                 // spatial counter
        ptx.AppendLine("CONV2D_BWD_WEIGHT:");
        ptx.AppendLine("    shr.u32 %r12, %r11, 4;");                           // y = s / W
        ptx.AppendLine("    and.b32 %r13, %r11, 15;");                          // x = s % W
        ptx.AppendLine("    add.s32 %r14, %r12, %r9;");                         // iy = y + dy
        ptx.AppendLine("    add.s32 %r15, %r13, %r10;");                        // ix = x + dx
        ptx.AppendLine("    setp.ge.s32 %p0, %r14, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r14, {Height};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r15, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r15, {Width};");
        ptx.AppendLine("    and.pred %p4, %p0, %p1;");
        ptx.AppendLine("    and.pred %p4, %p4, %p2;");
        ptx.AppendLine("    and.pred %p4, %p4, %p3;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");                    // dOut[co,s]
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p4 ld.global.nc.f32 %f1, [%rd4];");               // in[ci, iy, ix]
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
        ptx.AppendLine("    add.u64 %rd3, %rd3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, 4;");
        ptx.AppendLine("    add.u32 %r11, %r11, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p5, %r11, {SpatialElements};");
        ptx.AppendLine("    @%p5 bra CONV2D_BWD_WEIGHT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
