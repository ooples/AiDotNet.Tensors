using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW grouped deformable 3x3 convolution forward (DCNv2 with G
/// deformable groups; stride 1, pad 1, dilation 1). Each input channel belongs to a
/// deformable group g = ci / (Cin/G) with its own learned offset and modulation-mask
/// field:
/// <c>Y[co,oy,ox] = sum_ci sum_ky sum_kx W[co,ci,ky,kx] * mask[g*kHkW+t, oy,ox] * bilinear(X[ci], oy+ky-1+offY, ox+kx-1+offX)</c>
/// with <c>offY = offsets[g*2*kHkW + 2*t, oy, ox]</c>, <c>offX = offsets[g*2*kHkW + 2*t+1, oy, ox]</c>,
/// t = ky*3+kx, over the exact N1/Cin4/H8/W8/Cout4 G2 geometry. One thread owns one output
/// element and runs an input-channel loop; because the sampling field is per-group, the
/// four bilinear corners are recomputed for every (input channel, tap). Thread-private,
/// deterministic, no atomics. Offsets are [G*2*kHkW, outH, outW] and mask is
/// [G*kHkW, outH, outW]; all tensors map to byte-exact extents.
/// </summary>
internal sealed class PtxDeformableConv2DGroupedNchw3x3F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_deformconv2d_grouped_n1_cin4_h8_w8_cout4_r3_s1_p1_g2_v2";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 4;
    internal const int Height = 8;
    internal const int Width = 8;
    internal const int OutputChannels = 4;
    internal const int OutHeight = Height;
    internal const int OutWidth = Width;
    internal const int KernelSize = 3;
    internal const int DeformableGroups = 2;
    internal const int ChannelsPerGroup = InputChannels / DeformableGroups; // 2
    internal const int TapsPerChannel = KernelSize * KernelSize;          // 9
    internal const int SpatialElements = OutHeight * OutWidth;            // 64
    internal const int OutputElements = Batch * OutputChannels * SpatialElements; // 256
    internal const int OffsetChannels = DeformableGroups * 2 * TapsPerChannel; // 36
    internal const int MaskChannels = DeformableGroups * TapsPerChannel;       // 18
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long OffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);
    internal const long MaskBytes = (long)MaskChannels * SpatialElements * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDeformableConv2DGroupedNchw3x3F32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Grouped deformable Conv2D 3x3 has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, KernelSize, KernelSize);
        var offsets = new DirectPtxExtent(OffsetChannels, OutHeight, OutWidth);
        var mask = new DirectPtxExtent(MaskChannels, OutHeight, OutWidth);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin4-h8-w8-cout4-r3-s1-p1-g2-v2-fp32",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offsets", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    offsets, offsets, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "Y[co,oy,ox] = sum_ci sum_ky sum_kx W[co,ci,ky,kx] * mask[g*9+t,oy,ox] * bilinear(X[ci], oy+ky-1+offY, ox+kx-1+offX), g=ci/(Cin/G)",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["sampling"] = "bilinear (floor + 4 corner, zero-pad) per (channel, tap)",
                ["modulation"] = "v2 mask multiply",
                ["deformable-groups"] = "2",
                ["layout"] = "nchw input/output; oihw weights; [G*2*kHkW,outH,outW] offsets; [G*kHkW,outH,outW] mask",
                ["padding"] = "zero-pad-1 base + learned per-group offset bilinear",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView offsets,
        DirectPtxTensorView mask,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(offsets, Blueprint.Tensors[2], nameof(offsets));
        Require(mask, Blueprint.Tensors[3], nameof(mask));
        Require(output, Blueprint.Tensors[4], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr offsetPointer = offsets.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &offsetPointer;
        arguments[3] = &maskPointer;
        arguments[4] = &outputPointer;
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
                "Only the experimental SM86 grouped deformable Conv2D 3x3 emitter exists.");

        const int inChannelStrideBytes = Height * Width * sizeof(float);              // 256 per ci
        const int weightOutStrideBytes = InputChannels * TapsPerChannel * sizeof(float); // 144 per co
        const int weightChannelStrideBytes = TapsPerChannel * sizeof(float);          // 36 per ci
        const int planeStrideBytes = OutHeight * OutWidth * sizeof(float);            // 256 per offset/mask channel
        const int offsetGroupStrideBytes = 2 * TapsPerChannel * planeStrideBytes;     // 4608 per group
        const int maskGroupStrideBytes = TapsPerChannel * planeStrideBytes;           // 2304 per group

        var ptx = new StringBuilder(49152);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 offsets_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<16>;");
        ptx.AppendLine("    .reg .b32 %r<48>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [offsets_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // output id = co*HW + spatial
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial s = oy*W + ox
        ptx.AppendLine("    shr.u32 %r4, %r3, 3;");                            // oy
        ptx.AppendLine($"    and.b32 %r5, %r3, {Width - 1};");                 // ox
        ptx.AppendLine("    shr.u32 %r6, %r2, 6;");                            // co (64 = outH*outW)
        // Weight base for co: weights + co*144 (wci starts here, +36 per ci).
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r6, {weightOutStrideBytes};");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd5;");                        // wci
        ptx.AppendLine("    mov.u64 %rd9, %rd0;");                             // xci = input (+256 per ci)
        ptx.AppendLine("    mul.wide.u32 %rd20, %r3, 4;");                      // spatial byte offset s*4
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accumulator
        ptx.AppendLine("    mov.f32 %f30, 0f3F800000;");                       // 1.0
        ptx.AppendLine("    mov.u32 %r7, 0;");                                 // ci counter
        ptx.AppendLine("DEFORM_GROUPED_CI:");
        // Group g = ci / ChannelsPerGroup; group offset/mask bases at this spatial position.
        ptx.AppendLine($"    div.u32 %r8, %r7, {ChannelsPerGroup};");          // g
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r8, {offsetGroupStrideBytes};");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd20;");                       // offgrp_s = offsets + g*4608 + s*4
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r8, {maskGroupStrideBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd3, %rd7;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd20;");                       // maskgrp_s = mask + g*2304 + s*4
        for (int t = 0; t < TapsPerChannel; t++)
        {
            int ky = t / KernelSize;
            int kx = t % KernelSize;
            int dy = ky - 1;
            int dx = kx - 1;
            int offYByte = 2 * t * planeStrideBytes;       // relative to offgrp_s
            int offXByte = (2 * t + 1) * planeStrideBytes;
            int maskByte = t * planeStrideBytes;           // relative to maskgrp_s
            int weightByte = t * (int)sizeof(float);       // relative to wci
            ptx.AppendLine(offYByte == 0
                ? "    ld.global.nc.f32 %f1, [%rd6];"
                : $"    ld.global.nc.f32 %f1, [%rd6+{offYByte}];");
            ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd6+{offXByte}];");
            ptx.AppendLine(maskByte == 0
                ? "    ld.global.nc.f32 %f3, [%rd7];"
                : $"    ld.global.nc.f32 %f3, [%rd7+{maskByte}];");
            // py = (oy+dy) + offY, px = (ox+dx) + offX.
            if (dy == 0) ptx.AppendLine("    mov.s32 %r9, %r4;");
            else ptx.AppendLine($"    add.s32 %r9, %r4, {dy};");
            if (dx == 0) ptx.AppendLine("    mov.s32 %r10, %r5;");
            else ptx.AppendLine($"    add.s32 %r10, %r5, {dx};");
            ptx.AppendLine("    cvt.rn.f32.s32 %f4, %r9;");
            ptx.AppendLine("    add.rn.f32 %f4, %f4, %f1;");                    // py
            ptx.AppendLine("    cvt.rn.f32.s32 %f5, %r10;");
            ptx.AppendLine("    add.rn.f32 %f5, %f5, %f2;");                    // px
            ptx.AppendLine("    cvt.rmi.s32.f32 %r11, %f4;");                   // y0
            ptx.AppendLine("    cvt.rmi.s32.f32 %r12, %f5;");                   // x0
            ptx.AppendLine("    add.s32 %r13, %r11, 1;");                       // y1
            ptx.AppendLine("    add.s32 %r14, %r12, 1;");                       // x1
            ptx.AppendLine("    cvt.rn.f32.s32 %f6, %r11;");
            ptx.AppendLine("    sub.rn.f32 %f7, %f4, %f6;");                    // wy1
            ptx.AppendLine("    sub.rn.f32 %f8, %f30, %f7;");                   // wy0
            ptx.AppendLine("    cvt.rn.f32.s32 %f9, %r12;");
            ptx.AppendLine("    sub.rn.f32 %f10, %f5, %f9;");                   // wx1
            ptx.AppendLine("    sub.rn.f32 %f11, %f30, %f10;");                 // wx0
            ptx.AppendLine("    mul.rn.f32 %f12, %f8, %f11;");                  // c00
            ptx.AppendLine("    mul.rn.f32 %f13, %f8, %f10;");                  // c01
            ptx.AppendLine("    mul.rn.f32 %f14, %f7, %f11;");                  // c10
            ptx.AppendLine("    mul.rn.f32 %f15, %f7, %f10;");                  // c11
            ptx.AppendLine("    setp.ge.s32 %p0, %r11, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p1, %r11, {Height};");
            ptx.AppendLine("    and.pred %p0, %p0, %p1;");
            ptx.AppendLine("    setp.ge.s32 %p2, %r13, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p3, %r13, {Height};");
            ptx.AppendLine("    and.pred %p2, %p2, %p3;");
            ptx.AppendLine("    setp.ge.s32 %p4, %r12, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p5, %r12, {Width};");
            ptx.AppendLine("    and.pred %p4, %p4, %p5;");
            ptx.AppendLine("    setp.ge.s32 %p6, %r14, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p7, %r14, {Width};");
            ptx.AppendLine("    and.pred %p6, %p6, %p7;");
            ptx.AppendLine("    and.pred %p8, %p0, %p4;");
            ptx.AppendLine("    and.pred %p9, %p0, %p6;");
            ptx.AppendLine("    and.pred %p10, %p2, %p4;");
            ptx.AppendLine("    and.pred %p11, %p2, %p6;");
            ptx.AppendLine($"    mul.lo.s32 %r15, %r11, {Width};");
            ptx.AppendLine($"    mul.lo.s32 %r16, %r13, {Width};");
            ptx.AppendLine("    add.s32 %r17, %r15, %r12;");
            ptx.AppendLine("    add.s32 %r18, %r15, %r14;");
            ptx.AppendLine("    add.s32 %r19, %r16, %r12;");
            ptx.AppendLine("    add.s32 %r20, %r16, %r14;");
            ptx.AppendLine("    shl.b32 %r17, %r17, 2;");
            ptx.AppendLine("    shl.b32 %r18, %r18, 2;");
            ptx.AppendLine("    shl.b32 %r19, %r19, 2;");
            ptx.AppendLine("    shl.b32 %r20, %r20, 2;");
            ptx.AppendLine("    cvt.s64.s32 %rd10, %r17;");
            ptx.AppendLine("    add.s64 %rd10, %rd9, %rd10;");
            ptx.AppendLine("    mov.f32 %f16, 0f00000000;");
            ptx.AppendLine("    @%p8 ld.global.nc.f32 %f16, [%rd10];");
            ptx.AppendLine("    cvt.s64.s32 %rd11, %r18;");
            ptx.AppendLine("    add.s64 %rd11, %rd9, %rd11;");
            ptx.AppendLine("    mov.f32 %f17, 0f00000000;");
            ptx.AppendLine("    @%p9 ld.global.nc.f32 %f17, [%rd11];");
            ptx.AppendLine("    cvt.s64.s32 %rd12, %r19;");
            ptx.AppendLine("    add.s64 %rd12, %rd9, %rd12;");
            ptx.AppendLine("    mov.f32 %f18, 0f00000000;");
            ptx.AppendLine("    @%p10 ld.global.nc.f32 %f18, [%rd12];");
            ptx.AppendLine("    cvt.s64.s32 %rd13, %r20;");
            ptx.AppendLine("    add.s64 %rd13, %rd9, %rd13;");
            ptx.AppendLine("    mov.f32 %f19, 0f00000000;");
            ptx.AppendLine("    @%p11 ld.global.nc.f32 %f19, [%rd13];");
            ptx.AppendLine("    mul.rn.f32 %f20, %f12, %f16;");
            ptx.AppendLine("    fma.rn.f32 %f20, %f13, %f17, %f20;");
            ptx.AppendLine("    fma.rn.f32 %f20, %f14, %f18, %f20;");
            ptx.AppendLine("    fma.rn.f32 %f20, %f15, %f19, %f20;");           // sample
            ptx.AppendLine(weightByte == 0
                ? "    ld.global.nc.f32 %f21, [%rd8];"
                : $"    ld.global.nc.f32 %f21, [%rd8+{weightByte}];");
            ptx.AppendLine("    mul.rn.f32 %f22, %f3, %f20;");                  // mask * sample
            ptx.AppendLine("    fma.rn.f32 %f0, %f21, %f22, %f0;");            // acc += W * (mask*sample)
        }
        ptx.AppendLine($"    add.u64 %rd8, %rd8, {weightChannelStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd9, %rd9, {inChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p12, %r7, {InputChannels};");
        ptx.AppendLine("    @%p12 bra DEFORM_GROUPED_CI;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd4, %rd14;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
