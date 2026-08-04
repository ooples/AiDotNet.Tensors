using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW grouped deformable 3x3 convolution backward-mask (DCNv2 with G
/// deformable groups; stride 1, pad 1, dilation 1). The modulation-mask gradient for mask
/// channel gm = g*kHkW+t at output position (oy,ox). Only the input channels that belong
/// to deformable group g contribute (they are the ones that read this group's mask):
/// <c>dMask[g*9+t, oy,ox] = sum_co dOut[co,oy,ox] * sum_{ci in group g} W[co,ci,t] * bilinear(X[ci], py, px)</c>
/// with py=oy+ky-1+offY[g*18+2t,oy,ox], px=ox+kx-1+offX[g*18+2t+1,oy,ox], t=ky*3+kx, over
/// the exact N1/Cin4/H8/W8/Cout4 G2 geometry. One thread owns one mask element; the
/// bilinear geometry is computed once, then a ChannelsPerGroup loop samples each in-group
/// X[ci] and an inner output-channel loop reduces dOut*W. Thread-private, deterministic, no
/// atomics. Second of the grouped-backward family.
/// </summary>
internal sealed class PtxDeformableConv2DGroupedNchw3x3BackwardMaskF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_deformconv2d_grouped_n1_cin4_h8_w8_cout4_r3_s1_p1_g2_v2_bwd_mask";
    internal const int BlockThreads = 128;   // MaskChannels*SpatialElements = 1152 = 128*9
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
    internal const int GradMaskElements = MaskChannels * SpatialElements;      // 1152
    internal const long GradOutputBytes = (long)OutputElements * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long OffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);
    internal const long GradMaskBytes = (long)GradMaskElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDeformableConv2DGroupedNchw3x3BackwardMaskF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Grouped deformable Conv2D 3x3 backward-mask has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, KernelSize, KernelSize);
        var offsets = new DirectPtxExtent(OffsetChannels, OutHeight, OutWidth);
        var gradMask = new DirectPtxExtent(MaskChannels, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped-backward-mask",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin4-h8-w8-cout4-r3-s1-p1-g2-v2-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offsets", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    offsets, offsets, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradMask, gradMask, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dMask[g*9+t,oy,ox] = sum_co dOut[co] * sum_{ci in group g} W[co,ci,t] * bilinear(X[ci], oy+ky-1+offY, ox+kx-1+offX)",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad-mask"] = "fp32",
                ["sampling"] = "bilinear (floor + 4 corner, zero-pad) computed once per thread",
                ["reduction"] = "over in-group ci and co (thread-private, no atomics)",
                ["deformable-groups"] = "2",
                ["layout"] = "nchw grad/input; oihw weights; [G*2*kHkW,outH,outW] offsets; [G*kHkW,outH,outW] grad-mask",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView offsets,
        DirectPtxTensorView gradMask)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(weights, Blueprint.Tensors[2], nameof(weights));
        Require(offsets, Blueprint.Tensors[3], nameof(offsets));
        Require(gradMask, Blueprint.Tensors[4], nameof(gradMask));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr offsetPointer = offsets.Pointer;
        IntPtr gradMaskPointer = gradMask.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &weightPointer;
        arguments[3] = &offsetPointer;
        arguments[4] = &gradMaskPointer;
        _module.Launch(
            _function, GradMaskElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 grouped deformable Conv2D 3x3 backward-mask emitter exists.");

        const int inChannelStrideBytes = Height * Width * sizeof(float);          // 256 per ci
        const int weightChannelStrideBytes = TapsPerChannel * sizeof(float);      // 36 per ci
        const int weightOutStrideBytes = InputChannels * TapsPerChannel * sizeof(float); // 144 per co
        const int gradOutChannelStrideBytes = SpatialElements * sizeof(float);     // 256 per co
        const int planeStrideBytes = SpatialElements * sizeof(float);              // 256 (offY->offX)
        const int inGroupBaseStrideBytes = ChannelsPerGroup * inChannelStrideBytes; // 512 (input + g*512)
        const int weightGroupBaseStrideBytes = ChannelsPerGroup * weightChannelStrideBytes; // 72 (weights + t*4 + g*72)
        const int maskChannelSpanElems = 2 * TapsPerChannel * SpatialElements;     // g index scale for offsets = 1152

        var ptx = new StringBuilder(16384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 offsets_ptr,");
        ptx.AppendLine("    .param .u64 grad_mask_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<16>;");
        ptx.AppendLine("    .reg .b32 %r<48>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [offsets_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [grad_mask_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // grad_mask id = gm*HW + s
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");                            // gm (mask channel, 64 = outH*outW)
        ptx.AppendLine($"    and.b32 %r4, %r2, {SpatialElements - 1};");        // s
        ptx.AppendLine("    div.u32 %r5, %r3, 9;");                            // g = gm / 9
        ptx.AppendLine("    mul.lo.u32 %r6, %r5, 9;");
        ptx.AppendLine("    sub.u32 %r6, %r3, %r6;");                          // t = gm - g*9
        ptx.AppendLine("    shr.u32 %r7, %r4, 3;");                            // oy
        ptx.AppendLine($"    and.b32 %r8, %r4, {Width - 1};");                 // ox
        ptx.AppendLine("    div.u32 %r9, %r6, 3;");                            // ky
        ptx.AppendLine("    mul.lo.u32 %r10, %r9, 3;");
        ptx.AppendLine("    sub.u32 %r10, %r6, %r10;");                        // kx
        ptx.AppendLine("    sub.s32 %r11, %r9, 1;");                           // ky - 1
        ptx.AppendLine("    sub.s32 %r12, %r10, 1;");                          // kx - 1
        // offsets: offY at flat index g*1152 + t*128 + s (offY channel = g*18 + 2t), offX at +256 bytes.
        ptx.AppendLine($"    mad.lo.u32 %r13, %r5, {maskChannelSpanElems}, %r4;");
        ptx.AppendLine("    mad.lo.u32 %r13, %r6, 128, %r13;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd3, %rd5;");                        // offY addr
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");                   // offY
        ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd5+{planeStrideBytes}];"); // offX
        // py, px (fixed per thread).
        ptx.AppendLine("    add.s32 %r14, %r7, %r11;");                        // oy + ky - 1
        ptx.AppendLine("    cvt.rn.f32.s32 %f3, %r14;");
        ptx.AppendLine("    add.rn.f32 %f3, %f3, %f1;");                        // py
        ptx.AppendLine("    add.s32 %r15, %r8, %r12;");                        // ox + kx - 1
        ptx.AppendLine("    cvt.rn.f32.s32 %f4, %r15;");
        ptx.AppendLine("    add.rn.f32 %f4, %f4, %f2;");                        // px
        ptx.AppendLine("    mov.f32 %f7, 0f3F800000;");                        // 1.0
        ptx.AppendLine("    cvt.rmi.s32.f32 %r16, %f3;");                       // y0
        ptx.AppendLine("    cvt.rmi.s32.f32 %r17, %f4;");                       // x0
        ptx.AppendLine("    add.s32 %r18, %r16, 1;");                           // y1
        ptx.AppendLine("    add.s32 %r19, %r17, 1;");                           // x1
        ptx.AppendLine("    cvt.rn.f32.s32 %f5, %r16;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f5;");                        // wy1
        ptx.AppendLine("    sub.rn.f32 %f8, %f7, %f6;");                        // wy0
        ptx.AppendLine("    cvt.rn.f32.s32 %f9, %r17;");
        ptx.AppendLine("    sub.rn.f32 %f10, %f4, %f9;");                       // wx1
        ptx.AppendLine("    sub.rn.f32 %f11, %f7, %f10;");                      // wx0
        ptx.AppendLine("    mul.rn.f32 %f12, %f8, %f11;");                      // c00
        ptx.AppendLine("    mul.rn.f32 %f13, %f8, %f10;");                      // c01
        ptx.AppendLine("    mul.rn.f32 %f14, %f6, %f11;");                      // c10
        ptx.AppendLine("    mul.rn.f32 %f15, %f6, %f10;");                      // c11
        ptx.AppendLine("    setp.ge.s32 %p0, %r16, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r16, {Height};");
        ptx.AppendLine("    and.pred %p0, %p0, %p1;");
        ptx.AppendLine("    setp.ge.s32 %p2, %r18, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r18, {Height};");
        ptx.AppendLine("    and.pred %p2, %p2, %p3;");
        ptx.AppendLine("    setp.ge.s32 %p4, %r17, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p5, %r17, {Width};");
        ptx.AppendLine("    and.pred %p4, %p4, %p5;");
        ptx.AppendLine("    setp.ge.s32 %p6, %r19, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p7, %r19, {Width};");
        ptx.AppendLine("    and.pred %p6, %p6, %p7;");
        ptx.AppendLine("    and.pred %p8, %p0, %p4;");
        ptx.AppendLine("    and.pred %p9, %p0, %p6;");
        ptx.AppendLine("    and.pred %p10, %p2, %p4;");
        ptx.AppendLine("    and.pred %p11, %p2, %p6;");
        ptx.AppendLine($"    mul.lo.s32 %r20, %r16, {Width};");
        ptx.AppendLine($"    mul.lo.s32 %r21, %r18, {Width};");
        ptx.AppendLine("    add.s32 %r22, %r20, %r17;");
        ptx.AppendLine("    add.s32 %r23, %r20, %r19;");
        ptx.AppendLine("    add.s32 %r24, %r21, %r17;");
        ptx.AppendLine("    add.s32 %r25, %r21, %r19;");
        ptx.AppendLine("    shl.b32 %r22, %r22, 2;");
        ptx.AppendLine("    shl.b32 %r23, %r23, 2;");
        ptx.AppendLine("    shl.b32 %r24, %r24, 2;");
        ptx.AppendLine("    shl.b32 %r25, %r25, 2;");
        // In-group bases: xci = input + g*512 (+256 per j); wci = weights + t*4 + g*72 (+36 per j).
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r5, {inGroupBaseStrideBytes};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd6;");                        // xci
        ptx.AppendLine("    mul.wide.u32 %rd7, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        ptx.AppendLine($"    mul.wide.u32 %rd15, %r5, {weightGroupBaseStrideBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd7, %rd15;");                       // wci
        ptx.AppendLine("    mul.wide.u32 %rd8, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd8;");                        // dOut spatial base
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accumulator
        ptx.AppendLine("    mov.u32 %r26, 0;");                                // in-group channel counter j
        ptx.AppendLine("DEFORM_GROUPED_BWD_MASK_J:");
        ptx.AppendLine("    cvt.s64.s32 %rd10, %r22;");
        ptx.AppendLine("    add.s64 %rd10, %rd6, %rd10;");
        ptx.AppendLine("    mov.f32 %f16, 0f00000000;");
        ptx.AppendLine("    @%p8 ld.global.nc.f32 %f16, [%rd10];");
        ptx.AppendLine("    cvt.s64.s32 %rd11, %r23;");
        ptx.AppendLine("    add.s64 %rd11, %rd6, %rd11;");
        ptx.AppendLine("    mov.f32 %f17, 0f00000000;");
        ptx.AppendLine("    @%p9 ld.global.nc.f32 %f17, [%rd11];");
        ptx.AppendLine("    cvt.s64.s32 %rd12, %r24;");
        ptx.AppendLine("    add.s64 %rd12, %rd6, %rd12;");
        ptx.AppendLine("    mov.f32 %f18, 0f00000000;");
        ptx.AppendLine("    @%p10 ld.global.nc.f32 %f18, [%rd12];");
        ptx.AppendLine("    cvt.s64.s32 %rd13, %r25;");
        ptx.AppendLine("    add.s64 %rd13, %rd6, %rd13;");
        ptx.AppendLine("    mov.f32 %f19, 0f00000000;");
        ptx.AppendLine("    @%p11 ld.global.nc.f32 %f19, [%rd13];");
        ptx.AppendLine("    mul.rn.f32 %f20, %f12, %f16;");
        ptx.AppendLine("    fma.rn.f32 %f20, %f13, %f17, %f20;");
        ptx.AppendLine("    fma.rn.f32 %f20, %f14, %f18, %f20;");
        ptx.AppendLine("    fma.rn.f32 %f20, %f15, %f19, %f20;");               // sample_ci
        ptx.AppendLine("    mov.u64 %rd9, %rd7;");                             // wco = wci
        ptx.AppendLine("    mov.u64 %rd14, %rd8;");                            // dco = dOut spatial base
        ptx.AppendLine("    mov.u32 %r27, 0;");                                // co counter
        ptx.AppendLine("DEFORM_GROUPED_BWD_MASK_CO:");
        ptx.AppendLine("    ld.global.nc.f32 %f21, [%rd9];");                  // W[co,ci,t]
        ptx.AppendLine("    ld.global.nc.f32 %f22, [%rd14];");                 // dOut[co,s]
        ptx.AppendLine("    mul.rn.f32 %f23, %f21, %f20;");                    // W * sample_ci
        ptx.AppendLine("    fma.rn.f32 %f0, %f22, %f23, %f0;");                // acc += dOut * (W*sample)
        ptx.AppendLine($"    add.u64 %rd9, %rd9, {weightOutStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd14, %rd14, {gradOutChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r27, %r27, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p12, %r27, {OutputChannels};");
        ptx.AppendLine("    @%p12 bra DEFORM_GROUPED_BWD_MASK_CO;");
        ptx.AppendLine($"    add.u64 %rd6, %rd6, {inChannelStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd7, %rd7, {weightChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r26, %r26, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p13, %r26, {ChannelsPerGroup};");
        ptx.AppendLine("    @%p13 bra DEFORM_GROUPED_BWD_MASK_J;");
        ptx.AppendLine("    mul.wide.u32 %rd16, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd16, %rd4, %rd16;");
        ptx.AppendLine("    st.global.f32 [%rd16], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
