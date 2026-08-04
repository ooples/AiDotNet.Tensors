using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW deformable 3x3 convolution backward-offset (DCNv2; stride 1,
/// pad 1, dilation 1, single deformable group). The offset gradient flows through the
/// spatial derivative of the bilinear sampler. With fy=py-y0, fx=px-x0 and predicated
/// corner values V00,V01,V10,V11 (zero outside the frame):
/// <c>dBil/dpy = (1-fx)(V10-V00) + fx(V11-V01)</c>,
/// <c>dBil/dpx = (1-fy)(V01-V00) + fy(V11-V10)</c>, and
/// <c>dOffY[t,oy,ox] = mask[t,oy,ox] * sum_ci (sum_co dOut[co,oy,ox] W[co,ci,t]) * dBil/dpy_ci</c>
/// (dOffX analogously with dBil/dpx), over the exact N1/Cin4/H8/W8/Cout4 geometry. One
/// thread owns one (tap, position) and writes BOTH the y- and x-offset gradients. The
/// bilinear geometry is computed once; an input-channel loop forms the per-channel
/// derivatives and an inner output-channel loop reduces dOut*W (thread-private,
/// deterministic, no atomics). All tensors map to byte-exact extents.
/// </summary>
internal sealed class PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_deformconv2d_n1_cin4_h8_w8_cout4_r3_s1_p1_v2_bwd_offset";
    internal const int BlockThreads = 192;   // TapsPerChannel*SpatialElements = 576 = 192*3
    internal const int Batch = 1;
    internal const int InputChannels = 4;
    internal const int Height = 8;
    internal const int Width = 8;
    internal const int OutputChannels = 4;
    internal const int OutHeight = Height;
    internal const int OutWidth = Width;
    internal const int KernelSize = 3;
    internal const int TapsPerChannel = KernelSize * KernelSize;         // 9
    internal const int SpatialElements = OutHeight * OutWidth;           // 64
    internal const int OutputElements = Batch * OutputChannels * SpatialElements; // 256
    internal const int TapPositionElements = TapsPerChannel * SpatialElements; // 576 (threads)
    internal const int OffsetChannels = 2 * TapsPerChannel;             // 18
    internal const long GradOutputBytes = (long)OutputElements * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long OffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);
    internal const long MaskBytes = (long)TapsPerChannel * SpatialElements * sizeof(float);
    internal const long GradOffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDeformableConv2DNchw3x3BackwardOffsetF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Deformable Conv2D 3x3 backward-offset has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var mask = new DirectPtxExtent(TapsPerChannel, OutHeight, OutWidth);
        var gradOffset = new DirectPtxExtent(OffsetChannels, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-backward-offset",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin4-h8-w8-cout4-r3-s1-p1-v2-fp32",
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
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_offset", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOffset, gradOffset, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dOffY[t,oy,ox] = mask * sum_ci (sum_co dOut[co] W[co,ci,t]) * dBil/dpy; dOffX with dBil/dpx",
                ["dbil-dpy"] = "(1-fx)(V10-V00) + fx(V11-V01)",
                ["dbil-dpx"] = "(1-fy)(V01-V00) + fy(V11-V10)",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-fma (dual: offY, offX)",
                ["grad-offset"] = "fp32 [2*kHkW,outH,outW]",
                ["reduction"] = "over ci and co (thread-private, no atomics)",
                ["deformable-groups"] = "1",
                ["layout"] = "nchw grad/input; oihw weights; [2*kHkW,outH,outW] offsets/grad-offset; [kHkW,outH,outW] mask",
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
        DirectPtxTensorView mask,
        DirectPtxTensorView gradOffset)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(weights, Blueprint.Tensors[2], nameof(weights));
        Require(offsets, Blueprint.Tensors[3], nameof(offsets));
        Require(mask, Blueprint.Tensors[4], nameof(mask));
        Require(gradOffset, Blueprint.Tensors[5], nameof(gradOffset));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr offsetPointer = offsets.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr gradOffsetPointer = gradOffset.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &weightPointer;
        arguments[3] = &offsetPointer;
        arguments[4] = &maskPointer;
        arguments[5] = &gradOffsetPointer;
        _module.Launch(
            _function, TapPositionElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 deformable Conv2D 3x3 backward-offset emitter exists.");

        const int inChannelStrideBytes = Height * Width * sizeof(float);          // 256 per ci
        const int weightChannelStrideBytes = TapsPerChannel * sizeof(float);      // 36 per ci
        const int weightOutStrideBytes = InputChannels * TapsPerChannel * sizeof(float); // 144 per co
        const int gradOutChannelStrideBytes = SpatialElements * sizeof(float);     // 256 per co
        const int offsetPlaneStrideBytes = SpatialElements * sizeof(float);        // 256 (offY->offX / dOffY->dOffX)

        var ptx = new StringBuilder(20480);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 offsets_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 grad_offset_ptr");
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
        ptx.AppendLine("    ld.param.u64 %rd4, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [grad_offset_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // id = t*HW + s
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");                            // t
        ptx.AppendLine($"    and.b32 %r4, %r2, {SpatialElements - 1};");        // s
        ptx.AppendLine("    shr.u32 %r5, %r4, 3;");                            // oy
        ptx.AppendLine($"    and.b32 %r6, %r4, {Width - 1};");                 // ox
        ptx.AppendLine("    div.u32 %r7, %r3, 3;");                            // ky
        ptx.AppendLine("    mul.lo.u32 %r8, %r7, 3;");
        ptx.AppendLine("    sub.u32 %r9, %r3, %r8;");                          // kx
        ptx.AppendLine("    sub.s32 %r10, %r7, 1;");                           // ky - 1
        ptx.AppendLine("    sub.s32 %r11, %r9, 1;");                           // kx - 1
        // offsets: offY at (t*128 + s)*4, offX at +256; reuse idx for grad_offset stores.
        ptx.AppendLine("    mad.lo.u32 %r24, %r3, 128, %r4;");                  // t*128 + s
        ptx.AppendLine("    mul.wide.u32 %rd6, %r24, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd6;");                        // offY addr
        ptx.AppendLine("    ld.global.nc.f32 %f20, [%rd6];");                  // offY
        ptx.AppendLine($"    ld.global.nc.f32 %f21, [%rd6+{offsetPlaneStrideBytes}];"); // offX
        ptx.AppendLine("    add.s32 %r12, %r5, %r10;");                        // base_y
        ptx.AppendLine("    add.s32 %r13, %r6, %r11;");                        // base_x
        ptx.AppendLine("    cvt.rn.f32.s32 %f16, %r12;");
        ptx.AppendLine("    add.rn.f32 %f16, %f16, %f20;");                     // py
        ptx.AppendLine("    cvt.rn.f32.s32 %f17, %r13;");
        ptx.AppendLine("    add.rn.f32 %f17, %f17, %f21;");                     // px
        ptx.AppendLine("    mov.f32 %f7, 0f3F800000;");                        // 1.0
        ptx.AppendLine("    cvt.rmi.s32.f32 %r14, %f16;");                      // y0
        ptx.AppendLine("    cvt.rmi.s32.f32 %r15, %f17;");                      // x0
        ptx.AppendLine("    add.s32 %r16, %r14, 1;");                           // y1
        ptx.AppendLine("    add.s32 %r17, %r15, 1;");                           // x1
        ptx.AppendLine("    cvt.rn.f32.s32 %f18, %r14;");
        ptx.AppendLine("    sub.rn.f32 %f4, %f16, %f18;");                      // fy = wy1
        ptx.AppendLine("    sub.rn.f32 %f3, %f7, %f4;");                        // wy0 = 1 - fy
        ptx.AppendLine("    cvt.rn.f32.s32 %f19, %r15;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f17, %f19;");                      // fx = wx1
        ptx.AppendLine("    sub.rn.f32 %f5, %f7, %f6;");                        // wx0 = 1 - fx
        ptx.AppendLine("    setp.ge.s32 %p0, %r14, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r14, {Height};");
        ptx.AppendLine("    and.pred %p0, %p0, %p1;");                          // y0 valid
        ptx.AppendLine("    setp.ge.s32 %p2, %r16, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r16, {Height};");
        ptx.AppendLine("    and.pred %p2, %p2, %p3;");                          // y1 valid
        ptx.AppendLine("    setp.ge.s32 %p4, %r15, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p5, %r15, {Width};");
        ptx.AppendLine("    and.pred %p4, %p4, %p5;");                          // x0 valid
        ptx.AppendLine("    setp.ge.s32 %p6, %r17, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p7, %r17, {Width};");
        ptx.AppendLine("    and.pred %p6, %p6, %p7;");                          // x1 valid
        ptx.AppendLine("    and.pred %p8, %p0, %p4;");                          // V00
        ptx.AppendLine("    and.pred %p9, %p0, %p6;");                          // V01
        ptx.AppendLine("    and.pred %p10, %p2, %p4;");                         // V10
        ptx.AppendLine("    and.pred %p11, %p2, %p6;");                         // V11
        ptx.AppendLine($"    mul.lo.s32 %r18, %r14, {Width};");                // yb0
        ptx.AppendLine($"    mul.lo.s32 %r19, %r16, {Width};");                // yb1
        ptx.AppendLine("    add.s32 %r20, %r18, %r15;");
        ptx.AppendLine("    add.s32 %r21, %r18, %r17;");
        ptx.AppendLine("    add.s32 %r22, %r19, %r15;");
        ptx.AppendLine("    add.s32 %r23, %r19, %r17;");
        ptx.AppendLine("    shl.b32 %r20, %r20, 2;");                          // li00
        ptx.AppendLine("    shl.b32 %r21, %r21, 2;");                          // li01
        ptx.AppendLine("    shl.b32 %r22, %r22, 2;");                          // li10
        ptx.AppendLine("    shl.b32 %r23, %r23, 2;");                          // li11
        // mask[id] = mask + id*4 (mask index == t*HW + s == id).
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");                   // mask
        // Bases for the ci/co loops.
        ptx.AppendLine("    mul.wide.u32 %rd8, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd8;");                        // dOut + s*4 (reset each ci)
        ptx.AppendLine("    mul.wide.u32 %rd9, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");                        // weights + t*4 (wbase_ci)
        ptx.AppendLine("    mov.u64 %rd10, %rd1;");                            // input (Xci base)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accY
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                        // accX
        ptx.AppendLine("    mov.u32 %r25, 0;");                                // ci counter
        ptx.AppendLine("DEFORM_BWD_OFFSET_CI:");
        // Four predicated corner values.
        ptx.AppendLine("    cvt.s64.s32 %rd11, %r20;");
        ptx.AppendLine("    add.s64 %rd11, %rd10, %rd11;");
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        ptx.AppendLine("    @%p8 ld.global.nc.f32 %f8, [%rd11];");             // V00
        ptx.AppendLine("    cvt.s64.s32 %rd12, %r21;");
        ptx.AppendLine("    add.s64 %rd12, %rd10, %rd12;");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
        ptx.AppendLine("    @%p9 ld.global.nc.f32 %f9, [%rd12];");             // V01
        ptx.AppendLine("    cvt.s64.s32 %rd13, %r22;");
        ptx.AppendLine("    add.s64 %rd13, %rd10, %rd13;");
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
        ptx.AppendLine("    @%p10 ld.global.nc.f32 %f10, [%rd13];");           // V10
        ptx.AppendLine("    cvt.s64.s32 %rd14, %r23;");
        ptx.AppendLine("    add.s64 %rd14, %rd10, %rd14;");
        ptx.AppendLine("    mov.f32 %f11, 0f00000000;");
        ptx.AppendLine("    @%p11 ld.global.nc.f32 %f11, [%rd14];");           // V11
        // dpy = wx0*(V10-V00) + wx1*(V11-V01).
        ptx.AppendLine("    sub.rn.f32 %f14, %f10, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f12, %f5, %f14;");
        ptx.AppendLine("    sub.rn.f32 %f14, %f11, %f9;");
        ptx.AppendLine("    fma.rn.f32 %f12, %f6, %f14, %f12;");                // dpy_ci
        // dpx = wy0*(V01-V00) + wy1*(V11-V10).
        ptx.AppendLine("    sub.rn.f32 %f14, %f9, %f8;");
        ptx.AppendLine("    mul.rn.f32 %f13, %f3, %f14;");
        ptx.AppendLine("    sub.rn.f32 %f14, %f11, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f13, %f4, %f14, %f13;");                // dpx_ci
        // G_ci = sum_co dOut[co,s] * W[co,ci,t].
        ptx.AppendLine("    mov.u64 %rd15, %rd9;");                            // wco = wbase_ci
        ptx.AppendLine("    mov.u64 %rd16, %rd8;");                            // dco = dOut spatial base
        ptx.AppendLine("    mov.f32 %f15, 0f00000000;");                       // G_ci
        ptx.AppendLine("    mov.u32 %r26, 0;");                                // co counter
        ptx.AppendLine("DEFORM_BWD_OFFSET_CO:");
        ptx.AppendLine("    ld.global.nc.f32 %f22, [%rd15];");                 // W[co,ci,t]
        ptx.AppendLine("    ld.global.nc.f32 %f23, [%rd16];");                 // dOut[co,s]
        ptx.AppendLine("    fma.rn.f32 %f15, %f23, %f22, %f15;");              // G_ci += dOut*W
        ptx.AppendLine($"    add.u64 %rd15, %rd15, {weightOutStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd16, %rd16, {gradOutChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r26, %r26, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p12, %r26, {OutputChannels};");
        ptx.AppendLine("    @%p12 bra DEFORM_BWD_OFFSET_CO;");
        // accY += G_ci * dpy_ci ; accX += G_ci * dpx_ci.
        ptx.AppendLine("    fma.rn.f32 %f0, %f15, %f12, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f1, %f15, %f13, %f1;");
        // Advance to next input channel.
        ptx.AppendLine($"    add.u64 %rd10, %rd10, {inChannelStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd9, %rd9, {weightChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r25, %r25, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p13, %r25, {InputChannels};");
        ptx.AppendLine("    @%p13 bra DEFORM_BWD_OFFSET_CI;");
        // Scale by mask and store dOffY, dOffX (grad_offset channels 2t, 2t+1).
        ptx.AppendLine("    mul.rn.f32 %f24, %f2, %f0;");                       // dOffY = mask*accY
        ptx.AppendLine("    mul.rn.f32 %f25, %f2, %f1;");                       // dOffX = mask*accX
        ptx.AppendLine("    mul.wide.u32 %rd17, %r24, 4;");
        ptx.AppendLine("    add.u64 %rd17, %rd5, %rd17;");                      // grad_offset dOffY addr
        ptx.AppendLine("    st.global.f32 [%rd17], %f24;");
        ptx.AppendLine($"    st.global.f32 [%rd17+{offsetPlaneStrideBytes}], %f25;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
