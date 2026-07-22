using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW deformable 3x3 convolution backward-mask (DCNv2; stride 1,
/// pad 1, dilation 1, single deformable group). The modulation-mask gradient for tap
/// t=ky*3+kx at output position (oy,ox):
/// <c>dMask[t,oy,ox] = sum_co dOut[co,oy,ox] * sum_ci W[co,ci,t] * bilinear(X[ci], py, px)</c>
/// with <c>py=oy+ky-1+offY[t,oy,ox]</c>, <c>px=ox+kx-1+offX[t,oy,ox]</c>, over the exact
/// N1/Cin4/H8/W8/Cout4 geometry. One thread owns one mask element; since (t,oy,ox) — and
/// hence (py,px) — are fixed per thread, the four bilinear corner weights, validity
/// predicates and corner offsets are computed once, then an input-channel loop samples
/// each X[ci] and an inner output-channel loop reduces dOut*W over co (thread-private,
/// deterministic, no atomics). All tensors map to byte-exact extents.
/// </summary>
internal sealed class PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_deformconv2d_n1_cin4_h8_w8_cout4_r3_s1_p1_v2_bwd_mask";
    internal const int BlockThreads = 192;   // TapsPerChannel*SpatialElements = 576 = 192*3, no bounds check
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
    internal const int MaskElements = TapsPerChannel * SpatialElements;  // 576
    internal const int OffsetChannels = 2 * TapsPerChannel;             // 18
    internal const long GradOutputBytes = (long)OutputElements * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long OffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);
    internal const long GradMaskBytes = (long)MaskElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDeformableConv2DNchw3x3BackwardMaskF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Deformable Conv2D 3x3 backward-mask has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradMask = new DirectPtxExtent(TapsPerChannel, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-backward-mask",
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
                ["equation"] = "dMask[t,oy,ox] = sum_co dOut[co,oy,ox] * sum_ci W[co,ci,t] * bilinear(X[ci], oy+ky-1+offY, ox+kx-1+offX)",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad-mask"] = "fp32",
                ["sampling"] = "bilinear (floor + 4 corner, zero-pad) computed once per thread",
                ["reduction"] = "over ci and co (thread-private, no atomics)",
                ["deformable-groups"] = "1",
                ["layout"] = "nchw grad/input; oihw weights; [2*kHkW,outH,outW] offsets; [kHkW,outH,outW] grad-mask",
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
            _function, MaskElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 deformable Conv2D 3x3 backward-mask emitter exists.");

        const int inChannelStrideBytes = Height * Width * sizeof(float);          // 256 per ci
        const int weightChannelStrideBytes = TapsPerChannel * sizeof(float);      // 36 per ci
        const int weightOutStrideBytes = InputChannels * TapsPerChannel * sizeof(float); // 144 per co
        const int gradOutChannelStrideBytes = SpatialElements * sizeof(float);     // 256 per co
        const int offsetPlaneStrideBytes = SpatialElements * sizeof(float);        // 256 (offY->offX)

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
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // mask id = t*HW + s
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");                            // t (64 = outH*outW)
        ptx.AppendLine($"    and.b32 %r4, %r2, {SpatialElements - 1};");        // s = oy*W + ox
        ptx.AppendLine("    shr.u32 %r5, %r4, 3;");                            // oy
        ptx.AppendLine($"    and.b32 %r6, %r4, {Width - 1};");                 // ox
        ptx.AppendLine("    div.u32 %r7, %r3, 3;");                            // ky
        ptx.AppendLine("    mul.lo.u32 %r8, %r7, 3;");
        ptx.AppendLine("    sub.u32 %r9, %r3, %r8;");                          // kx
        ptx.AppendLine("    sub.s32 %r10, %r7, 1;");                           // ky - 1
        ptx.AppendLine("    sub.s32 %r11, %r9, 1;");                           // kx - 1
        // offsets: offY at (t*128 + s)*4, offX at +256.
        ptx.AppendLine("    mad.lo.u32 %r12, %r3, 128, %r4;");                  // t*128 + s
        ptx.AppendLine("    mul.wide.u32 %rd5, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd3, %rd5;");                        // offY addr
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");                   // offY
        ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd5+{offsetPlaneStrideBytes}];"); // offX
        // py, px.
        ptx.AppendLine("    add.s32 %r13, %r5, %r10;");                        // base_y = oy + ky - 1
        ptx.AppendLine("    add.s32 %r14, %r6, %r11;");                        // base_x = ox + kx - 1
        ptx.AppendLine("    cvt.rn.f32.s32 %f3, %r13;");
        ptx.AppendLine("    add.rn.f32 %f3, %f3, %f1;");                        // py
        ptx.AppendLine("    cvt.rn.f32.s32 %f4, %r14;");
        ptx.AppendLine("    add.rn.f32 %f4, %f4, %f2;");                        // px
        ptx.AppendLine("    mov.f32 %f7, 0f3F800000;");                        // 1.0
        ptx.AppendLine("    cvt.rmi.s32.f32 %r17, %f3;");                       // y0
        ptx.AppendLine("    cvt.rmi.s32.f32 %r18, %f4;");                       // x0
        ptx.AppendLine("    add.s32 %r19, %r17, 1;");                           // y1
        ptx.AppendLine("    add.s32 %r20, %r18, 1;");                           // x1
        ptx.AppendLine("    cvt.rn.f32.s32 %f5, %r17;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f5;");                        // wy1
        ptx.AppendLine("    sub.rn.f32 %f8, %f7, %f6;");                        // wy0
        ptx.AppendLine("    cvt.rn.f32.s32 %f9, %r18;");
        ptx.AppendLine("    sub.rn.f32 %f10, %f4, %f9;");                       // wx1
        ptx.AppendLine("    sub.rn.f32 %f11, %f7, %f10;");                      // wx0
        ptx.AppendLine("    mul.rn.f32 %f12, %f8, %f11;");                      // c00
        ptx.AppendLine("    mul.rn.f32 %f13, %f8, %f10;");                      // c01
        ptx.AppendLine("    mul.rn.f32 %f14, %f6, %f11;");                      // c10
        ptx.AppendLine("    mul.rn.f32 %f15, %f6, %f10;");                      // c11
        ptx.AppendLine("    setp.ge.s32 %p0, %r17, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r17, {Height};");
        ptx.AppendLine("    and.pred %p0, %p0, %p1;");                          // y0 valid
        ptx.AppendLine("    setp.ge.s32 %p2, %r19, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r19, {Height};");
        ptx.AppendLine("    and.pred %p2, %p2, %p3;");                          // y1 valid
        ptx.AppendLine("    setp.ge.s32 %p4, %r18, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p5, %r18, {Width};");
        ptx.AppendLine("    and.pred %p4, %p4, %p5;");                          // x0 valid
        ptx.AppendLine("    setp.ge.s32 %p6, %r20, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p7, %r20, {Width};");
        ptx.AppendLine("    and.pred %p6, %p6, %p7;");                          // x1 valid
        ptx.AppendLine("    and.pred %p8, %p0, %p4;");
        ptx.AppendLine("    and.pred %p9, %p0, %p6;");
        ptx.AppendLine("    and.pred %p10, %p2, %p4;");
        ptx.AppendLine("    and.pred %p11, %p2, %p6;");
        ptx.AppendLine($"    mul.lo.s32 %r21, %r17, {Width};");                // yb0
        ptx.AppendLine($"    mul.lo.s32 %r22, %r19, {Width};");                // yb1
        ptx.AppendLine("    add.s32 %r23, %r21, %r18;");
        ptx.AppendLine("    add.s32 %r24, %r21, %r20;");
        ptx.AppendLine("    add.s32 %r25, %r22, %r18;");
        ptx.AppendLine("    add.s32 %r26, %r22, %r20;");
        ptx.AppendLine("    shl.b32 %r23, %r23, 2;");                          // li00 bytes
        ptx.AppendLine("    shl.b32 %r24, %r24, 2;");                          // li01 bytes
        ptx.AppendLine("    shl.b32 %r25, %r25, 2;");                          // li10 bytes
        ptx.AppendLine("    shl.b32 %r26, %r26, 2;");                          // li11 bytes
        // dOut spatial base and weight/input bases for the ci/co loops.
        ptx.AppendLine("    mul.wide.u32 %rd6, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd6;");                        // dOut + s*4 (reset each ci)
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd7;");                        // weights + t*4 (wbase_ci, +36 per ci)
        ptx.AppendLine("    mov.u64 %rd8, %rd1;");                             // input (Xci base, +256 per ci)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accumulator
        ptx.AppendLine("    mov.u32 %r15, 0;");                                // ci counter
        ptx.AppendLine("DEFORM_BWD_MASK_CI:");
        // Bilinear sample of X[ci] at the four fixed corners.
        ptx.AppendLine("    cvt.s64.s32 %rd10, %r23;");
        ptx.AppendLine("    add.s64 %rd10, %rd8, %rd10;");
        ptx.AppendLine("    mov.f32 %f17, 0f00000000;");
        ptx.AppendLine("    @%p8 ld.global.nc.f32 %f17, [%rd10];");
        ptx.AppendLine("    cvt.s64.s32 %rd11, %r24;");
        ptx.AppendLine("    add.s64 %rd11, %rd8, %rd11;");
        ptx.AppendLine("    mov.f32 %f18, 0f00000000;");
        ptx.AppendLine("    @%p9 ld.global.nc.f32 %f18, [%rd11];");
        ptx.AppendLine("    cvt.s64.s32 %rd12, %r25;");
        ptx.AppendLine("    add.s64 %rd12, %rd8, %rd12;");
        ptx.AppendLine("    mov.f32 %f19, 0f00000000;");
        ptx.AppendLine("    @%p10 ld.global.nc.f32 %f19, [%rd12];");
        ptx.AppendLine("    cvt.s64.s32 %rd13, %r26;");
        ptx.AppendLine("    add.s64 %rd13, %rd8, %rd13;");
        ptx.AppendLine("    mov.f32 %f20, 0f00000000;");
        ptx.AppendLine("    @%p11 ld.global.nc.f32 %f20, [%rd13];");
        ptx.AppendLine("    mul.rn.f32 %f21, %f12, %f17;");
        ptx.AppendLine("    fma.rn.f32 %f21, %f13, %f18, %f21;");
        ptx.AppendLine("    fma.rn.f32 %f21, %f14, %f19, %f21;");
        ptx.AppendLine("    fma.rn.f32 %f21, %f15, %f20, %f21;");               // sample_ci
        // Inner co reduction: acc += dOut[co,s] * W[co,ci,t] * sample_ci.
        ptx.AppendLine("    mov.u64 %rd9, %rd7;");                             // wco = wbase_ci
        ptx.AppendLine("    mov.u64 %rd14, %rd6;");                            // dco = dOut spatial base
        ptx.AppendLine("    mov.u32 %r16, 0;");                                // co counter
        ptx.AppendLine("DEFORM_BWD_MASK_CO:");
        ptx.AppendLine("    ld.global.nc.f32 %f22, [%rd9];");                  // W[co,ci,t]
        ptx.AppendLine("    ld.global.nc.f32 %f23, [%rd14];");                 // dOut[co,s]
        ptx.AppendLine("    mul.rn.f32 %f24, %f22, %f21;");                    // W * sample_ci
        ptx.AppendLine("    fma.rn.f32 %f0, %f23, %f24, %f0;");                // acc += dOut * (W*sample)
        ptx.AppendLine($"    add.u64 %rd9, %rd9, {weightOutStrideBytes};");    // next co weight (+144)
        ptx.AppendLine($"    add.u64 %rd14, %rd14, {gradOutChannelStrideBytes};"); // next co dOut (+256)
        ptx.AppendLine("    add.u32 %r16, %r16, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p12, %r16, {OutputChannels};");
        ptx.AppendLine("    @%p12 bra DEFORM_BWD_MASK_CO;");
        // Advance to next input channel.
        ptx.AppendLine($"    add.u64 %rd8, %rd8, {inChannelStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd7, %rd7, {weightChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r15, %r15, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p13, %r15, {InputChannels};");
        ptx.AppendLine("    @%p13 bra DEFORM_BWD_MASK_CI;");
        ptx.AppendLine("    mul.wide.u32 %rd15, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd15, %rd4, %rd15;");
        ptx.AppendLine("    st.global.f32 [%rd15], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
