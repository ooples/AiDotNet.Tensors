using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW grouped deformable 3x3 convolution backward-input (DCNv2 with G
/// deformable groups; stride 1, pad 1, dilation 1). Formulated as a deterministic gather
/// (the transpose of the forward bilinear sampler): the weight with which input pixel
/// (iy,ix) contributes to a sample at (py,px) is g(iy-py)*g(ix-px) with g(d)=max(0,1-|d|).
/// The input channel ci uses its own deformable group gg = ci/(Cin/G) for the offset/mask
/// field, so:
/// <c>dX[ci,iy,ix] = sum_{t,oy,ox} g(iy-py)g(ix-px) * mask[gg*9+t,oy,ox] * sum_co dOut[co,oy,ox] W[co,ci,t]</c>
/// with py=oy+ky-1+offY[gg*18+2t,oy,ox], px=ox+kx-1+offX[gg*18+2t+1,oy,ox], over the exact
/// N1/Cin4/H8/W8/Cout4 G2 geometry. One thread owns one input element and sweeps every
/// (tap, output position) via a triple loop, reducing dOut*W over co. Thread-private,
/// deterministic, no atomics, no input tensor needed. Last of the grouped-backward family.
/// </summary>
internal sealed class PtxDeformableConv2DGroupedNchw3x3BackwardInputF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_deformconv2d_grouped_n1_cin4_h8_w8_cout4_r3_s1_p1_g2_v2_bwd_input";
    internal const int BlockThreads = 128;   // Cin*H*W = 256 = 128*2
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
    internal const int InputElements = Batch * InputChannels * Height * Width;    // 256
    internal const int OffsetChannels = DeformableGroups * 2 * TapsPerChannel; // 36
    internal const int MaskChannels = DeformableGroups * TapsPerChannel;       // 18
    internal const long GradOutputBytes = (long)OutputElements * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long OffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);
    internal const long MaskBytes = (long)MaskChannels * SpatialElements * sizeof(float);
    internal const long GradInputBytes = (long)InputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDeformableConv2DGroupedNchw3x3BackwardInputF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Grouped deformable Conv2D 3x3 backward-input has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, KernelSize, KernelSize);
        var offsets = new DirectPtxExtent(OffsetChannels, OutHeight, OutWidth);
        var mask = new DirectPtxExtent(MaskChannels, OutHeight, OutWidth);
        var gradInput = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-grouped-backward-input",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin4-h8-w8-cout4-r3-s1-p1-g2-v2-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offsets", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    offsets, offsets, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradInput, gradInput, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dX[ci,iy,ix] = sum_{t,oy,ox} g(iy-py)g(ix-px) * mask[gg*9+t] * sum_co dOut[co] W[co,ci,t]; gg=ci/(Cin/G), g(d)=max(0,1-|d|)",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad-input"] = "fp32",
                ["sampling"] = "bilinear transpose gather (g(d)=max(0,1-|d|), no input tensor needed)",
                ["reduction"] = "over tap, output position, co (thread-private, no atomics)",
                ["deformable-groups"] = "2",
                ["layout"] = "nchw grad-output/grad-input; oihw weights; [G*2*kHkW,outH,outW] offsets; [G*kHkW,outH,outW] mask",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView weights,
        DirectPtxTensorView offsets,
        DirectPtxTensorView mask,
        DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(offsets, Blueprint.Tensors[2], nameof(offsets));
        Require(mask, Blueprint.Tensors[3], nameof(mask));
        Require(gradInput, Blueprint.Tensors[4], nameof(gradInput));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr offsetPointer = offsets.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &offsetPointer;
        arguments[3] = &maskPointer;
        arguments[4] = &gradInputPointer;
        _module.Launch(
            _function, InputElements / BlockThreads, 1, 1,
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
                "Only the experimental SM86 grouped deformable Conv2D 3x3 backward-input emitter exists.");

        const int weightChannelStrideBytes = InputChannels * TapsPerChannel * sizeof(float); // 144 per co
        const int gradOutChannelStrideBytes = SpatialElements * sizeof(float);                // 256 per co / offY->offX plane
        const int ciWeightBaseBytes = TapsPerChannel * sizeof(float);                         // ci*36
        const int offsetGroupSpanElems = 2 * TapsPerChannel * SpatialElements;                // 1152 (g scale for offsets)
        const int maskGroupSpanElems = TapsPerChannel * SpatialElements;                      // 576 (g scale for mask)

        var ptx = new StringBuilder(16384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 offsets_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 grad_input_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<40>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [offsets_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [grad_input_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // input id = ci*HW + s_in
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");                            // ci
        ptx.AppendLine($"    and.b32 %r4, %r2, {SpatialElements - 1};");        // s_in
        ptx.AppendLine("    shr.u32 %r5, %r4, 3;");                            // iy
        ptx.AppendLine($"    and.b32 %r6, %r4, {Width - 1};");                 // ix
        ptx.AppendLine("    shr.u32 %r7, %r3, 1;");                            // gg = ci / ChannelsPerGroup (2)
        ptx.AppendLine($"    mul.lo.u32 %r8, %r7, {offsetGroupSpanElems};");   // gg*1152 (offset group base elems)
        ptx.AppendLine($"    mul.lo.u32 %r9, %r7, {maskGroupSpanElems};");     // gg*576 (mask group base elems)
        ptx.AppendLine("    cvt.rn.f32.s32 %f13, %r5;");                        // iy as float
        ptx.AppendLine("    cvt.rn.f32.s32 %f14, %r6;");                        // ix as float
        ptx.AppendLine("    mov.f32 %f15, 0f3F800000;");                        // 1.0
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r3, {ciWeightBaseBytes};");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd5;");                        // weights + ci*36
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accumulator
        ptx.AppendLine("    mov.u32 %r10, 0;");                                // tap counter t
        ptx.AppendLine("DEFORM_GROUPED_BWD_INPUT_T:");
        ptx.AppendLine("    div.u32 %r11, %r10, 3;");                          // ky
        ptx.AppendLine("    mul.lo.u32 %r12, %r11, 3;");
        ptx.AppendLine("    sub.u32 %r13, %r10, %r12;");                       // kx
        ptx.AppendLine("    sub.s32 %r14, %r11, 1;");                          // ky - 1
        ptx.AppendLine("    sub.s32 %r15, %r13, 1;");                          // kx - 1
        ptx.AppendLine("    mul.wide.u32 %rd6, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd5, %rd6;");                        // wtap = weights + ci*36 + t*4
        ptx.AppendLine("    mov.u32 %r16, 0;");                                // spatial counter s
        ptx.AppendLine("DEFORM_GROUPED_BWD_INPUT_S:");
        ptx.AppendLine("    shr.u32 %r17, %r16, 3;");                          // oy
        ptx.AppendLine($"    and.b32 %r18, %r16, {Width - 1};");               // ox
        // offsets: offY flat index = gg*1152 + t*128 + s; offX at +256 bytes.
        ptx.AppendLine("    mad.lo.u32 %r19, %r10, 128, %r16;");
        ptx.AppendLine("    add.u32 %r19, %r19, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r19, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd7];");                   // offY
        ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd7+{gradOutChannelStrideBytes}];"); // offX (+256)
        // mask flat index = gg*576 + t*64 + s.
        ptx.AppendLine("    mad.lo.u32 %r20, %r10, 64, %r16;");
        ptx.AppendLine("    add.u32 %r20, %r20, %r9;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r20, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd8;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd8];");                   // mask_ts
        // py, px.
        ptx.AppendLine("    add.s32 %r21, %r17, %r14;");                        // oy + ky - 1
        ptx.AppendLine("    cvt.rn.f32.s32 %f4, %r21;");
        ptx.AppendLine("    add.rn.f32 %f4, %f4, %f1;");                        // py
        ptx.AppendLine("    add.s32 %r22, %r18, %r15;");                        // ox + kx - 1
        ptx.AppendLine("    cvt.rn.f32.s32 %f5, %r22;");
        ptx.AppendLine("    add.rn.f32 %f5, %f5, %f2;");                        // px
        // gy = max(0, 1 - |iy - py|), gx = max(0, 1 - |ix - px|).
        ptx.AppendLine("    sub.rn.f32 %f6, %f13, %f4;");
        ptx.AppendLine("    abs.f32 %f6, %f6;");
        ptx.AppendLine("    sub.rn.f32 %f6, %f15, %f6;");
        ptx.AppendLine("    max.f32 %f6, %f6, 0f00000000;");                    // gy
        ptx.AppendLine("    sub.rn.f32 %f7, %f14, %f5;");
        ptx.AppendLine("    abs.f32 %f7, %f7;");
        ptx.AppendLine("    sub.rn.f32 %f7, %f15, %f7;");
        ptx.AppendLine("    max.f32 %f7, %f7, 0f00000000;");                    // gx
        ptx.AppendLine("    mul.rn.f32 %f8, %f6, %f7;");                        // coeff
        ptx.AppendLine("    mul.rn.f32 %f9, %f8, %f3;");                        // cm = coeff * mask_ts
        // G = sum_co dOut[co,s] * W[co,ci,t].
        ptx.AppendLine("    mul.wide.u32 %rd9, %r16, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd9;");                        // dOut + s*4 (dco)
        ptx.AppendLine("    mov.u64 %rd10, %rd6;");                            // wco = wtap
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");                       // G
        ptx.AppendLine("    mov.u32 %r23, 0;");                                // co counter
        ptx.AppendLine("DEFORM_GROUPED_BWD_INPUT_CO:");
        ptx.AppendLine("    ld.global.nc.f32 %f11, [%rd10];");                 // W[co,ci,t]
        ptx.AppendLine("    ld.global.nc.f32 %f12, [%rd9];");                  // dOut[co,s]
        ptx.AppendLine("    fma.rn.f32 %f10, %f12, %f11, %f10;");
        ptx.AppendLine($"    add.u64 %rd10, %rd10, {weightChannelStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd9, %rd9, {gradOutChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r23, %r23, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r23, {OutputChannels};");
        ptx.AppendLine("    @%p0 bra DEFORM_GROUPED_BWD_INPUT_CO;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f9, %f10, %f0;");                  // acc += cm * G
        ptx.AppendLine("    add.u32 %r16, %r16, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p1, %r16, {SpatialElements};");
        ptx.AppendLine("    @%p1 bra DEFORM_GROUPED_BWD_INPUT_S;");
        ptx.AppendLine("    add.u32 %r10, %r10, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r10, {TapsPerChannel};");
        ptx.AppendLine("    @%p2 bra DEFORM_GROUPED_BWD_INPUT_T;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd4, %rd11;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
