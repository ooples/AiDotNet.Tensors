using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW deformable 3x3 convolution backward-weight (DCNv2; stride 1,
/// pad 1, dilation 1, single deformable group). The weight gradient is a reduction over
/// output positions of the modulated bilinear sample used in the forward pass:
/// <c>dW[co,ci,ky,kx] = sum_{oy,ox} dOut[co,oy,ox] * mask[t,oy,ox] * bilinear(X[ci], py, px)</c>
/// with <c>t=ky*3+kx</c>, <c>py=oy+ky-1+offY[t,oy,ox]</c>, <c>px=ox+kx-1+offX[t,oy,ox]</c>,
/// over the exact N1/Cin4/H8/W8/Cout4 geometry. One thread owns one weight; it decomposes
/// its flat id into (co,ci,ky,kx), then sweeps all 64 output positions, recomputing the
/// same bilinear sample (floor + 4 corner, zero-padded) as the forward kernel and
/// accumulating (thread-private, deterministic, no atomics). All tensors map to byte-exact
/// extents.
/// </summary>
internal sealed class PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_deformconv2d_n1_cin4_h8_w8_cout4_r3_s1_p1_v2_bwd_weight";
    internal const int BlockThreads = 144;   // Cout*Cin*9 = one weight per thread, single block
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
    internal const int WeightElements = OutputChannels * InputChannels * TapsPerChannel; // 144
    internal const int OffsetChannels = 2 * TapsPerChannel;             // 18
    internal const long GradOutputBytes = (long)OutputElements * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long OffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);
    internal const long MaskBytes = (long)TapsPerChannel * SpatialElements * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDeformableConv2DNchw3x3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Deformable Conv2D 3x3 backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var offsets = new DirectPtxExtent(OffsetChannels, OutHeight, OutWidth);
        var mask = new DirectPtxExtent(TapsPerChannel, OutHeight, OutWidth);
        var gradWeight = new DirectPtxExtent(OutputChannels, InputChannels, KernelSize, KernelSize);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d-backward-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin4-h8-w8-cout4-r3-s1-p1-v2-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("offsets", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    offsets, offsets, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mask", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    mask, mask, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_weight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    gradWeight, gradWeight, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // 0 = per-thread register ceiling derived from the device register
                // file at validation time; not pinned to a hardcoded literal.
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "dW[co,ci,ky,kx] = sum_{oy,ox} dOut[co,oy,ox] * mask[t,oy,ox] * bilinear(X[ci], oy+ky-1+offY, ox+kx-1+offX)",
                ["grad-output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad-weight"] = "fp32",
                ["sampling"] = "bilinear (floor + 4 corner, zero-pad) recomputed per output position",
                ["reduction"] = "spatial (thread-private per weight, no atomics)",
                ["deformable-groups"] = "1",
                ["layout"] = "nchw grad/input; oihw grad-weight; [2*kHkW,outH,outW] offsets; [kHkW,outH,outW] mask",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView offsets,
        DirectPtxTensorView mask,
        DirectPtxTensorView gradWeight)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(offsets, Blueprint.Tensors[2], nameof(offsets));
        Require(mask, Blueprint.Tensors[3], nameof(mask));
        Require(gradWeight, Blueprint.Tensors[4], nameof(gradWeight));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr offsetPointer = offsets.Pointer;
        IntPtr maskPointer = mask.Pointer;
        IntPtr gradWeightPointer = gradWeight.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &offsetPointer;
        arguments[3] = &maskPointer;
        arguments[4] = &gradWeightPointer;
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
                "Only the experimental SM86 deformable Conv2D 3x3 backward-weight emitter exists.");

        const int inChannelStrideBytes = Height * Width * sizeof(float);   // 256 per ci
        const int planeStrideBytes = OutHeight * OutWidth * sizeof(float); // 256 per offset/mask/grad-output channel
        const int offsetTapStrideBytes = 2 * planeStrideBytes;            // 512 per tap (offY then offX plane)

        var ptx = new StringBuilder(16384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 offsets_ptr,");
        ptx.AppendLine("    .param .u64 mask_ptr,");
        ptx.AppendLine("    .param .u64 grad_weight_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<16>;");
        ptx.AppendLine("    .reg .b32 %r<48>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<32>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [offsets_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [mask_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [grad_weight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // flat weight id
        // Decompose id = co*36 + ci*9 + t.
        ptx.AppendLine("    div.u32 %r3, %r2, 9;");                            // rest = id / 9
        ptx.AppendLine("    mul.lo.u32 %r4, %r3, 9;");
        ptx.AppendLine("    sub.u32 %r5, %r2, %r4;");                          // t = id - rest*9
        ptx.AppendLine("    and.b32 %r6, %r3, 3;");                            // ci = rest & 3
        ptx.AppendLine("    shr.u32 %r7, %r3, 2;");                            // co = rest >> 2
        ptx.AppendLine("    div.u32 %r8, %r5, 3;");                            // ky = t / 3
        ptx.AppendLine("    mul.lo.u32 %r9, %r8, 3;");
        ptx.AppendLine("    sub.u32 %r10, %r5, %r9;");                         // kx = t - ky*3
        ptx.AppendLine("    sub.s32 %r11, %r8, 1;");                           // ky - 1
        ptx.AppendLine("    sub.s32 %r12, %r10, 1;");                          // kx - 1
        // Per-thread bases (advance by +4 bytes per spatial step).
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r5, {offsetTapStrideBytes};");
        ptx.AppendLine("    add.u64 %rd5, %rd2, %rd5;");                        // offY plane base (offsets + t*512)
        ptx.AppendLine($"    add.u64 %rd6, %rd5, {planeStrideBytes};");        // offX plane base (+256)
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r5, {planeStrideBytes};");
        ptx.AppendLine("    add.u64 %rd7, %rd3, %rd7;");                        // mask plane base (mask + t*256)
        ptx.AppendLine($"    mul.wide.u32 %rd8, %r7, {planeStrideBytes};");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd8;");                        // grad_output co base (+co*256)
        ptx.AppendLine($"    mul.wide.u32 %rd9, %r6, {inChannelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd9;");                        // input ci base (+ci*256)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accumulator
        ptx.AppendLine("    mov.f32 %f23, 0f3F800000;");                       // constant 1.0
        ptx.AppendLine("    mov.u32 %r13, 0;");                                // spatial counter s
        ptx.AppendLine("DEFORM_BWD_WEIGHT:");
        ptx.AppendLine("    shr.u32 %r14, %r13, 3;");                          // oy = s / W
        ptx.AppendLine($"    and.b32 %r15, %r13, {Width - 1};");               // ox = s % W
        ptx.AppendLine("    add.s32 %r16, %r14, %r11;");                        // base_y = oy + ky - 1
        ptx.AppendLine("    add.s32 %r17, %r15, %r12;");                        // base_x = ox + kx - 1
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");                   // offY
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");                   // offX
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd7];");                   // mask_s
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd8];");                   // dOut_s
        ptx.AppendLine("    cvt.rn.f32.s32 %f5, %r16;");
        ptx.AppendLine("    add.rn.f32 %f5, %f5, %f1;");                        // py
        ptx.AppendLine("    cvt.rn.f32.s32 %f6, %r17;");
        ptx.AppendLine("    add.rn.f32 %f6, %f6, %f2;");                        // px
        ptx.AppendLine("    cvt.rmi.s32.f32 %r18, %f5;");                       // y0
        ptx.AppendLine("    cvt.rmi.s32.f32 %r19, %f6;");                       // x0
        ptx.AppendLine("    add.s32 %r20, %r18, 1;");                           // y1
        ptx.AppendLine("    add.s32 %r21, %r19, 1;");                           // x1
        ptx.AppendLine("    cvt.rn.f32.s32 %f7, %r18;");
        ptx.AppendLine("    sub.rn.f32 %f8, %f5, %f7;");                        // wy1
        ptx.AppendLine("    sub.rn.f32 %f9, %f23, %f8;");                       // wy0
        ptx.AppendLine("    cvt.rn.f32.s32 %f10, %r19;");
        ptx.AppendLine("    sub.rn.f32 %f11, %f6, %f10;");                      // wx1
        ptx.AppendLine("    sub.rn.f32 %f12, %f23, %f11;");                     // wx0
        ptx.AppendLine("    mul.rn.f32 %f13, %f9, %f12;");                      // c00
        ptx.AppendLine("    mul.rn.f32 %f14, %f9, %f11;");                      // c01
        ptx.AppendLine("    mul.rn.f32 %f15, %f8, %f12;");                      // c10
        ptx.AppendLine("    mul.rn.f32 %f16, %f8, %f11;");                      // c11
        ptx.AppendLine("    setp.ge.s32 %p0, %r18, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r18, {Height};");
        ptx.AppendLine("    and.pred %p0, %p0, %p1;");                          // y0 valid
        ptx.AppendLine("    setp.ge.s32 %p2, %r20, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r20, {Height};");
        ptx.AppendLine("    and.pred %p2, %p2, %p3;");                          // y1 valid
        ptx.AppendLine("    setp.ge.s32 %p4, %r19, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p5, %r19, {Width};");
        ptx.AppendLine("    and.pred %p4, %p4, %p5;");                          // x0 valid
        ptx.AppendLine("    setp.ge.s32 %p6, %r21, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p7, %r21, {Width};");
        ptx.AppendLine("    and.pred %p6, %p6, %p7;");                          // x1 valid
        ptx.AppendLine("    and.pred %p8, %p0, %p4;");
        ptx.AppendLine("    and.pred %p9, %p0, %p6;");
        ptx.AppendLine("    and.pred %p10, %p2, %p4;");
        ptx.AppendLine("    and.pred %p11, %p2, %p6;");
        ptx.AppendLine($"    mul.lo.s32 %r22, %r18, {Width};");                // yb0
        ptx.AppendLine($"    mul.lo.s32 %r23, %r20, {Width};");                // yb1
        ptx.AppendLine("    add.s32 %r24, %r22, %r19;");
        ptx.AppendLine("    add.s32 %r25, %r22, %r21;");
        ptx.AppendLine("    add.s32 %r26, %r23, %r19;");
        ptx.AppendLine("    add.s32 %r27, %r23, %r21;");
        ptx.AppendLine("    shl.b32 %r24, %r24, 2;");
        ptx.AppendLine("    shl.b32 %r25, %r25, 2;");
        ptx.AppendLine("    shl.b32 %r26, %r26, 2;");
        ptx.AppendLine("    shl.b32 %r27, %r27, 2;");
        ptx.AppendLine("    cvt.s64.s32 %rd10, %r24;");
        ptx.AppendLine("    add.s64 %rd10, %rd9, %rd10;");
        ptx.AppendLine("    mov.f32 %f17, 0f00000000;");
        ptx.AppendLine("    @%p8 ld.global.nc.f32 %f17, [%rd10];");
        ptx.AppendLine("    cvt.s64.s32 %rd11, %r25;");
        ptx.AppendLine("    add.s64 %rd11, %rd9, %rd11;");
        ptx.AppendLine("    mov.f32 %f18, 0f00000000;");
        ptx.AppendLine("    @%p9 ld.global.nc.f32 %f18, [%rd11];");
        ptx.AppendLine("    cvt.s64.s32 %rd12, %r26;");
        ptx.AppendLine("    add.s64 %rd12, %rd9, %rd12;");
        ptx.AppendLine("    mov.f32 %f19, 0f00000000;");
        ptx.AppendLine("    @%p10 ld.global.nc.f32 %f19, [%rd12];");
        ptx.AppendLine("    cvt.s64.s32 %rd13, %r27;");
        ptx.AppendLine("    add.s64 %rd13, %rd9, %rd13;");
        ptx.AppendLine("    mov.f32 %f20, 0f00000000;");
        ptx.AppendLine("    @%p11 ld.global.nc.f32 %f20, [%rd13];");
        ptx.AppendLine("    mul.rn.f32 %f21, %f13, %f17;");
        ptx.AppendLine("    fma.rn.f32 %f21, %f14, %f18, %f21;");
        ptx.AppendLine("    fma.rn.f32 %f21, %f15, %f19, %f21;");
        ptx.AppendLine("    fma.rn.f32 %f21, %f16, %f20, %f21;");               // sample
        ptx.AppendLine("    mul.rn.f32 %f22, %f3, %f21;");                      // mask_s * sample
        ptx.AppendLine("    fma.rn.f32 %f0, %f4, %f22, %f0;");                  // acc += dOut_s * (mask*sample)
        ptx.AppendLine("    add.u64 %rd5, %rd5, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, 4;");
        ptx.AppendLine("    add.u32 %r13, %r13, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p12, %r13, {SpatialElements};");
        ptx.AppendLine("    @%p12 bra DEFORM_BWD_WEIGHT;");
        ptx.AppendLine("    mul.wide.u32 %rd14, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd4, %rd14;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
