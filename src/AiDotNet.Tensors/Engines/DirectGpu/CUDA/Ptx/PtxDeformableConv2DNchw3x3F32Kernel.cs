using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW deformable 3x3 convolution forward (v2, modulated; stride 1,
/// pad 1, dilation 1, single deformable group). Each of the nine taps samples the input
/// at a learned, per-position sub-pixel location via bilinear interpolation, scales it by
/// a learned modulation mask, and accumulates the shared convolution weight:
/// <c>Y[co,oy,ox] = sum_ci sum_ky sum_kx W[co,ci,ky,kx] * mask[ky*3+kx,oy,ox] * bilinear(X[ci], py, px)</c>
/// with <c>py = oy+ky-1 + offY[ky,kx,oy,ox]</c>, <c>px = ox+kx-1 + offX[ky,kx,oy,ox]</c>,
/// over the exact N1/Cin4/H8/W8/Cout4 geometry. One thread owns one output element. The
/// four bilinear corner weights and validity predicates are computed once per tap
/// (independent of the input channel); the channel loop then gathers the four corners
/// (zero-padded outside the frame) and reduces (thread-private, deterministic, no
/// atomics). Offsets are [2*kH*kW, outH, outW] and mask is [kH*kW, outH, outW]; all
/// tensors map to byte-exact extents.
/// </summary>
internal sealed class PtxDeformableConv2DNchw3x3F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_deformconv2d_n1_cin4_h8_w8_cout4_r3_s1_p1_v2";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 4;
    internal const int Height = 8;
    internal const int Width = 8;
    internal const int OutputChannels = 4;
    internal const int OutHeight = Height;   // stride 1, pad 1, dilation 1
    internal const int OutWidth = Width;
    internal const int KernelSize = 3;
    internal const int TapsPerChannel = KernelSize * KernelSize;         // 9
    internal const int SpatialElements = OutHeight * OutWidth;           // 64
    internal const int OutputElements = Batch * OutputChannels * SpatialElements; // 256
    internal const int OffsetChannels = 2 * TapsPerChannel;             // 18
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long OffsetBytes = (long)OffsetChannels * SpatialElements * sizeof(float);
    internal const long MaskBytes = (long)TapsPerChannel * SpatialElements * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDeformableConv2DNchw3x3F32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Deformable Conv2D 3x3 has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var mask = new DirectPtxExtent(TapsPerChannel, OutHeight, OutWidth);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "deformable-conv2d",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin4-h8-w8-cout4-r3-s1-p1-v2-fp32",
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
                // 0 = per-thread register ceiling derived from the device register
                // file at validation time; not pinned to a hardcoded literal.
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 2),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "Y[co,oy,ox] = sum_ci sum_ky sum_kx W[co,ci,ky,kx] * mask[ky*3+kx,oy,ox] * bilinear(X[ci], oy+ky-1+offY, ox+kx-1+offX)",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["sampling"] = "bilinear (floor + 4 corner, zero-pad outside frame)",
                ["modulation"] = "v2 mask multiply",
                ["deformable-groups"] = "1",
                ["layout"] = "nchw input/output; oihw weights; [2*kHkW,outH,outW] offsets; [kHkW,outH,outW] mask",
                ["padding"] = "zero-pad-1 base + learned offset bilinear",
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
                "Only the experimental SM86 deformable Conv2D 3x3 emitter exists.");

        const int inChannelStrideBytes = Height * Width * sizeof(float);          // 256 per ci
        const int weightOutStrideBytes = InputChannels * TapsPerChannel * sizeof(float);   // 144 per co
        const int weightChannelStrideBytes = TapsPerChannel * sizeof(float);      // 36 per ci
        const int planeStrideBytes = OutHeight * OutWidth * sizeof(float);        // 256 per offset/mask channel

        var ptx = new StringBuilder(32768);
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
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial = oy*W + ox
        ptx.AppendLine("    shr.u32 %r4, %r3, 3;");                            // oy
        ptx.AppendLine($"    and.b32 %r5, %r3, {Width - 1};");                 // ox
        ptx.AppendLine("    shr.u32 %r6, %r2, 6;");                            // co (64 = outH*outW)
        // Weight base for co: weights + co*144.
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r6, {weightOutStrideBytes};");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        // Offset/mask spatial base (channel 0): +spatial*4.
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd6;");                        // offsets + spatial*4
        ptx.AppendLine("    mul.wide.u32 %rd7, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd3, %rd7;");                        // mask + spatial*4
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                        // accumulator
        ptx.AppendLine("    mov.f32 %f7, 0f3F800000;");                        // constant 1.0
        for (int t = 0; t < TapsPerChannel; t++)
        {
            int ky = t / KernelSize;
            int kx = t % KernelSize;
            int constY = ky - 1;
            int constX = kx - 1;
            int offYByte = 2 * t * planeStrideBytes;
            int offXByte = (2 * t + 1) * planeStrideBytes;
            int maskByte = t * planeStrideBytes;
            // base_y = oy + constY, base_x = ox + constX.
            if (constY == 0) ptx.AppendLine("    mov.s32 %r10, %r4;");
            else ptx.AppendLine($"    add.s32 %r10, %r4, {constY};");
            if (constX == 0) ptx.AppendLine("    mov.s32 %r11, %r5;");
            else ptx.AppendLine($"    add.s32 %r11, %r5, {constX};");
            // offY, offX.
            ptx.AppendLine(offYByte == 0
                ? "    ld.global.nc.f32 %f1, [%rd6];"
                : $"    ld.global.nc.f32 %f1, [%rd6+{offYByte}];");
            ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd6+{offXByte}];");
            // py = (float)base_y + offY ; px = (float)base_x + offX.
            ptx.AppendLine("    cvt.rn.f32.s32 %f3, %r10;");
            ptx.AppendLine("    add.rn.f32 %f3, %f3, %f1;");                    // py
            ptx.AppendLine("    cvt.rn.f32.s32 %f4, %r11;");
            ptx.AppendLine("    add.rn.f32 %f4, %f4, %f2;");                    // px
            // Floor to integer corners.
            ptx.AppendLine("    cvt.rmi.s32.f32 %r12, %f3;");                   // y0 = floor(py)
            ptx.AppendLine("    cvt.rmi.s32.f32 %r13, %f4;");                   // x0 = floor(px)
            ptx.AppendLine("    add.s32 %r14, %r12, 1;");                       // y1
            ptx.AppendLine("    add.s32 %r15, %r13, 1;");                       // x1
            // Fractional weights.
            ptx.AppendLine("    cvt.rn.f32.s32 %f5, %r12;");                    // y0f
            ptx.AppendLine("    sub.rn.f32 %f6, %f3, %f5;");                    // wy1 = py - y0f
            ptx.AppendLine("    sub.rn.f32 %f8, %f7, %f6;");                    // wy0 = 1 - wy1
            ptx.AppendLine("    cvt.rn.f32.s32 %f9, %r13;");                    // x0f
            ptx.AppendLine("    sub.rn.f32 %f10, %f4, %f9;");                   // wx1 = px - x0f
            ptx.AppendLine("    sub.rn.f32 %f11, %f7, %f10;");                  // wx0 = 1 - wx1
            // Corner coefficients.
            ptx.AppendLine("    mul.rn.f32 %f12, %f8, %f11;");                  // c00 = wy0*wx0
            ptx.AppendLine("    mul.rn.f32 %f13, %f8, %f10;");                  // c01 = wy0*wx1
            ptx.AppendLine("    mul.rn.f32 %f14, %f6, %f11;");                  // c10 = wy1*wx0
            ptx.AppendLine("    mul.rn.f32 %f15, %f6, %f10;");                  // c11 = wy1*wx1
            // Bounds predicates: y0/y1 in [0,H), x0/x1 in [0,W).
            ptx.AppendLine("    setp.ge.s32 %p0, %r12, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p1, %r12, {Height};");
            ptx.AppendLine("    and.pred %p0, %p0, %p1;");                      // y0 valid
            ptx.AppendLine("    setp.ge.s32 %p2, %r14, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p3, %r14, {Height};");
            ptx.AppendLine("    and.pred %p2, %p2, %p3;");                      // y1 valid
            ptx.AppendLine("    setp.ge.s32 %p4, %r13, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p5, %r13, {Width};");
            ptx.AppendLine("    and.pred %p4, %p4, %p5;");                      // x0 valid
            ptx.AppendLine("    setp.ge.s32 %p6, %r15, 0;");
            ptx.AppendLine($"    setp.lt.s32 %p7, %r15, {Width};");
            ptx.AppendLine("    and.pred %p6, %p6, %p7;");                      // x1 valid
            ptx.AppendLine("    and.pred %p8, %p0, %p4;");                      // corner (y0,x0)
            ptx.AppendLine("    and.pred %p9, %p0, %p6;");                      // corner (y0,x1)
            ptx.AppendLine("    and.pred %p10, %p2, %p4;");                     // corner (y1,x0)
            ptx.AppendLine("    and.pred %p11, %p2, %p6;");                     // corner (y1,x1)
            // Corner byte offsets li = (y*W + x)*4.
            ptx.AppendLine($"    mul.lo.s32 %r16, %r12, {Width};");            // yb0 = y0*W
            ptx.AppendLine($"    mul.lo.s32 %r17, %r14, {Width};");            // yb1 = y1*W
            ptx.AppendLine("    add.s32 %r18, %r16, %r13;");                    // li00 = yb0 + x0
            ptx.AppendLine("    add.s32 %r19, %r16, %r15;");                    // li01 = yb0 + x1
            ptx.AppendLine("    add.s32 %r20, %r17, %r13;");                    // li10 = yb1 + x0
            ptx.AppendLine("    add.s32 %r21, %r17, %r15;");                    // li11 = yb1 + x1
            ptx.AppendLine("    shl.b32 %r18, %r18, 2;");
            ptx.AppendLine("    shl.b32 %r19, %r19, 2;");
            ptx.AppendLine("    shl.b32 %r20, %r20, 2;");
            ptx.AppendLine("    shl.b32 %r21, %r21, 2;");
            // mask[t].
            ptx.AppendLine(maskByte == 0
                ? "    ld.global.nc.f32 %f16, [%rd7];"
                : $"    ld.global.nc.f32 %f16, [%rd7+{maskByte}];");
            // Weight pointer (co, ci=0, t) and input pointer (ci=0).
            ptx.AppendLine(t == 0
                ? "    mov.u64 %rd8, %rd5;"
                : $"    add.u64 %rd8, %rd5, {t * (int)sizeof(float)};");
            ptx.AppendLine("    mov.u64 %rd9, %rd0;");
            ptx.AppendLine("    mov.u32 %r22, 0;");                             // ci counter
            string ciLabel = $"DEFORM_TAP{t}_CI";
            ptx.AppendLine($"{ciLabel}:");
            // Four bilinear corners (zero outside the frame).
            ptx.AppendLine("    cvt.s64.s32 %rd10, %r18;");
            ptx.AppendLine("    add.s64 %rd10, %rd9, %rd10;");
            ptx.AppendLine("    mov.f32 %f17, 0f00000000;");
            ptx.AppendLine("    @%p8 ld.global.nc.f32 %f17, [%rd10];");
            ptx.AppendLine("    cvt.s64.s32 %rd11, %r19;");
            ptx.AppendLine("    add.s64 %rd11, %rd9, %rd11;");
            ptx.AppendLine("    mov.f32 %f18, 0f00000000;");
            ptx.AppendLine("    @%p9 ld.global.nc.f32 %f18, [%rd11];");
            ptx.AppendLine("    cvt.s64.s32 %rd12, %r20;");
            ptx.AppendLine("    add.s64 %rd12, %rd9, %rd12;");
            ptx.AppendLine("    mov.f32 %f19, 0f00000000;");
            ptx.AppendLine("    @%p10 ld.global.nc.f32 %f19, [%rd12];");
            ptx.AppendLine("    cvt.s64.s32 %rd13, %r21;");
            ptx.AppendLine("    add.s64 %rd13, %rd9, %rd13;");
            ptx.AppendLine("    mov.f32 %f20, 0f00000000;");
            ptx.AppendLine("    @%p11 ld.global.nc.f32 %f20, [%rd13];");
            // sample = c00*v00 + c01*v01 + c10*v10 + c11*v11.
            ptx.AppendLine("    mul.rn.f32 %f21, %f12, %f17;");
            ptx.AppendLine("    fma.rn.f32 %f21, %f13, %f18, %f21;");
            ptx.AppendLine("    fma.rn.f32 %f21, %f14, %f19, %f21;");
            ptx.AppendLine("    fma.rn.f32 %f21, %f15, %f20, %f21;");
            // contribution = W[co,ci,t] * (mask[t] * sample); acc += contribution.
            ptx.AppendLine("    ld.global.nc.f32 %f22, [%rd8];");
            ptx.AppendLine("    mul.rn.f32 %f23, %f16, %f21;");
            ptx.AppendLine("    fma.rn.f32 %f0, %f22, %f23, %f0;");
            // Advance to next input channel.
            ptx.AppendLine($"    add.u64 %rd8, %rd8, {weightChannelStrideBytes};");
            ptx.AppendLine($"    add.u64 %rd9, %rd9, {inChannelStrideBytes};");
            ptx.AppendLine("    add.u32 %r22, %r22, 1;");
            ptx.AppendLine($"    setp.lt.u32 %p12, %r22, {InputChannels};");
            ptx.AppendLine($"    @%p12 bra {ciLabel};");
        }
        // Y[co,oy,ox] = output + id*4.
        ptx.AppendLine("    mul.wide.u32 %rd14, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd4, %rd14;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
