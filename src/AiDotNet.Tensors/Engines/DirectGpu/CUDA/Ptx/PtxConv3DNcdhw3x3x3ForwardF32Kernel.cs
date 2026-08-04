using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCDHW 3x3x3 3D convolution (stride 1, zero pad 1). For each
/// output voxel
/// <c>out[co,d,h,w] = sum_ci sum_{kd,kh,kw} W[co,ci,kd,kh,kw] * in[ci, d+kd-1, h+kh-1, w+kw-1]</c>
/// over the exact N1/Cin16/D8/H8/W8/Cout16 geometry, OIDHW weights. One thread owns
/// one output voxel and runs an input-channel loop with all 27 taps unrolled; the
/// six depth/height/width boundary predicates are computed once from (d,h,w) and
/// combined per tap (cube corners AND three). NCDHW/OIDHW logical shapes are rank-5,
/// so the ABI contract uses byte-exact collapsed rank-4 extents; the emitter is the
/// authority on the true strides. Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConv3DNcdhw3x3x3ForwardF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv3d_n1_cin16_d8_h8_w8_cout16_r3_s1_p1";
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
    internal const int OutputElements = Batch * OutputChannels * SpatialElements;
    internal const long InputBytes = (long)Batch * InputChannels * SpatialElements * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv3DNcdhw3x3x3ForwardF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv3D 3x3x3 has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        // NCDHW/OIDHW are rank-5; collapse to byte-exact rank-4 extents for the ABI
        // contract (spatial folded into one axis). Element counts (hence RequiredBytes)
        // are exact; the emitter owns the true NCDHW/OIDHW strides.
        var input = new DirectPtxExtent(Batch, InputChannels, Depth, Height * Width);
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, TapsPerChannel, 1);
        var output = new DirectPtxExtent(Batch, OutputChannels, Depth, Height * Width);
        return new DirectPtxKernelBlueprint(
            Operation: "conv3d",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin16-d8-h8-w8-cout16-r3-s1-p1-fp32",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "out[co,d,h,w] = sum_ci sum_{kd,kh,kw} W[co,ci,kd,kh,kw] * in[ci, d+kd-1, h+kh-1, w+kw-1]",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["layout"] = "ncdhw/oidhw (abi extents collapsed to rank-4 byte-exact)",
                ["padding"] = "zero-pad-1 halo-predicated on d/h/w",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &outputPointer;
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
                "Only the experimental SM86 Conv3D 3x3x3 emitter exists.");

        const int inChannelStrideBytes = SpatialElements * sizeof(float);                 // per ci
        const int weightChannelStrideBytes = TapsPerChannel * sizeof(float);              // per (co,ci)
        const int weightOutStrideBytes = InputChannels * TapsPerChannel * sizeof(float);  // per co
        const int hw = Height * Width;

        var ptx = new StringBuilder(32768);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<11>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global output id
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // s = d*H*W + h*W + w
        ptx.AppendLine("    shr.u32 %r7, %r2, 9;");                            // co (512 = D*H*W)
        ptx.AppendLine($"    shr.u32 %r4, %r3, 6;");                           // d = s / (H*W)  (64)
        ptx.AppendLine($"    and.b32 %r9, %r3, {hw - 1};");                    // rem = s % (H*W)
        ptx.AppendLine("    shr.u32 %r5, %r9, 3;");                            // h = rem / W  (8)
        ptx.AppendLine($"    and.b32 %r6, %r9, {Width - 1};");                 // w = rem % W
        // Six boundary predicates from (d,h,w).
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");                         // d >= 1
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {Depth - 1};");             // d <= D-2
        ptx.AppendLine("    setp.ge.s32 %p2, %r5, 1;");                         // h >= 1
        ptx.AppendLine($"    setp.lt.s32 %p3, %r5, {Height - 1};");            // h <= H-2
        ptx.AppendLine("    setp.ge.s32 %p4, %r6, 1;");                         // w >= 1
        ptx.AppendLine($"    setp.lt.s32 %p5, %r6, {Width - 1};");             // w <= W-2
        // Weight base for co: weights_ptr + co*(Cin*27*4); advances +108 per ci.
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r7, {weightOutStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        // Input center base: in_ptr + s*4 (channel 0); advances +2048 per ci.
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");                                  // input-channel counter
        ptx.AppendLine("CONV3D_CHANNELS:");
        for (int m = 0; m < TapsPerChannel; m++)
        {
            int kd = m / (KernelSize * KernelSize);
            int kh = (m / KernelSize) % KernelSize;
            int kw = m % KernelSize;
            int dd = kd - 1;
            int dh = kh - 1;
            int dw = kw - 1;
            int tapOffsetBytes = (dd * hw + dh * Width + dw) * sizeof(float);
            if (m == 0)
                ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");
            else
                ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{m * sizeof(float)}];");
            // Collect the boundary predicates required by this tap's non-zero axes.
            var guards = new List<string>(3);
            if (dd != 0) guards.Add(dd < 0 ? "%p0" : "%p1");
            if (dh != 0) guards.Add(dh < 0 ? "%p2" : "%p3");
            if (dw != 0) guards.Add(dw < 0 ? "%p4" : "%p5");
            if (guards.Count == 0)
            {
                ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");           // center tap (always valid)
            }
            else
            {
                string guard;
                if (guards.Count == 1)
                {
                    guard = guards[0];
                }
                else
                {
                    ptx.AppendLine($"    and.pred %p6, {guards[0]}, {guards[1]};");
                    if (guards.Count == 3)
                        ptx.AppendLine($"    and.pred %p6, %p6, {guards[2]};");
                    guard = "%p6";
                }
                if (tapOffsetBytes >= 0)
                    ptx.AppendLine($"    add.u64 %rd5, %rd4, {tapOffsetBytes};");
                else
                    ptx.AppendLine($"    sub.u64 %rd5, %rd4, {-tapOffsetBytes};");
                ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
                ptx.AppendLine($"    @{guard} ld.global.nc.f32 %f1, [%rd5];");
            }
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        ptx.AppendLine($"    add.u64 %rd3, %rd3, {weightChannelStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd4, %rd4, {inChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        // Loop-continue uses %p7, keeping the boundary predicates (%p0-%p5) intact.
        ptx.AppendLine($"    setp.lt.u32 %p7, %r8, {InputChannels};");
        ptx.AppendLine("    @%p7 bra CONV3D_CHANNELS;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
