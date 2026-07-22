using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCDHW 3x3x3 3D convolution backward-input gradient (stride 1,
/// zero pad 1) — the transpose of <see cref="PtxConv3DNcdhw3x3x3ForwardF32Kernel"/>.
/// <c>dIn[ci,d,h,w] = sum_co sum_{kd,kh,kw} W[co,ci,kd,kh,kw] * dOut[co, d-(kd-1), h-(kh-1), w-(kw-1)]</c>
/// over the exact N1/Cin16/D8/H8/W8/Cout16 geometry, OIDHW weights. One thread owns
/// one input-gradient voxel and runs an output-channel loop with all 27 taps
/// unrolled at the negated read offset; the six d/h/w boundary predicates are
/// computed once and combined per tap. Byte-exact collapsed rank-4 ABI extents.
/// Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConv3DNcdhw3x3x3BackwardInputF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv3d_bwd_input_n1_cin16_d8_h8_w8_cout16_r3_s1_p1";
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
    internal const int InputElements = Batch * InputChannels * SpatialElements;
    internal const long GradOutputBytes = (long)Batch * OutputChannels * SpatialElements * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * TapsPerChannel * sizeof(float);
    internal const long GradInputBytes = (long)InputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv3DNcdhw3x3x3BackwardInputF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv3D backward-input has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, TapsPerChannel, 1);
        var gradInput = new DirectPtxExtent(Batch, InputChannels, Depth, Height * Width);
        return new DirectPtxKernelBlueprint(
            Operation: "conv3d-bwd-input",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin16-d8-h8-w8-cout16-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradInput, gradInput, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "dIn[ci,d,h,w] = sum_co sum_{kd,kh,kw} W[co,ci,kd,kh,kw] * dOut[co, d-(kd-1), h-(kh-1), w-(kw-1)]",
                ["grad_output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_input"] = "fp32",
                ["layout"] = "ncdhw/oidhw (abi extents collapsed to rank-4 byte-exact), transposed offset",
                ["padding"] = "zero-pad-1 halo-predicated on d/h/w",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView weights,
        DirectPtxTensorView gradInput)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(gradInput, Blueprint.Tensors[2], nameof(gradInput));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr gradInputPointer = gradInput.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &gradInputPointer;
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
                "Only the experimental SM86 Conv3D backward-input emitter exists.");

        const int gradOutChannelStrideBytes = SpatialElements * sizeof(float);            // per co
        const int weightInBaseStrideBytes = TapsPerChannel * sizeof(float);               // per ci (base)
        const int weightOutStrideBytes = InputChannels * TapsPerChannel * sizeof(float);  // per co
        const int hw = Height * Width;

        var ptx = new StringBuilder(32768);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 gradout_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 gradin_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<11>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradin_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global input-grad id
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // s = d*H*W + h*W + w
        ptx.AppendLine("    shr.u32 %r7, %r2, 9;");                            // ci (512 = D*H*W)
        ptx.AppendLine("    shr.u32 %r4, %r3, 6;");                            // d
        ptx.AppendLine($"    and.b32 %r9, %r3, {hw - 1};");                    // rem
        ptx.AppendLine("    shr.u32 %r5, %r9, 3;");                            // h
        ptx.AppendLine($"    and.b32 %r6, %r9, {Width - 1};");                 // w
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");                         // d >= 1
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {Depth - 1};");             // d <= D-2
        ptx.AppendLine("    setp.ge.s32 %p2, %r5, 1;");                         // h >= 1
        ptx.AppendLine($"    setp.lt.s32 %p3, %r5, {Height - 1};");            // h <= H-2
        ptx.AppendLine("    setp.ge.s32 %p4, %r6, 1;");                         // w >= 1
        ptx.AppendLine($"    setp.lt.s32 %p5, %r6, {Width - 1};");             // w <= W-2
        // Weight base for input channel ci (OIDHW): weights_ptr + ci*(27*4); advances +Cin*27*4 per co.
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r7, {weightInBaseStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        // grad-output center base: gradout_ptr + s*4 (output channel 0); advances +2048 per co.
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");                                  // output-channel counter
        ptx.AppendLine("CONV3D_BWD_INPUT:");
        for (int m = 0; m < TapsPerChannel; m++)
        {
            int kd = m / (KernelSize * KernelSize);
            int kh = (m / KernelSize) % KernelSize;
            int kw = m % KernelSize;
            // Backward-input reads dOut at (d-(kd-1), ...) = (d + (1-kd), ...); negated offset.
            int ddr = 1 - kd;
            int dhr = 1 - kh;
            int dwr = 1 - kw;
            int tapOffsetBytes = (ddr * hw + dhr * Width + dwr) * sizeof(float);
            if (m == 0)
                ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");
            else
                ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{m * sizeof(float)}];");
            var guards = new List<string>(3);
            if (ddr != 0) guards.Add(ddr < 0 ? "%p0" : "%p1");
            if (dhr != 0) guards.Add(dhr < 0 ? "%p2" : "%p3");
            if (dwr != 0) guards.Add(dwr < 0 ? "%p4" : "%p5");
            if (guards.Count == 0)
            {
                ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
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
        ptx.AppendLine($"    add.u64 %rd3, %rd3, {weightOutStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd4, %rd4, {gradOutChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p7, %r8, {OutputChannels};");
        ptx.AppendLine("    @%p7 bra CONV3D_BWD_INPUT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
