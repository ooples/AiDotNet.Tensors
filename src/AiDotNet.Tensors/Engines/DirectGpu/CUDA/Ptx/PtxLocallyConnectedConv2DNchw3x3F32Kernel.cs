using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW locally-connected 3x3 convolution (stride 1, zero pad 1).
/// Unlike a shared convolution, every output spatial position owns its own filter
/// bank (no weight sharing):
/// <c>out[co,oy,ox] = sum_ci sum_ky sum_kx W[oy,ox,co,ci,ky,kx] * in[ci, oy+ky-1, ox+kx-1]</c>
/// over the exact N1/Cin8/H8/W8/Cout8 geometry. Weights are laid out per-position as
/// [outH*outW, Cout, Cin, kH, kW]; one thread owns one output element and derives its
/// per-position weight base from (spatial, co) before running the input-channel loop
/// with all nine taps unrolled and halo-predicated. The weight tensor is rank-6, so the
/// ABI contract uses byte-exact collapsed rank-4 extents; the emitter owns the true
/// strides. Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxLocallyConnectedConv2DNchw3x3F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_localconn2d_n1_cin8_h8_w8_cout8_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 8;
    internal const int Height = 8;
    internal const int Width = 8;
    internal const int OutputChannels = 8;
    internal const int OutHeight = Height;   // stride 1, pad 1
    internal const int OutWidth = Width;
    internal const int KernelSize = 3;
    internal const int TapsPerChannel = KernelSize * KernelSize;         // 9
    internal const int SpatialElements = OutHeight * OutWidth;           // 64
    internal const int OutputElements = Batch * OutputChannels * SpatialElements; // 512
    internal const int WeightElements =
        SpatialElements * OutputChannels * InputChannels * TapsPerChannel; // 36864
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long WeightBytes = (long)WeightElements * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxLocallyConnectedConv2DNchw3x3F32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Locally-connected Conv2D 3x3 has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        // The locally-connected weight tensor is rank-6 [outH*outW, Cout, Cin, kH, kW];
        // collapse to byte-exact rank-4 extents for the ABI contract (element count,
        // hence RequiredBytes, is exact). The emitter owns the true per-position strides.
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var weights = new DirectPtxExtent(SpatialElements, OutputChannels, InputChannels, TapsPerChannel);
        var output = new DirectPtxExtent(Batch, OutputChannels, OutHeight, OutWidth);
        return new DirectPtxKernelBlueprint(
            Operation: "locally-connected-conv2d",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin8-h8-w8-cout8-r3-s1-p1-fp32",
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
                ["equation"] = "out[co,oy,ox] = sum_ci sum_ky sum_kx W[oy,ox,co,ci,ky,kx] * in[ci, oy+ky-1, ox+kx-1]",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["weight-sharing"] = "none (per output position)",
                ["layout"] = "nchw input/output; per-position [outHW,Cout,Cin,kH,kW] weights (abi extents collapsed to rank-4 byte-exact)",
                ["padding"] = "zero-pad-1 halo-predicated on oy/ox",
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
                "Only the experimental SM86 locally-connected Conv2D 3x3 emitter exists.");

        const int inChannelStrideBytes = Height * Width * sizeof(float);          // per ci  (256)
        const int weightChannelStrideBytes = TapsPerChannel * sizeof(float);      // per (pos,co,ci) block  (36)
        const int weightPositionStrideBytes =                                     // per (pos,co)  (288)
            InputChannels * TapsPerChannel * sizeof(float);

        var ptx = new StringBuilder(16384);
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
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global output id = co*HW + spatial
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial = oy*W + ox
        ptx.AppendLine("    shr.u32 %r6, %r2, 6;");                            // co (64 = outH*outW)
        ptx.AppendLine("    shr.u32 %r4, %r3, 3;");                            // oy = spatial / W  (8)
        ptx.AppendLine($"    and.b32 %r5, %r3, {Width - 1};");                 // ox = spatial % W
        // Four boundary predicates from (oy, ox).
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");                         // oy >= 1
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {Height - 1};");            // oy <= H-2
        ptx.AppendLine("    setp.ge.s32 %p2, %r5, 1;");                         // ox >= 1
        ptx.AppendLine($"    setp.lt.s32 %p3, %r5, {Width - 1};");             // ox <= W-2
        // Per-position weight base: weights_ptr + (spatial*Cout + co)*Cin*9*4; advances +36 per ci.
        ptx.AppendLine($"    mad.lo.u32 %r7, %r3, {OutputChannels}, %r6;");     // widx = spatial*Cout + co
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r7, {weightPositionStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        // Input center base: in_ptr + spatial*4 (channel 0); advances +256 per ci.
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r8, 0;");                                  // input-channel counter
        ptx.AppendLine("LOCALCONN_CHANNELS:");
        for (int m = 0; m < TapsPerChannel; m++)
        {
            int dy = m / KernelSize - 1;
            int dx = m % KernelSize - 1;
            int tapOffsetBytes = (dy * Width + dx) * sizeof(float);
            if (m == 0)
                ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");
            else
                ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{m * sizeof(float)}];");
            bool needY = dy != 0;
            bool needX = dx != 0;
            if (!needY && !needX)
            {
                ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");           // center tap (always valid)
            }
            else
            {
                string guard;
                string yGuard = dy < 0 ? "%p0" : "%p1";
                string xGuard = dx < 0 ? "%p2" : "%p3";
                if (needY && needX)
                {
                    ptx.AppendLine($"    and.pred %p4, {yGuard}, {xGuard};");
                    guard = "%p4";
                }
                else
                {
                    guard = needY ? yGuard : xGuard;
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
        // Loop-continue uses %p5, keeping the boundary predicates (%p0-%p3) intact.
        ptx.AppendLine($"    setp.lt.u32 %p5, %r8, {InputChannels};");
        ptx.AppendLine("    @%p5 bra LOCALCONN_CHANNELS;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
