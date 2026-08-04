using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW depthwise 3x3 backward-weight gradient (stride 1, zero
/// pad 1). For each channel the nine weight gradients are
/// <c>dW[c,ky,kx] = sum_{y,x} dOut[c,y,x] * in[c, y+(ky-1), x+(kx-1)]</c>. One warp
/// owns one channel (grid = C blocks of 32 threads); each lane sweeps its stripe
/// of the H*W spatial grid accumulating nine partial sums, then a single
/// shuffle-butterfly warp reduction collapses each tap and lane 0 writes the nine
/// results. Deterministic (fixed reduction order, no atomics), zero shared memory,
/// zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxDepthwiseConv2D3x3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_depthwise_conv2d_bwd_weight_n1_c64_h16_w16_r3_s1_p1";
    internal const int WarpSize = 32;
    internal const int BlockThreads = WarpSize;
    internal const int Batch = 1;
    internal const int Channels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int KernelSize = 3;
    internal const int SpatialElements = Height * Width;
    internal const int SpatialStepsPerLane = SpatialElements / WarpSize;
    internal const long GradOutputBytes = (long)Batch * Channels * SpatialElements * sizeof(float);
    internal const long InputBytes = (long)Batch * Channels * SpatialElements * sizeof(float);
    internal const long GradWeightBytes = (long)Channels * KernelSize * KernelSize * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = new(
        Batch, Channels, Height, Width, Channels, Height, Width,
        KernelSize, KernelSize, 1, 1, 1, 1, 1, 1);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxDepthwiseConv2D3x3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Depthwise backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, Channels, Height, Width);
        var input = new DirectPtxExtent(Batch, Channels, Height, Width);
        var gradWeight = new DirectPtxExtent(Channels, 1, KernelSize, KernelSize);
        return new DirectPtxKernelBlueprint(
            Operation: "depthwise-conv2d-bwd-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-c64-h16-w16-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_weight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    gradWeight, gradWeight, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "dW[c,ky,kx] = sum_{y,x} dOut[c,y,x] * in[c,y+(ky-1),x+(kx-1)]",
                ["grad_output"] = "fp32",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_weight"] = "fp32",
                ["reduction"] = "warp-shuffle-butterfly-deterministic",
                ["layout"] = "nchw/oihw-depthwise",
                ["padding"] = "zero-pad-1-halo-predicated",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView input,
        DirectPtxTensorView gradWeight)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(input, Blueprint.Tensors[1], nameof(input));
        Require(gradWeight, Blueprint.Tensors[2], nameof(gradWeight));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr gradWeightPointer = gradWeight.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &inputPointer;
        arguments[2] = &gradWeightPointer;
        // One warp (32 threads) per channel.
        _module.Launch(
            _function, Channels, 1, 1,
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
                "Only the experimental SM86 depthwise backward-weight emitter exists.");

        const int channelStrideBytes = SpatialElements * sizeof(float);
        const int weightStrideBytes = KernelSize * KernelSize * sizeof(float);

        var ptx = new StringBuilder(16384);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 gradout_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 gradweight_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<13>;");    // f0..f8 accumulators, f9 dOut, f10 in/shfl, f11 spare
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradweight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");                     // lane
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                   // channel c
        // Per-channel bases.
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r1, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");                // grad-output base for c
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r1, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");                // input base for c
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r1, {weightStrideBytes};");
        ptx.AppendLine("    add.u64 %rd5, %rd2, %rd5;");                // grad-weight base for c
        // Nine tap accumulators.
        for (int k = 0; k < KernelSize * KernelSize; k++)
            ptx.AppendLine($"    mov.f32 %f{k}, 0f00000000;");
        // Spatial sweep: lane starts at s = lane, strides by the warp width.
        ptx.AppendLine("    mov.u32 %r2, %r0;");                        // s = lane
        ptx.AppendLine("    mov.u32 %r5, 0;");                          // loop counter
        ptx.AppendLine("REDUCE_SPATIAL:");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");                     // y = s / W
        ptx.AppendLine("    and.b32 %r4, %r2, 15;");                    // x = s % W
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd3, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f9, [%rd7];");            // dOut[c,y,x]
        ptx.AppendLine("    add.u64 %rd8, %rd4, %rd6;");                // input center addr = inbase + s*4
        for (int k = 0; k < KernelSize * KernelSize; k++)
        {
            int dy = k / KernelSize - 1;
            int dx = k % KernelSize - 1;
            int tapOffsetBytes = (dy * Width + dx) * sizeof(float);
            ptx.AppendLine($"    // tap ky={dy + 1} kx={dx + 1}: in at (y+{dy}, x+{dx})");
            bool needY = dy != 0;
            bool needX = dx != 0;
            if (!needY && !needX)
            {
                ptx.AppendLine("    ld.global.nc.f32 %f10, [%rd8];");
            }
            else
            {
                string guard;
                if (needY)
                    ptx.AppendLine(dy < 0
                        ? "    setp.ge.s32 %p0, %r3, 1;"                // y >= 1
                        : $"    setp.lt.s32 %p0, %r3, {Height - 1};");  // y <= H-2
                if (needX)
                    ptx.AppendLine(dx < 0
                        ? "    setp.ge.s32 %p1, %r4, 1;"                // x >= 1
                        : $"    setp.lt.s32 %p1, %r4, {Width - 1};");   // x <= W-2
                if (needY && needX)
                {
                    ptx.AppendLine("    and.pred %p2, %p0, %p1;");
                    guard = "%p2";
                }
                else
                {
                    guard = needY ? "%p0" : "%p1";
                }
                if (tapOffsetBytes >= 0)
                    ptx.AppendLine($"    add.u64 %rd9, %rd8, {tapOffsetBytes};");
                else
                    ptx.AppendLine($"    sub.u64 %rd9, %rd8, {-tapOffsetBytes};");
                ptx.AppendLine("    mov.f32 %f10, 0f00000000;");
                ptx.AppendLine($"    @{guard} ld.global.nc.f32 %f10, [%rd9];");
            }
            ptx.AppendLine($"    fma.rn.f32 %f{k}, %f9, %f10, %f{k};");
        }
        ptx.AppendLine($"    add.u32 %r2, %r2, {WarpSize};");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p3, %r5, {SpatialStepsPerLane};");
        ptx.AppendLine("    @%p3 bra REDUCE_SPATIAL;");
        // Warp-butterfly reduction of each tap accumulator across the 32 lanes.
        for (int k = 0; k < KernelSize * KernelSize; k++)
            foreach (int delta in new[] { 16, 8, 4, 2, 1 })
            {
                ptx.AppendLine($"    mov.b32 %r8, %f{k};");
                ptx.AppendLine($"    shfl.sync.bfly.b32 %r9, %r8, {delta}, 31, 0xffffffff;");
                ptx.AppendLine("    mov.b32 %f10, %r9;");
                ptx.AppendLine($"    add.rn.f32 %f{k}, %f{k}, %f10;");
            }
        // Lane 0 writes the nine reduced weight gradients.
        ptx.AppendLine("    setp.eq.u32 %p4, %r0, 0;");
        for (int k = 0; k < KernelSize * KernelSize; k++)
        {
            if (k == 0)
                ptx.AppendLine($"    @%p4 st.global.f32 [%rd5], %f{k};");
            else
                ptx.AppendLine($"    @%p4 st.global.f32 [%rd5+{k * sizeof(float)}], %f{k};");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
