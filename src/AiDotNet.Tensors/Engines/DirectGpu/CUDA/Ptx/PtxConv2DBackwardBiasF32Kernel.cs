using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW convolution bias gradient: dBias[k] = sum over all
/// spatial positions of dOut[k,y,x], for the exact N1/K64/H16/W16 output geometry
/// of the golden-slice fused conv. One warp owns one output channel (grid = K
/// blocks of 32 threads); each lane sums its stride-32 slice of the H*W grid into
/// one accumulator, a shuffle-butterfly warp reduction collapses the lanes, and
/// lane 0 writes the channel's bias gradient. Deterministic (fixed warp-reduction
/// order, no atomics), zero shared memory, zero global intermediates, zero local
/// bytes. Complements <see cref="PtxFusedConv2DNchwK1Kernel"/>.
/// </summary>
internal sealed class PtxConv2DBackwardBiasF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv2d_bwd_bias_n1_k64_h16_w16";
    internal const int WarpSize = 32;
    internal const int BlockThreads = WarpSize;
    internal const int Batch = 1;
    internal const int OutputChannels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int SpatialElements = Height * Width;
    internal const int SpatialStepsPerLane = SpatialElements / WarpSize;
    internal const long GradOutputBytes = (long)Batch * OutputChannels * SpatialElements * sizeof(float);
    internal const long GradBiasBytes = (long)OutputChannels * sizeof(float);

    // The bias-gradient contract is defined by the fused conv's output geometry.
    internal static readonly DirectPtxConvolutionShape Shape = PtxFusedConv2DNchwK1Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv2DBackwardBiasF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv2D backward-bias has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var gradOutput = new DirectPtxExtent(Batch, OutputChannels, Height, Width);
        var gradBias = new DirectPtxExtent(OutputChannels);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-bwd-bias",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-k64-h16-w16-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    gradBias, gradBias, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "dBias[k] = sum_{y,x} dOut[k,y,x]",
                ["grad_output"] = "fp32",
                ["accumulator"] = "fp32",
                ["grad_bias"] = "fp32",
                ["reduction"] = "warp-shuffle-butterfly-deterministic",
                ["layout"] = "nchw-to-vector",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView gradBias)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(gradBias, Blueprint.Tensors[1], nameof(gradBias));

        IntPtr gradOutputPointer = gradOutput.Pointer;
        IntPtr gradBiasPointer = gradBias.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &gradOutputPointer;
        arguments[1] = &gradBiasPointer;
        // One warp (32 threads) per output channel.
        _module.Launch(
            _function, OutputChannels, 1, 1,
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
                "Only the experimental SM86 conv backward-bias emitter exists.");

        const int channelStrideBytes = SpatialElements * sizeof(float);

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 gradout_ptr,");
        ptx.AppendLine("    .param .u64 gradbias_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [gradbias_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");                     // lane
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                   // output channel k
        // Grad-output base for channel k, offset to this lane's first element.
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r1, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd2, %rd0, %rd2;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd2, %rd3;");                // &dOut[k, lane]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        // Sum this lane's stride-32 slice of the H*W grid (unrolled).
        for (int i = 0; i < SpatialStepsPerLane; i++)
        {
            int offsetBytes = i * WarpSize * sizeof(float);
            if (offsetBytes == 0)
                ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
            else
                ptx.AppendLine($"    ld.global.nc.f32 %f1, [%rd4+{offsetBytes}];");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        }
        // Warp-butterfly reduction over the 32 lanes.
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine("    mov.b32 %r8, %f0;");
            ptx.AppendLine($"    shfl.sync.bfly.b32 %r9, %r8, {delta}, 31, 0xffffffff;");
            ptx.AppendLine("    mov.b32 %f1, %r9;");
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        }
        // Lane 0 writes this channel's bias gradient.
        ptx.AppendLine("    setp.eq.u32 %p0, %r0, 0;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd5;");
        ptx.AppendLine("    @%p0 st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
