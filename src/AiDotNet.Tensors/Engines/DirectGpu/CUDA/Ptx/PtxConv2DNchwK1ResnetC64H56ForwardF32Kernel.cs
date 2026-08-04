using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW 1x1 convolution with bias and ReLU at the ResNet-class
/// geometry N32/C64/H56/W56/K64 (stride 1, pad 0). This is the reproducible v1
/// starting point for the #841 promotion track at a realistic, compute-bound
/// shape: a per-output-channel dot over the 64 input channels, thread-per-output.
/// It is NOT expected to beat cuBLASLt/implicit-GEMM as written; it is the
/// correct, contract-clean baseline that the shared-memory weight-staged tiled
/// GEMM optimization is measured against (see the ResNet-class evidence campaign
/// in docs/research/2026-07-22-direct-ptx-convolution-blueprint.md). Because the
/// spatial extent 56x56 is not a power of two, the output-id decompose uses
/// explicit divides rather than shifts/masks.
/// </summary>
internal sealed class PtxConv2DNchwK1ResnetC64H56ForwardF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv2d_n32_c64_h56_w56_k64_k1_bias_relu";
    internal const int BlockThreads = 128;
    internal const int Batch = 32;
    internal const int InputChannels = 64;
    internal const int Height = 56;
    internal const int Width = 56;
    internal const int OutputChannels = 64;
    internal const int SpatialElements = Height * Width;              // 3136
    internal const int ChannelSpan = OutputChannels * SpatialElements; // K*H*W = C*H*W = 200704
    internal const int OutputElements = Batch * OutputChannels * SpatialElements; // 6,422,528
    internal const long InputBytes = (long)Batch * InputChannels * SpatialElements * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * sizeof(float);
    internal const long BiasBytes = OutputChannels * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv2DNchwK1ResnetC64H56ForwardF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"ResNet 1x1 convolution has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, 1, 1);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-1x1-bias-relu",
            Version: 1,
            Architecture: architecture,
            Variant: "n32-c64-h56-w56-k64-k1-s1-p0-fp32-resnet",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Oihw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[n,k,s] = relu(bias[k] + sum_c in[n,c,s] * W[k,c])",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["output"] = "fp32",
                ["layout"] = "nchw/oihw 1x1",
                ["epilogue"] = "bias-add + relu (fused, no intermediate)",
                ["regime"] = "resnet-class compute-bound (evidence track)",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-baseline-pending-tiled-gemm"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView weights,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(weights, Blueprint.Tensors[1], nameof(weights));
        Require(bias, Blueprint.Tensors[2], nameof(bias));
        Require(output, Blueprint.Tensors[3], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr weightPointer = weights.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &weightPointer;
        arguments[2] = &biasPointer;
        arguments[3] = &outputPointer;
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
                "Only the experimental SM86 ResNet 1x1 convolution emitter exists.");

        const int inChannelStrideBytes = SpatialElements * sizeof(float);   // per c (12544)
        const int weightChannelStrideBytes = sizeof(float);                 // per c (4)

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 weights_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // output id = n*KHW + k*HW + s
        // Decompose (56x56 is not a power of two -> explicit divides).
        ptx.AppendLine($"    div.u32 %r3, %r2, {ChannelSpan};");               // n
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {ChannelSpan};");
        ptx.AppendLine("    sub.u32 %r5, %r2, %r4;");                          // rem = k*HW + s
        ptx.AppendLine($"    div.u32 %r6, %r5, {SpatialElements};");           // k
        ptx.AppendLine($"    mul.lo.u32 %r7, %r6, {SpatialElements};");
        ptx.AppendLine("    sub.u32 %r8, %r5, %r7;");                          // s
        // input base for (n,s), channel 0: input + (n*ChannelSpan + s)*4; +12544 per c.
        ptx.AppendLine($"    mad.lo.u32 %r9, %r3, {ChannelSpan}, %r8;");        // n*ChannelSpan + s
        ptx.AppendLine("    mul.wide.u32 %rd4, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");                        // xci
        // weight base for k, channel 0: weights + k*C*4; +4 per c.
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r6, {InputChannels * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd5;");                        // wc
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r10, 0;");                                // channel counter
        ptx.AppendLine("CONV1X1_RESNET_CHANNELS:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");                   // in[n,c,s]
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd5];");                   // W[k,c]
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine($"    add.u64 %rd4, %rd4, {inChannelStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd5, %rd5, {weightChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r10, %r10, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r10, {InputChannels};");
        ptx.AppendLine("    @%p0 bra CONV1X1_RESNET_CHANNELS;");
        // bias[k] + relu.
        ptx.AppendLine("    mul.wide.u32 %rd6, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
