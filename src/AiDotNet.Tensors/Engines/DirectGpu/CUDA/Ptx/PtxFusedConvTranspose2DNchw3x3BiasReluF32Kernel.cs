using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW 2D transposed convolution with fused bias and ReLU
/// (kernel 3, stride 1, zero pad 1), IOHW weights:
/// <c>Y[co,y,x] = relu(bias[co] + sum_ci sum_ky sum_kx W[ci,co,ky,kx] * X[ci, y+1-ky, x+1-kx])</c>
/// over the exact N1/Cin64/H16/W16/Cout64 geometry. The bias add and activation are
/// folded into the accumulator's final register, so no intermediate transposed
/// result is written to global memory. Structurally
/// <see cref="PtxConvTranspose2DNchw3x3ForwardF32Kernel"/> plus a four-instruction
/// epilogue.
/// </summary>
internal sealed class PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv_transpose2d_n1_cin64_h16_w16_cout64_r3_s1_p1_bias_relu";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int OutputChannels = 64;
    internal const int KernelSize = 3;
    internal const int SpatialElements = Height * Width;
    internal const int OutputElements = Batch * OutputChannels * SpatialElements;
    internal const long InputBytes = (long)Batch * InputChannels * SpatialElements * sizeof(float);
    internal const long WeightBytes = (long)InputChannels * OutputChannels * KernelSize * KernelSize * sizeof(float);
    internal const long BiasBytes = OutputChannels * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxConvTranspose2DNchw3x3ForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedConvTranspose2DNchw3x3BiasReluF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Fused ConvTranspose2D has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var weights = new DirectPtxExtent(InputChannels, OutputChannels, KernelSize, KernelSize);
        var bias = new DirectPtxExtent(OutputChannels);
        var output = new DirectPtxExtent(Batch, OutputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "conv-transpose2d-bias-relu",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin64-h16-w16-cout64-r3-s1-p1-fp32",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Iohw,
                    weights, weights, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
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
                ["equation"] = "relu(bias[co] + sum_ci sum_ky sum_kx W[ci,co,ky,kx] * X[ci, y+1-ky, x+1-kx])",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["epilogue"] = "bias-add + relu (fused, no intermediate)",
                ["output"] = "fp32",
                ["layout"] = "nchw input/output, iohw transposed weights",
                ["padding"] = "zero-pad-1-halo-predicated",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
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
                "Only the experimental SM86 fused ConvTranspose2D emitter exists.");

        const int inChannelStrideBytes = SpatialElements * sizeof(float);                   // per ci
        const int weightInStrideBytes = OutputChannels * KernelSize * KernelSize * sizeof(float); // per ci
        const int weightOutBaseStrideBytes = KernelSize * KernelSize * sizeof(float);        // per co (base)

        var ptx = new StringBuilder(16384);
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
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global output id
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial
        ptx.AppendLine("    shr.u32 %r6, %r2, 8;");                            // output channel co
        ptx.AppendLine("    shr.u32 %r4, %r3, 4;");                            // y
        ptx.AppendLine("    and.b32 %r5, %r3, 15;");                           // x
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {Height - 1};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r5, 1;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r5, {Width - 1};");
        // Weight base for output channel co (IOHW): weights_ptr + co*(9*4); advances +2304 per ci.
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r6, {weightOutBaseStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        // Input center base: input_ptr + spatial*4 (input channel 0); advances +1024 per ci.
        ptx.AppendLine("    mul.wide.u32 %rd5, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r7, 0;");                                  // input-channel counter
        ptx.AppendLine("CONVT2D_FUSED_CHANNELS:");
        for (int m = 0; m < KernelSize * KernelSize; m++)
        {
            int dyr = 1 - m / KernelSize;
            int dxr = 1 - m % KernelSize;
            int tapOffsetBytes = (dyr * Width + dxr) * sizeof(float);
            if (m == 0)
                ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd4];");
            else
                ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd4+{m * sizeof(float)}];");
            bool needY = dyr != 0;
            bool needX = dxr != 0;
            if (!needY && !needX)
            {
                ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");
            }
            else
            {
                string guard;
                string yGuard = dyr < 0 ? "%p0" : "%p1";
                string xGuard = dxr < 0 ? "%p2" : "%p3";
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
                    ptx.AppendLine($"    add.u64 %rd6, %rd5, {tapOffsetBytes};");
                else
                    ptx.AppendLine($"    sub.u64 %rd6, %rd5, {-tapOffsetBytes};");
                ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
                ptx.AppendLine($"    @{guard} ld.global.nc.f32 %f1, [%rd6];");
            }
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        }
        ptx.AppendLine($"    add.u64 %rd4, %rd4, {weightInStrideBytes};");
        ptx.AppendLine($"    add.u64 %rd5, %rd5, {inChannelStrideBytes};");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p5, %r7, {InputChannels};");
        ptx.AppendLine("    @%p5 bra CONVT2D_FUSED_CHANNELS;");
        // Fused epilogue: + bias[co], then ReLU. No intermediate written.
        ptx.AppendLine("    mul.wide.u32 %rd7, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd7];");                    // bias[co]
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("    max.f32 %f0, %f0, 0f00000000;");                    // ReLU
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd3, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
