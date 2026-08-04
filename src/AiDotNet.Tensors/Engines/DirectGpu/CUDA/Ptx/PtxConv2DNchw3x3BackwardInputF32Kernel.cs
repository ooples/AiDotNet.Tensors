using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW general multi-channel 3x3 convolution backward-input
/// gradient (stride 1, zero pad 1) — the transpose of
/// <see cref="PtxConv2DNchw3x3ForwardF32Kernel"/>. For each input position
/// <c>dIn[ci,y,x] = sum_co sum_ky sum_kx W[co,ci,ky,kx] * dOut[co, y-(ky-1), x-(kx-1)]</c>.
/// One thread owns one input element and runs an output-channel loop with the nine
/// taps unrolled (read offset negated relative to the forward); the four boundary
/// predicates are computed once from (y,x) and combined per tap. Zero global
/// intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConv2DNchw3x3BackwardInputF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv2d_bwd_input_n1_cin64_h16_w16_cout64_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int OutputChannels = 64;
    internal const int KernelSize = 3;
    internal const int SpatialElements = Height * Width;
    internal const int InputElements = Batch * InputChannels * SpatialElements;
    internal const long GradOutputBytes = (long)Batch * OutputChannels * SpatialElements * sizeof(float);
    internal const long WeightBytes = (long)OutputChannels * InputChannels * KernelSize * KernelSize * sizeof(float);
    internal const long GradInputBytes = (long)InputElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxConv2DNchw3x3ForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConv2DNchw3x3BackwardInputF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Conv2D 3x3 backward-input has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var weights = new DirectPtxExtent(OutputChannels, InputChannels, KernelSize, KernelSize);
        var gradInput = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        return new DirectPtxKernelBlueprint(
            Operation: "conv2d-bwd-input",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin64-h16-w16-cout64-r3-s1-p1-fp32",
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
                ["equation"] = "dIn[ci,y,x] = sum_co sum_ky sum_kx W[co,ci,ky,kx] * dOut[co, y-(ky-1), x-(kx-1)]",
                ["grad_output"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_input"] = "fp32",
                ["layout"] = "nchw/oihw-transposed",
                ["padding"] = "zero-pad-1-halo-predicated",
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
                "Only the experimental SM86 Conv2D 3x3 backward-input emitter exists.");

        const int gradOutChannelStrideBytes = SpatialElements * sizeof(float);             // per co
        const int weightOutStrideBytes = InputChannels * KernelSize * KernelSize * sizeof(float); // per co
        const int weightInBaseStrideBytes = KernelSize * KernelSize * sizeof(float);        // per ci (base)

        var ptx = new StringBuilder(16384);
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
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<11>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [weights_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradin_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global input element id
        ptx.AppendLine($"    and.b32 %r3, %r2, {SpatialElements - 1};");        // spatial = y*W + x
        ptx.AppendLine("    shr.u32 %r6, %r2, 8;");                            // input channel ci (256 = H*W)
        ptx.AppendLine("    shr.u32 %r4, %r3, 4;");                            // y
        ptx.AppendLine("    and.b32 %r5, %r3, 15;");                           // x
        // Four boundary predicates from (y,x); combined per tap.
        ptx.AppendLine("    setp.ge.s32 %p0, %r4, 1;");                         // y >= 1   (read dy=-1 valid)
        ptx.AppendLine($"    setp.lt.s32 %p1, %r4, {Height - 1};");            // y <= H-2 (read dy=+1 valid)
        ptx.AppendLine("    setp.ge.s32 %p2, %r5, 1;");                         // x >= 1
        ptx.AppendLine($"    setp.lt.s32 %p3, %r5, {Width - 1};");             // x <= W-2
        // Weight base for input channel ci at output channel 0: weights_ptr + ci*(9*4).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r6, {weightInBaseStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd3;");
        // grad-output center base: gradout_ptr + spatial*4 (output channel 0); advances +1024 per co.
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd4;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r7, 0;");                                  // output-channel counter
        ptx.AppendLine("CONV2D_BWD_INPUT:");
        for (int m = 0; m < KernelSize * KernelSize; m++)
        {
            // Backward-input reads dOut at (y-(ky-1), x-(kx-1)); the read offset is
            // the negated forward tap offset. dyr = 1-ky, dxr = 1-kx.
            int dyr = 1 - m / KernelSize;
            int dxr = 1 - m % KernelSize;
            int tapOffsetBytes = (dyr * Width + dxr) * sizeof(float);
            if (m == 0)
                ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");
            else
                ptx.AppendLine($"    ld.global.nc.f32 %f2, [%rd3+{m * sizeof(float)}];");
            bool needY = dyr != 0;
            bool needX = dxr != 0;
            if (!needY && !needX)
            {
                ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
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
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        // Loop-continue uses %p5, keeping the boundary predicates (%p0-%p3) intact.
        ptx.AppendLine($"    setp.lt.u32 %p5, %r7, {OutputChannels};");
        ptx.AppendLine("    @%p5 bra CONV2D_BWD_INPUT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
