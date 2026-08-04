using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW 2D transposed convolution backward-weight gradient
/// (kernel 3, stride 1, zero pad 1) with IOHW weights:
/// <c>dW[ci,co,ky,kx] = sum_{oy,ox} dY[co,oy,ox] * X[ci, oy+1-ky, ox+1-kx]</c> over the
/// exact N1/Cin64/H16/W16/Cout64 geometry. One thread owns one weight and evaluates
/// its full H*W dot with the transpose (negated) read offset, halo-predicated per
/// position, so there is no cross-thread reduction: the result is deterministic.
/// Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_conv_transpose2d_bwd_weight_n1_cin64_h16_w16_cout64_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int OutputChannels = 64;
    internal const int KernelSize = 3;
    internal const int SpatialElements = Height * Width;
    internal const int WeightElements = InputChannels * OutputChannels * KernelSize * KernelSize;
    internal const long GradOutputBytes = (long)Batch * OutputChannels * SpatialElements * sizeof(float);
    internal const long InputBytes = (long)Batch * InputChannels * SpatialElements * sizeof(float);
    internal const long GradWeightBytes = (long)WeightElements * sizeof(float);

    internal static readonly DirectPtxConvolutionShape Shape = PtxConvTranspose2DNchw3x3ForwardF32Kernel.Shape;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxConvTranspose2DNchw3x3BackwardWeightF32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"ConvTranspose2D backward-weight has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var input = new DirectPtxExtent(Batch, InputChannels, Height, Width);
        var gradWeight = new DirectPtxExtent(InputChannels, OutputChannels, KernelSize, KernelSize);
        return new DirectPtxKernelBlueprint(
            Operation: "conv-transpose2d-bwd-weight",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin64-h16-w16-cout64-r3-s1-p1-fp32",
            Tensors:
            [
                new("grad_output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad_weight", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Iohw,
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
                ["equation"] = "dW[ci,co,ky,kx] = sum_{oy,ox} dY[co,oy,ox] * X[ci, oy+1-ky, ox+1-kx]",
                ["grad_output"] = "fp32",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-fma",
                ["grad_weight"] = "fp32",
                ["reduction"] = "thread-private-full-dot-deterministic",
                ["layout"] = "nchw grad-out/input, iohw grad-weight",
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
                "Only the experimental SM86 ConvTranspose2D backward-weight emitter exists.");

        const int channelStrideBytes = SpatialElements * sizeof(float);
        const int weightsPerBlock = KernelSize * KernelSize; // 9 per (ci,co)

        var ptx = new StringBuilder(8192);
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
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [gradout_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [gradweight_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // weight id (IOHW)
        // Decompose: tmp = id/9 (= ci*Cout + co), m = id - tmp*9, co = tmp&63, ci = tmp>>6.
        ptx.AppendLine($"    div.u32 %r3, %r2, {weightsPerBlock};");            // tmp
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {weightsPerBlock};");
        ptx.AppendLine("    sub.u32 %r4, %r2, %r4;");                           // m
        ptx.AppendLine("    and.b32 %r5, %r3, 63;");                            // co
        ptx.AppendLine("    shr.u32 %r6, %r3, 6;");                             // ci
        ptx.AppendLine($"    div.u32 %r7, %r4, {KernelSize};");                 // ky
        ptx.AppendLine($"    mul.lo.u32 %r8, %r7, {KernelSize};");
        ptx.AppendLine("    sub.u32 %r8, %r4, %r8;");                           // kx
        ptx.AppendLine("    sub.s32 %r9, %r7, 1;");
        ptx.AppendLine("    neg.s32 %r9, %r9;");                                // dyr = 1 - ky
        ptx.AppendLine("    sub.s32 %r10, %r8, 1;");
        ptx.AppendLine("    neg.s32 %r10, %r10;");                              // dxr = 1 - kx
        // dY row base for output channel co (advances +4 per position).
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r5, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        // X base for input channel ci + transpose tap offset: in_ptr + ci*stride + (dyr*W+dxr)*4.
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r6, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        ptx.AppendLine($"    mul.lo.s32 %r11, %r9, {Width};");                  // dyr*W
        ptx.AppendLine("    add.s32 %r11, %r11, %r10;");                        // dyr*W + dxr
        ptx.AppendLine("    mul.lo.s32 %r11, %r11, 4;");
        ptx.AppendLine("    cvt.s64.s32 %rd6, %r11;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, %rd6;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r11, 0;");                                 // spatial counter
        ptx.AppendLine("CONVT2D_BWD_WEIGHT:");
        ptx.AppendLine("    shr.u32 %r12, %r11, 4;");                           // oy
        ptx.AppendLine("    and.b32 %r13, %r11, 15;");                          // ox
        ptx.AppendLine("    add.s32 %r14, %r12, %r9;");                         // iy = oy + dyr
        ptx.AppendLine("    add.s32 %r15, %r13, %r10;");                        // ix = ox + dxr
        ptx.AppendLine("    setp.ge.s32 %p0, %r14, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r14, {Height};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r15, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r15, {Width};");
        ptx.AppendLine("    and.pred %p4, %p0, %p1;");
        ptx.AppendLine("    and.pred %p4, %p4, %p2;");
        ptx.AppendLine("    and.pred %p4, %p4, %p3;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd3];");                    // dY[co, s]
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p4 ld.global.nc.f32 %f1, [%rd4];");               // X[ci, iy, ix]
        ptx.AppendLine("    fma.rn.f32 %f0, %f2, %f1, %f0;");
        ptx.AppendLine("    add.u64 %rd3, %rd3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd4, 4;");
        ptx.AppendLine("    add.u32 %r11, %r11, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p5, %r11, {SpatialElements};");
        ptx.AppendLine("    @%p5 bra CONVT2D_BWD_WEIGHT;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
