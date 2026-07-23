using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted FP32 NCHW unfold / im2col patch extraction (kernel 3, stride 1,
/// zero pad 1). Materializes each output spatial position's flattened 3x3xC patch
/// as a column of a <c>[C*9, H*W]</c> row-major matrix:
/// <c>unfold[c*9 + ky*3 + kx, oy*W + ox] = in[c, oy+ky-1, ox+kx-1]</c> (0 outside).
/// One thread owns one matrix element and performs a single halo-masked gather —
/// no arithmetic, so the copy is bit-exact. Exact geometry baked into PTX (two
/// pointers). Zero global intermediates, zero local bytes.
/// </summary>
internal sealed class PtxUnfoldIm2ColNchw3x3F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_unfold_im2col_n1_c64_h16_w16_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int Channels = 64;
    internal const int Height = 16;
    internal const int Width = 16;
    internal const int KernelSize = 3;
    internal const int TapsPerChannel = KernelSize * KernelSize;      // 9
    internal const int SpatialElements = Height * Width;              // 256 (columns)
    internal const int UnfoldRows = Channels * TapsPerChannel;        // 576
    internal const int OutputElements = UnfoldRows * SpatialElements; // 147456
    internal const long InputBytes = (long)Batch * Channels * SpatialElements * sizeof(float);
    internal const long UnfoldBytes = (long)OutputElements * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxUnfoldIm2ColNchw3x3F32Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Unfold/im2col has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var input = new DirectPtxExtent(Batch, Channels, Height, Width);
        var unfold = new DirectPtxExtent(UnfoldRows, SpatialElements);
        return new DirectPtxKernelBlueprint(
            Operation: "unfold-im2col",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-c64-h16-w16-r3-s1-p1-fp32",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("unfold", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    unfold, unfold, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
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
                ["equation"] = "unfold[c*9 + ky*3 + kx, oy*W + ox] = in[c, oy+ky-1, ox+kx-1] (0 outside)",
                ["input"] = "fp32",
                ["unfold"] = "fp32",
                ["operation"] = "halo-masked gather (bit-exact, no arithmetic)",
                ["layout"] = "nchw input, [C*9, H*W] row-major output",
                ["padding"] = "zero-pad-1-halo-predicated",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView unfold)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(unfold, Blueprint.Tensors[1], nameof(unfold));

        IntPtr inputPointer = input.Pointer;
        IntPtr unfoldPointer = unfold.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &unfoldPointer;
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
                "Only the experimental SM86 unfold/im2col emitter exists.");

        const int channelStrideBytes = SpatialElements * sizeof(float); // per channel of input

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 unfold_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<2>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [unfold_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // global element id
        ptx.AppendLine($"    shr.u32 %r3, %r2, 8;");                           // row (H*W = 256 columns)
        ptx.AppendLine($"    and.b32 %r4, %r2, {SpatialElements - 1};");        // col
        // row -> c, m; m -> ky, kx.
        ptx.AppendLine($"    div.u32 %r5, %r3, {TapsPerChannel};");            // c
        ptx.AppendLine($"    mul.lo.u32 %r6, %r5, {TapsPerChannel};");
        ptx.AppendLine("    sub.u32 %r6, %r3, %r6;");                          // m = ky*3 + kx
        ptx.AppendLine($"    div.u32 %r7, %r6, {KernelSize};");               // ky
        ptx.AppendLine($"    mul.lo.u32 %r8, %r7, {KernelSize};");
        ptx.AppendLine("    sub.u32 %r8, %r6, %r8;");                          // kx
        // col -> oy, ox.
        ptx.AppendLine("    shr.u32 %r9, %r4, 4;");                            // oy
        ptx.AppendLine("    and.b32 %r10, %r4, 15;");                          // ox
        // iy = oy + ky - 1; ix = ox + kx - 1.
        ptx.AppendLine("    add.s32 %r11, %r9, %r7;");
        ptx.AppendLine("    sub.s32 %r11, %r11, 1;");                          // iy
        ptx.AppendLine("    add.s32 %r12, %r10, %r8;");
        ptx.AppendLine("    sub.s32 %r12, %r12, 1;");                          // ix
        ptx.AppendLine("    setp.ge.s32 %p0, %r11, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r11, {Height};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r12, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r12, {Width};");
        ptx.AppendLine("    and.pred %p4, %p0, %p1;");
        ptx.AppendLine("    and.pred %p4, %p4, %p2;");
        ptx.AppendLine("    and.pred %p4, %p4, %p3;");
        // in addr = in_ptr + c*(H*W*4) + (iy*W + ix)*4.
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r5, {channelStrideBytes};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        ptx.AppendLine($"    mul.lo.s32 %r13, %r11, {Width};");
        ptx.AppendLine("    add.s32 %r13, %r13, %r12;");                       // iy*W + ix
        ptx.AppendLine("    mul.lo.s32 %r13, %r13, 4;");
        ptx.AppendLine("    cvt.s64.s32 %rd4, %r13;");
        ptx.AppendLine("    add.u64 %rd4, %rd3, %rd4;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    @%p4 ld.global.nc.f32 %f0, [%rd4];");
        // Write the matrix element.
        ptx.AppendLine("    mul.wide.u32 %rd9, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd9;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
