using System;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Hand-emitted im2col that lowers an FP32 NCHW input to a row-major [K, N] FP16 patch
/// matrix for a downstream FP16 Tensor-Core GEMM (weights [Cout, K] times patches
/// [K, N]). For the exact N1/Cin4/H8/W8 3x3 stride-1 pad-1 geometry:
/// K = Cin*kH*kW = 36 (patch dimension) and N = outH*outW = 64 (spatial positions).
/// Output element (k, n) holds <c>X[ci, oy+ky-1, ox+kx-1]</c> (zero outside the frame)
/// converted to FP16, where k decomposes to (ci, ky, kx) and n to (oy, ox). One thread
/// owns one output element: it reads the single halo-predicated FP32 input value,
/// round-to-nearest converts it to FP16, and stores two bytes. Deterministic, no atomics.
/// The FP16 result is bit-identical to a host <c>(Half)</c> cast of the same FP32 patch.
/// </summary>
internal sealed class PtxIm2colKNFp16Nchw3x3Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_im2col_kn_fp16_n1_cin4_h8_w8_r3_s1_p1";
    internal const int BlockThreads = 128;
    internal const int Batch = 1;
    internal const int InputChannels = 4;
    internal const int Height = 8;
    internal const int Width = 8;
    internal const int OutHeight = Height;   // stride 1, pad 1
    internal const int OutWidth = Width;
    internal const int KernelSize = 3;
    internal const int PatchDim = InputChannels * KernelSize * KernelSize; // K = 36
    internal const int SpatialElements = OutHeight * OutWidth;             // N = 64
    internal const int OutputElements = PatchDim * SpatialElements;        // 2304
    internal const long InputBytes = (long)Batch * InputChannels * Height * Width * sizeof(float);
    internal const long OutputBytes = (long)OutputElements * 2;            // FP16

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxIm2colKNFp16Nchw3x3Kernel(DirectPtxRuntime runtime)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasExperimentalConvolution(
                runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Im2col-KN-FP16 has no experimental SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} specialization.");

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
        var output = new DirectPtxExtent(PatchDim, SpatialElements);
        return new DirectPtxKernelBlueprint(
            Operation: "im2col-kn-fp16",
            Version: 1,
            Architecture: architecture,
            Variant: "n1-cin4-h8-w8-r3-s1-p1-fp32-to-fp16",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Nchw,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float16, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 0,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 4),
            Semantics: new System.Collections.Generic.Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "out[k,n] = (half) X[ci, oy+ky-1, ox+kx-1]; k=(ci,ky,kx), n=(oy,ox)",
                ["input"] = "fp32",
                ["output"] = "fp16 (round-to-nearest-even, bit-identical to host (Half) cast)",
                ["layout"] = "nchw input; row-major [K=Cin*kHkW, N=outHW] fp16 output",
                ["padding"] = "zero-pad-1 halo-predicated",
                ["purpose"] = "fp16 tensor-core gemm preparation",
                ["intermediate-global-bytes"] = "0",
                ["shape-selection"] = "host-only-exact-contract",
                ["promotion"] = "experimental-pending-gpu-evidence"
            });
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
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
                "Only the experimental SM86 Im2col-KN-FP16 emitter exists.");

        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{computeCapabilityMajor}{computeCapabilityMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b16 %h<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<6>;");
        ptx.AppendLine("    .reg .f32 %f<2>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");       // output id = k*N + n
        ptx.AppendLine($"    shr.u32 %r3, %r2, 6;");                           // k (N = 64)
        ptx.AppendLine($"    and.b32 %r4, %r2, {SpatialElements - 1};");        // n
        // Decompose k = ci*9 + ky*3 + kx.
        ptx.AppendLine("    div.u32 %r5, %r3, 9;");                            // ci = k / 9
        ptx.AppendLine("    mul.lo.u32 %r6, %r5, 9;");
        ptx.AppendLine("    sub.u32 %r6, %r3, %r6;");                          // tap = k - ci*9
        ptx.AppendLine("    div.u32 %r7, %r6, 3;");                            // ky = tap / 3
        ptx.AppendLine("    mul.lo.u32 %r8, %r7, 3;");
        ptx.AppendLine("    sub.u32 %r8, %r6, %r8;");                          // kx = tap - ky*3
        // Decompose n = oy*W + ox.
        ptx.AppendLine("    shr.u32 %r9, %r4, 3;");                            // oy
        ptx.AppendLine($"    and.b32 %r10, %r4, {OutWidth - 1};");             // ox
        // iy = oy + ky - 1, ix = ox + kx - 1.
        ptx.AppendLine("    add.s32 %r11, %r9, %r7;");
        ptx.AppendLine("    sub.s32 %r11, %r11, 1;");                          // iy
        ptx.AppendLine("    add.s32 %r12, %r10, %r8;");
        ptx.AppendLine("    sub.s32 %r12, %r12, 1;");                          // ix
        // Halo predicate: iy in [0,H) and ix in [0,W).
        ptx.AppendLine("    setp.ge.s32 %p0, %r11, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p1, %r11, {Height};");
        ptx.AppendLine("    setp.ge.s32 %p2, %r12, 0;");
        ptx.AppendLine($"    setp.lt.s32 %p3, %r12, {Width};");
        ptx.AppendLine("    and.pred %p0, %p0, %p1;");
        ptx.AppendLine("    and.pred %p2, %p2, %p3;");
        ptx.AppendLine("    and.pred %p0, %p0, %p2;");
        // Input index = ci*(H*W) + iy*W + ix.
        ptx.AppendLine($"    mad.lo.s32 %r13, %r11, {Width}, %r12;");
        ptx.AppendLine($"    mad.lo.s32 %r13, %r5, {Height * Width}, %r13;");
        ptx.AppendLine("    mul.wide.s32 %rd3, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd3;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    @%p0 ld.global.nc.f32 %f0, [%rd3];");             // X[ci,iy,ix] or 0 in pad
        ptx.AppendLine("    cvt.rn.f16.f32 %h0, %f0;");                       // round-to-nearest FP16
        // Output byte = id*2 (row-major [K,N]).
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 2;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        ptx.AppendLine("    st.global.b16 [%rd4], %h0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    public void Dispose() => _module.Dispose();
}
