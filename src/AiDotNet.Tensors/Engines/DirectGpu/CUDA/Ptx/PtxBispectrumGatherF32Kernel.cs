using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Bispectrum gather for issue #850, matching the NVRTC <c>bispectrum_gather</c> kernel:
/// <c>B(f1,f2) = X(f1) * X(f2) * conj(X(f1+f2))</c> - a third-order spectral cumulant used for
/// quadratic-phase-coupling analysis. One thread owns one output bin <c>(f1,f2)</c>, reads the three
/// spectrum samples from the split buffers (the <c>f1+f2</c> sample conjugated), and forms the triple
/// complex product. Every product uses the multiply-then-fma contraction (nvcc's default fused evaluation),
/// so the spec is a tolerance-based fp64-oracle comparison. <c>maxF1</c>, <c>maxF2</c>, and the spectrum
/// length are baked; the launch rounds up and a single guard drops the tail lanes. Four pointers reach the
/// launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxBispectrumGatherF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_bispectrum_gather_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int MaxF1 { get; }
    internal int MaxF2 { get; }
    internal int SpecLength { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxBispectrumGatherF32Kernel(
        DirectPtxRuntime runtime, int maxF1, int maxF2, int specLength, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in bispectrum-gather specialization is admitted only on SM86.");
        Validate(maxF1, maxF2, specLength);
        ValidateBlockThreads(blockThreads);
        MaxF1 = maxF1;
        MaxF2 = maxF2;
        SpecLength = specLength;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, maxF1, maxF2, specLength, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, maxF1, maxF2, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView specReal, DirectPtxTensorView specImag,
        DirectPtxTensorView outReal, DirectPtxTensorView outImag)
    {
        Require(specReal, Blueprint.Tensors[0], nameof(specReal));
        Require(specImag, Blueprint.Tensors[1], nameof(specImag));
        Require(outReal, Blueprint.Tensors[2], nameof(outReal));
        Require(outImag, Blueprint.Tensors[3], nameof(outImag));

        IntPtr specRealPointer = specReal.Pointer, specImagPointer = specImag.Pointer;
        IntPtr outRealPointer = outReal.Pointer, outImagPointer = outImag.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &specRealPointer;
        arguments[1] = &specImagPointer;
        arguments[2] = &outRealPointer;
        arguments[3] = &outImagPointer;
        int total = MaxF1 * MaxF2;
        _module.Launch(
            _function,
            (uint)((total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int maxF1, int maxF2, int blockThreads = DefaultBlockThreads)
    {
        ValidateShape(maxF1, maxF2);
        ValidateBlockThreads(blockThreads);
        int total = checked(maxF1 * maxF2);

        var ptx = new StringBuilder(3_072);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape maxF1={maxF1} maxF2={maxF2} block={blockThreads} op=bispectrum-gather");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 specreal_ptr,");
        ptx.AppendLine("    .param .u64 specimag_ptr,");
        ptx.AppendLine("    .param .u64 outreal_ptr,");
        ptx.AppendLine("    .param .u64 outimag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<6>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [specreal_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [specimag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [outreal_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [outimag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $BSP_RET;");
        ptx.AppendLine($"    div.u32 %r3, %r2, {maxF2};");               // f1
        ptx.AppendLine($"    rem.u32 %r4, %r2, {maxF2};");               // f2
        ptx.AppendLine("    add.u32 %r5, %r3, %r4;");                     // sumIdx = f1 + f2
        // load X(f1), X(f2), conj X(f1+f2)
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd5];");   // ar
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");   // ai
        ptx.AppendLine("    mul.wide.u32 %rd7, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd8];");   // br
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd9];");   // bi
        ptx.AppendLine("    mul.wide.u32 %rd10, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd0, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd1, %rd10;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd11];");  // cr
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd12];");  // specImag[sumIdx]
        ptx.AppendLine("    neg.f32 %f5, %f5;");               // ci = -specImag[sumIdx]
        // ab = a * b : abr = ar*br - ai*bi, abi = ar*bi + ai*br
        ptx.AppendLine("    mul.rn.f32 %f6, %f1, %f3;");       // ai*bi
        ptx.AppendLine("    neg.f32 %f6, %f6;");
        ptx.AppendLine("    fma.rn.f32 %f6, %f0, %f2, %f6;");  // abr
        ptx.AppendLine("    mul.rn.f32 %f7, %f1, %f2;");       // ai*br
        ptx.AppendLine("    fma.rn.f32 %f7, %f0, %f3, %f7;");  // abi
        // out = ab * c : outReal = abr*cr - abi*ci, outImag = abr*ci + abi*cr
        ptx.AppendLine("    mul.rn.f32 %f8, %f7, %f5;");       // abi*ci
        ptx.AppendLine("    neg.f32 %f8, %f8;");
        ptx.AppendLine("    fma.rn.f32 %f8, %f6, %f4, %f8;");  // outReal
        ptx.AppendLine("    mul.rn.f32 %f9, %f7, %f4;");       // abi*cr
        ptx.AppendLine("    fma.rn.f32 %f9, %f6, %f5, %f9;");  // outImag
        ptx.AppendLine("    mul.wide.u32 %rd13, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd2, %rd13;");
        ptx.AppendLine("    add.u64 %rd15, %rd3, %rd13;");
        ptx.AppendLine("    st.global.f32 [%rd14], %f8;");
        ptx.AppendLine("    st.global.f32 [%rd15], %f9;");
        ptx.AppendLine("$BSP_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int maxF1, int maxF2, int specLength, int blockThreads)
    {
        var specExtent = new DirectPtxExtent(specLength);
        var outExtent = new DirectPtxExtent(checked(maxF1 * maxF2));
        return new DirectPtxKernelBlueprint(
            Operation: "bispectrum-gather-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-f1{maxF1}-f2{maxF2}-spec{specLength}",
            Tensors:
            [
                new("specReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    specExtent, specExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("specImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    specExtent, specExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("outImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "B(f1,f2) = X(f1) * X(f2) * conj(X(f1+f2))",
                ["mode"] = "inference-forward-bispectrum-gather",
                ["arithmetic"] = "triple complex product via multiply-then-fma contraction; tolerance-based parity",
                ["conjugate"] = "the f1+f2 sample is conjugated (neg.f32 on its imaginary lane)",
                ["bounds-check"] = "single guard drops lanes past maxF1*maxF2; caller sizes spec >= maxF1+maxF2-1",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int maxF1, int maxF2, int specLength) =>
        maxF1 >= 1 && maxF2 >= 1 && specLength >= maxF1 + maxF2 - 1 &&
        (long)maxF1 * maxF2 <= (1L << 26) && specLength <= (1 << 24);

    internal static bool IsPromotedShape(int maxF1, int maxF2, int specLength) => false;

    private static void Validate(int maxF1, int maxF2, int specLength)
    {
        if (!IsSupportedShape(maxF1, maxF2, specLength))
            throw new ArgumentOutOfRangeException(nameof(specLength),
                "The bispectrum-gather family requires maxF1>=1, maxF2>=1, specLength>=maxF1+maxF2-1<=2^24, " +
                "and maxF1*maxF2<=2^26.");
    }

    private static void ValidateShape(int maxF1, int maxF2)
    {
        if (maxF1 < 1 || maxF2 < 1 || (long)maxF1 * maxF2 > (1L << 26))
            throw new ArgumentOutOfRangeException(nameof(maxF1),
                "The bispectrum-gather family requires maxF1>=1, maxF2>=1, and maxF1*maxF2<=2^26.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Bispectrum-gather block threads must be 128, 256, or 512.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
