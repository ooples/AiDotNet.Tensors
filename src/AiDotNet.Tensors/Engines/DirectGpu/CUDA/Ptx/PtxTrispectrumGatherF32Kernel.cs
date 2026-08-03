using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Trispectrum gather for issue #850, matching the NVRTC <c>trispectrum_gather</c> kernel:
/// <c>T(f1,f2,f3) = X(f1) * X(f2) * X(f3) * conj(X(f1+f2+f3))</c> - a fourth-order spectral cumulant used
/// for cubic-phase-coupling analysis. One thread owns one output bin <c>(f1,f2,f3)</c>, reads the four
/// spectrum samples from the split buffers (the <c>f1+f2+f3</c> sample conjugated), and forms the quadruple
/// complex product with the multiply-then-fma contraction (nvcc's default fused evaluation), so the spec is
/// a tolerance-based fp64-oracle comparison. <c>maxF1</c>, <c>maxF2</c>, <c>maxF3</c>, and the spectrum
/// length are baked; the launch rounds up and a single guard drops the tail lanes. Four pointers reach the
/// launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxTrispectrumGatherF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_trispectrum_gather_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int MaxF1 { get; }
    internal int MaxF2 { get; }
    internal int MaxF3 { get; }
    internal int SpecLength { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxTrispectrumGatherF32Kernel(
        DirectPtxRuntime runtime, int maxF1, int maxF2, int maxF3, int specLength, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in trispectrum-gather specialization is admitted only on SM86.");
        Validate(maxF1, maxF2, maxF3, specLength);
        ValidateBlockThreads(blockThreads);
        MaxF1 = maxF1;
        MaxF2 = maxF2;
        MaxF3 = maxF3;
        SpecLength = specLength;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, maxF1, maxF2, maxF3, specLength, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, maxF1, maxF2, maxF3, blockThreads);
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
        int total = MaxF1 * MaxF2 * MaxF3;
        _module.Launch(
            _function,
            (uint)((total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int maxF1, int maxF2, int maxF3, int blockThreads = DefaultBlockThreads)
    {
        ValidateShape(maxF1, maxF2, maxF3);
        ValidateBlockThreads(blockThreads);
        int total = checked(maxF1 * maxF2 * maxF3);
        int f23 = checked(maxF2 * maxF3);

        var ptx = new StringBuilder(3_840);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape maxF1={maxF1} maxF2={maxF2} maxF3={maxF3} block={blockThreads} op=trispectrum-gather");
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
        ptx.AppendLine("    .reg .b32 %r<9>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [specreal_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [specimag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [outreal_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [outimag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $TSP_RET;");
        ptx.AppendLine($"    div.u32 %r3, %r2, {f23};");                 // f1
        ptx.AppendLine($"    mad.lo.u32 %r4, %r3, {f23}, 0;");           // f1*f23
        ptx.AppendLine("    sub.u32 %r4, %r2, %r4;");                     // rem
        ptx.AppendLine($"    div.u32 %r5, %r4, {maxF3};");               // f2
        ptx.AppendLine($"    rem.u32 %r6, %r4, {maxF3};");               // f3
        ptx.AppendLine("    add.u32 %r7, %r3, %r5;");                     // f1+f2
        ptx.AppendLine("    add.u32 %r7, %r7, %r6;");                     // sumIdx = f1+f2+f3
        // load a=X(f1), b=X(f2), c=X(f3), d=conj X(sumIdx)
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd5];");   // ar
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");   // ai
        ptx.AppendLine("    mul.wide.u32 %rd7, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd8];");   // br
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd9];");   // bi
        ptx.AppendLine("    mul.wide.u32 %rd10, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd0, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd1, %rd10;");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd11];");  // cr
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd12];");  // ci
        ptx.AppendLine("    mul.wide.u32 %rd13, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd0, %rd13;");
        ptx.AppendLine("    add.u64 %rd15, %rd1, %rd13;");
        ptx.AppendLine("    ld.global.nc.f32 %f6, [%rd14];");  // dr
        ptx.AppendLine("    ld.global.nc.f32 %f7, [%rd15];");  // specImag[sumIdx]
        ptx.AppendLine("    neg.f32 %f7, %f7;");               // di = -specImag[sumIdx]
        // t1 = a*b
        ptx.AppendLine("    mul.rn.f32 %f8, %f1, %f3;");       // ai*bi
        ptx.AppendLine("    neg.f32 %f8, %f8;");
        ptx.AppendLine("    fma.rn.f32 %f8, %f0, %f2, %f8;");  // t1r
        ptx.AppendLine("    mul.rn.f32 %f9, %f1, %f2;");       // ai*br
        ptx.AppendLine("    fma.rn.f32 %f9, %f0, %f3, %f9;");  // t1i
        // t2 = t1*c
        ptx.AppendLine("    mul.rn.f32 %f10, %f9, %f5;");      // t1i*ci
        ptx.AppendLine("    neg.f32 %f10, %f10;");
        ptx.AppendLine("    fma.rn.f32 %f10, %f8, %f4, %f10;"); // t2r
        ptx.AppendLine("    mul.rn.f32 %f11, %f9, %f4;");      // t1i*cr
        ptx.AppendLine("    fma.rn.f32 %f11, %f8, %f5, %f11;"); // t2i
        // out = t2*d
        ptx.AppendLine("    mul.rn.f32 %f12, %f11, %f7;");     // t2i*di
        ptx.AppendLine("    neg.f32 %f12, %f12;");
        ptx.AppendLine("    fma.rn.f32 %f12, %f10, %f6, %f12;"); // outReal
        ptx.AppendLine("    mul.rn.f32 %f13, %f11, %f6;");     // t2i*dr
        ptx.AppendLine("    fma.rn.f32 %f13, %f10, %f7, %f13;"); // outImag
        ptx.AppendLine("    mul.wide.u32 %rd16, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd17, %rd2, %rd16;");
        ptx.AppendLine("    add.u64 %rd18, %rd3, %rd16;");
        ptx.AppendLine("    st.global.f32 [%rd17], %f12;");
        ptx.AppendLine("    st.global.f32 [%rd18], %f13;");
        ptx.AppendLine("$TSP_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int maxF1, int maxF2, int maxF3, int specLength, int blockThreads)
    {
        var specExtent = new DirectPtxExtent(specLength);
        var outExtent = new DirectPtxExtent(checked(maxF1 * maxF2 * maxF3));
        return new DirectPtxKernelBlueprint(
            Operation: "trispectrum-gather-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-f1{maxF1}-f2{maxF2}-f3{maxF3}-spec{specLength}",
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
                MaxRegistersPerThread: 28,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "T(f1,f2,f3) = X(f1) * X(f2) * X(f3) * conj(X(f1+f2+f3))",
                ["mode"] = "inference-forward-trispectrum-gather",
                ["arithmetic"] = "quadruple complex product via multiply-then-fma contraction; tolerance-based parity",
                ["conjugate"] = "the f1+f2+f3 sample is conjugated (neg.f32 on its imaginary lane)",
                ["bounds-check"] = "single guard drops lanes past maxF1*maxF2*maxF3; caller sizes spec >= maxF1+maxF2+maxF3-2",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int maxF1, int maxF2, int maxF3, int specLength) =>
        maxF1 >= 1 && maxF2 >= 1 && maxF3 >= 1 && specLength >= maxF1 + maxF2 + maxF3 - 2 &&
        (long)maxF1 * maxF2 * maxF3 <= (1L << 26) && specLength <= (1 << 24);

    internal static bool IsPromotedShape(int maxF1, int maxF2, int maxF3, int specLength) => false;

    private static void Validate(int maxF1, int maxF2, int maxF3, int specLength)
    {
        if (!IsSupportedShape(maxF1, maxF2, maxF3, specLength))
            throw new ArgumentOutOfRangeException(nameof(specLength),
                "The trispectrum-gather family requires maxF1>=1, maxF2>=1, maxF3>=1, " +
                "specLength>=maxF1+maxF2+maxF3-2<=2^24, and maxF1*maxF2*maxF3<=2^26.");
    }

    private static void ValidateShape(int maxF1, int maxF2, int maxF3)
    {
        if (maxF1 < 1 || maxF2 < 1 || maxF3 < 1 || (long)maxF1 * maxF2 * maxF3 > (1L << 26))
            throw new ArgumentOutOfRangeException(nameof(maxF1),
                "The trispectrum-gather family requires maxF1>=1, maxF2>=1, maxF3>=1, and maxF1*maxF2*maxF3<=2^26.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Trispectrum-gather block threads must be 128, 256, or 512.");
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
