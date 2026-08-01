using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Spherical softmax (issue #854), matching the NVRTC <c>spherical_softmax</c> kernel:
/// L2-normalize each row (<c>x / sqrt(||x||^2 + 1e-12)</c>) then take a numerically-stable softmax
/// over the normalized row. One warp owns one row and its lanes walk the <c>innerSize</c> axis in
/// four passes (norm, normalize+max, exp+sum, scale). Warp shuffles reduce the norm, maximum, and
/// exponential sum without shared memory; intermediate values round-trip through the output row
/// exactly as in the NVRTC kernel. <c>expf</c> is reconstructed as
/// <c>ex2.approx.f32(x * log2(e))</c>.
///
/// Shape (outerSize, innerSize) is baked into the PTX, so the launch takes buffer pointers only.
/// 256 threads/block (eight rows/block), grid = outerSize/8 (a positive multiple of eight), so
/// there is no divergent row bounds guard.
/// </summary>
internal sealed class PtxSphericalSoftmaxKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const int MaxOuter = 2048 * 4096;
    internal const int MaxInner = 4096;
    internal const string EntryPoint = "aidotnet_spherical_softmax";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int OuterSize { get; }
    internal int InnerSize { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSphericalSoftmaxKernel(DirectPtxRuntime runtime, int outerSize, int innerSize)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedScientific(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in spherical-softmax specialization is measured only on GA10x/SM86.");
        ValidateShape(outerSize, innerSize);
        OuterSize = outerSize;
        InnerSize = innerSize;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, outerSize, innerSize);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, outerSize, innerSize);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        DirectPtxAbi.Require(input, Blueprint.Tensors[0], nameof(input));
        DirectPtxAbi.Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function, (uint)(OuterSize / (BlockThreads / 32)), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();


    internal static string EmitPtx(int ccMajor, int ccMinor, int outerSize, int innerSize)
    {
        ValidateShape(outerSize, innerSize);
        string normEps = DirectPtxPtxText.Hex(1e-12f), sumEps = DirectPtxPtxText.Hex(1e-10f), log2e = DirectPtxPtxText.Hex(1.4426950408889634f);
        const string one = "0f3F800000", negInf = "0fFF800000";

        var ptx = new StringBuilder(5_000);
        DirectPtxPtxText.AppendModuleHeader(ptx, ccMajor, ccMinor, disableLoopUnrolling: true);
        ptx.AppendLine($"// spherical-softmax outer={outerSize} inner={innerSize}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 in_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [in_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r0, %r1, {BlockThreads}, %r0;"); // global thread
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");                       // warp owns row
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");                      // lane owns feature
        ptx.AppendLine($"    mul.lo.u32 %r4, %r2, {innerSize};");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");                   // inBase
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");                   // outBase

        // Pass 1: lane partial norm, then warp sum.
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine("    mov.u32 %r5, %r3;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @!%p0 bra $SS_P1_DONE;");
        ptx.AppendLine("$SS_P1:");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd6];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f5, %f5, %f0;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 128;");
        ptx.AppendLine("    add.u32 %r5, %r5, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @%p0 bra $SS_P1;");
        ptx.AppendLine("$SS_P1_DONE:");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f0", "%r6", "%r7", "%f6");
        ptx.AppendLine("    mov.b32 %r6, %f0;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r7, %r6, 0, 31, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f0, %r7;");
        ptx.AppendLine($"    add.rn.f32 %f0, %f0, {normEps};");
        ptx.AppendLine("    sqrt.rn.f32 %f1, %f0;");
        ptx.AppendLine($"    div.rn.f32 %f1, {one}, %f1;");

        // Pass 2: normalize and store lane elements, then warp maximum.
        ptx.AppendLine($"    mov.f32 %f2, {negInf};");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd5;");
        ptx.AppendLine("    mov.u32 %r5, %r3;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @!%p0 bra $SS_P2_DONE;");
        ptx.AppendLine("$SS_P2:");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd6];");
        ptx.AppendLine("    mul.rn.f32 %f7, %f5, %f1;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f7;");
        ptx.AppendLine("    max.f32 %f2, %f2, %f7;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 128;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 128;");
        ptx.AppendLine("    add.u32 %r5, %r5, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @%p0 bra $SS_P2;");
        ptx.AppendLine("$SS_P2_DONE:");
        DirectPtxPtxText.AppendWarpMax(ptx, "%f2", "%r6", "%r7", "%f6");
        ptx.AppendLine("    mov.b32 %r6, %f2;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r7, %r6, 0, 31, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f2, %r7;");

        // Pass 3: exponentiate lane elements, then warp sum.
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd5;");
        ptx.AppendLine("    mov.u32 %r5, %r3;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @!%p0 bra $SS_P3_DONE;");
        ptx.AppendLine("$SS_P3:");
        ptx.AppendLine("    ld.global.f32 %f5, [%rd7];");
        ptx.AppendLine("    sub.rn.f32 %f7, %f5, %f2;");
        ptx.AppendLine($"    mul.rn.f32 %f7, %f7, {log2e};");
        ptx.AppendLine("    ex2.approx.f32 %f7, %f7;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f7;");
        ptx.AppendLine("    add.rn.f32 %f3, %f3, %f7;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 128;");
        ptx.AppendLine("    add.u32 %r5, %r5, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @%p0 bra $SS_P3;");
        ptx.AppendLine("$SS_P3_DONE:");
        DirectPtxPtxText.AppendWarpSum(ptx, "%f3", "%r6", "%r7", "%f6");
        ptx.AppendLine("    mov.b32 %r6, %f3;");
        ptx.AppendLine("    shfl.sync.idx.b32 %r7, %r6, 0, 31, 0xffffffff;");
        ptx.AppendLine("    mov.b32 %f3, %r7;");
        ptx.AppendLine($"    add.rn.f32 %f3, %f3, {sumEps};");
        ptx.AppendLine($"    div.rn.f32 %f4, {one}, %f3;");

        // Pass 4: scale lane elements.
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd5;");
        ptx.AppendLine("    mov.u32 %r5, %r3;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @!%p0 bra $SS_DONE;");
        ptx.AppendLine("$SS_P4:");
        ptx.AppendLine("    ld.global.f32 %f5, [%rd7];");
        ptx.AppendLine("    mul.rn.f32 %f5, %f5, %f4;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f5;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 128;");
        ptx.AppendLine("    add.u32 %r5, %r5, 32;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r5, {innerSize};");
        ptx.AppendLine("    @%p0 bra $SS_P4;");
        ptx.AppendLine("$SS_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int outerSize, int innerSize)
    {
        var extent = new DirectPtxExtent(outerSize * innerSize);
        return new DirectPtxKernelBlueprint(
            Operation: "spherical-softmax",
            Version: 2,
            Architecture: architecture,
            Variant: $"fp32-warp-o{outerSize}-i{innerSize}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: DirectPtxResourceBudget.FromDriverMeasurement(
                measuredRegistersPerThread: 24,
                maxStaticSharedBytes: 0,
                maxLocalBytesPerThread: 0,
                minBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "out = softmax_stable(normalize_L2(x)); norm eps 1e-12, sum eps 1e-10",
                ["approximation"] = "expf via ex2.approx.f32(x*log2e)",
                ["reduction"] = "one warp per row; shuffle sum/max; lane-strided features",
                ["global-intermediates"] = "output row reused across passes (matches NVRTC)",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none"
            });
    }

    internal static bool IsSupportedShape(int outerSize, int innerSize)
    {
        if (outerSize <= 0 || innerSize <= 0 || innerSize > MaxInner) return false;
        return outerSize % BlockThreads == 0 && outerSize <= MaxOuter;
    }

    internal static bool IsPromotedShape(int outerSize, int innerSize) => false;

    private static void ValidateShape(int outerSize, int innerSize)
    {
        if (!IsSupportedShape(outerSize, innerSize))
            throw new ArgumentOutOfRangeException(
                nameof(outerSize),
                $"Spherical softmax requires positive dims with innerSize<={MaxInner} and outerSize a multiple of {BlockThreads} up to {MaxOuter}.");
    }

}
