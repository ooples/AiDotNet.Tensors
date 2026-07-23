using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Spherical softmax (issue #854), matching the NVRTC <c>spherical_softmax</c> kernel:
/// L2-normalize each row (<c>x / sqrt(||x||^2 + 1e-12)</c>) then take a numerically-stable softmax
/// over the normalized row. One thread owns one row and walks the <c>innerSize</c> axis serially in
/// four passes (norm, normalize+max, exp+sum, scale) — no shared memory, no reduction; intermediate
/// values round-trip through the output row exactly as in the NVRTC kernel. <c>expf</c> is
/// reconstructed as <c>ex2.approx.f32(x * log2(e))</c>.
///
/// Shape (outerSize, innerSize) is baked into the PTX, so the launch takes buffer pointers only.
/// 256 threads/block, grid = outerSize/256 (a positive multiple of 256), so there is no divergent
/// bounds guard.
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
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(_function, (uint)(OuterSize / BlockThreads), 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int outerSize, int innerSize)
    {
        ValidateShape(outerSize, innerSize);
        string normEps = Hex(1e-12f), sumEps = Hex(1e-10f), log2e = Hex(1.4426950408889634f);
        const string one = "0f3F800000", negInf = "0fFF800000";

        var ptx = new StringBuilder(5_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// spherical-softmax outer={outerSize} inner={innerSize}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 in_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [in_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");   // row
        ptx.AppendLine($"    mul.lo.u32 %r3, %r2, {innerSize};");          // row*innerSize
        ptx.AppendLine("    mul.wide.u32 %rd2, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");                   // inBase
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");                   // outBase

        // Pass 1: norm_sq = sum in[j]^2 ; norm = sqrt(norm_sq + 1e-12)
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // norm_sq
        ptx.AppendLine("    mov.u64 %rd5, %rd3;");
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("$SS_P1:");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd5];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f5, %f5, %f0;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, 4;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r4, {innerSize};");
        ptx.AppendLine("    @%p0 bra $SS_P1;");
        ptx.AppendLine($"    add.rn.f32 %f0, %f0, {normEps};");
        ptx.AppendLine("    sqrt.rn.f32 %f1, %f0;");                      // norm
        ptx.AppendLine($"    div.rn.f32 %f1, {one}, %f1;");              // invNorm

        // Pass 2: out[j] = in[j]*invNorm ; max_val = max over out[j]
        ptx.AppendLine($"    mov.f32 %f2, {negInf};");                   // max_val
        ptx.AppendLine("    mov.u64 %rd5, %rd3;");                       // in walker
        ptx.AppendLine("    mov.u64 %rd6, %rd4;");                       // out walker
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("$SS_P2:");
        ptx.AppendLine("    ld.global.nc.f32 %f5, [%rd5];");
        ptx.AppendLine("    mul.rn.f32 %f6, %f5, %f1;");                 // normalized
        ptx.AppendLine("    st.global.f32 [%rd6], %f6;");
        ptx.AppendLine("    max.f32 %f2, %f2, %f6;");
        ptx.AppendLine("    add.u64 %rd5, %rd5, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p1, %r4, {innerSize};");
        ptx.AppendLine("    @%p1 bra $SS_P2;");

        // Pass 3: out[j] = exp(out[j] - max_val) ; sum_exp += out[j]
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");                  // sum_exp
        ptx.AppendLine("    mov.u64 %rd6, %rd4;");
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("$SS_P3:");
        ptx.AppendLine("    ld.global.f32 %f5, [%rd6];");
        ptx.AppendLine("    sub.rn.f32 %f6, %f5, %f2;");                 // out - max
        ptx.AppendLine($"    mul.rn.f32 %f6, %f6, {log2e};");
        ptx.AppendLine("    ex2.approx.f32 %f6, %f6;");                  // exp
        ptx.AppendLine("    st.global.f32 [%rd6], %f6;");
        ptx.AppendLine("    add.rn.f32 %f3, %f3, %f6;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p2, %r4, {innerSize};");
        ptx.AppendLine("    @%p2 bra $SS_P3;");
        ptx.AppendLine($"    add.rn.f32 %f3, %f3, {sumEps};");
        ptx.AppendLine($"    div.rn.f32 %f4, {one}, %f3;");             // inv_sum

        // Pass 4: out[j] *= inv_sum
        ptx.AppendLine("    mov.u64 %rd6, %rd4;");
        ptx.AppendLine("    mov.u32 %r4, 0;");
        ptx.AppendLine("$SS_P4:");
        ptx.AppendLine("    ld.global.f32 %f5, [%rd6];");
        ptx.AppendLine("    mul.rn.f32 %f5, %f5, %f4;");
        ptx.AppendLine("    st.global.f32 [%rd6], %f5;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p3, %r4, {innerSize};");
        ptx.AppendLine("    @%p3 bra $SS_P4;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture, int outerSize, int innerSize)
    {
        var extent = new DirectPtxExtent(outerSize * innerSize);
        return new DirectPtxKernelBlueprint(
            Operation: "spherical-softmax",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-o{outerSize}-i{innerSize}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "out = softmax_stable(normalize_L2(x)); norm eps 1e-12, sum eps 1e-10",
                ["approximation"] = "expf via ex2.approx.f32(x*log2e)",
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

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
