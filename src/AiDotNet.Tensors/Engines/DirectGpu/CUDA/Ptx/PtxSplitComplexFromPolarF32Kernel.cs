using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Split real/imag FP32 polar-to-Cartesian conversion for issue #850:
/// <c>outReal[i] = mag[i] * cos(phase[i])</c>, <c>outImag[i] = mag[i] * sin(phase[i])</c>. One thread
/// owns one element and uses the hardware <c>cos.approx.f32</c>/<c>sin.approx.f32</c>, so - like the
/// phase operator - this is covered by a tolerance-based parity spec rather than a bit-exact one. No
/// shared memory, no reduction, no branch; the launch covers exactly the element count. Four pointers
/// reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxSplitComplexFromPolarF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_split_complex_from_polar_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Count { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSplitComplexFromPolarF32Kernel(
        DirectPtxRuntime runtime, int count, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in split-complex from-polar specialization is admitted only on SM86.");
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        Count = count;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, count, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, count, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView mag, DirectPtxTensorView phase,
        DirectPtxTensorView outReal, DirectPtxTensorView outImag)
    {
        Require(mag, Blueprint.Tensors[0], nameof(mag));
        Require(phase, Blueprint.Tensors[1], nameof(phase));
        Require(outReal, Blueprint.Tensors[2], nameof(outReal));
        Require(outImag, Blueprint.Tensors[3], nameof(outImag));

        IntPtr magPointer = mag.Pointer, phasePointer = phase.Pointer;
        IntPtr outRealPointer = outReal.Pointer, outImagPointer = outImag.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &magPointer;
        arguments[1] = &phasePointer;
        arguments[2] = &outRealPointer;
        arguments[3] = &outImagPointer;
        _module.Launch(
            _function,
            checked((uint)(Count / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int count, int blockThreads = DefaultBlockThreads)
    {
        Validate(count);
        ValidateBlockThreads(count, blockThreads);

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape count={count} block={blockThreads} layout=split-complex-fp32 op=split-complex-from-polar");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 mag_ptr,");
        ptx.AppendLine("    .param .u64 phase_ptr,");
        ptx.AppendLine("    .param .u64 out_real_ptr,");
        ptx.AppendLine("    .param .u64 out_imag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [mag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [phase_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [out_imag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd5];");   // m
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");   // p
        ptx.AppendLine("    cos.approx.f32 %f2, %f1;");        // cos(p)
        ptx.AppendLine("    sin.approx.f32 %f3, %f1;");        // sin(p)
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f2;");       // m*cos(p)
        ptx.AppendLine("    mul.rn.f32 %f3, %f0, %f3;");       // m*sin(p)
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd4;");
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f2;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f3;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int count, int blockThreads)
    {
        var extent = new DirectPtxExtent(count);
        return new DirectPtxKernelBlueprint(
            Operation: "split-complex-from-polar-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"split-b{blockThreads}-n{count}",
            Tensors:
            [
                new("mag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("phase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("outImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "outReal[i] = mag[i]*cos(phase[i]); outImag[i] = mag[i]*sin(phase[i])",
                ["mode"] = "inference-forward-split-complex-from-polar",
                ["approximation"] = "cos.approx/sin.approx; tolerance-based parity, not bit-exact",
                ["input-layout"] = "canonical-split-mag-phase-contiguous",
                ["output-layout"] = "canonical-split-real-imag-contiguous",
                ["global-input-reads"] = "two-fp32-per-element",
                ["global-output-writes"] = "two-fp32-per-element",
                ["bounds-check"] = "none - the launch covers exactly the element count",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int count) =>
        count is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int count) => false;

    private static void Validate(int count)
    {
        if (!IsSupportedShape(count))
            throw new ArgumentOutOfRangeException(nameof(count),
                "The first split-complex from-polar family supports exact counts 65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int count, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || count % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Split complex from-polar block threads must be 128, 256, or 512 and evenly tile the element count.");
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
