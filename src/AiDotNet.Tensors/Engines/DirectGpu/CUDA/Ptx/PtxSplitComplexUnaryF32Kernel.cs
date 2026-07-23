using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Which split-buffer unary reduction a <see cref="PtxSplitComplexUnaryF32Kernel"/> applies. Both
/// read the canonical split real/imag layout (two contiguous FP32 vectors) and write one contiguous
/// scalar output, so each is a separate module and a separate issue-#850 coverage cell.
/// </summary>
internal enum DirectPtxSplitComplexUnaryOp
{
    /// <summary>output[i] = sqrt(re[i]^2 + im[i]^2).</summary>
    Magnitude,

    /// <summary>output[i] = re[i]^2 + im[i]^2.</summary>
    MagnitudeSquared
}

/// <summary>
/// Exact contiguous split real/imag FP32 unary reductions for issue #850, the split siblings of the
/// interleaved <see cref="PtxFusedComplexUnaryF32Kernel"/>. One thread owns one element, reads the two
/// input lanes from their separate real/imag buffers, keeps everything in registers, and writes the
/// scalar result once. The launch covers exactly the element count, so neither op needs the
/// reference's bounds guard and neither emits a branch. Three pointers reach the launch ABI.
///
/// The magnitude operator follows <c>sqrtf(re*re + im*im)</c> exactly: <c>sqrt.rn</c> is the
/// IEEE-correct square root <c>sqrtf</c> compiles to, and the sum is formed with an unfused
/// multiply-multiply-add so it rounds the same way. An <c>fma</c> would be faster and MORE accurate,
/// which is precisely why it is not used - it would silently disagree with the kernel it replaces.
/// The magnitude-squared operator is the same sum without the root.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs
/// clear the release gate.
/// </summary>
internal sealed class PtxSplitComplexUnaryF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxSplitComplexUnaryOp Op { get; }
    internal int Count { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal static string EntryPointFor(DirectPtxSplitComplexUnaryOp op) => op switch
    {
        DirectPtxSplitComplexUnaryOp.Magnitude => "aidotnet_split_complex_magnitude_f32",
        DirectPtxSplitComplexUnaryOp.MagnitudeSquared => "aidotnet_split_complex_magnitude_squared_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    internal PtxSplitComplexUnaryF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxSplitComplexUnaryOp op,
        int count,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in split-complex unary specializations are admitted only on SM86.");
        ValidateOp(op);
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        Op = op;
        Count = count;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, op, count, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, op, count, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointFor(op), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointFor(op), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView inReal, DirectPtxTensorView inImag, DirectPtxTensorView output)
    {
        Require(inReal, Blueprint.Tensors[0], nameof(inReal));
        Require(inImag, Blueprint.Tensors[1], nameof(inImag));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inRealPointer = inReal.Pointer;
        IntPtr inImagPointer = inImag.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inRealPointer;
        arguments[1] = &inImagPointer;
        arguments[2] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)(Count / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, DirectPtxSplitComplexUnaryOp op, int count, int blockThreads = DefaultBlockThreads)
    {
        ValidateOp(op);
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        bool isMagnitude = op == DirectPtxSplitComplexUnaryOp.Magnitude;
        string entryPoint = EntryPointFor(op);

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape count={count} block={blockThreads} layout=split-complex-fp32 op={OpTag(op)}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        ptx.AppendLine("    .param .u64 in_real_ptr,");
        ptx.AppendLine("    .param .u64 in_imag_ptr,");
        ptx.AppendLine("    .param .u64 out_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [in_real_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [in_imag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [out_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd3;");
        // Streamed once, read through the read-only data cache.
        ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd4];");            // re
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");            // im
        // Unfused mul-mul-add so the rounding matches the reference sum.
        ptx.AppendLine("    mul.rn.f32 %f2, %f0, %f0;");                // re*re
        ptx.AppendLine("    mul.rn.f32 %f3, %f1, %f1;");                // im*im
        ptx.AppendLine("    add.rn.f32 %f2, %f2, %f3;");                // re*re + im*im
        if (isMagnitude)
            ptx.AppendLine("    sqrt.rn.f32 %f2, %f2;");                // IEEE-correct sqrtf
        ptx.AppendLine("    st.global.f32 [%rd6], %f2;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string OpTag(DirectPtxSplitComplexUnaryOp op) => op switch
    {
        DirectPtxSplitComplexUnaryOp.Magnitude => "split-complex-magnitude",
        DirectPtxSplitComplexUnaryOp.MagnitudeSquared => "split-complex-magnitude-squared",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, DirectPtxSplitComplexUnaryOp op, int count, int blockThreads)
    {
        var extent = new DirectPtxExtent(count);
        bool isMagnitude = op == DirectPtxSplitComplexUnaryOp.Magnitude;
        return new DirectPtxKernelBlueprint(
            Operation: $"{OpTag(op)}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"split-b{blockThreads}-n{count}",
            Tensors:
            [
                new("inReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("inImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = isMagnitude
                    ? "output[i] = sqrt(re[i]*re[i] + im[i]*im[i])"
                    : "output[i] = re[i]*re[i] + im[i]*im[i]",
                ["mode"] = $"inference-forward-{OpTag(op)}",
                ["input-layout"] = "canonical-split-real-imag-contiguous",
                ["output-layout"] = "contiguous-scalar",
                ["arithmetic"] = isMagnitude
                    ? "unfused mul-mul-add then sqrt.rn, matching sqrtf(re*re + im*im) rounding"
                    : "unfused mul-mul-add, matching the reference power sum",
                ["fma-use"] = "deliberately none - an fma would be more accurate and would disagree with the reference",
                ["global-input-reads"] = "two-fp32-per-element",
                ["global-output-writes"] = "one-fp32-per-element",
                ["bounds-check"] = "none - the launch covers exactly the element count",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>Mirrors the interleaved complex-unary family's exact element counts.</summary>
    internal static bool IsSupportedShape(int count) =>
        count is 65_536 or 262_144 or 1_048_576 or 4_194_304;

    internal static bool IsPromotedShape(int count) => false;

    internal static bool IsSupportedOp(DirectPtxSplitComplexUnaryOp op) =>
        op is DirectPtxSplitComplexUnaryOp.Magnitude or DirectPtxSplitComplexUnaryOp.MagnitudeSquared;

    private static void ValidateOp(DirectPtxSplitComplexUnaryOp op)
    {
        if (!IsSupportedOp(op))
            throw new ArgumentOutOfRangeException(nameof(op),
                "The split complex unary family covers Magnitude and MagnitudeSquared.");
    }

    private static void Validate(int count)
    {
        if (!IsSupportedShape(count))
            throw new ArgumentOutOfRangeException(nameof(count),
                "The first split-complex unary family supports exact counts 65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int count, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || count % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Split complex unary block threads must be 128, 256, or 512 and evenly tile the element count.");
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
