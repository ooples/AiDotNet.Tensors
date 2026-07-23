using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Which layout conversion a <see cref="PtxComplexInterleaveF32Kernel"/> performs between the split
/// real/imag layout and the interleaved <c>[pair, real-imag]</c> layout. Each direction is a separate
/// module and a separate issue-#850 coverage cell.
/// </summary>
internal enum DirectPtxComplexInterleaveDirection
{
    /// <summary>split (real[i], imag[i]) -> interleaved[2i]=real[i], interleaved[2i+1]=imag[i].</summary>
    Interleave,

    /// <summary>interleaved[2i],[2i+1] -> real[i], imag[i].</summary>
    Deinterleave
}

/// <summary>
/// Exact contiguous FP32 complex layout conversions for issue #850 - the split↔interleaved bridges.
/// One thread owns one complex element and moves both lanes with a single 8-byte <c>v2</c> transaction
/// on the interleaved side (a load for deinterleave, a store for interleave) and two scalar transactions
/// on the split side. Both directions are pure data movement, so they are bit-exact copies with no
/// arithmetic. The launch covers exactly the element count, so neither needs the reference's bounds
/// guard and neither emits a branch. Three pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxComplexInterleaveF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxComplexInterleaveDirection Direction { get; }
    internal int Count { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal static string EntryPointFor(DirectPtxComplexInterleaveDirection direction) => direction switch
    {
        DirectPtxComplexInterleaveDirection.Interleave => "aidotnet_interleave_complex_f32",
        DirectPtxComplexInterleaveDirection.Deinterleave => "aidotnet_deinterleave_complex_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(direction))
    };

    internal PtxComplexInterleaveF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxComplexInterleaveDirection direction,
        int count,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in complex interleave specializations are admitted only on SM86.");
        ValidateDirection(direction);
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        Direction = direction;
        Count = count;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, direction, count, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, direction, count, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointFor(direction), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointFor(direction), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    /// <summary>
    /// For Interleave the views are (real, imag, interleaved); for Deinterleave they are
    /// (interleaved, real, imag) - the tensor order matches the launch ABI in both cases.
    /// </summary>
    internal unsafe void Launch(DirectPtxTensorView a, DirectPtxTensorView b, DirectPtxTensorView c)
    {
        Require(a, Blueprint.Tensors[0], nameof(a));
        Require(b, Blueprint.Tensors[1], nameof(b));
        Require(c, Blueprint.Tensors[2], nameof(c));

        IntPtr aPointer = a.Pointer, bPointer = b.Pointer, cPointer = c.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &aPointer;
        arguments[1] = &bPointer;
        arguments[2] = &cPointer;
        _module.Launch(
            _function,
            checked((uint)(Count / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, DirectPtxComplexInterleaveDirection direction, int count, int blockThreads = DefaultBlockThreads)
    {
        ValidateDirection(direction);
        Validate(count);
        ValidateBlockThreads(count, blockThreads);
        bool isInterleave = direction == DirectPtxComplexInterleaveDirection.Interleave;
        string entryPoint = EntryPointFor(direction);

        var ptx = new StringBuilder(2_048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape count={count} block={blockThreads} layout=split-interleaved-fp32 op={OpTag(direction)}");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        ptx.AppendLine("    .param .u64 p0_ptr,");
        ptx.AppendLine("    .param .u64 p1_ptr,");
        ptx.AppendLine("    .param .u64 p2_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<3>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [p0_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [p1_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [p2_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");   // split scalar offset = idx*4
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 8;");   // interleaved pair offset = idx*8
        if (isInterleave)
        {
            // in: real (%rd0), imag (%rd1); out: interleaved (%rd2).
            ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
            ptx.AppendLine("    add.u64 %rd6, %rd1, %rd3;");
            ptx.AppendLine("    ld.global.nc.f32 %f0, [%rd5];");           // real[i]
            ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");           // imag[i]
            ptx.AppendLine("    add.u64 %rd7, %rd2, %rd4;");
            ptx.AppendLine("    st.global.v2.f32 [%rd7], {%f0, %f1};");    // interleaved[2i], [2i+1]
        }
        else
        {
            // in: interleaved (%rd0); out: real (%rd1), imag (%rd2).
            ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
            ptx.AppendLine("    ld.global.nc.v2.f32 {%f0, %f1}, [%rd5];"); // interleaved[2i], [2i+1]
            ptx.AppendLine("    add.u64 %rd6, %rd1, %rd3;");
            ptx.AppendLine("    add.u64 %rd7, %rd2, %rd3;");
            ptx.AppendLine("    st.global.f32 [%rd6], %f0;");             // real[i]
            ptx.AppendLine("    st.global.f32 [%rd7], %f1;");             // imag[i]
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string OpTag(DirectPtxComplexInterleaveDirection direction) => direction switch
    {
        DirectPtxComplexInterleaveDirection.Interleave => "interleave-complex",
        DirectPtxComplexInterleaveDirection.Deinterleave => "deinterleave-complex",
        _ => throw new ArgumentOutOfRangeException(nameof(direction))
    };

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, DirectPtxComplexInterleaveDirection direction, int count, int blockThreads)
    {
        var scalarExtent = new DirectPtxExtent(count);
        var pairExtent = new DirectPtxExtent(count, 2);
        bool isInterleave = direction == DirectPtxComplexInterleaveDirection.Interleave;
        DirectPtxTensorContract Split(string name, DirectPtxTensorAccess access) => new(
            name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
            scalarExtent, scalarExtent, 16, access, DirectPtxExtentMode.Exact);
        DirectPtxTensorContract Interleaved(string name, DirectPtxTensorAccess access) => new(
            name, DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
            pairExtent, pairExtent, 16, access, DirectPtxExtentMode.Exact);
        return new DirectPtxKernelBlueprint(
            Operation: $"{OpTag(direction)}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"pairwise-v2-b{blockThreads}-n{count}",
            Tensors: isInterleave
                ?
                [
                    Split("real", DirectPtxTensorAccess.Read),
                    Split("imag", DirectPtxTensorAccess.Read),
                    Interleaved("interleaved", DirectPtxTensorAccess.Write)
                ]
                :
                [
                    Interleaved("interleaved", DirectPtxTensorAccess.Read),
                    Split("real", DirectPtxTensorAccess.Write),
                    Split("imag", DirectPtxTensorAccess.Write)
                ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 16,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = isInterleave
                    ? "interleaved[2i]=real[i]; interleaved[2i+1]=imag[i]"
                    : "real[i]=interleaved[2i]; imag[i]=interleaved[2i+1]",
                ["mode"] = $"inference-forward-{OpTag(direction)}",
                ["arithmetic"] = "none - pure bit-exact data movement",
                ["interleaved-access"] = "one-fp32x2-per-pair",
                ["split-access"] = "two-fp32-per-element",
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

    private static void ValidateDirection(DirectPtxComplexInterleaveDirection direction)
    {
        if (direction is not (DirectPtxComplexInterleaveDirection.Interleave or DirectPtxComplexInterleaveDirection.Deinterleave))
            throw new ArgumentOutOfRangeException(nameof(direction),
                "The complex interleave family covers Interleave and Deinterleave.");
    }

    private static void Validate(int count)
    {
        if (!IsSupportedShape(count))
            throw new ArgumentOutOfRangeException(nameof(count),
                "The first complex interleave family supports exact counts 65536, 262144, 1048576, and 4194304.");
    }

    private static void ValidateBlockThreads(int count, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || count % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Complex interleave block threads must be 128, 256, or 512 and evenly tile the element count.");
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
