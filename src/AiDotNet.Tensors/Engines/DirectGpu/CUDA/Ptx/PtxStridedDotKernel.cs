using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact semantic specialization for the explicitly strided dot-product API.
/// The host reduces the baked offset/step bounds to one valid i interval, so
/// PTX receives no scalar shape/stride values and performs no per-element
/// bounds check. Input allocations themselves remain canonical vectors.
/// </summary>
internal sealed class PtxStridedDotKernel : IDisposable
{
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_strided_dot";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxStridedDotKernel(
        DirectPtxRuntime runtime,
        int aSize,
        int bSize,
        int bOffset,
        int bStep)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedFusedLinear(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The strided-dot PTX specialization is measured only on GA10x/SM86.");
        Validate(aSize, bSize);
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, aSize, bSize, bOffset, bStep);
        Ptx = EmitPtx(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            aSize, bSize, bOffset, bStep);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module);
    }

    internal unsafe void Launch(
        DirectPtxTensorView left,
        DirectPtxTensorView right,
        DirectPtxTensorView output)
    {
        Require(left, Blueprint.Tensors[0], nameof(left));
        Require(right, Blueprint.Tensors[1], nameof(right));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, left) || Overlaps(output, right))
            throw new ArgumentException("Strided-dot output may not alias an input.");
        IntPtr leftPointer = left.Pointer;
        IntPtr rightPointer = right.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &leftPointer;
        arguments[1] = &rightPointer;
        arguments[2] = &outputPointer;
        _module.Launch(
            _function, 1, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int aSize,
        int bSize,
        int bOffset,
        int bStep)
    {
        Validate(aSize, bSize);
        (int first, int last) = ValidInterval(aSize, bSize, bOffset, bStep);
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($"// exact strided dot a={aSize} b={bSize} offset={bOffset} step={bStep}; valid-i=[{first},{last}]");
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 left_ptr,");
        ptx.AppendLine("    .param .u64 right_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {BlockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        ptx.AppendLine($"    .shared .align 16 .b8 partial[{BlockThreads * sizeof(float)}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [left_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [right_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine($"    add.u32 %r1, %r0, {first};");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("STRIDED_LOOP:");
        ptx.AppendLine($"    setp.gt.s32 %p0, %r1, {last};");
        ptx.AppendLine("    @%p0 bra.uni STRIDED_REDUCE;");
        ptx.AppendLine($"    mad.lo.s32 %r2, %r1, {bStep}, {bOffset};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r1, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd4];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine($"    add.u32 %r1, %r1, {BlockThreads};");
        ptx.AppendLine("    bra.uni STRIDED_LOOP;");
        ptx.AppendLine("STRIDED_REDUCE:");
        ptx.AppendLine("    mov.u64 %rd7, partial;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd7, %rd8;");
        ptx.AppendLine("    st.shared.f32 [%rd9], %f0;");
        ptx.AppendLine("    bar.sync 0;");
        foreach (int offset in new[] { 128, 64, 32, 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    setp.ge.u32 %p1, %r0, {offset};");
            ptx.AppendLine($"    @%p1 bra.uni STRIDED_SKIP_{offset};");
            ptx.AppendLine($"    add.u32 %r3, %r0, {offset};");
            ptx.AppendLine("    mul.wide.u32 %rd10, %r3, 4;");
            ptx.AppendLine("    add.u64 %rd11, %rd7, %rd10;");
            ptx.AppendLine("    ld.shared.f32 %f3, [%rd9];");
            ptx.AppendLine("    ld.shared.f32 %f4, [%rd11];");
            ptx.AppendLine("    add.rn.f32 %f3, %f3, %f4;");
            ptx.AppendLine("    st.shared.f32 [%rd9], %f3;");
            ptx.AppendLine($"STRIDED_SKIP_{offset}:");
            ptx.AppendLine("    bar.sync 0;");
        }
        ptx.AppendLine("    setp.ne.u32 %p2, %r0, 0;");
        ptx.AppendLine("    @%p2 bra.uni STRIDED_DONE;");
        ptx.AppendLine("    ld.shared.f32 %f5, [%rd7];");
        ptx.AppendLine("    st.global.f32 [%rd2], %f5;");
        ptx.AppendLine("STRIDED_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    internal static (int First, int Last) ValidInterval(
        int aSize,
        int bSize,
        int bOffset,
        int bStep)
    {
        Validate(aSize, bSize);
        long first;
        long last;
        if (bStep > 0)
        {
            first = Math.Max(0, CeilDiv(-(long)bOffset, bStep));
            last = Math.Min(aSize - 1L, FloorDiv((long)bSize - 1 - bOffset, bStep));
        }
        else if (bStep < 0)
        {
            long step = -(long)bStep;
            first = Math.Max(0, CeilDiv((long)bOffset - (bSize - 1L), step));
            last = Math.Min(aSize - 1L, FloorDiv(bOffset, step));
        }
        else if (bOffset >= 0 && bOffset < bSize)
        {
            first = 0;
            last = aSize - 1L;
        }
        else
        {
            first = 0;
            last = -1;
        }
        if (first > last) return (0, -1);
        return (checked((int)first), checked((int)last));
    }

    private static long FloorDiv(long value, long positiveDivisor)
    {
        long quotient = value / positiveDivisor;
        long remainder = value % positiveDivisor;
        return remainder < 0 ? quotient - 1 : quotient;
    }

    private static long CeilDiv(long value, long positiveDivisor) =>
        -FloorDiv(-value, positiveDivisor);

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int aSize,
        int bSize,
        int bOffset,
        int bStep)
    {
        var left = new DirectPtxExtent(aSize);
        var right = new DirectPtxExtent(bSize);
        var output = new DirectPtxExtent(1);
        (int first, int last) = ValidInterval(aSize, bSize, bOffset, bStep);
        return new DirectPtxKernelBlueprint(
            Operation: "strided-dot",
            Version: 1,
            Architecture: architecture,
            Variant: $"fp32-a{aSize}-b{bSize}-o{bOffset}-s{bStep}",
            Tensors:
            [
                new("left", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    left, left, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("right", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    right, right, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: BlockThreads * sizeof(float),
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "sum(left[i]*right[bOffset+i*bStep]) for in-bounds right indices",
                ["b-offset"] = bOffset.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["b-step"] = bStep.ToString(System.Globalization.CultureInfo.InvariantCulture),
                ["valid-i"] = $"[{first},{last}]",
                ["runtime-scalar-parameters"] = "none",
                ["runtime-bounds-checks"] = "none",
                ["temporary-device-allocation"] = "none"
            });
    }

    private static void Validate(int aSize, int bSize)
    {
        if (aSize <= 0 || aSize > 1_048_576)
            throw new ArgumentOutOfRangeException(nameof(aSize));
        if (bSize <= 0 || bSize > 1_048_576)
            throw new ArgumentOutOfRangeException(nameof(bSize));
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = PtxCompat.ToNuint(left.Pointer);
        nuint rightStart = PtxCompat.ToNuint(right.Pointer);
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
