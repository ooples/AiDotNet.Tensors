using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 row-softmax specialization. A warp owns a row and
/// retains every lane value from the sole global load through both reductions
/// to the sole final global store.
/// </summary>
internal sealed class PtxFusedSoftmaxF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_softmax_f32";
    internal const int DefaultBlockThreads = 512;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int Columns { get; }
    internal int BlockThreads { get; }
    internal int WarpsPerBlock => BlockThreads / 32;
    internal bool UsesCtaRowReduction => IsCtaRowVariant(Rows, Columns, BlockThreads);
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedSoftmaxF32Kernel(
        DirectPtxRuntime runtime,
        int rows,
        int columns,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedRowSoftmax(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP32 softmax specialization is measured only on GA10x/SM86.");
        Validate(rows, columns);
        ValidateBlockThreads(rows, blockThreads);
        Rows = rows;
        Columns = columns;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(
            runtime.ArchitectureFamily, rows, columns, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            rows, columns, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));
        if (Overlaps(input, output))
            throw new ArgumentException("The first softmax specialization does not admit aliasing.");

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)(UsesCtaRowReduction
                ? Rows
                : (Rows + WarpsPerBlock - 1) / WarpsPerBlock)),
            1,
            1,
            checked((uint)BlockThreads),
            1,
            1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int rows,
        int columns,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(rows, columns);
        ValidateBlockThreads(rows, blockThreads);
        if (IsCtaRowVariant(rows, columns, blockThreads))
            return EmitCtaRowPtx(ccMajor, ccMinor, rows, columns, blockThreads);
        int warpsPerBlock = blockThreads / 32;
        int valuesPerLane = columns / 32;
        var valueRegisterNames = new string[valuesPerLane];
        for (int i = 0; i < valuesPerLane; i++)
            valueRegisterNames[i] = $"%f{i}";
        string valueRegisters = string.Join(", ", valueRegisterNames);
        var ptx = new StringBuilder(12_288);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} columns={columns} block={blockThreads} strategy=warp-row cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {warpsPerBlock}, %r2;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r4, {columns * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r3, {valuesPerLane * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd5;");
        ptx.AppendLine("    mov.f32 %f8, 0fFF800000;");
        ptx.AppendLine(
            $"    ld.global.ca.v{valuesPerLane}.f32 {{{valueRegisters}}}, [%rd6];");
        for (int i = 0; i < valuesPerLane; i++)
            ptx.AppendLine($"    max.ftz.f32 %f8, %f8, %f{i};");
        EmitShuffleReduction(ptx, "max.ftz.f32", "%f8");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");
        for (int i = 0; i < valuesPerLane; i++)
        {
            ptx.AppendLine($"    sub.rn.ftz.f32 %f{i}, %f{i}, %f8;");
            ptx.AppendLine($"    mul.rn.ftz.f32 %f{i}, %f{i}, 0f3FB8AA3B;");
            ptx.AppendLine($"    ex2.approx.ftz.f32 %f{i}, %f{i};");
            ptx.AppendLine($"    add.rn.ftz.f32 %f9, %f9, %f{i};");
        }
        EmitShuffleReduction(ptx, "add.rn.ftz.f32", "%f9");
        ptx.AppendLine("    rcp.approx.ftz.f32 %f10, %f9;");
        for (int i = 0; i < valuesPerLane; i++)
            ptx.AppendLine($"    mul.rn.ftz.f32 %f{i}, %f{i}, %f10;");
        ptx.AppendLine(
            $"    st.global.v{valuesPerLane}.f32 [%rd7], {{{valueRegisters}}};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string EmitCtaRowPtx(
        int ccMajor,
        int ccMinor,
        int rows,
        int columns,
        int blockThreads)
    {
        var ptx = new StringBuilder(8_192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} columns={columns} block={blockThreads} strategy=cta-row cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<16>;");
        ptx.AppendLine("    .shared .align 16 .b8 scratch[16];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r1, {columns * sizeof(float)};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r0, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd2;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd2;");
        ptx.AppendLine("    add.u64 %rd6, %rd4, %rd3;");
        ptx.AppendLine("    add.u64 %rd7, %rd5, %rd3;");
        ptx.AppendLine("    mov.u64 %rd8, scratch;");
        ptx.AppendLine("    ld.global.ca.f32 %f0, [%rd6];");
        ptx.AppendLine("    mov.f32 %f8, %f0;");
        EmitShuffleReduction(ptx, "max.ftz.f32", "%f8");
        ptx.AppendLine("    setp.eq.u32 %p0, %r3, 0;");
        ptx.AppendLine("    mul.lo.u32 %r4, %r2, 4;");
        ptx.AppendLine("    cvt.u64.u32 %rd9, %r4;");
        ptx.AppendLine("    add.u64 %rd10, %rd8, %rd9;");
        ptx.AppendLine("    @%p0 st.shared.f32 [%rd10], %f8;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.lt.u32 %p1, %r3, 4;");
        ptx.AppendLine("    mul.lo.u32 %r5, %r3, 4;");
        ptx.AppendLine("    cvt.u64.u32 %rd9, %r5;");
        ptx.AppendLine("    add.u64 %rd11, %rd8, %rd9;");
        ptx.AppendLine("    @%p1 ld.shared.f32 %f8, [%rd11];");
        ptx.AppendLine("    @!%p1 mov.f32 %f8, 0fFF800000;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r2, 0;");
        ptx.AppendLine("    setp.eq.u32 %p3, %r0, 0;");
        ptx.AppendLine("    @%p2 bra CTA_MAX_REDUCED;");
        EmitShuffleReduction(ptx, "max.ftz.f32", "%f8");
        ptx.AppendLine("    @%p3 st.shared.f32 [scratch], %f8;");
        ptx.AppendLine("CTA_MAX_REDUCED:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    ld.shared.f32 %f8, [scratch];");
        ptx.AppendLine("    sub.rn.ftz.f32 %f0, %f0, %f8;");
        ptx.AppendLine("    mul.rn.ftz.f32 %f0, %f0, 0f3FB8AA3B;");
        ptx.AppendLine("    ex2.approx.ftz.f32 %f0, %f0;");
        ptx.AppendLine("    mov.f32 %f9, %f0;");
        EmitShuffleReduction(ptx, "add.rn.ftz.f32", "%f9");
        ptx.AppendLine("    @%p0 st.shared.f32 [%rd10], %f9;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    @%p1 ld.shared.f32 %f9, [%rd11];");
        ptx.AppendLine("    @!%p1 mov.f32 %f9, 0f00000000;");
        ptx.AppendLine("    @%p2 bra CTA_SUM_REDUCED;");
        EmitShuffleReduction(ptx, "add.rn.ftz.f32", "%f9");
        ptx.AppendLine("    @%p3 st.shared.f32 [scratch], %f9;");
        ptx.AppendLine("CTA_SUM_REDUCED:");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    ld.shared.f32 %f9, [scratch];");
        ptx.AppendLine("    rcp.approx.ftz.f32 %f10, %f9;");
        ptx.AppendLine("    mul.rn.ftz.f32 %f0, %f0, %f10;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    // shfl.sync.bfly.b32 is a bit-manipulation instruction: its operands are
    // .b32 registers, not .f32. Reinterpret the float accumulator through a
    // .b32 register for the shuffle, then reinterpret the shuffled bits back to
    // .f32 before the arithmetic reduction step. This is the ISA-correct idiom
    // (matching the fused QKV/RoPE warp reduction) rather than relying on the
    // assembler tolerating an .f32 register on a .b32 shuffle.
    private static void EmitShuffleReduction(
        StringBuilder ptx,
        string operation,
        string accumulator)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r6, {accumulator};");
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 %r7, %r6, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 %f11, %r7;");
            ptx.AppendLine($"    {operation} {accumulator}, {accumulator}, %f11;");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows,
        int columns,
        int blockThreads)
    {
        var extent = new DirectPtxExtent(rows, columns);
        bool ctaRow = IsCtaRowVariant(rows, columns, blockThreads);
        return new DirectPtxKernelBlueprint(
            Operation: "softmax-forward-f32",
            Version: 1,
            Architecture: architecture,
            Variant: ctaRow
                ? $"cta-row-w{blockThreads / 32}-r{rows}-c{columns}"
                : $"warp-row-w{blockThreads / 32}-r{rows}-c{columns}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: ctaRow ? 16 : 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: ctaRow ? 8 : 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "exp(x-max(x))/sum(exp(x-max(x)))",
                ["axis"] = "last",
                ["mode"] = "inference-forward-fast-exp2-rcp",
                ["input"] = "fp32",
                ["accumulator"] = ctaRow
                    ? "fp32-warp-cta-reduction-register-resident"
                    : "fp32-warp-reduction-register-resident",
                ["output"] = "fp32",
                ["global-input-reads"] = "one-per-element",
                ["global-output-writes"] = "one-final-per-element",
                ["lane-vector-transaction"] = ctaRow
                    ? "coalesced-fp32-scalar"
                    : columns == 64
                    ? "aligned-fp32x2"
                    : "aligned-fp32x4",
                ["shared-intermediate"] = ctaRow
                    ? "four-warp-partials-reused-for-max-and-sum-16-bytes"
                    : "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["alias-policy"] = "input-output-disjoint",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int rows, int columns) =>
        (rows, columns) is
            (256, 128) or (2048, 64) or (2048, 128) or (8192, 128);

    internal static bool IsPromotedShape(int rows, int columns) => false;

    private static bool IsCtaRowVariant(int rows, int columns, int blockThreads) =>
        rows == 256 && columns == 128 && blockThreads == 128;

    private static void Validate(int rows, int columns)
    {
        if (!IsSupportedShape(rows, columns))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first softmax family supports exact (rows,columns) buckets " +
                "(256,128), (2048,64), (2048,128), and (8192,128).");
        if (columns % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(columns));
    }

    private static void ValidateBlockThreads(int rows, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            rows % (blockThreads / 32) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Softmax block threads must be 128, 256, or 512 and evenly tile rows.");
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
