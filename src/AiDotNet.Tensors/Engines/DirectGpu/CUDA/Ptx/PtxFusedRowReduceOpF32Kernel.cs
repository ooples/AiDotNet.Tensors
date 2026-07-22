using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// The reduction operator a <see cref="PtxFusedRowReduceOpF32Kernel"/> applies
/// along a row. Each value is a separate module and a separate coverage cell in
/// issue #843; they share one emitter because the memory schedule - one
/// vectorized load, a register-resident lane fold, a butterfly-shuffle warp
/// fold, one store from lane zero - is identical, and only the identity, the
/// combine, and the finalize step differ.
/// </summary>
internal enum DirectPtxRowReduceOp
{
    /// <summary>Arithmetic mean: sum, then scale by a baked 1/columns.</summary>
    Mean,

    /// <summary>Row maximum.</summary>
    Max,

    /// <summary>Row minimum.</summary>
    Min,

    /// <summary>Sum of squares, folded with FMA so no intermediate is rounded twice.</summary>
    SumOfSquares
}

/// <summary>
/// Exact contiguous FP32 row reduction over a <c>[rows, columns]</c> input for
/// issue #843, covering the mean, max, min, and sum-of-squares operators. A warp
/// owns one row, keeps every lane value in registers from a single vectorized
/// global load through a butterfly-shuffle fold, and lane zero commits one FP32
/// element of the <c>[rows]</c> output.
///
/// This is the operator-parameterized sibling of
/// <see cref="PtxFusedRowReduceF32Kernel"/> (row sum). There are no
/// shared-memory, local-memory, global-intermediate, temporary-allocation,
/// stride, division, or scalar shape parameters - only two tensor pointers reach
/// the launch ABI, and for mean the reciprocal is baked into the module as an
/// IEEE-754 literal so the kernel multiplies rather than divides.
///
/// Max and Min use PTX <c>max.f32</c> / <c>min.f32</c>, which return the
/// non-NaN operand when exactly one operand is NaN - the same quieting
/// behaviour as CUDA's <c>fmaxf</c> / <c>fminf</c>, so the established kernel's
/// semantics are preserved.
///
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedRowReduceOpF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxRowReduceOp Op { get; }
    internal int Rows { get; }
    internal int Columns { get; }
    internal int BlockThreads { get; }
    internal int WarpsPerBlock => BlockThreads / 32;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    /// <summary>The module entry point for <paramref name="op"/>.</summary>
    internal static string EntryPointFor(DirectPtxRowReduceOp op) => op switch
    {
        DirectPtxRowReduceOp.Mean => "aidotnet_fused_row_mean_f32",
        DirectPtxRowReduceOp.Max => "aidotnet_fused_row_max_f32",
        DirectPtxRowReduceOp.Min => "aidotnet_fused_row_min_f32",
        DirectPtxRowReduceOp.SumOfSquares => "aidotnet_fused_row_sumsq_f32",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    internal PtxFusedRowReduceOpF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxRowReduceOp op,
        int rows,
        int columns,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedRowReduction(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP32 row-reduction specializations are measured only on GA10x/SM86.");
        ValidateOp(op);
        Validate(rows, columns);
        ValidateBlockThreads(rows, blockThreads);
        Op = op;
        Rows = rows;
        Columns = columns;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, op, rows, columns, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            op, rows, columns, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointFor(op), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointFor(op), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
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
        _module.Launch(
            _function,
            checked((uint)(Rows / WarpsPerBlock)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxRowReduceOp op,
        int rows,
        int columns,
        int blockThreads = DefaultBlockThreads)
    {
        ValidateOp(op);
        Validate(rows, columns);
        ValidateBlockThreads(rows, blockThreads);
        int warpsPerBlock = blockThreads / 32;
        int valuesPerLane = columns / 32;
        var valueRegisterNames = new string[valuesPerLane];
        for (int i = 0; i < valuesPerLane; i++)
            valueRegisterNames[i] = $"%f{i}";
        string valueRegisters = string.Join(", ", valueRegisterNames);
        int accReg = valuesPerLane;
        int scratchReg = valuesPerLane + 1;
        string entryPoint = EntryPointFor(op);

        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} columns={columns} block={blockThreads} strategy=warp-row op={OpTag(op)} cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<7>;");
        ptx.AppendLine("    .reg .b64 %rd<9>;");
        ptx.AppendLine($"    .reg .f32 %f<{valuesPerLane + 3}>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {warpsPerBlock}, %r2;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r4, {columns * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r3, {valuesPerLane * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine(
            $"    ld.global.ca.v{valuesPerLane}.f32 {{{valueRegisters}}}, [%rd6];");

        // Identity, then the register-resident lane fold.
        ptx.AppendLine($"    mov.f32 %f{accReg}, {IdentityLiteral(op)};");
        for (int i = 0; i < valuesPerLane; i++)
            ptx.AppendLine(LaneFold(op, accReg, i));

        EmitShuffleReduction(ptx, op, $"%f{accReg}", scratchReg);

        // Mean divides once, at the end, by a baked reciprocal.
        if (op == DirectPtxRowReduceOp.Mean)
        {
            string reciprocal = "0f" + PtxCompat.SingleToUInt32Bits(1f / columns)
                .ToString("X8", System.Globalization.CultureInfo.InvariantCulture);
            ptx.AppendLine($"    mul.rn.f32 %f{accReg}, %f{accReg}, {reciprocal};");
        }

        ptx.AppendLine("    setp.eq.u32 %p0, %r3, 0;");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r4, {sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine($"    @%p0 st.global.f32 [%rd8], %f{accReg};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string OpTag(DirectPtxRowReduceOp op) => op switch
    {
        DirectPtxRowReduceOp.Mean => "mean",
        DirectPtxRowReduceOp.Max => "max",
        DirectPtxRowReduceOp.Min => "min",
        DirectPtxRowReduceOp.SumOfSquares => "sumsq",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    /// <summary>
    /// The fold identity. Max and Min seed with the infinities rather than
    /// float.MinValue/MaxValue so a row of all-infinities still reduces exactly.
    /// </summary>
    private static string IdentityLiteral(DirectPtxRowReduceOp op) => op switch
    {
        DirectPtxRowReduceOp.Mean or DirectPtxRowReduceOp.SumOfSquares => "0f00000000",
        DirectPtxRowReduceOp.Max => "0fFF800000",   // -inf
        DirectPtxRowReduceOp.Min => "0f7F800000",   // +inf
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    private static string LaneFold(DirectPtxRowReduceOp op, int accReg, int i) => op switch
    {
        DirectPtxRowReduceOp.Mean =>
            $"    add.rn.f32 %f{accReg}, %f{accReg}, %f{i};",
        // One FMA per element: the square and the accumulate round once, not twice.
        DirectPtxRowReduceOp.SumOfSquares =>
            $"    fma.rn.f32 %f{accReg}, %f{i}, %f{i}, %f{accReg};",
        DirectPtxRowReduceOp.Max =>
            $"    max.f32 %f{accReg}, %f{accReg}, %f{i};",
        DirectPtxRowReduceOp.Min =>
            $"    min.f32 %f{accReg}, %f{accReg}, %f{i};",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    private static string WarpCombine(DirectPtxRowReduceOp op, string accumulator, int scratchIndex) => op switch
    {
        DirectPtxRowReduceOp.Mean or DirectPtxRowReduceOp.SumOfSquares =>
            $"    add.rn.f32 {accumulator}, {accumulator}, %f{scratchIndex};",
        DirectPtxRowReduceOp.Max =>
            $"    max.f32 {accumulator}, {accumulator}, %f{scratchIndex};",
        DirectPtxRowReduceOp.Min =>
            $"    min.f32 {accumulator}, {accumulator}, %f{scratchIndex};",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    // shfl.sync.bfly.b32 is a bit-manipulation instruction whose operands are
    // .b32 registers, not .f32. Reinterpret the float accumulator through a
    // .b32 register for the shuffle, then reinterpret the shuffled bits back to
    // .f32 before the arithmetic combine - the same ISA-correct idiom the row-sum
    // kernel uses.
    private static void EmitShuffleReduction(
        StringBuilder ptx,
        DirectPtxRowReduceOp op,
        string accumulator,
        int scratchIndex)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r5, {accumulator};");
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 %r6, %r5, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 %f{scratchIndex}, %r6;");
            ptx.AppendLine(WarpCombine(op, accumulator, scratchIndex));
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxRowReduceOp op,
        int rows,
        int columns,
        int blockThreads)
    {
        var inputExtent = new DirectPtxExtent(rows, columns);
        var outputExtent = new DirectPtxExtent(rows);
        return new DirectPtxKernelBlueprint(
            Operation: $"row-{OpTag(op)}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-row-b{blockThreads}-r{rows}-c{columns}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = FormulaFor(op),
                ["mode"] = $"inference-forward-warp-row-{OpTag(op)}",
                ["input"] = "fp32 row-major [rows, columns]",
                ["output"] = "fp32 [rows]",
                ["values-per-lane"] = (columns / 32).ToString(),
                ["global-input-reads"] = "one-vector-per-lane",
                ["global-output-writes"] = "one-fp32-per-row-from-lane-zero",
                ["warp-reduction"] = "butterfly-shuffle-through-b32",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["division"] = op == DirectPtxRowReduceOp.Mean
                    ? "none-reciprocal-baked-as-literal"
                    : "none",
                ["nan-policy"] = op is DirectPtxRowReduceOp.Max or DirectPtxRowReduceOp.Min
                    ? "max.f32/min.f32 return the non-NaN operand, matching fmaxf/fminf"
                    : "propagates",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    private static string FormulaFor(DirectPtxRowReduceOp op) => op switch
    {
        DirectPtxRowReduceOp.Mean => "output[r] = (1/columns) * sum_c input[r, c]",
        DirectPtxRowReduceOp.Max => "output[r] = max_c input[r, c]",
        DirectPtxRowReduceOp.Min => "output[r] = min_c input[r, c]",
        DirectPtxRowReduceOp.SumOfSquares => "output[r] = sum_c input[r, c] * input[r, c]",
        _ => throw new ArgumentOutOfRangeException(nameof(op))
    };

    /// <summary>The operator set mirrors the row-sum family's exact shape domain.</summary>
    internal static bool IsSupportedShape(int rows, int columns) =>
        rows is 256 or 512 or 1024 or 2048 or 4096 &&
        columns is 128 or 256 or 512 or 1024;

    internal static bool IsPromotedShape(int rows, int columns) => false;

    internal static bool IsSupportedOp(DirectPtxRowReduceOp op) =>
        op is DirectPtxRowReduceOp.Mean or DirectPtxRowReduceOp.Max
           or DirectPtxRowReduceOp.Min or DirectPtxRowReduceOp.SumOfSquares;

    private static void ValidateOp(DirectPtxRowReduceOp op)
    {
        if (!IsSupportedOp(op))
            throw new ArgumentOutOfRangeException(nameof(op),
                "The row-reduction family covers Mean, Max, Min, and SumOfSquares.");
    }

    private static void Validate(int rows, int columns)
    {
        if (!IsSupportedShape(rows, columns))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first row-reduction family supports exact rows 256, 512, 1024, 2048, 4096 " +
                "and exact columns 128, 256, 512, 1024.");
        // One warp per row, one vectorized load per lane: columns must split into
        // 32 lanes with a legal vector width.
        int valuesPerLane = columns / 32;
        if (columns % 32 != 0 || valuesPerLane is not (2 or 4))
            throw new ArgumentOutOfRangeException(nameof(columns),
                "Columns must divide into 32 lanes at a v2 or v4 vector width.");
    }

    private static void ValidateBlockThreads(int rows, int blockThreads)
    {
        if (blockThreads is not (64 or 128 or 256) || rows % (blockThreads / 32) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Row-reduction block threads must be 64, 128, or 256 and evenly tile the rows.");
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
}
