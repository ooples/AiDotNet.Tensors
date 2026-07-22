#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 row-sum reduction (issue #843). A warp owns one row of
/// a <c>[rows, columns]</c> input, keeps every lane value in registers from the
/// single vectorized global load through a butterfly-shuffle add reduction, and
/// lane zero commits one FP32 element of the <c>[rows]</c> output. There are no
/// shared-memory, local-memory, global-intermediate, temporary-allocation,
/// stride, division, or scalar shape parameters — only two tensor pointers reach
/// the launch ABI. The specialization stays disabled by default and fails closed
/// until three clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedRowReduceF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_row_sum_f32";
    internal const int DefaultBlockThreads = 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int Columns { get; }
    internal int BlockThreads { get; }
    internal int WarpsPerBlock => BlockThreads / 32;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedRowReduceF32Kernel(
        DirectPtxRuntime runtime,
        int rows,
        int columns,
        int blockThreads = DefaultBlockThreads)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "The checked-in FP32 row-sum specialization is validated only on Ampere.");
        Validate(rows, columns);
        ValidateBlockThreads(rows, blockThreads);
        Rows = rows;
        Columns = columns;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows, columns, blockThreads);
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
            throw new ArgumentException("The row-sum specialization does not admit aliasing.");

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)((Rows + WarpsPerBlock - 1) / WarpsPerBlock)),
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
        int warpsPerBlock = blockThreads / 32;
        int valuesPerLane = columns / 32;
        var valueRegisterNames = new string[valuesPerLane];
        for (int i = 0; i < valuesPerLane; i++)
            valueRegisterNames[i] = $"%f{i}";
        string valueRegisters = string.Join(", ", valueRegisterNames);
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} columns={columns} block={blockThreads} strategy=warp-row op=sum cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<6>;");
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
        ptx.AppendLine($"    mov.f32 %f{valuesPerLane}, 0f00000000;");
        for (int i = 0; i < valuesPerLane; i++)
            ptx.AppendLine($"    add.rn.f32 %f{valuesPerLane}, %f{valuesPerLane}, %f{i};");
        EmitShuffleAddReduction(ptx, $"%f{valuesPerLane}", valuesPerLane + 1);
        ptx.AppendLine("    setp.eq.u32 %p0, %r3, 0;");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r4, {sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine($"    @%p0 st.global.f32 [%rd8], %f{valuesPerLane};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitShuffleAddReduction(
        StringBuilder ptx,
        string accumulator,
        int scratchIndex)
    {
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 %f{scratchIndex}, {accumulator}, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    add.rn.f32 {accumulator}, {accumulator}, %f{scratchIndex};");
        }
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows,
        int columns,
        int blockThreads)
    {
        var inputExtent = new DirectPtxExtent(rows, columns);
        var outputExtent = new DirectPtxExtent(rows);
        return new DirectPtxKernelBlueprint(
            Operation: "row-sum-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-row-w{blockThreads / 32}-r{rows}-c{columns}",
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
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "sum(x, axis=last)",
                ["axis"] = "last",
                ["mode"] = "inference-forward-fp32-tree-reduce",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-warp-reduction-register-resident",
                ["output"] = "fp32",
                ["global-input-reads"] = "one-per-element",
                ["global-output-writes"] = "one-per-row",
                ["lane-vector-transaction"] = columns == 64 ? "aligned-fp32x2" : "aligned-fp32x4",
                ["shared-intermediate"] = "none",
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

    private static void Validate(int rows, int columns)
    {
        if (!IsSupportedShape(rows, columns))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first row-sum family supports exact (rows,columns) buckets " +
                "(256,128), (2048,64), (2048,128), and (8192,128).");
        if (columns % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(columns));
    }

    private static void ValidateBlockThreads(int rows, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            rows % (blockThreads / 32) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Row-sum block threads must be 128, 256, or 512 and evenly tile rows.");
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
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }
}
#endif
