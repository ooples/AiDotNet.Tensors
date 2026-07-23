using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 row L2-normalization (issue #843, fused reduction):
/// <c>output[i,:] = input[i,:] * rsqrt(sum_j input[i,j]^2 + eps)</c> with
/// <c>eps = 1e-12</c>. A warp owns one row and retains every lane value in
/// registers from the sole global load, accumulates the sum of squares with
/// <c>fma.rn.f32</c>, reduces it across the warp, and rescales the retained
/// values before the sole final global store — so the per-row norm is
/// <b>never materialized</b> and the input is read exactly once. This is the
/// fusion analogue of the bare row-sum kernel: it folds a reduction and a
/// broadcast rescale into a single pass, removing the extra global round-trip a
/// two-kernel reduce-then-divide would pay. Zero shared/local, no global
/// intermediates, no temporary allocation, no stride or scalar shape parameters.
/// Disabled by default; fails closed until three clean promotion runs.
/// </summary>
internal sealed class PtxFusedRowL2NormalizeF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_row_l2normalize_f32";
    internal const int DefaultBlockThreads = 128;
    internal const float Epsilon = 1e-12f;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int Columns { get; }
    internal int BlockThreads { get; }
    internal int WarpsPerBlock => BlockThreads / 32;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedRowL2NormalizeF32Kernel(
        DirectPtxRuntime runtime,
        int rows,
        int columns,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedRowReduction(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in FP32 row-L2-normalize specialization is measured only on GA10x/SM86.");
        Validate(rows, columns);
        ValidateBlockThreads(rows, blockThreads);
        Rows = rows;
        Columns = columns;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows, columns, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, rows, columns, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));
        if (Overlaps(input, output))
            throw new ArgumentException("The row-L2-normalize specialization does not admit aliasing.");

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)((Rows + WarpsPerBlock - 1) / WarpsPerBlock)),
            1, 1,
            checked((uint)BlockThreads),
            1, 1,
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
        string epsilonLiteral = "0f" + PtxCompat.SingleToUInt32Bits(Epsilon).ToString("X8",
            System.Globalization.CultureInfo.InvariantCulture);
        var valueRegisterNames = new string[valuesPerLane];
        for (int i = 0; i < valuesPerLane; i++)
            valueRegisterNames[i] = $"%f{i}";
        string valueRegisters = string.Join(", ", valueRegisterNames);
        int accReg = valuesPerLane;
        int shuffleReg = valuesPerLane + 1;
        int invNormReg = valuesPerLane + 2;
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} columns={columns} block={blockThreads} strategy=warp-row op=l2normalize cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        // %r5/%r6 are the .b32 scratch pair the warp shuffle reinterprets through.
        ptx.AppendLine("    .reg .b32 %r<7>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
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
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd2;");
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r3, {valuesPerLane * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd4, %rd5;");
        // Streamed once and never revisited, so this goes through the
        // read-only data cache instead of displacing L1.
        ptx.AppendLine($"    ld.global.nc.v{valuesPerLane}.f32 {{{valueRegisters}}}, [%rd6];");
        ptx.AppendLine($"    mov.f32 %f{accReg}, 0f00000000;");
        for (int i = 0; i < valuesPerLane; i++)
            ptx.AppendLine($"    fma.rn.f32 %f{accReg}, %f{i}, %f{i}, %f{accReg};");
        // shfl.sync.bfly.b32 is a bit-manipulation instruction whose operands are
        // .b32 registers, not .f32. Reinterpret the float accumulator through a
        // .b32 register for the shuffle, then reinterpret the shuffled bits back
        // to .f32 before the arithmetic add — the same ISA-correct idiom the
        // row-reduce kernel uses, rather than relying on the assembler.
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r5, %f{accReg};");
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 %r6, %r5, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 %f{shuffleReg}, %r6;");
            ptx.AppendLine($"    add.rn.f32 %f{accReg}, %f{accReg}, %f{shuffleReg};");
        }
        ptx.AppendLine($"    add.rn.f32 %f{accReg}, %f{accReg}, {epsilonLiteral};");
        ptx.AppendLine($"    rsqrt.approx.f32 %f{invNormReg}, %f{accReg};");
        for (int i = 0; i < valuesPerLane; i++)
            ptx.AppendLine($"    mul.rn.f32 %f{i}, %f{i}, %f{invNormReg};");
        ptx.AppendLine($"    st.global.v{valuesPerLane}.f32 [%rd7], {{{valueRegisters}}};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows,
        int columns,
        int blockThreads)
    {
        var extent = new DirectPtxExtent(rows, columns);
        return new DirectPtxKernelBlueprint(
            Operation: "row-l2-normalize-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-row-w{blockThreads / 32}-r{rows}-c{columns}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i,:] = input[i,:] * rsqrt(sum_j input[i,j]^2 + 1e-12)",
                ["axis"] = "last",
                ["epsilon"] = "1e-12",
                ["mode"] = "inference-forward-fused-reduce-rescale-rsqrt-approx",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-warp-reduction-register-resident",
                ["output"] = "fp32",
                ["global-input-reads"] = "one-per-element",
                ["global-output-writes"] = "one-per-element",
                ["lane-vector-transaction"] = columns == 64 ? "aligned-fp32x2" : "aligned-fp32x4",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none-no-materialized-norm-vector",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["alias-policy"] = "input-output-disjoint",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int rows, int columns) =>
        (rows, columns) is (256, 128) or (2048, 64) or (2048, 128) or (8192, 128);

    internal static bool IsPromotedShape(int rows, int columns) => false;

    private static void Validate(int rows, int columns)
    {
        if (!IsSupportedShape(rows, columns))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first row-L2-normalize family supports exact (rows,columns) buckets " +
                "(256,128), (2048,64), (2048,128), and (8192,128).");
        if (columns % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(columns));
    }

    private static void ValidateBlockThreads(int rows, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            rows % (blockThreads / 32) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Row-L2-normalize block threads must be 128, 256, or 512 and evenly tile rows.");
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
