#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 per-sample mean-squared-error loss (issue #847):
/// <c>loss[i] = mean_j (predictions[i,j] - targets[i,j])^2</c>. A warp owns one
/// row, loads its prediction and target lanes once, fuses the difference,
/// square, warp reduction, and mean scale entirely in registers, and lane zero
/// commits one FP32 element — no materialized squared-error intermediate. There
/// are no shared-memory, local-memory, global-intermediate, temporary-
/// allocation, division, remainder, stride, or scalar shape parameters; only
/// three tensor pointers reach the launch ABI. The specialization stays disabled
/// by default and fails closed until three clean promotion runs clear the gate.
/// </summary>
internal sealed class PtxFusedMseLossF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_mse_loss_f32";
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

    internal PtxFusedMseLossF32Kernel(
        DirectPtxRuntime runtime,
        int rows,
        int columns,
        int blockThreads = DefaultBlockThreads)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "The checked-in FP32 MSE-loss specialization is validated only on Ampere.");
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
        DirectPtxTensorView predictions,
        DirectPtxTensorView targets,
        DirectPtxTensorView loss)
    {
        Require(predictions, Blueprint.Tensors[0], nameof(predictions));
        Require(targets, Blueprint.Tensors[1], nameof(targets));
        Require(loss, Blueprint.Tensors[2], nameof(loss));

        IntPtr predictionsPointer = predictions.Pointer;
        IntPtr targetsPointer = targets.Pointer;
        IntPtr lossPointer = loss.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &predictionsPointer;
        arguments[1] = &targetsPointer;
        arguments[2] = &lossPointer;
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
        // 1/columns as an IEEE-754 single-precision hex literal.
        uint invColumnsBits = BitConverter.SingleToUInt32Bits(1.0f / columns);
        string invColumns = "0f" + invColumnsBits.ToString("X8", CultureInfo.InvariantCulture);
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} columns={columns} block={blockThreads} strategy=warp-row op=mse-loss cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 pred_ptr,");
        ptx.AppendLine("    .param .u64 target_ptr,");
        ptx.AppendLine("    .param .u64 loss_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<6>;");
        ptx.AppendLine("    .reg .b64 %rd<13>;");
        ptx.AppendLine($"    .reg .f32 %f<{2 * valuesPerLane + 3}>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [pred_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [target_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [loss_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {warpsPerBlock}, %r2;");
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r4, {columns * sizeof(float)};");
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r3, {valuesPerLane * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
        ptx.AppendLine("    add.u64 %rd6, %rd5, %rd4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");
        ptx.AppendLine("    add.u64 %rd8, %rd7, %rd4;");
        int predBase = 0;
        int targetBase = valuesPerLane;
        int accReg = 2 * valuesPerLane;
        int shuffleReg = 2 * valuesPerLane + 1;
        string predRegs = string.Join(", ", RegisterRange(predBase, valuesPerLane));
        string targetRegs = string.Join(", ", RegisterRange(targetBase, valuesPerLane));
        ptx.AppendLine($"    ld.global.ca.v{valuesPerLane}.f32 {{{predRegs}}}, [%rd6];");
        ptx.AppendLine($"    ld.global.ca.v{valuesPerLane}.f32 {{{targetRegs}}}, [%rd8];");
        ptx.AppendLine($"    mov.f32 %f{accReg}, 0f00000000;");
        for (int i = 0; i < valuesPerLane; i++)
        {
            ptx.AppendLine($"    sub.rn.f32 %f{predBase + i}, %f{predBase + i}, %f{targetBase + i};");
            ptx.AppendLine($"    fma.rn.f32 %f{accReg}, %f{predBase + i}, %f{predBase + i}, %f{accReg};");
        }
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 %f{shuffleReg}, %f{accReg}, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    add.rn.f32 %f{accReg}, %f{accReg}, %f{shuffleReg};");
        }
        ptx.AppendLine($"    mul.rn.f32 %f{accReg}, %f{accReg}, {invColumns};");
        ptx.AppendLine("    setp.eq.u32 %p0, %r3, 0;");
        ptx.AppendLine($"    mul.wide.u32 %rd9, %r4, {sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine($"    @%p0 st.global.f32 [%rd10], %f{accReg};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static IEnumerable<string> RegisterRange(int start, int count)
    {
        for (int i = 0; i < count; i++)
            yield return $"%f{start + i}";
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows,
        int columns,
        int blockThreads)
    {
        var matrixExtent = new DirectPtxExtent(rows, columns);
        var lossExtent = new DirectPtxExtent(rows);
        return new DirectPtxKernelBlueprint(
            Operation: "mse-loss-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-row-w{blockThreads / 32}-r{rows}-c{columns}",
            Tensors:
            [
                new("predictions", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrixExtent, matrixExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("targets", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrixExtent, matrixExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("loss", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    lossExtent, lossExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "loss[i] = mean_j (pred[i,j] - target[i,j])^2",
                ["reduction"] = "mean-over-features",
                ["mode"] = "inference-forward-fused-diff-square-reduce",
                ["input"] = "fp32",
                ["accumulator"] = "fp32-warp-reduction-register-resident",
                ["output"] = "fp32",
                ["global-input-reads"] = "one-pred-plus-one-target-per-element",
                ["global-output-writes"] = "one-per-row",
                ["lane-vector-transaction"] = columns == 64 ? "aligned-fp32x2" : "aligned-fp32x4",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none-no-materialized-squared-error",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
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
                "The first MSE-loss family supports exact (rows,columns) buckets " +
                "(256,128), (2048,64), (2048,128), and (8192,128).");
        if (columns % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(columns));
    }

    private static void ValidateBlockThreads(int rows, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            rows % (blockThreads / 32) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "MSE-loss block threads must be 128, 256, or 512 and evenly tile rows.");
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
#endif
