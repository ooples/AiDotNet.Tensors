using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape 2D FP32 transpose (issue #845), the shared-tile replacement for
/// the naive NVRTC <c>transpose_2d</c> kernel.
///
/// The established kernel reads one element per thread at
/// <c>[row * cols + col]</c> and writes it at <c>[col * rows + row]</c>. The read
/// is coalesced but the write is not: adjacent threads in a warp write addresses
/// <c>rows</c> floats apart, so every warp store fragments into 32 separate
/// memory transactions.
///
/// This specialization stages a 32x32 tile through shared memory so BOTH the
/// global read and the global write are fully coalesced. The tile row stride is
/// 33 floats rather than 32: the odd stride shifts each successive tile row by
/// one bank, so the transposed shared read (which walks a tile column) touches
/// 32 distinct banks instead of hitting one bank 32 times. That padding is the
/// whole reason the kernel is bank-conflict free, and it costs 128 bytes.
///
/// Launched as a 2D grid of (32, 8) blocks; each thread moves four elements, so
/// one block drains a full 32x32 tile. Only two pointers reach the launch ABI -
/// both extents are baked into the module, so there is no bounds branch, no
/// stride parameter, and no tail path. The specialization stays disabled by
/// default and fails closed until three clean promotion runs clear the release
/// gate.
/// </summary>
internal sealed class PtxFusedTranspose2DF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_transpose2d_f32";

    /// <summary>Tile edge. One block stages a <see cref="TileDim"/> square tile.</summary>
    internal const int TileDim = 32;

    /// <summary>Rows of threads per block; each thread moves <see cref="TileDim"/> / <see cref="BlockRows"/> elements.</summary>
    internal const int BlockRows = 8;

    /// <summary>Shared tile row stride in floats. 33, not 32, to break bank conflicts.</summary>
    internal const int TileStride = TileDim + 1;

    internal const int BlockThreads = TileDim * BlockRows;
    internal const int SharedBytes = TileDim * TileStride * sizeof(float);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int Columns { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedTranspose2DF32Kernel(DirectPtxRuntime runtime, int rows, int columns)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedTranspose2D(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in 2D transpose specialization is admitted only on SM86.");
        Validate(rows, columns);
        Rows = rows;
        Columns = columns;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows, columns);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, rows, columns);
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
        // One block per 32x32 tile: grid.x walks input columns, grid.y walks rows.
        _module.Launch(
            _function,
            checked((uint)(Columns / TileDim)),
            checked((uint)(Rows / TileDim)),
            1,
            TileDim,
            BlockRows,
            1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int rows, int columns)
    {
        Validate(rows, columns);
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} columns={columns} tile={TileDim}x{TileDim} block={TileDim}x{BlockRows} strategy=shared-tile op=transpose2d-f32");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {TileDim}, {BlockRows}, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<2>;");
        ptx.AppendLine($"    .shared .align 4 .b32 tile[{TileDim * TileStride}];");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u64 %rd2, tile;");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.y;");
        ptx.AppendLine("    mov.u32 %r2, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r3, %ctaid.y;");
        // Read coordinates: column = ctaid.x*32 + tid.x, row = ctaid.y*32 + tid.y.
        ptx.AppendLine($"    mad.lo.u32 %r4, %r2, {TileDim}, %r0;");
        ptx.AppendLine($"    mad.lo.u32 %r5, %r3, {TileDim}, %r1;");

        // Stage the tile. Consecutive threads read consecutive columns, so each
        // warp read is one coalesced transaction.
        for (int j = 0; j < TileDim; j += BlockRows)
        {
            ptx.AppendLine($"    add.u32 %r6, %r5, {j};");
            ptx.AppendLine($"    mad.lo.u32 %r7, %r6, {columns}, %r4;");
            ptx.AppendLine("    mul.wide.u32 %rd3, %r7, 4;");
            ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
            ptx.AppendLine("    ld.global.ca.f32 %f0, [%rd4];");
            ptx.AppendLine($"    add.u32 %r8, %r1, {j};");
            ptx.AppendLine($"    mad.lo.u32 %r9, %r8, {TileStride}, %r0;");
            ptx.AppendLine("    mul.wide.u32 %rd5, %r9, 4;");
            ptx.AppendLine("    add.u64 %rd6, %rd2, %rd5;");
            ptx.AppendLine("    st.shared.f32 [%rd6], %f0;");
        }
        ptx.AppendLine("    bar.sync 0;");

        // Write coordinates: the tile's block indices swap, so the output write
        // is also consecutive-per-thread and therefore coalesced.
        ptx.AppendLine($"    mad.lo.u32 %r10, %r3, {TileDim}, %r0;");
        ptx.AppendLine($"    mad.lo.u32 %r11, %r2, {TileDim}, %r1;");
        for (int j = 0; j < TileDim; j += BlockRows)
        {
            ptx.AppendLine($"    add.u32 %r12, %r11, {j};");
            ptx.AppendLine($"    mad.lo.u32 %r13, %r12, {rows}, %r10;");
            ptx.AppendLine("    mul.wide.u32 %rd7, %r13, 4;");
            ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
            // Transposed shared read: walk a tile column. The 33-float stride is
            // what keeps these 32 lanes on 32 distinct banks.
            ptx.AppendLine($"    add.u32 %r14, %r1, {j};");
            ptx.AppendLine($"    mad.lo.u32 %r15, %r0, {TileStride}, %r14;");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r15, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
            ptx.AppendLine("    ld.shared.f32 %f1, [%rd10];");
            ptx.AppendLine("    st.global.f32 [%rd8], %f1;");
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows,
        int columns)
    {
        var inputExtent = new DirectPtxExtent(rows, columns);
        var outputExtent = new DirectPtxExtent(columns, rows);
        return new DirectPtxKernelBlueprint(
            Operation: "transpose2d-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"shared-tile{TileDim}-r{rows}-c{columns}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: SharedBytes,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[c, r] = input[r, c]",
                ["mode"] = "inference-forward-shared-tile-transpose",
                ["input"] = "fp32 row-major [rows, columns]",
                ["output"] = "fp32 row-major [columns, rows]",
                ["tile"] = $"{TileDim}x{TileDim}",
                ["elements-per-thread"] = (TileDim / BlockRows).ToString(),
                ["global-input-reads"] = "one-coalesced-float-per-element",
                ["global-output-writes"] = "one-coalesced-float-per-element",
                ["shared-intermediate"] = $"one {TileDim}x{TileStride} fp32 tile ({SharedBytes} bytes)",
                ["shared-bank-conflicts"] = "none-odd-row-stride-33",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["synchronization"] = "one-bar-sync-between-stage-and-drain",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>
    /// Both extents come from a closed set of exact multiples of the tile edge,
    /// so the kernel never needs a bounds branch or a tail path.
    /// </summary>
    internal static bool IsSupportedShape(int rows, int columns) =>
        IsSupportedExtent(rows) && IsSupportedExtent(columns);

    private static bool IsSupportedExtent(int extent) =>
        extent is 512 or 1024 or 2048 or 4096;

    internal static bool IsPromotedShape(int rows, int columns) => false;

    private static void Validate(int rows, int columns)
    {
        if (!IsSupportedShape(rows, columns))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first 2D transpose family supports exact extents 512, 1024, 2048, and 4096 " +
                "on both axes.");
        if (rows % TileDim != 0 || columns % TileDim != 0)
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The exact-shape transpose requires both extents to be whole tiles.");
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
