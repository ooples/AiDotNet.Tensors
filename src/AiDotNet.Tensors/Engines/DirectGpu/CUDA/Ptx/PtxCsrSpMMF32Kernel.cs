using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP32 CSR SpMM (issue #852): <c>output[M,N] = A[M,K] * B[K,N]</c> with A in CSR form
/// (INT32 row pointers and column indices, FP32 values) and B, output dense row-major.
/// <para>
/// N / 4 adjacent lanes own one output row, 4 columns each, so a 256-thread block covers 256 / (N / 4) rows and every
/// lane has work. (The NVRTC <c>csr_spmm_vec4</c> it ports runs one 256-thread block per row with only N / 4 lanes
/// active.) Each lane walks its row's nonzeros in stored order, reads the column index and value once per nonzero
/// (lanes of a row read the same address, so the warp issues one transaction), streams the matching 16-byte slice of
/// B through the read-only cache and accumulates with <c>fma.rn</c> in the same order as the NVRTC kernel, so the
/// result is bit-identical to it. The output row is written once with a 16-byte store.
/// </para>
/// <para>
/// M, K and N are baked into the module; only the five tensor pointers reach the launch ABI. The nonzero count is
/// data (row_ptr[M]), not shape: the values and column-index tensors are at-least views checked once against it on
/// the host. There is no shared memory, no local memory, no global intermediate and no temporary allocation. The
/// specialization is experimental and fails closed until three clean promotion runs clear the gate.
/// </para>
/// </summary>
internal sealed class PtxCsrSpMMF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_csr_spmm_f32";
    internal const int DefaultBlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int Inner { get; }
    internal int Columns { get; }
    internal int BlockThreads { get; }
    internal int LanesPerRow => Columns / 4;
    internal int RowsPerBlock => BlockThreads / LanesPerRow;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCsrSpMMF32Kernel(DirectPtxRuntime runtime, int rows, int inner, int columns,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedCsrSpMM(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException("The checked-in FP32 CSR SpMM specialization is admitted only on SM86.");
        Validate(rows, inner, columns, blockThreads);
        Rows = rows;
        Inner = inner;
        Columns = columns;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows, inner, columns, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, rows, inner, columns, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal uint GridBlocks => checked((uint)((Rows + RowsPerBlock - 1) / RowsPerBlock));

    /// <summary>Launches on the runtime's stream. <paramref name="nonzeros"/> is row_ptr[M].</summary>
    internal unsafe void Launch(
        DirectPtxTensorView rowPointers,
        DirectPtxTensorView columnIndices,
        DirectPtxTensorView values,
        DirectPtxTensorView dense,
        DirectPtxTensorView output,
        int nonzeros)
    {
        Require(rowPointers, Blueprint.Tensors[0], nameof(rowPointers));
        Require(columnIndices, Blueprint.Tensors[1], nameof(columnIndices));
        Require(values, Blueprint.Tensors[2], nameof(values));
        Require(dense, Blueprint.Tensors[3], nameof(dense));
        Require(output, Blueprint.Tensors[4], nameof(output));
        nuint nonzeroBytes = checked((nuint)nonzeros * sizeof(int));
        if (nonzeros < 0 || columnIndices.AllocationByteLength < nonzeroBytes || values.AllocationByteLength < nonzeroBytes)
            throw new ArgumentException($"The CSR index and value tensors must hold {nonzeros} nonzeros.", nameof(nonzeros));

        IntPtr rowPointer = rowPointers.Pointer, indexPointer = columnIndices.Pointer, valuePointer = values.Pointer;
        IntPtr densePointer = dense.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &rowPointer;
        arguments[1] = &indexPointer;
        arguments[2] = &valuePointer;
        arguments[3] = &densePointer;
        arguments[4] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, checked((uint)BlockThreads), 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int rows, int inner, int columns,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(rows, inner, columns, blockThreads);
        int lanesPerRow = columns / 4;
        int laneShift = Log2(lanesPerRow);
        int rowsPerBlock = blockThreads / lanesPerRow;
        int rowBytes = columns * sizeof(float);
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape rows={rows} inner={inner} columns={columns} block={blockThreads} " +
            $"strategy=lanes-per-row-{lanesPerRow} op=csr-spmm order=stored-fma");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 row_ptr_ptr,");
        ptx.AppendLine("    .param .u64 col_idx_ptr,");
        ptx.AppendLine("    .param .u64 values_ptr,");
        ptx.AppendLine("    .param .u64 dense_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<9>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [row_ptr_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [col_idx_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [values_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [dense_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        // row = ctaid * rowsPerBlock + tid / lanesPerRow, lane = tid % lanesPerRow
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    shr.u32 %r2, %r0, {laneShift};");
        ptx.AppendLine($"    and.b32 %r3, %r0, {lanesPerRow - 1};");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {rowsPerBlock}, %r2;");
        if (rows % rowsPerBlock != 0)
        {
            // The last block covers rows past M; M is baked, so this is a constant tail guard.
            ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {rows};");
            ptx.AppendLine("    @%p0 bra DONE;");
        }
        // [start, end) = row_ptr[row], row_ptr[row + 1]
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd5;");
        ptx.AppendLine("    ld.global.nc.u32 %r5, [%rd5];");
        ptx.AppendLine("    ld.global.nc.u32 %r6, [%rd5+4];");
        // Cursors into col_idx and values at start; this lane's 16-byte slice of B and of the output row.
        ptx.AppendLine("    mul.wide.s32 %rd6, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd6;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r3, 16;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd9;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    sub.s32 %r7, %r6, %r5;");
        ptx.AppendLine("    setp.le.s32 %p1, %r7, 0;");
        ptx.AppendLine("    @%p1 bra STORE;");
        ptx.AppendLine("LOOP:");
        ptx.AppendLine("    ld.global.nc.u32 %r8, [%rd7];");
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd8];");
        ptx.AppendLine($"    mul.wide.s32 %rd11, %r8, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd12, %rd10, %rd11;");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f5, %f6, %f7, %f8}, [%rd12];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f4, %f5, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f1, %f4, %f6, %f1;");
        ptx.AppendLine("    fma.rn.f32 %f2, %f4, %f7, %f2;");
        ptx.AppendLine("    fma.rn.f32 %f3, %f4, %f8, %f3;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, 4;");
        ptx.AppendLine("    sub.s32 %r7, %r7, 1;");
        ptx.AppendLine("    setp.gt.s32 %p2, %r7, 0;");
        ptx.AppendLine("    @%p2 bra LOOP;");
        ptx.AppendLine("STORE:");
        ptx.AppendLine($"    mul.wide.u32 %rd13, %r4, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd13;");
        ptx.AppendLine("    add.u64 %rd14, %rd13, %rd9;");
        ptx.AppendLine("    st.global.v4.f32 [%rd14], {%f0, %f1, %f2, %f3};");
        ptx.AppendLine("DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture,
        int rows, int inner, int columns, int blockThreads)
    {
        var rowPointerExtent = new DirectPtxExtent(rows + 1);
        var nonzeroExtent = new DirectPtxExtent(1);
        var denseExtent = new DirectPtxExtent(inner, columns);
        var outputExtent = new DirectPtxExtent(rows, columns);
        return new DirectPtxKernelBlueprint(
            Operation: "csr-spmm-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"lanes-per-row-{columns / 4}-b{blockThreads}-m{rows}-k{inner}-n{columns}",
            Tensors:
            [
                new("row_ptr", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    rowPointerExtent, rowPointerExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("col_idx", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    nonzeroExtent, nonzeroExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("values", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    nonzeroExtent, nonzeroExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("dense", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    denseExtent, denseExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // ptxas allocates 40 (64-bit cursors for col_idx, values and the B slice). 40 x 256 threads x 6
                // blocks = 61,440 of SM86's 65,536 registers, so full occupancy (1,536 threads) still holds.
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: Math.Min(16, 1536 / blockThreads)),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i,:] = sum over p in [row_ptr[i], row_ptr[i+1]) of values[p] * dense[col_idx[p],:]",
                ["mode"] = "forward",
                ["index"] = "int32 row pointers and column indices, zero-based, rows in any column order",
                ["accumulation"] = "fp32 fma.rn in stored nonzero order per output element (bit-identical to NVRTC csr_spmm_vec4)",
                ["empty-row"] = "zeros",
                ["global-input-reads"] = "two row pointers per lane; one index and one value per nonzero per lane (warp-coalesced to one transaction); one fp32x4 slice of B per nonzero per lane",
                ["global-output-writes"] = "one fp32x4 per lane",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["index-bounds"] = "trusted-in-range (row pointers non-decreasing, row_ptr[M] = nonzeros, column indices in [0, K))",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>The exact (M, K, N) cells: the SpMM workload catalog of #379 / #515 (GNN and recommender shapes).</summary>
    internal static bool IsSupportedShape(int rows, int inner, int columns) =>
        (rows, inner, columns) is (2708, 2708, 64) or (19717, 500, 128) or (10000, 10000, 32) or (512, 512, 64);

    /// <summary>No cell is promoted until three clean runs clear the release gate.</summary>
    internal static bool IsPromotedShape(int rows, int inner, int columns) => false;

    private static void Validate(int rows, int inner, int columns, int blockThreads)
    {
        if (!IsSupportedShape(rows, inner, columns))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first CSR SpMM family supports exact (M, K, N) cells (2708,2708,64), (19717,500,128), " +
                "(10000,10000,32) and (512,512,64).");
        if (blockThreads is not (64 or 128 or 256 or 512) || blockThreads < columns / 4)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "CSR SpMM block threads must be 64, 128, 256 or 512 and hold at least one row.");
    }

    private static int Log2(int value)
    {
        int shift = 0;
        while ((1 << shift) < value) shift++;
        return shift;
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        bool exactExtent = contract.ExtentMode == DirectPtxExtentMode.Exact;
        bool allocationOk = exactExtent
            ? view.AllocationByteLength == contract.RequiredBytes
            : view.AllocationByteLength >= contract.RequiredBytes;
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || !allocationOk)
            throw new ArgumentException($"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
