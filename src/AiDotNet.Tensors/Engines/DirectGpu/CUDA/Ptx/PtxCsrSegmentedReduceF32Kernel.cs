using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>The reduction a <see cref="PtxCsrSegmentedReduceF32Kernel"/> applies across a CSR row's gathered rows.</summary>
internal enum CsrSegmentedReduction
{
    Max,
    Min
}

/// <summary>
/// Exact-shape FP32 CSR segmented max / min (issue #852): <c>output[i,j] = max (or min) over p in
/// [row_ptr[i], row_ptr[i+1]) of input[col_idx[p], j]</c>, with <c>input</c> [K,N] and <c>output</c> [M,N] dense
/// row-major, and 0 for an empty row. It ports the NVRTC <c>csr_segmented_max</c> / <c>csr_segmented_min</c> kernels
/// behind <c>CudaBackend.CsrSegmentedMax</c> / <c>CsrSegmentedMin</c>, including their index ABI: the column indices and
/// row pointers are FP32 buffers holding integral values, truncated with <c>cvt.rzi</c> as the C <c>(int)</c> cast does.
/// <para>
/// N / 4 adjacent lanes own one output row, 4 columns each, as in <see cref="PtxCsrSpMMF32Kernel"/>. (The NVRTC kernels
/// launch one thread per output element over an M x ceil(N / 256) grid, so a row of 64 columns leaves 192 of each
/// block's 256 threads idle.) Each lane walks its row's nonzeros in stored order, reads the column index once per
/// nonzero (lanes of a row read the same address, so the warp issues one transaction) and streams the matching 16-byte
/// slice of <c>input</c> through the read-only cache. The NVRTC kernels are compiled with <c>--use_fast_math</c>
/// (<c>CudaBackend.CompileKernelModule</c>'s default), under which <c>fmaxf</c> / <c>fminf</c> and the <c>(int)</c> casts
/// become <c>max.ftz.f32</c> / <c>min.ftz.f32</c> and <c>cvt.rzi.ftz.s32.f32</c>: subnormal inputs flush to zero. This
/// emits exactly those instructions, starting from the same literal, <c>∓3.402823e+38f</c> (0xFF7FFFFD / 0x7F7FFFFD,
/// two ulps inside FLT_MAX), so NaN, signed-zero and subnormal handling match. The output row is written once with a
/// 16-byte store.
/// </para>
/// <para>
/// M, K and N are baked into the module; only the four tensor pointers reach the launch ABI. The nonzero count is data
/// (row_ptr[M]), not shape: the column-index tensor is an at-least view checked once against it on the host. There is
/// no shared memory, no local memory, no global intermediate and no temporary allocation. The specialization is
/// experimental and has not run on hardware yet: it fails closed outside SM86 and is never dispatched until three clean
/// promotion runs clear the gate.
/// </para>
/// </summary>
internal sealed class PtxCsrSegmentedReduceF32Kernel : IDisposable
{
    internal const string MaxEntryPoint = "aidotnet_csr_segmented_max_f32";
    internal const string MinEntryPoint = "aidotnet_csr_segmented_min_f32";
    internal const int DefaultBlockThreads = 256;

    // The NVRTC kernels' starting values, -3.402823e+38f and 3.402823e+38f, as PTX hexadecimal float literals.
    internal const string MaxInitialLiteral = "0fFF7FFFFD";
    internal const string MinInitialLiteral = "0f7F7FFFFD";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal CsrSegmentedReduction Reduction { get; }
    internal int Rows { get; }
    internal int Inner { get; }
    internal int Columns { get; }
    internal int BlockThreads { get; }
    internal int LanesPerRow => Columns / 4;
    internal int RowsPerBlock => BlockThreads / LanesPerRow;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCsrSegmentedReduceF32Kernel(DirectPtxRuntime runtime, CsrSegmentedReduction reduction, int rows, int inner,
        int columns, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasTargetedCsrSegmentedReduce(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException("The FP32 CSR segmented max / min specialization is admitted only on SM86.");
        Validate(reduction, rows, inner, columns, blockThreads);
        Reduction = reduction;
        Rows = rows;
        Inner = inner;
        Columns = columns;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, reduction, rows, inner, columns, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, reduction, rows, inner, columns, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPointFor(reduction), out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPointFor(reduction), info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module);
    }

    internal uint GridBlocks => checked((uint)((Rows + RowsPerBlock - 1) / RowsPerBlock));

    internal static string EntryPointFor(CsrSegmentedReduction reduction) =>
        reduction == CsrSegmentedReduction.Max ? MaxEntryPoint : MinEntryPoint;

    /// <summary>
    /// Launches on the runtime's stream, in the NVRTC kernels' argument order. <paramref name="nonzeros"/> is
    /// row_ptr[M].
    /// </summary>
    internal unsafe void Launch(
        DirectPtxTensorView columnIndices,
        DirectPtxTensorView rowPointers,
        DirectPtxTensorView input,
        DirectPtxTensorView output,
        int nonzeros)
    {
        Require(columnIndices, Blueprint.Tensors[0], nameof(columnIndices));
        Require(rowPointers, Blueprint.Tensors[1], nameof(rowPointers));
        Require(input, Blueprint.Tensors[2], nameof(input));
        Require(output, Blueprint.Tensors[3], nameof(output));
        if (nonzeros < 0 || columnIndices.AllocationByteLength < checked((nuint)nonzeros * sizeof(float)))
            throw new ArgumentException($"The CSR column-index tensor must hold {nonzeros} nonzeros.", nameof(nonzeros));

        IntPtr indexPointer = columnIndices.Pointer, rowPointer = rowPointers.Pointer;
        IntPtr inputPointer = input.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &indexPointer;
        arguments[1] = &rowPointer;
        arguments[2] = &inputPointer;
        arguments[3] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, checked((uint)BlockThreads), 1, 1, 0, arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, CsrSegmentedReduction reduction, int rows, int inner,
        int columns, int blockThreads = DefaultBlockThreads)
    {
        Validate(reduction, rows, inner, columns, blockThreads);
        int lanesPerRow = columns / 4;
        int laneShift = Log2(lanesPerRow);
        int rowsPerBlock = blockThreads / lanesPerRow;
        int rowBytes = columns * sizeof(float);
        bool max = reduction == CsrSegmentedReduction.Max;
        string op = max ? "max.ftz.f32" : "min.ftz.f32";
        string initial = max ? MaxInitialLiteral : MinInitialLiteral;
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape rows={rows} inner={inner} columns={columns} block={blockThreads} " +
            $"strategy=lanes-per-row-{lanesPerRow} op=csr-segmented-{(max ? "max" : "min")} order=stored index=fp32-truncated");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPointFor(reduction)}(");
        ptx.AppendLine("    .param .u64 col_idx_ptr,");
        ptx.AppendLine("    .param .u64 row_ptr_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<15>;");
        ptx.AppendLine("    .reg .f32 %f<11>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [col_idx_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [row_ptr_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
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
        // [start, end) = (int)row_ptr[row], (int)row_ptr[row + 1] -- FP32 row pointers, truncated as the C cast does.
        ptx.AppendLine("    mul.wide.u32 %rd4, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f8, [%rd4];");
        ptx.AppendLine("    ld.global.nc.f32 %f9, [%rd4+4];");
        ptx.AppendLine("    cvt.rzi.ftz.s32.f32 %r5, %f8;");
        ptx.AppendLine("    cvt.rzi.ftz.s32.f32 %r6, %f9;");
        // This lane's 16-byte slice of the output row, and of input row 0.
        ptx.AppendLine("    mul.wide.u32 %rd5, %r3, 16;");
        ptx.AppendLine($"    mul.wide.u32 %rd6, %r4, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd6;");
        ptx.AppendLine("    add.u64 %rd6, %rd6, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd5;");
        // An empty row (start >= end) writes zeros, the NVRTC kernels' contract.
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    setp.ge.s32 %p1, %r5, %r6;");
        ptx.AppendLine("    @%p1 bra STORE;");
        ptx.AppendLine($"    mov.f32 %f0, {initial};");
        ptx.AppendLine($"    mov.f32 %f1, {initial};");
        ptx.AppendLine($"    mov.f32 %f2, {initial};");
        ptx.AppendLine($"    mov.f32 %f3, {initial};");
        ptx.AppendLine("    mul.wide.s32 %rd8, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd8;");
        ptx.AppendLine("    sub.s32 %r7, %r6, %r5;");
        ptx.AppendLine("LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f10, [%rd8];");
        ptx.AppendLine("    cvt.rzi.ftz.s32.f32 %r8, %f10;");
        ptx.AppendLine($"    mul.wide.s32 %rd9, %r8, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd9, %rd7, %rd9;");
        ptx.AppendLine("    ld.global.nc.v4.f32 {%f4, %f5, %f6, %f7}, [%rd9];");
        ptx.AppendLine($"    {op} %f0, %f0, %f4;");
        ptx.AppendLine($"    {op} %f1, %f1, %f5;");
        ptx.AppendLine($"    {op} %f2, %f2, %f6;");
        ptx.AppendLine($"    {op} %f3, %f3, %f7;");
        ptx.AppendLine("    add.u64 %rd8, %rd8, 4;");
        ptx.AppendLine("    sub.s32 %r7, %r7, 1;");
        ptx.AppendLine("    setp.gt.s32 %p2, %r7, 0;");
        ptx.AppendLine("    @%p2 bra LOOP;");
        ptx.AppendLine("STORE:");
        ptx.AppendLine("    st.global.v4.f32 [%rd6], {%f0, %f1, %f2, %f3};");
        ptx.AppendLine("DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture,
        CsrSegmentedReduction reduction, int rows, int inner, int columns, int blockThreads)
    {
        var nonzeroExtent = new DirectPtxExtent(1);
        var rowPointerExtent = new DirectPtxExtent(rows + 1);
        var inputExtent = new DirectPtxExtent(inner, columns);
        var outputExtent = new DirectPtxExtent(rows, columns);
        string name = reduction == CsrSegmentedReduction.Max ? "max" : "min";
        return new DirectPtxKernelBlueprint(
            Operation: $"csr-segmented-{name}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"lanes-per-row-{columns / 4}-b{blockThreads}-m{rows}-k{inner}-n{columns}",
            Tensors:
            [
                new("col_idx", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    nonzeroExtent, nonzeroExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("row_ptr", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    rowPointerExtent, rowPointerExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                // ptxas 13.3.33 allocates RegistersUsed for both reductions at every cell (sm_86, no spills, no local
                // bytes). At 256 threads x 6 blocks that is well inside SM86's 65,536 registers.
                MaxRegistersPerThread: RegistersUsed,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: Math.Min(16, 1536 / blockThreads)),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = $"output[i,j] = {name} over p in [row_ptr[i], row_ptr[i+1]) of input[col_idx[p], j]",
                ["mode"] = "forward",
                ["index"] = "fp32 row pointers and column indices holding integral values, truncated toward zero (cvt.rzi.ftz); zero-based",
                ["reduction"] = $"fp32 {name}.ftz.f32 in stored nonzero order from {(reduction == CsrSegmentedReduction.Max ? "-" : "")}3.402823e+38f (the NVRTC csr_segmented_{name} contract under --use_fast_math: NaN, signed-zero and flush-to-zero handling)",
                ["empty-row"] = "zeros",
                ["global-input-reads"] = "two row pointers per lane; one column index per nonzero per lane (warp-coalesced to one transaction); one fp32x4 slice of input per nonzero per lane",
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

    /// <summary>Registers ptxas 13.3.33 allocates on sm_86 for every admitted cell (see the resource budget).</summary>
    internal const int RegistersUsed = 34;

    /// <summary>The exact (M, K, N) cells: the CSR SpMM family's GNN and recommender cells (#379 / #515).</summary>
    internal static bool IsSupportedShape(int rows, int inner, int columns) =>
        PtxCsrSpMMF32Kernel.IsSupportedShape(rows, inner, columns);

    /// <summary>No cell is promoted until three clean runs clear the release gate.</summary>
    internal static bool IsPromotedShape(int rows, int inner, int columns) => false;

    private static void Validate(CsrSegmentedReduction reduction, int rows, int inner, int columns, int blockThreads)
    {
        if (reduction is not (CsrSegmentedReduction.Max or CsrSegmentedReduction.Min))
            throw new ArgumentOutOfRangeException(nameof(reduction));
        if (!IsSupportedShape(rows, inner, columns))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The CSR segmented max / min family supports exact (M, K, N) cells (2708,2708,64), (19717,500,128), " +
                "(10000,10000,32) and (512,512,64).");
        if (blockThreads is not (64 or 128 or 256 or 512) || blockThreads < columns / 4)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "CSR segmented max / min block threads must be 64, 128, 256 or 512 and hold at least one row.");
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
