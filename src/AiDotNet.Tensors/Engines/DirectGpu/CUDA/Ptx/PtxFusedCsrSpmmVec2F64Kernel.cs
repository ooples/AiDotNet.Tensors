#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP64 CSR SpMM specialization. Each thread owns two adjacent
/// output columns and retains both ordered FP64 reductions in registers.
/// </summary>
internal sealed class PtxFusedCsrSpmmVec2F64Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_csr_spmm_vec2_f64_m1024_k1024_n64_nnz16384";
    internal const int Rows = 1024;
    internal const int Inner = 1024;
    internal const int Columns = 64;
    internal const int NonZeros = 16384;
    internal const int BlockThreads = 256;
    internal const int ColumnsPerThread = 2;
    internal const int ColumnGroups = Columns / ColumnsPerThread;
    internal const int GridBlocks = Rows * ColumnGroups / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;

    internal PtxFusedCsrSpmmVec2F64Kernel(DirectPtxRuntime runtime)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"FP64 CSR SpMM has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");

        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int rows, int inner, int columns, int nonZeros) =>
        rows == Rows && inner == Inner && columns == Columns && nonZeros == NonZeros;

    internal static bool IsPromotedShape(int rows, int inner, int columns, int nonZeros) => false;

    internal unsafe void Launch(
        DirectPtxTensorView values,
        DirectPtxTensorView columnIndices,
        DirectPtxTensorView rowPointers,
        DirectPtxTensorView dense,
        DirectPtxTensorView output)
    {
        Require(values, Blueprint.Tensors[0], nameof(values));
        Require(columnIndices, Blueprint.Tensors[1], nameof(columnIndices));
        Require(rowPointers, Blueprint.Tensors[2], nameof(rowPointers));
        Require(dense, Blueprint.Tensors[3], nameof(dense));
        Require(output, Blueprint.Tensors[4], nameof(output));
        if (Overlaps(output, values) || Overlaps(output, columnIndices) ||
            Overlaps(output, rowPointers) || Overlaps(output, dense))
            throw new ArgumentException("FP64 CSR SpMM output must be disjoint from every input.", nameof(output));

        IntPtr valuesPointer = values.Pointer;
        IntPtr columnIndicesPointer = columnIndices.Pointer;
        IntPtr rowPointersPointer = rowPointers.Pointer;
        IntPtr densePointer = dense.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &valuesPointer;
        arguments[1] = &columnIndicesPointer;
        arguments[2] = &rowPointersPointer;
        arguments[3] = &densePointer;
        arguments[4] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var values = new DirectPtxExtent(NonZeros);
        var rowPointers = new DirectPtxExtent(Rows + 1);
        var dense = new DirectPtxExtent(Inner, Columns);
        var output = new DirectPtxExtent(Rows, Columns);
        return new DirectPtxKernelBlueprint(
            Operation: "csr-spmm-vec2-f64",
            Version: 1,
            Architecture: architecture,
            Variant: "m1024-k1024-n64-nnz16384-row-major",
            Tensors:
            [
                new("values", DirectPtxPhysicalType.Float64, DirectPtxPhysicalLayout.CsrValues,
                    values, values, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("column-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrColumnIndices,
                    values, values, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("row-pointers", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrRowPointers,
                    rowPointers, rowPointers, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("dense", DirectPtxPhysicalType.Float64, DirectPtxPhysicalLayout.RowMajor2D,
                    dense, dense, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float64, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(48, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "C=A(csr)*B",
                ["csr-index-base"] = "zero",
                ["csr-index-width"] = "int32",
                ["reduction-order"] = "stored-row-order",
                ["accumulator"] = "fp64-fma",
                ["dense-layout"] = "row-major-contiguous",
                ["output-layout"] = "row-major-contiguous",
                ["output-columns-per-thread"] = ColumnsPerThread.ToString(),
                ["intermediate-global-bytes"] = "0",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 values_ptr,");
        ptx.AppendLine("    .param .u64 column_indices_ptr,");
        ptx.AppendLine("    .param .u64 row_pointers_ptr,");
        ptx.AppendLine("    .param .u64 dense_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f64 %fd<6>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [values_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [column_indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [row_pointers_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [dense_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 5;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {ColumnGroups - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r4, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd5;");
        ptx.AppendLine("    ld.global.u32 %r6, [%rd6];");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd6+4];");
        ptx.AppendLine("    mov.u32 %r8, %r6;");
        ptx.AppendLine("    mov.f64 %fd0, 0d0000000000000000;");
        ptx.AppendLine("    mov.f64 %fd1, 0d0000000000000000;");
        ptx.AppendLine("CSR_F64_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r8, %r7;");
        ptx.AppendLine("    @%p0 bra CSR_F64_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r8, 8;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd7;");
        ptx.AppendLine("    add.u64 %rd10, %rd1, %rd8;");
        ptx.AppendLine("    ld.global.f64 %fd2, [%rd9];");
        ptx.AppendLine("    ld.global.u32 %r9, [%rd10];");
        ptx.AppendLine("    shl.b32 %r10, %r9, 6;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r10, 8;");
        ptx.AppendLine("    add.u64 %rd12, %rd3, %rd11;");
        ptx.AppendLine("    ld.global.v2.f64 {%fd3, %fd4}, [%rd12];");
        ptx.AppendLine("    fma.rn.f64 %fd0, %fd2, %fd3, %fd0;");
        ptx.AppendLine("    fma.rn.f64 %fd1, %fd2, %fd4, %fd1;");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine("    bra CSR_F64_LOOP;");
        ptx.AppendLine("CSR_F64_DONE:");
        ptx.AppendLine("    shl.b32 %r11, %r3, 6;");
        ptx.AppendLine("    add.u32 %r11, %r11, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r11, 8;");
        ptx.AppendLine("    add.u64 %rd14, %rd4, %rd13;");
        ptx.AppendLine("    st.global.v2.f64 [%rd14], {%fd0, %fd1};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer;
        nuint rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength);
        nuint rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
