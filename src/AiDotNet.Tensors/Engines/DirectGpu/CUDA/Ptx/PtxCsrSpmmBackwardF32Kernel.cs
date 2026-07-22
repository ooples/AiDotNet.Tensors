#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxCsrBackwardTarget
{
    DenseB,
    Values
}

/// <summary>
/// Exact CSR SpMM gradients. Dense-B uses fixed row/CSR traversal and one
/// overwrite; values uses one sampled dot product per stored CSR entry.
/// </summary>
internal sealed class PtxCsrSpmmBackwardF32Kernel : IDisposable
{
    internal const string DenseBEntryPoint = "aidotnet_csr_spmm_backward_b_vec4_f32_m1024_k1024_n64_nnz16384";
    internal const string ValuesEntryPoint = "aidotnet_csr_spmm_backward_values_f32_m1024_k1024_n64_nnz16384";
    internal const int Rows = 1024;
    internal const int Inner = 1024;
    internal const int Columns = 64;
    internal const int NonZeros = 16384;
    internal const int BlockThreads = 256;
    internal const int ColumnsPerThread = 4;
    internal const int ColumnGroups = Columns / ColumnsPerThread;
    internal const int DenseBGridBlocks = Inner * ColumnGroups / BlockThreads;
    internal const int ValuesGridBlocks = NonZeros / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxCsrBackwardTarget Target { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCsrSpmmBackwardF32Kernel(DirectPtxRuntime runtime, DirectPtxCsrBackwardTarget target)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"CSR SpMM backward has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Target = target;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, target);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, target);
        _module = runtime.LoadModule(Ptx);
        string entryPoint = target == DirectPtxCsrBackwardTarget.DenseB
            ? DenseBEntryPoint : ValuesEntryPoint;
        _function = _module.GetFunction(entryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(entryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int rows, int inner, int columns, int nonZeros) =>
        rows == Rows && inner == Inner && columns == Columns && nonZeros == NonZeros;

    internal unsafe void Launch(
        DirectPtxTensorView valuesOrColumns,
        DirectPtxTensorView columnsOrRows,
        DirectPtxTensorView rowsOrDense,
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView output)
    {
        Require(valuesOrColumns, Blueprint.Tensors[0], nameof(valuesOrColumns));
        Require(columnsOrRows, Blueprint.Tensors[1], nameof(columnsOrRows));
        Require(rowsOrDense, Blueprint.Tensors[2], nameof(rowsOrDense));
        Require(gradOutput, Blueprint.Tensors[3], nameof(gradOutput));
        Require(output, Blueprint.Tensors[4], nameof(output));
        if (Overlaps(output, valuesOrColumns) || Overlaps(output, columnsOrRows) ||
            Overlaps(output, rowsOrDense) || Overlaps(output, gradOutput))
            throw new ArgumentException("CSR backward output must be disjoint from every input.", nameof(output));
        IntPtr p0 = valuesOrColumns.Pointer;
        IntPtr p1 = columnsOrRows.Pointer;
        IntPtr p2 = rowsOrDense.Pointer;
        IntPtr p3 = gradOutput.Pointer;
        IntPtr p4 = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        arguments[4] = &p4;
        int grid = Target == DirectPtxCsrBackwardTarget.DenseB
            ? DenseBGridBlocks : ValuesGridBlocks;
        _module.Launch(_function, (uint)grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxCsrBackwardTarget target)
    {
        var nonZeros = new DirectPtxExtent(NonZeros);
        var rowPointers = new DirectPtxExtent(Rows + 1);
        var dense = new DirectPtxExtent(Inner, Columns);
        var gradOutput = new DirectPtxExtent(Rows, Columns);
        IReadOnlyList<DirectPtxTensorContract> tensors = target == DirectPtxCsrBackwardTarget.DenseB
            ?
            [
                new("values", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.CsrValues,
                    nonZeros, nonZeros, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("column-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrColumnIndices,
                    nonZeros, nonZeros, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("row-pointers", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrRowPointers,
                    rowPointers, rowPointers, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    dense, dense, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ]
            :
            [
                new("column-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrColumnIndices,
                    nonZeros, nonZeros, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("row-pointers", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrRowPointers,
                    rowPointers, rowPointers, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("dense-b", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    dense, dense, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    gradOutput, gradOutput, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("grad-values", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.CsrValues,
                    nonZeros, nonZeros, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ];
        return new DirectPtxKernelBlueprint(
            Operation: target == DirectPtxCsrBackwardTarget.DenseB
                ? "csr-spmm-backward-b-deterministic-vec4-f32"
                : "csr-spmm-backward-values-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "m1024-k1024-n64-nnz16384-row-major",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(56, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["target"] = target.ToString(),
                ["reduction-order"] = target == DirectPtxCsrBackwardTarget.DenseB
                    ? "ascending-row-then-stored-entry" : "ascending-output-column",
                ["output-initialization"] = "fused-register-initialization",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxCsrBackwardTarget target)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        return target == DirectPtxCsrBackwardTarget.DenseB
            ? EmitDenseB(ccMajor, ccMinor) : EmitValues(ccMajor, ccMinor);
    }

    private static string EmitDenseB(int ccMajor, int ccMinor)
    {
        var ptx = Header(ccMajor, ccMinor, DenseBEntryPoint,
            "values_ptr", "column_indices_ptr", "row_pointers_ptr", "grad_output_ptr", "grad_b_ptr");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<24>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        LoadParams(ptx, "values_ptr", "column_indices_ptr", "row_pointers_ptr", "grad_output_ptr", "grad_b_ptr");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {ColumnGroups - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r4, 2;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("ROW_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r6, {Rows};");
        ptx.AppendLine("    @%p0 bra DB_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd5;");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd6];");
        ptx.AppendLine("    ld.global.u32 %r8, [%rd6+4];");
        ptx.AppendLine("    mov.u32 %r9, %r7;");
        ptx.AppendLine("ENTRY_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p1, %r9, %r8;");
        ptx.AppendLine("    @%p1 bra NEXT_ROW;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine("    ld.global.u32 %r10, [%rd8];");
        ptx.AppendLine("    setp.ne.u32 %p2, %r10, %r3;");
        ptx.AppendLine("    @%p2 bra NEXT_ENTRY;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd7;");
        ptx.AppendLine("    ld.global.f32 %f4, [%rd9];");
        ptx.AppendLine("    shl.b32 %r11, %r6, 6;");
        ptx.AppendLine("    add.u32 %r11, %r11, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
        ptx.AppendLine("    ld.global.v4.f32 {%f5, %f6, %f7, %f8}, [%rd11];");
        for (int lane = 0; lane < 4; lane++)
            ptx.AppendLine($"    fma.rn.f32 %f{lane}, %f4, %f{lane + 5}, %f{lane};");
        ptx.AppendLine("NEXT_ENTRY:");
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine("    bra ENTRY_LOOP;");
        ptx.AppendLine("NEXT_ROW:");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra ROW_LOOP;");
        ptx.AppendLine("DB_DONE:");
        ptx.AppendLine("    shl.b32 %r12, %r3, 6;");
        ptx.AppendLine("    add.u32 %r12, %r12, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd12;");
        ptx.AppendLine("    st.global.v4.f32 [%rd13], {%f0, %f1, %f2, %f3};");
        Footer(ptx);
        return ptx.ToString();
    }

    private static string EmitValues(int ccMajor, int ccMinor)
    {
        var ptx = Header(ccMajor, ccMinor, ValuesEntryPoint,
            "column_indices_ptr", "row_pointers_ptr", "dense_b_ptr", "grad_output_ptr", "grad_values_ptr");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        LoadParams(ptx, "column_indices_ptr", "row_pointers_ptr", "dense_b_ptr", "grad_output_ptr", "grad_values_ptr");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    mov.u32 %r3, 0;");
        ptx.AppendLine("FIND_ROW:");
        ptx.AppendLine("    add.u32 %r4, %r3, 1;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.u32 %r5, [%rd6];");
        ptx.AppendLine("    setp.gt.u32 %p0, %r5, %r2;");
        ptx.AppendLine("    @%p0 bra ROW_FOUND;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra FIND_ROW;");
        ptx.AppendLine("ROW_FOUND:");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd0, %rd7;");
        ptx.AppendLine("    ld.global.u32 %r6, [%rd8];");
        ptx.AppendLine("    shl.b32 %r7, %r3, 6;");
        ptx.AppendLine("    shl.b32 %r8, %r6, 6;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r7, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
        ptx.AppendLine("    add.u64 %rd12, %rd2, %rd10;");
        ptx.AppendLine("    mov.f32 %f8, 0f00000000;");
        for (int byteOffset = 0; byteOffset < Columns * sizeof(float); byteOffset += 16)
        {
            ptx.AppendLine($"    ld.global.v4.f32 {{%f0, %f1, %f2, %f3}}, [%rd11+{byteOffset}];");
            ptx.AppendLine($"    ld.global.v4.f32 {{%f4, %f5, %f6, %f7}}, [%rd12+{byteOffset}];");
            for (int lane = 0; lane < 4; lane++)
                ptx.AppendLine($"    fma.rn.f32 %f8, %f{lane}, %f{lane + 4}, %f8;");
        }
        ptx.AppendLine("    add.u64 %rd13, %rd4, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd13], %f8;");
        Footer(ptx);
        return ptx.ToString();
    }

    private static StringBuilder Header(int major, int minor, string entry, params string[] parameters)
    {
        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{major}{minor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {entry}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        return ptx;
    }

    private static void LoadParams(StringBuilder ptx, params string[] parameters)
    {
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{parameters[i]}];");
    }

    private static void Footer(StringBuilder ptx)
    {
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
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
