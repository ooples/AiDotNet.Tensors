#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape fused FP32 CSR SpMM and column-bias specialization. Each thread
/// owns four adjacent outputs, initializes them directly from bias, performs
/// the sparse reduction in registers, and emits one final vector store.
/// </summary>
internal sealed class PtxFusedCsrSpmmBiasVec4F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_csr_spmm_bias_vec4_f32_m1024_k1024_n64_nnz16384";
    internal const string ReluEntryPoint = "aidotnet_csr_spmm_bias_relu_vec4_f32_m1024_k1024_n64_nnz16384";
    internal const int Rows = PtxFusedCsrSpmmVec4F32Kernel.Rows;
    internal const int Inner = PtxFusedCsrSpmmVec4F32Kernel.Inner;
    internal const int Columns = PtxFusedCsrSpmmVec4F32Kernel.Columns;
    internal const int NonZeros = PtxFusedCsrSpmmVec4F32Kernel.NonZeros;
    internal const int BlockThreads = PtxFusedCsrSpmmVec4F32Kernel.BlockThreads;
    internal const int ColumnsPerThread = PtxFusedCsrSpmmVec4F32Kernel.ColumnsPerThread;
    internal const int ColumnGroups = Columns / ColumnsPerThread;
    internal const int GridBlocks = Rows * ColumnGroups / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;
    internal bool FuseRelu { get; }

    internal PtxFusedCsrSpmmBiasVec4F32Kernel(DirectPtxRuntime runtime, bool fuseRelu = false)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"CSR SpMM+bias has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");

        FuseRelu = fuseRelu;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, fuseRelu);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, fuseRelu);
        _module = runtime.LoadModule(Ptx);
        string entryPoint = fuseRelu ? ReluEntryPoint : EntryPoint;
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

    internal static bool IsPromotedShape(int rows, int inner, int columns, int nonZeros) => false;

    internal unsafe void Launch(
        DirectPtxTensorView values,
        DirectPtxTensorView columnIndices,
        DirectPtxTensorView rowPointers,
        DirectPtxTensorView dense,
        DirectPtxTensorView bias,
        DirectPtxTensorView output)
    {
        Require(values, Blueprint.Tensors[0], nameof(values));
        Require(columnIndices, Blueprint.Tensors[1], nameof(columnIndices));
        Require(rowPointers, Blueprint.Tensors[2], nameof(rowPointers));
        Require(dense, Blueprint.Tensors[3], nameof(dense));
        Require(bias, Blueprint.Tensors[4], nameof(bias));
        Require(output, Blueprint.Tensors[5], nameof(output));
        if (Overlaps(output, values) || Overlaps(output, columnIndices) ||
            Overlaps(output, rowPointers) || Overlaps(output, dense) || Overlaps(output, bias))
            throw new ArgumentException("CSR SpMM+bias output must be disjoint from every input.", nameof(output));

        IntPtr valuesPointer = values.Pointer;
        IntPtr columnIndicesPointer = columnIndices.Pointer;
        IntPtr rowPointersPointer = rowPointers.Pointer;
        IntPtr densePointer = dense.Pointer;
        IntPtr biasPointer = bias.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &valuesPointer;
        arguments[1] = &columnIndicesPointer;
        arguments[2] = &rowPointersPointer;
        arguments[3] = &densePointer;
        arguments[4] = &biasPointer;
        arguments[5] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        bool fuseRelu = false)
    {
        var values = new DirectPtxExtent(NonZeros);
        var rowPointers = new DirectPtxExtent(Rows + 1);
        var dense = new DirectPtxExtent(Inner, Columns);
        var bias = new DirectPtxExtent(Columns);
        var output = new DirectPtxExtent(Rows, Columns);
        return new DirectPtxKernelBlueprint(
            Operation: fuseRelu ? "csr-spmm-bias-relu-vec4-f32" : "csr-spmm-bias-vec4-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "m1024-k1024-n64-nnz16384-row-major",
            Tensors:
            [
                new("values", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.CsrValues,
                    values, values, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("column-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrColumnIndices,
                    values, values, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("row-pointers", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CsrRowPointers,
                    rowPointers, rowPointers, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("dense", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    dense, dense, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("bias", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    bias, bias, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(48, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = fuseRelu ? "C=relu(A(csr)*B+bias)" : "C=A(csr)*B+bias",
                ["csr-index-base"] = "zero",
                ["csr-index-width"] = "int32",
                ["reduction-order"] = "bias-then-stored-row-order",
                ["accumulator"] = "fp32-fma",
                ["dense-layout"] = "row-major-contiguous",
                ["bias-layout"] = "contiguous-column-vector",
                ["epilogue"] = fuseRelu ? "relu" : "identity",
                ["output-columns-per-thread"] = ColumnsPerThread.ToString(),
                ["intermediate-global-bytes"] = "0",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor, bool fuseRelu = false)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {(fuseRelu ? ReluEntryPoint : EntryPoint)}(");
        ptx.AppendLine("    .param .u64 values_ptr,");
        ptx.AppendLine("    .param .u64 column_indices_ptr,");
        ptx.AppendLine("    .param .u64 row_pointers_ptr,");
        ptx.AppendLine("    .param .u64 dense_ptr,");
        ptx.AppendLine("    .param .u64 bias_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<22>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [values_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [column_indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [row_pointers_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [dense_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd4, [bias_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd5, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {ColumnGroups - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r4, 2;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd6;");
        ptx.AppendLine("    ld.global.u32 %r6, [%rd7];");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd7+4];");
        ptx.AppendLine("    mov.u32 %r8, %r6;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd4, %rd8;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd9];");
        ptx.AppendLine("CSR_BIAS_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r8, %r7;");
        ptx.AppendLine("    @%p0 bra CSR_BIAS_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd0, %rd10;");
        ptx.AppendLine("    add.u64 %rd12, %rd1, %rd10;");
        ptx.AppendLine("    ld.global.f32 %f4, [%rd11];");
        ptx.AppendLine("    ld.global.u32 %r9, [%rd12];");
        ptx.AppendLine("    shl.b32 %r10, %r9, 6;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd13, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd14, %rd3, %rd13;");
        ptx.AppendLine("    ld.global.v4.f32 {%f5, %f6, %f7, %f8}, [%rd14];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f4, %f5, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f1, %f4, %f6, %f1;");
        ptx.AppendLine("    fma.rn.f32 %f2, %f4, %f7, %f2;");
        ptx.AppendLine("    fma.rn.f32 %f3, %f4, %f8, %f3;");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine("    bra CSR_BIAS_LOOP;");
        ptx.AppendLine("CSR_BIAS_DONE:");
        if (fuseRelu)
        {
            ptx.AppendLine("    max.f32 %f0, %f0, 0f00000000;");
            ptx.AppendLine("    max.f32 %f1, %f1, 0f00000000;");
            ptx.AppendLine("    max.f32 %f2, %f2, 0f00000000;");
            ptx.AppendLine("    max.f32 %f3, %f3, 0f00000000;");
        }
        ptx.AppendLine("    shl.b32 %r11, %r3, 6;");
        ptx.AppendLine("    add.u32 %r11, %r11, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd15, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd16, %rd5, %rd15;");
        ptx.AppendLine("    st.global.v4.f32 [%rd16], {%f0, %f1, %f2, %f3};");
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
