#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSparseEngineOperation
{
    SpMV,
    SpMVTranspose,
    SpMM,
    SpSpMM,
    AddSparseDense,
    MultiplySparseDense,
    SparseGather,
    SparseScatter,
    SparseScatterAdd,
    SparseToDense,
    DenseToSparse,
    Coalesce,
    SparseTranspose,
    SparseMatMul,
    SparseMatMulPatternPreserving,
    SparseAddMM,
    SparseSampledAddMM,
    SparseSpGeMM,
    SparseSum,
    SparseMean,
    SparseSoftmax,
    SparseLogSoftmax
}

/// <summary>
/// Direct-PTX implementation family for the host-facing ISparseEngine contract.
/// The first specialization is deliberately exact: 256x256 matrices, 1024
/// stored entries, and 32-column dense products. Every interface operation has
/// a distinct module and pointer-only ABI so evidence cannot be borrowed from
/// a neighboring sparse primitive.
/// </summary>
internal sealed class PtxSparseEngineF32Kernel : IDisposable
{
    internal const int Rows = 256;
    internal const int Inner = 256;
    internal const int Columns = 32;
    internal const int NonZeros = 1024;
    internal const int DenseElements = Rows * Inner;
    internal const int ProductElements = Rows * Columns;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;
    private static readonly DirectPtxKernelBlueprint[] AdmissionBlueprints = CreateAdmissionBlueprints();

    internal DirectPtxSparseEngineOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSparseEngineF32Kernel(DirectPtxRuntime runtime, DirectPtxSparseEngineOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"ISparseEngine has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Operation = operation;
        EntryPoint = GetEntryPoint(operation);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, operation);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, operation);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static DirectPtxKernelBlueprint GetAdmissionBlueprint(DirectPtxSparseEngineOperation operation) =>
        AdmissionBlueprints[(int)operation];

    internal static bool SupportsCanonicalShape(int rows, int columns, int nonZeros) =>
        rows == Rows && columns == Inner && nonZeros == NonZeros;

    internal unsafe void Launch(ReadOnlySpan<DirectPtxTensorView> tensors)
    {
        if (tensors.Length != Blueprint.Tensors.Count)
            throw new ArgumentException("The ISparseEngine PTX ABI has the wrong arity.", nameof(tensors));
        for (int i = 0; i < tensors.Length; i++) Require(tensors[i], Blueprint.Tensors[i], $"tensor{i}");
        for (int i = 0; i < tensors.Length; i++)
        {
            if ((Blueprint.Tensors[i].Access & DirectPtxTensorAccess.Write) == 0) continue;
            for (int j = 0; j < tensors.Length; j++)
                if (i != j && Overlaps(tensors[i], tensors[j]))
                    throw new ArgumentException("ISparseEngine PTX outputs must be disjoint from every other ABI tensor.");
        }
        IntPtr* pointers = stackalloc IntPtr[tensors.Length];
        void** arguments = stackalloc void*[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
        {
            pointers[i] = tensors[i].Pointer;
            arguments[i] = &pointers[i];
        }
        int work = GetWorkItems(Operation);
        uint grid = checked((uint)((work + BlockThreads - 1) / BlockThreads));
        _module.Launch(_function, grid, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxSparseEngineOperation operation)
    {
        var values = new DirectPtxExtent(NonZeros);
        var rowPointers = new DirectPtxExtent(Rows + 1);
        var vectorRows = new DirectPtxExtent(Rows);
        var vectorInner = new DirectPtxExtent(Inner);
        var dense = new DirectPtxExtent(Rows, Inner);
        var product = new DirectPtxExtent(Rows, Columns);
        var rhs = new DirectPtxExtent(Inner, Columns);
        var scalar = new DirectPtxExtent(1);
        var flagsDense = new DirectPtxExtent(DenseElements);
        DirectPtxTensorContract F(string name, DirectPtxPhysicalLayout layout, DirectPtxExtent extent,
            DirectPtxTensorAccess access) => Exact(name, DirectPtxPhysicalType.Float32, layout, extent, access);
        DirectPtxTensorContract I(string name, DirectPtxPhysicalLayout layout, DirectPtxExtent extent,
            DirectPtxTensorAccess access = DirectPtxTensorAccess.Read) =>
            Exact(name, DirectPtxPhysicalType.Int32, layout, extent, access);
        DirectPtxTensorContract CsrValues(string prefix) => F(prefix + "-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Read);
        DirectPtxTensorContract CsrColumns(string prefix) => I(prefix + "-columns", DirectPtxPhysicalLayout.CsrColumnIndices, values);
        DirectPtxTensorContract CsrRows(string prefix) => I(prefix + "-row-pointers", DirectPtxPhysicalLayout.CsrRowPointers, rowPointers);
        DirectPtxTensorContract CooRows(string prefix, DirectPtxTensorAccess access = DirectPtxTensorAccess.Read) =>
            I(prefix + "-rows", DirectPtxPhysicalLayout.CooRowIndices, values, access);
        DirectPtxTensorContract CooColumns(string prefix, DirectPtxTensorAccess access = DirectPtxTensorAccess.Read) =>
            I(prefix + "-columns", DirectPtxPhysicalLayout.CooColumnIndices, values, access);

        IReadOnlyList<DirectPtxTensorContract> tensors = operation switch
        {
            DirectPtxSparseEngineOperation.SpMV =>
                [CsrValues("a"), CsrColumns("a"), CsrRows("a"),
                 F("x", DirectPtxPhysicalLayout.Vector, vectorInner, DirectPtxTensorAccess.Read),
                 F("y", DirectPtxPhysicalLayout.Vector, vectorRows, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SpMVTranspose =>
                [CsrValues("a"), CsrColumns("a"), CsrRows("a"),
                 F("x", DirectPtxPhysicalLayout.Vector, vectorRows, DirectPtxTensorAccess.Read),
                 F("y", DirectPtxPhysicalLayout.Vector, vectorInner, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SpMM or DirectPtxSparseEngineOperation.SparseMatMul or
            DirectPtxSparseEngineOperation.SparseMatMulPatternPreserving =>
                [CsrValues("a"), CsrColumns("a"), CsrRows("a"),
                 F("b", DirectPtxPhysicalLayout.RowMajor2D, rhs, DirectPtxTensorAccess.Read),
                 F("output", DirectPtxPhysicalLayout.RowMajor2D, product, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SpSpMM or DirectPtxSparseEngineOperation.SparseSpGeMM =>
                [CsrValues("a"), CsrColumns("a"), CsrRows("a"),
                 CsrValues("b"), CsrColumns("b"), CsrRows("b"),
                 F("output", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.AddSparseDense =>
                [CooRows("a"), CooColumns("a"), F("a-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Read),
                 F("dense", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Read),
                 F("output", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.MultiplySparseDense =>
                [CooRows("a"), CooColumns("a"), F("a-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Read),
                 F("dense", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Read),
                 F("output-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseGather =>
                [CooRows("indices"), CooColumns("indices"),
                 F("source", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Read),
                 F("output", DirectPtxPhysicalLayout.Vector, values, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseScatter or DirectPtxSparseEngineOperation.SparseToDense =>
                [CooRows("indices"), CooColumns("indices"),
                 F("values", DirectPtxPhysicalLayout.Vector, values, DirectPtxTensorAccess.Read),
                 F("output", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseScatterAdd =>
                [CooRows("indices"), CooColumns("indices"),
                 F("values", DirectPtxPhysicalLayout.Vector, values, DirectPtxTensorAccess.Read),
                 F("target", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.ReadWrite)],
            DirectPtxSparseEngineOperation.DenseToSparse =>
                [F("dense", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Read),
                 I("output-rows", DirectPtxPhysicalLayout.CooRowIndices, flagsDense, DirectPtxTensorAccess.Write),
                 I("output-columns", DirectPtxPhysicalLayout.CooColumnIndices, flagsDense, DirectPtxTensorAccess.Write),
                 F("output-values", DirectPtxPhysicalLayout.Vector, flagsDense, DirectPtxTensorAccess.Write),
                 I("output-flags", DirectPtxPhysicalLayout.Vector, flagsDense, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.Coalesce =>
                [CooRows("input"), CooColumns("input"), F("input-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Read),
                 CooRows("output", DirectPtxTensorAccess.Write), CooColumns("output", DirectPtxTensorAccess.Write),
                 F("output-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Write),
                 I("output-flags", DirectPtxPhysicalLayout.Vector, values, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseTranspose =>
                [CooRows("input"), CooColumns("input"), F("input-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Read),
                 CooRows("output", DirectPtxTensorAccess.Write), CooColumns("output", DirectPtxTensorAccess.Write),
                 F("output-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseAddMM =>
                [CsrValues("a"), CsrColumns("a"), CsrRows("a"),
                 F("b", DirectPtxPhysicalLayout.RowMajor2D, rhs, DirectPtxTensorAccess.Read),
                 F("c", DirectPtxPhysicalLayout.RowMajor2D, product, DirectPtxTensorAccess.Read),
                 F("output", DirectPtxPhysicalLayout.RowMajor2D, product, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseSampledAddMM =>
                [CooRows("pattern"), CooColumns("pattern"),
                 F("a", DirectPtxPhysicalLayout.RowMajor2D, dense, DirectPtxTensorAccess.Read),
                 F("b", DirectPtxPhysicalLayout.RowMajor2D, rhs, DirectPtxTensorAccess.Read),
                 F("c", DirectPtxPhysicalLayout.RowMajor2D, product, DirectPtxTensorAccess.Read),
                 F("output", DirectPtxPhysicalLayout.RowMajor2D, product, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseSum or DirectPtxSparseEngineOperation.SparseMean =>
                [F("values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Read),
                 F("output", DirectPtxPhysicalLayout.Vector, scalar, DirectPtxTensorAccess.Write)],
            DirectPtxSparseEngineOperation.SparseSoftmax or DirectPtxSparseEngineOperation.SparseLogSoftmax =>
                [CooRows("input"), F("input-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Read),
                 F("output-values", DirectPtxPhysicalLayout.CsrValues, values, DirectPtxTensorAccess.Write)],
            _ => throw new ArgumentOutOfRangeException(nameof(operation))
        };

        var semantics = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["shape"] = "rows=256,inner=256,features=32,nnz=1024",
            ["parameter-abi"] = "device-pointers-only",
            ["dynamic-strides"] = "forbidden",
            ["coordinate-validation"] = "host-pre-dispatch",
            ["reduction-order"] = "ascending-stored-index",
            ["workspace-bytes"] = "0",
            ["promotion"] = "experimental-hardware-evidence-pending"
        };
        if (operation == DirectPtxSparseEngineOperation.SparseSampledAddMM)
            semantics["pattern"] = "coo-unique";

        return new DirectPtxKernelBlueprint(
            Operation: "isparse-engine-" + GetOperationName(operation), Version: 1,
            Architecture: architecture, Variant: "m256-k256-n32-nnz1024-f32",
            Tensors: tensors, ResourceBudget: new DirectPtxResourceBudget(40, 0, 0, 4),
            Semantics: semantics);
    }

    internal static string EmitPtx(int ccMajor, int ccMinor, DirectPtxSparseEngineOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(24576);
        ptx.AppendLine(".version 7.1"); ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64"); ptx.AppendLine();
        switch (operation)
        {
            case DirectPtxSparseEngineOperation.SpMV: EmitSpmv(ptx, transpose: false); break;
            case DirectPtxSparseEngineOperation.SpMVTranspose: EmitSpmv(ptx, transpose: true); break;
            case DirectPtxSparseEngineOperation.SpMM:
            case DirectPtxSparseEngineOperation.SparseMatMul:
            case DirectPtxSparseEngineOperation.SparseMatMulPatternPreserving: EmitSpmm(ptx, operation, add: false); break;
            case DirectPtxSparseEngineOperation.SpSpMM:
            case DirectPtxSparseEngineOperation.SparseSpGeMM: EmitSpSp(ptx, operation); break;
            case DirectPtxSparseEngineOperation.AddSparseDense: EmitCooDense(ptx, operation, CooDenseMode.Add); break;
            case DirectPtxSparseEngineOperation.MultiplySparseDense: EmitCooDense(ptx, operation, CooDenseMode.Multiply); break;
            case DirectPtxSparseEngineOperation.SparseGather: EmitCooDense(ptx, operation, CooDenseMode.Gather); break;
            case DirectPtxSparseEngineOperation.SparseScatter: EmitScatter(ptx, operation, add: false); break;
            case DirectPtxSparseEngineOperation.SparseScatterAdd: EmitScatter(ptx, operation, add: true); break;
            case DirectPtxSparseEngineOperation.SparseToDense: EmitScatter(ptx, operation, add: false); break;
            case DirectPtxSparseEngineOperation.DenseToSparse: EmitDenseToSparse(ptx); break;
            case DirectPtxSparseEngineOperation.Coalesce: EmitCoalesce(ptx); break;
            case DirectPtxSparseEngineOperation.SparseTranspose: EmitTranspose(ptx); break;
            case DirectPtxSparseEngineOperation.SparseAddMM: EmitSpmm(ptx, operation, add: true); break;
            case DirectPtxSparseEngineOperation.SparseSampledAddMM: EmitSampledAddMm(ptx); break;
            case DirectPtxSparseEngineOperation.SparseSum: EmitScalarReduction(ptx, mean: false); break;
            case DirectPtxSparseEngineOperation.SparseMean: EmitScalarReduction(ptx, mean: true); break;
            case DirectPtxSparseEngineOperation.SparseSoftmax: EmitSparseSoftmax(ptx, logarithmic: false); break;
            case DirectPtxSparseEngineOperation.SparseLogSoftmax: EmitSparseSoftmax(ptx, logarithmic: true); break;
            default: throw new ArgumentOutOfRangeException(nameof(operation));
        }
        return ptx.ToString();
    }

    private static void EmitSpmv(StringBuilder ptx, bool transpose)
    {
        DirectPtxSparseEngineOperation op = transpose ? DirectPtxSparseEngineOperation.SpMVTranspose : DirectPtxSparseEngineOperation.SpMV;
        Header(ptx, op, 5); Registers(ptx); LoadPointers(ptx, 5); Linear(ptx);
        if (!transpose)
        {
            LoadCsrBounds(ptx, "%r2", 2, 3, 4); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
            ptx.AppendLine("SPMV_LOOP:"); ptx.AppendLine("    setp.ge.u32 %p0, %r3, %r4;"); ptx.AppendLine("    @%p0 bra SPMV_DONE;");
            LoadIndexedInt(ptx, 1, 3, 5, 5); LoadIndexedFloat(ptx, 0, 3, 6, 1); LoadIndexedFloat(ptx, 3, 5, 7, 2);
            ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;"); ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra SPMV_LOOP;");
            ptx.AppendLine("SPMV_DONE:"); StoreIndexedFloat(ptx, 4, 2, 8, 0); End(ptx); return;
        }
        ptx.AppendLine("    mov.u32 %r3, 0;"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("SPMVT_ROW:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {Rows};"); ptx.AppendLine("    @%p0 bra SPMVT_DONE;");
        LoadCsrBounds(ptx, "%r3", 2, 4, 5); ptx.AppendLine("SPMVT_ENTRY:"); ptx.AppendLine("    setp.ge.u32 %p1, %r4, %r5;");
        ptx.AppendLine("    @%p1 bra SPMVT_NEXT_ROW;"); LoadIndexedInt(ptx, 1, 4, 6, 6);
        ptx.AppendLine("    setp.ne.u32 %p2, %r6, %r2;"); ptx.AppendLine("    @%p2 bra SPMVT_NEXT_ENTRY;");
        LoadIndexedFloat(ptx, 0, 4, 7, 1); LoadIndexedFloat(ptx, 3, 3, 8, 2);
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;"); ptx.AppendLine("SPMVT_NEXT_ENTRY:");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;"); ptx.AppendLine("    bra SPMVT_ENTRY;");
        ptx.AppendLine("SPMVT_NEXT_ROW:"); ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra SPMVT_ROW;");
        ptx.AppendLine("SPMVT_DONE:"); StoreIndexedFloat(ptx, 4, 2, 9, 0); End(ptx);
    }

    private static void EmitSpmm(StringBuilder ptx, DirectPtxSparseEngineOperation operation, bool add)
    {
        Header(ptx, operation, add ? 6 : 5); Registers(ptx); LoadPointers(ptx, add ? 6 : 5); Linear(ptx);
        ptx.AppendLine("    shr.u32 %r6, %r2, 5;"); ptx.AppendLine("    and.b32 %r7, %r2, 31;");
        LoadCsrBounds(ptx, "%r6", 2, 3, 4); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("SPMM_LOOP:"); ptx.AppendLine("    setp.ge.u32 %p0, %r3, %r4;"); ptx.AppendLine("    @%p0 bra SPMM_DONE;");
        LoadIndexedInt(ptx, 1, 3, 10, 8); LoadIndexedFloat(ptx, 0, 3, 6, 1);
        ptx.AppendLine("    shl.b32 %r9, %r8, 5;"); ptx.AppendLine("    add.u32 %r9, %r9, %r7;");
        LoadIndexedFloat(ptx, 3, 9, 7, 2); ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra SPMM_LOOP;"); ptx.AppendLine("SPMM_DONE:");
        if (add) { LoadIndexedFloat(ptx, 4, 2, 8, 3); ptx.AppendLine("    add.rn.f32 %f0, %f0, %f3;"); }
        StoreIndexedFloat(ptx, add ? 5 : 4, 2, 9, 0); End(ptx);
    }

    private static void EmitSpSp(StringBuilder ptx, DirectPtxSparseEngineOperation operation)
    {
        Header(ptx, operation, 7); Registers(ptx); LoadPointers(ptx, 7); Linear(ptx);
        ptx.AppendLine("    shr.u32 %r6, %r2, 8;"); ptx.AppendLine("    and.b32 %r7, %r2, 255;");
        LoadCsrBounds(ptx, "%r6", 2, 3, 4); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("SPSP_A:"); ptx.AppendLine("    setp.ge.u32 %p0, %r3, %r4;"); ptx.AppendLine("    @%p0 bra SPSP_DONE;");
        LoadIndexedInt(ptx, 1, 3, 7, 8); LoadIndexedFloat(ptx, 0, 3, 8, 1); LoadCsrBounds(ptx, "%r8", 5, 9, 10);
        ptx.AppendLine("SPSP_B:"); ptx.AppendLine("    setp.ge.u32 %p1, %r9, %r10;"); ptx.AppendLine("    @%p1 bra SPSP_NEXT_A;");
        LoadIndexedInt(ptx, 4, 9, 11, 12); ptx.AppendLine("    setp.ne.u32 %p2, %r12, %r7;"); ptx.AppendLine("    @%p2 bra SPSP_NEXT_B;");
        LoadIndexedFloat(ptx, 3, 9, 12, 2); ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("SPSP_NEXT_B:"); ptx.AppendLine("    add.u32 %r9, %r9, 1;"); ptx.AppendLine("    bra SPSP_B;");
        ptx.AppendLine("SPSP_NEXT_A:"); ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra SPSP_A;");
        ptx.AppendLine("SPSP_DONE:"); StoreIndexedFloat(ptx, 6, 2, 13, 0); End(ptx);
    }

    private enum CooDenseMode { Add, Multiply, Gather }

    private static void EmitCooDense(StringBuilder ptx, DirectPtxSparseEngineOperation operation, CooDenseMode mode)
    {
        int count = mode == CooDenseMode.Gather ? 4 : 5;
        Header(ptx, operation, count); Registers(ptx); LoadPointers(ptx, count); Linear(ptx);
        if (mode == CooDenseMode.Add)
        {
            LoadIndexedFloat(ptx, 3, 2, 5, 0); ptx.AppendLine("    mov.u32 %r3, 0;");
            ptx.AppendLine("COO_ADD_LOOP:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {NonZeros};"); ptx.AppendLine("    @%p0 bra COO_ADD_DONE;");
            LoadIndexedInt(ptx, 0, 3, 6, 4); LoadIndexedInt(ptx, 1, 3, 7, 5); ptx.AppendLine("    shl.b32 %r6, %r4, 8;");
            ptx.AppendLine("    add.u32 %r6, %r6, %r5;"); ptx.AppendLine("    setp.ne.u32 %p1, %r6, %r2;");
            ptx.AppendLine("    @%p1 bra COO_ADD_NEXT;"); LoadIndexedFloat(ptx, 2, 3, 8, 1); ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
            ptx.AppendLine("COO_ADD_NEXT:"); ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra COO_ADD_LOOP;");
            ptx.AppendLine("COO_ADD_DONE:"); StoreIndexedFloat(ptx, 4, 2, 9, 0); End(ptx); return;
        }
        LoadIndexedInt(ptx, 0, 2, 5, 3); LoadIndexedInt(ptx, 1, 2, 6, 4);
        ptx.AppendLine("    shl.b32 %r5, %r3, 8;"); ptx.AppendLine("    add.u32 %r5, %r5, %r4;");
        LoadIndexedFloat(ptx, mode == CooDenseMode.Gather ? 2 : 3, 5, 7, 0);
        if (mode == CooDenseMode.Multiply)
        {
            LoadIndexedFloat(ptx, 2, 2, 8, 1); ptx.AppendLine("    mul.rn.f32 %f0, %f0, %f1;");
        }
        StoreIndexedFloat(ptx, mode == CooDenseMode.Gather ? 3 : 4, 2, 9, 0); End(ptx);
    }

    private static void EmitScatter(StringBuilder ptx, DirectPtxSparseEngineOperation operation, bool add)
    {
        Header(ptx, operation, 4); Registers(ptx); LoadPointers(ptx, 4); Linear(ptx);
        if (add) LoadIndexedFloat(ptx, 3, 2, 5, 0); else ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.u32 %r3, 0;"); ptx.AppendLine("SCATTER_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r3, {NonZeros};"); ptx.AppendLine("    @%p0 bra SCATTER_DONE;");
        LoadIndexedInt(ptx, 0, 3, 6, 4); LoadIndexedInt(ptx, 1, 3, 7, 5); ptx.AppendLine("    shl.b32 %r6, %r4, 8;");
        ptx.AppendLine("    add.u32 %r6, %r6, %r5;"); ptx.AppendLine("    setp.ne.u32 %p1, %r6, %r2;");
        ptx.AppendLine("    @%p1 bra SCATTER_NEXT;"); LoadIndexedFloat(ptx, 2, 3, 8, 1);
        ptx.AppendLine(add ? "    add.rn.f32 %f0, %f0, %f1;" : "    mov.f32 %f0, %f1;");
        ptx.AppendLine("SCATTER_NEXT:"); ptx.AppendLine("    add.u32 %r3, %r3, 1;"); ptx.AppendLine("    bra SCATTER_LOOP;");
        ptx.AppendLine("SCATTER_DONE:"); StoreIndexedFloat(ptx, 3, 2, 9, 0); End(ptx);
    }

    private static void EmitDenseToSparse(StringBuilder ptx)
    {
        Header(ptx, DirectPtxSparseEngineOperation.DenseToSparse, 5); Registers(ptx); LoadPointers(ptx, 5); Linear(ptx);
        LoadIndexedFloat(ptx, 0, 2, 5, 0); ptx.AppendLine("    abs.f32 %f1, %f0;");
        ptx.AppendLine("    setp.gt.f32 %p0, %f1, 0f00000000;"); ptx.AppendLine("    selp.u32 %r3, 1, 0, %p0;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 8;"); ptx.AppendLine("    and.b32 %r5, %r2, 255;");
        StoreIndexedInt(ptx, 1, 2, 6, 4); StoreIndexedInt(ptx, 2, 2, 7, 5);
        StoreIndexedFloat(ptx, 3, 2, 8, 0); StoreIndexedInt(ptx, 4, 2, 9, 3); End(ptx);
    }

    private static void EmitCoalesce(StringBuilder ptx)
    {
        Header(ptx, DirectPtxSparseEngineOperation.Coalesce, 7); Registers(ptx); LoadPointers(ptx, 7); Linear(ptx);
        LoadIndexedInt(ptx, 0, 2, 7, 3); LoadIndexedInt(ptx, 1, 2, 8, 4); LoadIndexedFloat(ptx, 2, 2, 9, 0);
        ptx.AppendLine("    mov.u32 %r5, 0;"); ptx.AppendLine("COALESCE_PRIOR:"); ptx.AppendLine("    setp.ge.u32 %p0, %r5, %r2;");
        ptx.AppendLine("    @%p0 bra COALESCE_SUM_START;"); LoadIndexedInt(ptx, 0, 5, 10, 6); LoadIndexedInt(ptx, 1, 5, 11, 7);
        ptx.AppendLine("    setp.ne.u32 %p1, %r6, %r3;"); ptx.AppendLine("    setp.ne.u32 %p2, %r7, %r4;");
        ptx.AppendLine("    or.pred %p3, %p1, %p2;"); ptx.AppendLine("    @!%p3 bra COALESCE_DUPLICATE;");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;"); ptx.AppendLine("    bra COALESCE_PRIOR;");
        ptx.AppendLine("COALESCE_SUM_START:"); ptx.AppendLine("    add.u32 %r5, %r2, 1;");
        ptx.AppendLine("COALESCE_SUM:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {NonZeros};"); ptx.AppendLine("    @%p0 bra COALESCE_UNIQUE;");
        LoadIndexedInt(ptx, 0, 5, 10, 6); LoadIndexedInt(ptx, 1, 5, 11, 7);
        ptx.AppendLine("    setp.ne.u32 %p1, %r6, %r3;"); ptx.AppendLine("    setp.ne.u32 %p2, %r7, %r4;"); ptx.AppendLine("    or.pred %p3, %p1, %p2;");
        ptx.AppendLine("    @%p3 bra COALESCE_SUM_NEXT;"); LoadIndexedFloat(ptx, 2, 5, 12, 1); ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("COALESCE_SUM_NEXT:"); ptx.AppendLine("    add.u32 %r5, %r5, 1;"); ptx.AppendLine("    bra COALESCE_SUM;");
        ptx.AppendLine("COALESCE_UNIQUE:"); StoreIndexedInt(ptx, 3, 2, 13, 3); StoreIndexedInt(ptx, 4, 2, 14, 4);
        StoreIndexedFloat(ptx, 5, 2, 15, 0); ptx.AppendLine("    mov.u32 %r8, 1;"); StoreIndexedInt(ptx, 6, 2, 16, 8);
        ptx.AppendLine("    bra COALESCE_EXIT;"); ptx.AppendLine("COALESCE_DUPLICATE:");
        ptx.AppendLine("    mov.u32 %r8, 0;"); StoreIndexedInt(ptx, 6, 2, 16, 8);
        ptx.AppendLine("COALESCE_EXIT:"); End(ptx);
    }

    private static void EmitTranspose(StringBuilder ptx)
    {
        Header(ptx, DirectPtxSparseEngineOperation.SparseTranspose, 6); Registers(ptx); LoadPointers(ptx, 6); Linear(ptx);
        LoadIndexedInt(ptx, 0, 2, 6, 3); LoadIndexedInt(ptx, 1, 2, 7, 4); LoadIndexedFloat(ptx, 2, 2, 8, 0);
        StoreIndexedInt(ptx, 3, 2, 9, 4); StoreIndexedInt(ptx, 4, 2, 10, 3); StoreIndexedFloat(ptx, 5, 2, 11, 0); End(ptx);
    }

    private static void EmitSampledAddMm(StringBuilder ptx)
    {
        Header(ptx, DirectPtxSparseEngineOperation.SparseSampledAddMM, 6); Registers(ptx); LoadPointers(ptx, 6); Linear(ptx);
        ptx.AppendLine("    shr.u32 %r3, %r2, 5;"); ptx.AppendLine("    and.b32 %r4, %r2, 31;"); ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("SAMPLED_PATTERN:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {NonZeros};"); ptx.AppendLine("    @%p0 bra SAMPLED_ZERO;");
        LoadIndexedInt(ptx, 0, 5, 6, 6); LoadIndexedInt(ptx, 1, 5, 7, 7); ptx.AppendLine("    setp.ne.u32 %p1, %r6, %r3;");
        ptx.AppendLine("    setp.ne.u32 %p2, %r7, %r4;"); ptx.AppendLine("    or.pred %p3, %p1, %p2;"); ptx.AppendLine("    @!%p3 bra SAMPLED_DOT_START;");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;"); ptx.AppendLine("    bra SAMPLED_PATTERN;");
        ptx.AppendLine("SAMPLED_DOT_START:"); ptx.AppendLine("    mov.u32 %r5, 0;"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("SAMPLED_DOT:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {Inner};"); ptx.AppendLine("    @%p0 bra SAMPLED_ADD_C;");
        ptx.AppendLine("    shl.b32 %r8, %r3, 8;"); ptx.AppendLine("    add.u32 %r8, %r8, %r5;"); LoadIndexedFloat(ptx, 2, 8, 8, 1);
        ptx.AppendLine("    shl.b32 %r9, %r5, 5;"); ptx.AppendLine("    add.u32 %r9, %r9, %r4;"); LoadIndexedFloat(ptx, 3, 9, 9, 2);
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;"); ptx.AppendLine("    add.u32 %r5, %r5, 1;"); ptx.AppendLine("    bra SAMPLED_DOT;");
        ptx.AppendLine("SAMPLED_ADD_C:"); LoadIndexedFloat(ptx, 4, 2, 10, 3); ptx.AppendLine("    add.rn.f32 %f0, %f0, %f3;"); ptx.AppendLine("    bra SAMPLED_STORE;");
        ptx.AppendLine("SAMPLED_ZERO:"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("SAMPLED_STORE:"); StoreIndexedFloat(ptx, 5, 2, 11, 0); End(ptx);
    }

    private static void EmitScalarReduction(StringBuilder ptx, bool mean)
    {
        DirectPtxSparseEngineOperation op = mean ? DirectPtxSparseEngineOperation.SparseMean : DirectPtxSparseEngineOperation.SparseSum;
        Header(ptx, op, 2); Registers(ptx); LoadPointers(ptx, 2); ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    setp.ne.u32 %p0, %r0, 0;"); ptx.AppendLine("    @%p0 bra REDUCE_EXIT;");
        ptx.AppendLine("    mov.u32 %r2, 0;"); ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("REDUCE_LOOP:"); ptx.AppendLine($"    setp.ge.u32 %p1, %r2, {NonZeros};"); ptx.AppendLine("    @%p1 bra REDUCE_DONE;");
        LoadIndexedFloat(ptx, 0, 2, 2, 1); ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;"); ptx.AppendLine("    add.u32 %r2, %r2, 1;"); ptx.AppendLine("    bra REDUCE_LOOP;");
        ptx.AppendLine("REDUCE_DONE:"); if (mean) ptx.AppendLine("    mul.rn.f32 %f0, %f0, 0f37800000;");
        ptx.AppendLine("    st.global.f32 [%rd1], %f0;"); ptx.AppendLine("REDUCE_EXIT:"); End(ptx);
    }

    private static void EmitSparseSoftmax(StringBuilder ptx, bool logarithmic)
    {
        DirectPtxSparseEngineOperation op = logarithmic ? DirectPtxSparseEngineOperation.SparseLogSoftmax : DirectPtxSparseEngineOperation.SparseSoftmax;
        Header(ptx, op, 3); Registers(ptx); LoadPointers(ptx, 3); Linear(ptx);
        LoadIndexedInt(ptx, 0, 2, 4, 3); LoadIndexedFloat(ptx, 1, 2, 5, 0);
        ptx.AppendLine("    mov.u32 %r4, 0;"); ptx.AppendLine("    mov.f32 %f1, 0ff800000;");
        ptx.AppendLine("SOFTMAX_MAX:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {NonZeros};"); ptx.AppendLine("    @%p0 bra SOFTMAX_SUM_START;");
        LoadIndexedInt(ptx, 0, 4, 6, 5); ptx.AppendLine("    setp.ne.u32 %p1, %r5, %r3;"); ptx.AppendLine("    @%p1 bra SOFTMAX_MAX_NEXT;");
        LoadIndexedFloat(ptx, 1, 4, 7, 2); ptx.AppendLine("    max.f32 %f1, %f1, %f2;"); ptx.AppendLine("SOFTMAX_MAX_NEXT:");
        ptx.AppendLine("    add.u32 %r4, %r4, 1;"); ptx.AppendLine("    bra SOFTMAX_MAX;");
        ptx.AppendLine("SOFTMAX_SUM_START:"); ptx.AppendLine("    mov.u32 %r4, 0;"); ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("SOFTMAX_SUM:"); ptx.AppendLine($"    setp.ge.u32 %p0, %r4, {NonZeros};"); ptx.AppendLine("    @%p0 bra SOFTMAX_DONE;");
        LoadIndexedInt(ptx, 0, 4, 6, 5); ptx.AppendLine("    setp.ne.u32 %p1, %r5, %r3;"); ptx.AppendLine("    @%p1 bra SOFTMAX_SUM_NEXT;");
        LoadIndexedFloat(ptx, 1, 4, 7, 2); ptx.AppendLine("    sub.rn.f32 %f2, %f2, %f1;"); ptx.AppendLine("    mul.rn.f32 %f2, %f2, 0f3fb8aa3b;");
        ptx.AppendLine("    ex2.approx.f32 %f2, %f2;"); ptx.AppendLine("    add.rn.f32 %f3, %f3, %f2;");
        ptx.AppendLine("SOFTMAX_SUM_NEXT:"); ptx.AppendLine("    add.u32 %r4, %r4, 1;"); ptx.AppendLine("    bra SOFTMAX_SUM;");
        ptx.AppendLine("SOFTMAX_DONE:");
        if (logarithmic)
        {
            ptx.AppendLine("    lg2.approx.f32 %f4, %f3;"); ptx.AppendLine("    mul.rn.f32 %f4, %f4, 0f3f317218;");
            ptx.AppendLine("    sub.rn.f32 %f0, %f0, %f1;"); ptx.AppendLine("    sub.rn.f32 %f0, %f0, %f4;");
        }
        else
        {
            ptx.AppendLine("    sub.rn.f32 %f0, %f0, %f1;"); ptx.AppendLine("    mul.rn.f32 %f0, %f0, 0f3fb8aa3b;");
            ptx.AppendLine("    ex2.approx.f32 %f0, %f0;"); ptx.AppendLine("    div.rn.f32 %f0, %f0, %f3;");
        }
        StoreIndexedFloat(ptx, 2, 2, 8, 0); End(ptx);
    }

    private static DirectPtxKernelBlueprint[] CreateAdmissionBlueprints()
    {
        var operations = (DirectPtxSparseEngineOperation[])Enum.GetValues(typeof(DirectPtxSparseEngineOperation));
        var result = new DirectPtxKernelBlueprint[operations.Length];
        foreach (DirectPtxSparseEngineOperation operation in operations)
            result[(int)operation] = CreateBlueprint(DirectPtxArchitectureFamily.Ampere, operation);
        return result;
    }

    private static int GetWorkItems(DirectPtxSparseEngineOperation operation) => operation switch
    {
        DirectPtxSparseEngineOperation.SpMV or DirectPtxSparseEngineOperation.SpMVTranspose => Rows,
        DirectPtxSparseEngineOperation.SpMM or DirectPtxSparseEngineOperation.SparseMatMul or
        DirectPtxSparseEngineOperation.SparseMatMulPatternPreserving or DirectPtxSparseEngineOperation.SparseAddMM or
        DirectPtxSparseEngineOperation.SparseSampledAddMM => ProductElements,
        DirectPtxSparseEngineOperation.SpSpMM or DirectPtxSparseEngineOperation.SparseSpGeMM or
        DirectPtxSparseEngineOperation.AddSparseDense or DirectPtxSparseEngineOperation.SparseScatter or
        DirectPtxSparseEngineOperation.SparseScatterAdd or DirectPtxSparseEngineOperation.SparseToDense or
        DirectPtxSparseEngineOperation.DenseToSparse => DenseElements,
        DirectPtxSparseEngineOperation.SparseSum or DirectPtxSparseEngineOperation.SparseMean => 1,
        _ => NonZeros
    };

    internal static string GetOperationName(DirectPtxSparseEngineOperation operation) => operation switch
    {
        DirectPtxSparseEngineOperation.SpMV => "spmv",
        DirectPtxSparseEngineOperation.SpMVTranspose => "spmv-transpose",
        DirectPtxSparseEngineOperation.SpMM => "spmm",
        DirectPtxSparseEngineOperation.SpSpMM => "spspmm",
        DirectPtxSparseEngineOperation.AddSparseDense => "add-sparse-dense",
        DirectPtxSparseEngineOperation.MultiplySparseDense => "multiply-sparse-dense",
        DirectPtxSparseEngineOperation.SparseGather => "sparse-gather",
        DirectPtxSparseEngineOperation.SparseScatter => "sparse-scatter",
        DirectPtxSparseEngineOperation.SparseScatterAdd => "sparse-scatter-add",
        DirectPtxSparseEngineOperation.SparseToDense => "sparse-to-dense",
        DirectPtxSparseEngineOperation.DenseToSparse => "dense-to-sparse",
        DirectPtxSparseEngineOperation.Coalesce => "coalesce",
        DirectPtxSparseEngineOperation.SparseTranspose => "sparse-transpose",
        DirectPtxSparseEngineOperation.SparseMatMul => "sparse-matmul",
        DirectPtxSparseEngineOperation.SparseMatMulPatternPreserving => "sparse-matmul-pattern-preserving",
        DirectPtxSparseEngineOperation.SparseAddMM => "sparse-addmm",
        DirectPtxSparseEngineOperation.SparseSampledAddMM => "sparse-sampled-addmm",
        DirectPtxSparseEngineOperation.SparseSpGeMM => "sparse-spgemm",
        DirectPtxSparseEngineOperation.SparseSum => "sparse-sum",
        DirectPtxSparseEngineOperation.SparseMean => "sparse-mean",
        DirectPtxSparseEngineOperation.SparseSoftmax => "sparse-softmax",
        DirectPtxSparseEngineOperation.SparseLogSoftmax => "sparse-log-softmax",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

    internal static string GetEntryPoint(DirectPtxSparseEngineOperation operation) =>
        $"aidotnet_isparse_{GetOperationName(operation).Replace('-', '_')}_f32_m256_k256_n32_nnz1024";

    private static void Header(StringBuilder ptx, DirectPtxSparseEngineOperation operation, int pointers)
    {
        ptx.AppendLine($".visible .entry {GetEntryPoint(operation)}(");
        for (int i = 0; i < pointers; i++) ptx.AppendLine($"    .param .u64 p{i}{(i + 1 == pointers ? string.Empty : ",")}");
        ptx.AppendLine(")"); ptx.AppendLine("{");
    }

    private static void Registers(StringBuilder ptx)
    {
        ptx.AppendLine("    .reg .pred %p<8>;"); ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;"); ptx.AppendLine("    .reg .f32 %f<12>;");
    }

    private static void LoadPointers(StringBuilder ptx, int count)
    {
        for (int i = 0; i < count; i++) ptx.AppendLine($"    ld.param.u64 %rd{i}, [p{i}];");
    }

    private static void Linear(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;"); ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static void LoadCsrBounds(StringBuilder ptx, string row, int pointer, int startRegister, int endRegister)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd20, {row}, 4;"); ptx.AppendLine($"    add.u64 %rd21, %rd{pointer}, %rd20;");
        ptx.AppendLine($"    ld.global.u32 %r{startRegister}, [%rd21];"); ptx.AppendLine("    add.u64 %rd21, %rd21, 4;");
        ptx.AppendLine($"    ld.global.u32 %r{endRegister}, [%rd21];");
    }

    private static void LoadIndexedInt(StringBuilder ptx, int pointer, int index, int address, int output) =>
        ptx.AppendLine($"    mul.wide.u32 %rd{address}, %r{index}, 4;\n    add.u64 %rd{address}, %rd{pointer}, %rd{address};\n    ld.global.u32 %r{output}, [%rd{address}];");

    private static void LoadIndexedFloat(StringBuilder ptx, int pointer, int index, int address, int output) =>
        ptx.AppendLine($"    mul.wide.u32 %rd{address}, %r{index}, 4;\n    add.u64 %rd{address}, %rd{pointer}, %rd{address};\n    ld.global.f32 %f{output}, [%rd{address}];");

    private static void StoreIndexedInt(StringBuilder ptx, int pointer, int index, int address, int input) =>
        ptx.AppendLine($"    mul.wide.u32 %rd{address}, %r{index}, 4;\n    add.u64 %rd{address}, %rd{pointer}, %rd{address};\n    st.global.u32 [%rd{address}], %r{input};");

    private static void StoreIndexedFloat(StringBuilder ptx, int pointer, int index, int address, int input) =>
        ptx.AppendLine($"    mul.wide.u32 %rd{address}, %r{index}, 4;\n    add.u64 %rd{address}, %rd{pointer}, %rd{address};\n    st.global.f32 [%rd{address}], %f{input};");

    private static void End(StringBuilder ptx) { ptx.AppendLine("    ret;"); ptx.AppendLine("}"); }

    private static DirectPtxTensorContract Exact(string name, DirectPtxPhysicalType type,
        DirectPtxPhysicalLayout layout, DirectPtxExtent extent, DirectPtxTensorAccess access) =>
        new(name, type, layout, extent, extent, 16, access, DirectPtxExtentMode.Exact);

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType || view.Layout != contract.Layout ||
            view.LogicalExtent != contract.LogicalExtent || view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || view.AllocationByteLength != contract.RequiredBytes || view.Access != contract.Access)
            throw new ArgumentException($"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
    }

    private static bool Overlaps(DirectPtxTensorView left, DirectPtxTensorView right)
    {
        nuint leftStart = (nuint)left.Pointer, rightStart = (nuint)right.Pointer;
        nuint leftEnd = checked(leftStart + left.ByteLength), rightEnd = checked(rightStart + right.ByteLength);
        return leftStart < rightEnd && rightStart < leftEnd;
    }

    public void Dispose() => _module.Dispose();
}
#endif
