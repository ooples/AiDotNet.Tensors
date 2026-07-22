#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxTensorScatterHighLevelOperation
{
    Mean,
    AddBackward
}

/// <summary>
/// Exact direct-PTX implementations for the public tensor-engine scatter
/// surfaces. These are deliberately separate from the resident-index backend
/// modules: the tensor ABI returns Int32 counts and owns its own production
/// admission, cache, and audit record.
/// </summary>
internal sealed class PtxTensorScatterHighLevelF32Kernel : IDisposable
{
    internal const int SourceRows = 16384;
    internal const int OutputRows = 1024;
    internal const int Features = 64;
    internal const int SourceElements = SourceRows * Features;
    internal const int OutputElements = OutputRows * Features;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxTensorScatterHighLevelOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxTensorScatterHighLevelF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxTensorScatterHighLevelOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Tensor scatter has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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

    internal static bool SupportsShape(int sourceRows, int features, int outputRows) =>
        sourceRows == SourceRows && features == Features && outputRows == OutputRows;

    internal unsafe void LaunchMean(
        DirectPtxTensorView source,
        DirectPtxTensorView indices,
        DirectPtxTensorView output,
        DirectPtxTensorView counts)
    {
        if (Operation != DirectPtxTensorScatterHighLevelOperation.Mean)
            throw new InvalidOperationException("This module exposes scatter-add backward.");
        Require(source, Blueprint.Tensors[0], nameof(source));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(output, Blueprint.Tensors[2], nameof(output));
        Require(counts, Blueprint.Tensors[3], nameof(counts));
        if (Overlaps(output, source) || Overlaps(output, indices) ||
            Overlaps(output, counts) || Overlaps(counts, source) || Overlaps(counts, indices))
            throw new ArgumentException("Tensor scatter-mean outputs must be disjoint from all inputs and each other.");
        IntPtr p0 = source.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = output.Pointer;
        IntPtr p3 = counts.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchAddBackward(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView indices,
        DirectPtxTensorView gradSource)
    {
        if (Operation != DirectPtxTensorScatterHighLevelOperation.AddBackward)
            throw new InvalidOperationException("This module exposes scatter-mean.");
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(gradSource, Blueprint.Tensors[2], nameof(gradSource));
        if (Overlaps(gradSource, gradOutput) || Overlaps(gradSource, indices))
            throw new ArgumentException(
                "Tensor scatter-add backward output must be disjoint from its inputs.",
                nameof(gradSource));
        IntPtr p0 = gradOutput.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = gradSource.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        _module.Launch(_function, SourceElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxTensorScatterHighLevelOperation operation)
    {
        var source = new DirectPtxExtent(SourceRows, Features);
        var grouped = new DirectPtxExtent(OutputRows, Features);
        var indices = new DirectPtxExtent(SourceRows);
        IReadOnlyList<DirectPtxTensorContract> tensors = operation ==
            DirectPtxTensorScatterHighLevelOperation.Mean
            ?
            [
                Exact("source", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.EdgeMajor2D, source, DirectPtxTensorAccess.Read),
                Exact("indices", DirectPtxPhysicalType.Int32,
                    DirectPtxPhysicalLayout.CooRowIndices, indices, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.RowMajor2D, grouped, DirectPtxTensorAccess.Write),
                Exact("counts", DirectPtxPhysicalType.Int32,
                    DirectPtxPhysicalLayout.Vector, new DirectPtxExtent(OutputRows),
                    DirectPtxTensorAccess.Write)
            ]
            :
            [
                Exact("grad-output", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.RowMajor2D, grouped, DirectPtxTensorAccess.Read),
                Exact("indices", DirectPtxPhysicalType.Int32,
                    DirectPtxPhysicalLayout.CooRowIndices, indices, DirectPtxTensorAccess.Read),
                Exact("grad-source", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.EdgeMajor2D, source, DirectPtxTensorAccess.Write)
            ];
        return new DirectPtxKernelBlueprint(
            Operation: operation == DirectPtxTensorScatterHighLevelOperation.Mean
                ? "tensor-scatter-mean-f32" : "tensor-scatter-add-backward-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "src16384-out1024-f64",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["invalid-index"] = operation == DirectPtxTensorScatterHighLevelOperation.Mean
                    ? "ignored" : "zero",
                ["counts-type"] = operation == DirectPtxTensorScatterHighLevelOperation.Mean
                    ? "int32" : "not-applicable",
                ["reduction-order"] = operation == DirectPtxTensorScatterHighLevelOperation.Mean
                    ? "ascending-source-row" : "not-applicable",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxTensorScatterHighLevelOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(6144);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        if (operation == DirectPtxTensorScatterHighLevelOperation.Mean)
            EmitMean(ptx);
        else
            EmitAddBackward(ptx);
        return ptx.ToString();
    }

    private static void EmitMean(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxTensorScatterHighLevelOperation.Mean),
            "source_ptr", "indices_ptr", "output_ptr", "counts_ptr");
        EmitRegisters(ptx);
        ptx.AppendLine("    ld.param.u64 %rd0, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [counts_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine("    and.b32 %r3, %r2, 63;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 6;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("TENSOR_MEAN_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {SourceRows};");
        ptx.AppendLine("    @%p0 bra TENSOR_MEAN_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.s32 %r7, [%rd5];");
        ptx.AppendLine("    setp.ne.s32 %p1, %r7, %r4;");
        ptx.AppendLine("    @%p1 bra TENSOR_MEAN_NEXT;");
        ptx.AppendLine("    shl.b32 %r8, %r5, 6;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd7];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("TENSOR_MEAN_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra TENSOR_MEAN_LOOP;");
        ptx.AppendLine("TENSOR_MEAN_DONE:");
        ptx.AppendLine("    cvt.rn.f32.u32 %f2, %r6;");
        ptx.AppendLine("    setp.eq.u32 %p2, %r6, 0;");
        ptx.AppendLine("    div.rn.f32 %f3, %f0, %f2;");
        ptx.AppendLine("    selp.f32 %f3, 0f00000000, %f3, %p2;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f3;");
        ptx.AppendLine("    setp.ne.u32 %p3, %r3, 0;");
        ptx.AppendLine("    @%p3 bra TENSOR_MEAN_EXIT;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
        ptx.AppendLine("    st.global.u32 [%rd11], %r6;");
        ptx.AppendLine("TENSOR_MEAN_EXIT:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitAddBackward(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxTensorScatterHighLevelOperation.AddBackward),
            "grad_output_ptr", "indices_ptr", "grad_source_ptr");
        EmitRegisters(ptx);
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_source_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");
        ptx.AppendLine("    and.b32 %r4, %r2, 63;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.s32 %r5, [%rd4];");
        ptx.AppendLine("    setp.lt.s32 %p0, %r5, 0;");
        ptx.AppendLine($"    setp.ge.s32 %p1, %r5, {OutputRows};");
        ptx.AppendLine("    or.pred %p2, %p0, %p1;");
        ptx.AppendLine("    @%p2 bra TENSOR_ADD_BACKWARD_ZERO;");
        ptx.AppendLine("    shl.b32 %r6, %r5, 6;");
        ptx.AppendLine("    add.u32 %r6, %r6, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd6];");
        ptx.AppendLine("    bra TENSOR_ADD_BACKWARD_STORE;");
        ptx.AppendLine("TENSOR_ADD_BACKWARD_ZERO:");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("TENSOR_ADD_BACKWARD_STORE:");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitHeader(StringBuilder ptx, string entryPoint, params string[] pointers)
    {
        ptx.AppendLine($".visible .entry {entryPoint}(");
        for (int i = 0; i < pointers.Length; i++)
            ptx.AppendLine($"    .param .u64 {pointers[i]}{(i + 1 == pointers.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
    }

    private static void EmitRegisters(StringBuilder ptx)
    {
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
    }

    private static void EmitLinearIndex(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static string GetEntryPoint(DirectPtxTensorScatterHighLevelOperation operation) =>
        operation == DirectPtxTensorScatterHighLevelOperation.Mean
            ? "aidotnet_tensor_scatter_mean_f32_src16384_out1024_f64"
            : "aidotnet_tensor_scatter_add_backward_f32_src16384_out1024_f64";

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalType type,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, type, layout, extent, extent, 16, access, DirectPtxExtentMode.Exact);

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent || view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
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
