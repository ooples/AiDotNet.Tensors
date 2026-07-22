#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxScatterBackwardRowsOperation
{
    Add,
    Mean
}

/// <summary>
/// Exact backward row gathers for scatter-add and scatter-mean. One thread
/// owns one source-row feature and gathers directly from the grouped gradient;
/// the mean variant applies the precomputed group count in registers.
/// </summary>
internal sealed class PtxScatterBackwardRowsF32Kernel : IDisposable
{
    internal const int SourceRows = 16384;
    internal const int OutputRows = 1024;
    internal const int Features = 64;
    internal const int SourceElements = SourceRows * Features;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxScatterBackwardRowsOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxScatterBackwardRowsF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxScatterBackwardRowsOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Scatter backward rows have no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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

    internal unsafe void LaunchAdd(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView indices,
        DirectPtxTensorView gradSource)
    {
        if (Operation != DirectPtxScatterBackwardRowsOperation.Add)
            throw new InvalidOperationException("This module requires scatter-mean counts.");
        LaunchCore(gradOutput, indices, default, gradSource);
    }

    internal unsafe void LaunchMean(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView indices,
        DirectPtxTensorView counts,
        DirectPtxTensorView gradSource)
    {
        if (Operation != DirectPtxScatterBackwardRowsOperation.Mean)
            throw new InvalidOperationException("This module has no counts parameter.");
        LaunchCore(gradOutput, indices, counts, gradSource);
    }

    private unsafe void LaunchCore(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView indices,
        DirectPtxTensorView counts,
        DirectPtxTensorView gradSource)
    {
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        if (Operation == DirectPtxScatterBackwardRowsOperation.Mean)
            Require(counts, Blueprint.Tensors[2], nameof(counts));
        int outputIndex = Operation == DirectPtxScatterBackwardRowsOperation.Mean ? 3 : 2;
        Require(gradSource, Blueprint.Tensors[outputIndex], nameof(gradSource));
        if (Overlaps(gradSource, gradOutput) || Overlaps(gradSource, indices) ||
            (counts.Pointer != IntPtr.Zero && Overlaps(gradSource, counts)))
            throw new ArgumentException("Scatter backward output must be disjoint from every input.", nameof(gradSource));
        IntPtr p0 = gradOutput.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = counts.Pointer;
        IntPtr p3 = gradSource.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        if (Operation == DirectPtxScatterBackwardRowsOperation.Mean)
        {
            arguments[2] = &p2;
            arguments[3] = &p3;
        }
        else
            arguments[2] = &p3;
        _module.Launch(_function, SourceElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxScatterBackwardRowsOperation operation)
    {
        var source = new DirectPtxExtent(SourceRows, Features);
        var grouped = new DirectPtxExtent(OutputRows, Features);
        var rows = new DirectPtxExtent(SourceRows);
        var counts = new DirectPtxExtent(OutputRows);
        var tensors = new List<DirectPtxTensorContract>
        {
            Exact("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                grouped, DirectPtxTensorAccess.Read),
            Exact("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                rows, DirectPtxTensorAccess.Read)
        };
        if (operation == DirectPtxScatterBackwardRowsOperation.Mean)
            tensors.Add(Exact("counts", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                counts, DirectPtxTensorAccess.Read));
        tensors.Add(Exact("grad-source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
            source, DirectPtxTensorAccess.Write));
        return new DirectPtxKernelBlueprint(
            Operation: operation == DirectPtxScatterBackwardRowsOperation.Add
                ? "scatter-add-backward-rows-f32" : "scatter-mean-backward-rows-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "src16384-out1024-f64",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["invalid-index"] = "zero",
                ["zero-mean-count-divisor"] = operation == DirectPtxScatterBackwardRowsOperation.Mean
                    ? "one" : "not-applicable",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxScatterBackwardRowsOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        bool mean = operation == DirectPtxScatterBackwardRowsOperation.Mean;
        var ptx = new StringBuilder(3072);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(operation)}(");
        ptx.AppendLine("    .param .u64 grad_output_ptr,");
        ptx.AppendLine("    .param .u64 indices_ptr,");
        if (mean) ptx.AppendLine("    .param .u64 counts_ptr,");
        ptx.AppendLine("    .param .u64 grad_source_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        if (mean) ptx.AppendLine("    ld.param.u64 %rd2, [counts_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd3, [grad_source_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {Features - 1};");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.s32 %r5, [%rd5];");
        ptx.AppendLine("    setp.lt.s32 %p0, %r5, 0;");
        ptx.AppendLine($"    setp.ge.s32 %p1, %r5, {OutputRows};");
        ptx.AppendLine("    or.pred %p2, %p0, %p1;");
        ptx.AppendLine("    @%p2 bra BACKWARD_ZERO;");
        ptx.AppendLine("    shl.b32 %r6, %r5, 6;");
        ptx.AppendLine("    add.u32 %r6, %r6, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd7];");
        if (mean)
        {
            ptx.AppendLine("    mul.wide.s32 %rd8, %r5, 4;");
            ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
            ptx.AppendLine("    ld.global.s32 %r7, [%rd9];");
            ptx.AppendLine("    max.s32 %r7, %r7, 1;");
            ptx.AppendLine("    cvt.rn.f32.s32 %f1, %r7;");
            ptx.AppendLine("    div.rn.f32 %f0, %f0, %f1;");
        }
        ptx.AppendLine("    bra BACKWARD_STORE;");
        ptx.AppendLine("BACKWARD_ZERO:");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("BACKWARD_STORE:");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string GetEntryPoint(DirectPtxScatterBackwardRowsOperation operation) =>
        operation == DirectPtxScatterBackwardRowsOperation.Add
            ? "aidotnet_scatter_add_backward_rows_f32_r16384_o1024_f64"
            : "aidotnet_scatter_mean_backward_rows_f32_r16384_o1024_f64";

    private static DirectPtxTensorContract Exact(
        string name,
        DirectPtxPhysicalType type,
        DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent,
        DirectPtxTensorAccess access) =>
        new(name, type, layout, extent, extent, 16, access, DirectPtxExtentMode.Exact);

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
