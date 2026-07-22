#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxResidentScatterAuxOperation
{
    MeanRowsWithCounts,
    MaxBackwardRows
}

/// <summary>
/// Exact deterministic resident scatter auxiliaries for the canonical graph
/// row layout. Shapes are baked into pointer-only PTX modules.
/// </summary>
internal sealed class PtxResidentScatterAuxF32Kernel : IDisposable
{
    internal const int SourceRows = 16384;
    internal const int OutputRows = 1024;
    internal const int Features = 64;
    internal const int SourceElements = SourceRows * Features;
    internal const int OutputElements = OutputRows * Features;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxResidentScatterAuxOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxResidentScatterAuxF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxResidentScatterAuxOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Resident scatter auxiliaries have no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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
        if (Operation != DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts)
            throw new InvalidOperationException("This module does not expose resident scatter mean.");
        Require(source, Blueprint.Tensors[0], nameof(source));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(output, Blueprint.Tensors[2], nameof(output));
        Require(counts, Blueprint.Tensors[3], nameof(counts));
        if (Overlaps(output, source) || Overlaps(output, indices) || Overlaps(output, counts) ||
            Overlaps(counts, source) || Overlaps(counts, indices))
            throw new ArgumentException("Resident scatter outputs must be disjoint from all inputs and each other.");
        IntPtr p0 = source.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = output.Pointer;
        IntPtr p3 = counts.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        Launch(arguments, OutputElements);
    }

    internal unsafe void LaunchMaxBackward(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView argmax,
        DirectPtxTensorView gradSource)
    {
        if (Operation != DirectPtxResidentScatterAuxOperation.MaxBackwardRows)
            throw new InvalidOperationException("This module does not expose resident scatter-max backward.");
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(argmax, Blueprint.Tensors[1], nameof(argmax));
        Require(gradSource, Blueprint.Tensors[2], nameof(gradSource));
        if (Overlaps(gradSource, gradOutput) || Overlaps(gradSource, argmax))
            throw new ArgumentException("Resident scatter gradient output must be disjoint from its inputs.", nameof(gradSource));
        IntPtr p0 = gradOutput.Pointer;
        IntPtr p1 = argmax.Pointer;
        IntPtr p2 = gradSource.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        Launch(arguments, SourceElements);
    }

    private unsafe void Launch(void** arguments, int elements) =>
        _module.Launch(_function, (uint)(elements / BlockThreads), 1, 1,
            BlockThreads, 1, 1, 0, arguments);

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxResidentScatterAuxOperation operation)
    {
        var source = new DirectPtxExtent(SourceRows, Features);
        var indices = new DirectPtxExtent(SourceRows);
        var grouped = new DirectPtxExtent(OutputRows, Features);
        IReadOnlyList<DirectPtxTensorContract> tensors =
            operation == DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts
                ?
                [
                    Exact("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                        source, DirectPtxTensorAccess.Read),
                    Exact("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                        indices, DirectPtxTensorAccess.Read),
                    Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                        grouped, DirectPtxTensorAccess.Write),
                    Exact("counts", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                        new DirectPtxExtent(OutputRows), DirectPtxTensorAccess.Write)
                ]
                :
                [
                    Exact("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                        grouped, DirectPtxTensorAccess.Read),
                    Exact("argmax", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.RowMajor2D,
                        grouped, DirectPtxTensorAccess.Read),
                    Exact("grad-source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                        source, DirectPtxTensorAccess.Write)
                ];
        return new DirectPtxKernelBlueprint(
            Operation: operation == DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts
                ? "resident-scatter-mean-rows-counts-f32"
                : "resident-scatter-max-backward-rows-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "src16384-g1024-f64",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction-order"] = "ascending-source-row",
                ["counts-type"] = operation == DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts
                    ? "float32-exact-integer" : "not-applicable",
                ["max-gradient-conflict"] = operation == DirectPtxResidentScatterAuxOperation.MaxBackwardRows
                    ? "last-matching-group-wins" : "not-applicable",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxResidentScatterAuxOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(6144);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        if (operation == DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts)
            EmitMean(ptx);
        else
            EmitMaxBackward(ptx);
        return ptx.ToString();
    }

    private static void EmitMean(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts),
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
        ptx.AppendLine("RESIDENT_MEAN_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {SourceRows};");
        ptx.AppendLine("    @%p0 bra RESIDENT_MEAN_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd5];");
        ptx.AppendLine("    setp.ne.u32 %p1, %r7, %r4;");
        ptx.AppendLine("    @%p1 bra RESIDENT_MEAN_NEXT;");
        ptx.AppendLine("    shl.b32 %r8, %r5, 6;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd7];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("RESIDENT_MEAN_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra RESIDENT_MEAN_LOOP;");
        ptx.AppendLine("RESIDENT_MEAN_DONE:");
        ptx.AppendLine("    cvt.rn.f32.u32 %f2, %r6;");
        ptx.AppendLine("    setp.eq.u32 %p2, %r6, 0;");
        ptx.AppendLine("    div.rn.f32 %f3, %f0, %f2;");
        ptx.AppendLine("    selp.f32 %f3, 0f00000000, %f3, %p2;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f3;");
        ptx.AppendLine("    setp.ne.u32 %p3, %r3, 0;");
        ptx.AppendLine("    @%p3 bra RESIDENT_MEAN_EXIT;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f2;");
        ptx.AppendLine("RESIDENT_MEAN_EXIT:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitMaxBackward(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxResidentScatterAuxOperation.MaxBackwardRows),
            "grad_output_ptr", "argmax_ptr", "grad_source_ptr");
        EmitRegisters(ptx);
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [argmax_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [grad_source_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine("    and.b32 %r3, %r2, 63;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 6;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("RESIDENT_MAX_BACKWARD_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {OutputRows};");
        ptx.AppendLine("    @%p0 bra RESIDENT_MAX_BACKWARD_DONE;");
        ptx.AppendLine("    shl.b32 %r6, %r5, 6;");
        ptx.AppendLine("    add.u32 %r6, %r6, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd4];");
        ptx.AppendLine("    setp.ne.u32 %p1, %r7, %r4;");
        ptx.AppendLine("    @%p1 bra RESIDENT_MAX_BACKWARD_NEXT;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd5];");
        ptx.AppendLine("RESIDENT_MAX_BACKWARD_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra RESIDENT_MAX_BACKWARD_LOOP;");
        ptx.AppendLine("RESIDENT_MAX_BACKWARD_DONE:");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f0;");
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
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
    }

    private static void EmitLinearIndex(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static string GetEntryPoint(DirectPtxResidentScatterAuxOperation operation) =>
        operation == DirectPtxResidentScatterAuxOperation.MeanRowsWithCounts
            ? "aidotnet_resident_scatter_mean_rows_counts_f32_src16384_g1024_f64"
            : "aidotnet_resident_scatter_max_backward_rows_f32_src16384_g1024_f64";

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
