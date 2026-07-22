#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxResidentScatterSoftmaxOperation
{
    Forward,
    Backward
}

/// <summary>
/// Exact grouped scatter-softmax forward/backward kernels. A group-feature
/// thread owns the full reduction and all matching output writes, eliminating
/// the reference kernel's repeated reduction for every source row.
/// </summary>
internal sealed class PtxResidentScatterSoftmaxF32Kernel : IDisposable
{
    internal const int SourceRows = 16384;
    internal const int Groups = 1024;
    internal const int Features = 64;
    internal const int SourceElements = SourceRows * Features;
    internal const int GroupElements = Groups * Features;
    internal const int WorkItems = SourceElements + GroupElements;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxResidentScatterSoftmaxOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxResidentScatterSoftmaxF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxResidentScatterSoftmaxOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Resident scatter softmax has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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

    internal static bool SupportsShape(int sourceRows, int features, int groups) =>
        sourceRows == SourceRows && features == Features && groups == Groups;

    internal unsafe void LaunchForward(
        DirectPtxTensorView source,
        DirectPtxTensorView indices,
        DirectPtxTensorView output)
    {
        if (Operation != DirectPtxResidentScatterSoftmaxOperation.Forward)
            throw new InvalidOperationException("This module does not expose scatter-softmax forward.");
        Require(source, Blueprint.Tensors[0], nameof(source));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, source) || Overlaps(output, indices))
            throw new ArgumentException("Scatter-softmax output must be disjoint from its inputs.", nameof(output));
        IntPtr p0 = source.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        Launch(arguments);
    }

    internal unsafe void LaunchBackward(
        DirectPtxTensorView gradOutput,
        DirectPtxTensorView output,
        DirectPtxTensorView indices,
        DirectPtxTensorView gradSource)
    {
        if (Operation != DirectPtxResidentScatterSoftmaxOperation.Backward)
            throw new InvalidOperationException("This module does not expose scatter-softmax backward.");
        Require(gradOutput, Blueprint.Tensors[0], nameof(gradOutput));
        Require(output, Blueprint.Tensors[1], nameof(output));
        Require(indices, Blueprint.Tensors[2], nameof(indices));
        Require(gradSource, Blueprint.Tensors[3], nameof(gradSource));
        if (Overlaps(gradSource, gradOutput) || Overlaps(gradSource, output) ||
            Overlaps(gradSource, indices))
            throw new ArgumentException("Scatter-softmax gradient output must be disjoint from its inputs.", nameof(gradSource));
        IntPtr p0 = gradOutput.Pointer;
        IntPtr p1 = output.Pointer;
        IntPtr p2 = indices.Pointer;
        IntPtr p3 = gradSource.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        Launch(arguments);
    }

    private unsafe void Launch(void** arguments) =>
        _module.Launch(_function, WorkItems / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxResidentScatterSoftmaxOperation operation)
    {
        var rows = new DirectPtxExtent(SourceRows, Features);
        var indices = new DirectPtxExtent(SourceRows);
        IReadOnlyList<DirectPtxTensorContract> tensors =
            operation == DirectPtxResidentScatterSoftmaxOperation.Forward
                ?
                [
                    Exact("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                        rows, DirectPtxTensorAccess.Read),
                    Exact("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                        indices, DirectPtxTensorAccess.Read),
                    Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                        rows, DirectPtxTensorAccess.Write)
                ]
                :
                [
                    Exact("grad-output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                        rows, DirectPtxTensorAccess.Read),
                    Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                        rows, DirectPtxTensorAccess.Read),
                    Exact("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                        indices, DirectPtxTensorAccess.Read),
                    Exact("grad-source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                        rows, DirectPtxTensorAccess.Write)
                ];
        return new DirectPtxKernelBlueprint(
            Operation: operation == DirectPtxResidentScatterSoftmaxOperation.Forward
                ? "resident-scatter-softmax-rows-f32"
                : "resident-scatter-softmax-backward-rows-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "src16384-g1024-f64-group-owned",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(40, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["ownership"] = "one-thread-per-group-feature-plus-invalid-row-zeroing",
                ["reduction-order"] = "ascending-source-row",
                ["exponential"] = "ex2-approx-log2e-scaled",
                ["invalid-index"] = "zero",
                ["kernel-launches"] = "1",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxResidentScatterSoftmaxOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(8192);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        if (operation == DirectPtxResidentScatterSoftmaxOperation.Forward)
            EmitForward(ptx);
        else
            EmitBackward(ptx);
        return ptx.ToString();
    }

    private static void EmitForward(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxResidentScatterSoftmaxOperation.Forward),
            "source_ptr", "indices_ptr", "output_ptr");
        EmitRegisters(ptx);
        ptx.AppendLine("    ld.param.u64 %rd0, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine($"    setp.lt.u32 %p0, %r2, {GroupElements};");
        ptx.AppendLine("    @%p0 bra SOFTMAX_GROUP;");
        EmitInvalidZeroRegion(ptx, "SOFTMAX_EXIT", "%rd2");
        ptx.AppendLine("SOFTMAX_GROUP:");
        ptx.AppendLine("    and.b32 %r3, %r2, 63;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 6;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0ff800000;");
        ptx.AppendLine("SOFTMAX_MAX_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r5, {SourceRows};");
        ptx.AppendLine("    @%p1 bra SOFTMAX_MAX_DONE;");
        EmitMatchAndValueLoad(ptx, "SOFTMAX_MAX_NEXT", "%rd0", "%f1");
        ptx.AppendLine("    max.f32 %f0, %f0, %f1;");
        ptx.AppendLine("SOFTMAX_MAX_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra SOFTMAX_MAX_LOOP;");
        ptx.AppendLine("SOFTMAX_MAX_DONE:");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("SOFTMAX_SUM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r5, {SourceRows};");
        ptx.AppendLine("    @%p1 bra SOFTMAX_SUM_DONE;");
        EmitMatchAndValueLoad(ptx, "SOFTMAX_SUM_NEXT", "%rd0", "%f1");
        EmitExpDifference(ptx, "%f1", "%f0", "%f3");
        ptx.AppendLine("    add.rn.f32 %f2, %f2, %f3;");
        ptx.AppendLine("SOFTMAX_SUM_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra SOFTMAX_SUM_LOOP;");
        ptx.AppendLine("SOFTMAX_SUM_DONE:");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("SOFTMAX_WRITE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r5, {SourceRows};");
        ptx.AppendLine("    @%p1 bra SOFTMAX_EXIT;");
        EmitMatchAndValueLoad(ptx, "SOFTMAX_WRITE_NEXT", "%rd0", "%f1");
        EmitExpDifference(ptx, "%f1", "%f0", "%f3");
        ptx.AppendLine("    div.rn.f32 %f3, %f3, %f2;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f3;");
        ptx.AppendLine("SOFTMAX_WRITE_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra SOFTMAX_WRITE_LOOP;");
        ptx.AppendLine("SOFTMAX_EXIT:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitBackward(StringBuilder ptx)
    {
        EmitHeader(ptx, GetEntryPoint(DirectPtxResidentScatterSoftmaxOperation.Backward),
            "grad_output_ptr", "output_ptr", "indices_ptr", "grad_source_ptr");
        EmitRegisters(ptx);
        ptx.AppendLine("    ld.param.u64 %rd0, [grad_output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd12, [grad_source_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine($"    setp.lt.u32 %p0, %r2, {GroupElements};");
        ptx.AppendLine("    @%p0 bra SOFTMAX_BACKWARD_GROUP;");
        EmitInvalidZeroRegion(ptx, "SOFTMAX_BACKWARD_EXIT", "%rd12", "%rd2");
        ptx.AppendLine("SOFTMAX_BACKWARD_GROUP:");
        ptx.AppendLine("    and.b32 %r3, %r2, 63;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 6;");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("SOFTMAX_BACKWARD_SUM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r5, {SourceRows};");
        ptx.AppendLine("    @%p1 bra SOFTMAX_BACKWARD_SUM_DONE;");
        EmitMatchAndOffset(ptx, "SOFTMAX_BACKWARD_SUM_NEXT", "%rd2");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd8];");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd9];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");
        ptx.AppendLine("SOFTMAX_BACKWARD_SUM_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra SOFTMAX_BACKWARD_SUM_LOOP;");
        ptx.AppendLine("SOFTMAX_BACKWARD_SUM_DONE:");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("SOFTMAX_BACKWARD_WRITE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r5, {SourceRows};");
        ptx.AppendLine("    @%p1 bra SOFTMAX_BACKWARD_EXIT;");
        EmitMatchAndOffset(ptx, "SOFTMAX_BACKWARD_WRITE_NEXT", "%rd2");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd8];");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f2, [%rd9];");
        ptx.AppendLine("    sub.rn.f32 %f3, %f2, %f0;");
        ptx.AppendLine("    mul.rn.f32 %f3, %f1, %f3;");
        ptx.AppendLine("    add.u64 %rd10, %rd12, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f3;");
        ptx.AppendLine("SOFTMAX_BACKWARD_WRITE_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra SOFTMAX_BACKWARD_WRITE_LOOP;");
        ptx.AppendLine("SOFTMAX_BACKWARD_EXIT:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
    }

    private static void EmitInvalidZeroRegion(
        StringBuilder ptx,
        string exitLabel,
        string outputPointer,
        string indicesPointer = "%rd1")
    {
        ptx.AppendLine($"    sub.u32 %r3, %r2, {GroupElements};");
        ptx.AppendLine("    shr.u32 %r4, %r3, 6;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r4, 4;");
        ptx.AppendLine($"    add.u64 %rd5, {indicesPointer}, %rd4;");
        ptx.AppendLine("    ld.global.s32 %r5, [%rd5];");
        ptx.AppendLine("    setp.lt.s32 %p1, %r5, 0;");
        ptx.AppendLine("    setp.ge.s32 %p2, %r5, 1024;");
        ptx.AppendLine("    or.pred %p3, %p1, %p2;");
        ptx.AppendLine($"    @!%p3 bra {exitLabel};");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine($"    add.u64 %rd7, {outputPointer}, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd7], 0f00000000;");
        ptx.AppendLine($"    bra {exitLabel};");
    }

    private static void EmitMatchAndValueLoad(
        StringBuilder ptx,
        string nextLabel,
        string sourcePointer,
        string valueRegister)
    {
        EmitMatchAndOffset(ptx, nextLabel, "%rd1");
        ptx.AppendLine($"    add.u64 %rd7, {sourcePointer}, %rd6;");
        ptx.AppendLine($"    ld.global.f32 {valueRegister}, [%rd7];");
    }

    private static void EmitMatchAndOffset(
        StringBuilder ptx,
        string nextLabel,
        string indicesPointer)
    {
        ptx.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
        ptx.AppendLine($"    add.u64 %rd4, {indicesPointer}, %rd3;");
        ptx.AppendLine("    ld.global.u32 %r6, [%rd4];");
        ptx.AppendLine("    setp.ne.u32 %p2, %r6, %r4;");
        ptx.AppendLine($"    @%p2 bra {nextLabel};");
        ptx.AppendLine("    shl.b32 %r7, %r5, 6;");
        ptx.AppendLine("    add.u32 %r7, %r7, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r7, 4;");
    }

    private static void EmitExpDifference(
        StringBuilder ptx,
        string value,
        string maximum,
        string result)
    {
        ptx.AppendLine($"    sub.rn.f32 {result}, {value}, {maximum};");
        ptx.AppendLine($"    mul.rn.f32 {result}, {result}, 0f3fb8aa3b;");
        ptx.AppendLine($"    ex2.approx.f32 {result}, {result};");
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
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
    }

    private static void EmitLinearIndex(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static string GetEntryPoint(DirectPtxResidentScatterSoftmaxOperation operation) =>
        operation == DirectPtxResidentScatterSoftmaxOperation.Forward
            ? "aidotnet_resident_scatter_softmax_rows_f32_src16384_g1024_f64"
            : "aidotnet_resident_scatter_softmax_backward_rows_f32_src16384_g1024_f64";

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
