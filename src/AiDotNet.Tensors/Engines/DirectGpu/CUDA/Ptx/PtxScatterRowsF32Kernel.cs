#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxScatterRowsOperation
{
    AddAtomic,
    AddDeterministic,
    MeanAccumulateAtomic,
    MeanAccumulateDeterministic,
    MeanNormalize
}

/// <summary>
/// Exact row scatter family for 16384 source rows, 1024 destination rows,
/// and 64 contiguous FP32 features. Atomic and fixed-order modules preserve
/// the established CUDA numerical modes while removing all shape parameters.
/// </summary>
internal sealed class PtxScatterRowsF32Kernel : IDisposable
{
    internal const int SourceRows = 16384;
    internal const int DestinationRows = 1024;
    internal const int Features = 64;
    internal const int SourceElements = SourceRows * Features;
    internal const int DestinationElements = DestinationRows * Features;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxScatterRowsOperation Operation { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxScatterRowsF32Kernel(DirectPtxRuntime runtime, DirectPtxScatterRowsOperation operation)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Row scatter has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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

    internal static bool SupportsShape(int sourceElements, int destinationRows, int features) =>
        sourceElements == SourceElements && destinationRows == DestinationRows && features == Features;

    internal unsafe void LaunchThree(
        DirectPtxTensorView first,
        DirectPtxTensorView second,
        DirectPtxTensorView third)
    {
        if (Operation is not (DirectPtxScatterRowsOperation.AddAtomic or
            DirectPtxScatterRowsOperation.AddDeterministic))
            throw new InvalidOperationException("This row-scatter module requires another ABI.");
        Require(first, Blueprint.Tensors[0], nameof(first));
        Require(second, Blueprint.Tensors[1], nameof(second));
        Require(third, Blueprint.Tensors[2], nameof(third));
        RequireDisjoint(third, first, second);
        IntPtr p0 = first.Pointer;
        IntPtr p1 = second.Pointer;
        IntPtr p2 = third.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        Launch(arguments);
    }

    internal unsafe void LaunchFour(
        DirectPtxTensorView source,
        DirectPtxTensorView indices,
        DirectPtxTensorView destination,
        DirectPtxTensorView counts)
    {
        if (Operation is not (DirectPtxScatterRowsOperation.MeanAccumulateAtomic or
            DirectPtxScatterRowsOperation.MeanAccumulateDeterministic))
            throw new InvalidOperationException("This row-scatter module does not expose mean accumulation.");
        Require(source, Blueprint.Tensors[0], nameof(source));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(destination, Blueprint.Tensors[2], nameof(destination));
        Require(counts, Blueprint.Tensors[3], nameof(counts));
        RequireDisjoint(destination, source, indices, counts);
        if (Overlaps(counts, source) || Overlaps(counts, indices) || Overlaps(counts, destination))
            throw new ArgumentException("Scatter-mean counts must be disjoint from all other tensors.", nameof(counts));
        IntPtr p0 = source.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = destination.Pointer;
        IntPtr p3 = counts.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        Launch(arguments);
    }

    internal unsafe void LaunchNormalize(
        DirectPtxTensorView destination,
        DirectPtxTensorView counts)
    {
        if (Operation != DirectPtxScatterRowsOperation.MeanNormalize)
            throw new InvalidOperationException("This row-scatter module does not expose normalization.");
        Require(destination, Blueprint.Tensors[0], nameof(destination));
        Require(counts, Blueprint.Tensors[1], nameof(counts));
        if (Overlaps(destination, counts))
            throw new ArgumentException("Scatter-mean output and counts must be disjoint.", nameof(destination));
        IntPtr p0 = destination.Pointer;
        IntPtr p1 = counts.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &p0;
        arguments[1] = &p1;
        Launch(arguments);
    }

    private unsafe void Launch(void** arguments)
    {
        int workItems = Operation is DirectPtxScatterRowsOperation.AddAtomic or
            DirectPtxScatterRowsOperation.MeanAccumulateAtomic
            ? SourceElements : DestinationElements;
        _module.Launch(_function, (uint)(workItems / BlockThreads), 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxScatterRowsOperation operation)
    {
        var source = new DirectPtxExtent(SourceRows, Features);
        var indices = new DirectPtxExtent(SourceRows);
        var destination = new DirectPtxExtent(DestinationRows, Features);
        var counts = new DirectPtxExtent(DestinationRows);
        IReadOnlyList<DirectPtxTensorContract> tensors = operation switch
        {
            DirectPtxScatterRowsOperation.MeanNormalize =>
            [
                Exact("destination", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    destination, DirectPtxTensorAccess.ReadWrite),
                Exact("counts", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    counts, DirectPtxTensorAccess.Read)
            ],
            DirectPtxScatterRowsOperation.MeanAccumulateAtomic or
            DirectPtxScatterRowsOperation.MeanAccumulateDeterministic =>
            [
                Exact("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                    source, DirectPtxTensorAccess.Read),
                Exact("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                    indices, DirectPtxTensorAccess.Read),
                Exact("destination", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    destination, DirectPtxTensorAccess.ReadWrite),
                Exact("counts", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    counts, DirectPtxTensorAccess.ReadWrite)
            ],
            _ =>
            [
                Exact("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                    source, DirectPtxTensorAccess.Read),
                Exact("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                    indices, DirectPtxTensorAccess.Read),
                Exact("destination", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    destination, operation == DirectPtxScatterRowsOperation.AddAtomic
                        ? DirectPtxTensorAccess.ReadWrite : DirectPtxTensorAccess.Write)
            ]
        };
        return new DirectPtxKernelBlueprint(
            Operation: "scatter-rows-" + operation.ToString().ToLowerInvariant(),
            Version: 1,
            Architecture: architecture,
            Variant: "src16384-dst1024-f64",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(40, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction-order"] = operation is DirectPtxScatterRowsOperation.AddAtomic or
                    DirectPtxScatterRowsOperation.MeanAccumulateAtomic
                    ? "unordered-global-fp32-atomic" : "ascending-source-row",
                ["features"] = Features.ToString(),
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxScatterRowsOperation operation)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(5120);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        string[] parameters = operation switch
        {
            DirectPtxScatterRowsOperation.MeanNormalize => ["destination_ptr", "counts_ptr"],
            DirectPtxScatterRowsOperation.MeanAccumulateAtomic or
            DirectPtxScatterRowsOperation.MeanAccumulateDeterministic =>
                ["source_ptr", "indices_ptr", "destination_ptr", "counts_ptr"],
            _ => ["source_ptr", "indices_ptr", "destination_ptr"]
        };
        ptx.AppendLine($".visible .entry {GetEntryPoint(operation)}(");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b32 %r<18>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<6>;");
        if (operation == DirectPtxScatterRowsOperation.MeanNormalize)
            EmitNormalize(ptx);
        else if (operation is DirectPtxScatterRowsOperation.AddAtomic or
            DirectPtxScatterRowsOperation.MeanAccumulateAtomic)
            EmitAtomic(ptx, operation == DirectPtxScatterRowsOperation.MeanAccumulateAtomic);
        else
            EmitDeterministic(ptx, operation == DirectPtxScatterRowsOperation.MeanAccumulateDeterministic);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitAtomic(StringBuilder ptx, bool mean)
    {
        ptx.AppendLine("    ld.param.u64 %rd0, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [destination_ptr];");
        if (mean) ptx.AppendLine("    ld.param.u64 %rd3, [counts_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {Features - 1};");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd5];");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.u32 %r5, [%rd7];");
        ptx.AppendLine("    shl.b32 %r6, %r5, 6;");
        ptx.AppendLine("    add.u32 %r6, %r6, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    atom.global.add.f32 %f1, [%rd9], %f0;");
        if (mean)
        {
            ptx.AppendLine("    setp.ne.u32 %p0, %r4, 0;");
            ptx.AppendLine("    @%p0 bra MEAN_ATOMIC_DONE;");
            ptx.AppendLine("    mul.wide.u32 %rd10, %r5, 4;");
            ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
            ptx.AppendLine("    atom.global.add.u32 %r7, [%rd11], 1;");
            ptx.AppendLine("MEAN_ATOMIC_DONE:");
        }
    }

    private static void EmitDeterministic(StringBuilder ptx, bool mean)
    {
        ptx.AppendLine("    ld.param.u64 %rd0, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [destination_ptr];");
        if (mean) ptx.AppendLine("    ld.param.u64 %rd3, [counts_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {Features - 1};");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("ROW_SCATTER_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {SourceRows};");
        ptx.AppendLine("    @%p0 bra ROW_SCATTER_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd5];");
        ptx.AppendLine("    setp.ne.u32 %p1, %r7, %r3;");
        ptx.AppendLine("    @%p1 bra ROW_SCATTER_NEXT;");
        ptx.AppendLine("    shl.b32 %r8, %r5, 6;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd7];");
        ptx.AppendLine("    add.rn.f32 %f0, %f0, %f1;");
        if (mean)
        {
            ptx.AppendLine("    setp.eq.u32 %p2, %r4, 0;");
            ptx.AppendLine("    @%p2 add.u32 %r6, %r6, 1;");
        }
        ptx.AppendLine("ROW_SCATTER_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra ROW_SCATTER_LOOP;");
        ptx.AppendLine("ROW_SCATTER_DONE:");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        if (mean)
        {
            ptx.AppendLine("    ld.global.f32 %f2, [%rd9];");
            ptx.AppendLine("    add.rn.f32 %f0, %f2, %f0;");
        }
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        if (mean)
        {
            ptx.AppendLine("    setp.ne.u32 %p3, %r4, 0;");
            ptx.AppendLine("    @%p3 bra ROW_COUNT_DONE;");
            ptx.AppendLine("    mul.wide.u32 %rd10, %r3, 4;");
            ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
            ptx.AppendLine("    ld.global.u32 %r9, [%rd11];");
            ptx.AppendLine("    add.u32 %r9, %r9, %r6;");
            ptx.AppendLine("    st.global.u32 [%rd11], %r9;");
            ptx.AppendLine("ROW_COUNT_DONE:");
        }
    }

    private static void EmitNormalize(StringBuilder ptx)
    {
        ptx.AppendLine("    ld.param.u64 %rd0, [destination_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [counts_ptr];");
        EmitLinearIndex(ptx);
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd1, %rd2;");
        ptx.AppendLine("    ld.global.u32 %r4, [%rd3];");
        ptx.AppendLine("    setp.eq.u32 %p0, %r4, 0;");
        ptx.AppendLine("    @%p0 bra NORMALIZE_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd5];");
        ptx.AppendLine("    cvt.rn.f32.u32 %f1, %r4;");
        ptx.AppendLine("    div.rn.f32 %f0, %f0, %f1;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f0;");
        ptx.AppendLine("NORMALIZE_DONE:");
    }

    private static void EmitLinearIndex(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
    }

    private static string GetEntryPoint(DirectPtxScatterRowsOperation operation) => operation switch
    {
        DirectPtxScatterRowsOperation.AddAtomic => "aidotnet_scatter_add_batched_atomic_f32_r16384_d1024_f64",
        DirectPtxScatterRowsOperation.AddDeterministic =>
            "aidotnet_scatter_add_batched_deterministic_f32_r16384_d1024_f64",
        DirectPtxScatterRowsOperation.MeanAccumulateAtomic =>
            "aidotnet_scatter_mean_accumulate_atomic_f32_r16384_d1024_f64",
        DirectPtxScatterRowsOperation.MeanAccumulateDeterministic =>
            "aidotnet_scatter_mean_accumulate_deterministic_f32_r16384_d1024_f64",
        DirectPtxScatterRowsOperation.MeanNormalize =>
            "aidotnet_scatter_mean_normalize_f32_d1024_f64",
        _ => throw new ArgumentOutOfRangeException(nameof(operation))
    };

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

    private static void RequireDisjoint(
        DirectPtxTensorView output,
        DirectPtxTensorView input0,
        DirectPtxTensorView input1,
        DirectPtxTensorView input2 = default)
    {
        if (Overlaps(output, input0) || Overlaps(output, input1) ||
            (input2.Pointer != IntPtr.Zero && Overlaps(output, input2)))
            throw new ArgumentException("Row-scatter output must be disjoint from every input.", nameof(output));
    }

    public void Dispose() => _module.Dispose();
}
#endif
