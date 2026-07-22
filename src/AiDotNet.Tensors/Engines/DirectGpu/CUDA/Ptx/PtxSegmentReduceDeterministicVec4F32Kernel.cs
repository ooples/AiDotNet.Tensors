#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact deterministic segmented reduction. One thread owns four features of
/// a segment, scans items in ascending order, and overwrites output once.
/// </summary>
internal sealed class PtxSegmentReduceDeterministicVec4F32Kernel : IDisposable
{
    internal const string SumEntryPoint = "aidotnet_segment_sum_deterministic_vec4_f32_i16384_s1024_f64";
    internal const string MeanEntryPoint = "aidotnet_segment_mean_deterministic_vec4_f32_i16384_s1024_f64";
    internal const string MaxEntryPoint = "aidotnet_segment_max_deterministic_vec4_f32_i16384_s1024_f64";
    internal const int Items = 16384;
    internal const int Segments = 1024;
    internal const int Features = 64;
    internal const int FeaturesPerThread = 4;
    internal const int FeatureGroups = Features / FeaturesPerThread;
    internal const int BlockThreads = 256;
    internal const int GridBlocks = Segments * FeatureGroups / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxSegmentReduction Reduction { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSegmentReduceDeterministicVec4F32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxSegmentReduction reduction)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Segmented reduction has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Reduction = reduction;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, reduction);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, reduction);
        _module = runtime.LoadModule(Ptx);
        string entryPoint = GetEntryPoint(reduction);
        _function = _module.GetFunction(entryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(entryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int items, int segments, int features) =>
        items == Items && segments == Segments && features == Features;

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView segmentIds,
        DirectPtxTensorView output)
    {
        if (Reduction == DirectPtxSegmentReduction.Mean)
            throw new InvalidOperationException("Segment mean requires the segment-sizes ABI.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(segmentIds, Blueprint.Tensors[1], nameof(segmentIds));
        Require(output, Blueprint.Tensors[2], nameof(output));
        RequireDisjoint(output, input, segmentIds);
        IntPtr inputPointer = input.Pointer;
        IntPtr idsPointer = segmentIds.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &idsPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchMean(
        DirectPtxTensorView input,
        DirectPtxTensorView segmentIds,
        DirectPtxTensorView segmentSizes,
        DirectPtxTensorView output)
    {
        if (Reduction != DirectPtxSegmentReduction.Mean)
            throw new InvalidOperationException("Only segment mean accepts segment sizes.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(segmentIds, Blueprint.Tensors[1], nameof(segmentIds));
        Require(segmentSizes, Blueprint.Tensors[2], nameof(segmentSizes));
        Require(output, Blueprint.Tensors[3], nameof(output));
        RequireDisjoint(output, input, segmentIds, segmentSizes);
        IntPtr inputPointer = input.Pointer;
        IntPtr idsPointer = segmentIds.Pointer;
        IntPtr sizesPointer = segmentSizes.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &idsPointer;
        arguments[2] = &sizesPointer;
        arguments[3] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxSegmentReduction reduction)
    {
        var input = new DirectPtxExtent(Items, Features);
        var items = new DirectPtxExtent(Items);
        var segments = new DirectPtxExtent(Segments);
        var output = new DirectPtxExtent(Segments, Features);
        var tensors = new List<DirectPtxTensorContract>
        {
            new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
            new("segment-ids", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.SegmentIds,
                items, items, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact)
        };
        if (reduction == DirectPtxSegmentReduction.Mean)
            tensors.Add(new("segment-sizes", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.SegmentSizes,
                segments, segments, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
        tensors.Add(new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
            output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact));
        string operation = reduction switch
        {
            DirectPtxSegmentReduction.Sum => "segment-sum-deterministic-vec4-f32",
            DirectPtxSegmentReduction.Mean => "segment-mean-deterministic-vec4-f32",
            DirectPtxSegmentReduction.Max => "segment-max-deterministic-vec4-f32",
            _ => throw new ArgumentOutOfRangeException(nameof(reduction))
        };
        return new DirectPtxKernelBlueprint(
            Operation: operation,
            Version: 1,
            Architecture: architecture,
            Variant: "i16384-s1024-f64-item-order",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(48, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction"] = reduction.ToString().ToLowerInvariant(),
                ["reduction-order"] = "ascending-item-index",
                ["empty-segment"] = reduction == DirectPtxSegmentReduction.Max ? "-FLT_MAX" : "zero",
                ["output-initialization"] = "fused-register-initialization",
                ["output-write"] = "single-overwrite",
                ["intermediate-global-bytes"] = "0",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxSegmentReduction reduction)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        bool mean = reduction == DirectPtxSegmentReduction.Mean;
        bool max = reduction == DirectPtxSegmentReduction.Max;
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(reduction)}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 segment_ids_ptr,");
        if (mean) ptx.AppendLine("    .param .u64 segment_sizes_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<18>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [segment_ids_ptr];");
        if (mean) ptx.AppendLine("    ld.param.u64 %rd2, [segment_sizes_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {FeatureGroups - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r4, 2;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        string initial = max ? "0dffefffffffffffff" : "0f00000000";
        if (max)
        {
            ptx.AppendLine("    mov.b32 %f0, 0xff7fffff;");
            ptx.AppendLine("    mov.b32 %f1, 0xff7fffff;");
            ptx.AppendLine("    mov.b32 %f2, 0xff7fffff;");
            ptx.AppendLine("    mov.b32 %f3, 0xff7fffff;");
        }
        else
        {
            ptx.AppendLine($"    mov.f32 %f0, {initial};");
            ptx.AppendLine($"    mov.f32 %f1, {initial};");
            ptx.AppendLine($"    mov.f32 %f2, {initial};");
            ptx.AppendLine($"    mov.f32 %f3, {initial};");
        }
        if (mean)
        {
            ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
            ptx.AppendLine("    add.u64 %rd5, %rd2, %rd4;");
            ptx.AppendLine("    ld.global.s32 %r7, [%rd5];");
            ptx.AppendLine("    setp.le.s32 %p2, %r7, 0;");
            ptx.AppendLine("    @%p2 bra REDUCE_DONE;");
            ptx.AppendLine("    cvt.rn.f32.s32 %f8, %r7;");
            ptx.AppendLine("    div.rn.f32 %f9, 0f3f800000, %f8;");
        }
        ptx.AppendLine("ITEM_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r6, {Items};");
        ptx.AppendLine("    @%p0 bra REDUCE_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.u32 %r8, [%rd7];");
        ptx.AppendLine("    setp.ne.u32 %p1, %r8, %r3;");
        ptx.AppendLine("    @%p1 bra ITEM_NEXT;");
        ptx.AppendLine("    shl.b32 %r9, %r6, 6;");
        ptx.AppendLine("    add.u32 %r9, %r9, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd8;");
        ptx.AppendLine("    ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd9];");
        string op = max ? "max.f32" : mean ? "fma.rn.f32" : "add.rn.f32";
        for (int lane = 0; lane < 4; lane++)
        {
            if (mean) ptx.AppendLine($"    {op} %f{lane}, %f{lane + 4}, %f9, %f{lane};");
            else ptx.AppendLine($"    {op} %f{lane}, %f{lane}, %f{lane + 4};");
        }
        ptx.AppendLine("ITEM_NEXT:");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra ITEM_LOOP;");
        ptx.AppendLine("REDUCE_DONE:");
        ptx.AppendLine("    shl.b32 %r10, %r3, 6;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd10;");
        ptx.AppendLine("    st.global.v4.f32 [%rd11], {%f0, %f1, %f2, %f3};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string GetEntryPoint(DirectPtxSegmentReduction reduction) => reduction switch
    {
        DirectPtxSegmentReduction.Sum => SumEntryPoint,
        DirectPtxSegmentReduction.Mean => MeanEntryPoint,
        DirectPtxSegmentReduction.Max => MaxEntryPoint,
        _ => throw new ArgumentOutOfRangeException(nameof(reduction))
    };

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
        DirectPtxTensorView input,
        DirectPtxTensorView segmentIds,
        DirectPtxTensorView segmentSizes = default)
    {
        if (Overlaps(output, input) || Overlaps(output, segmentIds) ||
            (segmentSizes.Pointer != IntPtr.Zero && Overlaps(output, segmentSizes)))
            throw new ArgumentException("Segment output must be disjoint from every input.", nameof(output));
    }

    public void Dispose() => _module.Dispose();
}
#endif
