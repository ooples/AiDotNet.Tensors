#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact deterministic graph aggregation. One thread owns four features of a
/// target node, scans edges in ascending order, retains four sums in registers,
/// and overwrites the final output once. Weighted/unweighted ABIs are separate.
/// </summary>
internal sealed class PtxGraphScatterAddDeterministicVec4F32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_graph_scatter_add_deterministic_vec4_f32_v1024_e16384_f64";
    internal const string WeightedEntryPoint = "aidotnet_graph_scatter_add_weighted_deterministic_vec4_f32_v1024_e16384_f64";
    internal const int Nodes = 1024;
    internal const int Edges = 16384;
    internal const int Features = 64;
    internal const int FeaturesPerThread = 4;
    internal const int FeatureGroups = Features / FeaturesPerThread;
    internal const int BlockThreads = 256;
    internal const int GridBlocks = Nodes * FeatureGroups / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal bool Weighted { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxGraphScatterAddDeterministicVec4F32Kernel(DirectPtxRuntime runtime, bool weighted)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Graph scatter-add has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Weighted = weighted;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, weighted);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, weighted);
        _module = runtime.LoadModule(Ptx);
        string entryPoint = weighted ? WeightedEntryPoint : EntryPoint;
        _function = _module.GetFunction(entryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(entryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int nodes, int edges, int features) =>
        nodes == Nodes && edges == Edges && features == Features;

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView sourceIndices,
        DirectPtxTensorView targetIndices,
        DirectPtxTensorView output)
    {
        if (Weighted) throw new InvalidOperationException("The weighted graph scatter ABI requires edge weights.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(sourceIndices, Blueprint.Tensors[1], nameof(sourceIndices));
        Require(targetIndices, Blueprint.Tensors[2], nameof(targetIndices));
        Require(output, Blueprint.Tensors[3], nameof(output));
        RequireDisjoint(output, input, sourceIndices, targetIndices);
        IntPtr inputPointer = input.Pointer;
        IntPtr sourcePointer = sourceIndices.Pointer;
        IntPtr targetPointer = targetIndices.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &inputPointer;
        arguments[1] = &sourcePointer;
        arguments[2] = &targetPointer;
        arguments[3] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchWeighted(
        DirectPtxTensorView input,
        DirectPtxTensorView sourceIndices,
        DirectPtxTensorView targetIndices,
        DirectPtxTensorView edgeWeights,
        DirectPtxTensorView output)
    {
        if (!Weighted) throw new InvalidOperationException("The unweighted graph scatter ABI has no edge weights.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(sourceIndices, Blueprint.Tensors[1], nameof(sourceIndices));
        Require(targetIndices, Blueprint.Tensors[2], nameof(targetIndices));
        Require(edgeWeights, Blueprint.Tensors[3], nameof(edgeWeights));
        Require(output, Blueprint.Tensors[4], nameof(output));
        RequireDisjoint(output, input, sourceIndices, targetIndices, edgeWeights);
        IntPtr inputPointer = input.Pointer;
        IntPtr sourcePointer = sourceIndices.Pointer;
        IntPtr targetPointer = targetIndices.Pointer;
        IntPtr weightsPointer = edgeWeights.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[5];
        arguments[0] = &inputPointer;
        arguments[1] = &sourcePointer;
        arguments[2] = &targetPointer;
        arguments[3] = &weightsPointer;
        arguments[4] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        bool weighted)
    {
        var nodes = new DirectPtxExtent(Nodes, Features);
        var edges = new DirectPtxExtent(Edges);
        var tensors = new List<DirectPtxTensorContract>
        {
            new("node-features", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                nodes, nodes, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
            new("source-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.GraphSourceIndices,
                edges, edges, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
            new("target-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.GraphTargetIndices,
                edges, edges, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact)
        };
        if (weighted)
            tensors.Add(new("edge-weights", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.GraphEdgeWeights,
                edges, edges, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact));
        tensors.Add(new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
            nodes, nodes, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact));
        return new DirectPtxKernelBlueprint(
            Operation: weighted
                ? "graph-scatter-add-weighted-deterministic-vec4-f32"
                : "graph-scatter-add-deterministic-vec4-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "v1024-e16384-f64-edge-order",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(48, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = weighted
                    ? "output[t,:]=sum_e(target[e]=t,input[source[e],:]*weight[e])"
                    : "output[t,:]=sum_e(target[e]=t,input[source[e],:])",
                ["reduction-order"] = "ascending-edge-index",
                ["output-initialization"] = "fused-zero-register-initialization",
                ["output-write"] = "single-overwrite",
                ["features-per-thread"] = FeaturesPerThread.ToString(),
                ["intermediate-global-bytes"] = "0",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor, bool weighted)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {(weighted ? WeightedEntryPoint : EntryPoint)}(");
        ptx.AppendLine("    .param .u64 node_features_ptr,");
        ptx.AppendLine("    .param .u64 source_indices_ptr,");
        ptx.AppendLine("    .param .u64 target_indices_ptr,");
        if (weighted) ptx.AppendLine("    .param .u64 edge_weights_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<10>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [node_features_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [source_indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [target_indices_ptr];");
        if (weighted) ptx.AppendLine("    ld.param.u64 %rd3, [edge_weights_ptr];");
        ptx.AppendLine($"    ld.param.u64 %rd4, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {FeatureGroups - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r4, 2;");
        ptx.AppendLine("    mov.u32 %r6, 0;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("EDGE_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r6, {Edges};");
        ptx.AppendLine("    @%p0 bra EDGE_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd2, %rd5;");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd6];");
        ptx.AppendLine("    setp.ne.u32 %p1, %r7, %r3;");
        ptx.AppendLine("    @%p1 bra EDGE_NEXT;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd5;");
        ptx.AppendLine("    ld.global.u32 %r8, [%rd7];");
        ptx.AppendLine("    shl.b32 %r9, %r8, 6;");
        ptx.AppendLine("    add.u32 %r9, %r9, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd0, %rd8;");
        ptx.AppendLine("    ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd9];");
        if (weighted)
        {
            ptx.AppendLine("    add.u64 %rd10, %rd3, %rd5;");
            ptx.AppendLine("    ld.global.f32 %f8, [%rd10];");
            ptx.AppendLine("    fma.rn.f32 %f0, %f4, %f8, %f0;");
            ptx.AppendLine("    fma.rn.f32 %f1, %f5, %f8, %f1;");
            ptx.AppendLine("    fma.rn.f32 %f2, %f6, %f8, %f2;");
            ptx.AppendLine("    fma.rn.f32 %f3, %f7, %f8, %f3;");
        }
        else
        {
            ptx.AppendLine("    add.rn.f32 %f0, %f0, %f4;");
            ptx.AppendLine("    add.rn.f32 %f1, %f1, %f5;");
            ptx.AppendLine("    add.rn.f32 %f2, %f2, %f6;");
            ptx.AppendLine("    add.rn.f32 %f3, %f3, %f7;");
        }
        ptx.AppendLine("EDGE_NEXT:");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra EDGE_LOOP;");
        ptx.AppendLine("EDGE_DONE:");
        ptx.AppendLine("    shl.b32 %r10, %r3, 6;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd11, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd12, %rd4, %rd11;");
        ptx.AppendLine("    st.global.v4.f32 [%rd12], {%f0, %f1, %f2, %f3};");
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

    private static void RequireDisjoint(
        DirectPtxTensorView output,
        DirectPtxTensorView input,
        DirectPtxTensorView sourceIndices,
        DirectPtxTensorView targetIndices,
        DirectPtxTensorView edgeWeights = default)
    {
        if (Overlaps(output, input) || Overlaps(output, sourceIndices) ||
            Overlaps(output, targetIndices) ||
            (edgeWeights.Pointer != IntPtr.Zero && Overlaps(output, edgeWeights)))
            throw new ArgumentException("Graph scatter-add output must be disjoint from every input.", nameof(output));
    }

    public void Dispose() => _module.Dispose();
}
#endif
