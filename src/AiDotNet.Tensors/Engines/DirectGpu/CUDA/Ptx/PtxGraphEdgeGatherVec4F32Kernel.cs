#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact graph edge gather. Source/target direction is baked into the module
/// identity and physical index contract; one thread transfers four features.
/// </summary>
internal sealed class PtxGraphEdgeGatherVec4F32Kernel : IDisposable
{
    internal const string SourceEntryPoint = "aidotnet_graph_gather_source_vec4_f32_v1024_e16384_f64";
    internal const string TargetEntryPoint = "aidotnet_graph_gather_target_vec4_f32_v1024_e16384_f64";
    internal const int Nodes = 1024;
    internal const int Edges = 16384;
    internal const int Features = 64;
    internal const int FeaturesPerThread = 4;
    internal const int FeatureGroups = Features / FeaturesPerThread;
    internal const int BlockThreads = 256;
    internal const int GridBlocks = Edges * FeatureGroups / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal bool GatherSource { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }
    internal string JitInfoLog => _module.JitInfoLog;

    internal PtxGraphEdgeGatherVec4F32Kernel(DirectPtxRuntime runtime, bool gatherSource)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Graph edge gather has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        GatherSource = gatherSource;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, gatherSource);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, gatherSource);
        _module = runtime.LoadModule(Ptx);
        string entryPoint = gatherSource ? SourceEntryPoint : TargetEntryPoint;
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
        DirectPtxTensorView nodeFeatures,
        DirectPtxTensorView edgeNodeIndices,
        DirectPtxTensorView edgeFeatures)
    {
        Require(nodeFeatures, Blueprint.Tensors[0], nameof(nodeFeatures));
        Require(edgeNodeIndices, Blueprint.Tensors[1], nameof(edgeNodeIndices));
        Require(edgeFeatures, Blueprint.Tensors[2], nameof(edgeFeatures));
        if (Overlaps(edgeFeatures, nodeFeatures) || Overlaps(edgeFeatures, edgeNodeIndices))
            throw new ArgumentException("Graph gather output must be disjoint from both inputs.", nameof(edgeFeatures));

        IntPtr featuresPointer = nodeFeatures.Pointer;
        IntPtr indicesPointer = edgeNodeIndices.Pointer;
        IntPtr outputPointer = edgeFeatures.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &featuresPointer;
        arguments[1] = &indicesPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        bool gatherSource)
    {
        var nodeFeatures = new DirectPtxExtent(Nodes, Features);
        var indices = new DirectPtxExtent(Edges);
        var edgeFeatures = new DirectPtxExtent(Edges, Features);
        return new DirectPtxKernelBlueprint(
            Operation: gatherSource ? "graph-gather-source-vec4-f32" : "graph-gather-target-vec4-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "v1024-e16384-f64-row-major",
            Tensors:
            [
                new("node-features", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    nodeFeatures, nodeFeatures, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new(gatherSource ? "source-indices" : "target-indices", DirectPtxPhysicalType.Int32,
                    gatherSource ? DirectPtxPhysicalLayout.GraphSourceIndices : DirectPtxPhysicalLayout.GraphTargetIndices,
                    indices, indices, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("edge-features", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                    edgeFeatures, edgeFeatures, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = gatherSource
                    ? "edgeFeatures[e,:]=nodeFeatures[source[e],:]"
                    : "edgeFeatures[e,:]=nodeFeatures[target[e],:]",
                ["index-base"] = "zero",
                ["index-width"] = "int32",
                ["node-layout"] = "row-major-contiguous",
                ["edge-layout"] = "row-major-contiguous",
                ["features-per-thread"] = FeaturesPerThread.ToString(),
                ["intermediate-global-bytes"] = "0",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor, bool gatherSource)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(2048);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {(gatherSource ? SourceEntryPoint : TargetEntryPoint)}(");
        ptx.AppendLine("    .param .u64 node_features_ptr,");
        ptx.AppendLine("    .param .u64 edge_node_indices_ptr,");
        ptx.AppendLine("    .param .u64 edge_features_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [node_features_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [edge_node_indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [edge_features_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {FeatureGroups - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r4, 2;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.u32 %r6, [%rd4];");
        ptx.AppendLine("    shl.b32 %r7, %r6, 6;");
        ptx.AppendLine("    add.u32 %r7, %r7, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd6];");
        ptx.AppendLine("    shl.b32 %r8, %r3, 6;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f0, %f1, %f2, %f3};");
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
