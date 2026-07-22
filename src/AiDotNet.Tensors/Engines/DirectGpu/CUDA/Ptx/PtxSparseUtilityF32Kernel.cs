#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxSparseUtility
{
    FillZero,
    FillNegativeInfinity,
    DegreeNormalize,
    SymmetricDegreeNormalize
}

/// <summary>Exact sparse initialization and normalization utility modules.</summary>
internal sealed class PtxSparseUtilityF32Kernel : IDisposable
{
    internal const string ZeroEntryPoint = "aidotnet_sparse_fill_zero_vec4_f32_e65536";
    internal const string NegativeInfinityEntryPoint = "aidotnet_sparse_fill_neg_inf_vec4_f32_e65536";
    internal const string DegreeEntryPoint = "aidotnet_degree_normalize_vec4_f32_v1024_f64";
    internal const string SymmetricEntryPoint = "aidotnet_symmetric_degree_normalize_f32_v1024_e16384";
    internal const int Nodes = 1024;
    internal const int Edges = 16384;
    internal const int Features = 64;
    internal const int Elements = Nodes * Features;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxSparseUtility Utility { get; }
    internal float Epsilon { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxSparseUtilityF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxSparseUtility utility,
        float epsilon = 1e-8f)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Sparse utility has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        if ((utility is DirectPtxSparseUtility.DegreeNormalize or
            DirectPtxSparseUtility.SymmetricDegreeNormalize) &&
            (!(epsilon >= 0f) || float.IsInfinity(epsilon)))
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        Utility = utility;
        Epsilon = epsilon;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, utility, epsilon);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, utility, epsilon);
        _module = runtime.LoadModule(Ptx);
        string entryPoint = GetEntryPoint(utility);
        _function = _module.GetFunction(entryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(entryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void LaunchFill(DirectPtxTensorView output)
    {
        if (Utility is not (DirectPtxSparseUtility.FillZero or DirectPtxSparseUtility.FillNegativeInfinity))
            throw new InvalidOperationException("This sparse utility is not a fill module.");
        Require(output, Blueprint.Tensors[0], nameof(output));
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[1];
        arguments[0] = &outputPointer;
        _module.Launch(_function, Elements / 4 / BlockThreads, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchDegree(
        DirectPtxTensorView input,
        DirectPtxTensorView degrees,
        DirectPtxTensorView output)
    {
        if (Utility != DirectPtxSparseUtility.DegreeNormalize)
            throw new InvalidOperationException("This sparse utility is not degree normalization.");
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(degrees, Blueprint.Tensors[1], nameof(degrees));
        Require(output, Blueprint.Tensors[2], nameof(output));
        RequireDisjoint(output, input, degrees);
        IntPtr inputPointer = input.Pointer;
        IntPtr degreesPointer = degrees.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &degreesPointer;
        arguments[2] = &outputPointer;
        _module.Launch(_function, Elements / 4 / BlockThreads, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal unsafe void LaunchSymmetric(
        DirectPtxTensorView edgeValues,
        DirectPtxTensorView sourceIndices,
        DirectPtxTensorView targetIndices,
        DirectPtxTensorView sourceDegrees,
        DirectPtxTensorView targetDegrees,
        DirectPtxTensorView output)
    {
        if (Utility != DirectPtxSparseUtility.SymmetricDegreeNormalize)
            throw new InvalidOperationException("This sparse utility is not symmetric degree normalization.");
        Require(edgeValues, Blueprint.Tensors[0], nameof(edgeValues));
        Require(sourceIndices, Blueprint.Tensors[1], nameof(sourceIndices));
        Require(targetIndices, Blueprint.Tensors[2], nameof(targetIndices));
        Require(sourceDegrees, Blueprint.Tensors[3], nameof(sourceDegrees));
        Require(targetDegrees, Blueprint.Tensors[4], nameof(targetDegrees));
        Require(output, Blueprint.Tensors[5], nameof(output));
        if (Overlaps(output, edgeValues) || Overlaps(output, sourceIndices) ||
            Overlaps(output, targetIndices) || Overlaps(output, sourceDegrees) ||
            Overlaps(output, targetDegrees))
            throw new ArgumentException("Symmetric normalization output must be disjoint from inputs.", nameof(output));
        IntPtr p0 = edgeValues.Pointer;
        IntPtr p1 = sourceIndices.Pointer;
        IntPtr p2 = targetIndices.Pointer;
        IntPtr p3 = sourceDegrees.Pointer;
        IntPtr p4 = targetDegrees.Pointer;
        IntPtr p5 = output.Pointer;
        void** arguments = stackalloc void*[6];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        arguments[4] = &p4;
        arguments[5] = &p5;
        _module.Launch(_function, Edges / BlockThreads, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxSparseUtility utility,
        float epsilon)
    {
        var elements = new DirectPtxExtent(Elements);
        var nodes = new DirectPtxExtent(Nodes);
        var matrix = new DirectPtxExtent(Nodes, Features);
        var edges = new DirectPtxExtent(Edges);
        IReadOnlyList<DirectPtxTensorContract> tensors = utility switch
        {
            DirectPtxSparseUtility.FillZero or DirectPtxSparseUtility.FillNegativeInfinity =>
            [
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    elements, elements, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            DirectPtxSparseUtility.DegreeNormalize =>
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("degrees", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    nodes, nodes, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    matrix, matrix, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            DirectPtxSparseUtility.SymmetricDegreeNormalize =>
            [
                new("edge-values", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.GraphEdgeWeights,
                    edges, edges, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("source-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.GraphSourceIndices,
                    edges, edges, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("target-indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.GraphTargetIndices,
                    edges, edges, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("source-degrees", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    nodes, nodes, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("target-degrees", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    nodes, nodes, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.GraphEdgeWeights,
                    edges, edges, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            _ => throw new ArgumentOutOfRangeException(nameof(utility))
        };
        return new DirectPtxKernelBlueprint(
            Operation: utility.ToString().ToLowerInvariant(),
            Version: 1,
            Architecture: architecture,
            Variant: utility is DirectPtxSparseUtility.DegreeNormalize or
                DirectPtxSparseUtility.SymmetricDegreeNormalize
                ? $"exact-eps{BitConverter.SingleToInt32Bits(epsilon):x8}" : "exact",
            Tensors: tensors,
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["utility"] = utility.ToString(),
                ["epsilon-bits"] = utility is DirectPtxSparseUtility.DegreeNormalize or
                    DirectPtxSparseUtility.SymmetricDegreeNormalize
                    ? BitConverter.SingleToInt32Bits(epsilon).ToString("x8") : "none",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int major, int minor, DirectPtxSparseUtility utility, float epsilon)
    {
        if (major <= 0 || minor < 0) throw new ArgumentOutOfRangeException(nameof(major));
        var ptx = new StringBuilder(4096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{major}{minor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(utility)}(");
        string[] parameters = utility switch
        {
            DirectPtxSparseUtility.FillZero or DirectPtxSparseUtility.FillNegativeInfinity => ["output_ptr"],
            DirectPtxSparseUtility.DegreeNormalize => ["input_ptr", "degrees_ptr", "output_ptr"],
            DirectPtxSparseUtility.SymmetricDegreeNormalize =>
                ["edge_values_ptr", "source_indices_ptr", "target_indices_ptr", "source_degrees_ptr", "target_degrees_ptr", "output_ptr"],
            _ => throw new ArgumentOutOfRangeException(nameof(utility))
        };
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    .param .u64 {parameters[i]}{(i + 1 == parameters.Length ? string.Empty : ",")}");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<12>;");
        for (int i = 0; i < parameters.Length; i++)
            ptx.AppendLine($"    ld.param.u64 %rd{i}, [{parameters[i]}];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        switch (utility)
        {
            case DirectPtxSparseUtility.FillZero:
            case DirectPtxSparseUtility.FillNegativeInfinity:
                ptx.AppendLine("    shl.b32 %r3, %r2, 2;");
                ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
                ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
                string value = utility == DirectPtxSparseUtility.FillZero ? "0f00000000" : "0fFF7FFFFF";
                ptx.AppendLine($"    mov.f32 %f0, {value};");
                ptx.AppendLine("    st.global.v4.f32 [%rd7], {%f0, %f0, %f0, %f0};");
                break;
            case DirectPtxSparseUtility.DegreeNormalize:
                ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
                ptx.AppendLine("    mul.wide.u32 %rd6, %r3, 4;");
                ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
                ptx.AppendLine("    ld.global.f32 %f0, [%rd7];");
                ptx.AppendLine($"    add.rn.f32 %f0, %f0, 0f{BitConverter.SingleToInt32Bits(epsilon):X8};");
                ptx.AppendLine("    rsqrt.approx.f32 %f1, %f0;");
                ptx.AppendLine("    shl.b32 %r4, %r2, 2;");
                ptx.AppendLine("    mul.wide.u32 %rd8, %r4, 4;");
                ptx.AppendLine("    add.u64 %rd9, %rd0, %rd8;");
                ptx.AppendLine("    ld.global.v4.f32 {%f2, %f3, %f4, %f5}, [%rd9];");
                for (int lane = 2; lane < 6; lane++) ptx.AppendLine($"    mul.rn.f32 %f{lane}, %f{lane}, %f1;");
                ptx.AppendLine("    add.u64 %rd10, %rd2, %rd8;");
                ptx.AppendLine("    st.global.v4.f32 [%rd10], {%f2, %f3, %f4, %f5};");
                break;
            case DirectPtxSparseUtility.SymmetricDegreeNormalize:
                ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
                ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
                ptx.AppendLine("    add.u64 %rd8, %rd2, %rd6;");
                ptx.AppendLine("    ld.global.u32 %r3, [%rd7];");
                ptx.AppendLine("    ld.global.u32 %r4, [%rd8];");
                ptx.AppendLine("    mul.wide.u32 %rd9, %r3, 4;");
                ptx.AppendLine("    mul.wide.u32 %rd10, %r4, 4;");
                ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");
                ptx.AppendLine("    add.u64 %rd12, %rd4, %rd10;");
                ptx.AppendLine("    ld.global.f32 %f0, [%rd11];");
                ptx.AppendLine("    ld.global.f32 %f1, [%rd12];");
                string eps = $"0f{BitConverter.SingleToInt32Bits(epsilon):X8}";
                ptx.AppendLine($"    add.rn.f32 %f0, %f0, {eps};");
                ptx.AppendLine($"    add.rn.f32 %f1, %f1, {eps};");
                ptx.AppendLine("    rsqrt.approx.f32 %f2, %f0;");
                ptx.AppendLine("    rsqrt.approx.f32 %f3, %f1;");
                ptx.AppendLine("    add.u64 %rd13, %rd0, %rd6;");
                ptx.AppendLine("    ld.global.f32 %f4, [%rd13];");
                ptx.AppendLine("    mul.rn.f32 %f4, %f4, %f2;");
                ptx.AppendLine("    mul.rn.f32 %f4, %f4, %f3;");
                ptx.AppendLine("    add.u64 %rd14, %rd5, %rd6;");
                ptx.AppendLine("    st.global.f32 [%rd14], %f4;");
                break;
        }
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string GetEntryPoint(DirectPtxSparseUtility utility) => utility switch
    {
        DirectPtxSparseUtility.FillZero => ZeroEntryPoint,
        DirectPtxSparseUtility.FillNegativeInfinity => NegativeInfinityEntryPoint,
        DirectPtxSparseUtility.DegreeNormalize => DegreeEntryPoint,
        DirectPtxSparseUtility.SymmetricDegreeNormalize => SymmetricEntryPoint,
        _ => throw new ArgumentOutOfRangeException(nameof(utility))
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
        DirectPtxTensorView degrees)
    {
        if (Overlaps(output, input) || Overlaps(output, degrees))
            throw new ArgumentException("Degree-normalized output must be disjoint from inputs.", nameof(output));
    }

    public void Dispose() => _module.Dispose();
}
#endif
