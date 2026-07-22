#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal enum DirectPtxCsrSegmentReduction
{
    Max,
    Min,
    StdDev
}

/// <summary>
/// Exact CSR gathered-row reduction for the legacy FP32-encoded CSR index ABI.
/// One thread owns four output features and writes them once.
/// </summary>
internal sealed class PtxCsrSegmentReduceVec4F32Kernel : IDisposable
{
    internal const string MaxEntryPoint = "aidotnet_csr_segmented_max_vec4_f32_m1024_k1024_n64_nnz16384";
    internal const string MinEntryPoint = "aidotnet_csr_segmented_min_vec4_f32_m1024_k1024_n64_nnz16384";
    internal const string StdDevEntryPoint = "aidotnet_csr_segmented_stddev_vec4_f32_m1024_k1024_n64_nnz16384";
    internal const int Rows = 1024;
    internal const int InnerRows = 1024;
    internal const int Features = 64;
    internal const int NonZeros = 16384;
    internal const int FeaturesPerThread = 4;
    internal const int FeatureGroups = Features / FeaturesPerThread;
    internal const int BlockThreads = 256;
    internal const int GridBlocks = Rows * FeatureGroups / BlockThreads;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxCsrSegmentReduction Reduction { get; }
    internal float Epsilon { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxCsrSegmentReduceVec4F32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxCsrSegmentReduction reduction,
        float epsilon)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"CSR segmented reduction has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        if (reduction == DirectPtxCsrSegmentReduction.StdDev &&
            (!(epsilon >= 0f) || float.IsInfinity(epsilon)))
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        Reduction = reduction;
        Epsilon = epsilon;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, reduction, epsilon);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, reduction, epsilon);
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

    internal static bool SupportsShape(int rows, int innerRows, int features, int nonZeros) =>
        rows == Rows && innerRows == InnerRows && features == Features && nonZeros == NonZeros;

    internal unsafe void Launch(
        DirectPtxTensorView columnIndices,
        DirectPtxTensorView rowPointers,
        DirectPtxTensorView input,
        DirectPtxTensorView output)
    {
        Require(columnIndices, Blueprint.Tensors[0], nameof(columnIndices));
        Require(rowPointers, Blueprint.Tensors[1], nameof(rowPointers));
        Require(input, Blueprint.Tensors[2], nameof(input));
        Require(output, Blueprint.Tensors[3], nameof(output));
        if (Overlaps(output, columnIndices) || Overlaps(output, rowPointers) || Overlaps(output, input))
            throw new ArgumentException("CSR segmented output must be disjoint from every input.", nameof(output));
        IntPtr columnsPointer = columnIndices.Pointer;
        IntPtr rowsPointer = rowPointers.Pointer;
        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &columnsPointer;
        arguments[1] = &rowsPointer;
        arguments[2] = &inputPointer;
        arguments[3] = &outputPointer;
        _module.Launch(_function, GridBlocks, 1, 1, BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxCsrSegmentReduction reduction,
        float epsilon)
    {
        var nonZeros = new DirectPtxExtent(NonZeros);
        var rowPointers = new DirectPtxExtent(Rows + 1);
        var input = new DirectPtxExtent(InnerRows, Features);
        var output = new DirectPtxExtent(Rows, Features);
        return new DirectPtxKernelBlueprint(
            Operation: $"csr-segmented-{reduction.ToString().ToLowerInvariant()}-vec4-f32",
            Version: 1,
            Architecture: architecture,
            Variant: reduction == DirectPtxCsrSegmentReduction.StdDev
                ? $"m1024-k1024-n64-nnz16384-eps{BitConverter.SingleToInt32Bits(epsilon):x8}"
                : "m1024-k1024-n64-nnz16384",
            Tensors:
            [
                new("column-indices", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.CsrFloatColumnIndices,
                    nonZeros, nonZeros, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("row-pointers", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.CsrFloatRowPointers,
                    rowPointers, rowPointers, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    input, input, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(56, 0, 0, 4),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction"] = reduction.ToString().ToLowerInvariant(),
                ["csr-index-encoding"] = "exact-integral-fp32",
                ["reduction-order"] = "stored-row-order",
                ["empty-row"] = "zero",
                ["epsilon-bits"] = reduction == DirectPtxCsrSegmentReduction.StdDev
                    ? BitConverter.SingleToInt32Bits(epsilon).ToString("x8") : "none",
                ["stddev-algorithm"] = reduction == DirectPtxCsrSegmentReduction.StdDev
                    ? "two-pass-population-variance" : "not-applicable",
                ["output-write"] = "single-overwrite",
                ["workspace-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxCsrSegmentReduction reduction,
        float epsilon)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        bool stddev = reduction == DirectPtxCsrSegmentReduction.StdDev;
        bool max = reduction == DirectPtxCsrSegmentReduction.Max;
        var ptx = new StringBuilder(6144);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(reduction)}(");
        ptx.AppendLine("    .param .u64 column_indices_ptr,");
        ptx.AppendLine("    .param .u64 row_pointers_ptr,");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<20>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [column_indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [row_pointers_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {FeatureGroups - 1};");
        ptx.AppendLine("    shl.b32 %r5, %r4, 2;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.f32 %f16, [%rd5];");
        ptx.AppendLine("    ld.global.f32 %f17, [%rd5+4];");
        ptx.AppendLine("    cvt.rzi.u32.f32 %r6, %f16;");
        ptx.AppendLine("    cvt.rzi.u32.f32 %r7, %f17;");
        ptx.AppendLine("    setp.ge.u32 %p2, %r6, %r7;");
        ptx.AppendLine("    @%p2 bra EMPTY_ROW;");
        if (stddev)
        {
            for (int lane = 0; lane < 4; lane++) ptx.AppendLine($"    mov.f32 %f{lane}, 0f00000000;");
            AppendLoop(ptx, "MEAN", "%f0", "%f1", "%f2", "%f3", "add.rn.f32");
            ptx.AppendLine("    sub.u32 %r12, %r7, %r6;");
            ptx.AppendLine("    cvt.rn.f32.u32 %f18, %r12;");
            for (int lane = 0; lane < 4; lane++) ptx.AppendLine($"    div.rn.f32 %f{lane}, %f{lane}, %f18;");
            for (int lane = 8; lane < 12; lane++) ptx.AppendLine($"    mov.f32 %f{lane}, 0f00000000;");
            AppendVarianceLoop(ptx);
            for (int lane = 0; lane < 4; lane++)
            {
                ptx.AppendLine($"    div.rn.f32 %f{lane + 8}, %f{lane + 8}, %f18;");
                ptx.AppendLine($"    add.rn.f32 %f{lane + 8}, %f{lane + 8}, 0f{BitConverter.SingleToInt32Bits(epsilon):X8};");
                ptx.AppendLine($"    sqrt.rn.f32 %f{lane}, %f{lane + 8};");
            }
        }
        else
        {
            string bits = max ? "0xff7fffff" : "0x7f7fffff";
            for (int lane = 0; lane < 4; lane++) ptx.AppendLine($"    mov.b32 %f{lane}, {bits};");
            AppendLoop(ptx, "REDUCE", "%f0", "%f1", "%f2", "%f3", max ? "max.f32" : "min.f32");
        }
        ptx.AppendLine("    bra STORE_RESULT;");
        ptx.AppendLine("EMPTY_ROW:");
        for (int lane = 0; lane < 4; lane++) ptx.AppendLine($"    mov.f32 %f{lane}, 0f00000000;");
        ptx.AppendLine("STORE_RESULT:");
        ptx.AppendLine("    shl.b32 %r13, %r3, 6;");
        ptx.AppendLine("    add.u32 %r13, %r13, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd18, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd19, %rd3, %rd18;");
        ptx.AppendLine("    st.global.v4.f32 [%rd19], {%f0, %f1, %f2, %f3};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void AppendLoop(
        StringBuilder ptx, string label, string a0, string a1, string a2, string a3, string op)
    {
        ptx.AppendLine("    mov.u32 %r8, %r6;");
        ptx.AppendLine($"{label}_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r8, %r7;");
        ptx.AppendLine($"    @%p0 bra {label}_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f16, [%rd7];");
        ptx.AppendLine("    cvt.rzi.u32.f32 %r9, %f16;");
        ptx.AppendLine("    shl.b32 %r10, %r9, 6;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd9];");
        ptx.AppendLine($"    {op} {a0}, {a0}, %f4;");
        ptx.AppendLine($"    {op} {a1}, {a1}, %f5;");
        ptx.AppendLine($"    {op} {a2}, {a2}, %f6;");
        ptx.AppendLine($"    {op} {a3}, {a3}, %f7;");
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine($"    bra {label}_LOOP;");
        ptx.AppendLine($"{label}_DONE:");
    }

    private static void AppendVarianceLoop(StringBuilder ptx)
    {
        ptx.AppendLine("    mov.u32 %r8, %r6;");
        ptx.AppendLine("VAR_LOOP:");
        ptx.AppendLine("    setp.ge.u32 %p0, %r8, %r7;");
        ptx.AppendLine("    @%p0 bra VAR_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd0, %rd10;");
        ptx.AppendLine("    ld.global.f32 %f16, [%rd11];");
        ptx.AppendLine("    cvt.rzi.u32.f32 %r9, %f16;");
        ptx.AppendLine("    shl.b32 %r10, %r9, 6;");
        ptx.AppendLine("    add.u32 %r10, %r10, %r5;");
        ptx.AppendLine("    mul.wide.u32 %rd12, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd2, %rd12;");
        ptx.AppendLine("    ld.global.v4.f32 {%f4, %f5, %f6, %f7}, [%rd13];");
        for (int lane = 0; lane < 4; lane++)
        {
            ptx.AppendLine($"    sub.rn.f32 %f{lane + 12}, %f{lane + 4}, %f{lane};");
            ptx.AppendLine($"    fma.rn.f32 %f{lane + 8}, %f{lane + 12}, %f{lane + 12}, %f{lane + 8};");
        }
        ptx.AppendLine("    add.u32 %r8, %r8, 1;");
        ptx.AppendLine("    bra VAR_LOOP;");
        ptx.AppendLine("VAR_DONE:");
    }

    private static string GetEntryPoint(DirectPtxCsrSegmentReduction reduction) => reduction switch
    {
        DirectPtxCsrSegmentReduction.Max => MaxEntryPoint,
        DirectPtxCsrSegmentReduction.Min => MinEntryPoint,
        DirectPtxCsrSegmentReduction.StdDev => StdDevEntryPoint,
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

    public void Dispose() => _module.Dispose();
}
#endif
