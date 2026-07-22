#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact deterministic row scatter-max. Each output cell scans source rows in
/// ascending order, so strict greater-than preserves first-on-ties behavior.
/// Empty groups receive negative infinity and an argmax of -1.
/// </summary>
internal sealed class PtxScatterMaxRowsF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_scatter_max_argmax_rows_f32_r16384_d1024_f64";
    internal const int SourceRows = 16384;
    internal const int DestinationRows = 1024;
    internal const int Features = 64;
    internal const int BlockThreads = 256;
    internal const int OutputElements = DestinationRows * Features;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxScatterMaxRowsF32Kernel(DirectPtxRuntime runtime)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Row scatter-max has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int sourceRows, int features, int destinationRows) =>
        sourceRows == SourceRows && features == Features && destinationRows == DestinationRows;

    internal unsafe void Launch(
        DirectPtxTensorView source,
        DirectPtxTensorView indices,
        DirectPtxTensorView output,
        DirectPtxTensorView argmax)
    {
        Require(source, Blueprint.Tensors[0], nameof(source));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(output, Blueprint.Tensors[2], nameof(output));
        Require(argmax, Blueprint.Tensors[3], nameof(argmax));
        if (Overlaps(output, source) || Overlaps(output, indices) || Overlaps(output, argmax) ||
            Overlaps(argmax, source) || Overlaps(argmax, indices))
            throw new ArgumentException("Scatter-max outputs must be mutually disjoint and disjoint from inputs.", nameof(output));
        IntPtr sourcePointer = source.Pointer;
        IntPtr indicesPointer = indices.Pointer;
        IntPtr outputPointer = output.Pointer;
        IntPtr argmaxPointer = argmax.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &sourcePointer;
        arguments[1] = &indicesPointer;
        arguments[2] = &outputPointer;
        arguments[3] = &argmaxPointer;
        _module.Launch(_function, OutputElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(DirectPtxArchitectureFamily architecture)
    {
        var source = new DirectPtxExtent(SourceRows, Features);
        var indices = new DirectPtxExtent(SourceRows);
        var output = new DirectPtxExtent(DestinationRows, Features);
        return new DirectPtxKernelBlueprint(
            Operation: "scatter-max-argmax-rows-deterministic-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "src16384-dst1024-f64-first-tie",
            Tensors:
            [
                new("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.EdgeMajor2D,
                    source, source, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.CooRowIndices,
                    indices, indices, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("argmax-float-encoded", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, output, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["equation"] = "output[g,f]=max_r(indices[r]=g,source[r,f])",
                ["tie-break"] = "lowest-source-row",
                ["empty-output"] = "negative-infinity",
                ["empty-argmax"] = "float-encoded-minus-one",
                ["reduction-order"] = "ascending-source-row",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        var ptx = new StringBuilder(3072);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 source_ptr,");
        ptx.AppendLine("    .param .u64 indices_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 argmax_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<18>;");
        ptx.AppendLine("    .reg .f32 %f<5>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [argmax_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 6;");
        ptx.AppendLine($"    and.b32 %r4, %r2, {Features - 1};");
        ptx.AppendLine("    mov.u32 %r5, 0;");
        ptx.AppendLine("    mov.s32 %r6, -1;");
        ptx.AppendLine("    mov.f32 %f0, 0fFF800000;");
        ptx.AppendLine("MAX_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {SourceRows};");
        ptx.AppendLine("    @%p0 bra MAX_DONE;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.u32 %r7, [%rd5];");
        ptx.AppendLine("    setp.ne.u32 %p1, %r7, %r3;");
        ptx.AppendLine("    @%p1 bra MAX_NEXT;");
        ptx.AppendLine("    shl.b32 %r8, %r5, 6;");
        ptx.AppendLine("    add.u32 %r8, %r8, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r8, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.f32 %f1, [%rd7];");
        ptx.AppendLine("    setp.gt.f32 %p2, %f1, %f0;");
        ptx.AppendLine("    selp.f32 %f0, %f1, %f0, %p2;");
        ptx.AppendLine("    selp.u32 %r6, %r5, %r6, %p2;");
        ptx.AppendLine("MAX_NEXT:");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra MAX_LOOP;");
        ptx.AppendLine("MAX_DONE:");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    add.u64 %rd10, %rd3, %rd8;");
        ptx.AppendLine("    cvt.rn.f32.s32 %f2, %r6;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    st.global.f32 [%rd10], %f2;");
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
