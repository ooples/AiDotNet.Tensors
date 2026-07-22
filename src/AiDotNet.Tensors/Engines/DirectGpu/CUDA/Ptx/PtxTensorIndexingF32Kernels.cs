#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>Exact row gather using one coalesced float4 transfer per thread.</summary>
internal sealed class PtxTensorGatherRowsF32Kernel : IDisposable
{
    internal const int SourceRows = 1024;
    internal const int Indices = 16_384;
    internal const int Features = 64;
    internal const int VectorsPerRow = Features / 4;
    internal const int OutputVectors = Indices * VectorsPerRow;
    internal const int BlockThreads = 256;
    internal const string EntryPoint = "aidotnet_tensor_gather_rows_f32_src1024_idx16384_f64";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxTensorGatherRowsF32Kernel(DirectPtxRuntime runtime)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Tensor gather has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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

    internal static bool SupportsShape(int indices, int features) =>
        indices == Indices && features == Features;

    internal unsafe void Launch(
        DirectPtxTensorView source,
        DirectPtxTensorView indices,
        DirectPtxTensorView output)
    {
        Require(source, Blueprint.Tensors[0], nameof(source));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(output, Blueprint.Tensors[2], nameof(output));
        if (Overlaps(output, source) || Overlaps(output, indices))
            throw new ArgumentException("Tensor gather output must be disjoint from source and indices.", nameof(output));
        IntPtr p0 = source.Pointer;
        IntPtr p1 = indices.Pointer;
        IntPtr p2 = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0; arguments[1] = &p1; arguments[2] = &p2;
        _module.Launch(_function, OutputVectors / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture)
    {
        var source = new DirectPtxExtent(SourceRows, Features);
        var indices = new DirectPtxExtent(Indices);
        var output = new DirectPtxExtent(Indices, Features);
        return new DirectPtxKernelBlueprint(
            Operation: "tensor-gather-rows-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "source1024-indices16384-features64-vec4",
            Tensors:
            [
                Exact("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    source, DirectPtxTensorAccess.Read),
                Exact("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.RowGatherIndices,
                    indices, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    output, DirectPtxTensorAccess.Write)
            ],
            ResourceBudget: new DirectPtxResourceBudget(24, 0, 0, 6),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["transfer"] = "one-coalesced-float4-per-thread",
                ["invalid-index"] = "zero-row",
                ["output-write"] = "single-overwrite",
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
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    shr.u32 %r3, %r2, 4;");
        ptx.AppendLine("    and.b32 %r4, %r2, 15;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.s32 %r5, [%rd4];");
        ptx.AppendLine("    setp.lt.s32 %p0, %r5, 0;");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r5, {SourceRows};");
        ptx.AppendLine("    or.pred %p2, %p0, %p1;");
        ptx.AppendLine("    @%p2 mov.f32 %f0, 0f00000000;");
        ptx.AppendLine("    @%p2 mov.f32 %f1, 0f00000000;");
        ptx.AppendLine("    @%p2 mov.f32 %f2, 0f00000000;");
        ptx.AppendLine("    @%p2 mov.f32 %f3, 0f00000000;");
        ptx.AppendLine("    @%p2 bra GATHER_STORE;");
        ptx.AppendLine("    shl.b32 %r6, %r5, 4;");
        ptx.AppendLine("    add.u32 %r6, %r6, %r4;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 16;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    ld.global.v4.f32 {%f0,%f1,%f2,%f3}, [%rd6];");
        ptx.AppendLine("GATHER_STORE:");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r2, 16;");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.v4.f32 [%rd8], {%f0,%f1,%f2,%f3};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxTensorContract Exact(
        string name, DirectPtxPhysicalType type, DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent, DirectPtxTensorAccess access) =>
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

internal enum DirectPtxTensorScatterReduceMode
{
    Sum,
    Product,
    Maximum,
    Minimum
}

/// <summary>Exact elementwise scatter-reduce with a baked atomic-CAS epilogue.</summary>
internal sealed class PtxTensorScatterReduceF32Kernel : IDisposable
{
    internal const int Outer = 32;
    internal const int SourceDimension = 512;
    internal const int DestinationDimension = 1024;
    internal const int Inner = 64;
    internal const int SourceElements = Outer * SourceDimension * Inner;
    internal const int OutputElements = Outer * DestinationDimension * Inner;
    internal const int BlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal DirectPtxTensorScatterReduceMode Mode { get; }
    internal string EntryPoint { get; }
    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxTensorScatterReduceF32Kernel(
        DirectPtxRuntime runtime,
        DirectPtxTensorScatterReduceMode mode)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Tensor scatter-reduce has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
        Mode = mode;
        EntryPoint = GetEntryPoint(mode);
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, mode);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, mode);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo functionInfo);
        FunctionInfo = functionInfo;
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, functionInfo, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, functionInfo,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal static bool SupportsShape(int outer, int sourceDimension, int destinationDimension, int inner) =>
        outer == Outer && sourceDimension == SourceDimension &&
        destinationDimension == DestinationDimension && inner == Inner;

    internal unsafe void Launch(
        DirectPtxTensorView output,
        DirectPtxTensorView source,
        DirectPtxTensorView indices)
    {
        Require(output, Blueprint.Tensors[0], nameof(output));
        Require(source, Blueprint.Tensors[1], nameof(source));
        Require(indices, Blueprint.Tensors[2], nameof(indices));
        if (Overlaps(output, source) || Overlaps(output, indices) || Overlaps(source, indices))
            throw new ArgumentException("Tensor scatter-reduce buffers must be disjoint.");
        IntPtr p0 = output.Pointer;
        IntPtr p1 = source.Pointer;
        IntPtr p2 = indices.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &p0; arguments[1] = &p1; arguments[2] = &p2;
        _module.Launch(_function, SourceElements / BlockThreads, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        DirectPtxTensorScatterReduceMode mode)
    {
        var source = new DirectPtxExtent(Outer, SourceDimension, Inner);
        var output = new DirectPtxExtent(Outer, DestinationDimension, Inner);
        return new DirectPtxKernelBlueprint(
            Operation: $"tensor-scatter-reduce-{mode.ToString().ToLowerInvariant()}-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "outer32-source512-destination1024-inner64",
            Tensors:
            [
                Exact("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.OuterDimensionInner,
                    output, DirectPtxTensorAccess.ReadWrite),
                Exact("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.OuterDimensionInner,
                    source, DirectPtxTensorAccess.Read),
                Exact("indices", DirectPtxPhysicalType.Int32,
                    DirectPtxPhysicalLayout.ElementScatterIndices, source, DirectPtxTensorAccess.Read)
            ],
            ResourceBudget: new DirectPtxResourceBudget(32, 0, 0, 5),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["reduction"] = mode.ToString().ToLowerInvariant(),
                ["reduction-order"] = "unordered-global-cas",
                ["output-seed"] = "preserved-include-self",
                ["invalid-index"] = "ignored",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        DirectPtxTensorScatterReduceMode mode)
    {
        if (ccMajor <= 0 || ccMinor < 0) throw new ArgumentOutOfRangeException(nameof(ccMajor));
        _ = GetEntryPoint(mode);
        var ptx = new StringBuilder(3072);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {GetEntryPoint(mode)}(");
        ptx.AppendLine("    .param .u64 output_ptr,");
        ptx.AppendLine("    .param .u64 source_ptr,");
        ptx.AppendLine("    .param .u64 indices_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<5>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [output_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [indices_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {BlockThreads}, %r0;");
        ptx.AppendLine("    and.b32 %r3, %r2, 63;");
        ptx.AppendLine("    shr.u32 %r4, %r2, 6;");
        ptx.AppendLine("    shr.u32 %r5, %r4, 9;");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd2, %rd3;");
        ptx.AppendLine("    ld.global.s32 %r6, [%rd4];");
        ptx.AppendLine("    setp.lt.s32 %p0, %r6, 0;");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r6, {DestinationDimension};");
        ptx.AppendLine("    or.pred %p2, %p0, %p1;");
        ptx.AppendLine("    @%p2 bra SCATTER_REDUCE_DONE;");
        ptx.AppendLine("    shl.b32 %r7, %r5, 16;");
        ptx.AppendLine("    shl.b32 %r8, %r6, 6;");
        ptx.AppendLine("    add.u32 %r7, %r7, %r8;");
        ptx.AppendLine("    add.u32 %r7, %r7, %r3;");
        ptx.AppendLine("    mul.wide.u32 %rd5, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.f32 %f0, [%rd7];");
        ptx.AppendLine("    ld.global.b32 %r9, [%rd6];");
        ptx.AppendLine("SCATTER_REDUCE_CAS:");
        ptx.AppendLine("    mov.b32 %r10, %r9;");
        ptx.AppendLine("    mov.b32 %f1, %r10;");
        ptx.AppendLine(mode switch
        {
            DirectPtxTensorScatterReduceMode.Sum => "    add.rn.f32 %f2, %f1, %f0;",
            DirectPtxTensorScatterReduceMode.Product => "    mul.rn.f32 %f2, %f1, %f0;",
            DirectPtxTensorScatterReduceMode.Maximum => "    max.f32 %f2, %f1, %f0;",
            DirectPtxTensorScatterReduceMode.Minimum => "    min.f32 %f2, %f1, %f0;",
            _ => throw new ArgumentOutOfRangeException(nameof(mode))
        });
        ptx.AppendLine("    mov.b32 %r11, %f2;");
        ptx.AppendLine("    atom.global.cas.b32 %r9, [%rd6], %r10, %r11;");
        ptx.AppendLine("    setp.ne.u32 %p3, %r9, %r10;");
        ptx.AppendLine("    @%p3 bra SCATTER_REDUCE_CAS;");
        ptx.AppendLine("SCATTER_REDUCE_DONE:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static string GetEntryPoint(DirectPtxTensorScatterReduceMode mode) => mode switch
    {
        DirectPtxTensorScatterReduceMode.Sum => "aidotnet_tensor_scatter_reduce_sum_f32_o32_s512_d1024_i64",
        DirectPtxTensorScatterReduceMode.Product => "aidotnet_tensor_scatter_reduce_product_f32_o32_s512_d1024_i64",
        DirectPtxTensorScatterReduceMode.Maximum => "aidotnet_tensor_scatter_reduce_maximum_f32_o32_s512_d1024_i64",
        DirectPtxTensorScatterReduceMode.Minimum => "aidotnet_tensor_scatter_reduce_minimum_f32_o32_s512_d1024_i64",
        _ => throw new ArgumentOutOfRangeException(nameof(mode))
    };

    private static DirectPtxTensorContract Exact(
        string name, DirectPtxPhysicalType type, DirectPtxPhysicalLayout layout,
        DirectPtxExtent extent, DirectPtxTensorAccess access) =>
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
