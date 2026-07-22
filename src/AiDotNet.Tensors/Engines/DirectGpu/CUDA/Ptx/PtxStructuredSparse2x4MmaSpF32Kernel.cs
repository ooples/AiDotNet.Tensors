#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact SM86 2:4 sparse Tensor Core matmul. Each warp owns one 16x8
/// output tile, converts the repository's packed FP32 values and dense B
/// values to FP16 fragments in registers, builds native metadata in a
/// register, and accumulates with eight genuine m16n8k32 mma.sp operations.
/// </summary>
internal sealed class PtxStructuredSparse2x4MmaSpF32Kernel : IDisposable
{
    internal const string EntryPoint =
        "aidotnet_sparse_2x4_mma_sp_f32_m256_n64_k256";
    internal const int Rows = 256;
    internal const int Inner = 256;
    internal const int Columns = 64;
    internal const int GroupsPerRow = Inner / 4;
    internal const int PackedValues = Rows * Inner / 2;
    internal const int MetadataElements = Rows * GroupsPerRow;
    internal const int OutputElements = Rows * Columns;
    internal const int BlockThreads = 128;
    internal const int Warps = (Rows / 16) * (Columns / 8);
    internal const int GridBlocks = Warps / (BlockThreads / 32);

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal string Ptx { get; }
    internal DirectPtxFunctionInfo FunctionInfo { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxStructuredSparse2x4MmaSpF32Kernel(DirectPtxRuntime runtime)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (!DirectPtxArchitecture.HasValidatedSparseGraph(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new NotSupportedException(
                $"Structured sparse mma.sp has no SM {runtime.ComputeCapabilityMajor}.{runtime.ComputeCapabilityMinor} experimental specialization.");
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

    internal static bool SupportsShape(int rows, int columns, int inner) =>
        rows == Rows && columns == Columns && inner == Inner;

    internal unsafe void Launch(
        DirectPtxTensorView values,
        DirectPtxTensorView metadata,
        DirectPtxTensorView denseB,
        DirectPtxTensorView output)
    {
        Require(values, Blueprint.Tensors[0], nameof(values));
        Require(metadata, Blueprint.Tensors[1], nameof(metadata));
        Require(denseB, Blueprint.Tensors[2], nameof(denseB));
        Require(output, Blueprint.Tensors[3], nameof(output));
        if (Overlaps(output, values) || Overlaps(output, metadata) ||
            Overlaps(output, denseB) || Overlaps(values, metadata) ||
            Overlaps(values, denseB) || Overlaps(metadata, denseB))
            throw new ArgumentException(
                "Structured sparse mma.sp tensors must occupy disjoint exact allocations.");
        IntPtr p0 = values.Pointer;
        IntPtr p1 = metadata.Pointer;
        IntPtr p2 = denseB.Pointer;
        IntPtr p3 = output.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &p0;
        arguments[1] = &p1;
        arguments[2] = &p2;
        arguments[3] = &p3;
        _module.Launch(_function, GridBlocks, 1, 1,
            BlockThreads, 1, 1, 0, arguments);
    }

    internal static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture)
    {
        var values = new DirectPtxExtent(PackedValues);
        var metadata = new DirectPtxExtent(MetadataElements);
        var denseB = new DirectPtxExtent(Inner, Columns);
        var output = new DirectPtxExtent(Rows, Columns);
        return new DirectPtxKernelBlueprint(
            Operation: "structured-sparse-2x4-mma-sp-f32",
            Version: 1,
            Architecture: architecture,
            Variant: "m256-n64-k256-warp16x8x32-fp16-input-fp32-acc",
            Tensors:
            [
                Exact("packed-values", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Values,
                    values, DirectPtxTensorAccess.Read),
                Exact("metadata", DirectPtxPhysicalType.UInt8,
                    DirectPtxPhysicalLayout.StructuredSparse2x4Metadata,
                    metadata, DirectPtxTensorAccess.Read),
                Exact("dense-b", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.RowMajor2D,
                    denseB, DirectPtxTensorAccess.Read),
                Exact("output", DirectPtxPhysicalType.Float32,
                    DirectPtxPhysicalLayout.RowMajor2D,
                    output, DirectPtxTensorAccess.Write)
            ],
            ResourceBudget: new DirectPtxResourceBudget(64, 0, 0, 2),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["instruction"] =
                    "mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32",
                ["tensor-core-operations-per-warp"] = "8",
                ["fragment-conversion"] = "fp32-to-fp16-registers",
                ["accumulator"] = "fp32-registers",
                ["metadata"] = "eight-repository-nibbles-packed-to-native-b32-per-k-tile",
                ["sparsity-selector"] = "0",
                ["output-write"] = "four-coalesced-scalar-overwrites-per-lane",
                ["workspace-bytes"] = "0",
                ["intermediate-global-bytes"] = "0"
            });
    }

    internal static string EmitPtx(int ccMajor, int ccMinor)
    {
        if (ccMajor < 8 || ccMinor < 0)
            throw new ArgumentOutOfRangeException(nameof(ccMajor), "mma.sp requires SM80 or newer.");
        var ptx = new StringBuilder(32768);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 values_ptr,");
        ptx.AppendLine("    .param .u64 metadata_ptr,");
        ptx.AppendLine("    .param .u64 dense_b_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b16 %h<16>;");
        ptx.AppendLine("    .reg .b32 %a<4>;");
        ptx.AppendLine("    .reg .b32 %b<4>;");
        ptx.AppendLine("    .reg .f32 %c<4>;");
        ptx.AppendLine("    .reg .f32 %d<4>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    .reg .b32 %r<48>;");
        ptx.AppendLine("    .reg .b64 %rd<24>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [values_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [metadata_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [dense_b_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    shr.u32 %r1, %r0, 5;");
        ptx.AppendLine("    mov.u32 %r2, %ctaid.x;");
        ptx.AppendLine("    mad.lo.u32 %r3, %r2, 4, %r1;");
        ptx.AppendLine("    shr.u32 %r4, %r3, 3;");
        ptx.AppendLine("    and.b32 %r5, %r3, 7;");
        ptx.AppendLine("    mov.u32 %r6, %laneid;");
        ptx.AppendLine("    shr.u32 %r7, %r6, 2;");
        ptx.AppendLine("    and.b32 %r8, %r6, 3;");
        ptx.AppendLine("    shl.b32 %r9, %r4, 4;");
        ptx.AppendLine("    add.u32 %r10, %r9, %r7;");
        ptx.AppendLine("    add.u32 %r11, %r10, 8;");
        ptx.AppendLine("    shl.b32 %r12, %r5, 3;");
        ptx.AppendLine("    shl.b32 %r13, %r8, 1;");
        ptx.AppendLine("    add.u32 %r14, %r12, %r13;");
        ptx.AppendLine("    add.u32 %r15, %r14, 1;");
        ptx.AppendLine("    mov.f32 %c0, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c1, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c2, 0f00000000;");
        ptx.AppendLine("    mov.f32 %c3, 0f00000000;");

        for (int kTile = 0; kTile < Inner / 32; kTile++)
            EmitKTile(ptx, kTile);

        EmitStores(ptx);
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static void EmitKTile(StringBuilder ptx, int kTile)
    {
        ptx.AppendLine($"    // K tile {kTile}: [{kTile * 32},{kTile * 32 + 31}]");
        ptx.AppendLine("    shl.b32 %r16, %r10, 7;");
        ptx.AppendLine("    shl.b32 %r17, %r11, 7;");
        ptx.AppendLine("    shl.b32 %r18, %r8, 1;");
        ptx.AppendLine($"    add.u32 %r19, %r18, {kTile * 16};");
        ptx.AppendLine("    add.u32 %r20, %r16, %r19;");
        ptx.AppendLine("    add.u32 %r21, %r20, 8;");
        ptx.AppendLine("    add.u32 %r22, %r17, %r19;");
        ptx.AppendLine("    add.u32 %r23, %r22, 8;");
        EmitPackedPairLoad(ptx, 20, 0, 0);
        EmitPackedPairLoad(ptx, 21, 1, 2);
        EmitPackedPairLoad(ptx, 22, 2, 4);
        EmitPackedPairLoad(ptx, 23, 3, 6);

        for (int fragment = 0; fragment < 4; fragment++)
        {
            int rowAdd = kTile * 32 + fragment * 8;
            ptx.AppendLine($"    shl.b32 %r24, %r8, 1;");
            ptx.AppendLine($"    add.u32 %r24, %r24, {rowAdd};");
            ptx.AppendLine("    shl.b32 %r25, %r24, 6;");
            ptx.AppendLine("    add.u32 %r25, %r25, %r14;");
            ptx.AppendLine("    add.u32 %r26, %r25, 64;");
            ptx.AppendLine("    mul.wide.u32 %rd8, %r25, 4;");
            ptx.AppendLine("    mul.wide.u32 %rd9, %r26, 4;");
            ptx.AppendLine("    add.u64 %rd10, %rd2, %rd8;");
            ptx.AppendLine("    add.u64 %rd11, %rd2, %rd9;");
            ptx.AppendLine("    ld.global.f32 %f8, [%rd10];");
            ptx.AppendLine("    ld.global.f32 %f9, [%rd11];");
            ptx.AppendLine("    cvt.rn.f16.f32 %h8, %f8;");
            ptx.AppendLine("    cvt.rn.f16.f32 %h9, %f9;");
            ptx.AppendLine($"    mov.b32 %b{fragment}, {{%h8,%h9}};");
        }

        ptx.AppendLine("    and.b32 %r27, %r8, 1;");
        ptx.AppendLine("    shl.b32 %r27, %r27, 3;");
        ptx.AppendLine("    add.u32 %r28, %r10, %r27;");
        ptx.AppendLine("    shl.b32 %r29, %r28, 6;");
        ptx.AppendLine($"    add.u32 %r29, %r29, {kTile * 8};");
        ptx.AppendLine("    cvt.u64.u32 %rd12, %r29;");
        ptx.AppendLine("    add.u64 %rd13, %rd1, %rd12;");
        ptx.AppendLine("    mov.u32 %r30, 0;");
        for (int group = 0; group < 8; group++)
        {
            ptx.AppendLine($"    ld.global.u8 %r31, [%rd13+{group}];");
            ptx.AppendLine("    and.b32 %r31, %r31, 15;");
            if (group != 0)
                ptx.AppendLine($"    shl.b32 %r31, %r31, {group * 4};");
            ptx.AppendLine("    or.b32 %r30, %r30, %r31;");
        }

        ptx.AppendLine("    mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32");
        ptx.AppendLine("        {%d0,%d1,%d2,%d3},");
        ptx.AppendLine("        {%a0,%a1,%a2,%a3},");
        ptx.AppendLine("        {%b0,%b1,%b2,%b3},");
        ptx.AppendLine("        {%c0,%c1,%c2,%c3}, %r30, 0x0;");
        ptx.AppendLine("    mov.f32 %c0, %d0;");
        ptx.AppendLine("    mov.f32 %c1, %d1;");
        ptx.AppendLine("    mov.f32 %c2, %d2;");
        ptx.AppendLine("    mov.f32 %c3, %d3;");
    }

    private static void EmitPackedPairLoad(
        StringBuilder ptx,
        int elementRegister,
        int packedRegister,
        int halfBase)
    {
        ptx.AppendLine($"    mul.wide.u32 %rd4, %r{elementRegister}, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine($"    ld.global.v2.f32 {{%f0,%f1}}, [%rd5];");
        ptx.AppendLine($"    cvt.rn.f16.f32 %h{halfBase}, %f0;");
        ptx.AppendLine($"    cvt.rn.f16.f32 %h{halfBase + 1}, %f1;");
        ptx.AppendLine($"    mov.b32 %a{packedRegister}, {{%h{halfBase},%h{halfBase + 1}}};");
    }

    private static void EmitStores(StringBuilder ptx)
    {
        ptx.AppendLine("    shl.b32 %r32, %r10, 6;");
        ptx.AppendLine("    add.u32 %r32, %r32, %r14;");
        ptx.AppendLine("    add.u32 %r33, %r32, 1;");
        ptx.AppendLine("    shl.b32 %r34, %r11, 6;");
        ptx.AppendLine("    add.u32 %r34, %r34, %r14;");
        ptx.AppendLine("    add.u32 %r35, %r34, 1;");
        for (int i = 0; i < 4; i++)
        {
            int register = 32 + i;
            ptx.AppendLine($"    mul.wide.u32 %rd{14 + i}, %r{register}, 4;");
            ptx.AppendLine($"    add.u64 %rd{18 + i}, %rd3, %rd{14 + i};");
            ptx.AppendLine($"    st.global.f32 [%rd{18 + i}], %c{i};");
        }
    }

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
            throw new ArgumentException(
                $"{parameter} does not satisfy exact physical ABI '{contract.Name}'.", parameter);
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
