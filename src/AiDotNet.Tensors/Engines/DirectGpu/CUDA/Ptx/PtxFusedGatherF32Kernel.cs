#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 embedding gather (issue #844):
/// <c>output[i, :] = source[indices[i], :]</c>. A warp owns one output row,
/// loads its INT32 index once, and streams the source row to the output row in
/// a single vectorized coalesced copy. There are no shared-memory, local-memory,
/// global-intermediate, temporary-allocation, division, remainder, or scalar
/// shape parameters — only three tensor pointers reach the launch ABI. The
/// source table is an at-least view (its row count is not part of the module
/// identity); indices are trusted in range, exactly as the established
/// embedding_forward kernel requires. The specialization stays disabled by
/// default and fails closed until three clean promotion runs clear the gate.
/// </summary>
internal sealed class PtxFusedGatherF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_gather_f32";
    internal const int DefaultBlockThreads = 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumIndices { get; }
    internal int FeatureSize { get; }
    internal int BlockThreads { get; }
    internal int WarpsPerBlock => BlockThreads / 32;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedGatherF32Kernel(
        DirectPtxRuntime runtime,
        int numIndices,
        int featureSize,
        int blockThreads = DefaultBlockThreads)
    {
        ArgumentNullException.ThrowIfNull(runtime);
        if (runtime.ArchitectureFamily != DirectPtxArchitectureFamily.Ampere)
            throw new PlatformNotSupportedException(
                "The checked-in FP32 gather specialization is validated only on Ampere.");
        Validate(numIndices, featureSize);
        ValidateBlockThreads(numIndices, blockThreads);
        NumIndices = numIndices;
        FeatureSize = featureSize;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numIndices, featureSize, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            numIndices, featureSize, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView indices,
        DirectPtxTensorView source,
        DirectPtxTensorView output)
    {
        Require(indices, Blueprint.Tensors[0], nameof(indices));
        Require(source, Blueprint.Tensors[1], nameof(source));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr indicesPointer = indices.Pointer;
        IntPtr sourcePointer = source.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &indicesPointer;
        arguments[1] = &sourcePointer;
        arguments[2] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)((NumIndices + WarpsPerBlock - 1) / WarpsPerBlock)),
            1,
            1,
            checked((uint)BlockThreads),
            1,
            1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int numIndices,
        int featureSize,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(numIndices, featureSize);
        ValidateBlockThreads(numIndices, blockThreads);
        int warpsPerBlock = blockThreads / 32;
        int valuesPerLane = featureSize / 32;
        int rowBytes = featureSize * sizeof(float);
        var valueRegisterNames = new string[valuesPerLane];
        for (int i = 0; i < valuesPerLane; i++)
            valueRegisterNames[i] = $"%f{i}";
        string valueRegisters = string.Join(", ", valueRegisterNames);
        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape indices={numIndices} feature={featureSize} block={blockThreads} strategy=warp-row op=gather cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 indices_ptr,");
        ptx.AppendLine("    .param .u64 source_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<7>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine($"    .reg .f32 %f<{Math.Max(valuesPerLane, 1)}>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [source_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {warpsPerBlock}, %r2;");
        // Load this warp's index.
        ptx.AppendLine("    mul.wide.u32 %rd3, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");
        ptx.AppendLine("    ld.global.ca.u32 %r5, [%rd4];");
        // Source row base = source + index * rowBytes (signed index).
        ptx.AppendLine($"    mul.wide.s32 %rd5, %r5, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");
        // Lane byte offset within a row.
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r3, {valuesPerLane * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd6, %rd7;");
        // Output row base = output + row * rowBytes.
        ptx.AppendLine($"    mul.wide.u32 %rd9, %r4, {rowBytes};");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");
        ptx.AppendLine("    add.u64 %rd11, %rd10, %rd7;");
        ptx.AppendLine(
            $"    ld.global.ca.v{valuesPerLane}.f32 {{{valueRegisters}}}, [%rd8];");
        ptx.AppendLine(
            $"    st.global.v{valuesPerLane}.f32 [%rd11], {{{valueRegisters}}};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int numIndices,
        int featureSize,
        int blockThreads)
    {
        var indexExtent = new DirectPtxExtent(numIndices);
        var sourceRowExtent = new DirectPtxExtent(1, featureSize);
        var outputExtent = new DirectPtxExtent(numIndices, featureSize);
        return new DirectPtxKernelBlueprint(
            Operation: "gather-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-row-w{blockThreads / 32}-n{numIndices}-f{featureSize}",
            Tensors:
            [
                new("indices", DirectPtxPhysicalType.Int32, DirectPtxPhysicalLayout.Vector,
                    indexExtent, indexExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("source", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    sourceRowExtent, sourceRowExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.AtLeast),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i,:] = source[indices[i],:]",
                ["mode"] = "inference-forward-indexed-copy",
                ["index"] = "int32",
                ["source"] = "fp32-at-least-table",
                ["output"] = "fp32",
                ["global-input-reads"] = "one-index-plus-one-row-per-output-row",
                ["global-output-writes"] = "one-per-element",
                ["lane-vector-transaction"] = featureSize == 64 ? "aligned-fp32x2" : "aligned-fp32x4",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["index-bounds"] = "trusted-in-range",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int numIndices, int featureSize) =>
        (numIndices, featureSize) is
            (256, 128) or (2048, 64) or (2048, 128) or (8192, 128);

    internal static bool IsPromotedShape(int numIndices, int featureSize) => false;

    private static void Validate(int numIndices, int featureSize)
    {
        if (!IsSupportedShape(numIndices, featureSize))
            throw new ArgumentOutOfRangeException(nameof(numIndices),
                "The first gather family supports exact (numIndices,featureSize) buckets " +
                "(256,128), (2048,64), (2048,128), and (8192,128).");
        if (featureSize % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(featureSize));
    }

    private static void ValidateBlockThreads(int numIndices, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            numIndices % (blockThreads / 32) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Gather block threads must be 128, 256, or 512 and evenly tile the index count.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        bool exactExtent = contract.ExtentMode == DirectPtxExtentMode.Exact;
        bool allocationOk = exactExtent
            ? view.AllocationByteLength == contract.RequiredBytes
            : view.AllocationByteLength >= contract.RequiredBytes;
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes || !allocationOk)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
#endif
