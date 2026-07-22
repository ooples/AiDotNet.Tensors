using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact contiguous FP32 index-select for issue #844: gathers
/// <c>numIndices</c> rows of width <c>innerSize</c> out of a table, one output
/// element per thread.
///
/// This is deliberately a SEPARATE kernel from
/// <see cref="PtxFusedGatherF32Kernel"/> rather than a reuse of it. The two ops
/// look interchangeable but their index contracts differ: the established
/// <c>embedding_forward</c> that backs Gather takes <c>const int*</c> indices,
/// while <c>index_select</c> takes <c>const float*</c> and applies a C
/// <c>(int)</c> cast. That cast is a numeric conversion, not a bit
/// reinterpretation - the caller stores 3.0f, not the bit pattern of 3 - and it
/// truncates toward zero, so this kernel uses <c>cvt.rzi.s32.f32</c> to match.
/// Routing index-select through the int-index gather would reinterpret the
/// index buffer and read garbage, which is the same class of bug the
/// embedding_forward source already documents having been bitten by.
///
/// The row width is baked into the module, so the per-thread index split is a
/// shift and mask rather than a division and remainder, and there is no bounds
/// branch: the launch covers exactly <c>numIndices * innerSize</c> elements.
/// Three pointers reach the launch ABI.
///
/// Out-of-range indices are the caller's responsibility, exactly as in the
/// established kernel, which also performs no bounds check. The dispatch guard
/// documents that; it is not silently different here.
///
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedIndexSelectF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_index_select_f32";
    internal const int DefaultBlockThreads = 256;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumIndices { get; }
    internal int InnerSize { get; }
    internal int BlockThreads { get; }
    internal int TableRows { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedIndexSelectF32Kernel(
        DirectPtxRuntime runtime,
        int numIndices,
        int innerSize,
        int tableRows,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedIndexSelect(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in index-select specialization is admitted only on SM86.");
        Validate(numIndices, innerSize, tableRows);
        ValidateBlockThreads(numIndices, innerSize, blockThreads);
        NumIndices = numIndices;
        InnerSize = innerSize;
        TableRows = tableRows;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numIndices, innerSize, tableRows, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            numIndices, innerSize, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView input,
        DirectPtxTensorView indices,
        DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(indices, Blueprint.Tensors[1], nameof(indices));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr indicesPointer = indices.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &inputPointer;
        arguments[1] = &indicesPointer;
        arguments[2] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)(NumIndices * InnerSize / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int numIndices,
        int innerSize,
        int blockThreads = DefaultBlockThreads)
    {
        ValidateShape(numIndices, innerSize);
        ValidateBlockThreads(numIndices, innerSize, blockThreads);
        int innerShift = PtxCompat.Log2((uint)innerSize);

        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape indices={numIndices} inner={innerSize} block={blockThreads} strategy=linear-element op=index-select-f32");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 indices_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [indices_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %ctaid.x;");
        ptx.AppendLine("    mov.u32 %r1, %tid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r0, {blockThreads}, %r1;");
        // innerSize is a baked power of two, so i = idx >> shift and
        // j = idx & (innerSize - 1) replace the reference's div and mod.
        ptx.AppendLine($"    shr.u32 %r3, %r2, {innerShift};");
        ptx.AppendLine($"    and.b32 %r4, %r2, {innerSize - 1};");
        // The index buffer holds FLOAT values; (int) in the reference truncates
        // toward zero, which is exactly cvt.rzi.s32.f32.
        ptx.AppendLine($"    mul.wide.u32 %rd3, %r3, {sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd4, %rd1, %rd3;");
        ptx.AppendLine("    ld.global.ca.f32 %f0, [%rd4];");
        ptx.AppendLine("    cvt.rzi.s32.f32 %r5, %f0;");
        // src = srcIdx * innerSize + j
        ptx.AppendLine($"    mul.lo.s32 %r6, %r5, {innerSize};");
        ptx.AppendLine("    add.s32 %r7, %r6, %r4;");
        ptx.AppendLine($"    mul.wide.s32 %rd5, %r7, {sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd5;");
        ptx.AppendLine("    ld.global.ca.f32 %f1, [%rd6];");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r2, {sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd2, %rd7;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f1;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int numIndices,
        int innerSize,
        int tableRows,
        int blockThreads)
    {
        var tableExtent = new DirectPtxExtent(tableRows, innerSize);
        var indexExtent = new DirectPtxExtent(numIndices);
        var outputExtent = new DirectPtxExtent(numIndices, innerSize);
        return new DirectPtxKernelBlueprint(
            Operation: "index-select-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"linear-element-b{blockThreads}-n{numIndices}-i{innerSize}-t{tableRows}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    tableExtent, tableExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                // Float, not int: this is the whole reason the kernel is separate
                // from the int-index gather.
                new("indices", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    indexExtent, indexExtent, 4, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i, j] = input[(int)indices[i], j]",
                ["mode"] = "inference-forward-linear-index-select",
                ["index-dtype"] = "fp32 holding integral values; (int) truncates toward zero",
                ["index-conversion"] = "cvt.rzi.s32.f32 - numeric truncation, NOT a bit reinterpretation",
                ["divide-and-remainder"] = "none - innerSize is a baked power of two, so shift and mask",
                ["bounds-check"] = "none, matching the established kernel; callers own index validity",
                ["global-input-reads"] = "one index element plus one table element per thread",
                ["global-output-writes"] = "one-fp32-per-thread",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>
    /// innerSize must be a power of two so the index split is a shift and mask;
    /// the total element count must tile the block exactly so there is no tail
    /// branch.
    /// </summary>
    internal static bool IsSupportedShape(int numIndices, int innerSize) =>
        innerSize is 32 or 64 or 128 or 256 or 512 &&
        numIndices is 1024 or 4096 or 16_384 or 65_536;

    internal static bool IsPromotedShape(int numIndices, int innerSize) => false;

    private static void ValidateShape(int numIndices, int innerSize)
    {
        if (!IsSupportedShape(numIndices, innerSize))
            throw new ArgumentOutOfRangeException(nameof(numIndices),
                "The first index-select family supports exact index counts 1024, 4096, 16384, and " +
                "65536 with inner sizes 32, 64, 128, 256, and 512.");
    }

    private static void Validate(int numIndices, int innerSize, int tableRows)
    {
        ValidateShape(numIndices, innerSize);
        if (tableRows <= 0)
            throw new ArgumentOutOfRangeException(nameof(tableRows),
                "The gathered table must have at least one row.");
    }

    private static void ValidateBlockThreads(int numIndices, int innerSize, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) ||
            checked(numIndices * innerSize) % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Index-select block threads must be 128, 256, or 512 and evenly tile the element count.");
    }

    private static void Require(
        DirectPtxTensorView view,
        DirectPtxTensorContract contract,
        string parameter)
    {
        if (view.Pointer == IntPtr.Zero || view.PhysicalType != contract.PhysicalType ||
            view.Layout != contract.Layout || view.LogicalExtent != contract.LogicalExtent ||
            view.PhysicalExtent != contract.PhysicalExtent ||
            view.ByteLength != contract.RequiredBytes ||
            view.AllocationByteLength != contract.RequiredBytes)
            throw new ArgumentException(
                $"{parameter} does not satisfy physical ABI '{contract.Name}'.", parameter);
    }
}
