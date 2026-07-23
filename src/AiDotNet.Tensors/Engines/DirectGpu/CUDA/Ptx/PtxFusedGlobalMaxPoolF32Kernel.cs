using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Exact-shape FP32 global max pooling over a <c>[batch*channels, H*W]</c>
/// input for issue #842, the max-reduction sibling of
/// <see cref="PtxFusedGlobalAvgPoolF32Kernel"/>.
///
/// The established <c>global_maxpool2d</c> kernel assigns ONE thread per
/// (batch, channel) plane and walks the whole HxW window in a serial nested
/// loop, so a plane of 128 elements costs 128 dependent iterations on a single
/// lane. This specialization gives each plane a full warp instead: every lane
/// takes a vectorized slice, folds it in registers, and a butterfly shuffle
/// finishes the reduction, so the same plane costs one vector load plus five
/// shuffle steps.
///
/// Only the value-producing path is ported. The established kernel can also
/// emit arg-max indices under a <c>saveIndices</c> flag, and reducing
/// (value, index) pairs through a warp needs a different shuffle - that path
/// stays on the established kernel and is recorded as a separate manifest cell,
/// so the direct-PTX lane is never silently used when indices are requested.
///
/// NaN behaviour matches the reference. The established kernel seeds
/// <c>-INFINITY</c> and updates only on a strict <c>val &gt; maxVal</c>, so NaN
/// inputs never win; PTX <c>max.f32</c> returns the non-NaN operand, so a plane
/// of all NaNs reduces to <c>-inf</c> in both.
///
/// The specialization stays disabled by default and fails closed until three
/// clean promotion runs clear the release gate.
/// </summary>
internal sealed class PtxFusedGlobalMaxPoolF32Kernel : IDisposable
{
    internal const string EntryPoint = "aidotnet_fused_global_maxpool_f32";
    internal const int DefaultBlockThreads = 128;

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Rows { get; }
    internal int Spatial { get; }
    internal int BlockThreads { get; }
    internal int WarpsPerBlock => BlockThreads / 32;
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxFusedGlobalMaxPoolF32Kernel(
        DirectPtxRuntime runtime,
        int rows,
        int spatial,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedGlobalMaxPool(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in global max-pool specialization is admitted only on SM86.");
        Validate(rows, spatial);
        ValidateBlockThreads(rows, blockThreads);
        Rows = rows;
        Spatial = spatial;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, rows, spatial, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            rows, spatial, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info,
            BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer;
        IntPtr outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)(Rows / WarpsPerBlock)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor,
        int ccMinor,
        int rows,
        int spatial,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(rows, spatial);
        ValidateBlockThreads(rows, blockThreads);
        int warpsPerBlock = blockThreads / 32;
        int valuesPerLane = spatial / 32;
        var valueRegisterNames = new string[valuesPerLane];
        for (int i = 0; i < valuesPerLane; i++)
            valueRegisterNames[i] = $"%f{i}";
        string valueRegisters = string.Join(", ", valueRegisterNames);
        int accReg = valuesPerLane;
        int scratchReg = valuesPerLane + 1;

        var ptx = new StringBuilder(4_096);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine(
            $"// exact-shape rows={rows} spatial={spatial} block={blockThreads} strategy=warp-plane op=global-maxpool cache=ca");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<7>;");
        ptx.AppendLine("    .reg .b64 %rd<9>;");
        ptx.AppendLine($"    .reg .f32 %f<{valuesPerLane + 3}>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine("    shr.u32 %r2, %r0, 5;");
        ptx.AppendLine("    and.b32 %r3, %r0, 31;");
        ptx.AppendLine($"    mad.lo.u32 %r4, %r1, {warpsPerBlock}, %r2;");
        ptx.AppendLine($"    mul.wide.u32 %rd2, %r4, {spatial * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine($"    mul.wide.u32 %rd5, %r3, {valuesPerLane * sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd6, %rd3, %rd5;");
        ptx.AppendLine(
            $"    ld.global.nc.v{valuesPerLane}.f32 {{{valueRegisters}}}, [%rd6];");
        // Seed -inf, matching the reference's maxVal = -INFINITY.
        ptx.AppendLine($"    mov.f32 %f{accReg}, 0fFF800000;");
        for (int i = 0; i < valuesPerLane; i++)
            ptx.AppendLine($"    max.f32 %f{accReg}, %f{accReg}, %f{i};");

        // shfl.sync.bfly.b32 operands are .b32, not .f32: reinterpret through a
        // .b32 scratch pair and back, the same idiom the row-reduce family uses.
        foreach (int delta in new[] { 16, 8, 4, 2, 1 })
        {
            ptx.AppendLine($"    mov.b32 %r5, %f{accReg};");
            ptx.AppendLine(
                $"    shfl.sync.bfly.b32 %r6, %r5, {delta}, 31, 0xffffffff;");
            ptx.AppendLine($"    mov.b32 %f{scratchReg}, %r6;");
            ptx.AppendLine($"    max.f32 %f{accReg}, %f{accReg}, %f{scratchReg};");
        }

        ptx.AppendLine("    setp.eq.u32 %p0, %r3, 0;");
        ptx.AppendLine($"    mul.wide.u32 %rd7, %r4, {sizeof(float)};");
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd7;");
        ptx.AppendLine($"    @%p0 st.global.f32 [%rd8], %f{accReg};");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture,
        int rows,
        int spatial,
        int blockThreads)
    {
        var inputExtent = new DirectPtxExtent(rows, spatial);
        var outputExtent = new DirectPtxExtent(rows);
        return new DirectPtxKernelBlueprint(
            Operation: "global-maxpool-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"warp-plane-b{blockThreads}-r{rows}-s{spatial}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    inputExtent, inputExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outputExtent, outputExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 8),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[bc] = max over the H*W plane of input[bc, :]",
                ["mode"] = "inference-forward-warp-plane-max",
                ["input"] = "fp32 row-major [batch*channels, H*W]",
                ["output"] = "fp32 [batch*channels]",
                ["values-per-lane"] = (spatial / 32).ToString(),
                ["identity"] = "-inf, matching the reference maxVal = -INFINITY",
                ["nan-policy"] = "max.f32 returns the non-NaN operand; an all-NaN plane yields -inf, as in the reference",
                ["arg-max-indices"] = "not produced; the saveIndices path stays on the established kernel",
                ["global-input-reads"] = "one-vector-per-lane",
                ["global-output-writes"] = "one-fp32-per-plane-from-lane-zero",
                ["warp-reduction"] = "butterfly-shuffle-through-b32",
                ["shared-intermediate"] = "none",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    /// <summary>Mirrors the global-average-pool family's exact shape buckets.</summary>
    internal static bool IsSupportedShape(int rows, int spatial) =>
        (rows, spatial) is (256, 128) or (2048, 64) or (2048, 128) or (8192, 128);

    internal static bool IsPromotedShape(int rows, int spatial) => false;

    private static void Validate(int rows, int spatial)
    {
        if (!IsSupportedShape(rows, spatial))
            throw new ArgumentOutOfRangeException(nameof(rows),
                "The first global-max-pool family supports exact (rows=batch*channels, spatial=H*W) " +
                "buckets (256,128), (2048,64), (2048,128), and (8192,128).");
        int valuesPerLane = spatial / 32;
        if (spatial % 32 != 0 || valuesPerLane is not (2 or 4))
            throw new ArgumentOutOfRangeException(nameof(spatial),
                "The plane must divide into 32 lanes at a v2 or v4 vector width.");
    }

    private static void ValidateBlockThreads(int rows, int blockThreads)
    {
        if (blockThreads is not (64 or 128 or 256) || rows % (blockThreads / 32) != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Global max-pool block threads must be 64, 128, or 256 and evenly tile the planes.");
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
