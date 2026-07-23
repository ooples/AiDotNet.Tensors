using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Segmented mel filterbank application for issue #850, matching the NVRTC <c>mel_filterbank_apply</c>
/// kernel: <c>melEnergy[seg,m] = sum_i powerSpec[seg,i] * melFilters[m,i]</c> over the spectral bins. One
/// thread owns one <c>(seg, mel)</c> output and reduces over <c>specBins</c> with <c>fma.rn</c>, matching
/// the reference's fused sum, so the parity spec is tolerance-based against the fp64 oracle. Unlike
/// <see cref="PtxApplyMelFilterbankF32Kernel"/> (which bakes exact frame/mel counts and launches without a
/// guard), this variant mirrors the perf-path backend method: <c>totalSegBatch</c>, <c>specBins</c>, and
/// <c>melBins</c> are baked, the launch rounds up, and a single guard drops the tail lanes. Three pointers
/// reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxMelFilterbankApplyF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_mel_filterbank_apply_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int TotalSegBatch { get; }
    internal int SpecBins { get; }
    internal int MelBins { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxMelFilterbankApplyF32Kernel(
        DirectPtxRuntime runtime, int totalSegBatch, int specBins, int melBins,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in segmented mel-filterbank specialization is admitted only on SM86.");
        Validate(totalSegBatch, specBins, melBins);
        ValidateBlockThreads(blockThreads);
        TotalSegBatch = totalSegBatch;
        SpecBins = specBins;
        MelBins = melBins;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, totalSegBatch, specBins, melBins, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            totalSegBatch, specBins, melBins, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView powerSpec, DirectPtxTensorView melFilters, DirectPtxTensorView melEnergy)
    {
        Require(powerSpec, Blueprint.Tensors[0], nameof(powerSpec));
        Require(melFilters, Blueprint.Tensors[1], nameof(melFilters));
        Require(melEnergy, Blueprint.Tensors[2], nameof(melEnergy));

        IntPtr powerPointer = powerSpec.Pointer, filterPointer = melFilters.Pointer, energyPointer = melEnergy.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &powerPointer;
        arguments[1] = &filterPointer;
        arguments[2] = &energyPointer;
        int total = TotalSegBatch * MelBins;
        _module.Launch(
            _function,
            (uint)((total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int totalSegBatch, int specBins, int melBins, int blockThreads = DefaultBlockThreads)
    {
        Validate(totalSegBatch, specBins, melBins);
        ValidateBlockThreads(blockThreads);
        int total = checked(totalSegBatch * melBins);

        var ptx = new StringBuilder(2_560);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape totalSegBatch={totalSegBatch} specBins={specBins} melBins={melBins} block={blockThreads} op=mel-filterbank-apply");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 power_ptr,");
        ptx.AppendLine("    .param .u64 filter_ptr,");
        ptx.AppendLine("    .param .u64 energy_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<9>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [power_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [filter_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [energy_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // outIdx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $MFA_RET;");
        ptx.AppendLine($"    div.u32 %r3, %r2, {melBins};");              // seg
        ptx.AppendLine($"    rem.u32 %r4, %r2, {melBins};");              // m
        ptx.AppendLine($"    mul.lo.u32 %r5, %r3, {specBins};");         // powerOff = seg*specBins
        ptx.AppendLine($"    mul.lo.u32 %r6, %r4, {specBins};");         // filtOff = m*specBins
        ptx.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");                   // powerBase
        ptx.AppendLine("    mul.wide.u32 %rd5, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");                   // filterBase
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                    // sum
        ptx.AppendLine("    mov.u32 %r7, 0;");                             // i
        ptx.AppendLine("$MFA_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r7, {specBins};");
        ptx.AppendLine("    @%p1 bra $MFA_WRITE;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r7, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd4, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd6, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd8];");   // powerSpec[i]
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd9];");   // melFilters[i]
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");  // sum += p*f (matches fused reference)
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine("    bra $MFA_LOOP;");
        ptx.AppendLine("$MFA_WRITE:");
        ptx.AppendLine("    mul.wide.u32 %rd10, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f0;");
        ptx.AppendLine("$MFA_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int totalSegBatch, int specBins, int melBins, int blockThreads)
    {
        var powerExtent = new DirectPtxExtent(checked(totalSegBatch * specBins));
        var filterExtent = new DirectPtxExtent(checked(melBins * specBins));
        var energyExtent = new DirectPtxExtent(checked(totalSegBatch * melBins));
        return new DirectPtxKernelBlueprint(
            Operation: "mel-filterbank-apply-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-seg{totalSegBatch}-spec{specBins}-mel{melBins}",
            Tensors:
            [
                new("powerSpec", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    powerExtent, powerExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("melFilters", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    filterExtent, filterExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("melEnergy", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    energyExtent, energyExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "melEnergy[seg,m] = sum_i powerSpec[seg,i] * melFilters[m,i]",
                ["mode"] = "inference-forward-mel-filterbank-apply",
                ["arithmetic"] = "fma.rn reduction matching the reference's fused sum; tolerance-based parity",
                ["loop"] = "per (seg,mel) output, reduce over specBins",
                ["bounds-check"] = "single guard drops lanes past totalSegBatch*melBins",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int totalSegBatch, int specBins, int melBins) =>
        totalSegBatch >= 1 && specBins >= 1 && melBins >= 1 &&
        (long)totalSegBatch * specBins <= (1L << 26) &&
        (long)totalSegBatch * melBins <= (1L << 24) &&
        (long)melBins * specBins <= (1L << 24);

    internal static bool IsPromotedShape(int totalSegBatch, int specBins, int melBins) => false;

    private static void Validate(int totalSegBatch, int specBins, int melBins)
    {
        if (!IsSupportedShape(totalSegBatch, specBins, melBins))
            throw new ArgumentOutOfRangeException(nameof(totalSegBatch),
                "The segmented mel-filterbank family requires totalSegBatch>=1, specBins>=1, melBins>=1, " +
                "totalSegBatch*specBins<=2^26, and totalSegBatch*melBins, melBins*specBins<=2^24.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Segmented mel-filterbank block threads must be 128, 256, or 512.");
    }

    private static void Require(DirectPtxTensorView view, DirectPtxTensorContract contract, string parameter)
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
