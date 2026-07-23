using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Mel filterbank application for issue #850, matching the NVRTC <c>apply_mel_filterbank</c> kernel:
/// <c>melSpec[frame, mel] = sum_f powerSpec[frame, f] * filterbank[mel, f]</c>. One thread owns one
/// (frame, mel) output cell and walks the <c>numFreqs</c> frequency axis serially in registers,
/// accumulating with <c>fma</c> to match the reference's fused evaluation - no shared memory, no
/// reduction across threads, no branch. Shape (numFrames, numFreqs, nMels) is baked into the PTX, so the
/// launch takes buffer pointers only. 256 threads/block, grid = (numFrames*nMels)/256.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxApplyMelFilterbankF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const int MaxCells = 2048 * 4096;
    internal const int MaxFreqs = 65536;
    internal const string EntryPoint = "aidotnet_apply_mel_filterbank_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumFrames { get; }
    internal int NumFreqs { get; }
    internal int NumMels { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxApplyMelFilterbankF32Kernel(
        DirectPtxRuntime runtime, int numFrames, int numFreqs, int nMels, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in mel-filterbank specialization is admitted only on SM86.");
        Validate(numFrames, numFreqs, nMels);
        ValidateBlockThreads(numFrames, nMels, blockThreads);
        NumFrames = numFrames;
        NumFreqs = numFreqs;
        NumMels = nMels;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numFrames, numFreqs, nMels, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, numFrames, numFreqs, nMels, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView powerSpec, DirectPtxTensorView filterbank, DirectPtxTensorView melSpec)
    {
        Require(powerSpec, Blueprint.Tensors[0], nameof(powerSpec));
        Require(filterbank, Blueprint.Tensors[1], nameof(filterbank));
        Require(melSpec, Blueprint.Tensors[2], nameof(melSpec));

        IntPtr powerPointer = powerSpec.Pointer, filterPointer = filterbank.Pointer, melPointer = melSpec.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &powerPointer;
        arguments[1] = &filterPointer;
        arguments[2] = &melPointer;
        _module.Launch(
            _function,
            checked((uint)((NumFrames * NumMels) / BlockThreads)), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(int ccMajor, int ccMinor, int numFrames, int numFreqs, int nMels, int blockThreads = DefaultBlockThreads)
    {
        Validate(numFrames, numFreqs, nMels);
        ValidateBlockThreads(numFrames, nMels, blockThreads);

        var ptx = new StringBuilder(3_000);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape frames={numFrames} freqs={numFreqs} mels={nMels} block={blockThreads} op=apply-mel-filterbank");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 power_ptr,");
        ptx.AppendLine("    .param .u64 filter_ptr,");
        ptx.AppendLine("    .param .u64 mel_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<2>;");
        ptx.AppendLine("    .reg .b32 %r<12>;");
        ptx.AppendLine("    .reg .b64 %rd<14>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [power_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [filter_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [mel_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // gid
        ptx.AppendLine($"    div.u32 %r3, %r2, {nMels};");                 // frame
        ptx.AppendLine($"    rem.u32 %r4, %r2, {nMels};");                 // mel
        // powerBase = frame*numFreqs ; filterBase = mel*numFreqs
        ptx.AppendLine($"    mul.lo.u32 %r5, %r3, {numFreqs};");
        ptx.AppendLine($"    mul.lo.u32 %r6, %r4, {numFreqs};");
        ptx.AppendLine("    mul.wide.u32 %rd3, %r5, 4;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd0, %rd3;");                   // &powerSpec[frame,0]
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd4;");                   // &filterbank[mel,0]
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                   // sum
        ptx.AppendLine("    mov.u32 %r7, 0;");                            // f = 0
        ptx.AppendLine("$MEL_F_LOOP:");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd6];");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");             // sum += power*filter
        ptx.AppendLine("    add.u64 %rd6, %rd6, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd7, 4;");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine($"    setp.lt.u32 %p0, %r7, {numFreqs};");
        ptx.AppendLine("    @%p0 bra $MEL_F_LOOP;");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int numFrames, int numFreqs, int nMels, int blockThreads)
    {
        var powerExtent = new DirectPtxExtent(numFrames * numFreqs);
        var filterExtent = new DirectPtxExtent(nMels * numFreqs);
        var melExtent = new DirectPtxExtent(numFrames * nMels);
        return new DirectPtxKernelBlueprint(
            Operation: "apply-mel-filterbank-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-frames{numFrames}-freqs{numFreqs}-mels{nMels}",
            Tensors:
            [
                new("powerSpec", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    powerExtent, powerExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("filterbank", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    filterExtent, filterExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("melSpec", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.RowMajor2D,
                    melExtent, melExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "melSpec[frame,mel] = sum_f powerSpec[frame,f] * filterbank[mel,f]",
                ["mode"] = "inference-forward-apply-mel-filterbank",
                ["arithmetic"] = "fma.rn accumulation matching the reference's fused sum",
                ["input-layout"] = "row-major powerSpec [frames,freqs] and filterbank [mels,freqs]",
                ["output-layout"] = "row-major melSpec [frames,mels]",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["stride-parameters"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int numFrames, int numFreqs, int nMels)
    {
        if (numFrames <= 0 || numFreqs <= 0 || nMels <= 0 || numFreqs > MaxFreqs) return false;
        long cells = (long)numFrames * nMels;
        return cells > 0 && cells % DefaultBlockThreads == 0 && cells <= MaxCells;
    }

    internal static bool IsPromotedShape(int numFrames, int numFreqs, int nMels) => false;

    private static void Validate(int numFrames, int numFreqs, int nMels)
    {
        if (!IsSupportedShape(numFrames, numFreqs, nMels))
            throw new ArgumentOutOfRangeException(nameof(numFrames),
                $"Mel filterbank requires positive dims with numFreqs<={MaxFreqs} and (numFrames*nMels) a multiple of {DefaultBlockThreads} up to {MaxCells}.");
    }

    private static void ValidateBlockThreads(int numFrames, int nMels, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512) || ((long)numFrames * nMels) % blockThreads != 0)
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Mel filterbank block threads must be 128, 256, or 512 and evenly tile (numFrames*nMels).");
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
