using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Analytic-signal frequency mask for issue #850, matching the NVRTC <c>analytic_signal_mask</c> kernel:
/// the frequency-domain step of a Hilbert transform. For each spectral bin <c>k</c> of every batch row it
/// multiplies the split complex spectrum by a Hilbert gain - <c>0</c> for negative frequencies,
/// <c>2</c> for positive frequencies, <c>1</c> for DC and Nyquist - additionally zeroed outside the
/// half-open pass band <c>[binLow, binHigh)</c>. One thread owns one bin; the gain is a compile-time-shaped
/// branch over <c>fftSize/binLow/binHigh</c> resolved with predicates, and the output is a single
/// <c>mul</c>, so the spec is bit-exact (the gains are exact powers of two). <c>batch</c>, <c>fftSize</c>,
/// <c>binLow</c>, and <c>binHigh</c> are baked; the launch rounds up and a single guard drops the tail
/// lanes. Four pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxAnalyticSignalMaskF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_analytic_signal_mask_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int FftSize { get; }
    internal int BinLow { get; }
    internal int BinHigh { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxAnalyticSignalMaskF32Kernel(
        DirectPtxRuntime runtime, int batch, int fftSize, int binLow, int binHigh,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexMultiply(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in analytic-signal-mask specialization is admitted only on SM86.");
        Validate(batch, fftSize, binLow, binHigh);
        ValidateBlockThreads(blockThreads);
        Batch = batch;
        FftSize = fftSize;
        BinLow = binLow;
        BinHigh = binHigh;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, fftSize, binLow, binHigh, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, fftSize, binLow, binHigh, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView specReal, DirectPtxTensorView specImag,
        DirectPtxTensorView outReal, DirectPtxTensorView outImag)
    {
        Require(specReal, Blueprint.Tensors[0], nameof(specReal));
        Require(specImag, Blueprint.Tensors[1], nameof(specImag));
        Require(outReal, Blueprint.Tensors[2], nameof(outReal));
        Require(outImag, Blueprint.Tensors[3], nameof(outImag));

        IntPtr specRealPointer = specReal.Pointer, specImagPointer = specImag.Pointer;
        IntPtr outRealPointer = outReal.Pointer, outImagPointer = outImag.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &specRealPointer;
        arguments[1] = &specImagPointer;
        arguments[2] = &outRealPointer;
        arguments[3] = &outImagPointer;
        int total = Batch * FftSize;
        _module.Launch(
            _function,
            (uint)((total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int batch, int fftSize, int binLow, int binHigh, int blockThreads = DefaultBlockThreads)
    {
        Validate(batch, fftSize, binLow, binHigh);
        ValidateBlockThreads(blockThreads);
        int total = checked(batch * fftSize);
        int halfN = fftSize >> 1;

        var ptx = new StringBuilder(2_816);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape batch={batch} fftSize={fftSize} binLow={binLow} binHigh={binHigh} halfN={halfN} block={blockThreads} op=analytic-signal-mask");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 specreal_ptr,");
        ptx.AppendLine("    .param .u64 specimag_ptr,");
        ptx.AppendLine("    .param .u64 outreal_ptr,");
        ptx.AppendLine("    .param .u64 outimag_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<8>;");
        ptx.AppendLine("    .reg .b32 %r<4>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [specreal_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [specimag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [outreal_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [outimag_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $ASM_RET;");
        ptx.AppendLine($"    rem.u32 %r3, %r2, {fftSize};");              // k
        // base = (k < halfN) ? 2.0 : 0.0
        ptx.AppendLine($"    setp.lt.u32 %p1, %r3, {halfN};");
        ptx.AppendLine("    selp.f32 %f0, 0f40000000, 0f00000000, %p1;");   // 2.0 : 0.0
        // isEdge = (k == 0) || (k == halfN)
        ptx.AppendLine("    setp.eq.u32 %p2, %r3, 0;");
        ptx.AppendLine($"    setp.eq.u32 %p3, %r3, {halfN};");
        ptx.AppendLine("    or.pred %p4, %p2, %p3;");
        ptx.AppendLine("    selp.f32 %f0, 0f3F800000, %f0, %p4;");          // isEdge ? 1.0 : base
        // outOfBand = (k < binLow) || (k >= binHigh)
        ptx.AppendLine($"    setp.lt.u32 %p5, %r3, {binLow};");
        ptx.AppendLine($"    setp.ge.u32 %p6, %r3, {binHigh};");
        ptx.AppendLine("    or.pred %p7, %p5, %p6;");
        ptx.AppendLine("    selp.f32 %f0, 0f00000000, %f0, %p7;");          // outOfBand ? 0.0 : gain
        // out = spec * gain
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");   // specReal
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd6];");   // specImag
        ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f0;");
        ptx.AppendLine("    mul.rn.f32 %f2, %f2, %f0;");
        ptx.AppendLine("    add.u64 %rd7, %rd2, %rd4;");
        ptx.AppendLine("    add.u64 %rd8, %rd3, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f1;");
        ptx.AppendLine("    st.global.f32 [%rd8], %f2;");
        ptx.AppendLine("$ASM_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batch, int fftSize, int binLow, int binHigh, int blockThreads)
    {
        var extent = new DirectPtxExtent(checked(batch * fftSize));
        return new DirectPtxKernelBlueprint(
            Operation: "analytic-signal-mask-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-bt{batch}-fft{fftSize}-lo{binLow}-hi{binHigh}",
            Tensors:
            [
                new("specReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("specImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("outReal", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("outImag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    extent, extent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "out = spec * hilbert_gain(k); gain=0 neg / 2 pos / 1 DC,Nyquist, zeroed outside [binLow,binHigh)",
                ["mode"] = "inference-forward-analytic-signal-mask",
                ["arithmetic"] = "predicated selp gain (exact powers of two) then one mul per lane; bit-exact",
                ["bounds-check"] = "single guard drops lanes past batch*fftSize",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int batch, int fftSize, int binLow, int binHigh) =>
        batch >= 1 && fftSize >= 2 && binLow >= 0 && binHigh >= binLow && binHigh <= fftSize &&
        (long)batch * fftSize <= (1L << 26);

    internal static bool IsPromotedShape(int batch, int fftSize, int binLow, int binHigh) => false;

    private static void Validate(int batch, int fftSize, int binLow, int binHigh)
    {
        if (!IsSupportedShape(batch, fftSize, binLow, binHigh))
            throw new ArgumentOutOfRangeException(nameof(fftSize),
                "The analytic-signal-mask family requires batch>=1, fftSize>=2, 0<=binLow<=binHigh<=fftSize, " +
                "and batch*fftSize<=2^26.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Analytic-signal-mask block threads must be 128, 256, or 512.");
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
