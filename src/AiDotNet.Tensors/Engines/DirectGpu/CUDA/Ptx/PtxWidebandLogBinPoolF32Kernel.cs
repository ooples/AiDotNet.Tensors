using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Wideband log-bin pooling for issue #850, matching the NVRTC <c>wideband_log_bin_pool</c> kernel: pool a
/// magnitude spectrum into <c>numBins</c> logarithmically-spaced bins and take <c>log1p</c> of each bin
/// average. One thread owns one output bin <c>(seg, k)</c>; it derives the quadratic log-bin edges
/// <c>binStart = 1 + floor((k/numBins)^2 * (usable-1))</c> (and the next edge for <c>binEnd</c>), averages
/// <c>magBuf</c> over that half-open range, and writes <c>log1p(avg)</c>. The bin math uses integer floor
/// casts and the log is <c>lg2.approx(1+avg)*ln(2)</c>, so the spec is TOLERANCE-based. <c>totalSegBatch</c>,
/// <c>fftSize</c>, <c>numBins</c>, and <c>usable</c> are baked; the launch rounds up and a single guard drops
/// the tail lanes. Two pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxWidebandLogBinPoolF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_wideband_log_bin_pool_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int TotalSegBatch { get; }
    internal int FftSize { get; }
    internal int NumBins { get; }
    internal int Usable { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxWidebandLogBinPoolF32Kernel(
        DirectPtxRuntime runtime, int totalSegBatch, int fftSize, int numBins, int usable,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in wideband-log-bin-pool specialization is admitted only on SM86.");
        Validate(totalSegBatch, fftSize, numBins, usable);
        ValidateBlockThreads(blockThreads);
        TotalSegBatch = totalSegBatch;
        FftSize = fftSize;
        NumBins = numBins;
        Usable = usable;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, totalSegBatch, fftSize, numBins, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, totalSegBatch, fftSize, numBins, usable, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView magBuf, DirectPtxTensorView output)
    {
        Require(magBuf, Blueprint.Tensors[0], nameof(magBuf));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr magPointer = magBuf.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &magPointer;
        arguments[1] = &outputPointer;
        int total = TotalSegBatch * NumBins;
        _module.Launch(
            _function,
            (uint)((total + BlockThreads - 1) / BlockThreads), 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int totalSegBatch, int fftSize, int numBins, int usable, int blockThreads = DefaultBlockThreads)
    {
        Validate(totalSegBatch, fftSize, numBins, usable);
        ValidateBlockThreads(blockThreads);
        int total = checked(totalSegBatch * numBins);
        string numBinsF = Hex(numBins), usm1F = Hex(usable - 1), ln2 = Hex((float)Math.Log(2.0));

        var ptx = new StringBuilder(3_328);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape totalSegBatch={totalSegBatch} fftSize={fftSize} numBins={numBins} usable={usable} block={blockThreads} op=wideband-log-bin-pool");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 mag_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<8>;");
        ptx.AppendLine("    .reg .f32 %f<8>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [mag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // outIdx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $WB_RET;");
        ptx.AppendLine($"    div.u32 %r3, %r2, {numBins};");            // seg
        ptx.AppendLine($"    rem.u32 %r4, %r2, {numBins};");            // k
        // binStart = 1 + (int)((k/numBins)^2 * (usable-1))
        ptx.AppendLine("    cvt.rn.f32.u32 %f0, %r4;");
        ptx.AppendLine($"    div.rn.f32 %f0, %f0, {numBinsF};");        // r0
        ptx.AppendLine("    mul.rn.f32 %f0, %f0, %f0;");
        ptx.AppendLine($"    mul.rn.f32 %f0, %f0, {usm1F};");
        ptx.AppendLine("    cvt.rzi.s32.f32 %r5, %f0;");
        ptx.AppendLine("    add.s32 %r5, %r5, 1;");                     // binStart
        // binEnd = 1 + (int)(((k+1)/numBins)^2 * (usable-1))
        ptx.AppendLine("    add.u32 %r6, %r4, 1;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f1, %r6;");
        ptx.AppendLine($"    div.rn.f32 %f1, %f1, {numBinsF};");        // r1
        ptx.AppendLine("    mul.rn.f32 %f1, %f1, %f1;");
        ptx.AppendLine($"    mul.rn.f32 %f1, %f1, {usm1F};");
        ptx.AppendLine("    cvt.rzi.s32.f32 %r6, %f1;");
        ptx.AppendLine("    add.s32 %r6, %r6, 1;");                     // binEnd
        // if (binEnd <= binStart) binEnd = binStart + 1
        ptx.AppendLine("    add.s32 %r7, %r5, 1;");
        ptx.AppendLine("    setp.le.s32 %p1, %r6, %r5;");
        ptx.AppendLine("    selp.b32 %r6, %r7, %r6, %p1;");
        // if (binEnd > usable) binEnd = usable
        ptx.AppendLine($"    min.s32 %r6, %r6, {usable};");
        ptx.AppendLine($"    mul.lo.u32 %r8, %r3, {fftSize};");        // magOff
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");                 // sum
        ptx.AppendLine("    mov.u32 %r9, 0;");                          // cnt
        ptx.AppendLine("    mov.s32 %r10, %r5;");                       // i = binStart
        ptx.AppendLine("$WB_LOOP:");
        ptx.AppendLine("    setp.ge.s32 %p2, %r10, %r6;");
        ptx.AppendLine("    @%p2 bra $WB_AVG;");
        ptx.AppendLine("    add.s32 %r11, %r8, %r10;");
        ptx.AppendLine("    mul.wide.s32 %rd2, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd3];");
        ptx.AppendLine("    add.rn.f32 %f2, %f2, %f3;");
        ptx.AppendLine("    add.u32 %r9, %r9, 1;");
        ptx.AppendLine("    add.s32 %r10, %r10, 1;");
        ptx.AppendLine("    bra $WB_LOOP;");
        ptx.AppendLine("$WB_AVG:");
        ptx.AppendLine("    cvt.rn.f32.u32 %f4, %r9;");
        ptx.AppendLine("    div.rn.f32 %f5, %f2, %f4;");
        ptx.AppendLine("    setp.gt.u32 %p3, %r9, 0;");
        ptx.AppendLine("    selp.f32 %f5, %f5, 0f00000000, %p3;");      // avg
        ptx.AppendLine("    add.rn.f32 %f5, %f5, 0f3F800000;");         // 1 + avg
        ptx.AppendLine("    lg2.approx.f32 %f5, %f5;");
        ptx.AppendLine($"    mul.rn.f32 %f5, %f5, {ln2};");            // log1p(avg)
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f5;");
        ptx.AppendLine("$WB_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int totalSegBatch, int fftSize, int numBins, int blockThreads)
    {
        var magExtent = new DirectPtxExtent(checked(totalSegBatch * fftSize));
        var outExtent = new DirectPtxExtent(checked(totalSegBatch * numBins));
        return new DirectPtxKernelBlueprint(
            Operation: "wideband-log-bin-pool-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-seg{totalSegBatch}-fft{fftSize}-bins{numBins}",
            Tensors:
            [
                new("magBuf", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    magExtent, magExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[seg,k] = log1p(mean magBuf over [binStart,binEnd)); quadratic log-bin edges",
                ["mode"] = "inference-forward-wideband-log-bin-pool",
                ["arithmetic"] = "integer floor bin edges, mean, lg2.approx(1+avg)*ln(2); tolerance-based parity",
                ["loop"] = "per output bin, over its quadratic log-spaced magnitude range",
                ["bounds-check"] = "single guard drops lanes past totalSegBatch*numBins; bin range clamped to usable",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int totalSegBatch, int fftSize, int numBins, int usable) =>
        totalSegBatch >= 1 && fftSize >= 2 && numBins >= 1 && usable >= 2 && usable <= fftSize &&
        (long)totalSegBatch * fftSize <= (1L << 26) && (long)totalSegBatch * numBins <= (1L << 24);

    internal static bool IsPromotedShape(int totalSegBatch, int fftSize, int numBins, int usable) => false;

    private static void Validate(int totalSegBatch, int fftSize, int numBins, int usable)
    {
        if (!IsSupportedShape(totalSegBatch, fftSize, numBins, usable))
            throw new ArgumentOutOfRangeException(nameof(usable),
                "The wideband-log-bin-pool family requires totalSegBatch>=1, fftSize>=2, numBins>=1, " +
                "2<=usable<=fftSize, totalSegBatch*fftSize<=2^26, and totalSegBatch*numBins<=2^24.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Wideband-log-bin-pool block threads must be 128, 256, or 512.");
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
