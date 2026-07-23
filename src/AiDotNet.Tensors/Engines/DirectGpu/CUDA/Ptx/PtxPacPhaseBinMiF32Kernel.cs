using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Phase-amplitude coupling (PAC) phase-binned modulation index for issue #850, matching the deterministic
/// NVRTC <c>pac_phase_bin_mi_deterministic</c> kernel: for one <c>(batch, gamma band)</c> it bins the gamma
/// amplitude by theta phase into 18 bins, then computes the Tort modulation index - the normalized
/// KL-divergence of the mean-amplitude distribution from uniform. The block is 18 threads (one phase bin
/// each); every thread scans all samples in fixed order and sums only the amplitudes whose phase falls in
/// its bin, writing to static shared memory. After a barrier, thread 0 reduces the histogram to the MI. The
/// per-bin scan is order-deterministic and the entropy uses <c>lg2.approx</c> for the natural log, so the
/// spec is TOLERANCE-based. <c>batch</c>, <c>numSamples</c>, <c>numGammaBands</c>, and <c>gammaIdx</c> are
/// baked; the launch uses one 18-thread block per batch row. Three pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxPacPhaseBinMiF32Kernel : IDisposable
{
    internal const int NumPhaseBins = 18;
    internal const string EntryPoint = "aidotnet_pac_phase_bin_mi_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int NumSamples { get; }
    internal int NumGammaBands { get; }
    internal int GammaIdx { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPacPhaseBinMiF32Kernel(
        DirectPtxRuntime runtime, int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in PAC modulation-index specialization is admitted only on SM86.");
        Validate(batch, numSamples, numGammaBands, gammaIdx);
        Batch = batch;
        NumSamples = numSamples;
        NumGammaBands = numGammaBands;
        GammaIdx = gammaIdx;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, numSamples, numGammaBands, gammaIdx);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, batch, numSamples, numGammaBands, gammaIdx);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, NumPhaseBins);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, NumPhaseBins, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, NumPhaseBins, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView thetaPhase, DirectPtxTensorView gammaAmp, DirectPtxTensorView output)
    {
        Require(thetaPhase, Blueprint.Tensors[0], nameof(thetaPhase));
        Require(gammaAmp, Blueprint.Tensors[1], nameof(gammaAmp));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr thetaPointer = thetaPhase.Pointer, gammaPointer = gammaAmp.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &thetaPointer;
        arguments[1] = &gammaPointer;
        arguments[2] = &outputPointer;
        _module.Launch(
            _function,
            checked((uint)Batch), 1, 1,
            NumPhaseBins, 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    private static string Hex(float value) => "0f" + BitConverter.ToInt32(BitConverter.GetBytes(value), 0).ToString("X8");

    internal static string EmitPtx(int ccMajor, int ccMinor, int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        Validate(batch, numSamples, numGammaBands, gammaIdx);
        string pi = Hex((float)Math.PI);
        string binScale = Hex((float)(NumPhaseBins / (2.0 * Math.PI)));   // (phase+pi) * 18/(2pi)
        string ln2 = Hex((float)Math.Log(2.0));
        string lnN = Hex((float)Math.Log(NumPhaseBins));
        string invLnN = Hex((float)(1.0 / Math.Log(NumPhaseBins)));
        string tiny = Hex(1e-12f);
        int gbBase = checked(gammaIdx * batch);      // (gammaIdx*batch + b) * numSamples
        int countsByteOffset = NumPhaseBins * 4;     // countsD starts after 18 sums

        var ptx = new StringBuilder(4_608);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape batch={batch} numSamples={numSamples} numGammaBands={numGammaBands} gammaIdx={gammaIdx} bins={NumPhaseBins} op=pac-phase-bin-mi");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 theta_ptr,");
        ptx.AppendLine("    .param .u64 gamma_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {NumPhaseBins}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine($"    .shared .align 4 .b32 pacHist[{2 * NumPhaseBins}];");   // [18 sums | 18 counts]
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<20>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [theta_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [gamma_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");                        // tid = phase bin
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");                      // b
        ptx.AppendLine("    mov.u32 %r2, pacHist;");                       // shared base
        // sampleOff = b*numSamples ; gammaOff = (gbBase + b)*numSamples
        ptx.AppendLine($"    mul.lo.u32 %r3, %r1, {numSamples};");        // sampleOff
        ptx.AppendLine($"    add.u32 %r4, %r1, {gbBase};");
        ptx.AppendLine($"    mul.lo.u32 %r4, %r4, {numSamples};");        // gammaOff
        ptx.AppendLine("    mul.wide.u32 %rd3, %r3, 4;");
        ptx.AppendLine("    add.u64 %rd4, %rd0, %rd3;");                   // thetaBase
        ptx.AppendLine("    mul.wide.u32 %rd5, %r4, 4;");
        ptx.AppendLine("    add.u64 %rd6, %rd1, %rd5;");                   // gammaBase
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                    // mySum
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                    // myCount
        ptx.AppendLine("    mov.u32 %r5, 0;");                             // i
        ptx.AppendLine("$PAC_HIST:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r5, {numSamples};");
        ptx.AppendLine("    @%p0 bra $PAC_STORE;");
        ptx.AppendLine("    mul.wide.u32 %rd7, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd8, %rd4, %rd7;");
        ptx.AppendLine("    add.u64 %rd9, %rd6, %rd7;");
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd8];");   // phase
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd9];");   // amp
        ptx.AppendLine($"    add.rn.f32 %f4, %f2, {pi};");
        ptx.AppendLine($"    mul.rn.f32 %f4, %f4, {binScale};");         // fbin
        ptx.AppendLine("    cvt.rzi.s32.f32 %r6, %f4;");                  // bin = (int)fbin
        ptx.AppendLine("    max.s32 %r6, %r6, 0;");
        ptx.AppendLine($"    min.s32 %r6, %r6, {NumPhaseBins - 1};");
        ptx.AppendLine("    setp.eq.u32 %p1, %r6, %r0;");                 // bin == tid
        ptx.AppendLine("    @%p1 add.rn.f32 %f0, %f0, %f3;");
        ptx.AppendLine("    @%p1 add.rn.f32 %f1, %f1, 0f3F800000;");
        ptx.AppendLine("    add.u32 %r5, %r5, 1;");
        ptx.AppendLine("    bra $PAC_HIST;");
        ptx.AppendLine("$PAC_STORE:");
        ptx.AppendLine("    shl.b32 %r7, %r0, 2;");                       // tid*4
        ptx.AppendLine("    add.u32 %r8, %r2, %r7;");                     // &sumsD[tid]
        ptx.AppendLine("    st.shared.f32 [%r8], %f0;");
        ptx.AppendLine($"    add.u32 %r9, %r8, {countsByteOffset};");     // &countsD[tid]
        ptx.AppendLine("    st.shared.f32 [%r9], %f1;");
        ptx.AppendLine("    bar.sync 0;");
        ptx.AppendLine("    setp.ne.u32 %p0, %r0, 0;");
        ptx.AppendLine("    @%p0 bra $PAC_RET;");
        // thread 0: totalAmp = sum_k avg_k
        ptx.AppendLine("    mov.f32 %f5, 0f00000000;");                    // totalAmp
        ptx.AppendLine("    mov.u32 %r10, 0;");                            // k
        ptx.AppendLine("$PAC_TOTAL:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r10, {NumPhaseBins};");
        ptx.AppendLine("    @%p1 bra $PAC_AFTER_TOTAL;");
        ptx.AppendLine("    shl.b32 %r11, %r10, 2;");
        ptx.AppendLine("    add.u32 %r12, %r2, %r11;");
        ptx.AppendLine("    ld.shared.f32 %f6, [%r12];");                 // sumsD[k]
        ptx.AppendLine($"    ld.shared.f32 %f7, [%r12+{countsByteOffset}];"); // countsD[k]
        ptx.AppendLine("    div.rn.f32 %f8, %f6, %f7;");
        ptx.AppendLine("    setp.gt.f32 %p2, %f7, 0f00000000;");
        ptx.AppendLine("    selp.f32 %f8, %f8, 0f00000000, %p2;");        // avg
        ptx.AppendLine("    add.rn.f32 %f5, %f5, %f8;");
        ptx.AppendLine("    add.u32 %r10, %r10, 1;");
        ptx.AppendLine("    bra $PAC_TOTAL;");
        ptx.AppendLine("$PAC_AFTER_TOTAL:");
        ptx.AppendLine("    mov.f32 %f9, 0f00000000;");                    // mi
        ptx.AppendLine("    setp.gt.f32 %p2, %f5, 0f00000000;");
        ptx.AppendLine("    @!%p2 bra $PAC_WRITE;");
        // entropy = -sum p*ln(p)
        ptx.AppendLine("    mov.f32 %f10, 0f00000000;");                   // entropy
        ptx.AppendLine("    mov.u32 %r10, 0;");
        ptx.AppendLine("$PAC_ENT:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r10, {NumPhaseBins};");
        ptx.AppendLine("    @%p1 bra $PAC_AFTER_ENT;");
        ptx.AppendLine("    shl.b32 %r11, %r10, 2;");
        ptx.AppendLine("    add.u32 %r12, %r2, %r11;");
        ptx.AppendLine("    ld.shared.f32 %f6, [%r12];");
        ptx.AppendLine($"    ld.shared.f32 %f7, [%r12+{countsByteOffset}];");
        ptx.AppendLine("    div.rn.f32 %f8, %f6, %f7;");
        ptx.AppendLine("    setp.gt.f32 %p2, %f7, 0f00000000;");
        ptx.AppendLine("    selp.f32 %f8, %f8, 0f00000000, %p2;");        // avg
        ptx.AppendLine("    div.rn.f32 %f11, %f8, %f5;");                 // p = avg/totalAmp
        ptx.AppendLine($"    setp.gt.f32 %p3, %f11, {tiny};");
        ptx.AppendLine("    lg2.approx.f32 %f12, %f11;");
        ptx.AppendLine($"    mul.rn.f32 %f12, %f12, {ln2};");            // ln(p)
        ptx.AppendLine("    mul.rn.f32 %f12, %f11, %f12;");              // p*ln(p)
        ptx.AppendLine("    sub.rn.f32 %f13, %f10, %f12;");              // entropy - p*ln(p)
        ptx.AppendLine("    selp.f32 %f10, %f13, %f10, %p3;");           // add only when p > tiny
        ptx.AppendLine("    add.u32 %r10, %r10, 1;");
        ptx.AppendLine("    bra $PAC_ENT;");
        ptx.AppendLine("$PAC_AFTER_ENT:");
        ptx.AppendLine($"    sub.rn.f32 %f9, {lnN}, %f10;");            // ln(N) - entropy
        ptx.AppendLine($"    mul.rn.f32 %f9, %f9, {invLnN};");          // / ln(N)
        ptx.AppendLine("$PAC_WRITE:");
        ptx.AppendLine($"    mad.lo.u32 %r13, %r1, {numGammaBands}, {gammaIdx};");   // b*numGammaBands + gammaIdx
        ptx.AppendLine("    mul.wide.u32 %rd10, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd2, %rd10;");
        ptx.AppendLine("    st.global.f32 [%rd11], %f9;");
        ptx.AppendLine("$PAC_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        var thetaExtent = new DirectPtxExtent(checked(batch * numSamples));
        var gammaExtent = new DirectPtxExtent(checked(numGammaBands * batch * numSamples));
        var outExtent = new DirectPtxExtent(checked(batch * numGammaBands));
        return new DirectPtxKernelBlueprint(
            Operation: "pac-phase-bin-mi-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"bins{NumPhaseBins}-bt{batch}-ns{numSamples}-gb{numGammaBands}-gi{gammaIdx}",
            Tensors:
            [
                new("thetaPhase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    thetaExtent, thetaExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("gammaAmp", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    gammaExtent, gammaExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 2 * NumPhaseBins * 4,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "MI = (ln(18) - H) / ln(18); H = -sum_k p_k ln(p_k); p_k = mean amp in phase bin k / total",
                ["mode"] = "inference-forward-pac-phase-bin-mi",
                ["arithmetic"] = "deterministic per-bin scan + lg2.approx natural log; tolerance-based parity",
                ["shared-memory"] = "18 amplitude sums + 18 counts (static, one block of 18 threads per batch row)",
                ["determinism"] = "order-fixed per-bin accumulation (no atomics), matching the deterministic reference",
                ["bounds-check"] = "block covers exactly the 18 phase bins; grid covers exactly the batch",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int batch, int numSamples, int numGammaBands, int gammaIdx) =>
        batch >= 1 && numSamples >= 1 && numGammaBands >= 1 && gammaIdx >= 0 && gammaIdx < numGammaBands &&
        (long)numGammaBands * batch * numSamples <= (1L << 27);

    internal static bool IsPromotedShape(int batch, int numSamples, int numGammaBands, int gammaIdx) => false;

    private static void Validate(int batch, int numSamples, int numGammaBands, int gammaIdx)
    {
        if (!IsSupportedShape(batch, numSamples, numGammaBands, gammaIdx))
            throw new ArgumentOutOfRangeException(nameof(gammaIdx),
                "The PAC modulation-index family requires batch>=1, numSamples>=1, numGammaBands>=1, " +
                "0<=gammaIdx<numGammaBands, and numGammaBands*batch*numSamples<=2^27.");
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
