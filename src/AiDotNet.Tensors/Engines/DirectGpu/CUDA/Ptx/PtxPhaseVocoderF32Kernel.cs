using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Phase-vocoder time scaling for issue #850, matching the NVRTC <c>parity210_phase_vocoder</c> kernel:
/// resample each frequency channel along time by <c>rate</c>, linearly interpolating the magnitude and
/// accumulating the wrapped phase advance. One thread owns one <c>(leading, frequency)</c> channel and
/// loops over the <c>outFrames</c> output frames: it reads the two bracketing source frames
/// <c>t0 = floor(t*rate)</c> and <c>t1 = min(t0+1, nFramesV-1)</c>, writes
/// <c>(1-frac)*m0 + frac*m1</c>, and advances a running phase by the wrapped forward difference
/// <c>dp - 2*pi*round(dp/(2*pi))</c>. The wrap uses <c>cvt.rni</c> and the interpolation uses fused
/// arithmetic, so the parity spec is TOLERANCE-based, not bit-exact. <c>leading</c>, <c>nFramesV</c>,
/// <c>nFreqV</c>, and <c>outFrames</c> are baked; <c>rate</c> is a per-launch <c>.param .f32</c>. The launch
/// covers exactly <c>leading * nFreqV</c> threads with no bounds guard. Four pointers plus one f32 scalar
/// reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxPhaseVocoderF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_phase_vocoder_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Leading { get; }
    internal int NFramesV { get; }
    internal int NFreqV { get; }
    internal int OutFrames { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxPhaseVocoderF32Kernel(
        DirectPtxRuntime runtime, int leading, int nFramesV, int nFreqV, int outFrames,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in phase-vocoder specialization is admitted only on SM86.");
        Validate(leading, nFramesV, nFreqV, outFrames);
        ValidateBlockThreads(leading, nFreqV, blockThreads);
        Leading = leading;
        NFramesV = nFramesV;
        NFreqV = nFreqV;
        OutFrames = outFrames;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, leading, nFramesV, nFreqV, outFrames, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            leading, nFramesV, nFreqV, outFrames, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView mag, DirectPtxTensorView phase,
        DirectPtxTensorView newMag, DirectPtxTensorView newPhase, float rate)
    {
        Require(mag, Blueprint.Tensors[0], nameof(mag));
        Require(phase, Blueprint.Tensors[1], nameof(phase));
        Require(newMag, Blueprint.Tensors[2], nameof(newMag));
        Require(newPhase, Blueprint.Tensors[3], nameof(newPhase));

        IntPtr magPointer = mag.Pointer, phasePointer = phase.Pointer;
        IntPtr newMagPointer = newMag.Pointer, newPhasePointer = newPhase.Pointer;
        float rateArg = rate;
        void** arguments = stackalloc void*[5];
        arguments[0] = &magPointer;
        arguments[1] = &phasePointer;
        arguments[2] = &newMagPointer;
        arguments[3] = &newPhasePointer;
        arguments[4] = &rateArg;
        int total = Leading * NFreqV;
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
        int ccMajor, int ccMinor, int leading, int nFramesV, int nFreqV, int outFrames,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(leading, nFramesV, nFreqV, outFrames);
        ValidateBlockThreads(leading, nFreqV, blockThreads);
        int stride = checked(nFramesV * nFreqV);
        int outStride = checked(outFrames * nFreqV);
        int lastFrame = nFramesV - 1;
        int total = checked(leading * nFreqV);
        string twoPi = Hex((float)(2.0 * Math.PI)), invTwoPi = Hex((float)(1.0 / (2.0 * Math.PI)));
        const string one = "0f3F800000";

        var ptx = new StringBuilder(4_352);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape leading={leading} nFramesV={nFramesV} nFreqV={nFreqV} outFrames={outFrames} block={blockThreads} op=phase-vocoder");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 mag_ptr,");
        ptx.AppendLine("    .param .u64 phase_ptr,");
        ptx.AppendLine("    .param .u64 newmag_ptr,");
        ptx.AppendLine("    .param .u64 newphase_ptr,");
        ptx.AppendLine("    .param .f32 rate_val");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<3>;");
        ptx.AppendLine("    .reg .b32 %r<18>;");
        ptx.AppendLine("    .reg .b64 %rd<20>;");
        ptx.AppendLine("    .reg .f32 %f<18>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [mag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [phase_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [newmag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [newphase_ptr];");
        ptx.AppendLine("    ld.param.f32 %f0, [rate_val];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p2, %r2, {total};");
        ptx.AppendLine("    @%p2 bra $PV_RET;");
        ptx.AppendLine($"    rem.u32 %r3, %r2, {nFreqV};");                // f
        ptx.AppendLine($"    div.u32 %r4, %r2, {nFreqV};");                // b
        ptx.AppendLine($"    mul.lo.u32 %r5, %r4, {stride};");            // baseB = b*stride
        ptx.AppendLine($"    mul.lo.u32 %r6, %r4, {outStride};");         // baseOut = b*outStride
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                    // accPhase
        ptx.AppendLine("    mov.u32 %r7, 0;");                             // t
        ptx.AppendLine("$PV_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p0, %r7, {outFrames};");
        ptx.AppendLine("    @%p0 bra $PV_RET;");
        ptx.AppendLine("    cvt.rn.f32.u32 %f2, %r7;");                    // tF
        ptx.AppendLine("    mul.rn.f32 %f3, %f2, %f0;");                   // srcT = t*rate
        ptx.AppendLine("    cvt.rmi.s32.f32 %r8, %f3;");                   // t0 = floor(srcT)
        ptx.AppendLine("    add.s32 %r13, %r8, 1;");                       // t0 + 1
        ptx.AppendLine($"    min.s32 %r9, %r13, {lastFrame};");           // t1 = min(t0+1, nFramesV-1)
        ptx.AppendLine("    cvt.rn.f32.s32 %f4, %r8;");                    // t0F
        ptx.AppendLine("    sub.rn.f32 %f5, %f3, %f4;");                   // frac = srcT - t0
        ptx.AppendLine($"    sub.rn.f32 %f6, {one}, %f5;");               // 1 - frac
        // mag indices: baseB + t*nFreqV + f
        ptx.AppendLine($"    mad.lo.u32 %r10, %r8, {nFreqV}, %r3;");      // t0*nFreqV + f
        ptx.AppendLine("    add.u32 %r10, %r10, %r5;");                    // + baseB (idx0)
        ptx.AppendLine($"    mad.lo.u32 %r11, %r9, {nFreqV}, %r3;");      // t1*nFreqV + f
        ptx.AppendLine("    add.u32 %r11, %r11, %r5;");                    // + baseB (idx1)
        ptx.AppendLine("    mul.wide.u32 %rd4, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r11, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f7, [%rd5];");   // m0
        ptx.AppendLine("    ld.global.nc.f32 %f8, [%rd7];");   // m1
        ptx.AppendLine("    mul.rn.f32 %f9, %f6, %f7;");       // (1-frac)*m0
        ptx.AppendLine("    fma.rn.f32 %f9, %f5, %f8, %f9;");  // + frac*m1
        ptx.AppendLine($"    mad.lo.u32 %r12, %r7, {nFreqV}, %r3;");      // t*nFreqV + f
        ptx.AppendLine("    add.u32 %r12, %r12, %r6;");                    // + baseOut (outIdx)
        ptx.AppendLine("    mul.wide.u32 %rd8, %r12, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f9;");     // newMag
        // phase advance: dp = phase[t0+1] - phase[t0], wrapped; 0 when t0+1 >= nFramesV
        ptx.AppendLine("    mov.f32 %f12, 0f00000000;");                   // dp
        ptx.AppendLine($"    setp.ge.s32 %p1, %r13, {nFramesV};");
        ptx.AppendLine("    @%p1 bra $PV_ADD;");
        ptx.AppendLine($"    mad.lo.u32 %r14, %r13, {nFreqV}, %r3;");     // (t0+1)*nFreqV + f
        ptx.AppendLine("    add.u32 %r14, %r14, %r5;");                    // + baseB
        ptx.AppendLine("    mul.wide.u32 %rd10, %r14, 4;");
        ptx.AppendLine("    add.u64 %rd11, %rd1, %rd10;");
        ptx.AppendLine("    ld.global.nc.f32 %f10, [%rd11];");   // phase[t0+1]
        ptx.AppendLine("    mul.wide.u32 %rd12, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd13, %rd1, %rd12;");
        ptx.AppendLine("    ld.global.nc.f32 %f11, [%rd13];");   // phase[t0]
        ptx.AppendLine("    sub.rn.f32 %f12, %f10, %f11;");      // dp
        ptx.AppendLine($"    mul.rn.f32 %f13, %f12, {invTwoPi};");
        ptx.AppendLine("    cvt.rni.f32.f32 %f14, %f13;");       // round(dp/2pi)
        ptx.AppendLine($"    mul.rn.f32 %f15, %f14, {twoPi};");
        ptx.AppendLine("    sub.rn.f32 %f12, %f12, %f15;");      // dp -= 2pi*round(...)
        ptx.AppendLine("$PV_ADD:");
        ptx.AppendLine("    add.rn.f32 %f1, %f1, %f12;");        // accPhase += dp
        ptx.AppendLine("    add.u64 %rd14, %rd3, %rd8;");        // newPhase at outIdx
        ptx.AppendLine("    st.global.f32 [%rd14], %f1;");
        ptx.AppendLine("    add.u32 %r7, %r7, 1;");
        ptx.AppendLine("    bra $PV_LOOP;");
        ptx.AppendLine("$PV_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int leading, int nFramesV, int nFreqV, int outFrames, int blockThreads)
    {
        var inExtent = new DirectPtxExtent(checked(leading * nFramesV * nFreqV));
        var outExtent = new DirectPtxExtent(checked(leading * outFrames * nFreqV));
        return new DirectPtxKernelBlueprint(
            Operation: "phase-vocoder-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-ld{leading}-fr{nFramesV}-fq{nFreqV}-out{outFrames}",
            Tensors:
            [
                new("mag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inExtent, inExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("phase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inExtent, inExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("newMag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("newPhase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 32,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1024 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "newMag=lerp(m[t0],m[t1],frac); newPhase=running sum of wrapped forward phase differences",
                ["mode"] = "inference-forward-phase-vocoder",
                ["arithmetic"] = "fma lerp + cvt.rni phase wrap; tolerance-based parity, not bit-exact",
                ["scalar"] = "rate is a per-launch .param .f32",
                ["loop"] = "per (leading,freq) channel, over outFrames output frames",
                ["bounds-check"] = "none - the launch covers exactly leading*nFreqV threads",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int leading, int nFramesV, int nFreqV, int outFrames) =>
        leading >= 1 && nFramesV >= 2 && nFreqV >= 1 && outFrames >= 1 &&
        (long)leading * nFramesV * nFreqV <= (1L << 26) &&
        (long)leading * outFrames * nFreqV <= (1L << 26);

    internal static bool IsPromotedShape(int leading, int nFramesV, int nFreqV, int outFrames) => false;

    private static void Validate(int leading, int nFramesV, int nFreqV, int outFrames)
    {
        if (!IsSupportedShape(leading, nFramesV, nFreqV, outFrames))
            throw new ArgumentOutOfRangeException(nameof(nFramesV),
                "The phase-vocoder family requires leading>=1, nFramesV>=2, nFreqV>=1, outFrames>=1, " +
                "and both leading*nFramesV*nFreqV and leading*outFrames*nFreqV <= 2^26.");
    }

    private static void ValidateBlockThreads(int leading, int nFreqV, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Phase-vocoder block threads must be 128, 256, or 512.");
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
