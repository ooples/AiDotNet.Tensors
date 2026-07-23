using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Direct windowed STFT magnitude/phase for issue #850, matching the NVRTC <c>parity210_stft_mag_phase</c>
/// kernel: for each output bin <c>(batch b, frequency k, frame)</c> evaluate a windowed length-<c>nFft</c>
/// DFT of the padded signal and emit its magnitude and phase. One thread owns one output bin, loops over
/// the <c>nFft</c> window taps accumulating <c>padded[b*Lp + frame*hop + i] * window[i]</c> against
/// <c>cos</c>/<c>sin</c> of <c>-2*pi*k*i/nFft</c> with <c>fma.rn</c>, then writes
/// <c>mag = sqrt(re^2 + im^2)</c> and <c>phase = atan2(im, re)</c>. The twiddle uses
/// <c>cos.approx</c>/<c>sin.approx</c> and the phase uses a minimax atan2, so the parity spec is
/// TOLERANCE-based, not bit-exact. <c>batch</c>, <c>Lp</c>, <c>nFft</c>, <c>hop</c>, <c>numFrames</c>, and
/// <c>numFreqs</c> are baked; the launch covers exactly <c>batch * numFreqs * numFrames</c> threads with no
/// bounds guard. Four pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxStftMagPhaseF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_stft_mag_phase_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Batch { get; }
    internal int Lp { get; }
    internal int NFft { get; }
    internal int Hop { get; }
    internal int NumFrames { get; }
    internal int NumFreqs { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxStftMagPhaseF32Kernel(
        DirectPtxRuntime runtime, int batch, int lp, int nFft, int hop, int numFrames, int numFreqs,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in STFT mag/phase specialization is admitted only on SM86.");
        Validate(batch, lp, nFft, hop, numFrames, numFreqs);
        ValidateBlockThreads(batch, numFrames, numFreqs, blockThreads);
        Batch = batch;
        Lp = lp;
        NFft = nFft;
        Hop = hop;
        NumFrames = numFrames;
        NumFreqs = numFreqs;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, batch, lp, nFft, numFrames, numFreqs, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            batch, lp, nFft, hop, numFrames, numFreqs, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView padded, DirectPtxTensorView window,
        DirectPtxTensorView mag, DirectPtxTensorView phase)
    {
        Require(padded, Blueprint.Tensors[0], nameof(padded));
        Require(window, Blueprint.Tensors[1], nameof(window));
        Require(mag, Blueprint.Tensors[2], nameof(mag));
        Require(phase, Blueprint.Tensors[3], nameof(phase));

        IntPtr paddedPointer = padded.Pointer, windowPointer = window.Pointer;
        IntPtr magPointer = mag.Pointer, phasePointer = phase.Pointer;
        void** arguments = stackalloc void*[4];
        arguments[0] = &paddedPointer;
        arguments[1] = &windowPointer;
        arguments[2] = &magPointer;
        arguments[3] = &phasePointer;
        int total = Batch * NumFreqs * NumFrames;
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
        int ccMajor, int ccMinor, int batch, int lp, int nFft, int hop, int numFrames, int numFreqs,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(batch, lp, nFft, hop, numFrames, numFreqs);
        ValidateBlockThreads(batch, numFrames, numFreqs, blockThreads);
        string negTwoPiOverN = Hex((float)(-2.0 * Math.PI / nFft));
        int magStride = checked(numFreqs * numFrames);
        int total = checked(batch * numFreqs * numFrames);
        string c0 = Hex(0.9998660f), c1 = Hex(-0.3302995f), c2 = Hex(0.1801410f),
               c3 = Hex(-0.0851330f), c4 = Hex(0.0208351f);
        string pi = Hex((float)Math.PI), halfPi = Hex((float)(Math.PI / 2.0)), tiny = Hex(1e-20f);
        const string negOne = "0fBF800000";

        var ptx = new StringBuilder(5_120);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape batch={batch} Lp={lp} nFft={nFft} hop={hop} numFrames={numFrames} numFreqs={numFreqs} block={blockThreads} op=stft-mag-phase");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 padded_ptr,");
        ptx.AppendLine("    .param .u64 window_ptr,");
        ptx.AppendLine("    .param .u64 mag_ptr,");
        ptx.AppendLine("    .param .u64 phase_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<16>;");
        ptx.AppendLine("    .reg .b64 %rd<16>;");
        ptx.AppendLine("    .reg .f32 %f<28>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [padded_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [window_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [mag_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd3, [phase_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p5, %r2, {total};");
        ptx.AppendLine("    @%p5 bra $STFT_RET;");
        ptx.AppendLine($"    rem.u32 %r3, %r2, {numFrames};");             // frame
        ptx.AppendLine($"    div.u32 %r4, %r2, {numFrames};");             // tmp
        ptx.AppendLine($"    rem.u32 %r5, %r4, {numFreqs};");              // k
        ptx.AppendLine($"    div.u32 %r6, %r4, {numFreqs};");              // b
        ptx.AppendLine($"    mul.lo.u32 %r7, %r3, {hop};");               // start = frame*hop
        ptx.AppendLine($"    mul.lo.u32 %r8, %r6, {lp};");                // inOff = b*Lp
        ptx.AppendLine("    add.u32 %r9, %r8, %r7;");                      // inStart = inOff + start
        ptx.AppendLine("    mul.wide.u32 %rd4, %r9, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");                   // paddedBase (at inStart)
        ptx.AppendLine("    cvt.rn.f32.u32 %f2, %r5;");                    // kF
        ptx.AppendLine($"    mul.rn.f32 %f2, %f2, {negTwoPiOverN};");     // bk = -2pi*k/nFft
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                    // re
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                    // im
        ptx.AppendLine("    mov.u32 %r10, 0;");                            // i
        ptx.AppendLine("$STFT_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p4, %r10, {nFft};");
        ptx.AppendLine("    @%p4 bra $STFT_AFTER;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd5, %rd6;");   // &padded[inStart+i]
        ptx.AppendLine("    add.u64 %rd8, %rd1, %rd6;");   // &window[i]
        ptx.AppendLine("    ld.global.nc.f32 %f3, [%rd7];");   // pv
        ptx.AppendLine("    ld.global.nc.f32 %f4, [%rd8];");   // wv
        ptx.AppendLine("    mul.rn.f32 %f5, %f3, %f4;");       // x = pv*wv
        ptx.AppendLine("    cvt.rn.f32.u32 %f6, %r10;");       // iF
        ptx.AppendLine("    mul.rn.f32 %f7, %f2, %f6;");       // a = bk*iF
        ptx.AppendLine("    cos.approx.f32 %f8, %f7;");
        ptx.AppendLine("    sin.approx.f32 %f9, %f7;");
        ptx.AppendLine("    fma.rn.f32 %f0, %f5, %f8, %f0;"); // re += x*cos(a)
        ptx.AppendLine("    fma.rn.f32 %f1, %f5, %f9, %f1;"); // im += x*sin(a)
        ptx.AppendLine("    add.u32 %r10, %r10, 1;");
        ptx.AppendLine("    bra $STFT_LOOP;");
        ptx.AppendLine("$STFT_AFTER:");
        // magnitude = sqrt(re^2 + im^2)
        ptx.AppendLine("    mul.rn.f32 %f10, %f0, %f0;");
        ptx.AppendLine("    fma.rn.f32 %f10, %f1, %f1, %f10;");
        ptx.AppendLine("    sqrt.rn.f32 %f11, %f10;");
        // outOff = b*numFreqs*numFrames + k*numFrames + frame
        ptx.AppendLine($"    mad.lo.u32 %r12, %r5, {numFrames}, %r3;");
        ptx.AppendLine($"    mad.lo.u32 %r13, %r6, {magStride}, %r12;");
        ptx.AppendLine("    mul.wide.u32 %rd9, %r13, 4;");
        ptx.AppendLine("    add.u64 %rd10, %rd2, %rd9;");   // magAddr
        ptx.AppendLine("    st.global.f32 [%rd10], %f11;");
        // phase = atan2(im, re) : minimax atan on [0,1] + quadrant folding (x=re=%f0, y=im=%f1)
        ptx.AppendLine("    abs.f32 %f12, %f0;");               // ax
        ptx.AppendLine("    abs.f32 %f13, %f1;");               // ay
        ptx.AppendLine("    max.f32 %f14, %f12, %f13;");        // mx
        ptx.AppendLine("    min.f32 %f15, %f12, %f13;");        // mn
        ptx.AppendLine($"    setp.lt.f32 %p0, %f14, {tiny};");
        ptx.AppendLine("    rcp.approx.f32 %f16, %f14;");
        ptx.AppendLine("    mul.rn.f32 %f17, %f15, %f16;");     // a = mn/mx
        ptx.AppendLine("    mul.rn.f32 %f18, %f17, %f17;");     // t = a^2
        ptx.AppendLine($"    mov.f32 %f19, {c4};");
        ptx.AppendLine($"    fma.rn.f32 %f19, %f19, %f18, {c3};");
        ptx.AppendLine($"    fma.rn.f32 %f19, %f19, %f18, {c2};");
        ptx.AppendLine($"    fma.rn.f32 %f19, %f19, %f18, {c1};");
        ptx.AppendLine($"    fma.rn.f32 %f19, %f19, %f18, {c0};");
        ptx.AppendLine("    mul.rn.f32 %f20, %f17, %f19;");     // r = atan(a)
        ptx.AppendLine($"    mul.rn.f32 %f21, %f20, {negOne};");
        ptx.AppendLine($"    add.rn.f32 %f21, %f21, {halfPi};");
        ptx.AppendLine("    setp.gt.f32 %p1, %f13, %f12;");     // ay > ax
        ptx.AppendLine("    selp.f32 %f22, %f21, %f20, %p1;");
        ptx.AppendLine($"    mul.rn.f32 %f23, %f22, {negOne};");
        ptx.AppendLine($"    add.rn.f32 %f23, %f23, {pi};");
        ptx.AppendLine("    setp.lt.f32 %p2, %f0, 0f00000000;"); // re < 0
        ptx.AppendLine("    selp.f32 %f24, %f23, %f22, %p2;");
        ptx.AppendLine("    neg.f32 %f25, %f24;");
        ptx.AppendLine("    setp.lt.f32 %p3, %f1, 0f00000000;"); // im < 0
        ptx.AppendLine("    selp.f32 %f24, %f25, %f24, %p3;");
        ptx.AppendLine("    selp.f32 %f24, 0f00000000, %f24, %p0;");
        ptx.AppendLine("    add.u64 %rd11, %rd3, %rd9;");   // phaseAddr
        ptx.AppendLine("    st.global.f32 [%rd11], %f24;");
        ptx.AppendLine("$STFT_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int batch, int lp, int nFft, int numFrames, int numFreqs, int blockThreads)
    {
        var paddedExtent = new DirectPtxExtent(checked(batch * lp));
        var windowExtent = new DirectPtxExtent(nFft);
        var outExtent = new DirectPtxExtent(checked(batch * numFreqs * numFrames));
        return new DirectPtxKernelBlueprint(
            Operation: "stft-mag-phase-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-bt{batch}-lp{lp}-fft{nFft}-fr{numFrames}-nf{numFreqs}",
            Tensors:
            [
                new("padded", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    paddedExtent, paddedExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("window", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    windowExtent, windowExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("mag", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("phase", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1024 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "windowed length-nFft DFT per (b,k,frame); mag=sqrt(re^2+im^2), phase=atan2(im,re)",
                ["mode"] = "inference-forward-stft-mag-phase",
                ["arithmetic"] = "cos.approx/sin.approx twiddle, fma.rn accumulation, minimax atan2; tolerance-based parity",
                ["loop"] = "per output bin, over nFft window taps",
                ["bounds-check"] = "none - the launch covers exactly batch*numFreqs*numFrames threads",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int batch, int lp, int nFft, int hop, int numFrames, int numFreqs) =>
        batch >= 1 && nFft >= 4 && hop >= 1 && numFrames >= 1 && numFreqs >= 2 && numFreqs <= nFft / 2 + 1 &&
        lp >= (numFrames - 1) * hop + nFft &&
        (long)batch * lp <= (1L << 26) && (long)batch * numFreqs * numFrames <= (1L << 24);

    internal static bool IsPromotedShape(int batch, int lp, int nFft, int hop, int numFrames, int numFreqs) => false;

    private static void Validate(int batch, int lp, int nFft, int hop, int numFrames, int numFreqs)
    {
        if (!IsSupportedShape(batch, lp, nFft, hop, numFrames, numFreqs))
            throw new ArgumentOutOfRangeException(nameof(nFft),
                "The STFT mag/phase family requires batch>=1, nFft>=4, hop>=1, numFrames>=1, " +
                "2<=numFreqs<=nFft/2+1, Lp>=(numFrames-1)*hop+nFft, batch*Lp<=2^26, and " +
                "batch*numFreqs*numFrames<=2^24.");
    }

    private static void ValidateBlockThreads(int batch, int numFrames, int numFreqs, int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "STFT mag/phase block threads must be 128, 256, or 512.");
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
