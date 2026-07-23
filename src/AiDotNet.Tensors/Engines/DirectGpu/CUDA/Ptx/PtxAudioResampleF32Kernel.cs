using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// Polyphase Hann-windowed sinc resampling for issue #850, matching the NVRTC <c>audio_resample</c> kernel:
/// resample each row from <c>inLen</c> to <c>outLen</c> samples at ratio <c>up/down</c>. One thread owns one
/// output sample; it centers a windowed sinc at <c>srcIdx = ot*down/up</c> and accumulates
/// <c>sinc((idx-srcIdx)*cutoff) * hann(k) * input[idx]</c> over the tap window <c>[-halfWidth, halfWidth]</c>,
/// dividing by the window sum. PTX has no full-precision sin/cos, so each transcendental is argument-reduced
/// to <c>[-pi, pi]</c> (<c>x - 2*pi*round(x/2*pi)</c>) before <c>sin.approx</c>/<c>cos.approx</c> - far more
/// accurate than the fast-math intrinsics the reference warns against, but still a TOLERANCE-based parity
/// spec, not bit-exact. <c>leading</c>, <c>inLen</c>, <c>outLen</c>, <c>up</c>, <c>down</c>, and
/// <c>halfWidth</c> are baked; the launch rounds up and a single guard drops the tail lanes. Two pointers
/// reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxAudioResampleF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_audio_resample_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int Leading { get; }
    internal int InLen { get; }
    internal int OutLen { get; }
    internal int Up { get; }
    internal int Down { get; }
    internal int HalfWidth { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxAudioResampleF32Kernel(
        DirectPtxRuntime runtime, int leading, int inLen, int outLen, int up, int down, int halfWidth,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in audio-resample specialization is admitted only on SM86.");
        Validate(leading, inLen, outLen, up, down, halfWidth);
        ValidateBlockThreads(blockThreads);
        Leading = leading;
        InLen = inLen;
        OutLen = outLen;
        Up = up;
        Down = down;
        HalfWidth = halfWidth;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, leading, inLen, outLen, up, down, halfWidth, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor, leading, inLen, outLen, up, down, halfWidth, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView input, DirectPtxTensorView output)
    {
        Require(input, Blueprint.Tensors[0], nameof(input));
        Require(output, Blueprint.Tensors[1], nameof(output));

        IntPtr inputPointer = input.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &inputPointer;
        arguments[1] = &outputPointer;
        int total = Leading * OutLen;
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
        int ccMajor, int ccMinor, int leading, int inLen, int outLen, int up, int down, int halfWidth,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(leading, inLen, outLen, up, down, halfWidth);
        ValidateBlockThreads(blockThreads);
        int total = checked(leading * outLen);
        int twoHalfWidth = 2 * halfWidth;
        string cutoff = Hex(1f / (up > down ? up : down));
        string downF = Hex(down), upF = Hex(up);
        string piC = Hex((float)Math.PI), twoPi = Hex((float)(2.0 * Math.PI)), inv2Pi = Hex((float)(1.0 / (2.0 * Math.PI)));
        string piOverHalf = Hex((float)(Math.PI / halfWidth));
        string tiny = Hex(1e-12f);
        const string half = "0f3F000000", negHalf = "0fBF000000", one = "0f3F800000", zero = "0f00000000";

        var ptx = new StringBuilder(5_632);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape leading={leading} inLen={inLen} outLen={outLen} up={up} down={down} halfWidth={halfWidth} block={blockThreads} op=audio-resample");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 input_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<6>;");
        ptx.AppendLine("    .reg .b32 %r<14>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<28>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [input_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // gid
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {total};");
        ptx.AppendLine("    @%p0 bra $RS_RET;");
        ptx.AppendLine($"    rem.u32 %r3, %r2, {outLen};");              // ot
        ptx.AppendLine($"    div.u32 %r4, %r2, {outLen};");              // row
        ptx.AppendLine($"    mul.lo.u32 %r9, %r4, {inLen};");           // sBase = row*inLen
        // srcIdx = (float)ot * down / up
        ptx.AppendLine("    cvt.rn.f32.u32 %f0, %r3;");                    // otF
        ptx.AppendLine($"    mul.rn.f32 %f0, %f0, {downF};");
        ptx.AppendLine($"    div.rn.f32 %f0, %f0, {upF};");              // srcIdx
        ptx.AppendLine("    cvt.rmi.s32.f32 %r5, %f0;");                   // centre = floor(srcIdx)
        ptx.AppendLine("    mov.f32 %f1, 0f00000000;");                    // acc
        ptx.AppendLine("    mov.f32 %f2, 0f00000000;");                    // wSum
        ptx.AppendLine("    mov.u32 %r6, 0;");                             // kk = 0..2*halfWidth
        ptx.AppendLine("$RS_LOOP:");
        ptx.AppendLine($"    setp.gt.u32 %p1, %r6, {twoHalfWidth};");
        ptx.AppendLine("    @%p1 bra $RS_AFTER;");
        ptx.AppendLine($"    sub.s32 %r7, %r6, {halfWidth};");           // k = kk - halfWidth
        ptx.AppendLine("    add.s32 %r8, %r5, %r7;");                     // idx = centre + k
        ptx.AppendLine("    setp.lt.s32 %p2, %r8, 0;");
        ptx.AppendLine("    @%p2 bra $RS_SKIP;");
        ptx.AppendLine($"    setp.ge.s32 %p3, %r8, {inLen};");
        ptx.AppendLine("    @%p3 bra $RS_SKIP;");
        // t = (idx - srcIdx) * cutoff
        ptx.AppendLine("    cvt.rn.f32.s32 %f3, %r8;");                    // idxF
        ptx.AppendLine("    sub.rn.f32 %f4, %f3, %f0;");
        ptx.AppendLine($"    mul.rn.f32 %f4, %f4, {cutoff};");           // t
        ptx.AppendLine($"    mul.rn.f32 %f5, %f4, {piC};");             // sincArg = pi*t
        ptx.AppendLine("    abs.f32 %f6, %f4;");
        ptx.AppendLine($"    setp.lt.f32 %p4, %f6, {tiny};");           // |t| < 1e-12
        // sin(sincArg) with range reduction to [-pi, pi]
        ptx.AppendLine($"    mul.rn.f32 %f7, %f5, {inv2Pi};");
        ptx.AppendLine("    cvt.rni.f32.f32 %f7, %f7;");                  // round(x/2pi)
        ptx.AppendLine($"    mul.rn.f32 %f8, %f7, {twoPi};");
        ptx.AppendLine("    sub.rn.f32 %f8, %f5, %f8;");                  // reduced sinc arg
        ptx.AppendLine("    sin.approx.f32 %f9, %f8;");
        ptx.AppendLine("    div.rn.f32 %f10, %f9, %f5;");                 // sin(pi t)/(pi t)
        ptx.AppendLine($"    selp.f32 %f11, {one}, %f10, %p4;");        // sinc
        // hann = 0.5 - 0.5*cos(pi*kk/halfWidth), range reduced
        ptx.AppendLine("    cvt.rn.f32.u32 %f12, %r6;");                  // kkF
        ptx.AppendLine($"    mul.rn.f32 %f13, %f12, {piOverHalf};");    // hannArg
        ptx.AppendLine($"    mul.rn.f32 %f14, %f13, {inv2Pi};");
        ptx.AppendLine("    cvt.rni.f32.f32 %f14, %f14;");
        ptx.AppendLine($"    mul.rn.f32 %f15, %f14, {twoPi};");
        ptx.AppendLine("    sub.rn.f32 %f15, %f13, %f15;");              // reduced hann arg
        ptx.AppendLine("    cos.approx.f32 %f16, %f15;");
        ptx.AppendLine($"    fma.rn.f32 %f17, {negHalf}, %f16, {half};"); // 0.5 - 0.5*cos
        ptx.AppendLine("    mul.rn.f32 %f18, %f11, %f17;");               // w = sinc*hann
        // acc += w * input[sBase + idx]; wSum += w
        ptx.AppendLine("    add.u32 %r10, %r9, %r8;");
        ptx.AppendLine("    mul.wide.u32 %rd2, %r10, 4;");
        ptx.AppendLine("    add.u64 %rd3, %rd0, %rd2;");
        ptx.AppendLine("    ld.global.nc.f32 %f19, [%rd3];");
        ptx.AppendLine("    fma.rn.f32 %f1, %f18, %f19, %f1;");
        ptx.AppendLine("    add.rn.f32 %f2, %f2, %f18;");
        ptx.AppendLine("$RS_SKIP:");
        ptx.AppendLine("    add.u32 %r6, %r6, 1;");
        ptx.AppendLine("    bra $RS_LOOP;");
        ptx.AppendLine("$RS_AFTER:");
        ptx.AppendLine($"    setp.gt.f32 %p5, %f2, {zero};");
        ptx.AppendLine("    div.rn.f32 %f20, %f1, %f2;");
        ptx.AppendLine($"    selp.f32 %f20, %f20, {zero}, %p5;");       // wSum>0 ? acc/wSum : 0
        ptx.AppendLine("    mul.wide.u32 %rd4, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    st.global.f32 [%rd5], %f20;");
        ptx.AppendLine("$RS_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int leading, int inLen, int outLen, int up, int down, int halfWidth, int blockThreads)
    {
        var inExtent = new DirectPtxExtent(checked(leading * inLen));
        var outExtent = new DirectPtxExtent(checked(leading * outLen));
        return new DirectPtxKernelBlueprint(
            Operation: "audio-resample-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-ld{leading}-in{inLen}-out{outLen}-up{up}-dn{down}-hw{halfWidth}",
            Tensors:
            [
                new("input", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    inExtent, inExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 40,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1024 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[row,ot] = sum_k sinc((idx-srcIdx)*cutoff)*hann(k)*input[row,idx] / sum_k w; srcIdx=ot*down/up",
                ["mode"] = "inference-forward-audio-resample",
                ["arithmetic"] = "range-reduced sin.approx/cos.approx (reduced to [-pi,pi]) + fma; tolerance-based parity",
                ["loop"] = "per output sample, over the [-halfWidth, halfWidth] tap window with a bounds skip",
                ["bounds-check"] = "single guard drops lanes past leading*outLen; per-tap idx-range skip inside the loop",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int leading, int inLen, int outLen, int up, int down, int halfWidth) =>
        leading >= 1 && inLen >= 1 && outLen >= 1 && up >= 1 && down >= 1 && halfWidth >= 1 && halfWidth <= 1024 &&
        (long)leading * inLen <= (1L << 26) && (long)leading * outLen <= (1L << 26);

    internal static bool IsPromotedShape(int leading, int inLen, int outLen, int up, int down, int halfWidth) => false;

    private static void Validate(int leading, int inLen, int outLen, int up, int down, int halfWidth)
    {
        if (!IsSupportedShape(leading, inLen, outLen, up, down, halfWidth))
            throw new ArgumentOutOfRangeException(nameof(halfWidth),
                "The audio-resample family requires leading>=1, inLen>=1, outLen>=1, up>=1, down>=1, " +
                "1<=halfWidth<=1024, and leading*inLen, leading*outLen <= 2^26.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "Audio-resample block threads must be 128, 256, or 512.");
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
