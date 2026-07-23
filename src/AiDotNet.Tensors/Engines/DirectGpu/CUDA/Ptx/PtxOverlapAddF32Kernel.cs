using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// ISTFT overlap-add for issue #850, matching the NVRTC <c>overlap_add</c> kernel: reconstruct a waveform
/// by windowing each of <c>numFrames</c> length-<c>nFft</c> synthesis frames and summing them at their hop
/// positions. One thread owns one output sample <c>idx</c> and loops over the frames, accumulating
/// <c>frames[frame*nFft + localIdx] * window[localIdx]</c> for every frame whose support
/// (<c>localIdx = idx - frame*hopLength</c> in <c>[0, nFft)</c>) covers the sample. The accumulation uses
/// <c>fma.rn</c>, matching the reference's fused <c>sum += frame*window</c>, so the parity spec is
/// tolerance-based against the fp64 oracle. <c>numFrames</c>, <c>nFft</c>, <c>hopLength</c>, and
/// <c>outputLength</c> are baked; the launch rounds up and a single guard drops lanes past the output
/// length. Three pointers reach the launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxOverlapAddF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_overlap_add_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NumFrames { get; }
    internal int NFft { get; }
    internal int HopLength { get; }
    internal int OutputLength { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxOverlapAddF32Kernel(
        DirectPtxRuntime runtime, int numFrames, int nFft, int hopLength, int outputLength,
        int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in ISTFT overlap-add specialization is admitted only on SM86.");
        Validate(numFrames, nFft, hopLength, outputLength);
        ValidateBlockThreads(blockThreads);
        NumFrames = numFrames;
        NFft = nFft;
        HopLength = hopLength;
        OutputLength = outputLength;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, numFrames, nFft, outputLength, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            numFrames, nFft, hopLength, outputLength, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(
        DirectPtxTensorView frames, DirectPtxTensorView window, DirectPtxTensorView output)
    {
        Require(frames, Blueprint.Tensors[0], nameof(frames));
        Require(window, Blueprint.Tensors[1], nameof(window));
        Require(output, Blueprint.Tensors[2], nameof(output));

        IntPtr framesPointer = frames.Pointer, windowPointer = window.Pointer, outputPointer = output.Pointer;
        void** arguments = stackalloc void*[3];
        arguments[0] = &framesPointer;
        arguments[1] = &windowPointer;
        arguments[2] = &outputPointer;
        uint grid = (uint)((OutputLength + BlockThreads - 1) / BlockThreads);
        _module.Launch(
            _function,
            grid, 1, 1,
            checked((uint)BlockThreads), 1, 1,
            0,
            arguments);
    }

    public void Dispose() => _module.Dispose();

    internal static string EmitPtx(
        int ccMajor, int ccMinor, int numFrames, int nFft, int hopLength, int outputLength,
        int blockThreads = DefaultBlockThreads)
    {
        Validate(numFrames, nFft, hopLength, outputLength);
        ValidateBlockThreads(blockThreads);

        var ptx = new StringBuilder(3_072);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape numFrames={numFrames} nFft={nFft} hop={hopLength} outLen={outputLength} block={blockThreads} op=overlap-add");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 frames_ptr,");
        ptx.AppendLine("    .param .u64 window_ptr,");
        ptx.AppendLine("    .param .u64 output_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<10>;");
        ptx.AppendLine("    .reg .b64 %rd<12>;");
        ptx.AppendLine("    .reg .f32 %f<4>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [frames_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [window_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd2, [output_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {outputLength};");
        ptx.AppendLine("    @%p0 bra $OLA_RET;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                    // sum
        ptx.AppendLine("    mov.u32 %r3, 0;");                             // frame
        ptx.AppendLine("$OLA_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r3, {numFrames};");
        ptx.AppendLine("    @%p1 bra $OLA_WRITE;");
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {hopLength};");         // frameStart
        ptx.AppendLine("    sub.s32 %r5, %r2, %r4;");                      // localIdx = idx - frameStart
        ptx.AppendLine("    setp.lt.s32 %p2, %r5, 0;");
        ptx.AppendLine("    @%p2 bra $OLA_SKIP;");
        ptx.AppendLine($"    setp.ge.s32 %p3, %r5, {nFft};");
        ptx.AppendLine("    @%p3 bra $OLA_SKIP;");
        ptx.AppendLine($"    mad.lo.u32 %r6, %r3, {nFft}, %r5;");         // frame*nFft + localIdx
        ptx.AppendLine("    mul.wide.u32 %rd4, %r6, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd0, %rd4;");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd1, %rd6;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");   // frames sample
        ptx.AppendLine("    ld.global.nc.f32 %f2, [%rd7];");   // window
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f2, %f0;");  // sum += frame*window (matches fused reference)
        ptx.AppendLine("$OLA_SKIP:");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra $OLA_LOOP;");
        ptx.AppendLine("$OLA_WRITE:");
        ptx.AppendLine("    mul.wide.u32 %rd8, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd9, %rd2, %rd8;");
        ptx.AppendLine("    st.global.f32 [%rd9], %f0;");
        ptx.AppendLine("$OLA_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int numFrames, int nFft, int outputLength, int blockThreads)
    {
        var framesExtent = new DirectPtxExtent(checked(numFrames * nFft));
        var windowExtent = new DirectPtxExtent(nFft);
        var outExtent = new DirectPtxExtent(outputLength);
        return new DirectPtxKernelBlueprint(
            Operation: "overlap-add-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-nf{numFrames}-fft{nFft}-out{outputLength}",
            Tensors:
            [
                new("frames", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    framesExtent, framesExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("window", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    windowExtent, windowExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact),
                new("output", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 24,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "output[i] = sum_frame frames[frame*nFft + (i-frame*hop)] * window[i-frame*hop] over covering frames",
                ["mode"] = "inference-forward-istft-overlap-add",
                ["arithmetic"] = "fma.rn accumulation matching the reference's fused sum; tolerance-based parity",
                ["loop"] = "per output sample, over numFrames frames with support guard localIdx in [0,nFft)",
                ["bounds-check"] = "single guard drops lanes past outputLength; frame support guards inside the loop",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int numFrames, int nFft, int hopLength, int outputLength) =>
        numFrames >= 1 && nFft >= 2 && hopLength >= 1 && hopLength <= nFft && outputLength >= nFft &&
        (long)numFrames * nFft <= (1L << 26) && outputLength <= (1 << 24);

    internal static bool IsPromotedShape(int numFrames, int nFft, int hopLength, int outputLength) => false;

    private static void Validate(int numFrames, int nFft, int hopLength, int outputLength)
    {
        if (!IsSupportedShape(numFrames, nFft, hopLength, outputLength))
            throw new ArgumentOutOfRangeException(nameof(numFrames),
                "The ISTFT overlap-add family requires numFrames>=1, nFft>=2, 1<=hopLength<=nFft, " +
                "outputLength in [nFft, 2^24], and numFrames*nFft<=2^26.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "ISTFT overlap-add block threads must be 128, 256, or 512.");
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
