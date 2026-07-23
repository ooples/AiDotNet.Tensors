using System;
using System.Collections.Generic;
using System.Text;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

/// <summary>
/// ISTFT window normalization for issue #850, matching the NVRTC <c>window_sum_squares</c> kernel: build
/// the per-sample sum of squared window values so the overlap-add reconstruction can be divided by it. One
/// thread owns one output sample <c>idx</c> and loops over the <c>numFrames = (outputLength - nFft)/hop + 1</c>
/// frames, accumulating <c>window[idx - frame*hop]^2</c> for every frame whose support
/// (<c>localIdx in [0, nFft)</c>) covers the sample. The accumulation uses <c>fma.rn</c>, matching the
/// reference's fused <c>sum += w*w</c>, so the parity spec is tolerance-based against the fp64 oracle.
/// <c>nFft</c>, <c>hopLength</c>, and <c>outputLength</c> are baked (<c>numFrames</c> derived from them);
/// the launch rounds up and a single guard drops lanes past the output length. Two pointers reach the
/// launch ABI.
///
/// The specialization stays disabled by default and fails closed until three clean promotion runs clear
/// the release gate.
/// </summary>
internal sealed class PtxWindowSumSquaresF32Kernel : IDisposable
{
    internal const int DefaultBlockThreads = 256;
    internal const string EntryPoint = "aidotnet_window_sum_squares_f32";

    private readonly DirectPtxModule _module;
    private readonly IntPtr _function;

    internal int NFft { get; }
    internal int HopLength { get; }
    internal int OutputLength { get; }
    internal int NumFrames { get; }
    internal int BlockThreads { get; }
    internal string Ptx { get; }
    internal DirectPtxKernelBlueprint Blueprint { get; }
    internal DirectPtxKernelAudit Audit { get; }

    internal PtxWindowSumSquaresF32Kernel(
        DirectPtxRuntime runtime, int nFft, int hopLength, int outputLength, int blockThreads = DefaultBlockThreads)
    {
        PtxCompat.ThrowIfNull(runtime, nameof(runtime));
        if (!DirectPtxArchitecture.HasValidatedComplexUnary(
            runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor))
            throw new PlatformNotSupportedException(
                "The checked-in ISTFT window-sum-squares specialization is admitted only on SM86.");
        Validate(nFft, hopLength, outputLength);
        ValidateBlockThreads(blockThreads);
        NFft = nFft;
        HopLength = hopLength;
        OutputLength = outputLength;
        NumFrames = (outputLength - nFft) / hopLength + 1;
        BlockThreads = blockThreads;
        Blueprint = CreateBlueprint(runtime.ArchitectureFamily, nFft, outputLength, blockThreads);
        Ptx = EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor,
            nFft, hopLength, outputLength, blockThreads);
        _module = runtime.LoadModule(Ptx);
        _function = _module.GetFunction(EntryPoint, out DirectPtxFunctionInfo info);
        int activeBlocks = _module.GetActiveBlocksPerMultiprocessor(_function, BlockThreads);
        Blueprint.ResourceBudget.Validate(EntryPoint, info, BlockThreads, activeBlocks);
        Audit = DirectPtxKernelAudit.Create(
            Blueprint, runtime.DeviceFingerprint, Ptx, info, BlockThreads, activeBlocks, _module.JitInfoLog);
    }

    internal unsafe void Launch(DirectPtxTensorView winSqSum, DirectPtxTensorView window)
    {
        Require(winSqSum, Blueprint.Tensors[0], nameof(winSqSum));
        Require(window, Blueprint.Tensors[1], nameof(window));

        IntPtr winSqSumPointer = winSqSum.Pointer, windowPointer = window.Pointer;
        void** arguments = stackalloc void*[2];
        arguments[0] = &winSqSumPointer;
        arguments[1] = &windowPointer;
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
        int ccMajor, int ccMinor, int nFft, int hopLength, int outputLength, int blockThreads = DefaultBlockThreads)
    {
        Validate(nFft, hopLength, outputLength);
        ValidateBlockThreads(blockThreads);
        int numFrames = (outputLength - nFft) / hopLength + 1;

        var ptx = new StringBuilder(2_816);
        ptx.AppendLine(".version 7.1");
        ptx.AppendLine($".target sm_{ccMajor}{ccMinor}");
        ptx.AppendLine(".address_size 64");
        ptx.AppendLine($"// exact-shape nFft={nFft} hop={hopLength} outLen={outputLength} numFrames={numFrames} block={blockThreads} op=window-sum-squares");
        ptx.AppendLine();
        ptx.AppendLine($".visible .entry {EntryPoint}(");
        ptx.AppendLine("    .param .u64 winsqsum_ptr,");
        ptx.AppendLine("    .param .u64 window_ptr");
        ptx.AppendLine(")");
        ptx.AppendLine($".maxntid {blockThreads}, 1, 1");
        ptx.AppendLine("{");
        ptx.AppendLine("    .reg .pred %p<4>;");
        ptx.AppendLine("    .reg .b32 %r<8>;");
        ptx.AppendLine("    .reg .b64 %rd<10>;");
        ptx.AppendLine("    .reg .f32 %f<3>;");
        ptx.AppendLine("    ld.param.u64 %rd0, [winsqsum_ptr];");
        ptx.AppendLine("    ld.param.u64 %rd1, [window_ptr];");
        ptx.AppendLine("    mov.u32 %r0, %tid.x;");
        ptx.AppendLine("    mov.u32 %r1, %ctaid.x;");
        ptx.AppendLine($"    mad.lo.u32 %r2, %r1, {blockThreads}, %r0;");   // idx
        ptx.AppendLine($"    setp.ge.u32 %p0, %r2, {outputLength};");
        ptx.AppendLine("    @%p0 bra $WSS_RET;");
        ptx.AppendLine("    mov.f32 %f0, 0f00000000;");                    // sum
        ptx.AppendLine("    mov.u32 %r3, 0;");                             // frame
        ptx.AppendLine("$WSS_LOOP:");
        ptx.AppendLine($"    setp.ge.u32 %p1, %r3, {numFrames};");
        ptx.AppendLine("    @%p1 bra $WSS_WRITE;");
        ptx.AppendLine($"    mul.lo.u32 %r4, %r3, {hopLength};");         // frameStart
        ptx.AppendLine("    sub.s32 %r5, %r2, %r4;");                      // localIdx
        ptx.AppendLine("    setp.lt.s32 %p2, %r5, 0;");
        ptx.AppendLine("    @%p2 bra $WSS_SKIP;");
        ptx.AppendLine($"    setp.ge.s32 %p3, %r5, {nFft};");
        ptx.AppendLine("    @%p3 bra $WSS_SKIP;");
        ptx.AppendLine("    mul.wide.u32 %rd4, %r5, 4;");
        ptx.AppendLine("    add.u64 %rd5, %rd1, %rd4;");
        ptx.AppendLine("    ld.global.nc.f32 %f1, [%rd5];");   // window[localIdx]
        ptx.AppendLine("    fma.rn.f32 %f0, %f1, %f1, %f0;");  // sum += w*w (matches fused reference)
        ptx.AppendLine("$WSS_SKIP:");
        ptx.AppendLine("    add.u32 %r3, %r3, 1;");
        ptx.AppendLine("    bra $WSS_LOOP;");
        ptx.AppendLine("$WSS_WRITE:");
        ptx.AppendLine("    mul.wide.u32 %rd6, %r2, 4;");
        ptx.AppendLine("    add.u64 %rd7, %rd0, %rd6;");
        ptx.AppendLine("    st.global.f32 [%rd7], %f0;");
        ptx.AppendLine("$WSS_RET:");
        ptx.AppendLine("    ret;");
        ptx.AppendLine("}");
        return ptx.ToString();
    }

    private static DirectPtxKernelBlueprint CreateBlueprint(
        DirectPtxArchitectureFamily architecture, int nFft, int outputLength, int blockThreads)
    {
        var outExtent = new DirectPtxExtent(outputLength);
        var windowExtent = new DirectPtxExtent(nFft);
        return new DirectPtxKernelBlueprint(
            Operation: "window-sum-squares-f32",
            Version: 1,
            Architecture: architecture,
            Variant: $"b{blockThreads}-fft{nFft}-out{outputLength}",
            Tensors:
            [
                new("winSqSum", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    outExtent, outExtent, 16, DirectPtxTensorAccess.Write, DirectPtxExtentMode.Exact),
                new("window", DirectPtxPhysicalType.Float32, DirectPtxPhysicalLayout.Vector,
                    windowExtent, windowExtent, 16, DirectPtxTensorAccess.Read, DirectPtxExtentMode.Exact)
            ],
            ResourceBudget: new DirectPtxResourceBudget(
                MaxRegistersPerThread: 20,
                MaxStaticSharedBytes: 0,
                MaxLocalBytesPerThread: 0,
                MinBlocksPerMultiprocessor: 1536 / blockThreads),
            Semantics: new Dictionary<string, string>(StringComparer.Ordinal)
            {
                ["formula"] = "winSqSum[i] = sum_frame window[i-frame*hop]^2 over covering frames",
                ["mode"] = "inference-forward-istft-window-sum-squares",
                ["arithmetic"] = "fma.rn accumulation matching the reference's fused sum; tolerance-based parity",
                ["loop"] = "per output sample, over derived numFrames with support guard localIdx in [0,nFft)",
                ["num-frames"] = "(outputLength - nFft)/hopLength + 1 (baked)",
                ["bounds-check"] = "single guard drops lanes past outputLength; frame support guards inside the loop",
                ["global-intermediates"] = "none",
                ["temporary-device-allocation"] = "none",
                ["byte-offset"] = "zero-entire-allocation-view",
                ["padding"] = "none-logical-equals-physical"
            });
    }

    internal static bool IsSupportedShape(int nFft, int hopLength, int outputLength) =>
        nFft >= 2 && hopLength >= 1 && hopLength <= nFft && outputLength >= nFft && outputLength <= (1 << 24) &&
        (outputLength - nFft) % hopLength == 0;

    internal static bool IsPromotedShape(int nFft, int hopLength, int outputLength) => false;

    private static void Validate(int nFft, int hopLength, int outputLength)
    {
        if (!IsSupportedShape(nFft, hopLength, outputLength))
            throw new ArgumentOutOfRangeException(nameof(nFft),
                "The ISTFT window-sum-squares family requires nFft>=2, 1<=hopLength<=nFft, outputLength in " +
                "[nFft, 2^24], and (outputLength - nFft) divisible by hopLength.");
    }

    private static void ValidateBlockThreads(int blockThreads)
    {
        if (blockThreads is not (128 or 256 or 512))
            throw new ArgumentOutOfRangeException(nameof(blockThreads),
                "ISTFT window-sum-squares block threads must be 128, 256, or 512.");
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
