using System;
using System.Buffers;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
#if NET5_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using static AiDotNet.Tensors.Compatibility.MethodImplHelper;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>The conv passes <see cref="DirectConvAvx2"/> may take; the rest keep the im2col routes.</summary>
[Flags]
internal enum DirectConvPasses
{
    None = 0,
    Forward = 1,
    BackwardInput = 2,
    BackwardInputStrided = 4,
    BackwardKernel = 8,
    All = Forward | BackwardInput | BackwardInputStrided | BackwardKernel,
}

/// <summary>
/// Direct float32 convolution on an 8-channel-blocked layout, after oneDNN's jit:avx2 forward and backward-weights
/// kernels (the implementations PyTorch's CPU conv dispatches to). No im2col: the input is repacked once into a
/// zero-padded [N][C/8][H+2p][W+2p][8] buffer, so every tap is an in-bounds load, and each output tile is held in
/// registers for the whole reduction. On ResNet-sized small planes the im2col route spent as long building and
/// scattering its column matrix as in the GEMM; this removes that traffic.
/// </summary>
internal static partial class DirectConvAvx2
{
    private const int Block = 8;

    // Forward: 3 output positions x 4 output-channel blocks = 12 accumulators, plus 3 broadcasts and 1 weight
    // register - all 16 ymm registers, as in oneDNN's avx2 forward kernel.
    private const int PositionUnroll = 3;
    private const int OutputBlocksPerTile = 4;

    // Enough tasks to keep the pool busy without splitting a tile's weight stream into slivers.
    private const int TargetForwardTasks = 128;

    // Backward weights needs one (output block, input block) pair per task; fewer pairs than this starve the pool.
    private const int MinBackwardKernelTasks = 16;

    // Weight-gradient tasks to aim for when splitting the batch across partial gradients.
    private const int TargetBackwardKernelTasks = 128;

    // Most floats the backward-kernel batch split may hold in per-split partials (16 MB). Shape-only, so the split
    // (and the summation order) never depends on the thread budget.
    private const long MaxBackwardKernelPartialFloats = 4L * 1024 * 1024;

    // Set only through SuppressOnThisThread(): tests that pin the choice between the im2col routes run their shapes with
    // the direct route out of the way, without a process-wide switch that would reach convolutions on other threads.
    [ThreadStatic]
    private static bool t_suppressed;

    /// <summary>Keeps this thread's convolutions off the direct kernels until disposed (restores the previous state).</summary>
    internal static IDisposable SuppressOnThisThread() => new Suppression();

    /// <summary>Whether <see cref="SuppressOnThisThread"/> is in force (the Winograd kernel honours it too).</summary>
    internal static bool SuppressedOnThisThread => t_suppressed;

    private sealed class Suppression : IDisposable
    {
        private readonly bool _previous;
        internal Suppression() { _previous = t_suppressed; t_suppressed = true; }
        public void Dispose() => t_suppressed = _previous;
    }

    /// <summary>Passes allowed onto the direct kernels (process-wide; for A/B measurement and kernel tuning).</summary>
    internal static DirectConvPasses EnabledPasses { get; set; } = DirectConvPasses.All;

#if NET5_0_OR_GREATER
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;
#else
    public static bool IsSupported => false;
#endif

    // Interleaved A/B on a 3950X (batch 8, ResNet-18 shapes), existing route -> direct, after the register-tile and
    // L1-chunking work: the direct kernels win every 3x3 pass up to 32x32 planes (64 channels on 32x32: forward 1.42
    // -> 1.29, dX 1.41 -> 1.18, dW 1.23 -> 1.10 ms; 64->128 stride 2: forward 2.2 -> 0.75, dX 2.5 -> 0.92, dW 2.8 ->
    // 0.88 ms; 512 channels on 4x4: dW 4.4 -> 1.0 ms) and lose on 1x1 kernels (64->128 stride 2: forward 0.46 -> 0.58).
    // Larger planes stay on the Winograd / implicit-GEMM routes until measured.
    private const int MaxStride1PlanePositions = 1024;
    // A strided input gradient runs one stride-1 conv per phase over a grid of (H/stride) x (W/stride); below this
    // many grid positions per image the tiles are too short. It was 64 (256->512 stride 2 on 8x8, a 4x4 grid, measured
    // 1.2 ms -> 1.4); with runs that cross grid rows and a destination-ordered phase pack, that shape takes 5.3 ms direct
    // against 12.2 ms on the im2col route (one thread, emulated CI runner; PyTorch 4.7-6.6 ms).
    internal static int MinStridedPhasePositions = 16;

    /// <summary>
    /// Whether <paramref name="shape"/>'s pass runs on the direct kernel, and with what forward task target (0 = the
    /// default). A configuration activated for the exact shape (<see cref="DirectConvTuning"/>, normally a tuned
    /// winner) decides; otherwise the measured rules below do. Shapes the kernels cannot run are never routed.
    /// </summary>
    public static bool TryChoose(in DirectConvShape shape, out int targetTasks)
    {
        targetTasks = 0;
        if (t_suppressed || !IsEligible(shape)) return false;
        if (DirectConvTuning.TryGet(shape, out DirectConvConfiguration configuration))
        {
            targetTasks = configuration.TargetTasks;
            return configuration.Route == DirectConvRoute.Direct;
        }
        return DefaultChoosesDirect(shape);
    }

    /// <summary>Whether the direct kernels can run <paramref name="shape"/> at all (layout and alignment, not speed).</summary>
    public static bool IsEligible(in DirectConvShape shape)
    {
        if (!IsSupported || shape.Batch < 1 || shape.OutputHeight <= 0 || shape.OutputWidth <= 0) return false;
        switch (shape.Pass)
        {
            case DirectConvPass.Forward:
                // Input channels need not fill a block: the packing zero-pads them (a 3-channel image stem).
                return (EnabledPasses & DirectConvPasses.Forward) != 0
                    && shape.OutChannels % (Block * OutputBlocksPerTile) == 0;
            case DirectConvPass.BackwardInput:
                return (EnabledPasses & (shape.StrideH == 1 && shape.StrideW == 1
                        ? DirectConvPasses.BackwardInput : DirectConvPasses.BackwardInputStrided)) != 0
                    && shape.DilationH == 1 && shape.DilationW == 1
                    && shape.PadH <= shape.KernelHeight - 1 && shape.PadW <= shape.KernelWidth - 1
                    && shape.OutChannels % Block == 0
                    && shape.InChannels % (Block * OutputBlocksPerTile) == 0;
            case DirectConvPass.BackwardKernel:
                return (EnabledPasses & DirectConvPasses.BackwardKernel) != 0
                    && shape.InChannels % Block == 0
                    && shape.OutChannels % Block == 0;
            default:
                return false;
        }
    }

    /// <summary>The measured routing rules (see the constants above) for a shape with no activated configuration.</summary>
    /// <summary>
    /// #653: a single image whose stride-1 output plane is at or past the batch rule's 1024 positions, up to 160x160, with at
    /// least 64 input channels. Measured in-process against the im2col GEMM routes (alternating calls, 1 and 16 threads),
    /// direct over GEMM: 64ch@40 1.15-1.86x, 64ch@48 1.29-2.29x, 64ch@64 1.27-1.44x, 96ch@64 1.42-1.94x, 128ch@64
    /// 1.10-1.72x, 64ch@80 1.41-1.49x, and with the staged output store 64ch@112 1.51-1.54x, 64ch@128 1.4-1.5x,
    /// 96ch@128 1.68-1.88x, 128ch@128 1.54-1.71x, 256ch@128 1.27-1.72x, 64ch@160 1.34-1.52x. Outside it the GEMM routes
    /// win: 32 channels at 16 threads (0.6-0.7x) and planes of 1024 or less at one thread (0.94-0.97x). Larger planes are
    /// unmeasured. Shape-only, so the route (and the result) is the same at every thread count.
    /// <para>
    /// A 32x32 plane (exactly 1024) joins only when the forward runs its input channels in more than one cache-sized
    /// pass; a channel count that fits one pass keeps GEMM. It joined once those passes existed: a
    /// Kandinsky-sized UNet's 768ch@32 is 106 against 134 ms on one thread and 20-27 against 37 ms on eight, its
    /// decoder's 2304->768ch@32 322 against 375 and 57-60 against 132 ms. A 16x16 plane stays on GEMM: direct wins
    /// at eight threads (1536ch 29-35 against 38 ms) but loses 5-11% on one.
    /// </para>
    /// </summary>
    internal static bool SingleImageLargePlaneForward(in DirectConvShape shape)
    {
        if (shape.Pass != DirectConvPass.Forward || shape.Batch != 1 || shape.StrideH != 1 || shape.StrideW != 1) return false;
        long plane = (long)shape.OutputHeight * shape.OutputWidth;
        if (shape.InChannels < 64 || plane < MaxStride1PlanePositions || plane > MaxSingleImagePlanePositions) return false;
        // Exactly 1024 positions joined BECAUSE of the input-channel passes (768ch measured above). A channel count
        // that fits one pass (64-112 at 3x3) is the single-pass configuration that measured 0.94-0.97x against GEMM
        // on one thread, so it stays on GEMM.
        if (plane == MaxStride1PlanePositions)
        {
#if NET5_0_OR_GREATER
            int inBlocks = (shape.InChannels + Block - 1) / Block;
            return ForwardInputBlocksPerPass(inBlocks, shape.KernelHeight * shape.KernelWidth) < inBlocks;
#else
            return false;   // the input-channel passes (and the direct kernel) exist on .NET 5+ only
#endif
        }
        return true;
    }

    // Largest output plane SingleImageLargePlaneForward takes: the largest measured (160x160).
    private const int MaxSingleImagePlanePositions = 160 * 160;

    public static bool DefaultChoosesDirect(in DirectConvShape shape)
    {
        if (shape.KernelHeight * shape.KernelWidth == 1) return false;
        bool singleImage = SingleImageLargePlaneForward(shape);
        if (shape.Batch < 2 && !singleImage) return false;
        // A partial input-channel block (an image stem) wastes most of the forward tile; measured slower on 32x32
        // (3 channels: 0.30 ms existing, 0.41 direct), so it is left to tuning.
        if (shape.Pass == DirectConvPass.Forward && shape.InChannels % Block != 0) return false;
        bool strided = shape.StrideH > 1 || shape.StrideW > 1;
        int plane = shape.Pass == DirectConvPass.BackwardInput
            ? shape.Height * shape.Width
            : shape.OutputHeight * shape.OutputWidth;
        if (!strided && plane > MaxStride1PlanePositions && !singleImage) return false;
        return shape.Pass switch
        {
            DirectConvPass.BackwardInput => !strided
                || (shape.Height / shape.StrideH) * (shape.Width / shape.StrideW) >= MinStridedPhasePositions,
            DirectConvPass.BackwardKernel => (shape.InChannels / Block) * (shape.OutChannels / Block) >= MinBackwardKernelTasks,
            _ => true,
        };
    }
#if NET5_0_OR_GREATER
    /// <summary>output[n, oc, oh, ow] (=, or += when <paramref name="accumulate"/>) conv(input, kernel), NCHW / OIHW.</summary>
    [MethodImpl(Hot)]
    public static void Forward(
        float[] input, int inputOffset, float[] kernel, int kernelOffset, float[] output, int outputOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outputHeight, int outputWidth,
        int targetTasks = 0)
    {
        int paddedH = height + 2 * padH, paddedW = width + 2 * padW;
        int inBlocks = (inChannels + Block - 1) / Block;
        var pool = ArrayPool<float>.Shared;
        var packedInput = pool.Rent(batch * inBlocks * paddedH * paddedW * Block);
        var packedKernel = pool.Rent(outChannels * inBlocks * Block * kernelHeight * kernelWidth);
        try
        {
            PackInput(input, inputOffset, packedInput, batch, inChannels, height, width, padH, padW, paddedH, paddedW);
            bool interleave = Interleave512(outChannels);
            PackKernel(kernel, kernelOffset, packedKernel, outChannels, inChannels, kernelHeight, kernelWidth, transposeAndFlip: false, interleave);
            ForwardPacked(packedInput, packedKernel, interleave, output, outputOffset, accumulate,
                ForwardGeometry.Plain(batch, inBlocks, paddedH, paddedW, outChannels, kernelHeight, kernelWidth,
                    strideH, strideW, dilationH, dilationW, outputHeight, outputWidth), targetTasks);
        }
        finally
        {
            pool.Return(packedInput);
            pool.Return(packedKernel);
        }
    }

    /// <summary>
    /// dX (=, or +=) for a stride-1, dilation-1 conv: the forward conv of the output gradient with
    /// W'[i, o, y, x] = W[o, i, kH-1-y, kW-1-x] and padding k-1-p.
    /// </summary>
    [MethodImpl(Hot)]
    public static void BackwardInput(
        float[] gradOutput, int gradOutputOffset, float[] kernel, int kernelOffset, float[] dest, int destOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int outputHeight, int outputWidth, int targetTasks = 0)
    {
        if (strideH > 1 || strideW > 1)
        {
            BackwardInputStrided(gradOutput, gradOutputOffset, kernel, kernelOffset, dest, destOffset, accumulate,
                batch, inChannels, height, width, outChannels, kernelHeight, kernelWidth,
                strideH, strideW, padH, padW, outputHeight, outputWidth, targetTasks);
            return;
        }
        int padHt = kernelHeight - 1 - padH, padWt = kernelWidth - 1 - padW;
        int paddedH = outputHeight + 2 * padHt, paddedW = outputWidth + 2 * padWt;
        int gradBlocks = outChannels / Block;
        var pool = ArrayPool<float>.Shared;
        var packedGrad = pool.Rent(batch * gradBlocks * paddedH * paddedW * Block);
        var packedKernel = pool.Rent(outChannels * inChannels * kernelHeight * kernelWidth);
        try
        {
            PackInput(gradOutput, gradOutputOffset, packedGrad, batch, outChannels, outputHeight, outputWidth, padHt, padWt, paddedH, paddedW);
            bool interleave = Interleave512(inChannels);
            PackKernel(kernel, kernelOffset, packedKernel, outChannels, inChannels, kernelHeight, kernelWidth, transposeAndFlip: true, interleave);
            ForwardPacked(packedGrad, packedKernel, interleave, dest, destOffset, accumulate,
                ForwardGeometry.Plain(batch, gradBlocks, paddedH, paddedW, inChannels, kernelHeight, kernelWidth,
                    1, 1, 1, 1, height, width), targetTasks);
        }
        finally
        {
            pool.Return(packedGrad);
            pool.Return(packedKernel);
        }
    }

    /// <summary>
    /// dX for a strided conv, as oneDNN computes it: split dX into strideH x strideW phases (ih mod strideH, iw mod
    /// strideW). Phase (ry, rx) receives only the kernel taps kh = (ry + padH) mod strideH + strideH*t (likewise kw),
    /// and over its own grid of rows ry + strideH*j it is a stride-1 correlation of the output gradient with that
    /// flipped sub-kernel, so the forward tile runs it unchanged. No column matrix, no scatter.
    /// </summary>
    [MethodImpl(Hot)]
    private static void BackwardInputStrided(
        float[] gradOutput, int gradOutputOffset, float[] kernel, int kernelOffset, float[] dest, int destOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int outputHeight, int outputWidth, int targetTasks)
    {
        int taskTarget = targetTasks > 0 ? targetTasks : TargetForwardTasks;
        // Every phase origin lies within a kernel extent of the gradient plane, so a kernel-sized border covers all reads.
        int borderH = kernelHeight, borderW = kernelWidth;
        int paddedH = outputHeight + 2 * borderH, paddedW = outputWidth + 2 * borderW;
        int gradBlocks = outChannels / Block;
        int tiles = inChannels / (Block * OutputBlocksPerTile);

        // The phases that receive a tap. Their sub-kernels partition the kernel, so they pack into one kernel-sized buffer.
        var phases = new List<(ForwardGeometry Geometry, int KernelOffset, int FirstH, int TapsH, int FirstW, int TapsW, int Chunks)>();
        int packedOffset = 0;
        for (int ry = 0; ry < strideH; ry++)
        {
            for (int rx = 0; rx < strideW; rx++)
            {
                int firstH = (ry + padH) % strideH, firstW = (rx + padW) % strideW;
                int tapsH = firstH < kernelHeight ? (kernelHeight - 1 - firstH) / strideH + 1 : 0;
                int tapsW = firstW < kernelWidth ? (kernelWidth - 1 - firstW) / strideW + 1 : 0;
                int gridH = ry < height ? (height - ry + strideH - 1) / strideH : 0;
                int gridW = rx < width ? (width - rx + strideW - 1) / strideW : 0;
                if (gridH == 0 || gridW == 0) continue;
                if (tapsH == 0 || tapsW == 0)
                {
                    if (!accumulate) ClearPhase(dest, destOffset, batch * inChannels, height, width, ry, rx, strideH, strideW);
                    continue;
                }
                // Grid row j, sub-kernel row y (= tap tapsH-1-y) reads gradient row j + (ry+padH)/strideH - tapsH + 1 + y.
                int originRow = (ry + padH) / strideH - tapsH + 1 + borderH;
                int originCol = (rx + padW) / strideW - tapsW + 1 + borderW;
                var geometry = new ForwardGeometry(batch, gradBlocks, paddedH * paddedW * Block, paddedW, inChannels,
                    tapsH, tapsW, 1, 1, 1, 1, gridH, gridW, (originRow * paddedW + originCol) * Block,
                    height, width, strideH, strideW, ry, rx);
                int chunks = (geometry.TotalPositions + PositionUnroll - 1) / PositionUnroll;
                phases.Add((geometry, packedOffset, firstH, tapsH, firstW, tapsW, chunks));
                packedOffset += outChannels * inChannels * tapsH * tapsW;
            }
        }
        if (phases.Count == 0) return;

        var pool = ArrayPool<float>.Shared;
        var packedGrad = pool.Rent(batch * gradBlocks * paddedH * paddedW * Block);
        var packedKernel = pool.Rent(Math.Max(1, packedOffset));
        try
        {
            PackInput(gradOutput, gradOutputOffset, packedGrad, batch, outChannels, outputHeight, outputWidth, borderH, borderW, paddedH, paddedW);
            int inBlocks = inChannels / Block;
            CpuParallelSettings.ParallelForOrSerial(0, phases.Count * inBlocks, (long)outChannels * inChannels * kernelHeight * kernelWidth, [MethodImpl(Hot)] (task) =>
            {
                var phase = phases[task / inBlocks];
                PackKernelPhase(kernel, kernelOffset, packedKernel, phase.KernelOffset, task % inBlocks, outChannels, inChannels,
                    kernelHeight, kernelWidth, phase.FirstH, strideH, phase.TapsH, phase.FirstW, strideW, phase.TapsW);
            }, deterministicSafe: true);

            // Each task runs EVERY phase over the same share of the grid (the same fraction of each phase's positions,
            // so the same images and rows): the phases interleave in dX (stride 2: alternate columns), and as separate
            // tasks up to four cores wrote each dX cache line at different times. The phases write disjoint positions.
            int minChunks = int.MaxValue;
            foreach (var phase in phases) minChunks = Math.Min(minChunks, phase.Chunks);
            // At least four position triples per phase in a share: shorter runs cost more in setup than they save.
            int sharesPerTile = Math.Max(1, Math.Min(minChunks / 4, (taskTarget + tiles - 1) / tiles));
            CpuParallelSettings.ParallelForOrSerial(0, tiles * sharesPerTile,
                (long)batch * height * width * inChannels * outChannels * kernelHeight * kernelWidth / (strideH * strideW),
                [MethodImpl(Hot)] (task) =>
                {
                    int tile = task / sharesPerTile, share = task % sharesPerTile;
                    foreach (var phase in phases)
                    {
                        int total = phase.Geometry.TotalPositions;
                        int first = ShareBound(total, share, sharesPerTile), last = ShareBound(total, share + 1, sharesPerTile);
                        if (first >= last) continue;
                        if (s_stagedStore)
                            ForwardTileRuns(packedGrad, packedKernel, phase.KernelOffset, dest, destOffset, accumulate, phase.Geometry, tile, first, last, interleaved: false);
                        else
                            ForwardTile(packedGrad, packedKernel, phase.KernelOffset, dest, destOffset, accumulate, phase.Geometry, tile, first, last);
                    }
                },
                deterministicSafe: true);
        }
        finally
        {
            pool.Return(packedGrad);
            pool.Return(packedKernel);
        }
    }
    // Start of share `share` of `shares` over `total` positions, on a PositionUnroll boundary (the last ends at total).
    private static int ShareBound(int total, int share, int shares)
        => share >= shares ? total : Math.Min(total, (int)((long)total * share / shares) / PositionUnroll * PositionUnroll);

    /// <summary>Zeroes phase (ry, rx) of every plane: the input positions no kernel tap reaches (a 1x1 stride-2 conv's odd rows).</summary>
    [MethodImpl(Hot)]
    private static void ClearPhase(float[] dest, int destOffset, int planes, int height, int width, int ry, int rx, int strideH, int strideW)
    {
        CpuParallelSettings.ParallelForOrSerial(0, planes, (long)planes * height * width / (strideH * strideW), [MethodImpl(Hot)] (plane) =>
        {
            int baseIndex = destOffset + plane * height * width;
            for (int y = ry; y < height; y += strideH)
                for (int x = rx; x < width; x += strideW)
                    dest[baseIndex + y * width + x] = 0f;
        }, deterministicSafe: true);
    }

    /// <summary>
    /// One input block of one strided-dX phase's packed kernel, at <paramref name="packedOffset"/>:
    /// [I/8][O/8][tapsH][tapsW][8 out][8 in] with sub-kernel tap (y, x)
    /// = W[o, i, firstH + strideH*(tapsH-1-y), firstW + strideW*(tapsW-1-x)].
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void PackKernelPhase(float[] source, int sourceOffset, float[] packed, int packedOffset, int ib,
        int outChannels, int inChannels, int kernelHeight, int kernelWidth,
        int firstH, int strideH, int tapsH, int firstW, int strideW, int tapsW)
    {
        int taps = tapsH * tapsW;
        int outBlocks = outChannels / Block;
        fixed (float* ps = source)
        fixed (float* pd = packed)
        {
            // Destination order (each 8x8 block written contiguously); the lane-outer order it replaces revisited every
            // destination line eight times, and was 46% of a 256->512 8x8 stride-2 input gradient (one thread).
            float* d = pd + packedOffset + (long)ib * outBlocks * taps * Block * Block;
            int kernelPlane = kernelHeight * kernelWidth;
            for (int ob = 0; ob < outBlocks; ob++)
                for (int y = 0; y < tapsH; y++)
                {
                    int kh = firstH + strideH * (tapsH - 1 - y);
                    for (int x = 0; x < tapsW; x++)
                    {
                        int tap = kh * kernelWidth + firstW + strideW * (tapsW - 1 - x);
                        for (int oo = 0; oo < Block; oo++)
                        {
                            float* s = ps + sourceOffset + ((long)(ob * Block + oo) * inChannels + ib * Block) * kernelPlane + tap;
                            for (int lane = 0; lane < Block; lane++) *d++ = s[lane * kernelPlane];
                        }
                    }
                }
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RunBackwardKernelTile(bool wide, float[] packedInput, float[] packedGrad, int[] offsets, float[] dest,
        int destOffset, bool accumulate, int ob, int it, int batchBegin, int batchEnd, int inChannels, int outBlocks,
        int paddedW, int planeIn, int kernelHeight, int kernelWidth, int dilationH, int dilationW, int positions)
    {
#if NET9_0_OR_GREATER
        if (wide)
        {
            BackwardKernelTile512(packedInput, packedGrad, offsets, dest, destOffset, accumulate, ob, it, batchBegin, batchEnd,
                inChannels, outBlocks, paddedW, planeIn, kernelHeight, kernelWidth, dilationH, dilationW, positions);
            return;
        }
#endif
        BackwardKernelTile(packedInput, packedGrad, offsets, dest, destOffset, accumulate, ob, it, batchBegin, batchEnd,
            inChannels, outBlocks, paddedW, planeIn, kernelHeight, kernelWidth, dilationH, dilationW, positions);
    }

    /// <summary>dW[oc, ic, kh, kw] (=, or += when <paramref name="accumulate"/>) for the conv of input with kernel.</summary>
    [MethodImpl(Hot)]
    public static void BackwardKernel(
        float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset, float[] dest, int destOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outputHeight, int outputWidth,
        int targetTasks = 0)
    {
        int paddedH = height + 2 * padH, paddedW = width + 2 * padW;
        int inBlocks = inChannels / Block, outBlocks = outChannels / Block;
        int positions = outputHeight * outputWidth;
        int planeIn = paddedH * paddedW * Block;
#if NET9_0_OR_GREATER
        // AVX-512: one task per output block x pair of input blocks (sixteen input channels per vector).
        bool wide = UseAvx512BackwardKernel(inChannels);
#else
        const bool wide = false;
#endif
        int inTiles = wide ? inBlocks / 2 : inBlocks;
        int pairs = outBlocks * inTiles;
        int kernelSize = outChannels * inChannels * kernelHeight * kernelWidth;
        // Few block pairs leave cores idle (32 pairs for 32->64 channels), so the batch is split too: each slice
        // sums into its own partial gradient and the partials are reduced in a fixed order afterwards.
        // The split count follows the 8-lane block pairs whichever tile runs, so both sum the same partials in the same
        // order (the AVX-512 tile's results are bit-identical to the AVX2 tile's).
        int blockPairs = outBlocks * inBlocks;
        int splits = Math.Max(1, Math.Min(batch, ((targetTasks > 0 ? targetTasks : TargetBackwardKernelTasks) + blockPairs - 1) / blockPairs));
        // Each split adds a kernel-sized partial that ReducePartials reads back, so bound the partials' total size.
        splits = Math.Max(1, Math.Min(splits, (int)Math.Min(int.MaxValue, MaxBackwardKernelPartialFloats / Math.Max(1, kernelSize))));
        int imagesPerSplit = (batch + splits - 1) / splits;
        splits = (batch + imagesPerSplit - 1) / imagesPerSplit;
        var pool = ArrayPool<float>.Shared;
        var packedInput = pool.Rent(batch * inBlocks * planeIn);
        var packedGrad = pool.Rent(batch * outChannels * positions);
        var offsets = ArrayPool<int>.Shared.Rent(batch * positions);
        var partials = splits > 1 ? pool.Rent(splits * kernelSize) : null;
        try
        {
            PackInput(input, inputOffset, packedInput, batch, inChannels, height, width, padH, padW, paddedH, paddedW);
            PackGradByBlock(gradOutput, gradOutputOffset, packedGrad, batch, outChannels, positions);
            for (int b = 0; b < batch; b++)
                for (int q = 0; q < positions; q++)
                    offsets[b * positions + q] = b * inBlocks * planeIn
                        + ((q / outputWidth) * strideH * paddedW + (q % outputWidth) * strideW) * Block;

            if (partials is null)
            {
                CpuParallelSettings.ParallelForOrSerial(0, pairs, (long)kernelSize * batch * positions,
                    [MethodImpl(Hot)] (task) => RunBackwardKernelTile(wide, packedInput, packedGrad, offsets, dest, destOffset,
                        accumulate, task / inTiles, task % inTiles, 0, batch, inChannels, outBlocks, paddedW, planeIn,
                        kernelHeight, kernelWidth, dilationH, dilationW, positions),
                    deterministicSafe: true);
            }
            else
            {
                CpuParallelSettings.ParallelForOrSerial(0, pairs * splits, (long)kernelSize * batch * positions, [MethodImpl(Hot)] (task) =>
                {
                    int split = task / pairs, pair = task % pairs;
                    int b0 = split * imagesPerSplit;
                    RunBackwardKernelTile(wide, packedInput, packedGrad, offsets, partials, split * kernelSize, false,
                        pair / inTiles, pair % inTiles, b0, Math.Min(batch, b0 + imagesPerSplit), inChannels, outBlocks,
                        paddedW, planeIn, kernelHeight, kernelWidth, dilationH, dilationW, positions);
                }, deterministicSafe: true);
                ReducePartials(partials, splits, kernelSize, dest, destOffset, accumulate);
            }
        }
        finally
        {
            pool.Return(packedInput);
            pool.Return(packedGrad);
            ArrayPool<int>.Shared.Return(offsets);
            if (partials is not null) pool.Return(partials);
        }
    }

    [MethodImpl(Hot)]
    private static unsafe void ReducePartials(float[] partials, int splits, int length, float[] dest, int destOffset, bool accumulate)
    {
        const int Chunk = 2048;
        int chunks = (length + Chunk - 1) / Chunk;
        CpuParallelSettings.ParallelForOrSerial(0, chunks, (long)splits * length, [MethodImpl(Hot)] (chunk) =>
        {
            int begin = chunk * Chunk, end = Math.Min(length, begin + Chunk);
            fixed (float* pp = partials)
            fixed (float* pd = dest)
            {
                float* d = pd + destOffset;
                int i = begin;
                for (; i + Block <= end; i += Block)
                {
                    var sum = accumulate ? Avx.LoadVector256(d + i) : Vector256<float>.Zero;
                    for (int s = 0; s < splits; s++) sum = Avx.Add(sum, Avx.LoadVector256(pp + (long)s * length + i));
                    Avx.Store(d + i, sum);
                }
                for (; i < end; i++)
                {
                    float sum = accumulate ? d[i] : 0f;
                    for (int s = 0; s < splits; s++) sum += pp[(long)s * length + i];
                    d[i] = sum;
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>NCHW -> zero-padded [N][ceil(C/8)][H+2pH][W+2pW][8]; channels past C are zero.</summary>
    [MethodImpl(Hot)]
    private static unsafe void PackInput(float[] source, int sourceOffset, float[] packed,
        int batch, int channels, int height, int width, int padH, int padW, int paddedH, int paddedW)
    {
        int blocks = (channels + Block - 1) / Block;
        int plane = paddedH * paddedW * Block;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * blocks * plane, [MethodImpl(Hot)] (task) =>
        {
            int b = task / blocks, blk = task % blocks;
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                float* d = pd + (long)task * plane;
                int blockChannels = Math.Min(Block, channels - blk * Block);
                if (blockChannels == Block)
                {
                    // Every lane of the interior is written below, so only the halo needs zeroing. Clearing the whole
                    // plane first was 15% of DenseNet-121 training's main-thread zeroing.
                    int rowFloats = paddedW * Block, left = padW * Block, right = (paddedW - padW - width) * Block;
                    new Span<float>(d, padH * rowFloats).Clear();
                    new Span<float>(d + (long)(padH + height) * rowFloats, (paddedH - padH - height) * rowFloats).Clear();
                    // The side halos are a few 8-float pixels per row: stored directly, as a Span.Clear call per row
                    // cost about as much as clearing the whole plane.
                    var zero = Vector256<float>.Zero;
                    for (int y = 0; y < height; y++)
                    {
                        float* row = d + (long)(y + padH) * rowFloats;
                        for (int i = 0; i < left; i += Block) Avx.Store(row + i, zero);
                        float* tail = row + left + width * Block;
                        for (int i = 0; i < right; i += Block) Avx.Store(tail + i, zero);
                    }
                }
                else
                {
                    new Span<float>(d, plane).Clear();   // lanes past the last channel stay zero
                }
                long channelPlane = (long)height * width;
                float* s0 = ps + sourceOffset + ((long)b * channels + blk * Block) * channelPlane;
                // A full block goes 8 pixels x 8 channels at a time through a register transpose; the strided scalar
                // scatter it replaces was 7% of a 64-channel forward.
                int vectorWidth = blockChannels == Block ? width & ~(Block - 1) : 0;
                for (int y = 0; y < height; y++)
                {
                    float* drow = d + ((y + padH) * paddedW + padW) * Block;
                    float* srow = s0 + y * width;
                    for (int x = 0; x < vectorWidth; x += Block)
                        Transpose8(srow + x, channelPlane, drow + x * Block, Block, accumulate: false);
                    for (int c = 0; c < blockChannels; c++)
                    {
                        float* sc = srow + c * channelPlane;
                        for (int x = vectorWidth; x < width; x++) drow[x * Block + c] = sc[x];
                    }
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>NCHW gradient -> [N][C/8][P][8].</summary>
    [MethodImpl(Hot)]
    private static unsafe void PackGradByBlock(float[] source, int sourceOffset, float[] packed, int batch, int channels, int positions)
    {
        int blocks = channels / Block;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * channels * positions, [MethodImpl(Hot)] (task) =>
        {
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                float* d = pd + (long)task * positions * Block;
                float* s = ps + sourceOffset + (long)task * Block * positions;
                // Eight positions x eight channels per register transpose (the scalar scatter was 5-10% of the AVX-512
                // weight gradient at 32 -> 64 channels).
                int q0 = 0;
                for (; q0 + Block <= positions; q0 += Block)
                    Transpose8(s + q0, positions, d + q0 * Block, Block, accumulate: false);
                for (int lane = 0; lane < Block; lane++)
                {
                    float* sl = s + lane * positions;
                    for (int q = q0; q < positions; q++) d[q * Block + lane] = sl[q];
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>
    /// OIHW -> [O/8][I/8][kH][kW][8 in][8 out]. With <paramref name="transposeAndFlip"/> the result is the packed
    /// kernel of the transposed conv: its outputs are the original inputs, and the taps are reversed.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void PackKernel(float[] source, int sourceOffset, float[] packed,
        int outChannels, int inChannels, int kernelHeight, int kernelWidth, bool transposeAndFlip, bool interleave = false)
    {
        int taps = kernelHeight * kernelWidth;
        int packedOut = transposeAndFlip ? inChannels : outChannels;
        int packedIn = transposeAndFlip ? outChannels : inChannels;
        int packedInBlocks = (packedIn + Block - 1) / Block;
        // One task per (output block, input block) pair, and the destination is written strictly in
        // order: [tap][input lane][output lane]. The source reads stay within Block rows of
        // Block*taps contiguous floats, so both sides stream instead of striding a cache line per
        // store (the old lane-outer order touched every destination line Block times).
        CpuParallelSettings.ParallelForOrSerial(0, (packedOut / Block) * packedInBlocks, (long)outChannels * inChannels * taps, [MethodImpl(Hot)] (task) =>
        {
            int ob = task / packedInBlocks, ib = task % packedInBlocks;
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                // Interleaved (see Interleave512): a tile's four output blocks share one [ib][tap][il][32] run, block
                // ob % 4 at lanes 8 (ob % 4) .. +8, so consecutive 8-vectors of this block sit vecStride floats apart.
                int vecStride = interleave ? Block * OutputBlocksPerTile : Block;
                float* d = interleave
                    ? pd + ((long)(ob / OutputBlocksPerTile) * packedInBlocks + ib) * taps * Block * vecStride + (ob % OutputBlocksPerTile) * Block
                    : pd + ((long)ob * packedInBlocks + ib) * taps * Block * Block;
                float* src = ps + sourceOffset;
                // The eight output lanes of one (tap, input lane) sit a fixed stride apart in the source, so a full input
                // block gathers them in one instruction; the scalar walk below (index arithmetic per element) was a
                // third of a 512-channel 4x4 forward (one thread).
                long laneStride = transposeAndFlip ? taps : (long)inChannels * taps;
                if (!transposeAndFlip && (ib + 1) * Block <= packedIn)
                {
                    // Forward packing: the block's 8 output rows each hold [8 input lanes][taps] contiguously, so an 8x8
                    // transpose of columns c..c+7 of that 8 x 8*taps matrix yields, for each column c = il*taps + tap,
                    // the 8 output lanes of (tap, il) -- the destination vector.
                    float* rows = src + ((long)ob * Block * inChannels + (long)ib * Block) * taps;
                    float* d0 = d;
                    for (int c0 = 0; c0 < Block * taps; c0 += Block)
                        TransposeScatter(rows + c0, laneStride, d0, c0, taps, vecStride);
                    return;
                }
                if (transposeAndFlip && (ib + 1) * Block <= packedIn && taps <= MaxTransposePackTaps)
                {
                    // Input-gradient packing: input lane il's row holds [8 output lanes][taps] contiguously. A first
                    // transpose (columns c = lane*taps + tap of the 8 rows) gives tile[c] = the 8 input lanes of
                    // (lane, tap); a second, of tile[tap], tile[taps + tap], ..., gives each (tap, il)'s 8 output lanes.
                    float* tile = stackalloc float[MaxTransposePackTaps * Block * Block];
                    float* rows = src + ((long)ib * Block * inChannels + (long)ob * Block) * taps;
                    for (int c0 = 0; c0 < Block * taps; c0 += Block)
                        Transpose8(rows + c0, (long)inChannels * taps, tile + c0 * Block, Block, accumulate: false);
                    for (int t = 0; t < taps; t++)
                        Transpose8(tile + (taps - 1 - t) * Block, (long)taps * Block, d + (long)t * Block * vecStride, vecStride, accumulate: false);
                    return;
                }
                if (Avx2.IsSupported && (ib + 1) * Block <= packedIn && 7 * laneStride <= int.MaxValue)
                {
                    int ls = (int)laneStride;
                    var lanes = Vector256.Create(0, ls, 2 * ls, 3 * ls, 4 * ls, 5 * ls, 6 * ls, 7 * ls);
                    for (int t = 0; t < taps; t++)
                    {
                        int tap = transposeAndFlip ? taps - 1 - t : t;
                        for (int il = 0; il < Block; il++, d += vecStride)
                        {
                            int pi = ib * Block + il, po = ob * Block;
                            long row = transposeAndFlip ? (long)pi * inChannels + po : (long)po * inChannels + pi;
                            Avx.Store(d, Avx2.GatherVector256(src + row * taps + tap, lanes, 4));
                        }
                    }
                    return;
                }
                for (int t = 0; t < taps; t++)
                {
                    int tap = transposeAndFlip ? taps - 1 - t : t;
                    for (int il = 0; il < Block; il++, d += vecStride)
                    {
                        int pi = ib * Block + il;
                        if (pi >= packedIn)
                        {
                            new Span<float>(d, Block).Clear();
                            continue;
                        }
                        for (int lane = 0; lane < Block; lane++)
                        {
                            int po = ob * Block + lane;
                            long row = transposeAndFlip ? (long)pi * inChannels + po : (long)po * inChannels + pi;
                            d[lane] = src[row * taps + tap];
                        }
                    }
                }
            }
        }, deterministicSafe: true);
    }

    // A/B switch for the staged output store (on by default).
    internal static bool s_stagedStore = true;

    [MethodImpl(Hot)]
    private static void ForwardPacked(float[] packedInput, float[] packedKernel, bool interleaved, float[] output, int outputOffset, bool accumulate,
        ForwardGeometry geometry, int targetTasks)
    {
        int taskTarget = targetTasks > 0 ? targetTasks : TargetForwardTasks;
        int tiles = geometry.OutChannels / (Block * OutputBlocksPerTile);
        int totalPositions = geometry.TotalPositions;
        int chunks = (totalPositions + PositionUnroll - 1) / PositionUnroll;
        int chunksPerTask = Math.Max(1, (int)(((long)chunks * tiles + taskTarget - 1) / taskTarget));
        int tasksPerTile = (chunks + chunksPerTask - 1) / chunksPerTask;
        CpuParallelSettings.ParallelForOrSerial(0, tiles * tasksPerTile,
            (long)totalPositions * geometry.OutChannels * geometry.InBlocks * Block * geometry.KernelHeight * geometry.KernelWidth,
            [MethodImpl(Hot)] (task) =>
            {
                int tile = task / tasksPerTile, part = task % tasksPerTile;
                int first = part * chunksPerTask * PositionUnroll;
                int last = Math.Min(totalPositions, first + chunksPerTask * PositionUnroll);
                if (s_stagedStore)
                    ForwardTileRuns(packedInput, packedKernel, 0, output, outputOffset, accumulate, geometry, tile, first, last, interleaved);
                else
                    ForwardTile(packedInput, packedKernel, 0, output, outputOffset, accumulate, geometry, tile, first, last);
            },
            deterministicSafe: true);
    }

    /// <summary>
    /// Where a forward tile reads and writes. Positions run over a grid of <see cref="GridPositions"/> per image;
    /// position (row, col) reads the packed input at <see cref="OriginBase"/> + (row*strideH, col*strideW) and writes
    /// output plane element (<see cref="OffsetH"/> + row*<see cref="StepH"/>, <see cref="OffsetW"/> +
    /// col*<see cref="StepW"/>). A plain conv uses the whole output plane as its grid; one phase of a strided input
    /// gradient uses every StepH-th row and StepW-th column of the input-gradient plane.
    /// </summary>
    private readonly struct ForwardGeometry
    {
        public ForwardGeometry(int batch, int inBlocks, int planeIn, int paddedW, int outChannels, int kernelHeight, int kernelWidth,
            int strideH, int strideW, int dilationH, int dilationW, int gridHeight, int gridWidth, int originBase,
            int outHeight, int outWidth, int stepH, int stepW, int offsetH, int offsetW)
        {
            InBlocks = inBlocks; PlaneIn = planeIn; PaddedW = paddedW; OutChannels = outChannels;
            KernelHeight = kernelHeight; KernelWidth = kernelWidth; StrideH = strideH; StrideW = strideW;
            DilationH = dilationH; DilationW = dilationW; GridWidth = gridWidth; GridPositions = gridHeight * gridWidth;
            TotalPositions = batch * GridPositions; OriginBase = originBase;
            OutPlane = outHeight * outWidth; OutWidth = outWidth; StepH = stepH; StepW = stepW; OffsetH = offsetH; OffsetW = offsetW;
        }

        /// <summary>A plain conv: the grid is the output plane, read from the packed input's origin.</summary>
        public static ForwardGeometry Plain(int batch, int inBlocks, int paddedH, int paddedW, int outChannels,
            int kernelHeight, int kernelWidth, int strideH, int strideW, int dilationH, int dilationW, int outputHeight, int outputWidth)
            => new ForwardGeometry(batch, inBlocks, paddedH * paddedW * Block, paddedW, outChannels, kernelHeight, kernelWidth,
                strideH, strideW, dilationH, dilationW, outputHeight, outputWidth, 0, outputHeight, outputWidth, 1, 1, 0, 0);

        // Consecutive positions of one image are consecutive output floats (stride-1 grid covering whole rows).
        public bool PlainContiguous => StepH == 1 && StepW == 1 && OffsetH == 0 && OffsetW == 0 && GridWidth == OutWidth;

        public int InBlocks { get; }
        public int PlaneIn { get; }
        public int PaddedW { get; }
        public int OutChannels { get; }
        public int KernelHeight { get; }
        public int KernelWidth { get; }
        public int StrideH { get; }
        public int StrideW { get; }
        public int DilationH { get; }
        public int DilationW { get; }
        public int GridWidth { get; }
        public int GridPositions { get; }
        public int TotalPositions { get; }
        public int OriginBase { get; }
        public int OutPlane { get; }
        public int OutWidth { get; }
        public int StepH { get; }
        public int StepW { get; }
        public int OffsetH { get; }
        public int OffsetW { get; }

        /// <summary>Offset of flattened grid position q's receptive-field origin in the packed input.</summary>
        public int InputOrigin(int q)
        {
            int b = q / GridPositions, r = q % GridPositions;
            return b * InBlocks * PlaneIn + OriginBase
                + ((r / GridWidth) * StrideH * PaddedW + (r % GridWidth) * StrideW) * Block;
        }

        /// <summary>Offset, in one output channel's plane, of flattened grid position q, and q's image.</summary>
        public int OutputIndex(int q, out int image)
        {
            image = q / GridPositions;
            int r = q % GridPositions;
            return (OffsetH + (r / GridWidth) * StepH) * OutWidth + OffsetW + (r % GridWidth) * StepW;
        }
    }

    /// <summary>
    /// The forward tile for a plain output grid (consecutive positions are consecutive output floats), at any batch. The
    /// positions run in runs of up to <see cref="s_forwardRunPositions"/> within one image; for each run the input channels
    /// go in passes whose kernel slice fits L1 (<see cref="s_forwardRunPassKernelBytes"/>), every position triple carrying
    /// its accumulators from pass to pass in the run's stage ([position][32 channels], vector loads and stores), and the
    /// run is written to the NCHW output once, as contiguous per-channel runs.
    /// </summary>
    /// <remarks>
    /// Every position triple used to walk the tile's whole kernel slice (4 output blocks x all input blocks x taps: 73 KB
    /// at 64 channels, 3x3), past the 32 KB L1, so each 12 FMAs pulled two weight lines from L2: 15% of loads missed L1
    /// and the kernel ran at L2 bandwidth (64ch@32 b8, one thread: 9.1 ms, 66 GFLOP/s, against PyTorch's 5.2 ms). A pass
    /// now streams its slice from L2 once per run instead of once per triple. A batch above one also used to store each
    /// position's 32 channels as scattered scalars (the stage was single-image only); runs now stop at image boundaries.
    /// Each output's FMA chain is unchanged (from zero, input blocks, taps and channels in order), so the result is
    /// bit-identical to the single-pass tile; an accumulating call adds the finished sum to the output, as that tile did.
    /// </remarks>
    [MethodImpl(Hot)]
    private static unsafe void ForwardTileRuns(float[] packedInput, float[] packedKernel, int kernelOffset, float[] output, int outputOffset,
        bool accumulate, in ForwardGeometry g, int tile, int first, int last, bool interleaved)
    {
        int kernelHeight = g.KernelHeight, kernelWidth = g.KernelWidth, inBlocks = g.InBlocks, planeIn = g.PlaneIn;
        int rowStep = g.DilationH * g.PaddedW * Block, colStep = g.DilationW * Block;
        int taps = kernelHeight * kernelWidth;
        long blockStride = (long)inBlocks * taps * Block * Block;
        int gridPositions = g.GridPositions;
        int runLength = Math.Max(PositionUnroll, Math.Min(s_forwardRunPositions, MaxForwardRunPositions));
        int pass = ForwardRunInputBlocksPerPass(inBlocks, taps);
        int channelBase = tile * OutputBlocksPerTile * Block;
        float* stage = stackalloc float[(MaxForwardRunPositions + RowPositions512) * 32];
        // The run's receptive-field origins, computed once per run (InputOrigin divides; per triple and pass it was a
        // tenth of the forward's time) and stepped along a grid row; the tail repeats the last real position.
        int* origin = stackalloc int[MaxForwardRunPositions + RowPositions512];
        // An interleaved kernel (packed for the 512-bit tile) is only ever read by it.
        bool wide = interleaved || UseAvx512Forward;
        bool rowKernel = wide && s_useRowKernel512 && g.StrideW == 1;
        int gridWidth = g.GridWidth, columnStep = g.StrideW * Block;
        bool plain = g.PlainContiguous;
        fixed (float* px = packedInput)
        fixed (float* pw = packedKernel)
        fixed (float* po = output)
        {
            float* tileKernel = pw + kernelOffset + tile * OutputBlocksPerTile * blockStride;
            float* o = po + outputOffset;
            for (int runStart = first; runStart < last;)
            {
                // A run may span images (the packed input addresses any position), so a small plane (512 channels at
                // 4x4: 16 positions an image) still streams each pass's kernel slice once per run, not once per image;
                // only the flush splits it per image.
                int runEnd = Math.Min(last, runStart + runLength);
                int run = runEnd - runStart;
                for (int i = 0, column = gridWidth; i < run; i++)
                {
                    int q = runStart + i;
                    if (column >= gridWidth) { origin[i] = g.InputOrigin(q); column = q % gridPositions % gridWidth; }
                    else origin[i] = origin[i - 1] + columnStep;
                    column++;
                }
                for (int i = run; i < run + Avx512Positions; i++) origin[i] = origin[run - 1];
                for (int cb0 = 0; cb0 < inBlocks; cb0 += pass)
                {
                    int cbEnd = Math.Min(inBlocks, cb0 + pass);
                    float* passKernel = tileKernel + (long)cb0 * taps * Block * Block * (interleaved ? OutputBlocksPerTile : 1);
                    if (wide)
                    {
                        for (int q = runStart; q < runEnd;)
                        {
                            int i = q - runStart;
                            // Twelve positions of a stride-1 conv as three groups of four consecutive along a grid row
                            // (a row of twelve, or 8 + 4 on an 8-wide plane, or three rows of a 4-wide one): a pointer per
                            // group, 24 accumulators, each weight load feeding twelve positions instead of the sextet's six.
                            if (rowKernel && runEnd - q >= RowPositions512
                                && QuadGroup(origin + i) && QuadGroup(origin + i + 4) && QuadGroup(origin + i + 8))
                            {
                                if (interleaved)
                                    ForwardRowRun512<Interleaved512>(px + origin[i], px + origin[i + 4], px + origin[i + 8], passKernel, blockStride, cbEnd - cb0, cb0 * planeIn, planeIn,
                                        kernelHeight, kernelWidth, rowStep, colStep, stage + i * 32, cb0 > 0);
                                else
                                    ForwardRowRun512<Blocked512>(px + origin[i], px + origin[i + 4], px + origin[i + 8], passKernel, blockStride, cbEnd - cb0, cb0 * planeIn, planeIn,
                                        kernelHeight, kernelWidth, rowStep, colStep, stage + i * 32, cb0 > 0);
                                q += RowPositions512;
                                continue;
                            }
                            int count = Math.Min(Avx512Positions, runEnd - q);
                            if (interleaved)
                                ForwardSextetRun512<Interleaved512>(px, origin + i, passKernel, blockStride, cbEnd - cb0, cb0 * planeIn, planeIn,
                                    kernelHeight, kernelWidth, rowStep, colStep, stage + i * 32, cb0 > 0, count);
                            else
                                ForwardSextetRun512<Blocked512>(px, origin + i, passKernel, blockStride, cbEnd - cb0, cb0 * planeIn, planeIn,
                                    kernelHeight, kernelWidth, rowStep, colStep, stage + i * 32, cb0 > 0, count);
                            q += Avx512Positions;
                        }
                        continue;
                    }
                    for (int q = runStart; q < runEnd; q += PositionUnroll)
                    {
                        int count = Math.Min(PositionUnroll, runEnd - q), i = q - runStart;
                        // A short final triple repeats its last position; only the real positions are stored.
                        ForwardTripleRun(px + origin[i], px + origin[i + 1], px + origin[i + 2],
                            passKernel, blockStride, cbEnd - cb0, cb0 * planeIn, planeIn, kernelHeight, kernelWidth, rowStep, colStep,
                            stage + (q - runStart) * 32, cb0 > 0, count);
                    }
                }
                for (int segment = runStart; segment < runEnd;)
                {
                    int segmentEnd = Math.Min(runEnd, (segment / gridPositions + 1) * gridPositions);
                    float* segmentStage = stage + (segment - runStart) * 32;
                    if (plain) FlushStage(o, accumulate, g, channelBase, segment, segmentEnd - segment, segmentStage);
                    else FlushStageStrided(o, accumulate, g, channelBase, segment, segmentEnd - segment, segmentStage);
                    segment = segmentEnd;
                }
                runStart = runEnd;
            }
        }
    }

    /// <summary>
    /// One input-channel pass for a position triple: the 3 x 32 accumulators from <paramref name="slot"/> (or zero on the
    /// first pass), the pass's FMAs, and the <paramref name="count"/> real positions stored back. A method of its own so
    /// the twelve accumulators, three broadcasts and the weight register get all sixteen ymm registers with nothing live
    /// across a call.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void ForwardTripleRun(float* x0, float* x1, float* x2, float* w, long blockStride, int blocks,
        int firstBlockOffset, int planeIn, int kernelHeight, int kernelWidth, int rowStep, int colStep,
        float* slot, bool seeded, int count)
    {
        Vector256<float> a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23;
        if (seeded)
        {
            // Slots past the real positions hold no position; a short triple reloads its last real one there.
            float* s1 = count > 1 ? slot + 32 : slot, s2 = count > 2 ? slot + 64 : s1;
            a00 = Avx.LoadVector256(slot); a01 = Avx.LoadVector256(slot + 8); a02 = Avx.LoadVector256(slot + 16); a03 = Avx.LoadVector256(slot + 24);
            a10 = Avx.LoadVector256(s1); a11 = Avx.LoadVector256(s1 + 8); a12 = Avx.LoadVector256(s1 + 16); a13 = Avx.LoadVector256(s1 + 24);
            a20 = Avx.LoadVector256(s2); a21 = Avx.LoadVector256(s2 + 8); a22 = Avx.LoadVector256(s2 + 16); a23 = Avx.LoadVector256(s2 + 24);
        }
        else
        {
            a00 = Vector256<float>.Zero; a01 = a00; a02 = a00; a03 = a00;
            a10 = a00; a11 = a00; a12 = a00; a13 = a00;
            a20 = a00; a21 = a00; a22 = a00; a23 = a00;
        }
        long stride2 = 2 * blockStride, stride3 = 3 * blockStride;
        for (int cb = 0; cb < blocks; cb++)
        {
            int blockOffset = firstBlockOffset + cb * planeIn;
            for (int y = 0; y < kernelHeight; y++)
            {
                int rowOffset = blockOffset + y * rowStep;
                for (int x = 0; x < kernelWidth; x++)
                {
                    int tapOffset = rowOffset + x * colStep;
                    float* p0 = x0 + tapOffset, p1 = x1 + tapOffset, p2 = x2 + tapOffset;
                    // The eight channels unrolled: the loop counter and branch were a sixth of the uops.
                    Vector256<float> b0, b1, b2, wv;
                    b0 = Vector256.Create(p0[0]); b1 = Vector256.Create(p1[0]); b2 = Vector256.Create(p2[0]);
                    wv = Avx.LoadVector256(w); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    b0 = Vector256.Create(p0[1]); b1 = Vector256.Create(p1[1]); b2 = Vector256.Create(p2[1]);
                    wv = Avx.LoadVector256(w + 8); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride + 8); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2 + 8); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3 + 8); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    b0 = Vector256.Create(p0[2]); b1 = Vector256.Create(p1[2]); b2 = Vector256.Create(p2[2]);
                    wv = Avx.LoadVector256(w + 16); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride + 16); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2 + 16); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3 + 16); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    b0 = Vector256.Create(p0[3]); b1 = Vector256.Create(p1[3]); b2 = Vector256.Create(p2[3]);
                    wv = Avx.LoadVector256(w + 24); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride + 24); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2 + 24); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3 + 24); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    b0 = Vector256.Create(p0[4]); b1 = Vector256.Create(p1[4]); b2 = Vector256.Create(p2[4]);
                    wv = Avx.LoadVector256(w + 32); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride + 32); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2 + 32); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3 + 32); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    b0 = Vector256.Create(p0[5]); b1 = Vector256.Create(p1[5]); b2 = Vector256.Create(p2[5]);
                    wv = Avx.LoadVector256(w + 40); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride + 40); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2 + 40); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3 + 40); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    b0 = Vector256.Create(p0[6]); b1 = Vector256.Create(p1[6]); b2 = Vector256.Create(p2[6]);
                    wv = Avx.LoadVector256(w + 48); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride + 48); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2 + 48); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3 + 48); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    b0 = Vector256.Create(p0[7]); b1 = Vector256.Create(p1[7]); b2 = Vector256.Create(p2[7]);
                    wv = Avx.LoadVector256(w + 56); a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                    wv = Avx.LoadVector256(w + blockStride + 56); a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                    wv = Avx.LoadVector256(w + stride2 + 56); a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                    wv = Avx.LoadVector256(w + stride3 + 56); a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                    w += Block * Block;
                }
            }
        }
        Avx.Store(slot, a00); Avx.Store(slot + 8, a01); Avx.Store(slot + 16, a02); Avx.Store(slot + 24, a03);
        if (count > 1) { Avx.Store(slot + 32, a10); Avx.Store(slot + 40, a11); Avx.Store(slot + 48, a12); Avx.Store(slot + 56, a13); }
        if (count > 2) { Avx.Store(slot + 64, a20); Avx.Store(slot + 72, a21); Avx.Store(slot + 80, a22); Avx.Store(slot + 88, a23); }
    }

    /// <summary>
    /// The AVX-512 form of <see cref="ForwardTripleRun"/>: six positions x the tile's 32 output channels as two 16-lane
    /// registers (each joined from two of the packed 8-lane output blocks), twelve accumulators. Each output's FMA chain
    /// is the AVX2 kernel's (same order, one rounding per FMA), so the results are bit-identical; the hosted CI runners
    /// that lost to PyTorch were AVX-512 machines, where PyTorch's oneDNN kernels run 16-wide.
    /// </summary>
#if NET9_0_OR_GREATER
    [MethodImpl(Hot)]
    private static unsafe void ForwardSextetRun512<TLayout>(float* px, int* origin, float* w, long blockStride, int blocks,
        int firstBlockOffset, int planeIn, int kernelHeight, int kernelWidth, int rowStep, int colStep,
        float* slot, bool seeded, int count) where TLayout : struct, IWeightLayout512
    {
        float* x0 = px + origin[0], x1 = px + origin[1], x2 = px + origin[2];
        float* x3 = px + origin[3], x4 = px + origin[4], x5 = px + origin[5];
        Vector512<float> a0l, a0h, a1l, a1h, a2l, a2h, a3l, a3h, a4l, a4h, a5l, a5h;
        if (seeded)
        {
            // Slots past the real positions repeat the last real one (their origins do too), so their sums are discarded.
            float* s0 = slot;
            float* s1 = slot + 32 * Math.Min(1, count - 1);
            float* s2 = slot + 32 * Math.Min(2, count - 1);
            float* s3 = slot + 32 * Math.Min(3, count - 1);
            float* s4 = slot + 32 * Math.Min(4, count - 1);
            float* s5 = slot + 32 * Math.Min(5, count - 1);
            a0l = Vector512.Load(s0); a0h = Vector512.Load(s0 + 16);
            a1l = Vector512.Load(s1); a1h = Vector512.Load(s1 + 16);
            a2l = Vector512.Load(s2); a2h = Vector512.Load(s2 + 16);
            a3l = Vector512.Load(s3); a3h = Vector512.Load(s3 + 16);
            a4l = Vector512.Load(s4); a4h = Vector512.Load(s4 + 16);
            a5l = Vector512.Load(s5); a5h = Vector512.Load(s5 + 16);
        }
        else
        {
            a0l = Vector512<float>.Zero; a0h = a0l; a1l = a0l; a1h = a0l; a2l = a0l; a2h = a0l;
            a3l = a0l; a3h = a0l; a4l = a0l; a4h = a0l; a5l = a0l; a5h = a0l;
        }
        long stride2 = 2 * blockStride, stride3 = 3 * blockStride;
        for (int cb = 0; cb < blocks; cb++)
        {
            int blockOffset = firstBlockOffset + cb * planeIn;
            for (int y = 0; y < kernelHeight; y++)
            {
                int rowOffset = blockOffset + y * rowStep;
                for (int x = 0; x < kernelWidth; x++)
                {
                    int tapOffset = rowOffset + x * colStep;
                    float* p0 = x0 + tapOffset, p1 = x1 + tapOffset, p2 = x2 + tapOffset;
                    float* p3 = x3 + tapOffset, p4 = x4 + tapOffset, p5 = x5 + tapOffset;
                    for (int c = 0; c < Block; c++)
                    {
                        // Output channels 0-15 of the tile, then 16-31 (see IWeightLayout512).
                        var wl = default(TLayout).Low(w, blockStride);
                        var wh = default(TLayout).High(w, stride2, stride3);
                        var b = Vector512.Create(p0[c]);
                        a0l = Vector512.FusedMultiplyAdd(b, wl, a0l); a0h = Vector512.FusedMultiplyAdd(b, wh, a0h);
                        b = Vector512.Create(p1[c]);
                        a1l = Vector512.FusedMultiplyAdd(b, wl, a1l); a1h = Vector512.FusedMultiplyAdd(b, wh, a1h);
                        b = Vector512.Create(p2[c]);
                        a2l = Vector512.FusedMultiplyAdd(b, wl, a2l); a2h = Vector512.FusedMultiplyAdd(b, wh, a2h);
                        b = Vector512.Create(p3[c]);
                        a3l = Vector512.FusedMultiplyAdd(b, wl, a3l); a3h = Vector512.FusedMultiplyAdd(b, wh, a3h);
                        b = Vector512.Create(p4[c]);
                        a4l = Vector512.FusedMultiplyAdd(b, wl, a4l); a4h = Vector512.FusedMultiplyAdd(b, wh, a4h);
                        b = Vector512.Create(p5[c]);
                        a5l = Vector512.FusedMultiplyAdd(b, wl, a5l); a5h = Vector512.FusedMultiplyAdd(b, wh, a5h);
                        w += default(TLayout).Step;
                    }
                }
            }
        }
        a0l.Store(slot); a0h.Store(slot + 16);
        if (count > 1) { a1l.Store(slot + 32); a1h.Store(slot + 48); }
        if (count > 2) { a2l.Store(slot + 64); a2h.Store(slot + 80); }
        if (count > 3) { a3l.Store(slot + 96); a3h.Store(slot + 112); }
        if (count > 4) { a4l.Store(slot + 128); a4h.Store(slot + 144); }
        if (count > 5) { a5l.Store(slot + 160); a5h.Store(slot + 176); }
    }
#else
    private static unsafe void ForwardSextetRun512<TLayout>(float* px, int* origin, float* w, long blockStride, int blocks,
        int firstBlockOffset, int planeIn, int kernelHeight, int kernelWidth, int rowStep, int colStep,
        float* slot, bool seeded, int count) where TLayout : struct, IWeightLayout512 => throw new PlatformNotSupportedException();
#endif

#if NET9_0_OR_GREATER
    /// <summary>Positions per <see cref="ForwardRowRun512"/> call.</summary>
    private const int RowPositions512 = 12;

    /// <summary>Whether the AVX-512 forward takes <see cref="ForwardRowRun512"/> for row-contiguous positions (A/B switch,
    /// AIDOTNET_DIRECT_CONV_ROW512=0 turns it off; tests).</summary>
    internal static bool s_useRowKernel512 = Environment.GetEnvironmentVariable("AIDOTNET_DIRECT_CONV_ROW512") != "0";

    /// <summary>
    /// <see cref="ForwardSextetRun512"/> for twelve positions consecutive along a grid row of a stride-1 conv: position k
    /// reads the packed input at <paramref name="x"/> + k Block, so one pointer with constant offsets addresses all twelve
    /// (twelve separate pointers spilled the general registers), and the 24 accumulators fill the zmm file. Each output's
    /// FMA chain is the AVX2 kernel's: bit-identical results.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void ForwardRowRun512<TLayout>(float* x, float* x4, float* x8, float* w, long blockStride, int blocks,
        int firstBlockOffset, int planeIn, int kernelHeight, int kernelWidth, int rowStep, int colStep,
        float* slot, bool seeded) where TLayout : struct, IWeightLayout512
    {
        Vector512<float> a0l, a0h, a1l, a1h, a2l, a2h, a3l, a3h, a4l, a4h, a5l, a5h, a6l, a6h, a7l, a7h, a8l, a8h, a9l, a9h, a10l, a10h, a11l, a11h;
        if (seeded)
        {
            a0l = Vector512.Load(slot + 0); a0h = Vector512.Load(slot + 16);
            a1l = Vector512.Load(slot + 32); a1h = Vector512.Load(slot + 48);
            a2l = Vector512.Load(slot + 64); a2h = Vector512.Load(slot + 80);
            a3l = Vector512.Load(slot + 96); a3h = Vector512.Load(slot + 112);
            a4l = Vector512.Load(slot + 128); a4h = Vector512.Load(slot + 144);
            a5l = Vector512.Load(slot + 160); a5h = Vector512.Load(slot + 176);
            a6l = Vector512.Load(slot + 192); a6h = Vector512.Load(slot + 208);
            a7l = Vector512.Load(slot + 224); a7h = Vector512.Load(slot + 240);
            a8l = Vector512.Load(slot + 256); a8h = Vector512.Load(slot + 272);
            a9l = Vector512.Load(slot + 288); a9h = Vector512.Load(slot + 304);
            a10l = Vector512.Load(slot + 320); a10h = Vector512.Load(slot + 336);
            a11l = Vector512.Load(slot + 352); a11h = Vector512.Load(slot + 368);
        }
        else
        {
            a0l = Vector512<float>.Zero; a0h = a0l;
            a1l = a0l; a1h = a0l; a2l = a0l; a2h = a0l; a3l = a0l; a3h = a0l; a4l = a0l; a4h = a0l; a5l = a0l; a5h = a0l; a6l = a0l; a6h = a0l; a7l = a0l; a7h = a0l; a8l = a0l; a8h = a0l; a9l = a0l; a9h = a0l; a10l = a0l; a10h = a0l; a11l = a0l; a11h = a0l;
        }
        long stride2 = 2 * blockStride, stride3 = 3 * blockStride;
        for (int cb = 0; cb < blocks; cb++)
        {
            int blockOffset = firstBlockOffset + cb * planeIn;
            for (int y = 0; y < kernelHeight; y++)
            {
                int rowOffset = blockOffset + y * rowStep;
                for (int t = 0; t < kernelWidth; t++)
                {
                    int tapOffset = rowOffset + t * colStep;
                    float* p = x + tapOffset, q = x4 + tapOffset, r = x8 + tapOffset;
                    for (int c = 0; c < Block; c++)
                    {
                        // Output channels 0-15 of the tile, then 16-31 (see IWeightLayout512).
                        var wl = default(TLayout).Low(w, blockStride);
                        var wh = default(TLayout).High(w, stride2, stride3);
                        Vector512<float> b;
                        b = Vector512.Create(p[c]); a0l = Vector512.FusedMultiplyAdd(b, wl, a0l); a0h = Vector512.FusedMultiplyAdd(b, wh, a0h);
                        b = Vector512.Create(p[8 + c]); a1l = Vector512.FusedMultiplyAdd(b, wl, a1l); a1h = Vector512.FusedMultiplyAdd(b, wh, a1h);
                        b = Vector512.Create(p[16 + c]); a2l = Vector512.FusedMultiplyAdd(b, wl, a2l); a2h = Vector512.FusedMultiplyAdd(b, wh, a2h);
                        b = Vector512.Create(p[24 + c]); a3l = Vector512.FusedMultiplyAdd(b, wl, a3l); a3h = Vector512.FusedMultiplyAdd(b, wh, a3h);
                        b = Vector512.Create(q[c]); a4l = Vector512.FusedMultiplyAdd(b, wl, a4l); a4h = Vector512.FusedMultiplyAdd(b, wh, a4h);
                        b = Vector512.Create(q[8 + c]); a5l = Vector512.FusedMultiplyAdd(b, wl, a5l); a5h = Vector512.FusedMultiplyAdd(b, wh, a5h);
                        b = Vector512.Create(q[16 + c]); a6l = Vector512.FusedMultiplyAdd(b, wl, a6l); a6h = Vector512.FusedMultiplyAdd(b, wh, a6h);
                        b = Vector512.Create(q[24 + c]); a7l = Vector512.FusedMultiplyAdd(b, wl, a7l); a7h = Vector512.FusedMultiplyAdd(b, wh, a7h);
                        b = Vector512.Create(r[c]); a8l = Vector512.FusedMultiplyAdd(b, wl, a8l); a8h = Vector512.FusedMultiplyAdd(b, wh, a8h);
                        b = Vector512.Create(r[8 + c]); a9l = Vector512.FusedMultiplyAdd(b, wl, a9l); a9h = Vector512.FusedMultiplyAdd(b, wh, a9h);
                        b = Vector512.Create(r[16 + c]); a10l = Vector512.FusedMultiplyAdd(b, wl, a10l); a10h = Vector512.FusedMultiplyAdd(b, wh, a10h);
                        b = Vector512.Create(r[24 + c]); a11l = Vector512.FusedMultiplyAdd(b, wl, a11l); a11h = Vector512.FusedMultiplyAdd(b, wh, a11h);
                        w += default(TLayout).Step;
                    }
                }
            }
        }
        a0l.Store(slot + 0); a0h.Store(slot + 16);
        a1l.Store(slot + 32); a1h.Store(slot + 48);
        a2l.Store(slot + 64); a2h.Store(slot + 80);
        a3l.Store(slot + 96); a3h.Store(slot + 112);
        a4l.Store(slot + 128); a4h.Store(slot + 144);
        a5l.Store(slot + 160); a5h.Store(slot + 176);
        a6l.Store(slot + 192); a6h.Store(slot + 208);
        a7l.Store(slot + 224); a7h.Store(slot + 240);
        a8l.Store(slot + 256); a8h.Store(slot + 272);
        a9l.Store(slot + 288); a9h.Store(slot + 304);
        a10l.Store(slot + 320); a10h.Store(slot + 336);
        a11l.Store(slot + 352); a11h.Store(slot + 368);
    }
#else
    private const int RowPositions512 = 12;
    internal static bool s_useRowKernel512;
    private static unsafe void ForwardRowRun512<TLayout>(float* x, float* x4, float* x8, float* w, long blockStride, int blocks,
        int firstBlockOffset, int planeIn, int kernelHeight, int kernelWidth, int rowStep, int colStep,
        float* slot, bool seeded) where TLayout : struct, IWeightLayout512 => throw new PlatformNotSupportedException();
#endif

    /// <summary>
    /// How the 512-bit forward tiles read a packed weight vector: output channels 0-15 of the tile (<see cref="Low"/>)
    /// and 16-31 (<see cref="High"/>) for one input lane, and the floats to the next input lane (<see cref="Step"/>).
    /// A struct type argument, so each layout gets its own compiled kernel with nothing branching inside.
    /// </summary>
    private interface IWeightLayout512
    {
        unsafe Vector512<float> Low(float* w, long blockStride);
        unsafe Vector512<float> High(float* w, long stride2, long stride3);
        int Step { get; }
    }

    /// <summary>The AVX2 layout ([O/8][I/8][tap][8 in][8 out]): each 16-lane vector joined from two blocks.</summary>
    private readonly struct Blocked512 : IWeightLayout512
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe Vector512<float> Low(float* w, long blockStride) => Vector512.Create(Vector256.Load(w), Vector256.Load(w + blockStride));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe Vector512<float> High(float* w, long stride2, long stride3) => Vector512.Create(Vector256.Load(w + stride2), Vector256.Load(w + stride3));
        public int Step => Block;
    }

    /// <summary>
    /// The tile-interleaved layout ([O/32][I/8][tap][8 in][32 out]): each 16-lane vector one load. The join above is a
    /// 256-bit load plus an insert, a shuffle-port uop, and on Intel the 512-bit FMAs issue on that port too: two per
    /// twelve or 24 FMAs. Same values in the same lanes, so the same FMA chains and bit-identical results.
    /// </summary>
    private readonly struct Interleaved512 : IWeightLayout512
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe Vector512<float> Low(float* w, long blockStride) => Vector512.Load(w);
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public unsafe Vector512<float> High(float* w, long stride2, long stride3) => Vector512.Load(w + 2 * Block);
        public int Step => Block * OutputBlocksPerTile;
    }

    /// <summary>A/B switch: AIDOTNET_DIRECT_CONV_IL512=0 packs the 512-bit forward's weights in the AVX2 layout.</summary>
    internal static bool s_interleave512 = Environment.GetEnvironmentVariable("AIDOTNET_DIRECT_CONV_IL512") != "0";

    /// <summary>
    /// Whether a forward (or stride-1 input-gradient) conv with <paramref name="packedOut"/> output channels packs its
    /// weights tile-interleaved: only where the staged 512-bit tile reads them, and every tile is full.
    /// </summary>
    private static bool Interleave512(int packedOut)
        => s_interleave512 && UseAvx512Forward && s_stagedStore && packedOut % (Block * OutputBlocksPerTile) == 0;

    /// <summary>Whether the four positions at <paramref name="origin"/> are consecutive along one grid row (Block apart).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe bool QuadGroup(int* origin)
        => origin[1] == origin[0] + Block && origin[2] == origin[0] + 2 * Block && origin[3] == origin[0] + 3 * Block;

    /// <summary>Positions per <see cref="ForwardSextetRun512"/> call.</summary>
    private const int Avx512Positions = 6;

    /// <summary>Whether <see cref="ForwardTileRuns"/> runs the AVX-512 kernel (switch for A/B and tests).</summary>
    internal static bool s_useAvx512 = Environment.GetEnvironmentVariable("AIDOTNET_DIRECT_CONV_AVX512") != "0";

    // Vector512.FusedMultiplyAdd (one rounding, like Fma.MultiplyAdd) is .NET 9+. Written with the cross-platform
    // Vector512 API, the kernel is AVX-512 where the hardware accelerates it and a correct software fallback elsewhere,
    // which s_forceAvx512ForTest uses to test it on AVX2-only hosts.
#if NET9_0_OR_GREATER
    private static bool UseAvx512Forward => s_useAvx512 && (Vector512.IsHardwareAccelerated || s_forceAvx512ForTest);
#else
    private static bool UseAvx512Forward => false;
#endif

    /// <summary>Test hook: run the AVX-512 kernel on hosts without AVX-512 (software Vector512, slow but exact).</summary>
    internal static bool s_forceAvx512ForTest;

    /// <summary>Positions per run of <see cref="ForwardTileRuns"/> (tuning; capped at <see cref="MaxForwardRunPositions"/>).</summary>
    internal static int s_forwardRunPositions = 48;
    private const int MaxForwardRunPositions = 192;

    /// <summary>Bytes of kernel slice one <see cref="ForwardTileRuns"/> pass may span: what stays in L1 beside the run's input rows.</summary>
    internal static int s_forwardRunPassKernelBytes = 16 * 1024;

    /// <summary>Input blocks per <see cref="ForwardTileRuns"/> pass: the most whose tile kernel slice fits the L1 budget.</summary>
    internal static int ForwardRunInputBlocksPerPass(int inBlocks, int taps)
    {
        int perBlock = taps * Block * Block * OutputBlocksPerTile * sizeof(float);
        int fit = Math.Max(1, s_forwardRunPassKernelBytes / Math.Max(1, perBlock));
        if (fit >= inBlocks) return inBlocks;
        int passes = (inBlocks + fit - 1) / fit;
        return (inBlocks + passes - 1) / passes;
    }

    [MethodImpl(Hot)]
    private static unsafe void ForwardTile(float[] packedInput, float[] packedKernel, int kernelOffset, float[] output, int outputOffset,
        bool accumulate, in ForwardGeometry g, int tile, int first, int last)
    {
        // Geometry in locals: read through the `in` reference, every field is a memory load inside the tap loops.
        int kernelHeight = g.KernelHeight, kernelWidth = g.KernelWidth, inBlocks = g.InBlocks, planeIn = g.PlaneIn;
        int rowStep = g.DilationH * g.PaddedW * Block, colStep = g.DilationW * Block;
        int taps = kernelHeight * kernelWidth;
        long blockStride = (long)inBlocks * taps * Block * Block;
        fixed (float* px = packedInput)
        fixed (float* pw = packedKernel)
        fixed (float* po = output)
        {
            float* tileKernel = pw + kernelOffset + tile * OutputBlocksPerTile * blockStride;
            // #653: a position's 32 channels sit a whole output plane apart, so storing them per position wrote 32 scattered
            // floats each time (64 KB apart on a 128x128 plane). A run of positions is staged and written as one contiguous
            // run per channel instead: 64ch@128 single-threaded 38-41 -> 21-22 ms, bit-identical. One image only, so a run
            // never crosses into the next image's planes.
            bool staged = s_stagedStore && g.PlainContiguous && g.TotalPositions == g.GridPositions;
            float* stage = stackalloc float[(StageRun + PositionUnroll) * 32];
            // The input channels run in passes whose kernel slice stays cache-resident: every position triple walks the
            // pass's whole slice, so a slice past L2 (1152 input channels: 1.33 MB) streamed from L3 once per triple.
            // A later pass starts its accumulators from the output the one before wrote, so the passes are one FMA
            // chain, bit-identical to a single pass. Summing a pass from zero and adding it on lost small terms to
            // cancellation: 1e8 in one pass, then -1e8 and 1 in the next, gave 0 instead of 1.
            int pass = ForwardInputBlocksPerPass(inBlocks, taps);
            for (int cb0 = 0; cb0 < inBlocks; cb0 += pass)
            {
            int cbEnd = Math.Min(inBlocks, cb0 + pass);
            bool seeded = cb0 > 0;
            bool accumulatePass = accumulate && !seeded;
            float* passKernel = tileKernel + (long)cb0 * taps * Block * Block;
            int channelBase = tile * OutputBlocksPerTile * Block;
            int runStart = first;
            for (int q = first; q < last; q += PositionUnroll)
            {
                int count = Math.Min(PositionUnroll, last - q);
                // A short final chunk repeats its last position; only the real positions are stored.
                float* x0 = px + g.InputOrigin(q);
                float* x1 = px + g.InputOrigin(count > 1 ? q + 1 : q);
                float* x2 = px + g.InputOrigin(count > 2 ? q + 2 : q);
                // Each position's output base, computed before any accumulator is live: in a method this size the JIT
                // leaves OutputIndex a call, and a call while the accumulators were live (all ymm registers are
                // caller-saved on Unix) made RyuJIT keep all twelve on the stack through the FMA loop.
                float* o = po + outputOffset;
                long outPlane = g.OutPlane;
                float* d0 = PositionBase(o, g, channelBase, q);
                float* d1 = PositionBase(o, g, channelBase, count > 1 ? q + 1 : q);
                float* d2 = PositionBase(o, g, channelBase, count > 2 ? q + 2 : q);
                Vector256<float> a00, a01, a02, a03, a10, a11, a12, a13, a20, a21, a22, a23;
                if (!seeded)
                {
                    a00 = Vector256<float>.Zero; a01 = a00; a02 = a00; a03 = a00;
                    a10 = a00; a11 = a00; a12 = a00; a13 = a00;
                    a20 = a00; a21 = a00; a22 = a00; a23 = a00;
                }
                else if (staged)
                {
                    // The run's outputs are read into the stage once, as contiguous runs per channel.
                    int slot = q - runStart;
                    if (slot == 0) SeedStage(po + outputOffset, g, channelBase, runStart, Math.Min(last - runStart, StageRun + PositionUnroll), stage);
                    float* s = stage + slot * 32;
                    a00 = Avx.LoadVector256(s); a01 = Avx.LoadVector256(s + 8); a02 = Avx.LoadVector256(s + 16); a03 = Avx.LoadVector256(s + 24);
                    a10 = Avx.LoadVector256(s + 32); a11 = Avx.LoadVector256(s + 40); a12 = Avx.LoadVector256(s + 48); a13 = Avx.LoadVector256(s + 56);
                    a20 = Avx.LoadVector256(s + 64); a21 = Avx.LoadVector256(s + 72); a22 = Avx.LoadVector256(s + 80); a23 = Avx.LoadVector256(s + 88);
                }
                else
                {
                    long blockStep = Block * outPlane;
                    a00 = LoadLanes(d0, outPlane); a01 = LoadLanes(d0 + blockStep, outPlane);
                    a02 = LoadLanes(d0 + 2 * blockStep, outPlane); a03 = LoadLanes(d0 + 3 * blockStep, outPlane);
                    a10 = LoadLanes(d1, outPlane); a11 = LoadLanes(d1 + blockStep, outPlane);
                    a12 = LoadLanes(d1 + 2 * blockStep, outPlane); a13 = LoadLanes(d1 + 3 * blockStep, outPlane);
                    a20 = LoadLanes(d2, outPlane); a21 = LoadLanes(d2 + blockStep, outPlane);
                    a22 = LoadLanes(d2 + 2 * blockStep, outPlane); a23 = LoadLanes(d2 + 3 * blockStep, outPlane);
                }
                float* w = passKernel;
                for (int cb = cb0; cb < cbEnd; cb++)
                {
                    int blockOffset = cb * planeIn;
                    for (int y = 0; y < kernelHeight; y++)
                    {
                        int rowOffset = blockOffset + y * rowStep;
                        for (int x = 0; x < kernelWidth; x++)
                        {
                            int tapOffset = rowOffset + x * colStep;
                            float* s0 = x0 + tapOffset, s1 = x1 + tapOffset, s2 = x2 + tapOffset;
                            // The tap's 8 input channels, one per iteration, addressed by pointer bumps. Two per
                            // iteration made RyuJIT spill eight accumulators to the stack and reload them on every
                            // iteration (all 16 ymm registers live across the hand-unrolled halves).
                            for (int c = 0; c < Block; c++)
                            {
                                var b0 = Vector256.Create(s0[c]); var b1 = Vector256.Create(s1[c]); var b2 = Vector256.Create(s2[c]);
                                var wv = Avx.LoadVector256(w);
                                a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                                wv = Avx.LoadVector256(w + blockStride);
                                a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                                wv = Avx.LoadVector256(w + 2 * blockStride);
                                a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                                wv = Avx.LoadVector256(w + 3 * blockStride);
                                a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
                                w += Block;
                            }
                        }
                    }
                }
                if (staged)
                {
                    // Stage this position chunk's 32 channels, flushed as contiguous per-channel runs (the plain grid's
                    // consecutive positions are consecutive output floats).
                    int slot = q - runStart;
                    Avx.Store(stage + slot * 32, a00); Avx.Store(stage + slot * 32 + 8, a01); Avx.Store(stage + slot * 32 + 16, a02); Avx.Store(stage + slot * 32 + 24, a03);
                    if (count > 1) { Avx.Store(stage + (slot + 1) * 32, a10); Avx.Store(stage + (slot + 1) * 32 + 8, a11); Avx.Store(stage + (slot + 1) * 32 + 16, a12); Avx.Store(stage + (slot + 1) * 32 + 24, a13); }
                    if (count > 2) { Avx.Store(stage + (slot + 2) * 32, a20); Avx.Store(stage + (slot + 2) * 32 + 8, a21); Avx.Store(stage + (slot + 2) * 32 + 16, a22); Avx.Store(stage + (slot + 2) * 32 + 24, a23); }
                    int runEnd = q + count;
                    if (runEnd - runStart >= StageRun || runEnd >= last)
                    {
                        FlushStage(po + outputOffset, accumulatePass, g, channelBase, runStart, runEnd - runStart, stage);
                        runStart = runEnd;
                    }
                }
                else
                {
                StorePosition(d0, outPlane, accumulatePass, a00, a01, a02, a03);
                if (count > 1) StorePosition(d1, outPlane, accumulatePass, a10, a11, a12, a13);
                if (count > 2) StorePosition(d2, outPlane, accumulatePass, a20, a21, a22, a23);
                }
            }
            }
        }
    }

    /// <summary>Bytes of a forward tile's kernel slice one input-channel pass may span (see ForwardTile).</summary>
    internal static int s_forwardPassKernelBytes = 128 * 1024;

    /// <summary>Input blocks per forward pass: the most whose tile kernel slice (taps x 8 x 32 floats each) fits the budget.</summary>
    internal static int ForwardInputBlocksPerPass(int inBlocks, int taps)
    {
        int perBlock = taps * Block * Block * OutputBlocksPerTile * sizeof(float);
        int fit = Math.Max(1, s_forwardPassKernelBytes / Math.Max(1, perBlock));
        if (fit >= inBlocks) return inBlocks;
        int passes = (inBlocks + fit - 1) / fit;
        return (inBlocks + passes - 1) / passes; // even passes, so the last is not a sliver
    }
    private const int StageRun = 48;

    // Writes `run` consecutive positions x 32 channels from the stage ([position][channel]) as one contiguous run per
    // channel. Only for the plain grid, where position q's output index is q's offset in the plane.
    [MethodImpl(Hot)]
    private static unsafe void FlushStage(float* output, bool accumulate, in ForwardGeometry g, int channelBase, int q0, int run, float* stage)
    {
        int index = g.OutputIndex(q0, out int b);
        long plane = g.OutPlane;
        float* d = output + ((long)b * g.OutChannels + channelBase) * plane + index;
        // Eight positions x eight channels at a time through a register transpose (the scalar walk was 8% of a forward).
        int vectorRun = run & ~(Block - 1);
        for (int i = 0; i < vectorRun; i += Block)
            for (int cb = 0; cb < 32; cb += Block)
                Transpose8(stage + i * 32 + cb, 32, d + cb * plane + i, plane, accumulate);
        if (vectorRun == run) return;
        for (int ch = 0; ch < 32; ch++)
        {
            float* dst = d + ch * plane;
            if (accumulate) for (int i = vectorRun; i < run; i++) dst[i] += stage[i * 32 + ch];
            else for (int i = vectorRun; i < run; i++) dst[i] = stage[i * 32 + ch];
        }
    }

    // FlushStage for a run of a non-plain grid (a strided input gradient's phase), within one image: consecutive
    // positions are StepW apart within a grid row, and a row wrap moves StepH output rows down. Phase grids are a few
    // positions wide (4 at 8x8 stride 2), so runs that stopped at row ends were too short to amortize the passes.
    [MethodImpl(Hot)]
    private static unsafe void FlushStageStrided(float* output, bool accumulate, in ForwardGeometry g, int channelBase, int q0, int run, float* stage)
    {
        g.OutputIndex(q0, out int b);
        long plane = g.OutPlane;
        int* index = stackalloc int[MaxForwardRunPositions];
        int gridWidth = g.GridWidth, column = q0 % g.GridPositions % gridWidth;
        index[0] = g.OutputIndex(q0, out _);
        for (int i = 1; i < run; i++)
        {
            if (++column == gridWidth) { column = 0; index[i] = g.OutputIndex(q0 + i, out _); }
            else index[i] = index[i - 1] + g.StepW;
        }
        float* d = output + ((long)b * g.OutChannels + channelBase) * plane;
        for (int ch = 0; ch < 32; ch++)
        {
            float* dst = d + ch * plane;
            if (accumulate) for (int i = 0; i < run; i++) dst[index[i]] += stage[i * 32 + ch];
            else for (int i = 0; i < run; i++) dst[index[i]] = stage[i * 32 + ch];
        }
    }

    /// <summary>
    /// Transposes an 8x8 float block: row r of the source (<paramref name="s"/> + r * <paramref name="sStride"/>) becomes
    /// column r of the destination, stored over (or added to) the destination rows.
    /// </summary>
    [MethodImpl(HotInline)]
    private static unsafe void Transpose8(float* s, long sStride, float* d, long dStride, bool accumulate)
    {
        var r0 = Avx.LoadVector256(s); var r1 = Avx.LoadVector256(s + sStride);
        var r2 = Avx.LoadVector256(s + 2 * sStride); var r3 = Avx.LoadVector256(s + 3 * sStride);
        var r4 = Avx.LoadVector256(s + 4 * sStride); var r5 = Avx.LoadVector256(s + 5 * sStride);
        var r6 = Avx.LoadVector256(s + 6 * sStride); var r7 = Avx.LoadVector256(s + 7 * sStride);
        var t0 = Avx.UnpackLow(r0, r1); var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3); var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5); var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7); var t7 = Avx.UnpackHigh(r6, r7);
        var u0 = Avx.Shuffle(t0, t2, 0x44); var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44); var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44); var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44); var u7 = Avx.Shuffle(t5, t7, 0xEE);
        StoreRow(d, accumulate, Avx.Permute2x128(u0, u4, 0x20));
        StoreRow(d + dStride, accumulate, Avx.Permute2x128(u1, u5, 0x20));
        StoreRow(d + 2 * dStride, accumulate, Avx.Permute2x128(u2, u6, 0x20));
        StoreRow(d + 3 * dStride, accumulate, Avx.Permute2x128(u3, u7, 0x20));
        StoreRow(d + 4 * dStride, accumulate, Avx.Permute2x128(u0, u4, 0x31));
        StoreRow(d + 5 * dStride, accumulate, Avx.Permute2x128(u1, u5, 0x31));
        StoreRow(d + 6 * dStride, accumulate, Avx.Permute2x128(u2, u6, 0x31));
        StoreRow(d + 7 * dStride, accumulate, Avx.Permute2x128(u3, u7, 0x31));
    }

    // Transposes the 8x8 block at s (rows sStride apart); column c0 + j's 8 values go to the packed forward kernel's
    // (tap, input lane) slot d[(tap * 8 + il) * 8], where c0 + j = il * taps + tap.
    [MethodImpl(HotInline)]
    private static unsafe void TransposeScatter(float* s, long sStride, float* d, int c0, int taps, int vecStride)
    {
        var r0 = Avx.LoadVector256(s); var r1 = Avx.LoadVector256(s + sStride);
        var r2 = Avx.LoadVector256(s + 2 * sStride); var r3 = Avx.LoadVector256(s + 3 * sStride);
        var r4 = Avx.LoadVector256(s + 4 * sStride); var r5 = Avx.LoadVector256(s + 5 * sStride);
        var r6 = Avx.LoadVector256(s + 6 * sStride); var r7 = Avx.LoadVector256(s + 7 * sStride);
        var t0 = Avx.UnpackLow(r0, r1); var t1 = Avx.UnpackHigh(r0, r1);
        var t2 = Avx.UnpackLow(r2, r3); var t3 = Avx.UnpackHigh(r2, r3);
        var t4 = Avx.UnpackLow(r4, r5); var t5 = Avx.UnpackHigh(r4, r5);
        var t6 = Avx.UnpackLow(r6, r7); var t7 = Avx.UnpackHigh(r6, r7);
        var u0 = Avx.Shuffle(t0, t2, 0x44); var u1 = Avx.Shuffle(t0, t2, 0xEE);
        var u2 = Avx.Shuffle(t1, t3, 0x44); var u3 = Avx.Shuffle(t1, t3, 0xEE);
        var u4 = Avx.Shuffle(t4, t6, 0x44); var u5 = Avx.Shuffle(t4, t6, 0xEE);
        var u6 = Avx.Shuffle(t5, t7, 0x44); var u7 = Avx.Shuffle(t5, t7, 0xEE);
        Avx.Store(d + PackedSlot(c0, taps, vecStride), Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(d + PackedSlot(c0 + 1, taps, vecStride), Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(d + PackedSlot(c0 + 2, taps, vecStride), Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(d + PackedSlot(c0 + 3, taps, vecStride), Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(d + PackedSlot(c0 + 4, taps, vecStride), Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(d + PackedSlot(c0 + 5, taps, vecStride), Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(d + PackedSlot(c0 + 6, taps, vecStride), Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(d + PackedSlot(c0 + 7, taps, vecStride), Avx.Permute2x128(u3, u7, 0x31));
    }

    // Largest kernel (taps) the two-transpose input-gradient pack stages on the stack (7x7).
    private const int MaxTransposePackTaps = 49;

    [MethodImpl(HotInline)]
    private static int PackedSlot(int column, int taps, int vecStride) => ((column % taps) * Block + column / taps) * vecStride;

    [MethodImpl(HotInline)]
    private static unsafe void StoreRow(float* d, bool accumulate, Vector256<float> v)
        => Avx.Store(d, accumulate ? Avx.Add(Avx.LoadVector256(d), v) : v);

    // The reverse of FlushStage: `run` consecutive positions x 32 channels of the output into the stage.
    [MethodImpl(Hot)]
    private static unsafe void SeedStage(float* output, in ForwardGeometry g, int channelBase, int q0, int run, float* stage)
    {
        int index = g.OutputIndex(q0, out int b);
        float* d = output + ((long)b * g.OutChannels + channelBase) * g.OutPlane + index;
        for (int ch = 0; ch < 32; ch++)
        {
            float* src = d + (long)ch * g.OutPlane;
            for (int i = 0; i < run; i++) stage[i * 32 + ch] = src[i];
        }
    }

    // Where position q's first channel (channelBase) sits in the NCHW output; its other channels follow a plane apart.
    [MethodImpl(HotInline)]
    private static unsafe float* PositionBase(float* output, in ForwardGeometry g, int channelBase, int q)
    {
        int index = g.OutputIndex(q, out int b);
        return output + ((long)b * g.OutChannels + channelBase) * g.OutPlane + index;
    }
    [MethodImpl(HotInline)]
    private static unsafe Vector256<float> LoadLanes(float* d, long stride)
        => Vector256.Create(d[0], d[stride], d[2 * stride], d[3 * stride], d[4 * stride], d[5 * stride], d[6 * stride], d[7 * stride]);

    [MethodImpl(HotInline)]
    private static unsafe void StorePosition(float* d, long outPlane, bool accumulate,
        Vector256<float> v0, Vector256<float> v1, Vector256<float> v2, Vector256<float> v3)
    {
        StoreLanes(d, outPlane, accumulate, v0);
        StoreLanes(d + Block * outPlane, outPlane, accumulate, v1);
        StoreLanes(d + 2 * Block * outPlane, outPlane, accumulate, v2);
        StoreLanes(d + 3 * Block * outPlane, outPlane, accumulate, v3);
    }
    [MethodImpl(HotInline)]
    private static unsafe void StoreLanes(float* d, long stride, bool accumulate, Vector256<float> v)
    {
        if (accumulate)
            for (int lane = 0; lane < Block; lane++) d[(long)lane * stride] += v.GetElement(lane);
        else
            for (int lane = 0; lane < Block; lane++) d[(long)lane * stride] = v.GetElement(lane);
    }

    // Positions per chunk of the kernel-gradient sweep: the chunk's gradient (8 KB) and the input rows it reads stay in
    // L1 across all of a block pair's (kernel row, column group, channel half) passes.
    private const int KernelGradPositionChunk = 256;

    /// <summary>
    /// One (output block, input block) pair of dW. For each kernel row the tile is 4 output channels x 3 kernel
    /// columns = 12 accumulators, each a vector over the block's 8 input channels. Per position that is 3 contiguous
    /// input loads (one per column: the blocked layout keeps a tap's 8 channels together), 4 gradient broadcasts and
    /// 12 FMAs. Vectorising over output channels instead needed 12 scalar input broadcasts per position (13 loads per
    /// 12 FMAs), which left the kernel load-bound. Positions are swept in L1-sized chunks, every pass (kernel row x
    /// column group x output-channel half) running over a chunk before the next, with each pass's accumulators parked
    /// in a small stack buffer between chunks: a whole-plane sweep per pass re-streamed the operands from L2.
    /// </summary>
    [MethodImpl(Hot)]
    private static unsafe void BackwardKernelTile(float[] packedInput, float[] packedGrad, int[] offsets, float[] dest, int destOffset,
        bool accumulate, int ob, int ib, int batchBegin, int batchEnd, int inChannels, int outBlocks, int paddedW, int planeIn,
        int kernelHeight, int kernelWidth, int dilationH, int dilationW, int positions)
    {
        int columnGroups = (kernelWidth + 2) / 3;
        int passes = kernelHeight * columnGroups * 2;
        const int AccumulatorsPerPass = 12;
        float* parked = stackalloc float[passes * AccumulatorsPerPass * Block];
        new Span<float>(parked, passes * AccumulatorsPerPass * Block).Clear();
        int step = dilationW * Block;
        fixed (float* px = packedInput)
        fixed (float* pg = packedGrad)
        fixed (int* pOff = offsets)
        fixed (float* pd = dest)
        {
            float* xb = px + (long)ib * planeIn;
            for (int b = batchBegin; b < batchEnd; b++)
            {
                float* gImage = pg + (long)(b * outBlocks + ob) * positions * Block;
                int* oImage = pOff + b * positions;
                for (int q0 = 0; q0 < positions; q0 += KernelGradPositionChunk)
                {
                    int q1 = Math.Min(positions, q0 + KernelGradPositionChunk);
                    int pass = 0;
                    for (int y = 0; y < kernelHeight; y++)
                    {
                        for (int x0 = 0; x0 < kernelWidth; x0 += 3)
                        {
                            int columns = Math.Min(3, kernelWidth - x0);
                            for (int o0 = 0; o0 < Block; o0 += 4, pass++)
                            {
                                // Accumulator (j, k): output channel o0+j, kernel column x0+k; lanes = input channels.
                                float* park = parked + pass * AccumulatorsPerPass * Block;
                                var a00 = Avx.LoadVector256(park); var a01 = Avx.LoadVector256(park + 8); var a02 = Avx.LoadVector256(park + 16);
                                var a10 = Avx.LoadVector256(park + 24); var a11 = Avx.LoadVector256(park + 32); var a12 = Avx.LoadVector256(park + 40);
                                var a20 = Avx.LoadVector256(park + 48); var a21 = Avx.LoadVector256(park + 56); var a22 = Avx.LoadVector256(park + 64);
                                var a30 = Avx.LoadVector256(park + 72); var a31 = Avx.LoadVector256(park + 80); var a32 = Avx.LoadVector256(park + 88);
                                float* xTap = xb + (y * dilationH * paddedW + x0 * dilationW) * Block;
                                float* gq = gImage + q0 * Block + o0;
                                if (columns == 3)
                                {
                                    for (int q = q0; q < q1; q++, gq += Block)
                                    {
                                        float* xs = xTap + oImage[q];
                                        var x0v = Avx.LoadVector256(xs);
                                        var x1v = Avx.LoadVector256(xs + step);
                                        var x2v = Avx.LoadVector256(xs + 2 * step);
                                        var g0 = Vector256.Create(gq[0]);
                                        a00 = Fma.MultiplyAdd(x0v, g0, a00); a01 = Fma.MultiplyAdd(x1v, g0, a01); a02 = Fma.MultiplyAdd(x2v, g0, a02);
                                        var g1 = Vector256.Create(gq[1]);
                                        a10 = Fma.MultiplyAdd(x0v, g1, a10); a11 = Fma.MultiplyAdd(x1v, g1, a11); a12 = Fma.MultiplyAdd(x2v, g1, a12);
                                        var g2 = Vector256.Create(gq[2]);
                                        a20 = Fma.MultiplyAdd(x0v, g2, a20); a21 = Fma.MultiplyAdd(x1v, g2, a21); a22 = Fma.MultiplyAdd(x2v, g2, a22);
                                        var g3 = Vector256.Create(gq[3]);
                                        a30 = Fma.MultiplyAdd(x0v, g3, a30); a31 = Fma.MultiplyAdd(x1v, g3, a31); a32 = Fma.MultiplyAdd(x2v, g3, a32);
                                    }
                                }
                                else
                                {
                                    for (int q = q0; q < q1; q++, gq += Block)
                                    {
                                        float* xs = xTap + oImage[q];
                                        var x0v = Avx.LoadVector256(xs);
                                        var x1v = columns > 1 ? Avx.LoadVector256(xs + step) : Vector256<float>.Zero;
                                        var g0 = Vector256.Create(gq[0]);
                                        a00 = Fma.MultiplyAdd(x0v, g0, a00); a01 = Fma.MultiplyAdd(x1v, g0, a01);
                                        var g1 = Vector256.Create(gq[1]);
                                        a10 = Fma.MultiplyAdd(x0v, g1, a10); a11 = Fma.MultiplyAdd(x1v, g1, a11);
                                        var g2 = Vector256.Create(gq[2]);
                                        a20 = Fma.MultiplyAdd(x0v, g2, a20); a21 = Fma.MultiplyAdd(x1v, g2, a21);
                                        var g3 = Vector256.Create(gq[3]);
                                        a30 = Fma.MultiplyAdd(x0v, g3, a30); a31 = Fma.MultiplyAdd(x1v, g3, a31);
                                    }
                                }
                                Avx.Store(park, a00); Avx.Store(park + 8, a01); Avx.Store(park + 16, a02);
                                Avx.Store(park + 24, a10); Avx.Store(park + 32, a11); Avx.Store(park + 40, a12);
                                Avx.Store(park + 48, a20); Avx.Store(park + 56, a21); Avx.Store(park + 64, a22);
                                Avx.Store(park + 72, a30); Avx.Store(park + 80, a31); Avx.Store(park + 88, a32);
                            }
                        }
                    }
                }
            }

            // dW[ob*8 + o0 + j, ib*8 + lane, y, x0 + k] = lane of accumulator (j, k); padded input lanes are dropped.
            int taps = kernelHeight * kernelWidth;
            int laneCount = Math.Min(Block, inChannels - ib * Block);
            float* d = pd + destOffset;
            int passIndex = 0;
            for (int y = 0; y < kernelHeight; y++)
            {
                for (int x0 = 0; x0 < kernelWidth; x0 += 3)
                {
                    int columns = Math.Min(3, kernelWidth - x0);
                    for (int o0 = 0; o0 < Block; o0 += 4, passIndex++)
                    {
                        float* park = parked + passIndex * AccumulatorsPerPass * Block;
                        for (int j = 0; j < 4; j++)
                        {
                            float* row = d + ((long)(ob * Block + o0 + j) * inChannels + ib * Block) * taps + y * kernelWidth + x0;
                            for (int k = 0; k < columns; k++)
                            {
                                float* acc = park + (j * 3 + k) * Block;
                                float* target = row + k;
                                if (accumulate)
                                    for (int lane = 0; lane < laneCount; lane++) target[(long)lane * taps] += acc[lane];
                                else
                                    for (int lane = 0; lane < laneCount; lane++) target[(long)lane * taps] = acc[lane];
                            }
                        }
                    }
                }
            }
        }
    }
#endif
}
