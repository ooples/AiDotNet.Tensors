using System;
using System.Buffers;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
#if NET5_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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
internal static class DirectConvAvx2
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
    // many grid positions per image the tiles are too short (256->512 stride 2 on 8x8, a 4x4 grid: 1.2 ms -> 1.4).
    private const int MinStridedPhasePositions = 64;

    /// <summary>
    /// Whether <paramref name="shape"/>'s pass runs on the direct kernel, and with what forward task target (0 = the
    /// default). A configuration activated for the exact shape (<see cref="DirectConvTuning"/>, normally a tuned
    /// winner) decides; otherwise the measured rules below do. Shapes the kernels cannot run are never routed.
    /// </summary>
    public static bool TryChoose(in DirectConvShape shape, out int targetTasks)
    {
        targetTasks = 0;
        if (!IsEligible(shape)) return false;
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
    public static bool DefaultChoosesDirect(in DirectConvShape shape)
    {
        if (shape.Batch < 2 || shape.KernelHeight * shape.KernelWidth == 1) return false;
        // A partial input-channel block (an image stem) wastes most of the forward tile; measured slower on 32x32
        // (3 channels: 0.30 ms existing, 0.41 direct), so it is left to tuning.
        if (shape.Pass == DirectConvPass.Forward && shape.InChannels % Block != 0) return false;
        bool strided = shape.StrideH > 1 || shape.StrideW > 1;
        int plane = shape.Pass == DirectConvPass.BackwardInput
            ? shape.Height * shape.Width
            : shape.OutputHeight * shape.OutputWidth;
        if (!strided && plane > MaxStride1PlanePositions) return false;
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
            PackKernel(kernel, kernelOffset, packedKernel, outChannels, inChannels, kernelHeight, kernelWidth, transposeAndFlip: false);
            ForwardPacked(packedInput, packedKernel, output, outputOffset, accumulate,
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
            PackKernel(kernel, kernelOffset, packedKernel, outChannels, inChannels, kernelHeight, kernelWidth, transposeAndFlip: true);
            ForwardPacked(packedGrad, packedKernel, dest, destOffset, accumulate,
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
        var phases = new List<(ForwardGeometry Geometry, int KernelOffset, int FirstH, int TapsH, int FirstW, int TapsW, int Chunks, int ChunksPerTask, int Tasks)>();
        int packedOffset = 0, totalTasks = 0;
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
                int chunksPerTask = Math.Max(1, (int)(((long)chunks * tiles * strideH * strideW + taskTarget - 1) / taskTarget));
                int tasks = tiles * ((chunks + chunksPerTask - 1) / chunksPerTask);
                phases.Add((geometry, packedOffset, firstH, tapsH, firstW, tapsW, chunks, chunksPerTask, tasks));
                packedOffset += outChannels * inChannels * tapsH * tapsW;
                totalTasks += tasks;
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
            CpuParallelSettings.ParallelForOrSerial(0, phases.Count * inBlocks, (long)outChannels * inChannels * kernelHeight * kernelWidth, task =>
            {
                var phase = phases[task / inBlocks];
                PackKernelPhase(kernel, kernelOffset, packedKernel, phase.KernelOffset, task % inBlocks, outChannels, inChannels,
                    kernelHeight, kernelWidth, phase.FirstH, strideH, phase.TapsH, phase.FirstW, strideW, phase.TapsW);
            }, deterministicSafe: true);

            // One dispatch over every phase's tiles: the phases write disjoint dX positions.
            var taskStart = new int[phases.Count + 1];
            for (int p = 0; p < phases.Count; p++) taskStart[p + 1] = taskStart[p] + phases[p].Tasks;
            CpuParallelSettings.ParallelForOrSerial(0, totalTasks,
                (long)batch * height * width * inChannels * outChannels * kernelHeight * kernelWidth / (strideH * strideW),
                task =>
                {
                    int p = 0;
                    while (taskStart[p + 1] <= task) p++;
                    var phase = phases[p];
                    int local = task - taskStart[p];
                    int tasksPerTile = phase.Tasks / tiles;
                    int tile = local / tasksPerTile, part = local % tasksPerTile;
                    int first = part * phase.ChunksPerTask * PositionUnroll;
                    int last = Math.Min(phase.Geometry.TotalPositions, first + phase.ChunksPerTask * PositionUnroll);
                    ForwardTile(packedGrad, packedKernel, phase.KernelOffset, dest, destOffset, accumulate, phase.Geometry, tile, first, last);
                },
                deterministicSafe: true);
        }
        finally
        {
            pool.Return(packedGrad);
            pool.Return(packedKernel);
        }
    }
    /// <summary>Zeroes phase (ry, rx) of every plane: the input positions no kernel tap reaches (a 1x1 stride-2 conv's odd rows).</summary>
    private static void ClearPhase(float[] dest, int destOffset, int planes, int height, int width, int ry, int rx, int strideH, int strideW)
    {
        CpuParallelSettings.ParallelForOrSerial(0, planes, (long)planes * height * width / (strideH * strideW), plane =>
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
    private static unsafe void PackKernelPhase(float[] source, int sourceOffset, float[] packed, int packedOffset, int ib,
        int outChannels, int inChannels, int kernelHeight, int kernelWidth,
        int firstH, int strideH, int tapsH, int firstW, int strideW, int tapsW)
    {
        int taps = tapsH * tapsW;
        int outBlocks = outChannels / Block;
        fixed (float* ps = source)
        fixed (float* pd = packed)
        {
            float* d0 = pd + packedOffset + (long)ib * outBlocks * taps * Block * Block;
            for (int lane = 0; lane < Block; lane++)
            {
                int i = ib * Block + lane;
                for (int o = 0; o < outChannels; o++)
                {
                    float* s = ps + sourceOffset + ((long)o * inChannels + i) * kernelHeight * kernelWidth;
                    float* d = d0 + (long)(o / Block) * taps * Block * Block + (o % Block) * Block + lane;
                    for (int y = 0; y < tapsH; y++)
                    {
                        int kh = firstH + strideH * (tapsH - 1 - y);
                        for (int x = 0; x < tapsW; x++)
                            d[(y * tapsW + x) * Block * Block] = s[kh * kernelWidth + firstW + strideW * (tapsW - 1 - x)];
                    }
                }
            }
        }
    }
    /// <summary>dW[oc, ic, kh, kw] (=, or += when <paramref name="accumulate"/>) for the conv of input with kernel.</summary>
    public static void BackwardKernel(
        float[] input, int inputOffset, float[] gradOutput, int gradOutputOffset, float[] dest, int destOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outputHeight, int outputWidth)
    {
        int paddedH = height + 2 * padH, paddedW = width + 2 * padW;
        int inBlocks = inChannels / Block, outBlocks = outChannels / Block;
        int positions = outputHeight * outputWidth;
        int planeIn = paddedH * paddedW * Block;
        var pool = ArrayPool<float>.Shared;
        var packedInput = pool.Rent(batch * inBlocks * planeIn);
        var packedGrad = pool.Rent(batch * outChannels * positions);
        var offsets = ArrayPool<int>.Shared.Rent(batch * positions);
        try
        {
            PackInput(input, inputOffset, packedInput, batch, inChannels, height, width, padH, padW, paddedH, paddedW);
            PackGradByBlock(gradOutput, gradOutputOffset, packedGrad, batch, outChannels, positions);
            for (int b = 0; b < batch; b++)
                for (int q = 0; q < positions; q++)
                    offsets[b * positions + q] = b * inBlocks * planeIn
                        + ((q / outputWidth) * strideH * paddedW + (q % outputWidth) * strideW) * Block;

            CpuParallelSettings.ParallelForOrSerial(0, outBlocks * inBlocks,
                (long)outChannels * inChannels * kernelHeight * kernelWidth * batch * positions,
                task => BackwardKernelTile(packedInput, packedGrad, offsets, dest, destOffset, accumulate,
                    task / inBlocks, task % inBlocks, batch, inChannels, outBlocks, paddedW, planeIn,
                    kernelHeight, kernelWidth, dilationH, dilationW, positions),
                deterministicSafe: true);
        }
        finally
        {
            pool.Return(packedInput);
            pool.Return(packedGrad);
            ArrayPool<int>.Shared.Return(offsets);
        }
    }

    /// <summary>NCHW -> zero-padded [N][ceil(C/8)][H+2pH][W+2pW][8]; channels past C are zero.</summary>
    private static unsafe void PackInput(float[] source, int sourceOffset, float[] packed,
        int batch, int channels, int height, int width, int padH, int padW, int paddedH, int paddedW)
    {
        int blocks = (channels + Block - 1) / Block;
        int plane = paddedH * paddedW * Block;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * blocks * plane, task =>
        {
            int b = task / blocks, blk = task % blocks;
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                float* d = pd + (long)task * plane;
                new Span<float>(d, plane).Clear();
                int blockChannels = Math.Min(Block, channels - blk * Block);
                for (int c = 0; c < blockChannels; c++)
                {
                    float* s = ps + sourceOffset + ((long)b * channels + blk * Block + c) * height * width;
                    for (int y = 0; y < height; y++)
                    {
                        float* drow = d + ((y + padH) * paddedW + padW) * Block + c;
                        float* srow = s + y * width;
                        for (int x = 0; x < width; x++) drow[x * Block] = srow[x];
                    }
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>NCHW gradient -> [N][C/8][P][8].</summary>
    private static unsafe void PackGradByBlock(float[] source, int sourceOffset, float[] packed, int batch, int channels, int positions)
    {
        int blocks = channels / Block;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * channels * positions, task =>
        {
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                float* d = pd + (long)task * positions * Block;
                float* s = ps + sourceOffset + (long)task * Block * positions;
                for (int lane = 0; lane < Block; lane++)
                {
                    float* sl = s + lane * positions;
                    for (int q = 0; q < positions; q++) d[q * Block + lane] = sl[q];
                }
            }
        }, deterministicSafe: true);
    }

    /// <summary>
    /// OIHW -> [O/8][I/8][kH][kW][8 in][8 out]. With <paramref name="transposeAndFlip"/> the result is the packed
    /// kernel of the transposed conv: its outputs are the original inputs, and the taps are reversed.
    /// </summary>
    private static unsafe void PackKernel(float[] source, int sourceOffset, float[] packed,
        int outChannels, int inChannels, int kernelHeight, int kernelWidth, bool transposeAndFlip)
    {
        int taps = kernelHeight * kernelWidth;
        int packedOut = transposeAndFlip ? inChannels : outChannels;
        int packedIn = transposeAndFlip ? outChannels : inChannels;
        int packedInBlocks = (packedIn + Block - 1) / Block;
        CpuParallelSettings.ParallelForOrSerial(0, packedOut / Block, (long)outChannels * inChannels * taps, ob =>
        {
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                float* d0 = pd + (long)ob * packedInBlocks * taps * Block * Block;
                // A partial last input block keeps zero weights for its missing channels.
                if (packedIn % Block != 0) new Span<float>(d0, packedInBlocks * taps * Block * Block).Clear();
                for (int lane = 0; lane < Block; lane++)
                {
                    int po = ob * Block + lane;
                    for (int pi = 0; pi < packedIn; pi++)
                    {
                        float* d = d0 + (long)(pi / Block) * taps * Block * Block + (pi % Block) * Block + lane;
                        if (transposeAndFlip)
                        {
                            float* s = ps + sourceOffset + ((long)pi * inChannels + po) * taps;
                            for (int t = 0; t < taps; t++) d[t * Block * Block] = s[taps - 1 - t];
                        }
                        else
                        {
                            float* s = ps + sourceOffset + ((long)po * inChannels + pi) * taps;
                            for (int t = 0; t < taps; t++) d[t * Block * Block] = s[t];
                        }
                    }
                }
            }
        }, deterministicSafe: true);
    }

    private static void ForwardPacked(float[] packedInput, float[] packedKernel, float[] output, int outputOffset, bool accumulate,
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
            task =>
            {
                int tile = task / tasksPerTile, part = task % tasksPerTile;
                int first = part * chunksPerTask * PositionUnroll;
                int last = Math.Min(totalPositions, first + chunksPerTask * PositionUnroll);
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
            for (int q = first; q < last; q += PositionUnroll)
            {
                int count = Math.Min(PositionUnroll, last - q);
                // A short final chunk repeats its last position; only the real positions are stored.
                float* x0 = px + g.InputOrigin(q);
                float* x1 = px + g.InputOrigin(count > 1 ? q + 1 : q);
                float* x2 = px + g.InputOrigin(count > 2 ? q + 2 : q);
                var a00 = Vector256<float>.Zero; var a01 = a00; var a02 = a00; var a03 = a00;
                var a10 = a00; var a11 = a00; var a12 = a00; var a13 = a00;
                var a20 = a00; var a21 = a00; var a22 = a00; var a23 = a00;
                float* w = tileKernel;
                for (int cb = 0; cb < inBlocks; cb++)
                {
                    int blockOffset = cb * planeIn;
                    for (int y = 0; y < kernelHeight; y++)
                    {
                        int rowOffset = blockOffset + y * rowStep;
                        for (int x = 0; x < kernelWidth; x++)
                        {
                            int tapOffset = rowOffset + x * colStep;
                            float* s0 = x0 + tapOffset, s1 = x1 + tapOffset, s2 = x2 + tapOffset;
                            // The tap's 8 input channels, two per iteration, addressed by pointer bumps.
                            for (int c = 0; c < Block; c += 2)
                            {
                                var b0 = Vector256.Create(s0[0]); var b1 = Vector256.Create(s1[0]); var b2 = Vector256.Create(s2[0]);
                                var wv = Avx.LoadVector256(w);
                                a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                                wv = Avx.LoadVector256(w + blockStride);
                                a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                                wv = Avx.LoadVector256(w + 2 * blockStride);
                                a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                                wv = Avx.LoadVector256(w + 3 * blockStride);
                                a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);

                                b0 = Vector256.Create(s0[1]); b1 = Vector256.Create(s1[1]); b2 = Vector256.Create(s2[1]);
                                wv = Avx.LoadVector256(w + Block);
                                a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                                wv = Avx.LoadVector256(w + Block + blockStride);
                                a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                                wv = Avx.LoadVector256(w + Block + 2 * blockStride);
                                a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                                wv = Avx.LoadVector256(w + Block + 3 * blockStride);
                                a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);

                                s0 += 2; s1 += 2; s2 += 2; w += 2 * Block;
                            }
                        }
                    }
                }
                int channelBase = tile * OutputBlocksPerTile * Block;
                StorePosition(po + outputOffset, accumulate, g, channelBase, q, a00, a01, a02, a03);
                if (count > 1) StorePosition(po + outputOffset, accumulate, g, channelBase, q + 1, a10, a11, a12, a13);
                if (count > 2) StorePosition(po + outputOffset, accumulate, g, channelBase, q + 2, a20, a21, a22, a23);
            }
        }
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StorePosition(float* output, bool accumulate, in ForwardGeometry g, int channelBase, int q,
        Vector256<float> v0, Vector256<float> v1, Vector256<float> v2, Vector256<float> v3)
    {
        int index = g.OutputIndex(q, out int b);
        float* d = output + ((long)b * g.OutChannels + channelBase) * g.OutPlane + index;
        StoreLanes(d, g.OutPlane, accumulate, v0);
        StoreLanes(d + (long)Block * g.OutPlane, g.OutPlane, accumulate, v1);
        StoreLanes(d + (long)2 * Block * g.OutPlane, g.OutPlane, accumulate, v2);
        StoreLanes(d + (long)3 * Block * g.OutPlane, g.OutPlane, accumulate, v3);
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreLanes(float* d, int stride, bool accumulate, Vector256<float> v)
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
    private static unsafe void BackwardKernelTile(float[] packedInput, float[] packedGrad, int[] offsets, float[] dest, int destOffset,
        bool accumulate, int ob, int ib, int batch, int inChannels, int outBlocks, int paddedW, int planeIn,
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
            for (int b = 0; b < batch; b++)
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
