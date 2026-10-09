using System;
using System.Buffers;
using AiDotNet.Tensors.Helpers;
#if NET5_0_OR_GREATER
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
#endif

namespace AiDotNet.Tensors.Engines.Simd;

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

#if NET5_0_OR_GREATER
    public static bool IsSupported => Avx2.IsSupported && Fma.IsSupported;
#else
    public static bool IsSupported => false;
#endif

    /// <summary>
    /// Whether the forward conv takes the direct kernel. Measured on a 3950X against the im2col routes (batch 8):
    /// it wins on stride-1 planes from 4x4 to 32x32 (256 channels on 8x8: 1.4 ms against 2.9) and on strided convs
    /// with a short reduction (64->128 3x3 stride 2 on 32x32: 0.8 ms against 3.2). A strided conv with a long
    /// reduction (256->512 3x3 stride 2, K = 2304) stays on the batched GEMM, which ran it in 0.76 ms against 1.2.
    /// </summary>
    public static bool ShouldUseForward(int batch, int inChannels, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int outputHeight, int outputWidth)
        => IsSupported
           && batch > 1
           && inChannels % Block == 0
           && outChannels % (Block * OutputBlocksPerTile) == 0
           && outputHeight * outputWidth > 0
           && !((strideH > 1 || strideW > 1) && inChannels * kernelHeight * kernelWidth > 1152);

    /// <summary>
    /// Whether a stride-1 input gradient takes the direct kernel (as a forward conv of the output gradient with the
    /// flipped, transposed kernel). Measured winning or tied on every stride-1 ResNet stage; 256 channels on 8x8
    /// ran 1.4 ms against 3.7.
    /// </summary>
    public static bool ShouldUseBackwardInput(int batch, int inChannels, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW)
        => IsSupported
           && batch > 1
           && strideH == 1 && strideW == 1 && dilationH == 1 && dilationW == 1
           && padH <= kernelHeight - 1 && padW <= kernelWidth - 1
           && outChannels % Block == 0
           && inChannels % (Block * OutputBlocksPerTile) == 0;

    /// <summary>
    /// Whether the kernel gradient takes the direct kernel. Measured winning or tied on every ResNet conv,
    /// strided and 1x1 included (512 channels on 4x4: 1.3 ms against 3.2).
    /// </summary>
    public static bool ShouldUseBackwardKernel(int inChannels, int outChannels)
        => IsSupported
           && inChannels % Block == 0
           && outChannels % Block == 0
           && (inChannels / Block) * (outChannels / Block) >= MinBackwardKernelTasks;

#if NET5_0_OR_GREATER
    /// <summary>output[n, oc, oh, ow] (=, or += when <paramref name="accumulate"/>) conv(input, kernel), NCHW / OIHW.</summary>
    public static void Forward(
        float[] input, int inputOffset, float[] kernel, int kernelOffset, float[] output, int outputOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int padH, int padW, int dilationH, int dilationW, int outputHeight, int outputWidth)
    {
        int paddedH = height + 2 * padH, paddedW = width + 2 * padW;
        int inBlocks = inChannels / Block;
        var pool = ArrayPool<float>.Shared;
        var packedInput = pool.Rent(batch * inBlocks * paddedH * paddedW * Block);
        var packedKernel = pool.Rent(outChannels * inChannels * kernelHeight * kernelWidth);
        try
        {
            PackInput(input, inputOffset, packedInput, batch, inChannels, height, width, padH, padW, paddedH, paddedW);
            PackKernel(kernel, kernelOffset, packedKernel, outChannels, inChannels, kernelHeight, kernelWidth, transposeAndFlip: false);
            ForwardPacked(packedInput, packedKernel, output, outputOffset, accumulate,
                batch, inBlocks, paddedH, paddedW, outChannels, kernelHeight, kernelWidth,
                strideH, strideW, dilationH, dilationW, outputHeight, outputWidth);
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
        int padH, int padW, int outputHeight, int outputWidth)
    {
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
                batch, gradBlocks, paddedH, paddedW, inChannels, kernelHeight, kernelWidth,
                1, 1, 1, 1, height, width);
        }
        finally
        {
            pool.Return(packedGrad);
            pool.Return(packedKernel);
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

    /// <summary>NCHW -> zero-padded [N][C/8][H+2pH][W+2pW][8].</summary>
    private static unsafe void PackInput(float[] source, int sourceOffset, float[] packed,
        int batch, int channels, int height, int width, int padH, int padW, int paddedH, int paddedW)
    {
        int blocks = channels / Block;
        int plane = paddedH * paddedW * Block;
        CpuParallelSettings.ParallelForOrSerial(0, batch * blocks, (long)batch * blocks * plane, task =>
        {
            int b = task / blocks, blk = task % blocks;
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                float* d = pd + (long)task * plane;
                new Span<float>(d, plane).Clear();
                for (int c = 0; c < Block; c++)
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
        int packedInBlocks = packedIn / Block;
        CpuParallelSettings.ParallelForOrSerial(0, packedOut / Block, (long)outChannels * inChannels * taps, ob =>
        {
            fixed (float* ps = source)
            fixed (float* pd = packed)
            {
                float* d0 = pd + (long)ob * packedInBlocks * taps * Block * Block;
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
        int batch, int inBlocks, int paddedH, int paddedW, int outChannels, int kernelHeight, int kernelWidth,
        int strideH, int strideW, int dilationH, int dilationW, int outputHeight, int outputWidth)
    {
        int positions = outputHeight * outputWidth;
        int totalPositions = batch * positions;
        int tiles = outChannels / (Block * OutputBlocksPerTile);
        int chunks = (totalPositions + PositionUnroll - 1) / PositionUnroll;
        int chunksPerTask = Math.Max(1, (int)(((long)chunks * tiles + TargetForwardTasks - 1) / TargetForwardTasks));
        int tasksPerTile = (chunks + chunksPerTask - 1) / chunksPerTask;
        var geometry = new ForwardGeometry(inBlocks, paddedH * paddedW * Block, paddedW, outChannels, kernelHeight, kernelWidth,
            strideH, strideW, dilationH, dilationW, outputWidth, positions, totalPositions);
        CpuParallelSettings.ParallelForOrSerial(0, tiles * tasksPerTile,
            (long)totalPositions * outChannels * inBlocks * Block * kernelHeight * kernelWidth,
            task =>
            {
                int tile = task / tasksPerTile, part = task % tasksPerTile;
                int first = part * chunksPerTask * PositionUnroll;
                int last = Math.Min(totalPositions, first + chunksPerTask * PositionUnroll);
                ForwardTile(packedInput, packedKernel, output, outputOffset, accumulate, geometry, tile, first, last);
            },
            deterministicSafe: true);
    }

    private readonly struct ForwardGeometry
    {
        public ForwardGeometry(int inBlocks, int planeIn, int paddedW, int outChannels, int kernelHeight, int kernelWidth,
            int strideH, int strideW, int dilationH, int dilationW, int outputWidth, int positions, int totalPositions)
        {
            InBlocks = inBlocks; PlaneIn = planeIn; PaddedW = paddedW; OutChannels = outChannels;
            KernelHeight = kernelHeight; KernelWidth = kernelWidth; StrideH = strideH; StrideW = strideW;
            DilationH = dilationH; DilationW = dilationW; OutputWidth = outputWidth; Positions = positions; TotalPositions = totalPositions;
        }

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
        public int OutputWidth { get; }
        public int Positions { get; }
        public int TotalPositions { get; }

        /// <summary>Offset of flattened output position q's receptive-field origin in the packed input.</summary>
        public int InputOrigin(int q)
        {
            int b = q / Positions, r = q % Positions;
            return b * InBlocks * PlaneIn + ((r / OutputWidth) * StrideH * PaddedW + (r % OutputWidth) * StrideW) * Block;
        }
    }

    private static unsafe void ForwardTile(float[] packedInput, float[] packedKernel, float[] output, int outputOffset, bool accumulate,
        in ForwardGeometry g, int tile, int first, int last)
    {
        int taps = g.KernelHeight * g.KernelWidth;
        long blockStride = (long)g.InBlocks * taps * Block * Block;
        fixed (float* px = packedInput)
        fixed (float* pw = packedKernel)
        fixed (float* po = output)
        {
            float* tileKernel = pw + tile * OutputBlocksPerTile * blockStride;
            for (int q = first; q < last; q += PositionUnroll)
            {
                int count = Math.Min(PositionUnroll, last - q);
                // A short final chunk repeats its last position; only the real positions are stored.
                int o0 = g.InputOrigin(q);
                int o1 = g.InputOrigin(count > 1 ? q + 1 : q);
                int o2 = g.InputOrigin(count > 2 ? q + 2 : q);
                var a00 = Vector256<float>.Zero; var a01 = a00; var a02 = a00; var a03 = a00;
                var a10 = a00; var a11 = a00; var a12 = a00; var a13 = a00;
                var a20 = a00; var a21 = a00; var a22 = a00; var a23 = a00;
                for (int cb = 0; cb < g.InBlocks; cb++)
                {
                    float* xc = px + (long)cb * g.PlaneIn;
                    float* wc = tileKernel + (long)cb * taps * Block * Block;
                    for (int y = 0; y < g.KernelHeight; y++)
                    {
                        for (int x = 0; x < g.KernelWidth; x++)
                        {
                            float* xs = xc + (y * g.DilationH * g.PaddedW + x * g.DilationW) * Block;
                            float* ws = wc + (y * g.KernelWidth + x) * Block * Block;
                            for (int c = 0; c < Block; c++)
                            {
                                var b0 = Vector256.Create(xs[o0 + c]);
                                var b1 = Vector256.Create(xs[o1 + c]);
                                var b2 = Vector256.Create(xs[o2 + c]);
                                float* wl = ws + c * Block;
                                var wv = Avx.LoadVector256(wl);
                                a00 = Fma.MultiplyAdd(b0, wv, a00); a10 = Fma.MultiplyAdd(b1, wv, a10); a20 = Fma.MultiplyAdd(b2, wv, a20);
                                wv = Avx.LoadVector256(wl + blockStride);
                                a01 = Fma.MultiplyAdd(b0, wv, a01); a11 = Fma.MultiplyAdd(b1, wv, a11); a21 = Fma.MultiplyAdd(b2, wv, a21);
                                wv = Avx.LoadVector256(wl + 2 * blockStride);
                                a02 = Fma.MultiplyAdd(b0, wv, a02); a12 = Fma.MultiplyAdd(b1, wv, a12); a22 = Fma.MultiplyAdd(b2, wv, a22);
                                wv = Avx.LoadVector256(wl + 3 * blockStride);
                                a03 = Fma.MultiplyAdd(b0, wv, a03); a13 = Fma.MultiplyAdd(b1, wv, a13); a23 = Fma.MultiplyAdd(b2, wv, a23);
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
        int b = q / g.Positions, r = q % g.Positions;
        float* d = output + ((long)b * g.OutChannels + channelBase) * g.Positions + r;
        StoreLanes(d, g.Positions, accumulate, v0);
        StoreLanes(d + (long)Block * g.Positions, g.Positions, accumulate, v1);
        StoreLanes(d + (long)2 * Block * g.Positions, g.Positions, accumulate, v2);
        StoreLanes(d + (long)3 * Block * g.Positions, g.Positions, accumulate, v3);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreLanes(float* d, int stride, bool accumulate, Vector256<float> v)
    {
        if (accumulate)
            for (int lane = 0; lane < Block; lane++) d[(long)lane * stride] += v.GetElement(lane);
        else
            for (int lane = 0; lane < Block; lane++) d[(long)lane * stride] = v.GetElement(lane);
    }

    /// <summary>
    /// One (output block, input block) pair of dW. For each kernel row the tile is 3 kernel columns x 4 input
    /// channels = 12 accumulators, each a vector over the block's 8 output channels; per position that is one
    /// gradient load, 12 broadcasts and 12 FMAs (oneDNN's avx2 backward-weights blocking).
    /// </summary>
    private static unsafe void BackwardKernelTile(float[] packedInput, float[] packedGrad, int[] offsets, float[] dest, int destOffset,
        bool accumulate, int ob, int ib, int batch, int inChannels, int outBlocks, int paddedW, int planeIn,
        int kernelHeight, int kernelWidth, int dilationH, int dilationW, int positions)
    {
        int total = batch * positions;
        fixed (float* px = packedInput)
        fixed (float* pg = packedGrad)
        fixed (int* pOff = offsets)
        fixed (float* pd = dest)
        {
            float* xb = px + (long)ib * planeIn;
            for (int y = 0; y < kernelHeight; y++)
            {
                for (int x0 = 0; x0 < kernelWidth; x0 += 3)
                {
                    int columns = Math.Min(3, kernelWidth - x0);
                    int step = dilationW * Block;
                    for (int c0 = 0; c0 < Block; c0 += 4)
                    {
                        var a00 = Vector256<float>.Zero; var a01 = a00; var a02 = a00; var a03 = a00;
                        var a10 = a00; var a11 = a00; var a12 = a00; var a13 = a00;
                        var a20 = a00; var a21 = a00; var a22 = a00; var a23 = a00;
                        int tapOffset = (y * dilationH * paddedW + x0 * dilationW) * Block + c0;
                        for (int b = 0; b < batch; b++)
                        {
                            float* gq = pg + (long)(b * outBlocks + ob) * positions * Block;
                            int* oq = pOff + b * positions;
                            for (int q = 0; q < positions; q++)
                            {
                                var gv = Avx.LoadVector256(gq + q * Block);
                                float* xs = xb + oq[q] + tapOffset;
                                a00 = Fma.MultiplyAdd(Vector256.Create(xs[0]), gv, a00);
                                a01 = Fma.MultiplyAdd(Vector256.Create(xs[1]), gv, a01);
                                a02 = Fma.MultiplyAdd(Vector256.Create(xs[2]), gv, a02);
                                a03 = Fma.MultiplyAdd(Vector256.Create(xs[3]), gv, a03);
                                if (columns > 1)
                                {
                                    a10 = Fma.MultiplyAdd(Vector256.Create(xs[step]), gv, a10);
                                    a11 = Fma.MultiplyAdd(Vector256.Create(xs[step + 1]), gv, a11);
                                    a12 = Fma.MultiplyAdd(Vector256.Create(xs[step + 2]), gv, a12);
                                    a13 = Fma.MultiplyAdd(Vector256.Create(xs[step + 3]), gv, a13);
                                    if (columns > 2)
                                    {
                                        a20 = Fma.MultiplyAdd(Vector256.Create(xs[2 * step]), gv, a20);
                                        a21 = Fma.MultiplyAdd(Vector256.Create(xs[2 * step + 1]), gv, a21);
                                        a22 = Fma.MultiplyAdd(Vector256.Create(xs[2 * step + 2]), gv, a22);
                                        a23 = Fma.MultiplyAdd(Vector256.Create(xs[2 * step + 3]), gv, a23);
                                    }
                                }
                            }
                        }
                        float* d = pd + destOffset;
                        int i0 = ib * Block + c0;
                        StoreKernelColumn(d, accumulate, ob, i0, x0, y, inChannels, kernelHeight, kernelWidth, a00, a01, a02, a03);
                        if (columns > 1) StoreKernelColumn(d, accumulate, ob, i0, x0 + 1, y, inChannels, kernelHeight, kernelWidth, a10, a11, a12, a13);
                        if (columns > 2) StoreKernelColumn(d, accumulate, ob, i0, x0 + 2, y, inChannels, kernelHeight, kernelWidth, a20, a21, a22, a23);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void StoreKernelColumn(float* d, bool accumulate, int ob, int i0, int x, int y,
        int inChannels, int kernelHeight, int kernelWidth,
        Vector256<float> c0, Vector256<float> c1, Vector256<float> c2, Vector256<float> c3)
    {
        int taps = kernelHeight * kernelWidth;
        float* tap = d + y * kernelWidth + x;
        // Lane l of input channel i0+c is dW[ob*8+l, i0+c, y, x].
        long laneStride = (long)inChannels * taps;
        float* o = tap + ((long)ob * Block * inChannels + i0) * taps;
        StoreLanes(o, (int)laneStride, accumulate, c0);
        StoreLanes(o + taps, (int)laneStride, accumulate, c1);
        StoreLanes(o + 2 * taps, (int)laneStride, accumulate, c2);
        StoreLanes(o + 3 * taps, (int)laneStride, accumulate, c3);
    }
#endif
}
