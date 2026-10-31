// Copyright (c) CrowdTrainer LLC. All rights reserved.
#if NET5_0_OR_GREATER
using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// The three passes of an unpadded 1x1 convolution at any stride (ResNet's downsample shortcuts, every pointwise
/// convolution), NCHW / OIHW, as GEMMs. With P = Ho*Wo output pixels and xs the input sampled at the stride
/// (xs[n, i, p] = x[n, i, (p / Wo) * strideH, (p % Wo) * strideW]):
/// <list type="bullet">
/// <item>forward: y[n] (O x P) = W (O x I) . xs[n] (I x P), one shared-weight GEMM per image;</item>
/// <item>input gradient: dxs[n] (I x P) = W^T (I x O) . dy[n] (O x P), scattered back to the stride grid (the other
/// input pixels receive no gradient);</item>
/// <item>weight gradient: dW (O x I) = dy' (O x N*P) . xs' (N*P x I), one GEMM over the whole batch (dy' is dy with
/// the image and channel axes swapped; xs' is xs pixel-major).</item>
/// </list>
/// </summary>
/// <remarks>
/// The direct kernel declines 1x1 shapes and the strided ones fell to the generic routes: on resnet18 b8's 64->128
/// 32x32 stride-2 shortcut (one thread) the forward took 0.87 ms, the input gradient 1.03 ms and the weight gradient
/// 1.67 ms, against PyTorch's 0.44, 0.81 and 0.40 ms; each pass is a 33.5 MFLOP GEMM. Every output element is summed
/// in one fixed order whatever the thread count (the GEMMs used are deterministic).
/// </remarks>
internal static class Conv1x1Gemm
{
    private const MethodImplOptions Hot = MethodImplOptions.AggressiveOptimization;

    /// <summary>Whether this class runs the pass: a 1x1, unpadded, undilated float convolution over a batch.</summary>
    public static bool Applies(int batch, int kernelHeight, int kernelWidth, int padH, int padW, int dilationH, int dilationW)
        => s_enabled && batch >= 2 && kernelHeight == 1 && kernelWidth == 1 && padH == 0 && padW == 0
            && dilationH == 1 && dilationW == 1 && Avx2.IsSupported && Fma.IsSupported;

    /// <summary>Switch for A/B measurement and tests.</summary>
    internal static bool s_enabled = true;

    /// <summary>y = conv1x1(x, w) at the stride (overwrites y).</summary>
    [MethodImpl(Hot)]
    public static void Forward(float[] x, int xOffset, float[] w, int wOffset, float[] y, int yOffset,
        int batch, int inChannels, int height, int width, int outChannels, int strideH, int strideW, int outputHeight, int outputWidth)
    {
        int pixels = outputHeight * outputWidth;
        if (strideH == 1 && strideW == 1)
        {
            SharedWeightGemm(w, wOffset, x, xOffset, y, yOffset, batch, inChannels, pixels, outChannels);
            return;
        }
        var pool = ArrayPool<float>.Shared;
        var sampled = pool.Rent(batch * inChannels * pixels);
        try
        {
            Subsample(x, xOffset, sampled, batch * inChannels, height, width, strideH, strideW, outputHeight, outputWidth);
            SharedWeightGemm(w, wOffset, sampled, 0, y, yOffset, batch, inChannels, pixels, outChannels);
        }
        finally
        {
            pool.Return(sampled);
        }
    }

    /// <summary>dx (=, or += when <paramref name="accumulate"/>) for y = conv1x1(x, w) at the stride.</summary>
    [MethodImpl(Hot)]
    public static void BackwardInput(float[] dy, int dyOffset, float[] w, int wOffset, float[] dx, int dxOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int strideH, int strideW, int outputHeight, int outputWidth)
    {
        int pixels = outputHeight * outputWidth;
        bool strided = strideH > 1 || strideW > 1;
        var pool = ArrayPool<float>.Shared;
        var transposed = pool.Rent(inChannels * outChannels);
        var staged = strided || accumulate ? pool.Rent(batch * inChannels * pixels) : null;
        try
        {
            SimdTranspose.Transpose(new ReadOnlySpan<float>(w, wOffset, outChannels * inChannels),
                new Span<float>(transposed, 0, inChannels * outChannels), outChannels, inChannels);
            if (staged is null)
            {
                SharedWeightGemm(transposed, 0, dy, dyOffset, dx, dxOffset, batch, outChannels, pixels, inChannels);
                return;
            }
            SharedWeightGemm(transposed, 0, dy, dyOffset, staged, 0, batch, outChannels, pixels, inChannels);
            Scatter(staged, dx, dxOffset, accumulate, batch * inChannels, height, width, strideH, strideW, outputHeight, outputWidth);
        }
        finally
        {
            pool.Return(transposed);
            if (staged is not null) pool.Return(staged);
        }
    }

    /// <summary>dW (=, or += when <paramref name="accumulate"/>) for y = conv1x1(x, w) at the stride.</summary>
    [MethodImpl(Hot)]
    public static void BackwardKernel(float[] x, int xOffset, float[] dy, int dyOffset, float[] dw, int dwOffset, bool accumulate,
        int batch, int inChannels, int height, int width, int outChannels, int strideH, int strideW, int outputHeight, int outputWidth)
    {
        int pixels = outputHeight * outputWidth;
        int depth = batch * pixels;
        var pool = ArrayPool<float>.Shared;
        var gradRows = pool.Rent(outChannels * depth);
        var inputRows = pool.Rent(depth * inChannels);
        try
        {
            // dy' [O][N*P]: image n's plane of channel o is the run at o*N*P + n*P.
            CpuParallelSettings.ParallelForOrSerial(0, outChannels, (long)outChannels * depth, [MethodImpl(Hot)] (o) =>
            {
                for (int n = 0; n < batch; n++)
                    Array.Copy(dy, dyOffset + ((long)n * outChannels + o) * pixels, gradRows, (long)o * depth + (long)n * pixels, pixels);
            }, deterministicSafe: true);
            PixelMajor(x, xOffset, inputRows, batch, inChannels, height, width, strideH, strideW, outputHeight, outputWidth);
            var c = new Span<float>(dw, dwOffset, outChannels * inChannels);
            var a = new ReadOnlySpan<float>(gradRows, 0, outChannels * depth);
            var b = new ReadOnlySpan<float>(inputRows, 0, depth * inChannels);
            if (accumulate) SimdGemm.SgemmAdd(a, depth, false, b, inChannels, false, c, outChannels, depth, inChannels);
            else SimdGemm.Sgemm(a, depth, false, b, inChannels, false, c, outChannels, depth, inChannels);
        }
        finally
        {
            pool.Return(gradRows);
            pool.Return(inputRows);
        }
    }

    // C[n] (m x pixels) = A (m x k) . B[n] (k x pixels) for each of the batch's images: the per-image GEMM route of the
    // stride-1 1x1 forward (CpuEngine.TryConv1x1PerImageGemm), with the same shape choice.
    private static void SharedWeightGemm(float[] a, int aOffset, float[] b, int bOffset, float[] c, int cOffset,
        int batch, int k, int pixels, int m)
    {
        int bStride = k * pixels, cStride = m * pixels;
        if (pixels < SharedColumnsMaxPixels)
        {
            SimdGemm.SgemmSharedAColumns(a, aOffset, b, bOffset, bStride, c, cOffset, cStride, m, k, pixels, batch);
            return;
        }
        if (SimdGemm.PrefersSharedA(m, k, pixels))
        {
            SimdGemm.SgemmSharedA(a, aOffset, b, bOffset, bStride, c, cOffset, cStride, m, k, pixels, batch);
            return;
        }
        CpuParallelSettings.ParallelForOrSerial(0, batch, 2L * batch * m * k * pixels, n =>
            SimdGemm.SgemmSequential(
                new ReadOnlySpan<float>(a, aOffset, m * k),
                new ReadOnlySpan<float>(b, bOffset + n * bStride, bStride),
                new Span<float>(c, cOffset + n * cStride, cStride),
                m, k, pixels),
            deterministicSafe: true);
    }

    private const int SharedColumnsMaxPixels = 64;

    // xs[plane, oy, ox] = x[plane, oy*strideH, ox*strideW] for every (image, channel) plane.
    [MethodImpl(Hot)]
    private static unsafe void Subsample(float[] x, int xOffset, float[] sampled, int planes, int height, int width,
        int strideH, int strideW, int outputHeight, int outputWidth)
    {
        int pixels = outputHeight * outputWidth;
        CpuParallelSettings.ParallelForOrSerial(0, planes, (long)planes * pixels, [MethodImpl(Hot)] (plane) =>
        {
            fixed (float* ps = x)
            fixed (float* pd = sampled)
            {
                float* s = ps + xOffset + (long)plane * height * width;
                float* d = pd + (long)plane * pixels;
                for (int oy = 0; oy < outputHeight; oy++)
                    SampleRow(s + (long)oy * strideH * width, d + oy * outputWidth, outputWidth, strideW);
            }
        }, deterministicSafe: true);
    }

    // d[i] = s[i * stride] for i < count; stride 2 (every ResNet shortcut) picks the even lanes of 16-float loads.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void SampleRow(float* s, float* d, int count, int stride)
    {
        int i = 0;
        if (stride == 1)
        {
            Buffer.MemoryCopy(s, d, count * sizeof(float), count * sizeof(float));
            return;
        }
        if (stride == 2)
        {
            // The last vector reads s[2i .. 2i+15]; keep it inside the row (2*count - 1 floats are in bounds).
            for (; i + 8 <= count && 2 * i + 16 <= 2 * count - 1; i += 8)
            {
                var lo = Avx.LoadVector256(s + 2 * i);
                var hi = Avx.LoadVector256(s + 2 * i + 8);
                // Shuffle picks lanes 0,2 of each 128-bit half of lo then hi: [l0 l2 h0 h2 | l4 l6 h4 h6]; the
                // 64-bit permute restores the order [l0 l2 l4 l6 h0 h2 h4 h6].
                var even = Avx.Shuffle(lo, hi, 0x88);
                Avx.Store(d + i, Avx2.Permute4x64(even.AsDouble(), 0xD8).AsSingle());
            }
        }
        for (; i < count; i++) d[i] = s[(long)i * stride];
    }

    // dx[plane] (=, or +=) staged[plane] on the stride grid; with overwrite, the pixels off the grid are zeroed.
    [MethodImpl(Hot)]
    private static unsafe void Scatter(float[] staged, float[] dx, int dxOffset, bool accumulate, int planes, int height, int width,
        int strideH, int strideW, int outputHeight, int outputWidth)
    {
        int pixels = outputHeight * outputWidth;
        bool strided = strideH > 1 || strideW > 1;
        CpuParallelSettings.ParallelForOrSerial(0, planes, (long)planes * height * width, [MethodImpl(Hot)] (plane) =>
        {
            fixed (float* ps = staged)
            fixed (float* pd = dx)
            {
                float* s = ps + (long)plane * pixels;
                float* d = pd + dxOffset + (long)plane * height * width;
                if (!accumulate && strided) new Span<float>(d, height * width).Clear();
                for (int oy = 0; oy < outputHeight; oy++)
                {
                    float* drow = d + (long)oy * strideH * width;
                    float* srow = s + oy * outputWidth;
                    if (strideW == 1)
                    {
                        int ox = 0;
                        for (; ox + 8 <= outputWidth; ox += 8)
                        {
                            var v = Avx.LoadVector256(srow + ox);
                            Avx.Store(drow + ox, accumulate ? Avx.Add(Avx.LoadVector256(drow + ox), v) : v);
                        }
                        for (; ox < outputWidth; ox++) drow[ox] = accumulate ? drow[ox] + srow[ox] : srow[ox];
                    }
                    else if (accumulate)
                    {
                        for (int ox = 0; ox < outputWidth; ox++) drow[(long)ox * strideW] += srow[ox];
                    }
                    else
                    {
                        for (int ox = 0; ox < outputWidth; ox++) drow[(long)ox * strideW] = srow[ox];
                    }
                }
            }
        }, deterministicSafe: true);
    }

    // xs' [N*P][I]: row n*P + p holds the I channels of image n's sampled pixel p (8x8 register transposes per block of
    // eight channels and eight pixels; the sampled rows are gathered into a stack tile first when strided).
    [MethodImpl(Hot)]
    private static unsafe void PixelMajor(float[] x, int xOffset, float[] rows, int batch, int inChannels, int height, int width,
        int strideH, int strideW, int outputHeight, int outputWidth)
    {
        int pixels = outputHeight * outputWidth;
        int channelBlocks = (inChannels + 7) / 8;
        CpuParallelSettings.ParallelForOrSerial(0, batch * channelBlocks, (long)batch * inChannels * pixels, [MethodImpl(Hot)] (task) =>
        {
            int n = task / channelBlocks, i0 = (task % channelBlocks) * 8;
            int lanes = Math.Min(8, inChannels - i0);
            float* tile = stackalloc float[8 * 8];
            long* offset = stackalloc long[8];
            fixed (float* ps = x)
            fixed (float* pd = rows)
            {
                float* image = ps + xOffset + ((long)n * inChannels + i0) * height * width;
                float* dImage = pd + (long)n * pixels * inChannels + i0;
                for (int p0 = 0; p0 < pixels; p0 += 8)
                {
                    int count = Math.Min(8, pixels - p0);
                    // The chunk's sampled offsets, shared by its channels (a division per element was most of the gather).
                    for (int j = 0, row = p0 / outputWidth, col = p0 % outputWidth; j < count; j++)
                    {
                        offset[j] = (long)row * strideH * width + (long)col * strideW;
                        if (++col == outputWidth) { col = 0; row++; }
                    }
                    if (count == 8 && lanes == 8 && offset[7] - offset[0] == 7)
                    {
                        // Eight consecutive pixels of one row (stride 1): transpose straight from the channel planes.
                        Transpose8x8(image + offset[0], (long)height * width, dImage + (long)p0 * inChannels, inChannels);
                    }
                    else if (count == 8 && lanes == 8)
                    {
                        // Gather each channel's eight sampled pixels into the tile's row, then transpose the tile.
                        for (int c = 0; c < 8; c++)
                        {
                            float* plane = image + (long)c * height * width;
                            for (int j = 0; j < 8; j++) tile[c * 8 + j] = plane[offset[j]];
                        }
                        Transpose8x8(tile, 8, dImage + (long)p0 * inChannels, inChannels);
                    }
                    else
                    {
                        for (int c = 0; c < lanes; c++)
                        {
                            float* plane = image + (long)c * height * width;
                            for (int j = 0; j < count; j++) dImage[(long)(p0 + j) * inChannels + c] = plane[offset[j]];
                        }
                    }
                }
            }
        }, deterministicSafe: true);
    }

    // d[j * dStride + c] = s[c * sStride + j] for an 8x8 block.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Transpose8x8(float* s, long sStride, float* d, int dStride)
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
        Avx.Store(d, Avx.Permute2x128(u0, u4, 0x20));
        Avx.Store(d + dStride, Avx.Permute2x128(u1, u5, 0x20));
        Avx.Store(d + 2L * dStride, Avx.Permute2x128(u2, u6, 0x20));
        Avx.Store(d + 3L * dStride, Avx.Permute2x128(u3, u7, 0x20));
        Avx.Store(d + 4L * dStride, Avx.Permute2x128(u0, u4, 0x31));
        Avx.Store(d + 5L * dStride, Avx.Permute2x128(u1, u5, 0x31));
        Avx.Store(d + 6L * dStride, Avx.Permute2x128(u2, u6, 0x31));
        Avx.Store(d + 7L * dStride, Avx.Permute2x128(u3, u7, 0x31));
    }
}
#endif
