using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// CUDA max and average pool over output planes smaller than its 16x16 block runs one thread per output cell (maxpool2d_flat, avgpool2d_flat).
/// Pins that it keeps maxpool2d's results and winner indices through padding, overlapping strided windows, odd
/// sizes, ties (the first of equal maxima wins) and a 1x1 output, against the CPU rule written out here.
/// </summary>
public class CudaMaxPoolSmallPlaneTests
{
    [SkippableTheory]
    [InlineData(3, 5, 7, 7, 3, 3, 2, 2, 1, 1)]   // padded, overlapping 3x3/2 windows: 4x4 outputs
    [InlineData(2, 9, 4, 4, 2, 2, 2, 2, 0, 0)]   // VGG's 2x2/2 on a 4x4 plane: 2x2 outputs
    [InlineData(2, 9, 2, 2, 2, 2, 2, 2, 0, 0)]   // 1x1 output
    [InlineData(1, 3, 5, 31, 3, 2, 1, 2, 1, 0)]  // 31 wide but 5 tall: the flat route through a wide plane
    public void SmallPlane_MatchesTheCpuRule(int batch, int channels, int h, int w, int kh, int kw, int sh, int sw,
        int ph, int pw)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, backend.InitializationError ?? "CUDA backend unavailable");

        int oh = (h + 2 * ph - kh) / sh + 1, ow = (w + 2 * pw - kw) / sw + 1;
        int planes = batch * channels;
        // Few distinct values, so windows hold ties.
        var input = Enumerable.Range(0, planes * h * w).Select(i => (float)((i * 7 + 3) % 5 - 2)).ToArray();
        using var inBuf = backend.AllocateBuffer(input);
        using var outBuf = backend.AllocateBuffer(Enumerable.Repeat(123.25f, planes * oh * ow).ToArray());
        using var idxBuf = backend.AllocateIntBuffer(planes * oh * ow);

        backend.MaxPool2D(inBuf, outBuf, idxBuf, batch, channels, h, w, oh, ow, kh, kw, sh, sw, ph, pw);

        var output = backend.DownloadBuffer(outBuf);
        var indices = new int[planes * oh * ow];
        backend.DownloadIntBuffer(idxBuf, indices);
        for (int p = 0; p < planes; p++)
        for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
        {
            float best = float.NegativeInfinity;
            int at = Math.Min(Math.Max(y * sh - ph, 0), h - 1) * w + Math.Min(Math.Max(x * sw - pw, 0), w - 1);
            bool have = false;
            for (int ky = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++)
            {
                int iy = y * sh - ph + ky, ix = x * sw - pw + kx;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                float v = input[(p * h + iy) * w + ix];
                if (!have || v > best) { best = v; at = iy * w + ix; have = true; }
            }
            int o = (p * oh + y) * ow + x;
            Assert.Equal(best, output[o]);
            Assert.Equal(at, indices[o]);
        }
    }

    [SkippableTheory]
    [InlineData(3, 5, 7, 7, 3, 3, 2, 2, 1, 1, true)]
    [InlineData(3, 5, 7, 7, 3, 3, 2, 2, 1, 1, false)]
    [InlineData(2, 9, 8, 8, 2, 2, 2, 2, 0, 0, true)]    // DenseNet's 2x2/2 transition pool
    public void AvgPoolSmallPlane_MatchesTheCpuRule(int batch, int channels, int h, int w, int kh, int kw, int sh,
        int sw, int ph, int pw, bool countIncludePad)
    {
        Skip.IfNot(CudaBackend.IsCudaAvailable, "CUDA driver or NVRTC is unavailable.");
        using var backend = new CudaBackend();
        Skip.IfNot(backend.IsAvailable, backend.InitializationError ?? "CUDA backend unavailable");

        int oh = (h + 2 * ph - kh) / sh + 1, ow = (w + 2 * pw - kw) / sw + 1;
        int planes = batch * channels;
        var input = Enumerable.Range(0, planes * h * w).Select(i => (float)((i * 37 + 11) % 97 - 48) / 64f).ToArray();
        using var inBuf = backend.AllocateBuffer(input);
        using var outBuf = backend.AllocateBuffer(Enumerable.Repeat(123.25f, planes * oh * ow).ToArray());

        backend.AvgPool2D(inBuf, outBuf, batch, channels, h, w, oh, ow, kh, kw, sh, sw, ph, pw, countIncludePad);

        var output = backend.DownloadBuffer(outBuf);
        for (int p = 0; p < planes; p++)
        for (int y = 0; y < oh; y++)
        for (int x = 0; x < ow; x++)
        {
            float sum = 0f; int count = 0;
            for (int ky = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++)
            {
                int iy = y * sh - ph + ky, ix = x * sw - pw + kx;
                if (iy >= 0 && iy < h && ix >= 0 && ix < w) { sum += input[(p * h + iy) * w + ix]; count++; }
                else if (countIncludePad) count++;
            }
            float expected = sum / Math.Max(countIncludePad ? kh * kw : count, 1);
            Assert.True(Math.Abs(expected - output[(p * oh + y) * ow + x]) <= 1e-6f,
                $"cell {(p * oh + y) * ow + x}: expected {expected}, got {output[(p * oh + y) * ow + x]}");
        }
    }
}
