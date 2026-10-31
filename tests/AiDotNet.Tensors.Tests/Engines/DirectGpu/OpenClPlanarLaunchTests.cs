#if NET6_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The OpenCL per-pixel kernels launched as a stack of planes (<c>ExecutePlanes</c>): the work group is shaped to the
/// plane, so a 1 x 1 or 2 x 2 output map puts several planes in one group, and the last group runs past the final plane.
/// Every output must still be written once, from its own plane, on maps smaller than, equal to and larger than 8 x 8.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClPlanarLaunchTests
{
    public static TheoryData<int, int, int, int, int, int, int> Pools => new()
    {
        // batch, channels, inH, inW, kernel, stride, pad
        { 3, 7, 2, 2, 2, 2, 0 },     // 1 x 1 output: 64 planes a group, 21 planes
        { 2, 5, 4, 4, 2, 2, 0 },     // 2 x 2
        { 1, 3, 3, 5, 3, 1, 1 },     // 3 x 5: a 4 x 8 group
        { 2, 3, 17, 9, 3, 2, 1 },    // 9 x 5
        { 2, 2, 20, 18, 2, 2, 0 },   // 10 x 9: two groups across each axis
    };

    [SkippableTheory]
    [MemberData(nameof(Pools))]
    public void OpenCl_MaxPool2D_MatchesReference(int batch, int channels, int inH, int inW, int k, int s, int p)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        int outH = (inH + 2 * p - k) / s + 1, outW = (inW + 2 * p - k) / s + 1;
        var input = Input(batch * channels * inH * inW);
        var expected = new float[batch * channels * outH * outW];
        var expectedIdx = new int[expected.Length];
        for (int plane = 0; plane < batch * channels; plane++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    float best = float.NegativeInfinity;
                    int bestIdx = 0;
                    for (int kh = 0; kh < k; kh++)
                        for (int kw = 0; kw < k; kw++)
                        {
                            int ih = oh * s - p + kh, iw = ow * s - p + kw;
                            if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                            float v = input[(plane * inH + ih) * inW + iw];
                            if (v > best) { best = v; bestIdx = ih * inW + iw; }
                        }
                    int o = (plane * outH + oh) * outW + ow;
                    expected[o] = best;
                    expectedIdx[o] = bestIdx;
                }
        using var inBuf = backend.AllocateBuffer(input);
        using var outBuf = backend.AllocateBuffer(Sentinel(expected.Length));
        using var idxBuf = backend.AllocateBuffer(expected.Length);
        backend.MaxPool2D(inBuf, outBuf, idxBuf, batch, channels, inH, inW, outH, outW, k, k, s, s, p, p);
        Assert.Equal(expected, backend.DownloadBuffer(outBuf));
        var idx = backend.DownloadBuffer(idxBuf);
        for (int i = 0; i < expectedIdx.Length; i++)
            Assert.Equal(expectedIdx[i], System.BitConverter.SingleToInt32Bits(idx[i]));
    }

    [SkippableTheory]
    [MemberData(nameof(Pools))]
    public void OpenCl_AvgPool2D_MatchesReference(int batch, int channels, int inH, int inW, int k, int s, int p)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        int outH = (inH + 2 * p - k) / s + 1, outW = (inW + 2 * p - k) / s + 1;
        var input = Input(batch * channels * inH * inW);
        var expected = new float[batch * channels * outH * outW];
        for (int plane = 0; plane < batch * channels; plane++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    float sum = 0f;
                    int count = 0;
                    for (int kh = 0; kh < k; kh++)
                        for (int kw = 0; kw < k; kw++)
                        {
                            int ih = oh * s - p + kh, iw = ow * s - p + kw;
                            if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                            sum += input[(plane * inH + ih) * inW + iw];
                            count++;
                        }
                    expected[(plane * outH + oh) * outW + ow] = sum / count;
                }
        using var inBuf = backend.AllocateBuffer(input);
        using var outBuf = backend.AllocateBuffer(Sentinel(expected.Length));
        backend.AvgPool2D(inBuf, outBuf, batch, channels, inH, inW, outH, outW, k, k, s, s, p, p, countIncludePad: false);
        var actual = backend.DownloadBuffer(outBuf);
        for (int i = 0; i < expected.Length; i++) Assert.InRange(actual[i] - expected[i], -1e-6f, 1e-6f);
    }

    [SkippableTheory]
    [MemberData(nameof(Pools))]
    public void OpenCl_DepthwiseConv2D_MatchesReference(int batch, int channels, int inH, int inW, int k, int s, int p)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        int outH = (inH + 2 * p - k) / s + 1, outW = (inW + 2 * p - k) / s + 1;
        var input = Input(batch * channels * inH * inW);
        var kernel = new float[channels * k * k];
        for (int i = 0; i < kernel.Length; i++) kernel[i] = (i % 5) * 0.25f - 0.5f;
        var expected = new float[batch * channels * outH * outW];
        for (int plane = 0; plane < batch * channels; plane++)
        {
            int c = plane % channels;
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    float sum = 0f;
                    for (int kh = 0; kh < k; kh++)
                        for (int kw = 0; kw < k; kw++)
                        {
                            int ih = oh * s - p + kh, iw = ow * s - p + kw;
                            if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) continue;
                            sum += input[(plane * inH + ih) * inW + iw] * kernel[(c * k + kh) * k + kw];
                        }
                    expected[(plane * outH + oh) * outW + ow] = sum;
                }
        }
        using var inBuf = backend.AllocateBuffer(input);
        using var kBuf = backend.AllocateBuffer(kernel);
        using var outBuf = backend.AllocateBuffer(Sentinel(expected.Length));
        backend.DepthwiseConv2D(inBuf, kBuf, outBuf, batch, channels, inH, inW, outH, outW, k, k, s, s, p, p);
        var actual = backend.DownloadBuffer(outBuf);
        for (int i = 0; i < expected.Length; i++) Assert.InRange(actual[i] - expected[i], -1e-5f, 1e-5f);
    }

    private static float[] Input(int count)
    {
        var x = new float[count];
        // Distinct values, so a pooled maximum has one position.
        for (int i = 0; i < count; i++) x[i] = ((i * 37) % 101) * 0.03125f - 1.5f + i * 1e-4f;
        return x;
    }

    private static float[] Sentinel(int count)
    {
        var x = new float[count];
        for (int i = 0; i < count; i++) x[i] = float.NaN;
        return x;
    }
}
#endif
