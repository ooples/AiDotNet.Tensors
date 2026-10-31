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
        // 2 x 2 at stride 2 over widths that are multiples of 4: max pooling's two-outputs-per-item kernel.
        { 3, 5, 8, 12, 2, 2, 0 },
        { 1, 1, 2, 4, 2, 2, 0 },
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
            Assert.Equal(expectedIdx[i], Bits(idx[i]));
    }

    [SkippableTheory]
    [MemberData(nameof(Pools))]
    public void OpenCl_MaxPool2D_WithoutIndices_MatchesReference(int batch, int channels, int inH, int inW, int k, int s, int p)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        int outH = (inH + 2 * p - k) / s + 1, outW = (inW + 2 * p - k) / s + 1;
        var input = Input(batch * channels * inH * inW);
        var (expected, _) = MaxPoolReference(input, batch * channels, inH, inW, outH, outW, k, s, p);
        using var inBuf = backend.AllocateBuffer(input);
        using var outBuf = backend.AllocateBuffer(Sentinel(expected.Length));
        backend.MaxPool2D(inBuf, outBuf, null, batch, channels, inH, inW, outH, outW, k, k, s, s, p, p);
        Assert.Equal(expected, backend.DownloadBuffer(outBuf));
    }

    [SkippableTheory]
    [MemberData(nameof(Pools))]
    public void OpenCl_MaxPool2DBackward_MatchesReference(int batch, int channels, int inH, int inW, int k, int s, int p)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        int outH = (inH + 2 * p - k) / s + 1, outW = (inW + 2 * p - k) / s + 1;
        int planes = batch * channels;
        var input = Input(planes * inH * inW);
        var (_, winners) = MaxPoolReference(input, planes, inH, inW, outH, outW, k, s, p);
        // Multiples of 1/4, so the atomic accumulation of overlapping windows is exact in any order.
        var gradOut = new float[planes * outH * outW];
        for (int i = 0; i < gradOut.Length; i++) gradOut[i] = ((i * 7) % 9) * 0.25f - 1f;
        var expected = new float[planes * inH * inW];
        for (int plane = 0; plane < planes; plane++)
            for (int o = 0; o < outH * outW; o++)
                expected[plane * inH * inW + winners[plane * outH * outW + o]] += gradOut[plane * outH * outW + o];
        var indexBits = new float[winners.Length];
        for (int i = 0; i < winners.Length; i++) indexBits[i] = FromBits(winners[i]);
        using var gradOutBuf = backend.AllocateBuffer(gradOut);
        using var idxBuf = backend.AllocateBuffer(indexBits);
        using var gradInBuf = backend.AllocateBuffer(new float[expected.Length]);
        backend.MaxPool2DBackward(gradOutBuf, idxBuf, gradInBuf, batch, channels, inH, inW, outH, outW, k, k, s, s, p, p);
        Assert.Equal(expected, backend.DownloadBuffer(gradInBuf));
    }

    public static TheoryData<int, int, int, int, int, int> AdaptivePools => new()
    {
        // batch, channels, inH, inW, outH, outW
        { 3, 7, 5, 6, 1, 1 },      // global pooling: 1 x 1 maps, many planes a group
        { 2, 5, 7, 9, 2, 2 },      // uneven windows
        { 1, 3, 14, 10, 7, 5 },    // exact halving
        { 2, 2, 9, 20, 9, 10 },    // two groups along x
    };

    [SkippableTheory]
    [MemberData(nameof(AdaptivePools))]
    public void OpenCl_AdaptivePool2D_MatchesReference(int batch, int channels, int inH, int inW, int outH, int outW)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        int planes = batch * channels;
        var input = Input(planes * inH * inW);
        var expectedMax = new float[planes * outH * outW];
        var expectedAvg = new float[expectedMax.Length];
        for (int plane = 0; plane < planes; plane++)
            for (int oh = 0; oh < outH; oh++)
                for (int ow = 0; ow < outW; ow++)
                {
                    int h0 = oh * inH / outH, h1 = ((oh + 1) * inH + outH - 1) / outH;
                    int w0 = ow * inW / outW, w1 = ((ow + 1) * inW + outW - 1) / outW;
                    float best = float.NegativeInfinity, sum = 0f;
                    for (int ih = h0; ih < h1; ih++)
                        for (int iw = w0; iw < w1; iw++)
                        {
                            float v = input[(plane * inH + ih) * inW + iw];
                            if (v > best) best = v;
                            sum += v;
                        }
                    int o = (plane * outH + oh) * outW + ow;
                    expectedMax[o] = best;
                    expectedAvg[o] = sum / ((h1 - h0) * (w1 - w0));
                }
        using var inBuf = backend.AllocateBuffer(input);
        using var maxBuf = backend.AllocateBuffer(Sentinel(expectedMax.Length));
        using var avgBuf = backend.AllocateBuffer(Sentinel(expectedAvg.Length));
        backend.AdaptiveMaxPool2D(inBuf, maxBuf, batch, channels, inH, inW, outH, outW);
        backend.AdaptiveAvgPool2D(inBuf, avgBuf, batch, channels, inH, inW, outH, outW);
        Assert.Equal(expectedMax, backend.DownloadBuffer(maxBuf));
        var avg = backend.DownloadBuffer(avgBuf);
        for (int i = 0; i < expectedAvg.Length; i++) Assert.InRange(avg[i] - expectedAvg[i], -1e-5f, 1e-5f);
    }

    // Max pooling over every plane: the pooled values and, per output, the winner's index within its input plane.
    private static (float[] Values, int[] Winners) MaxPoolReference(float[] input, int planes, int inH, int inW, int outH,
        int outW, int k, int s, int p)
    {
        var values = new float[planes * outH * outW];
        var winners = new int[values.Length];
        for (int plane = 0; plane < planes; plane++)
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
                    values[o] = best;
                    winners[o] = bestIdx;
                }
        return (values, winners);
    }

    // The pool kernels keep indices as int bits in a float buffer. BitConverter.SingleToInt32Bits and Int32BitsToSingle
    // are missing on .NET Framework, so these go through bytes, which every target framework has.
    private static int Bits(float value) => System.BitConverter.ToInt32(System.BitConverter.GetBytes(value), 0);

    private static float FromBits(int value) => System.BitConverter.ToSingle(System.BitConverter.GetBytes(value), 0);

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
