using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The shared Winograd pipeline (<see cref="WinogradF23Pipeline"/>) on each non-CUDA backend (#1368, #1370, #1371,
/// #1372), against a CPU convolution accumulated in double. The reference is computed on the host on purpose: a GPU
/// route as the reference could be the very candidate under test once the backend's slot is tuned.
/// </summary>
[Collection("DirectGpuSerial")]
public class WinogradF23PortTests
{
    private readonly ITestOutputHelper _output;

    public WinogradF23PortTests(ITestOutputHelper output) => _output = output;

    // batch, inC, outC, size, pad: odd sizes (a ragged last tile), every admitted padding, channel counts that are not
    // multiples of anything, and a 1x1 image.
    public static readonly TheoryData<int, int, int, int, int> Shapes = new()
    {
        { 2, 5, 7, 9, 1 },
        { 2, 16, 24, 11, 0 },
        { 3, 8, 6, 6, 2 },
        { 1, 3, 4, 1, 1 },
        { 4, 32, 32, 16, 1 },
    };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void OpenCl_winograd_routes_match_the_cpu_convolution(int batch, int inC, int outC, int size, int pad)
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        Check(backend, batch, inC, outC, size, pad,
            a => backend.TryConv2DForwardWinograd(a),
            a => backend.TryConv2DBackwardDataWinograd(a),
            a => backend.TryConv2DBackwardFilterWinograd(a));
    }

    private void Check(IDirectGpuBackend backend, int batch, int inC, int outC, int size, int pad,
        Func<TunedConv2DArgs, bool> forward, Func<TunedConv2DBackwardArgs, bool> backwardData,
        Func<TunedConv2DBackwardArgs, bool> backwardFilter)
    {
        int outSize = size + 2 * pad - 2;
        var x = Random(batch * inC * size * size, 1);
        var w = Random(outC * inC * 9, 2);
        var dy = Random(batch * outC * outSize * outSize, 3);
        using var bx = backend.AllocateBuffer(x);
        using var bw = backend.AllocateBuffer(w);
        using var bdy = backend.AllocateBuffer(dy);
        using var by = backend.AllocateBuffer(batch * outC * outSize * outSize);
        using var bdx = backend.AllocateBuffer(batch * inC * size * size);
        using var bdw = backend.AllocateBuffer(outC * inC * 9);

        Assert.True(forward(new TunedConv2DArgs(bx, bw, by, batch, inC, size, size, outC, outSize, outSize,
            3, 3, 1, 1, pad, pad, 1, 1)), "the Winograd forward refused an admitted shape");
        Assert.True(backwardData(new TunedConv2DBackwardArgs(bdy, bw, bdx, false, batch, inC, size, size, outC, outSize,
            outSize, 3, 3, 1, 1, pad, pad, 1, 1)), "the Winograd input gradient refused an admitted shape");
        Assert.True(backwardFilter(new TunedConv2DBackwardArgs(bx, bdy, bdw, true, batch, inC, size, size, outC, outSize,
            outSize, 3, 3, 1, 1, pad, pad, 1, 1)), "the Winograd weight gradient refused an admitted shape");
        backend.Synchronize();

        string label = $"N{batch} C{inC} K{outC} {size}x{size} pad{pad}";
        Compare("forward " + label, Forward(x, w, batch, inC, outC, size, pad), backend.DownloadBuffer(by));
        Compare("dX " + label, BackwardData(dy, w, batch, inC, outC, size, pad), backend.DownloadBuffer(bdx));
        Compare("dW " + label, BackwardFilter(x, dy, batch, inC, outC, size, pad), backend.DownloadBuffer(bdw));

        // The public entry points, whichever route the slots pick, must agree with the same reference.
        backend.Conv2D(bx, bw, by, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        backend.Conv2DBackwardInput(bdy, bw, bdx, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        backend.Conv2DBackwardKernel(bx, bdy, bdw, batch, inC, size, size, outC, outSize, outSize, 3, 3, 1, 1, pad, pad, 1, 1);
        backend.Synchronize();
        Compare("Conv2D " + label, Forward(x, w, batch, inC, outC, size, pad), backend.DownloadBuffer(by));
        Compare("Conv2DBackwardInput " + label, BackwardData(dy, w, batch, inC, outC, size, pad), backend.DownloadBuffer(bdx));
        Compare("Conv2DBackwardKernel " + label, BackwardFilter(x, dy, batch, inC, outC, size, pad), backend.DownloadBuffer(bdw));
    }

    // fp32 Winograd against a double-accumulated sum: the error is the fp32 rounding of transforms whose coefficients
    // are 0, +-1 and +-1/2, relative to the largest magnitude in the result.
    private const double Tolerance = 1e-4;

    private void Compare(string what, double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 1e-12, maxErr = 0;
        foreach (var e in expected) scale = Math.Max(scale, Math.Abs(e));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]) || float.IsInfinity(actual[i]), $"{what}: element {i} is {actual[i]}");
            maxErr = Math.Max(maxErr, Math.Abs(expected[i] - actual[i]) / scale);
        }
        _output.WriteLine($"[{what}] max err / max|ref| {maxErr:E2}");
        Assert.True(maxErr < Tolerance, $"{what}: max err / max|ref| {maxErr:E2}");
    }

    private static float[] Random(int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }

    private static double[] Forward(float[] x, float[] w, int n, int c, int k, int s, int pad)
    {
        int o = s + 2 * pad - 2;
        var y = new double[n * k * o * o];
        for (int b = 0; b < n; b++)
            for (int kk = 0; kk < k; kk++)
                for (int oh = 0; oh < o; oh++)
                    for (int ow = 0; ow < o; ow++)
                    {
                        double sum = 0;
                        for (int cc = 0; cc < c; cc++)
                            for (int i = 0; i < 3; i++)
                                for (int j = 0; j < 3; j++)
                                {
                                    int ih = oh + i - pad, iw = ow + j - pad;
                                    if (ih < 0 || ih >= s || iw < 0 || iw >= s) continue;
                                    sum += (double)x[((b * c + cc) * s + ih) * s + iw] * w[((kk * c + cc) * 3 + i) * 3 + j];
                                }
                        y[((b * k + kk) * o + oh) * o + ow] = sum;
                    }
        return y;
    }

    private static double[] BackwardData(float[] dy, float[] w, int n, int c, int k, int s, int pad)
    {
        int o = s + 2 * pad - 2;
        var dx = new double[n * c * s * s];
        for (int b = 0; b < n; b++)
            for (int kk = 0; kk < k; kk++)
                for (int oh = 0; oh < o; oh++)
                    for (int ow = 0; ow < o; ow++)
                    {
                        double g = dy[((b * k + kk) * o + oh) * o + ow];
                        for (int cc = 0; cc < c; cc++)
                            for (int i = 0; i < 3; i++)
                                for (int j = 0; j < 3; j++)
                                {
                                    int ih = oh + i - pad, iw = ow + j - pad;
                                    if (ih < 0 || ih >= s || iw < 0 || iw >= s) continue;
                                    dx[((b * c + cc) * s + ih) * s + iw] += g * w[((kk * c + cc) * 3 + i) * 3 + j];
                                }
                    }
        return dx;
    }

    private static double[] BackwardFilter(float[] x, float[] dy, int n, int c, int k, int s, int pad)
    {
        int o = s + 2 * pad - 2;
        var dw = new double[k * c * 9];
        for (int b = 0; b < n; b++)
            for (int kk = 0; kk < k; kk++)
                for (int oh = 0; oh < o; oh++)
                    for (int ow = 0; ow < o; ow++)
                    {
                        double g = dy[((b * k + kk) * o + oh) * o + ow];
                        for (int cc = 0; cc < c; cc++)
                            for (int i = 0; i < 3; i++)
                                for (int j = 0; j < 3; j++)
                                {
                                    int ih = oh + i - pad, iw = ow + j - pad;
                                    if (ih < 0 || ih >= s || iw < 0 || iw >= s) continue;
                                    dw[((kk * c + cc) * 3 + i) * 3 + j] += g * x[((b * c + cc) * s + ih) * s + iw];
                                }
                    }
        return dw;
    }
}