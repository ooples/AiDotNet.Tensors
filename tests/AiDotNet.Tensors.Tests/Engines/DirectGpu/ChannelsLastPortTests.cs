using System;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Engines.DirectGpu.Metal;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The channels-last operations (<see cref="IChannelsLastBackend"/>) on every backend against host references: the
/// layout change (exact), training BatchNorm with its activation over a [rows, channels] activation and its backward
/// (in double on the host), and the three convolution passes with every layout pairing against a direct NCHW
/// convolution in double. CUDA's convolutions run on cuDNN's TF32 tensor cores where it prefers channels-last, so the
/// convolution tolerance is TF32's (relative to the sum of the absolute products).
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class ChannelsLastPortTests
{
    [SkippableFact]
    public void Cuda_matches_the_host()
    {
        Skip.IfNot(AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.IsCudaAvailable, "CUDA is not available.");
        using var backend = new AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend();
        Skip.IfNot(backend.IsAvailable, "CUDA backend did not initialize.");
        // Without cuDNN the CUDA backend takes no channels-last convolution (and a plan keeps NCHW). FP32 plans do not
        // prefer the layout by default; the switch reports whether cuDNN would take it.
        bool prior = AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_channelsLastAtFp32;
        AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_channelsLastAtFp32 = true;
        bool cudnn;
        try
        {
            cudnn = ((IChannelsLastBackend)backend).PrefersChannelsLast;
        }
        finally
        {
            AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend.s_channelsLastAtFp32 = prior;
        }
        CheckAll(backend, convolutions: cudnn);
    }

    [SkippableFact]
    public void OpenCl_matches_the_host()
    {
        Skip.IfNot(DirectOpenClContext.IsAvailable && DirectOpenClContext.GetDeviceCount() > 0, "No OpenCL device.");
        using var backend = new OpenClBackend(deviceIndex: 0);
        Skip.IfNot(backend.IsAvailable, "OpenCL backend did not initialize.");
        CheckAll(backend, convolutions: true);
    }

    [SkippableFact]
    public void Vulkan_matches_the_host()
    {
        var backend = VulkanBackend.Instance;
        Skip.IfNot(backend.Initialize() && backend.IsGlslCompilerAvailable, "Vulkan with a GLSL compiler is not available.");
        CheckAll(backend, convolutions: true);
    }

    [SkippableFact]
    public void Metal_matches_the_host()
    {
        Skip.IfNot(System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX),
            "Metal is only available on macOS.");
        using var backend = new MetalBackend();
        Skip.IfNot(backend.IsAvailable, "Metal is not available.");
        CheckAll(backend, convolutions: true);
    }

    [SkippableFact]
    public void Hip_matches_the_host()
    {
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable, "HIP is not available.");
        CheckAll(backend, convolutions: true);
    }

#if NET7_0_OR_GREATER
    [SkippableFact]
    public void WebGpu_matches_the_host()
    {
        var backend = DawnHostedWebGpu.BackendOrSkip();
        CheckAll(backend, convolutions: true);
    }
#endif

    private static float[] Rand(int n, int seed, float scale = 1f, float offset = 0f)
    {
        var rng = new Random(seed);
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)((rng.NextDouble() * 2 - 1) * scale) + offset;
        return a;
    }

    // [n, c, s] <-> [n, s, c] on the host.
    private static float[] ToChannelsLast(float[] x, int n, int c, int s)
    {
        var y = new float[x.Length];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int p = 0; p < s; p++)
                    y[(b * s + p) * c + ch] = x[(b * c + ch) * s + p];
        return y;
    }

    private static float[] ToNchw(float[] x, int n, int c, int s)
    {
        var y = new float[x.Length];
        for (int b = 0; b < n; b++)
            for (int ch = 0; ch < c; ch++)
                for (int p = 0; p < s; p++)
                    y[(b * c + ch) * s + p] = x[(b * s + p) * c + ch];
        return y;
    }

    private static void CheckAll(IDirectGpuBackend backend, bool convolutions)
    {
        var channelsLast = Assert.IsAssignableFrom<IChannelsLastBackend>(backend);
        CheckLayout(backend, channelsLast, 2, 3, 5);
        CheckLayout(backend, channelsLast, 3, 64, 100);
        CheckLayout(backend, channelsLast, 1, 40, 33);
        foreach (int act in new[] { 0, 1, 2 })
        {
            CheckBatchNorm(backend, channelsLast, rows: 50, channels: 8, act);
            CheckBatchNorm(backend, channelsLast, rows: 2048, channels: 64, act);
        }
        if (!convolutions) return;
        // A strided 3x3 convolution with padding (a DCGAN discriminator layer) and a 4x4 stride-2 one (its first).
        CheckConvolution(backend, channelsLast, n: 2, c: 4, h: 8, w: 8, k: 8, r: 3, s: 3, pad: 1, stride: 2);
        CheckConvolution(backend, channelsLast, n: 3, c: 8, h: 16, w: 16, k: 4, r: 4, s: 4, pad: 1, stride: 2);
    }

    private static void CheckLayout(IDirectGpuBackend backend, IChannelsLastBackend channelsLast, int n, int c, int s)
    {
        var x = Rand(n * c * s, 11);
        using var xBuf = backend.AllocateBuffer(x);
        using var yBuf = backend.AllocateBuffer(x.Length);
        using var zBuf = backend.AllocateBuffer(x.Length);
        Assert.True(channelsLast.TryChangeLayout(xBuf, yBuf, n, c, s, toChannelsLast: true), $"layout refused ({n},{c},{s})");
        Assert.True(channelsLast.TryChangeLayout(yBuf, zBuf, n, c, s, toChannelsLast: false), $"layout back refused ({n},{c},{s})");
        backend.Synchronize();
        Assert.Equal(ToChannelsLast(x, n, c, s), backend.DownloadBuffer(yBuf));
        Assert.Equal(x, backend.DownloadBuffer(zBuf));
    }

    private static void CheckBatchNorm(IDirectGpuBackend backend, IChannelsLastBackend channelsLast, int rows,
        int channels, int act)
    {
        const float eps = 1e-5f, slope = 0.2f;
        var x = Rand(rows * channels, 21, 2f, 0.5f);
        var gamma = Rand(channels, 22, 0.5f, 1f);
        var beta = Rand(channels, 23, 0.5f);
        var dy = Rand(rows * channels, 24);
        using var xBuf = backend.AllocateBuffer(x);
        using var gBuf = backend.AllocateBuffer(gamma);
        using var bBuf = backend.AllocateBuffer(beta);
        using var dyBuf = backend.AllocateBuffer(dy);
        using var yBuf = backend.AllocateBuffer(x.Length);
        using var mBuf = backend.AllocateBuffer(channels);
        using var vBuf = backend.AllocateBuffer(channels);
        using var dxBuf = backend.AllocateBuffer(x.Length);
        using var dgBuf = backend.AllocateBuffer(channels);
        using var dbBuf = backend.AllocateBuffer(channels);
        Assert.True(channelsLast.TryBatchNormChannelsLast(xBuf, yBuf, gBuf, bBuf, mBuf, vBuf, rows, channels, eps, act, slope),
            $"BatchNorm refused (act {act})");
        Assert.True(channelsLast.TryBatchNormChannelsLastBackward(dyBuf, act != 0 ? yBuf : null, null, xBuf, gBuf, mBuf,
                vBuf, dxBuf, dgBuf, dbBuf, rows, channels, eps, act != 0 ? 1 : 0, act, slope, accumulate: false),
            $"BatchNorm backward refused (act {act})");
        backend.Synchronize();
        float[] y = backend.DownloadBuffer(yBuf), mean = backend.DownloadBuffer(mBuf), variance = backend.DownloadBuffer(vBuf);
        float[] dx = backend.DownloadBuffer(dxBuf), dg = backend.DownloadBuffer(dgBuf), db = backend.DownloadBuffer(dbBuf);
        string at = $"(rows {rows}, channels {channels}, act {act})";
        for (int ch = 0; ch < channels; ch++)
        {
            double m = 0, v = 0;
            for (int r = 0; r < rows; r++) m += x[r * channels + ch];
            m /= rows;
            for (int r = 0; r < rows; r++) { double d = x[r * channels + ch] - m; v += d * d; }
            v /= rows;
            Near(m, mean[ch], 1e-5, $"mean[{ch}] {at}");
            Near(v, variance[ch], 1e-5, $"variance[{ch}] {at}");
            double inv = 1.0 / Math.Sqrt(v + eps);
            double sumG = 0, sumGx = 0;
            var gPre = new double[rows];
            var xhat = new double[rows];
            for (int r = 0; r < rows; r++)
            {
                int i = r * channels + ch;
                xhat[r] = (x[i] - m) * inv;
                double pre = gamma[ch] * xhat[r] + beta[ch];
                double outV = act == 1 ? Math.Max(0, pre) : act == 2 ? (pre > 0 ? pre : slope * pre) : pre;
                Near(outV, y[i], 1e-5, $"y[{i}] {at}");
                gPre[r] = act == 0 || pre > 0 ? dy[i] : act == 2 ? slope * dy[i] : 0;
                sumG += gPre[r];
                sumGx += gPre[r] * xhat[r];
            }
            Near(sumG, db[ch], 1e-4, $"gradBeta[{ch}] {at}");
            Near(sumGx, dg[ch], 1e-4, $"gradGamma[{ch}] {at}");
            for (int r = 0; r < rows; r++)
            {
                double expected = gamma[ch] * inv / rows * (rows * gPre[r] - sumG - xhat[r] * sumGx);
                Near(expected, dx[r * channels + ch], 1e-4, $"dx[{r * channels + ch}] {at}");
            }
        }
    }

    private static void CheckConvolution(IDirectGpuBackend backend, IChannelsLastBackend channelsLast, int n, int c,
        int h, int w, int k, int r, int s, int pad, int stride)
    {
        int oh = (h + 2 * pad - r) / stride + 1, ow = (w + 2 * pad - s) / stride + 1;
        var x = Rand(n * c * h * w, 31);
        var wt = Rand(k * c * r * s, 32, 0.5f);
        var dy = Rand(n * k * oh * ow, 33);
        // Direct references in double, with each output's sum of absolute products for the TF32 tolerance.
        var y = new double[dy.Length]; var yMag = new double[dy.Length];
        var dx = new double[x.Length]; var dxMag = new double[x.Length];
        var dw = new double[wt.Length]; var dwMag = new double[wt.Length];
        for (int b = 0; b < n; b++)
            for (int o = 0; o < k; o++)
                for (int py = 0; py < oh; py++)
                    for (int px = 0; px < ow; px++)
                    {
                        int yi = ((b * k + o) * oh + py) * ow + px;
                        for (int ch = 0; ch < c; ch++)
                            for (int ky = 0; ky < r; ky++)
                                for (int kx = 0; kx < s; kx++)
                                {
                                    int iy = py * stride - pad + ky, ix = px * stride - pad + kx;
                                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                                    int xi = ((b * c + ch) * h + iy) * w + ix, wi = ((o * c + ch) * r + ky) * s + kx;
                                    y[yi] += (double)x[xi] * wt[wi]; yMag[yi] += Math.Abs((double)x[xi] * wt[wi]);
                                    dx[xi] += (double)wt[wi] * dy[yi]; dxMag[xi] += Math.Abs((double)wt[wi] * dy[yi]);
                                    dw[wi] += (double)x[xi] * dy[yi]; dwMag[wi] += Math.Abs((double)x[xi] * dy[yi]);
                                }
                    }
        foreach (var (inNhwc, outNhwc) in new[] { (true, true), (false, true), (true, false) })
        {
            string at = $"(n{n} c{c} {h}x{w} k{k} {r}x{s} p{pad} s{stride}, x {(inNhwc ? "NHWC" : "NCHW")}, y {(outNhwc ? "NHWC" : "NCHW")})";
            var xIn = inNhwc ? ToChannelsLast(x, n, c, h * w) : x;
            var dyIn = outNhwc ? ToChannelsLast(dy, n, k, oh * ow) : dy;
            using var xBuf = backend.AllocateBuffer(xIn);
            using var wBuf = backend.AllocateBuffer(wt);
            using var dyBuf = backend.AllocateBuffer(dyIn);
            using var yBuf = backend.AllocateBuffer(dy.Length);
            using var dxBuf = backend.AllocateBuffer(x.Length);
            using var dwBuf = backend.AllocateBuffer(wt.Length);
            Assert.True(channelsLast.TryConvolutionChannelsLast(0, xBuf, wBuf, yBuf, n, c, h, w, k, r, s, oh, ow, pad, pad,
                stride, stride, 1, 1, inNhwc, outNhwc), $"forward refused {at}");
            Assert.True(channelsLast.TryConvolutionChannelsLast(1, wBuf, dyBuf, dxBuf, n, c, h, w, k, r, s, oh, ow, pad, pad,
                stride, stride, 1, 1, inNhwc, outNhwc), $"backward data refused {at}");
            Assert.True(channelsLast.TryConvolutionChannelsLast(2, xBuf, dyBuf, dwBuf, n, c, h, w, k, r, s, oh, ow, pad, pad,
                stride, stride, 1, 1, inNhwc, outNhwc), $"backward filter refused {at}");
            backend.Synchronize();
            var yGot = backend.DownloadBuffer(yBuf);
            var dxGot = backend.DownloadBuffer(dxBuf);
            if (outNhwc) yGot = ToNchw(yGot, n, k, oh * ow);
            if (inNhwc) dxGot = ToNchw(dxGot, n, c, h * w);
            var dwGot = backend.DownloadBuffer(dwBuf);
            for (int i = 0; i < y.Length; i++) Within(y[i], yGot[i], yMag[i], $"y[{i}] {at}");
            for (int i = 0; i < dx.Length; i++) Within(dx[i], dxGot[i], dxMag[i], $"dx[{i}] {at}");
            for (int i = 0; i < dw.Length; i++) Within(dw[i], dwGot[i], dwMag[i], $"dw[{i}] {at}");
        }
    }

    private static void Near(double expected, float actual, double tolerance, string label)
        => Assert.True(Math.Abs(expected - actual) <= tolerance * Math.Max(1.0, Math.Abs(expected)),
            $"{label}: {actual} vs {expected}");

    // TF32 keeps 10 mantissa bits per operand: an error up to ~2^-10 of the summed magnitudes.
    private static void Within(double expected, float actual, double magnitude, string label)
        => Assert.True(Math.Abs(expected - actual) <= 2e-3 * magnitude + 1e-6, $"{label}: {actual} vs {expected}");
}
