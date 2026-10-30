#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The OpenCL epilogues that replaced separate bias and activation passes: a dense layer at batch 1 applies its bias and
/// activation in the split-K GEMV's reduce, and a convolution that writes NCHW directly (the generic route) applies them
/// in one in-place pass. Each against a double-precision reference and against the unfused path.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClEpilogueFusionTests
{
    private static float[] Values(int n, int seed, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < v.Length; i++) v[i] = (((i + seed) * 7919 % 97) - 48) * scale;
        return v;
    }

    private static double Activate(double x, int mode, double slope) => mode switch
    {
        1 => double.IsNaN(x) ? x : Math.Max(0.0, x),
        6 => x > 0.0 ? x : slope * x,
        _ => x,
    };

    private static void AssertClose(double expected, float actual, string what)
    {
        if (double.IsNaN(expected))
        {
            Assert.True(float.IsNaN(actual), $"{what}: expected NaN, got {actual}");
            return;
        }
        Assert.True(Math.Abs(expected - actual) <= 1e-4 * (1 + Math.Abs(expected)), $"{what}: expected {expected}, got {actual}");
    }

    [SkippableTheory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(6)]
    public void OpenCl_DenseEpilogue_AtBatchOne_MatchesReferenceAndTwoRowPath(int mode)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        const int k = 512, n = 64;
        const float slope = 0.2f;
        var a = Values(2 * k, 1, 0.01f);
        Array.Copy(a, 0, a, k, k);   // both rows equal, so the two-row product's first row is the one-row product
        var b = Values(k * n, 5, 0.02f);
        var bias = Values(n, 11, 0.05f);
        bias[5] = float.NaN;   // ReLU passes NaN through, as the separate relu kernel does
        using var bBuffer = backend.AllocateBuffer(b);
        using var biasBuffer = backend.AllocateBuffer(bias);
        using var oneRow = backend.AllocateBuffer(a.AsSpan(0, k).ToArray());
        using var twoRows = backend.AllocateBuffer(a);

        IGpuBuffer Run(IGpuBuffer x, int m) => mode switch
        {
            1 => backend.GemmBiasRelu(x, bBuffer, biasBuffer, m, n, k),
            6 => backend.GemmBiasLeakyRelu(x, bBuffer, biasBuffer, m, n, k, slope),
            _ => backend.GemmBias(x, bBuffer, biasBuffer, m, n, k),
        };

        using var fused = Run(oneRow, 1);
        using var unfused = Run(twoRows, 2);
        var actual = backend.DownloadBuffer(fused);
        var reference = backend.DownloadBuffer(unfused);
        for (int j = 0; j < n; j++)
        {
            double sum = bias[j];
            for (int i = 0; i < k; i++) sum += (double)a[i] * b[i * n + j];
            AssertClose(Activate(sum, mode, slope), actual[j], $"mode {mode} column {j} vs reference");
            AssertClose(reference[j], actual[j], $"mode {mode} column {j} vs the two-row path");
        }
    }

    [SkippableTheory]
    [InlineData(FusedActivationType.None)]
    [InlineData(FusedActivationType.ReLU)]
    public void OpenCl_GenericConvolutionEpilogue_MatchesReference(FusedActivationType activation)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var generic = backend.Conv2DForwardSlot.Candidate("opencl.conv2d.generic");
        Skip.If(generic is null, "The generic convolution route is not registered.");
        const int batch = 2, cin = 3, h = 9, w = 7, cout = 5;
        var input = Values(batch * cin * h * w, 3, 0.03f);
        var kernel = Values(cout * cin * 9, 7, 0.04f);
        var bias = Values(cout, 13, 0.1f);
        using var inputBuffer = backend.AllocateBuffer(input);
        using var kernelBuffer = backend.AllocateBuffer(kernel);
        using var biasBuffer = backend.AllocateBuffer(bias);
        using var output = backend.AllocateBuffer(new float[batch * cout * h * w]);
        generic.Execute(new TunedConv2DArgs(inputBuffer, kernelBuffer, output, batch, cin, h, w, cout, h, w,
            3, 3, 1, 1, 1, 1, 1, 1, biasBuffer, activation));
        var actual = backend.DownloadBuffer(output);
        for (int bi = 0; bi < batch; bi++)
            for (int co = 0; co < cout; co++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                    {
                        double sum = bias[co];
                        for (int ci = 0; ci < cin; ci++)
                            for (int ky = 0; ky < 3; ky++)
                                for (int kx = 0; kx < 3; kx++)
                                {
                                    int iy = y + ky - 1, ix = x + kx - 1;
                                    if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                                    sum += (double)input[((bi * cin + ci) * h + iy) * w + ix] * kernel[((co * cin + ci) * 3 + ky) * 3 + kx];
                                }
                        double expected = activation == FusedActivationType.ReLU ? Math.Max(0.0, sum) : sum;
                        AssertClose(expected, actual[((bi * cout + co) * h + y) * w + x], $"{activation} b{bi} c{co} ({y},{x})");
                    }
    }

    // Few outputs over a long K (a classifier head) run on the split-K GEMM: against a double reference, including
    // shapes that are not tile multiples and one at the four-tile limit.
    [SkippableTheory]
    [InlineData(32, 10, 4096)]
    [InlineData(5, 100, 2000)]
    [InlineData(128, 128, 1024)]
    [InlineData(1, 3, 5000)]
    public void OpenCl_SkinnyLongKGemm_MatchesReference(int m, int n, int k)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var a = Values(m * k, 2, 0.01f);
        var b = Values(k * n, 9, 0.01f);
        using var aBuffer = backend.AllocateBuffer(a);
        using var bBuffer = backend.AllocateBuffer(b);
        using var cBuffer = backend.AllocateBuffer(new float[m * n]);
        backend.Gemm(aBuffer, bBuffer, cBuffer, m, n, k);
        var actual = backend.DownloadBuffer(cBuffer);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += (double)a[i * k + p] * b[p * n + j];
                AssertClose(sum, actual[i * n + j], $"{m}x{n}x{k} [{i},{j}]");
            }
    }
}
#endif