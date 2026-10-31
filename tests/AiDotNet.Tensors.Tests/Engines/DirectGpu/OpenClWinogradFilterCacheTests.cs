#if NET6_0_OR_GREATER
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The F(2x2, 3x3) forward Winograd route keeps its filter transform per weight identity and version
/// (<see cref="ConvWeightIdentity"/>): a call at the same version reuses it, a new version transforms again, and a call
/// with no identity (a tape, a resident step) never reads or fills the cache.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class OpenClWinogradFilterCacheTests
{
    private const int B = 2, C = 8, H = 6, W = 6, K = 5;

    [SkippableFact]
    public void OpenCl_WinogradF23Forward_ReusesTheFilterTransformUntilTheVersionChanges()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        float[] x = Values(B * C * H * W, 1), w1 = Values(K * C * 9, 2), w2 = Values(K * C * 9, 3);
        using var xb = backend.AllocateBuffer(x);
        using var wb = backend.AllocateBuffer(w1);
        using var w1b = backend.AllocateBuffer(w1);
        using var w2b = backend.AllocateBuffer(w2);
        using var yb = backend.AllocateBuffer(B * K * H * W);
        var args = new TunedConv2DArgs(xb, wb, yb, B, C, H, W, K, H, W, 3, 3, 1, 1, 1, 1, 1, 1);
        var key = new object();

        Assert.True(Run(backend, args, key, version: 1));
        AssertClose(Reference(x, w1), backend.DownloadBuffer(yb));

        // New weights in the same buffer at the same version: the cached transform of w1 is used, which is the cache's
        // contract (the engine bumps the version whenever the host weights change). This is the positive control that
        // the cache is consulted at all.
        backend.Copy(w2b, wb, w2.Length);
        Assert.True(Run(backend, args, key, version: 1));
        AssertClose(Reference(x, w1), backend.DownloadBuffer(yb));

        // A new version transforms the weights again.
        Assert.True(Run(backend, args, key, version: 2));
        AssertClose(Reference(x, w2), backend.DownloadBuffer(yb));

        // No identity: no cache, the buffer's weights.
        backend.Copy(w1b, wb, w1.Length);
        Assert.True(backend.TryConv2DForwardWinograd(args));
        backend.Synchronize();
        AssertClose(Reference(x, w1), backend.DownloadBuffer(yb));
    }

    [SkippableFact]
    public void OpenCl_SmallMapGemm_ReusesTheGatheredWeightsUntilTheVersionChanges()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        const int b = 3, c = 6, k = 5;
        float[] x = Values(b * c * 4, 4), w1 = Values(k * c * 9, 5), w2 = Values(k * c * 9, 6);
        using var xb = backend.AllocateBuffer(x);
        using var wb = backend.AllocateBuffer(w1);
        using var w1b = backend.AllocateBuffer(w1);
        using var w2b = backend.AllocateBuffer(w2);
        using var yb = backend.AllocateBuffer(b * k * 4);
        var args = new TunedConv2DArgs(xb, wb, yb, b, c, 2, 2, k, 2, 2, 3, 3, 1, 1, 1, 1, 1, 1);
        var route = (AiDotNet.Tensors.Helpers.Autotune.TunedKernels.ITunedKernelCandidate<TunedConv2DArgs>)
            backend.Conv2DForwardSlot.Candidate("opencl.conv2d.small_map_gemm");
        Skip.If(!route.IsApplicable(args.Shape()), "The small-map GEMM's kernels did not compile.");
        var key = new object();

        void RunAt(int version)
        {
            using (ConvWeightIdentity.Enter(key, version)) route.Execute(args);
            backend.Synchronize();
        }

        RunAt(1);
        AssertClose(Reference2x2(x, w1, b, c, k), backend.DownloadBuffer(yb));
        // New weights at the same version: the cached gather of w1 is used (the positive control that it is cached).
        backend.Copy(w2b, wb, w2.Length);
        RunAt(1);
        AssertClose(Reference2x2(x, w1, b, c, k), backend.DownloadBuffer(yb));
        // A new version gathers again.
        RunAt(2);
        AssertClose(Reference2x2(x, w2, b, c, k), backend.DownloadBuffer(yb));
    }

    private static float[] Reference2x2(float[] x, float[] w, int b, int c, int k)
    {
        var y = new float[b * k * 4];
        for (int n = 0; n < b; n++)
            for (int o = 0; o < k; o++)
                for (int i = 0; i < 2; i++)
                    for (int j = 0; j < 2; j++)
                    {
                        double acc = 0;
                        for (int ci = 0; ci < c; ci++)
                            for (int u = 0; u < 3; u++)
                                for (int v = 0; v < 3; v++)
                                {
                                    int ih = i - 1 + u, iw = j - 1 + v;
                                    if (ih < 0 || ih >= 2 || iw < 0 || iw >= 2) continue;
                                    acc += (double)x[((n * c + ci) * 2 + ih) * 2 + iw] * w[((o * c + ci) * 3 + u) * 3 + v];
                                }
                        y[((n * k + o) * 2 + i) * 2 + j] = (float)acc;
                    }
        return y;
    }
    private static bool Run(OpenClBackend backend, in TunedConv2DArgs args, object key, int version)
    {
        bool ran;
        using (ConvWeightIdentity.Enter(key, version)) ran = backend.TryConv2DForwardWinograd(args);
        backend.Synchronize();
        return ran;
    }

    private static float[] Reference(float[] x, float[] w)
    {
        var y = new float[B * K * H * W];
        for (int b = 0; b < B; b++)
            for (int k = 0; k < K; k++)
                for (int i = 0; i < H; i++)
                    for (int j = 0; j < W; j++)
                    {
                        double acc = 0;
                        for (int c = 0; c < C; c++)
                            for (int u = 0; u < 3; u++)
                                for (int v = 0; v < 3; v++)
                                {
                                    int ih = i - 1 + u, iw = j - 1 + v;
                                    if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                                    acc += (double)x[((b * C + c) * H + ih) * W + iw] * w[((k * C + c) * 3 + u) * 3 + v];
                                }
                        y[((b * K + k) * H + i) * W + j] = (float)acc;
                    }
        return y;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        for (int i = 0; i < expected.Length; i++)
            Assert.True(System.Math.Abs(expected[i] - actual[i]) <= 1e-4f * (1f + System.Math.Abs(expected[i])),
                $"index {i}: expected {expected[i]}, actual {actual[i]}");
    }

    private static float[] Values(int count, int seed)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = ((i * 7 + seed * 13) % 17 - 8) * 0.0625f;
        return v;
    }
}
#endif
