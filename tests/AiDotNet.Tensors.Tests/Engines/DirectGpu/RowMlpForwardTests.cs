#if NET6_0_OR_GREATER
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// <see cref="IRowMlpForwardBackend"/>: a batch-1 MLP forward in one launch, against a double-precision layer-by-layer
/// reference. The shapes cover both of the kernel's layer forms (a layer at least as wide as the work-group takes whole
/// columns per thread; a narrower one splits K across the group), widths that are not multiples of 4, a missing bias,
/// and one to four layers.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class RowMlpForwardTests
{
    private static float[] Values(int n, int seed, float scale)
    {
        var v = new float[n];
        for (int i = 0; i < v.Length; i++) v[i] = (((i + seed) * 7919 % 97) - 48) * scale;
        return v;
    }

    public static TheoryData<int[], bool, FusedActivationType, FusedActivationType> Cases() => new()
    {
        { new[] { 784, 512, 128, 10 }, true, FusedActivationType.ReLU, FusedActivationType.None },
        { new[] { 300, 257, 3 }, true, FusedActivationType.ReLU, FusedActivationType.ReLU },
        { new[] { 64, 1 }, false, FusedActivationType.None, FusedActivationType.None },
        { new[] { 5, 2048, 255, 77, 9 }, true, FusedActivationType.ReLU, FusedActivationType.None },
        { new[] { 1000, 40, 40 }, false, FusedActivationType.ReLU, FusedActivationType.ReLU },
    };

    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void OpenCl_RowMlpForward_MatchesReference(int[] widths, bool withBias, FusedActivationType hidden,
        FusedActivationType output)
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        int layers = widths.Length - 1;
        var x = Values(widths[0], 1, 0.02f);
        var w = new float[layers][];
        var b = new float[layers][];
        var weightBuffers = new IGpuBuffer[layers];
        var biasBuffers = new IGpuBuffer?[layers];
        try
        {
            for (int l = 0; l < layers; l++)
            {
                w[l] = Values(widths[l] * widths[l + 1], 3 + l, 0.01f);
                b[l] = Values(widths[l + 1], 17 + l, 0.05f);
                weightBuffers[l] = backend.AllocateBuffer(w[l]);
                biasBuffers[l] = withBias ? backend.AllocateBuffer(b[l]) : null;
            }
            using var input = backend.AllocateBuffer(x);
            using var result = backend.AllocateBuffer(new float[widths[layers]]);
            Assert.True(((IRowMlpForwardBackend)backend).TryRowMlpForward(input, weightBuffers, biasBuffers, widths, hidden,
                output, result), $"[{string.Join(",", widths)}] declined");
            var actual = backend.DownloadBuffer(result);

            var cur = Array.ConvertAll(x, v => (double)v);
            for (int l = 0; l < layers; l++)
            {
                int k = widths[l], n = widths[l + 1];
                var next = new double[n];
                bool relu = (l == layers - 1 ? output : hidden) == FusedActivationType.ReLU;
                for (int j = 0; j < n; j++)
                {
                    double s = withBias ? b[l][j] : 0.0;
                    for (int i = 0; i < k; i++) s += cur[i] * w[l][i * n + j];
                    next[j] = relu ? Math.Max(0.0, s) : s;
                }
                cur = next;
            }
            for (int j = 0; j < cur.Length; j++)
                Assert.True(Math.Abs(cur[j] - actual[j]) <= 1e-4 * (1 + Math.Abs(cur[j])),
                    $"[{string.Join(",", widths)}] output {j}: expected {cur[j]}, got {actual[j]}");
        }
        finally
        {
            foreach (var buffer in weightBuffers) buffer?.Dispose();
            foreach (var buffer in biasBuffers) buffer?.Dispose();
        }
    }

    [SkippableFact]
    public void OpenCl_RowMlpForward_DeclinesWhatItDoesNotRun()
    {
        using var backend = new OpenClBackend();
        Skip.IfNot(backend.IsAvailable, "OpenCL is not available.");
        var rowMlp = (IRowMlpForwardBackend)backend;
        using var input = backend.AllocateBuffer(new float[8]);
        using var weight = backend.AllocateBuffer(new float[8 * 4]);
        using var output = backend.AllocateBuffer(new float[4]);
        Assert.False(rowMlp.TryRowMlpForward(input, new[] { weight }, new IGpuBuffer?[] { null }, new[] { 8, 4 },
            FusedActivationType.GELU, FusedActivationType.None, output));
        Assert.False(rowMlp.TryRowMlpForward(input, new[] { weight, weight, weight, weight, weight },
            new IGpuBuffer?[5], new[] { 8, 4, 4, 4, 4, 4 }, FusedActivationType.ReLU, FusedActivationType.None, output));
        using var wide = backend.AllocateBuffer(new float[8 * 4096]);
        using var wideOut = backend.AllocateBuffer(new float[4096]);
        Assert.False(rowMlp.TryRowMlpForward(input, new[] { wide }, new IGpuBuffer?[] { null }, new[] { 8, 4096 },
            FusedActivationType.ReLU, FusedActivationType.None, wideOut));
    }
}
#endif