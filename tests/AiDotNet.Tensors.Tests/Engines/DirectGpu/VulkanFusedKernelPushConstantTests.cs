using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.Vulkan;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The Vulkan fused kernels whose shaders take several push-constant words (extents, offsets, a pad value) must
/// receive every one of them. They were launched through a shorthand that pushed only the dispatch size, so each
/// shader read its remaining parameters from whatever followed a one-element array: an outer product wrote nothing,
/// and a sum's gradient on a Vulkan engine came back all zeros. Each op is checked against a host computation with
/// non-square, non-trivial extents (so a swapped or missing parameter changes the result).
/// </summary>
[Collection("VulkanGlobalState")]
public class VulkanFusedKernelPushConstantTests
{
    private static VulkanBackend BackendOrSkip()
    {
        VulkanBackend? backend = null;
        try
        {
            backend = VulkanBackend.Instance;
            if (!backend.Initialize() || !backend.IsGlslCompilerAvailable) backend = null;
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException)
        {
            backend = null;
        }
        Skip.If(backend is null, "No Vulkan device with the runtime GLSL compiler.");
        return backend!;
    }

    private static float[] Values(int n, int seed, float scale = 4f)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float)((rng.NextDouble() - 0.5) * scale);
        return v;
    }

    private static void AssertClose(string op, float[] expected, float[] actual, double tolerance = 1e-5)
    {
        Assert.True(actual.Length >= expected.Length, $"{op}: output holds {actual.Length} values, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1.0, Math.Abs(expected[i])),
                $"{op}[{i}]: expected {expected[i]:R}, vulkan {actual[i]:R}");
    }

    // Runs a two-buffer op into an output pre-filled with a sentinel, so an op that writes nothing is caught.
    private static float[] Run(VulkanBackend backend, float[] input, int outputLength, Action<IGpuBuffer, IGpuBuffer> op)
    {
        using var source = backend.AllocateBuffer(input);
        using var output = backend.AllocateBuffer(outputLength);
        backend.Fill(output, -777f, outputLength);
        op(source, output);
        return backend.DownloadBuffer(output);
    }

    [SkippableFact]
    public void OuterProduct_AndBatchOuterProduct()
    {
        var backend = BackendOrSkip();
        const int m = 5, n = 7, batch = 3;
        float[] a = Values(m, 1), b = Values(n, 2);
        using (var ba = backend.AllocateBuffer(a))
        using (var bb = backend.AllocateBuffer(b))
        using (var bo = backend.AllocateBuffer(m * n))
        {
            backend.Fill(bo, -777f, m * n);
            backend.OuterProduct(ba, bb, bo, m, n);
            var expected = new float[m * n];
            for (int i = 0; i < m; i++) for (int j = 0; j < n; j++) expected[i * n + j] = a[i] * b[j];
            AssertClose("OuterProduct", expected, backend.DownloadBuffer(bo));
        }

        float[] a2 = Values(batch * m, 3), b2 = Values(batch * n, 4);
        using var ba2 = backend.AllocateBuffer(a2);
        using var bb2 = backend.AllocateBuffer(b2);
        using var bo2 = backend.AllocateBuffer(batch * m * n);
        backend.Fill(bo2, -777f, batch * m * n);
        backend.BatchOuterProduct(ba2, bb2, bo2, batch, m, n);
        var expected2 = new float[batch * m * n];
        for (int s = 0; s < batch; s++) for (int i = 0; i < m; i++) for (int j = 0; j < n; j++)
            expected2[(s * m + i) * n + j] = a2[s * m + i] * b2[s * n + j];
        AssertClose("BatchOuterProduct", expected2, backend.DownloadBuffer(bo2));
    }

    [SkippableFact]
    public void TileLastAxis_AndRepeatElements()
    {
        var backend = BackendOrSkip();
        const int outer = 3, inner = 5, repeats = 4;
        float[] x = Values(outer * inner, 5);
        var tiled = new float[outer * inner * repeats];
        for (int o = 0; o < outer; o++) for (int r = 0; r < repeats; r++) for (int i = 0; i < inner; i++)
            tiled[o * inner * repeats + r * inner + i] = x[o * inner + i];
        AssertClose("TileLastAxis", tiled, Run(backend, x, tiled.Length, (s, d) => backend.TileLastAxis(s, d, outer, inner, repeats)));

        var repeated = new float[outer * inner * repeats];
        for (int i = 0; i < repeated.Length; i++) repeated[i] = x[i / repeats];
        AssertClose("RepeatElements", repeated, Run(backend, x, repeated.Length, (s, d) => backend.RepeatElements(s, d, outer, inner, repeats)));
    }

    [SkippableFact]
    public void Pad2D_UsesThePadValue_AndItsBackwardReadsTheWindow()
    {
        var backend = BackendOrSkip();
        const int batch = 2, channels = 3, inH = 4, inW = 5, padTop = 2, padLeft = 1, outH = 9, outW = 8;
        const float padValue = -3.5f;
        float[] x = Values(batch * channels * inH * inW, 6);
        var padded = new float[batch * channels * outH * outW];
        for (int p = 0; p < batch * channels; p++) for (int h = 0; h < outH; h++) for (int w = 0; w < outW; w++)
        {
            int sh = h - padTop, sw = w - padLeft;
            padded[(p * outH + h) * outW + w] = sh >= 0 && sh < inH && sw >= 0 && sw < inW ? x[(p * inH + sh) * inW + sw] : padValue;
        }
        AssertClose("Pad2D", padded, Run(backend, x, padded.Length,
            (s, d) => backend.Pad2D(s, d, batch, channels, inH, inW, outH, outW, padTop, padLeft, padValue)));

        float[] gradOutput = Values(padded.Length, 7);
        var gradInput = new float[x.Length];
        for (int p = 0; p < batch * channels; p++) for (int h = 0; h < inH; h++) for (int w = 0; w < inW; w++)
            gradInput[(p * inH + h) * inW + w] = gradOutput[(p * outH + h + padTop) * outW + w + padLeft];
        AssertClose("Pad2DBackward", gradInput, Run(backend, gradOutput, gradInput.Length,
            (s, d) => backend.Pad2DBackward(s, d, batch, channels, inH, inW, outH, outW, padTop, padLeft)));
    }

    [SkippableFact]
    public void Crop2D_AndItsBackward()
    {
        var backend = BackendOrSkip();
        const int batch = 2, channels = 2, inH = 7, inW = 9, outH = 3, outW = 4, offsetH = 2, offsetW = 5;
        float[] x = Values(batch * channels * inH * inW, 8);
        var cropped = new float[batch * channels * outH * outW];
        for (int p = 0; p < batch * channels; p++) for (int h = 0; h < outH; h++) for (int w = 0; w < outW; w++)
            cropped[(p * outH + h) * outW + w] = x[(p * inH + h + offsetH) * inW + w + offsetW];
        AssertClose("Crop2D", cropped, Run(backend, x, cropped.Length,
            (s, d) => backend.Crop2D(s, d, batch, channels, inH, inW, outH, outW, offsetH, offsetW)));

        float[] gradOutput = Values(cropped.Length, 9);
        var gradInput = new float[x.Length];   // zero outside the cropped window
        for (int p = 0; p < batch * channels; p++) for (int h = 0; h < outH; h++) for (int w = 0; w < outW; w++)
            gradInput[(p * inH + h + offsetH) * inW + w + offsetW] = gradOutput[(p * outH + h) * outW + w];
        AssertClose("Crop2DBackward", gradInput, Run(backend, gradOutput, gradInput.Length,
            (s, d) => backend.Crop2DBackward(s, d, batch, channels, inH, inW, outH, outW, offsetH, offsetW)));
    }

    [SkippableFact]
    public void PixelShuffle_AndItsBackward()
    {
        var backend = BackendOrSkip();
        const int batch = 2, channels = 3, inH = 2, inW = 5, scale = 2;
        int inChannels = channels * scale * scale, outH = inH * scale, outW = inW * scale;
        float[] x = Values(batch * inChannels * inH * inW, 10);
        var shuffled = new float[batch * channels * outH * outW];
        for (int b = 0; b < batch; b++) for (int c = 0; c < channels; c++) for (int h = 0; h < outH; h++) for (int w = 0; w < outW; w++)
        {
            int sourceChannel = c * scale * scale + (h % scale) * scale + (w % scale);
            shuffled[((b * channels + c) * outH + h) * outW + w] = x[((b * inChannels + sourceChannel) * inH + h / scale) * inW + w / scale];
        }
        AssertClose("PixelShuffle", shuffled, Run(backend, x, shuffled.Length, (s, d) => backend.PixelShuffle(s, d, batch, channels, inH, inW, scale)));

        // The shuffle is a permutation, so its backward is the inverse permutation.
        float[] gradOutput = Values(shuffled.Length, 11);
        var gradInput = new float[x.Length];
        for (int b = 0; b < batch; b++) for (int c = 0; c < channels; c++) for (int h = 0; h < outH; h++) for (int w = 0; w < outW; w++)
        {
            int sourceChannel = c * scale * scale + (h % scale) * scale + (w % scale);
            gradInput[((b * inChannels + sourceChannel) * inH + h / scale) * inW + w / scale] = gradOutput[((b * channels + c) * outH + h) * outW + w];
        }
        AssertClose("PixelShuffleBackward", gradInput, Run(backend, gradOutput, gradInput.Length,
            (s, d) => backend.PixelShuffleBackward(s, d, batch, channels, inH, inW, scale)));
    }

    [SkippableFact]
    public void Generators_OneHot_ExtractDiag_TriangularMask_Linspace()
    {
        var backend = BackendOrSkip();
        const int batch = 6, classes = 5;
        float[] labels = { 3, 0, 4, 1, 1, 2 };
        var oneHot = new float[batch * classes];
        for (int i = 0; i < batch; i++) oneHot[i * classes + (int)labels[i]] = 1f;
        AssertClose("OneHotKernel", oneHot, Run(backend, labels, oneHot.Length, (s, d) => backend.OneHotKernel(s, d, batch, classes)));

        const int n = 4, cols = 7;
        float[] matrix = Values(n * cols, 12);
        var diagonal = new float[n];
        for (int i = 0; i < n; i++) diagonal[i] = matrix[i * cols + i];
        AssertClose("ExtractDiagKernel", diagonal, Run(backend, matrix, n, (s, d) => backend.ExtractDiagKernel(s, d, n, cols)));

        const int rows = 5, maskCols = 6;
        foreach (int offset in new[] { 0, 2, -1 })
        {
            const float maskValue = -1e9f;
            var mask = new float[rows * maskCols];
            for (int r = 0; r < rows; r++) for (int c = 0; c < maskCols; c++) mask[r * maskCols + c] = c > r + offset ? maskValue : 0f;
            using var output = backend.AllocateBuffer(mask.Length);
            backend.Fill(output, -777f, mask.Length);
            backend.TriangularMask(output, rows, maskCols, offset, maskValue);
            AssertClose("TriangularMask(" + offset + ")", mask, backend.DownloadBuffer(output));
        }

        const int points = 11;
        const float start = -2.5f, step = 0.75f;
        var line = new float[points];
        for (int i = 0; i < points; i++) line[i] = start + step * i;
        using var lineOutput = backend.AllocateBuffer(points);
        backend.Fill(lineOutput, -777f, points);
        backend.LinspaceKernel(lineOutput, start, step, points);
        AssertClose("LinspaceKernel", line, backend.DownloadBuffer(lineOutput));
    }

    [SkippableFact]
    public void LogSoftmax_MatchesTheHost()
    {
        var backend = BackendOrSkip();
        const int outer = 4, inner = 9;
        float[] x = Values(outer * inner, 13, 8f);
        var expected = new float[x.Length];
        for (int r = 0; r < outer; r++)
        {
            double max = Enumerable.Range(0, inner).Max(i => (double)x[r * inner + i]);
            double logSum = max + Math.Log(Enumerable.Range(0, inner).Sum(i => Math.Exp(x[r * inner + i] - max)));
            for (int i = 0; i < inner; i++) expected[r * inner + i] = (float)(x[r * inner + i] - logSum);
        }
        AssertClose("LogSoftmax", expected, Run(backend, x, x.Length, (s, d) => backend.LogSoftmax(s, d, outer, inner)));
    }

    public static TheoryData<string> RowOps() => new() { "LogSoftmax", "Sparsemax", "TaylorSoftmax", "SphericalSoftmax" };

    // A row-wise op over [outer, inner] must give, for each row, what it gives for that row alone: true only when the
    // shader sees both extents.
    [SkippableTheory]
    [MemberData(nameof(RowOps))]
    public void RowOps_TreatEachRowIndependently(string op)
    {
        var backend = BackendOrSkip();
        const int outer = 5, inner = 6;
        float[] x = Values(outer * inner, 14, 6f);
        Action<IGpuBuffer, IGpuBuffer, int, int> run = op switch
        {
            "LogSoftmax" => backend.LogSoftmax,
            "Sparsemax" => backend.Sparsemax,
            "TaylorSoftmax" => backend.TaylorSoftmax,
            _ => backend.SphericalSoftmax,
        };
        float[] whole = Run(backend, x, x.Length, (s, d) => run(s, d, outer, inner));
        Assert.DoesNotContain(-777f, whole.Take(x.Length));
        for (int r = 0; r < outer; r++)
        {
            float[] row = x.Skip(r * inner).Take(inner).ToArray();
            float[] alone = Run(backend, row, inner, (s, d) => run(s, d, 1, inner));
            AssertClose(op + " row " + r, alone.Take(inner).ToArray(), whole.Skip(r * inner).Take(inner).ToArray(), 1e-6);
        }
        if (op is "Sparsemax" or "TaylorSoftmax")
            for (int r = 0; r < outer; r++)
                Assert.True(Math.Abs(whole.Skip(r * inner).Take(inner).Sum() - 1f) < 1e-4, $"{op} row {r} does not sum to 1");
    }

    [SkippableFact]
    public void ReduceBackwardKernels_SeeBothExtents()
    {
        var backend = BackendOrSkip();
        const int outer = 4, reduce = 6;
        float[] input = Values(outer * reduce, 15);
        float[] gradOutput = Values(outer, 16);
        var maxima = new float[outer];
        var means = new float[outer];
        var variances = new float[outer];
        for (int r = 0; r < outer; r++)
        {
            var row = input.Skip(r * reduce).Take(reduce).ToArray();
            maxima[r] = row.Max();
            means[r] = row.Average();
            variances[r] = row.Select(v => (v - means[r]) * (v - means[r])).Average();
        }

        using var go = backend.AllocateBuffer(gradOutput);
        using var inp = backend.AllocateBuffer(input);
        using var mx = backend.AllocateBuffer(maxima);
        using var mean = backend.AllocateBuffer(means);
        using var variance = backend.AllocateBuffer(variances);
        using var gi = backend.AllocateBuffer(outer * reduce);

        backend.Fill(gi, -777f, outer * reduce);
        backend.ReduceMaxBackward(go, inp, mx, gi, outer, reduce);
        var expectedMax = new float[outer * reduce];
        for (int i = 0; i < expectedMax.Length; i++) expectedMax[i] = input[i] == maxima[i / reduce] ? gradOutput[i / reduce] : 0f;
        AssertClose("ReduceMaxBackward", expectedMax, backend.DownloadBuffer(gi));

        // d var / d x_i = 2 (x_i - mean) / n, and d log var / d x_i is that over the variance.
        backend.Fill(gi, -777f, outer * reduce);
        backend.ReduceVarianceBackward(go, inp, mean, gi, outer, reduce);
        var expectedVariance = new float[outer * reduce];
        for (int i = 0; i < expectedVariance.Length; i++)
            expectedVariance[i] = gradOutput[i / reduce] * 2f * (input[i] - means[i / reduce]) / reduce;
        AssertClose("ReduceVarianceBackward", expectedVariance, backend.DownloadBuffer(gi), 1e-4);

        backend.Fill(gi, -777f, outer * reduce);
        backend.ReduceLogVarianceBackward(go, inp, mean, variance, gi, outer, reduce);
        float[] logVariance = backend.DownloadBuffer(gi);
        Assert.DoesNotContain(-777f, logVariance.Take(outer * reduce));
        for (int i = 0; i < outer * reduce; i++)
        {
            float expected = expectedVariance[i] / variances[i / reduce];
            Assert.True(Math.Abs(expected - logVariance[i]) <= 2e-3 * Math.Max(1.0, Math.Abs(expected)),
                $"ReduceLogVarianceBackward[{i}]: expected {expected:R}, vulkan {logVariance[i]:R}");
        }
    }

    // The symptom that exposed the bug: a sum's gradient is broadcast with an outer product against a ones vector.
    [SkippableFact]
    public void SumGradient_OnAVulkanEngine_IsNotZero()
    {
        var backend = BackendOrSkip();
        // Non-owning: the backend is a process-wide singleton.
        var engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend));
        var x = new Tensor<float>(new[] { 3, 4, 5 });
        var w = new Tensor<float>(new[] { 3, 4, 5 });
        float[] xv = Values(60, 17), wv = Values(60, 18);
        for (int i = 0; i < 60; i++) { x[i] = xv[i]; w[i] = wv[i]; }
        var previous = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            using var tape = new GradientTape<float>();
            var loss = engine.ReduceSum(engine.TensorMultiply(x, w), new[] { 0, 1, 2 }, keepDims: false);
            var gradients = tape.ComputeGradients(loss, new[] { x });
            AssertClose("d sum(x * w) / dx", wv, gradients[x].ToArray());
        }
        finally
        {
            AiDotNetEngine.Current = previous;
        }
    }

    // A kernel with a multi-word push block can no longer be launched with fewer words than it declares.
    [SkippableFact]
    public void Launch_RejectsAShortPushConstantArray()
    {
        var backend = BackendOrSkip();
        var method = typeof(VulkanBackend).GetMethod("GlslUnaryOp",
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic, null,
            new[] { typeof(string), typeof(IGpuBuffer), typeof(IGpuBuffer), typeof(int), typeof(uint) }, null);
        Assert.NotNull(method);
        using var a = backend.AllocateBuffer(16);
        using var b = backend.AllocateBuffer(16);
        var error = Assert.Throws<System.Reflection.TargetInvocationException>(
            () => method!.Invoke(backend, new object[] { VulkanGlslKernels.TileLastAxisGlsl, a, b, 16, 12u }));
        Assert.IsType<ArgumentException>(error.InnerException);
    }
}
