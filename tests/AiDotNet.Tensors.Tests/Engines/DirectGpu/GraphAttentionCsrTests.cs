using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Correctness of <see cref="CpuEngine.GraphAttentionCsr{T}"/>: the forward against a direct per-row reference, the
/// backward against central finite differences, and the GPU kernels against the CPU implementation (dropout included).
/// </summary>
[Collection("DirectGpuSerial")]
public class GraphAttentionCsrTests
{
    private const int Batch = 2, Nodes = 5, Heads = 2, Features = 3;

    // Self-loops plus an irregular neighbourhood (row 4 has only itself).
    private static CsrGraph Graph() => new CsrGraph(Nodes,
        new[] { 0, 3, 5, 8, 10, 11 },
        new[] { 0, 1, 3, 1, 2, 0, 2, 4, 3, 4, 4 });

    private static double[] Gen(int n, int s)
    {
        var a = new double[n];
        for (int i = 0; i < n; i++) a[i] = Math.Sin(0.37 * (i + 1) + 1.3 * s) * 0.8;
        return a;
    }

    private static double[] Reference(double[] wh, double[] attn, CsrGraph g, double slope)
    {
        var outp = new double[wh.Length];
        int width = Heads * Features;
        for (int b = 0; b < Batch; b++)
            for (int i = 0; i < Nodes; i++)
                for (int h = 0; h < Heads; h++)
                {
                    int start = g.RowPointers[i], end = g.RowPointers[i + 1];
                    var e = new double[end - start];
                    double self = 0;
                    for (int f = 0; f < Features; f++)
                        self += attn[h * 2 * Features + f] * wh[(b * Nodes + i) * width + h * Features + f];
                    double max = double.NegativeInfinity;
                    for (int k = start; k < end; k++)
                    {
                        int j = g.ColumnIndices[k];
                        double s = self;
                        for (int f = 0; f < Features; f++)
                            s += attn[h * 2 * Features + Features + f] * wh[(b * Nodes + j) * width + h * Features + f];
                        e[k - start] = s > 0 ? s : slope * s;
                        max = Math.Max(max, e[k - start]);
                    }
                    double sum = 0;
                    for (int k = 0; k < e.Length; k++) { e[k] = Math.Exp(e[k] - max); sum += e[k]; }
                    for (int k = start; k < end; k++)
                    {
                        int j = g.ColumnIndices[k];
                        for (int f = 0; f < Features; f++)
                            outp[(b * Nodes + i) * width + h * Features + f] +=
                                e[k - start] / sum * wh[(b * Nodes + j) * width + h * Features + f];
                    }
                }
        return outp;
    }

    [Fact]
    public void Forward_MatchesPerRowReference()
    {
        var engine = new CpuEngine();
        var g = Graph();
        var whData = Gen(Batch * Nodes * Heads * Features, 1);
        var attnData = Gen(Heads * 2 * Features, 2);
        var wh = new Tensor<double>(whData, new[] { Batch * Nodes, Heads * Features });
        var attn = new Tensor<double>(attnData, new[] { Heads, 2 * Features });
        var actual = engine.GraphAttentionCsr(wh, attn, g, 0.2).ToArray();
        var expected = Reference(whData, attnData, g, 0.2);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-10, $"out[{i}] expected={expected[i]} actual={actual[i]}");
    }

    [Fact]
    public void Backward_MatchesFiniteDifferences()
    {
        var engine = new CpuEngine();
        var g = Graph();
        var whData = Gen(Batch * Nodes * Heads * Features, 3);
        var attnData = Gen(Heads * 2 * Features, 4);
        var weights = Gen(whData.Length, 5);

        double Loss(double[] w, double[] a)
        {
            var o = Reference(w, a, g, 0.2);
            double l = 0;
            for (int i = 0; i < o.Length; i++) l += o[i] * weights[i];
            return l;
        }

        var wh = new Tensor<double>((double[])whData.Clone(), new[] { Batch * Nodes, Heads * Features });
        var attn = new Tensor<double>((double[])attnData.Clone(), new[] { Heads, 2 * Features });
        var weightT = new Tensor<double>((double[])weights.Clone(), new[] { Batch * Nodes, Heads * Features });
        double[] gWh, gAttn;
        using (var tape = new GradientTape<double>())
        {
            var outp = engine.GraphAttentionCsr(wh, attn, g, 0.2);
            var loss = engine.ReduceSum(engine.TensorMultiply(outp, weightT), new[] { 0, 1 }, keepDims: false);
            var grads = tape.ComputeGradients(loss, new[] { wh, attn });
            gWh = grads[wh].ToArray();
            gAttn = grads[attn].ToArray();
        }

        const double eps = 1e-6;
        for (int i = 0; i < whData.Length; i++)
        {
            var p = (double[])whData.Clone(); p[i] += eps;
            var m = (double[])whData.Clone(); m[i] -= eps;
            double fd = (Loss(p, attnData) - Loss(m, attnData)) / (2 * eps);
            Assert.True(Math.Abs(fd - gWh[i]) < 1e-6, $"dWh[{i}] fd={fd} analytic={gWh[i]}");
        }
        for (int i = 0; i < attnData.Length; i++)
        {
            var p = (double[])attnData.Clone(); p[i] += eps;
            var m = (double[])attnData.Clone(); m[i] -= eps;
            double fd = (Loss(whData, p) - Loss(whData, m)) / (2 * eps);
            Assert.True(Math.Abs(fd - gAttn[i]) < 1e-6, $"dAttn[{i}] fd={fd} analytic={gAttn[i]}");
        }
    }

    [SkippableTheory]
    [InlineData(0.0)]
    [InlineData(0.4)]
    public void Gpu_MatchesCpu_ForwardAndBackward(double rate)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); } catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.IfNot(gpu is not null && gpu.IsGpuAvailable, "GPU backend did not resolve.");
        bool savedFallback = DirectGpuTensorEngine.ThrowOnGpuKernelFallback;
        try
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = true;
            var cpu = new CpuEngine();
            var g = Graph();
            var shape = new[] { Batch * Nodes, Heads * Features };
            var wh = new Tensor<float>(Array.ConvertAll(Gen(Batch * Nodes * Heads * Features, 6), x => (float)x), shape);
            var attn = new Tensor<float>(Array.ConvertAll(Gen(Heads * 2 * Features, 7), x => (float)x), new[] { Heads, 2 * Features });
            var gradOut = new Tensor<float>(Array.ConvertAll(Gen(wh.Length, 8), x => (float)x), shape);
            const uint seed = 12345u;

            if (gpu!.GetBackend() is not AiDotNet.Tensors.Engines.DirectGpu.IGraphAttentionBackend)
            {
                var ex = Assert.Throws<NotSupportedException>(() => gpu.GraphAttentionCsrForward(wh, attn, g, 0.2, rate, seed));
                Assert.Contains(gpu.GetBackend()!.BackendName, ex.Message);
                Assert.Contains("CpuEngine", ex.Message);
                return;
            }

            Close(cpu.GraphAttentionCsrForward(wh, attn, g, 0.2, rate, seed).ToArray(),
                gpu!.GraphAttentionCsrForward(wh, attn, g, 0.2, rate, seed).ToArray(), "forward");
            var (cWh, cAttn) = cpu.GraphAttentionCsrBackward(gradOut, wh, attn, g, 0.2, rate, seed);
            var (gWh, gAttn) = gpu.GraphAttentionCsrBackward(gradOut, wh, attn, g, 0.2, rate, seed);
            Close(cWh.ToArray(), gWh.ToArray(), "dWh");
            Close(cAttn.ToArray(), gAttn.ToArray(), "dAttn");
        }
        finally
        {
            DirectGpuTensorEngine.ThrowOnGpuKernelFallback = savedFallback;
            gpu?.Dispose();
        }
    }

    [Fact]
    public void InvalidDropout_IsRejected()
    {
        var engine = new CpuEngine();
        var wh = new Tensor<float>(new[] { 1f }, new[] { 1, 1 });
        var attn = new Tensor<float>(new[] { 0f, 0f }, new[] { 1, 2 });
        var graph = new CsrGraph(1, new[] { 0, 1 }, new[] { 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.GraphAttentionCsr(wh, attn, graph, dropoutRate: double.NaN));
    }

    [Theory]
    [InlineData(0, 0, 1, 1)]
    [InlineData(1, 0, 1, 1)]
    public void DuplicateNeighbors_AreRejected(int a, int b, int c, int d)
        => Assert.Throws<ArgumentException>(() => new CsrGraph(2, new[] { 0, 3, 4 }, new[] { a, b, c, d }));

    [Fact]
    public void CreateGraphBackward_PreservesSecondDerivative()
    {
        var engine = new CpuEngine();
        var wh = new Tensor<double>(new[] { 1.0, 3.0 }, new[] { 2, 1 });
        var attn = new Tensor<double>(new double[2], new[] { 1, 2 });
        var graph = new CsrGraph(2, new[] { 0, 2, 4 }, new[] { 0, 1, 0, 1 });
        using var tape = new GradientTape<double>(new GradientTapeOptions { Persistent = true });
        var output = engine.GraphAttentionCsr(wh, attn, graph);
        var loss = engine.ReduceSum(engine.TensorMultiply(output, output), new[] { 0, 1 }, false);
        var first = tape.ComputeGradients(loss, new[] { wh }, createGraph: true);
        var sum = engine.ReduceSum(first[wh], new[] { 0, 1 }, false);
        var second = tape.ComputeGradients(sum, new[] { wh });
        Assert.Equal(new[] { 2.0, 2.0 }, second[wh].ToArray());
    }

    private static void Close(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) < 1e-4f * (1 + Math.Abs(expected[i])),
                $"{what}[{i}] cpu={expected[i]} gpu={actual[i]}");
    }
}
