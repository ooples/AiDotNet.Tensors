using System;
using System.Linq;
using System.Reflection;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// IEngine.CausalConv1dStep: one decode position of a causal depthwise convolution with the history shifted in place,
/// against a float64 causal convolution over the whole sequence, eagerly and replayed by a captured plan.
/// </summary>
[Collection("DirectGpuSerial")]
public class CausalConv1dStepTests
{
    private sealed class EngineScope : IDisposable
    {
        private readonly IEngine _prior = AiDotNetEngine.Current;
        public EngineScope(IEngine engine) => AiDotNetEngine.Current = engine;
        public void Dispose() => AiDotNetEngine.Current = _prior;
    }

    private static float[] R(Random rng, int n) => Enumerable.Range(0, n).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();

    // The full causal convolution in float64 at position t (zero padding before the sequence start).
    private static double[] Reference(float[][] xs, int t, float[] w, float[]? bias, bool silu, int batch, int channels, int K)
    {
        var y = new double[batch * channels];
        for (int i = 0; i < y.Length; i++)
        {
            int c = i % channels;
            double acc = bias?[c] ?? 0;
            for (int m = 0; m < K && t - m >= 0; m++) acc += (double)xs[t - m][i] * w[c * K + m];
            y[i] = silu ? acc / (1 + Math.Exp(-acc)) : acc;
        }
        return y;
    }

    private static void Close(double[] expect, float[] got, string what)
    {
        Assert.Equal(expect.Length, got.Length);
        for (int i = 0; i < got.Length; i++)
            Assert.True(Math.Abs(got[i] - expect[i]) <= 1e-5 * (1 + Math.Abs(expect[i])), $"{what}[{i}]: {got[i]} vs {expect[i]}");
    }

    private static void RunEager(IEngine engine, int K, int batch, int channels, bool withBias, bool silu)
    {
        var rng = new Random(K * 97 + channels);
        var w = R(rng, channels * K);
        var bias = withBias ? R(rng, channels) : null;
        var history = new Tensor<float>([K - 1, batch, channels]);
        var xs = new float[K + 3][];
        using (new EngineScope(engine))
            for (int t = 0; t < xs.Length; t++)
            {
                xs[t] = R(rng, batch * channels);
                var y = engine.CausalConv1dStep(history, new Tensor<float>([batch, channels], new Vector<float>(xs[t])),
                    new Tensor<float>([channels, K], new Vector<float>(w)), bias is null ? null : new Tensor<float>([channels], new Vector<float>(bias)), silu);
                Assert.Equal(new[] { batch, channels }, y.Shape.ToArray());
                Close(Reference(xs, t, w, bias, silu, batch, channels, K), y.ToArray(), $"t{t} y");
                // The history now holds the K - 1 most recent inputs, most recent first (zeros before the start).
                var h = history.ToArray();
                for (int m = 0; m < K - 1; m++)
                    for (int i = 0; i < batch * channels; i++)
                        Assert.Equal(t - m >= 0 ? xs[t - m][i] : 0f, h[m * batch * channels + i]);
            }
    }

    [Theory]
    [InlineData(4, 2, 5, true, true)]
    [InlineData(1, 3, 4, false, false)]
    [InlineData(17, 1, 3, true, false)]
    public void The_cpu_step_matches_the_causal_convolution(int K, int batch, int channels, bool withBias, bool silu)
        => RunEager(new CpuEngine(), K, batch, channels, withBias, silu);

    // The orientation cross-check against an independent op: CausalConv1dStep's kernel is newest-first (w[:, 0] weights
    // the current input) while DepthwiseConv1D, like PyTorch conv1d, weights it by w[:, K - 1]. Stepping a zero history
    // over a whole sequence with the tap-reversed kernel must equal DepthwiseConv1D with padding K - 1 truncated to the
    // first seqLen outputs; the unreversed kernel must not (the reference above uses the step's own orientation, so it
    // cannot catch a flipped layout).
    private static void RunAgainstDepthwiseConv1D(IEngine engine, int K, int batch, int channels, int seqLen)
    {
        var rng = new Random(K * 131 + channels * 7 + seqLen);
        var torchW = R(rng, channels * K);                                   // [channels, 1, K], oldest-first
        var x = R(rng, batch * channels * seqLen);                           // [batch, channels, seqLen]
        var stepW = new float[channels * K];                                 // [channels, K], newest-first
        for (int c = 0; c < channels; c++)
            for (int m = 0; m < K; m++) stepW[c * K + m] = torchW[c * K + (K - 1 - m)];

        var full = new CpuEngine().DepthwiseConv1D(new Tensor<float>([batch, channels, seqLen], new Vector<float>(x)),
            new Tensor<float>([channels, 1, K], new Vector<float>(torchW)), 1, K - 1);
        Assert.Equal(new[] { batch, channels, seqLen + K - 1 }, full.Shape.ToArray());
        var f = full.ToArray();
        int fullLen = seqLen + K - 1;

        var history = new Tensor<float>([K - 1, batch, channels]);
        var unreversedHistory = new Tensor<float>([K - 1, batch, channels]);
        var weight = new Tensor<float>([channels, K], new Vector<float>(stepW));
        var unreversed = new Tensor<float>([channels, K], new Vector<float>(torchW));
        double maxUnreversedGap = 0;
        using (new EngineScope(engine))
            for (int t = 0; t < seqLen; t++)
            {
                var xt = new float[batch * channels];
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < channels; c++) xt[b * channels + c] = x[(b * channels + c) * seqLen + t];
                var y = engine.CausalConv1dStep(history, new Tensor<float>([batch, channels], new Vector<float>(xt)), weight, null, false).ToArray();
                var yu = engine.CausalConv1dStep(unreversedHistory, new Tensor<float>([batch, channels], new Vector<float>(xt)), unreversed, null, false).ToArray();
                var expect = new double[batch * channels];
                for (int b = 0; b < batch; b++)
                    for (int c = 0; c < channels; c++) expect[b * channels + c] = f[(b * channels + c) * fullLen + t];
                Close(expect, y, $"t{t} reversed-kernel step vs DepthwiseConv1D");
                for (int i = 0; i < yu.Length; i++) maxUnreversedGap = Math.Max(maxUnreversedGap, Math.Abs(yu[i] - expect[i]));
            }
        if (K > 1) Assert.True(maxUnreversedGap > 1e-3, $"the unreversed kernel also matched DepthwiseConv1D (gap {maxUnreversedGap})");
    }

    [Theory]
    [InlineData(4, 2, 5, 9)]
    [InlineData(1, 3, 4, 6)]
    [InlineData(17, 1, 3, 20)]
    public void The_cpu_step_with_a_reversed_kernel_matches_DepthwiseConv1D(int K, int batch, int channels, int seqLen)
        => RunAgainstDepthwiseConv1D(new CpuEngine(), K, batch, channels, seqLen);

    [SkippableTheory]
    [InlineData(4, 2, 5, 9)]
    [InlineData(16, 2, 33, 20)]
    [InlineData(17, 1, 3, 20)]
    public void The_gpu_step_with_a_reversed_kernel_matches_DepthwiseConv1D(int K, int batch, int channels, int seqLen)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu) RunAgainstDepthwiseConv1D(gpu!, K, batch, channels, seqLen);
    }

    // K = 17 exceeds the kernel's register history and takes the composed path.
    [SkippableTheory]
    [InlineData(4, 2, 5, true, true)]
    [InlineData(4, 4, 300, false, true)]
    [InlineData(1, 3, 4, false, false)]
    [InlineData(16, 2, 33, true, false)]
    [InlineData(17, 1, 3, true, false)]
    public void The_gpu_step_matches_the_causal_convolution(int K, int batch, int channels, bool withBias, bool silu)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu) RunEager(gpu!, K, batch, channels, withBias, silu);
    }

    // A compiled plan stepping the convolution on a persistent history: every execution, the capture call included,
    // shifts the history once, and the captured graph copies nothing.
    [SkippableTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void A_compiled_plan_shifts_the_history_once_per_execution(bool capture)
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        Skip.If(gpu is null || !gpu.IsGpuAvailable, "GPU backend did not resolve.");
        using (gpu)
        {
            var prior = CompiledInferencePlan<float>.InferenceCudaGraphOverride;
            CompiledInferencePlan<float>.InferenceCudaGraphOverride = capture;
            try
            {
                const int K = 4, batch = 2, channels = 8;
                var rng = new Random(5);
                var w = R(rng, channels * K);
                var history = new Tensor<float>([K - 1, batch, channels]);
                var x = new Tensor<float>([batch, channels]);
                var weight = new Tensor<float>([channels, K], new Vector<float>(w));
                CompiledInferencePlan<float> plan;
                using (new EngineScope(gpu))
                using (var scope = GraphMode.EnableInference())
                {
                    var y = gpu!.TensorMultiplyScalar(gpu.CausalConv1dStep(history, x, weight, null, true), 2f);
                    plan = scope.CompileInference(y, new[] { x }, Array.Empty<(Tensor<float>, Tensor<float>)>());
                }
                using (plan)
                {
                    var xs = new float[7][];
                    for (int t = 0; t < xs.Length; t++)
                    {
                        xs[t] = R(rng, batch * channels);
                        plan.SetInputs(new[] { new Tensor<float>([batch, channels], new Vector<float>(xs[t])) });
                        var y = plan.Execute().ToArray();
                        Close(Reference(xs, t, w, null, true, batch, channels, K).Select(e => 2 * e).ToArray(), y, $"t{t} y");
                    }
                    if (capture && gpu!.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.CUDA.CudaBackend cuda)
                    {
                        var exec = typeof(CompiledInferencePlan<float>).GetField("_graphExec", BindingFlags.NonPublic | BindingFlags.Instance)!;
                        Assert.True(exec.GetValue(plan) is IntPtr h && h != IntPtr.Zero, "the plan is not replaying a captured graph");
                        Assert.Equal(0, cuda.LastCaptureMemcpyNodeCount);
                    }
                }
            }
            finally { CompiledInferencePlan<float>.InferenceCudaGraphOverride = prior; }
        }
    }
}
