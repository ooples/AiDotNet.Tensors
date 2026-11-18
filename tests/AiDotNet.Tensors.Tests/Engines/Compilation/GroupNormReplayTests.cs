using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A training trace records GroupNorm lazily: no eager normalization mid-trace, replay-time statistics in a holder the
/// objective signature compares by size only, and lazy mean/variance outputs scheduled after each replay's refresh.
/// </summary>
[Collection("CompilationGlobalState")]
public sealed class GroupNormReplayTests
{
    private const double Epsilon = 1e-5;

    // loss = sum(w * GroupNorm(x)) + sum(mean) + sum(variance); statistics carry no gradient (as on the eager tape).
    private static double Reference(double[] x, double[] gamma, double[] beta, double[] w, int[] shape, int groups,
        out double statisticSum)
    {
        int batch = shape[0], channels = shape[1], spatial = 1;
        for (int d = 2; d < shape.Length; d++) spatial *= shape[d];
        int perGroup = channels / groups;
        double loss = 0;
        statisticSum = 0;
        for (int b = 0; b < batch; b++)
        for (int g = 0; g < groups; g++)
        {
            double mean = 0, variance = 0;
            int count = perGroup * spatial;
            for (int c = g * perGroup; c < (g + 1) * perGroup; c++)
                for (int s = 0; s < spatial; s++) mean += x[(b * channels + c) * spatial + s];
            mean /= count;
            for (int c = g * perGroup; c < (g + 1) * perGroup; c++)
                for (int s = 0; s < spatial; s++)
                {
                    double centered = x[(b * channels + c) * spatial + s] - mean;
                    variance += centered * centered;
                }
            variance /= count;
            statisticSum += mean + variance;
            double invStd = 1.0 / Math.Sqrt(variance + Epsilon);
            for (int c = g * perGroup; c < (g + 1) * perGroup; c++)
                for (int s = 0; s < spatial; s++)
                {
                    int i = (b * channels + c) * spatial + s;
                    loss += w[i] * (gamma[c] * (x[i] - mean) * invStd + beta[c]);
                }
        }
        return loss;
    }

    public static TheoryData<int[], int> Shapes() => new()
    {
        { new[] { 2, 4, 3 }, 2 },       // rank 3 [B, C, L] (1D audio decoders)
        { new[] { 2, 4, 2, 2 }, 2 },    // rank 4 [B, C, H, W]
        { new[] { 1, 6, 2, 1, 2 }, 3 }, // rank 5 [B, C, D, H, W]
    };

    [Theory]
    [MemberData(nameof(Shapes))]
    public void Double_closure_objective_rebinds_and_matches_finite_differences(int[] shape, int groups)
    {
        var engine = new CpuEngine();
        int channels = shape[1];
        int count = shape.Aggregate(1, (a, d) => a * d);
        var x = new Tensor<double>(shape);
        var gamma = new Tensor<double>(new[] { channels });
        var beta = new Tensor<double>(new[] { channels });
        var parameters = new[] { x, gamma, beta };
        var weights = new Tensor<double>(shape);
        int sampledGroups = groups;
        Tensor<double> Objective()
        {
            var normalized = engine.GroupNorm(x, sampledGroups, gamma, beta, Epsilon, out var mean, out var variance);
            var weighted = engine.ReduceSum(engine.TensorMultiply(normalized, weights), null);
            return engine.TensorAdd(weighted,
                engine.TensorAdd(engine.ReduceSum(mean, null), engine.ReduceSum(variance, null)));
        }

        using var recording = ClosureObjectiveGuard<double>.BeginRecording();
        CompiledTrainingPlan<double> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
            plan = scope.CompileTraining(parameters, Objective());
        using (plan)
        {
            var guard = ClosureObjectiveGuard<double>.TakeRecorded(out var reason);
            Assert.True(guard is not null, reason);
            var signed = Assert.IsType<ClosureObjectiveGuard<double>>(guard);
            signed.AttachTo(plan);
            for (int step = 0; step < 4; step++)
            {
                var xs = new double[count];
                var gs = new double[channels];
                var bs = new double[channels];
                var ws = new double[count];
                for (int i = 0; i < count; i++)
                {
                    xs[i] = Math.Sin(1.3 * i + 0.7 * step) * (1 + 0.25 * step) + 0.1 * i;
                    ws[i] = Math.Cos(0.9 * i - 0.4 * step);
                }
                for (int c = 0; c < channels; c++) { gs[c] = 0.8 + 0.1 * c - 0.05 * step; bs[c] = 0.2 * step - 0.07 * c; }
                Load(x, xs); Load(gamma, gs); Load(beta, bs);
                // A fresh leaf and fresh statistics holder must match the same compiled objective.
                weights = new Tensor<double>(ws, shape);
                Assert.True(signed.TryRebind(Objective, parameters, out _) == ClosureGuardResult.Matched, signed.LastReason);

                double expected = Reference(xs, gs, bs, ws, shape, groups, out double statistics) + statistics;
                Assert.Equal(expected, plan.Step().ToArray()[0], 10);

                var all = xs.Concat(gs).Concat(bs).ToArray();
                int offset = 0;
                for (int tensor = 0; tensor < parameters.Length; tensor++)
                {
                    var gradients = plan.Gradients[tensor].ToArray();
                    for (int index = 0; index < gradients.Length; index++, offset++)
                    {
                        const double delta = 1e-5;
                        double saved = all[offset];
                        all[offset] = saved + delta;
                        double plus = Reference(Slice(all, 0, count), Slice(all, count, channels),
                            Slice(all, count + channels, channels), ws, shape, groups, out _);
                        all[offset] = saved - delta;
                        double minus = Reference(Slice(all, 0, count), Slice(all, count, channels),
                            Slice(all, count + channels, channels), ws, shape, groups, out _);
                        all[offset] = saved;
                        double fd = (plus - minus) / (2 * delta);
                        Assert.True(Math.Abs(fd - gradients[index]) < 2e-6,
                            $"shape=[{string.Join(",", shape)}] step={step} parameter={offset}: expected {fd:R}, got {gradients[index]:R}");
                    }
                }
            }
        }
    }

    [Fact]
    public void Float_compiled_training_matches_eager_tape_with_statistics_in_the_loss()
    {
        const int groups = 2;
        var shape = new[] { 2, 4, 3, 2 };
        int count = shape.Aggregate(1, (a, d) => a * d);
        var engine = new CpuEngine();
        var x = new Tensor<float>(shape);
        var gamma = new Tensor<float>(new[] { 4 });
        var beta = new Tensor<float>(new[] { 4 });
        var weights = new Tensor<float>(Enumerable.Range(0, count).Select(i => (float)Math.Cos(0.37 * i)).ToArray(), shape);
        var parameters = new[] { x, gamma, beta };
        Tensor<float> Loss(IEngine e, Tensor<float> xi, Tensor<float> gi, Tensor<float> bi)
        {
            var normalized = e.GroupNorm(xi, groups, gi, bi, Epsilon, out var mean, out var variance);
            return e.TensorAdd(e.ReduceSum(e.TensorMultiply(normalized, weights), null),
                e.ReduceSum(e.TensorMultiply(mean, variance), null));
        }

        Fill(x, gamma, beta, 0);
        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.EnableTraining(parameters))
        {
            var traced = Loss(engine, x, gamma, beta);
            // The trace ran nothing: the normalization and its statistics are unrealized lazy outputs.
            Assert.NotNull(traced.LazySource);
            plan = scope.CompileTraining(parameters, traced);
        }

        using (plan)
        {
            for (int step = 0; step < 3; step++)
            {
                Fill(x, gamma, beta, step);
                float compiled = plan.Step().ToArray()[0];

                var ex = new Tensor<float>(x.ToArray(), shape);
                var eg = new Tensor<float>(gamma.ToArray(), new[] { 4 });
                var eb = new Tensor<float>(beta.ToArray(), new[] { 4 });
                using var tape = new GradientTape<float>();
                var loss = Loss(new CpuEngine(), ex, eg, eb);
                var grads = tape.ComputeGradients(loss, new[] { ex, eg, eb });

                Assert.Equal(loss.ToArray()[0], compiled, 4);
                var eager = new[] { grads[ex].ToArray(), grads[eg].ToArray(), grads[eb].ToArray() };
                for (int p = 0; p < 3; p++)
                {
                    var actual = plan.Gradients[p].ToArray();
                    Assert.Equal(eager[p].Length, actual.Length);
                    for (int i = 0; i < actual.Length; i++)
                        Assert.True(Math.Abs(eager[p][i] - actual[i]) <= 1e-4f,
                            $"step {step}, parameter {p}[{i}]: eager {eager[p][i]:R} compiled {actual[i]:R}");
                }
            }
        }
    }

    [Fact]
    public void Probe_trace_records_group_norm_without_running_it()
    {
        var engine = new CpuEngine();
        var x = new Tensor<double>(new[] { 1, 4, 3 });
        var gamma = new Tensor<double>(new[] { 4 });
        var beta = new Tensor<double>(new[] { 4 });
        gamma.Fill(1.0);
        var parameters = new[] { x, gamma, beta };
        using var scope = GraphMode.EnableTraining(parameters);
        var normalized = engine.GroupNorm(x, 2, gamma, beta, Epsilon, out var mean, out var variance);
        var node = Assert.IsType<LazyNode<double>>(normalized.LazySource);
        Assert.False(node.IsRealized);
        Assert.Equal("GroupNorm", node.OpName);
        Assert.Equal(new[] { 1, 2 }, mean.Shape.ToArray());
        Assert.Equal(new[] { 1, 2 }, variance.Shape.ToArray());
        var state = Assert.IsAssignableFrom<object[]>(node.SavedState);
        Assert.Equal(2, Assert.IsType<int>(state[0]));
        Assert.Same(state[1], state[2]);
        Assert.Equal(Epsilon, Assert.IsType<double>(state[3]));
        Assert.Equal(2, Assert.IsAssignableFrom<ISignatureScratch>(state[1]).SignatureLength);
        // Reading a statistic realizes the forward first, so the holder is populated before the read.
        var means = mean.ToArray();
        Assert.Equal(2, means.Length);
        Assert.All(means, m => Assert.Equal(0.0, m));
    }

    [SkippableFact]
    public void Gpu_engine_records_group_norm_in_a_training_trace()
    {
        DirectGpuTensorEngine? gpu = null;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        bool required = Environment.GetEnvironmentVariable("AIDOTNET_REQUIRE_GPU_TESTS") == "1";
        if (gpu is null || !gpu.IsGpuAvailable)
        {
            gpu?.Dispose();
            Assert.False(required, "The validation lane requires a working GPU backend.");
            Skip.If(true, "No direct GPU backend available.");
        }
        using (gpu)
        {
            var engine = Assert.IsType<DirectGpuTensorEngine>(gpu);
            var x = new Tensor<float>(Enumerable.Range(0, 24).Select(i => (float)Math.Sin(i)).ToArray(), new[] { 2, 4, 3 });
            var gamma = new Tensor<float>(Enumerable.Repeat(1f, 4).ToArray(), new[] { 4 });
            var beta = new Tensor<float>(new[] { 4 });
            var parameters = new[] { x, gamma, beta };
            var reference = new CpuEngine().GroupNorm(
                new Tensor<float>(x.ToArray(), new[] { 2, 4, 3 }), 2, gamma, beta, Epsilon, out _, out _).ToArray();
            using var scope = GraphMode.EnableTraining(parameters);
            // A direct call on the device engine type must record too, not run eagerly into a constant.
            var normalized = engine.GroupNorm(x, 2, gamma, beta, Epsilon, out _, out _);
            Assert.IsType<LazyNode<float>>(normalized.LazySource);
            var values = normalized.ToArray();
            for (int i = 0; i < values.Length; i++) Assert.Equal(reference[i], values[i], 4);
        }
    }

    private static void Fill(Tensor<float> x, Tensor<float> gamma, Tensor<float> beta, int step)
    {
        for (int i = 0; i < x.Length; i++) x.SetFlat(i, (float)(Math.Sin(0.61 * i + step) * (1 + 0.3 * step)));
        for (int c = 0; c < gamma.Length; c++)
        {
            gamma.SetFlat(c, 0.9f + 0.1f * c - 0.05f * step);
            beta.SetFlat(c, 0.1f * step - 0.05f * c);
        }
        x.IncrementVersion(); gamma.IncrementVersion(); beta.IncrementVersion();
    }

    private static void Load(Tensor<double> tensor, double[] values)
    {
        for (int i = 0; i < values.Length; i++) tensor.SetFlat(i, values[i]);
        tensor.IncrementVersion();
    }

    private static double[] Slice(double[] all, int start, int length)
    {
        var result = new double[length];
        Array.Copy(all, start, result, 0, length);
        return result;
    }
}
