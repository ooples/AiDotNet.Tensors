using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Manual CUDA repro for the HarmonicEngine Cortex fused-training memory envelope.
/// This is intentionally opt-in because the default target shape can use many GB
/// of device memory and is meant for local bug isolation, not CI.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class CortexTransformerMemoryReproTests
{
    private readonly ITestOutputHelper _output;

    public CortexTransformerMemoryReproTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void CortexLikeD256L4CompiledStep_MemoryEnvelopeProbe()
    {
        Skip.IfNot(
            string.Equals(Environment.GetEnvironmentVariable("AIDOTNET_RUN_CORTEX_MEMORY_REPRO"), "1", StringComparison.Ordinal),
            "Manual CUDA memory-envelope repro. Set AIDOTNET_RUN_CORTEX_MEMORY_REPRO=1 to run.");

        using var gpu = new DirectGpuTensorEngine();
        Skip.IfNot(gpu.SupportsGpu, "DirectGpu backend is not available.");

        var previousEngine = AiDotNetEngine.Current;
        AiDotNetEngine.Current = gpu;
        try
        {
            int batch = ReadInt("AIDOTNET_REPRO_BATCH", 1024);
            int seqLen = ReadInt("AIDOTNET_REPRO_SEQLEN", 64);
            int dModel = ReadInt("AIDOTNET_REPRO_DMODEL", 256);
            int ffDim = ReadInt("AIDOTNET_REPRO_FFDIM", 512);
            int layers = ReadInt("AIDOTNET_REPRO_LAYERS", 4);
            int steps = ReadInt("AIDOTNET_REPRO_STEPS", 2);
            int maxLiveMb = ReadInt("AIDOTNET_REPRO_MAX_LIVE_MB", 0);
            int maxPeakMb = ReadInt("AIDOTNET_REPRO_MAX_PEAK_MB", 0);

            _output.WriteLine(
                $"Cortex-like repro shape: B={batch}, S={seqLen}, D={dModel}, FF={ffDim}, L={layers}, steps={steps}");

            var input = Rand([batch, seqLen, dModel], seed: 11, scale: 0.02f);
            var target = Rand([batch, seqLen, dModel], seed: 12, scale: 0.02f);

            var parameters = new Tensor<float>[layers * 6];
            for (int layer = 0; layer < layers; layer++)
            {
                int o = layer * 6;
                parameters[o + 0] = Ones([dModel]);
                parameters[o + 1] = Zeros([dModel]);
                parameters[o + 2] = Rand([dModel, ffDim], seed: 1000 + layer * 10 + 2, scale: 0.01f);
                parameters[o + 3] = Zeros([ffDim]);
                parameters[o + 4] = Rand([ffDim, dModel], seed: 1000 + layer * 10 + 4, scale: 0.01f);
                parameters[o + 5] = Zeros([dModel]);
            }

            double pL1Before = ParamL1(parameters);
            float[][] parameterSnapshot = SnapshotParameters(parameters);
            long memoryBaselineBytes = GpuMemoryTracker.LiveBytes;
            GpuMemoryTracker.ResetPeak();

            ICompiledTrainingPlan<float> plan;
            using (var scope = GraphMode.Enable())
            {
                var x = input;
                for (int layer = 0; layer < layers; layer++)
                {
                    int o = layer * 6;
                    var gamma = parameters[o + 0];
                    var beta = parameters[o + 1];
                    var w1 = parameters[o + 2];
                    var b1 = parameters[o + 3];
                    var w2 = parameters[o + 4];
                    var b2 = parameters[o + 5];

                    var normed = gpu.LayerNorm(x, gamma, beta, 1e-5, out _, out _);
                    var flat = gpu.Reshape(normed, [batch * seqLen, dModel]);
                    var hidden = gpu.FusedLinear(flat, w1, b1, FusedActivationType.ReLU);
                    var projected = gpu.FusedLinear(hidden, w2, b2, FusedActivationType.None);
                    var projected3D = gpu.Reshape(projected, [batch, seqLen, dModel]);
                    x = gpu.TensorAdd(x, projected3D);
                }

                var diff = gpu.TensorSubtract(x, target);
                var sq = gpu.TensorMultiply(diff, diff);
                var loss = gpu.ReduceSum(sq, null);
                plan = scope.CompileTraining(parameters, loss);
            }

            using (plan)
            {
                plan.ConfigureOptimizer(
                    OptimizerType.Adam,
                    learningRate: 1e-4f,
                    beta1: 0.9f,
                    beta2: 0.999f,
                    eps: 1e-8f,
                    weightDecay: 0f);

                float lastLoss = float.NaN;
                for (int step = 0; step < steps; step++)
                {
                    var loss = plan.Step();
                    lastLoss = loss.AsSpan()[0];
                    _output.WriteLine($"step {step + 1}/{steps}: loss={lastLoss:G9}");
                    _output.WriteLine(GpuMemoryTracker.Report(topN: 12));
                    AssertMemoryEnvelope(maxLiveMb, maxPeakMb, memoryBaselineBytes);
                }

                double pL1After = ParamL1(parameters);
                double delta = Math.Abs(pL1After - pL1Before);
                double maxElementDelta = MaxParameterDelta(parameters, parameterSnapshot);
                _output.WriteLine(
                    $"pL1 before={pL1Before:F6} after={pL1After:F6} delta={delta:G9}; maxElementDelta={maxElementDelta:G9}");

                Assert.True(!float.IsNaN(lastLoss) && !float.IsInfinity(lastLoss),
                    $"compiled step returned non-finite loss: {lastLoss}");
                Assert.True(maxElementDelta > 1e-7,
                    $"parameters did not move; max element delta={maxElementDelta:G9}, pL1 delta={delta:G9}");
            }
        }
        finally
        {
            AiDotNetEngine.Current = previousEngine;
        }
    }

    private static int ReadInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int parsed) && parsed > 0
            ? parsed
            : fallback;

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var tensor = new Tensor<float>(shape);
        var span = tensor.AsWritableSpan();
        var rng = new Random(seed);
        for (int i = 0; i < span.Length; i++)
            span[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        return tensor;
    }

    private static Tensor<float> Ones(int[] shape)
    {
        var tensor = new Tensor<float>(shape);
        tensor.AsWritableSpan().Fill(1f);
        return tensor;
    }

    private static Tensor<float> Zeros(int[] shape) => new(shape);

    private static void AssertMemoryEnvelope(int maxLiveMb, int maxPeakMb, long baselineLiveBytes)
    {
        const long bytesPerMiB = 1024L * 1024L;
        if (maxLiveMb > 0)
        {
            double liveMb = Math.Max(0L, GpuMemoryTracker.LiveBytes - baselineLiveBytes) / (double)bytesPerMiB;
            Assert.True(liveMb <= maxLiveMb, $"GPU live memory {liveMb:F1} MiB exceeded {maxLiveMb} MiB.");
        }

        if (maxPeakMb > 0)
        {
            double peakMb = Math.Max(0L, GpuMemoryTracker.PeakBytes - baselineLiveBytes) / (double)bytesPerMiB;
            Assert.True(peakMb <= maxPeakMb, $"GPU peak memory {peakMb:F1} MiB exceeded {maxPeakMb} MiB.");
        }
    }

    private static float[][] SnapshotParameters(Tensor<float>[] parameters)
    {
        var snapshot = new float[parameters.Length][];
        for (int i = 0; i < parameters.Length; i++)
            snapshot[i] = parameters[i].AsSpan().ToArray();
        return snapshot;
    }

    private static double MaxParameterDelta(Tensor<float>[] parameters, float[][] before)
    {
        double maxDelta = 0;
        for (int p = 0; p < parameters.Length; p++)
        {
            var after = parameters[p].AsSpan();
            Assert.Equal(before[p].Length, after.Length);
            for (int i = 0; i < after.Length; i++)
                maxDelta = Math.Max(maxDelta, Math.Abs(after[i] - before[p][i]));
        }

        return maxDelta;
    }

    private static double ParamL1(Tensor<float>[] parameters)
    {
        double sum = 0;
        foreach (var parameter in parameters)
        {
            var span = parameter.AsSpan();
            for (int i = 0; i < span.Length; i++)
                sum += Math.Abs(span[i]);
        }
        return sum;
    }
}
