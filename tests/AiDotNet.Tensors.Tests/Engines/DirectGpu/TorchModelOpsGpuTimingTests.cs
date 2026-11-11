using System;
using System.Diagnostics;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Wall-clock measurements of the model-op kernels at sizes that stress them (a vocabulary-sized top-k row, a long CRF
/// sequence), printed to the test output. Opt-in (AIDOTNET_RUN_GPU_TIMINGS=1): timings are for comparing builds on one
/// machine, not assertions. Correctness at these sizes is asserted by TorchModelOpsGpuTests and the op-parity registry.
/// </summary>
[Collection("DirectGpuSerial")]
public class TorchModelOpsGpuTimingTests : IClassFixture<DirectGpuTensorEngineTestFixture>, IClassFixture<CudaTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;
    private readonly CudaTensorEngineTestFixture _cuda;
    private readonly ITestOutputHelper _out;

    public TorchModelOpsGpuTimingTests(DirectGpuTensorEngineTestFixture fixture, CudaTensorEngineTestFixture cuda, ITestOutputHelper output)
    {
        _fixture = fixture;
        _cuda = cuda;
        _out = output;
    }

    private static Tensor<float> Rand(int seed, params int[] shape)
    {
        var rng = new Random(seed);
        return new Tensor<float>(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
            .Select(_ => (float)(rng.NextDouble() * 6 - 3)).ToArray(), shape);
    }

    private double Time(string what, Func<Tensor<float>> run, int reps)
    {
        _ = run().ToArray();   // warm-up: kernel compilation, buffer pools
        // The median of single runs: the machine is shared, and a mean takes in every stall.
        var samples = new double[Math.Max(reps, 15)];
        for (int i = 0; i < samples.Length; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = run().ToArray();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        double ms = samples[samples.Length / 2];
        _out.WriteLine($"{what}: {ms:F3} ms (median of {samples.Length})");
        return ms;
    }

    [SkippableFact]
    public void Measure()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("AIDOTNET_RUN_GPU_TIMINGS") == "1", "Set AIDOTNET_RUN_GPU_TIMINGS=1 to measure.");
        Skip.IfNot(_fixture.IsAvailable || _cuda.IsAvailable, "No GPU device.");
        if (_fixture.IsAvailable) MeasureOn(_fixture.Engine!);
        if (_cuda.IsAvailable) MeasureOn(_cuda.Engine!);
    }

    private void MeasureOn(DirectGpuTensorEngine gpu)
    {
        _out.WriteLine($"backend: {gpu.GetBackend()?.GetType().Name}");

        foreach (var (rows, width, k) in new[] { (1, 50000, 8), (8, 50000, 8), (64, 1024, 4) })
        {
            var x = gpu.TensorMultiplyScalar(Rand(1, rows, width), 1f);
            Time($"TopKSoftmax rows={rows} width={width} k={k}", () => gpu.TensorTopKSoftmax(x, k), 5);
        }

        foreach (var (b, s, tags) in new[] { (32, 128, 16), (8, 512, 32), (1, 1024, 8) })
        {
            var e = gpu.TensorMultiplyScalar(Rand(2, b, s, tags), 1f);
            var t = gpu.TensorMultiplyScalar(Rand(3, tags, tags), 1f);
            Time($"CrfForward b={b} s={s} k={tags}", () => gpu.TensorCrfForward(e, t), 5);
            Time($"CrfForward+backward b={b} s={s} k={tags}", () =>
            {
                using var tape = new GradientTape<float>();
                var z = gpu.TensorCrfForward(e, t);
                var loss = gpu.ReduceSum(z, new[] { 0 }, keepDims: false);
                var g = tape.ComputeGradients(loss, new[] { e, t });
                _ = g[t].ToArray();
                return g[e];
            }, 5);
        }
    }
}
