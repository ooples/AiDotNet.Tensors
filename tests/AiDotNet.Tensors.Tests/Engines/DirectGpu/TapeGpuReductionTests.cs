// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// ReduceSum and TensorLogSoftmax used to fall back to the CPU engine whenever a GradientTape was active, so the
/// cross-entropy of an LM training step downloaded its [tokens, vocab] logits every step. The GPU paths now record the
/// same tape nodes as the CPU engine. These tests pin forward, gradients and device residency to the CPU engine, and
/// pin the new numerically stable GPU log-softmax on rows whose spread overflows log(softmax(x)) in float32.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class TapeGpuReductionTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public TapeGpuReductionTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static void AssertClose(Tensor<float> expected, Tensor<float> actual, string name, double rel = 1e-4)
    {
        Assert.Equal(expected.Shape.ToArray(), actual.Shape.ToArray());
        var e = expected.ToArray();
        var a = actual.ToArray();
        double scale = Math.Max(e.Max(v => Math.Abs(v)), 1e-6);
        for (int i = 0; i < e.Length; i++)
        {
            Assert.False(float.IsNaN(a[i]) || float.IsInfinity(a[i]), $"{name}[{i}] is {a[i]}");
            Assert.True(Math.Abs(e[i] - a[i]) <= rel * scale, $"{name}[{i}]: cpu {e[i]}, gpu {a[i]}");
        }
    }

    private static bool OnDevice(Tensor<float> t) => t.IsGpuResident || t.HasPendingGpuData;

    [SkippableTheory]
    [InlineData(new[] { 64, 300 }, new[] { 1 }, true)]
    [InlineData(new[] { 4, 5, 6 }, new[] { 0, 2 }, false)]
    [InlineData(new[] { 32, 16 }, null, false)]
    public void ReduceSum_UnderTape_StaysOnDeviceAndMatchesCpu(int[] shape, int[]? axes, bool keepDims)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var x = Rand(shape, 1);
        (Tensor<float> Out, Tensor<float> Grad) Run(IEngine e)
        {
            using var tape = new GradientTape<float>();
            var reduced = e.ReduceSum(x, axes, keepDims);
            var weights = Rand(reduced.Shape.ToArray(), 2);
            var loss = e.ReduceSum(e.TensorMultiply(reduced, weights), null, false);
            return (reduced, tape.ComputeGradients(loss, [x])[x]);
        }

        var cpu = Run(new CpuEngine());
        var gpu = Run(_fixture.Engine!);
        Assert.True(OnDevice(gpu.Out), "ReduceSum under the tape ran on the host");
        AssertClose(cpu.Out, gpu.Out, "out");
        AssertClose(cpu.Grad, gpu.Grad, "dX");
    }

    [SkippableFact]
    public void LogSoftmax_UnderTape_StaysOnDeviceAndMatchesCpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var x = Rand([32, 1000], 3, scale: 8f);
        var r = Rand([32, 1000], 4);
        (Tensor<float> Out, Tensor<float> Grad) Run(IEngine e)
        {
            using var tape = new GradientTape<float>();
            var ls = e.TensorLogSoftmax(x, 1);
            var loss = e.ReduceSum(e.TensorMultiply(ls, r), null, false);
            return (ls, tape.ComputeGradients(loss, [x])[x]);
        }

        var cpu = Run(new CpuEngine());
        var gpu = Run(_fixture.Engine!);
        Assert.True(OnDevice(gpu.Out), "LogSoftmax under the tape ran on the host");
        AssertClose(cpu.Out, gpu.Out, "logsoftmax");
        AssertClose(cpu.Grad, gpu.Grad, "dX");
    }

    /// <summary>A row spread of 200 makes softmax underflow to 0 in float32, so log(softmax) was -inf.</summary>
    [SkippableFact]
    public void LogSoftmax_WideRows_IsFiniteAndMatchesCpu()
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var x = new Tensor<float>([2, 4], new Vector<float>([0f, -200f, 3f, -150f, 100f, -100f, 0f, 50f]));
        AssertClose(new CpuEngine().TensorLogSoftmax(x, 1), _fixture.Engine!.TensorLogSoftmax(x, 1), "wide", 1e-5);
    }
}
