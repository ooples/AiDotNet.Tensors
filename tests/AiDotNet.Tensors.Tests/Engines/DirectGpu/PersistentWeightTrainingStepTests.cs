// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A GPU inference pass registers a layer's weights in the persistent weight-buffer cache. An eager training step
/// taken afterwards (tape gradient, then an in-place <c>W -= update</c>) must see and change the real weights: measured
/// in AiDotNet, a float FeedForward net never moved after one GPU Predict (loss bit-identical every step) until the
/// weight cache was cleared before each step.
/// </summary>
[Collection("VulkanGlobalState")]
public sealed class PersistentWeightTrainingStepTests : IClassFixture<DirectGpuTensorEngineTestFixture>
{
    private readonly DirectGpuTensorEngineTestFixture _fixture;

    public PersistentWeightTrainingStepTests(DirectGpuTensorEngineTestFixture fixture) => _fixture = fixture;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5);
        return t;
    }

    [SkippableTheory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void EagerStep_AfterInferenceCachedTheWeight_UpdatesTheWeight(bool registerFirst, bool invalidateBeforeRead)
    {
        Skip.IfNot(_fixture.IsAvailable, "No GPU device.");
        var gpu = _fixture.Engine!;
        IEngine e = gpu;
        var x = Rand([16, 8], 1);
        var r = Rand([16, 4], 2);
        var w = Rand([8, 4], 3);
        var w0 = w.ToArray();

        // "Predict": the weight goes into the persistent cache.
        if (registerFirst) gpu.RegisterPersistentTensor(w, PersistentTensorRole.Weights);
        _ = e.TensorMatMul(x, w).ToArray();
        // A layer's GPU inference forward (DenseLayer -> FusedLinear) uploads the weight into the persistent cache.
        _ = e.FusedLinear(x, w, new Tensor<float>([4]), FusedActivationType.Tanh).ToArray();

        // Expected gradient dL/dW = x^T r for L = sum((x W) * r), computed on the CPU.
        var cpu = new CpuEngine();
        var gExpected = cpu.TensorMatMul(cpu.TensorTranspose(x), r).ToArray();

        Tensor<float> g;
        using (var tape = new GradientTape<float>())
        {
            var y = e.TensorMatMul(x, w);
            var loss = e.ReduceSum(e.TensorMultiply(y, r), null, false);
            g = tape.ComputeGradients(loss, [w])[w];
        }
        var gArr = g.ToArray();
        for (int i = 0; i < gArr.Length; i++)
            Assert.True(Math.Abs(gExpected[i] - gArr[i]) <= 1e-4, $"dW[{i}]: expected {gExpected[i]}, got {gArr[i]}");

        // A resident update operand (the optimizer's update is the output of GPU ops).
        var update = e.TensorAdd(g, new Tensor<float>(g.Shape.ToArray()));
        Assert.True(update.IsGpuResident || update.HasPendingGpuData, "precondition: the update must be resident");
        e.TensorSubtractInPlace(w, update);
        // AiDotNet's post-step contract: after an optimizer update the HOST weight is authoritative, so it drops the
        // weight's device state (GradientBasedOptimizerBase / NeuralNetworkBase) before anything reads it.
        if (invalidateBeforeRead) gpu.InvalidateResidentWeightBuffer(w);
        var host = w.ToArray();
        var viaDevice = e.TensorMatMul(new Tensor<float>([1, 8], new Vector<float>(Enumerable.Range(0, 8).Select(i => i == 0 ? 1f : 0f).ToArray())), w).ToArray();
        for (int i = 0; i < host.Length; i++)
            Assert.True(Math.Abs((w0[i] - gExpected[i]) - host[i]) <= 1e-4, $"W[{i}] after update: expected {w0[i] - gExpected[i]}, got {host[i]} (before {w0[i]})");
        for (int j = 0; j < 4; j++)
            Assert.True(Math.Abs(host[j] - viaDevice[j]) <= 1e-4, $"device read of W row 0 [{j}]: host {host[j]}, device {viaDevice[j]}");
    }
}
