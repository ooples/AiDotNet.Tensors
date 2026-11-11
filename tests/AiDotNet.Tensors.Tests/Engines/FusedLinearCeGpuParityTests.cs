using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.Cpu;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.Tests.TestHelpers;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// Parity tests for the GPU fused linear + cross-entropy path (#1464): the shared
/// <see cref="RecurrenceCpuKernels"/> host references (used by the Metal/Vulkan/WebGpu fallback)
/// must match <see cref="CpuEngine.FusedLinearCrossEntropyWithLogits{T}(Tensor{T},Tensor{T},Tensor{T},Tensor{int})"/>
/// and the dense overload.
/// </summary>
// Runs real GPU work on the shared CUDA context: serialized with the other GPU tests so it
// cannot launch during another test's stream capture (CUDA error 901, capture invalidated).
[Collection("DirectGpuSerial")]
public class FusedLinearCeGpuParityTests
{
    private static float[] Gen(int n, int s, float scale = 0.5f)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(Math.Sin(0.7 * (i + 1) + 1.3 * s) * scale);
        return a;
    }

    [Fact]
    public void HostReferenceIndex_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int n = 4, d = 5, vocab = 6;
        var hidden = new Tensor<float>(Gen(n * d, 1), new[] { n, d });
        var weight = new Tensor<float>(Gen(d * vocab, 2), new[] { d, vocab });
        var bias = new Tensor<float>(Gen(vocab, 3, 0.3f), new[] { vocab });
        var ids = new int[n];
        for (int r = 0; r < n; r++) ids[r] = (r * 7 + 2) % vocab;

        float reference = ((float[])(object)engine.FusedLinearCrossEntropyWithLogits(
            hidden, weight, bias, new Tensor<int>(ids, new[] { n })).GetDataArray()!)[0];
        float host = RecurrenceCpuKernels.FusedLinearCeIndex(F(hidden), F(weight), F(bias), ids, n, d, vocab);

        Assert.True(Math.Abs(reference - host) < 1e-4f, $"index loss cpu={reference} host={host}");
    }

    [Fact]
    public void HostReferenceDense_MatchesCpuEngine()
    {
        var engine = new CpuEngine();
        int n = 3, d = 4, vocab = 5;
        var hidden = new Tensor<float>(Gen(n * d, 1), new[] { n, d });
        var weight = new Tensor<float>(Gen(d * vocab, 2), new[] { d, vocab });
        var bias = new Tensor<float>(Gen(vocab, 3, 0.3f), new[] { vocab });
        var target = new float[n * vocab];
        for (int r = 0; r < n; r++) target[r * vocab + ((r * 3 + 1) % vocab)] = 1.0f; // one-hot

        float reference = ((float[])(object)engine.FusedLinearCrossEntropyWithLogits(
            hidden, weight, bias, new Tensor<float>(target, new[] { n, vocab })).GetDataArray()!)[0];
        float host = RecurrenceCpuKernels.FusedLinearCeDense(F(hidden), F(weight), F(bias), target, n, d, vocab);

        Assert.True(Math.Abs(reference - host) < 1e-4f, $"dense loss cpu={reference} host={host}");
    }

    [Fact]
    public void TapeActiveDirectGpuOverloads_ProduceAnalyticGradients()
    {
        var cpu = new CpuEngine();
        using var gpu = new DirectGpuTensorEngine();
        var ids = new Tensor<int>(new[] { 0, 2 }, new[] { 2 });
        var target = new Tensor<float>(new[] { 1f, 0f, 0f, 0f, 0f, 1f }, new[] { 2, 3 });

        AssertTapeParity(
            cpu,
            gpu,
            (engine, hidden, weight, bias) =>
                engine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, ids),
            "index");
        AssertTapeParity(
            cpu,
            gpu,
            (engine, hidden, weight, bias) =>
                engine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, ids, computeReadoutGrads: true),
            "index_5arg_true");
        AssertTapeParity(
            cpu,
            gpu,
            (engine, hidden, weight, bias) =>
                engine.FusedLinearCrossEntropyWithLogits(hidden, weight, bias, target),
            "dense");
        AssertDetachedTapeParity(cpu, gpu, ids);
    }

    private static void AssertDetachedTapeParity(CpuEngine cpu, DirectGpuTensorEngine gpu, Tensor<int> ids)
    {
        var cpuH = new Tensor<float>(Gen(8, 1), new[] { 2, 4 });
        var cpuW = new Tensor<float>(Gen(12, 2), new[] { 4, 3 });
        var cpuB = new Tensor<float>(Gen(3, 3), new[] { 3 });

        var gpuH = new Tensor<float>(Gen(8, 1), new[] { 2, 4 });
        var gpuW = new Tensor<float>(Gen(12, 2), new[] { 4, 3 });
        var gpuB = new Tensor<float>(Gen(3, 3), new[] { 3 });

        Tensor<float> cpuLoss;
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> cpuGrads;
        using (var tape = new GradientTape<float>())
        {
            cpuLoss = cpu.FusedLinearCrossEntropyWithLogits(cpuH, cpuW, cpuB, ids, computeReadoutGrads: false);
            cpuGrads = tape.ComputeGradients(cpuLoss, new[] { cpuH, cpuW, cpuB });
        }

        Tensor<float> gpuLoss;
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gpuGrads;
        using (var tape = new GradientTape<float>())
        {
            gpuLoss = gpu.FusedLinearCrossEntropyWithLogits(gpuH, gpuW, gpuB, ids, computeReadoutGrads: false);
            gpuGrads = tape.ComputeGradients(gpuLoss, new[] { gpuH, gpuW, gpuB });
        }

        Assert.InRange(Math.Abs(gpuLoss[0] - cpuLoss[0]), 0f, 1e-4f);
        Assert.True(cpuGrads.ContainsKey(cpuH) && gpuGrads.ContainsKey(gpuH));
        Assert.False(cpuGrads.ContainsKey(cpuW));
        Assert.False(cpuGrads.ContainsKey(cpuB));
        Assert.False(gpuGrads.ContainsKey(gpuW));
        Assert.False(gpuGrads.ContainsKey(gpuB));

        var expected = cpuGrads[cpuH].ToArray();
        var actual = gpuGrads[gpuH].ToArray();
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0f, 2e-4f);
    }

    [Fact]
    public void DetachedReadout_InPlaceMutationBeforeBackward_ThrowsInvalidOperationException()
    {
        var ids = new Tensor<int>(new[] { 1, 0 }, new[] { 2 });

        // 1. CPU double path (recomputes in FusedLinearCrossEntropyIndexBackward).
        var cpu = new CpuEngine();
        using (var tape = new GradientTape<double>())
        {
            var h = new Tensor<double>(new double[] { 0.2, -0.3, 0.5, 0.1, -0.4, 0.6 }, new[] { 2, 3 });
            var w = new Tensor<double>(new double[] { 0.1, -0.2, 0.3, 0.4, -0.1, 0.2 }, new[] { 3, 2 });
            var b = new Tensor<double>(new double[] { 0.05, -0.05 }, new[] { 2 });
            var loss = cpu.FusedLinearCrossEntropyWithLogits(h, w, b, ids, computeReadoutGrads: false);
            w[0] = 9.0;
            w.MarkModified();
            Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(loss, new[] { h }));
        }

        // 2. CPU float path: both before first backward AND before a repeated backward (after precomputed gradHidden is consumed).
        using (var tape = new GradientTape<float>(new GradientTapeOptions { Persistent = true }))
        {
            var h = new Tensor<float>(Gen(6, 11), new[] { 2, 3 });
            var w = new Tensor<float>(Gen(6, 12), new[] { 3, 2 });
            var b = new Tensor<float>(Gen(2, 13), new[] { 2 });
            var loss = cpu.FusedLinearCrossEntropyWithLogits(h, w, b, ids, computeReadoutGrads: false);
            var g1 = tape.ComputeGradients(loss, new[] { h });
            Assert.True(g1.ContainsKey(h));

            b[0] += 1.0f;
            b.MarkModified();
            Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(loss, new[] { h }));
        }

        // 3. DirectGpuTensorEngine path.
        using var gpu = new DirectGpuTensorEngine();
        using (var tape = new GradientTape<float>())
        {
            var h = new Tensor<float>(Gen(6, 21), new[] { 2, 3 });
            var w = new Tensor<float>(Gen(6, 22), new[] { 3, 2 });
            var b = new Tensor<float>(Gen(2, 23), new[] { 2 });
            var loss = gpu.FusedLinearCrossEntropyWithLogits(h, w, b, ids, computeReadoutGrads: false);
            w[0] += 2.0f;
            w.MarkModified();
            Assert.Throws<InvalidOperationException>(() => tape.ComputeGradients(loss, new[] { h }));
        }
    }

    [Fact]
    public void GraphMode_DetachedReadoutFromCapturedProducer_OrdersProducerBeforeLossAndSuppressesReadoutGrads()
    {
        var engine = new CpuEngine();
        var prior = AiDotNetEngine.Current;
        AiDotNetEngine.Current = engine;
        try
        {
            var hidden = new Tensor<double>(new double[] { 0.3, -0.2, 0.5, -0.4, 0.1, 0.6 }, new[] { 2, 3 });
            var rawWeight = new Tensor<double>(new double[] { 0.2, -0.1, 0.4, -0.3, 0.1, 0.5 }, new[] { 3, 2 });
            var rawBias = new Tensor<double>(new double[] { 0.05, -0.05 }, new[] { 2 });
            var ids = new Tensor<int>(new[] { 1, 0 }, new[] { 2 });

            Tensor<double> Forward(IEngine e)
            {
                var w = e.TensorMultiplyScalar(rawWeight, 1.5);
                var b = e.TensorAddScalar(rawBias, 0.2);
                return e.FusedLinearCrossEntropyWithLogits(hidden, w, b, ids, computeReadoutGrads: false);
            }

            ICompiledTrainingPlan<double> plan;
            using (var scope = GraphMode.EnableTraining(new[] { hidden }))
            {
                Forward(engine);
                plan = scope.CompileTraining(new[] { hidden });
            }

            using (plan)
            {
                for (int step = 0; step < 3; step++)
                {
                    for (int i = 0; i < rawWeight.Length; i++)
                        rawWeight.SetFlat(i, 0.15 * (step + 1) * ((i % 2 == 0) ? 1.0 : -0.7));
                    rawWeight.MarkModified();
                    rawBias.SetFlat(0, 0.1 * step);
                    rawBias.SetFlat(1, -0.2 * step);
                    rawBias.MarkModified();

                    double expectedLoss;
                    double[] expectedGrad;
                    using (var tape = new GradientTape<double>())
                    {
                        var l = Forward(engine);
                        expectedLoss = l.GetFlat(0);
                        expectedGrad = tape.ComputeGradients(l, new[] { hidden })[hidden].ToArray();
                    }

                    double actualLoss = plan.Step().GetFlat(0);
                    Assert.True(Math.Abs(expectedLoss - actualLoss) <= 1e-10,
                        $"step {step}: expected loss {expectedLoss:R}, got {actualLoss:R}");
                    var actualGrad = plan.Gradients[0].ToArray();
                    for (int i = 0; i < expectedGrad.Length; i++)
                        Assert.True(Math.Abs(expectedGrad[i] - actualGrad[i]) <= 1e-10,
                            $"step {step} grad[{i}]: expected {expectedGrad[i]:R}, got {actualGrad[i]:R}");
                }
            }
        }
        finally
        {
            AiDotNetEngine.Current = prior;
        }
    }

    private static void AssertTapeParity(
        CpuEngine cpu,
        DirectGpuTensorEngine gpu,
        Func<CpuEngine, Tensor<float>, Tensor<float>, Tensor<float>, Tensor<float>> forward,
        string label)
    {
        var cpuParameters = new[]
        {
            new Tensor<float>(Gen(8, 1), new[] { 2, 4 }),
            new Tensor<float>(Gen(12, 2), new[] { 4, 3 }),
            new Tensor<float>(Gen(3, 3), new[] { 3 })
        };
        var gpuParameters = new[]
        {
            new Tensor<float>(Gen(8, 1), new[] { 2, 4 }),
            new Tensor<float>(Gen(12, 2), new[] { 4, 3 }),
            new Tensor<float>(Gen(3, 3), new[] { 3 })
        };

        Tensor<float> cpuLoss;
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> cpuGradients;
        using (var tape = new GradientTape<float>())
        {
            cpuLoss = forward(cpu, cpuParameters[0], cpuParameters[1], cpuParameters[2]);
            cpuGradients = tape.ComputeGradients(cpuLoss, cpuParameters);
        }

        Tensor<float> gpuLoss;
        System.Collections.Generic.Dictionary<Tensor<float>, Tensor<float>> gpuGradients;
        using (var tape = new GradientTape<float>())
        {
            gpuLoss = forward(gpu, gpuParameters[0], gpuParameters[1], gpuParameters[2]);
            gpuGradients = tape.ComputeGradients(gpuLoss, gpuParameters);
        }

        Assert.True(MathCompat.IsFinite(gpuLoss[0]), $"{label} DirectGpu loss is not finite.");
        Assert.InRange(Math.Abs(gpuLoss[0] - cpuLoss[0]), 0f, 1e-4f);
        for (int parameterIndex = 0; parameterIndex < cpuParameters.Length; parameterIndex++)
        {
            var expected = cpuGradients[cpuParameters[parameterIndex]].ToArray();
            var actual = gpuGradients[gpuParameters[parameterIndex]].ToArray();
            Assert.Equal(expected.Length, actual.Length);
            for (int element = 0; element < expected.Length; element++)
            {
                Assert.True(MathCompat.IsFinite(actual[element]),
                    $"{label} DirectGpu gradient {parameterIndex}[{element}] is not finite.");
                Assert.InRange(Math.Abs(actual[element] - expected[element]), 0f, 2e-4f);
            }
        }

    }

    private static float[] F(Tensor<float> t) => (float[])(object)t.GetDataArray()!;
}
