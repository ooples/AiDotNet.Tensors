using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A compiled GPU plan leases each forward output no backward reads from the action that writes it to its last forward
/// reader, instead of keeping a buffer per output for the plan's life (PyTorch frees such an intermediate once it is
/// consumed). The graph has every case the lease follows: a convolution only added to a residual, a matrix product read
/// through a reshape view (the view shares its device buffer) by a permute, and a matrix product whose reshape view is
/// read by a multiply, whose backward needs it, so that product is kept. Trained for several SGD steps it must reach the
/// CPU plan's parameters with leases on and off, and the leased plan must hold fewer scratch bytes, with and without a
/// captured step graph.
/// </summary>
[Collection("DirectGpuSerial")]
public class CompiledGpuForwardLeaseTests
{
    private static Tensor<float> Filled(int[] shape, int seed, float scale)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)(rng.NextDouble() - 0.5) * scale;
        return t;
    }

    private static (float[] Parameters, long ScratchBytes, int Candidates) Train(IEngine engine, int steps, bool captured)
    {
        var x = Filled(new[] { 8, 8, 16, 16 }, 1, 2f);
        var flat = Filled(new[] { 64, 128 }, 2, 1f);
        var parameters = new System.Collections.Generic.List<Tensor<float>>();
        ICompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            // Residual convolutions: each convolution's output is read only by the add.
            var h = x;
            for (int b = 0; b < 3; b++)
            {
                var k = Filled(new[] { 8, 8, 3, 3 }, 10 + b, 0.1f);
                parameters.Add(k);
                var conv = engine.Conv2D(h, k, new[] { 1, 1 }, new[] { 1, 1 }, new[] { 1, 1 });
                h = engine.TensorAdd(h, conv);
            }
            // A product read through a reshape view by a permute, then added: the product and the permute are leased.
            var w = Filled(new[] { 128, 256 }, 20, 0.1f);
            var bias = Filled(new[] { 64, 16, 16 }, 21, 1f);
            parameters.Add(w); parameters.Add(bias);
            var product = engine.TensorMatMul(flat, w);
            var permuted = engine.TensorPermute(engine.Reshape(product, new[] { 64, 16, 16 }), new[] { 0, 2, 1 });
            var shifted = engine.TensorAdd(permuted, bias);
            // A product whose reshape view a multiply reads: its backward needs the view, so the product is kept.
            var w2 = Filled(new[] { 128, 256 }, 22, 0.1f);
            parameters.Add(w2);
            var kept = engine.Reshape(engine.TensorMatMul(flat, w2), new[] { 64, 16, 16 });
            var gated = engine.TensorMultiply(kept, kept);
            var loss = engine.TensorAdd(
                engine.ReduceSum(engine.TensorMultiply(h, h), null),
                engine.ReduceSum(engine.TensorMultiply(shifted, shifted), null));
            engine.TensorAdd(loss, engine.ReduceSum(gated, null));
            plan = scope.CompileTraining(parameters.ToArray());
        }
        using (plan)
        {
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-4f);
            if (!captured && plan is CompiledTrainingPlan<float> uncaptured) uncaptured.DisableGraphStep();
            for (int s = 0; s < steps; s++) plan.Step();
            var compiled = plan as CompiledTrainingPlan<float>;
            var all = new System.Collections.Generic.List<float>();
            foreach (var p in parameters) all.AddRange(p.ToArray());
            // An explicitly uncaptured plan hands its transient backing to the next one (CompiledTransientWorkspaceBroker),
            // so its pool reports what it inherited: measure the storage its own step used, before that transfer.
            long bytes = compiled is null ? -1
                : captured ? compiled.ScratchBufferBytes : compiled.LastStepScratchBytesBeforeTransfer;
            return (all.ToArray(), bytes, compiled?.ForwardSpanLeaseCandidates ?? -1);
        }
    }

    private static void AssertClose(float[] expected, float[] actual, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f * (1f + Math.Abs(expected[i])),
                $"{name} [{i}]: cpu {expected[i]}, gpu {actual[i]}");
    }

    private static void Run(bool captured)
    {
        DirectGpuTensorEngine gpu;
        try { gpu = new DirectGpuTensorEngine(); }
        catch (Exception ex) when (ex is InvalidOperationException or NotSupportedException or DllNotFoundException)
        {
            Skip.If(true, "No GPU engine: " + ex.GetType().Name);
            return;
        }
        Skip.IfNot(gpu.IsGpuAvailable, "No GPU device.");
        if (captured)
            Skip.IfNot(gpu.GetBackend() is AiDotNet.Tensors.Engines.DirectGpu.IGpuGraphCapture { SupportsGraphCapture: true },
                "The backend does not capture step graphs.");
        bool previous = CompiledTrainingPlan<float>.LeaseForwardOutputs;
        bool previousObservation = CompiledTrainingPlan<float>.ObserveScratchStorageForTests;
        using (gpu)
        {
            try
            {
                // Enough steps that a captured graph replays several times after the warm-up and capture steps.
                int steps = captured ? 8 : 4;
                CompiledTrainingPlan<float>.ObserveScratchStorageForTests = true;
                var expected = Train(new CpuEngine(), steps, captured: false).Parameters;
                CompiledTrainingPlan<float>.LeaseForwardOutputs = false;
                var keptAll = Train(gpu, steps, captured);
                CompiledTrainingPlan<float>.LeaseForwardOutputs = true;
                long replaysBefore = CompiledTrainingPlan<float>.GraphReplayCount;
                var leased = Train(gpu, steps, captured);
                if (captured)
                    Assert.True(CompiledTrainingPlan<float>.GraphReplayCount > replaysBefore,
                        "the leased plan never replayed a captured step graph, so this test did not exercise capture");

                AssertClose(expected, keptAll.Parameters, "kept");
                AssertClose(expected, leased.Parameters, "leased");
                // The three convolutions, the first product (through its view) and the permute.
                Assert.True(leased.Candidates >= 5, $"the plan leases {leased.Candidates} forward outputs");
                Assert.True(leased.ScratchBytes > 0 && leased.ScratchBytes < keptAll.ScratchBytes,
                    $"leased plan holds {leased.ScratchBytes} scratch bytes, kept plan {keptAll.ScratchBytes}");
            }
            finally
            {
                CompiledTrainingPlan<float>.LeaseForwardOutputs = previous;
                CompiledTrainingPlan<float>.ObserveScratchStorageForTests = previousObservation;
            }
        }
    }

    [SkippableFact]
    public void Leased_forward_outputs_train_like_the_cpu_and_hold_fewer_bytes() => Run(captured: false);

    [SkippableFact]
    public void Leased_forward_outputs_in_a_captured_step_graph_train_like_the_cpu_and_hold_fewer_bytes() => Run(captured: true);
}
