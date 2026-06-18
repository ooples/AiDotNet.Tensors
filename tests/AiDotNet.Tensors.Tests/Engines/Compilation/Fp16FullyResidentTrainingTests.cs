// End-to-end proof (#633): a small model trains on the GPU with FP16 activations while the AUTODIFF working set
// stays GPU-resident — the forward (Half-resident store), the fused Half backward, the on-device float↔Half
// gradient cast bridges, and the per-node scratch release keep activations / op results / cast-boundary
// gradients / scratch on the device across a full training step. Measured via
// DeferredArrayMaterializer.MaterializeCount (each fired callback = one real GPU→CPU download of a resident
// tensor): a full Step with no loss read pulls only a small bounded number of tensors to host — the per-PARAMETER
// gradient reads handed to the optimizer's master-weight update (count = trainable-parameter count, independent
// of activation size / model depth), NOT the activations/op-results/scratch (which never round-trip). Driving
// those last per-parameter grad reads to zero needs the optimizer master-update + forward param-cast on-device
// too (a coupled param-residency change, tracked separately). The loss also decreases over steps, proving the
// resident gradients are correct. Runs on any DirectGpuTensorEngine (CUDA/OpenCL/…); skips on a CPU-only host.

#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection(MixedPrecisionTestCollection.Name)]
public class Fp16FullyResidentTrainingTests
{
    private readonly ITestOutputHelper _out;
    public Fp16FullyResidentTrainingTests(ITestOutputHelper o) { _out = o; }

    private static float[] RandData(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return data;
    }

    [SkippableFact]
    public void Fp16Model_TrainsOnGpu_HeavyTensorsStayResident_AndLossDecreases()
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/…).");

        // A small trainable model: pred = (input·W1)·W2, loss = sum(pred²) — a smooth least-squares objective
        // minimized by shrinking pred, so plain SGD decreases it monotonically (stable, no target tensor needed).
        // The output-level square (TensorMultiply) gives the backward real per-op scratch; the two matmuls give
        // Half activations. Scaled-down init keeps the FP16 forward well inside range. Trains W1, W2 in place.
        var input = new Tensor<float>(RandData(new[] { 32, 64 }, 1, scale: 0.5f), new[] { 32, 64 });
        var w1 = new Tensor<float>(RandData(new[] { 64, 64 }, 2, scale: 0.1f), new[] { 64, 64 });
        var w2 = new Tensor<float>(RandData(new[] { 64, 64 }, 3, scale: 0.1f), new[] { 64, 64 });

        var prev = MixedPrecisionEmit.TestOverrideEnabled;
        MixedPrecisionEmit.TestOverrideEnabled = true; // force FP16 activation emission
        CompiledTrainingPlan<float> plan;
        try
        {
            using (var scope = GraphMode.Enable())
            using (new AutocastScope(PrecisionMode.Float16))
            {
                var h = gpu.TensorMatMul(input, w1);     // [32,64] Half activation
                var pred = gpu.TensorMatMul(h, w2);      // [32,64] Half activation
                var sqErr = gpu.TensorMultiply(pred, pred); // output-level square → backward scratch
                gpu.ReduceSum(sqErr, null);              // loss = sum(pred²) ≥ 0, minimized at pred→0
                plan = scope.CompileTraining(new[] { w1, w2 });
            }
        }
        finally { MixedPrecisionEmit.TestOverrideEnabled = prev; GraphMode.SetCurrent(null); }

        try
        {
            // FP16 activation storage is engaged (the captured graph holds Half nodes).
            Assert.True(plan.HasFp16HeteroForward, "FP16 hetero activation path must be engaged.");

            // Warm up once so the parameters acquire GPU buffers, then configure the FUSED ON-DEVICE optimizer
            // (the weight/grad/moment update runs on the GPU — no host round-trip of grads or weights).
            float firstLoss = plan.Step().GetFlat(0);
            Assert.True(!float.IsNaN(firstLoss) && !float.IsInfinity(firstLoss), $"first loss must be finite, got {firstLoss}.");
            plan.ConfigureOptimizer(OptimizerType.SGD, learningRate: 1e-3f);

            // TRAIN: the fully-resident step must actually learn — loss decreases. Reading the loss each step is
            // the one deliberate scalar host read; the heavy tensors stay on the GPU.
            float last = firstLoss;
            for (int i = 0; i < 15; i++) last = plan.Step().GetFlat(0);
            _out.WriteLine($"loss: first={firstLoss:G6} last={last:G6}");
            Assert.True(last < firstLoss * 0.99f, $"FP16 training should reduce the loss: first={firstLoss}, last={last}.");

            // RESIDENCY PROOF: a full Step — forward (Half-resident store) + fused Half backward + on-device
            // float↔Half gradient cast bridges + per-node scratch release — keeps the AUTODIFF working set
            // (activations, the matmul/elementwise op results, the cast-boundary gradients, the per-op scratch)
            // GPU-resident: nothing that scales with the activation working set or model depth round-trips. The
            // only host transfers are the small per-PARAMETER gradient reads handed to the optimizer's master-
            // weight update (one per trained parameter — here W1, W2 — independent of activation size / depth). A
            // non-resident path would instead download every op's result: dozens for this graph's forward+backward.
            // (Driving these last per-parameter grad reads to zero needs the optimizer's master-weight update +
            // the forward param-cast to run on-device too — a coupled param-residency change, tracked separately.)
            DeferredArrayMaterializer.ResetMaterializeCount();
            plan.Step(); // intentionally do NOT read the returned loss
            long downloads = DeferredArrayMaterializer.MaterializeCount;
            _out.WriteLine($"deferred GPU→CPU downloads during a full no-read FP16 step: {downloads} " +
                "(per-parameter gradient reads for the optimizer only; the autodiff activations/results/scratch stay resident)");
            // Bound = a small multiple of the trainable-parameter count (2 here), NOT the activation/op count. A
            // per-op host ping-pong of the forward+backward over this ~7-tensor graph would be far more.
            Assert.True(downloads <= 8,
                $"a full FP16 step pulled {downloads} tensors to host — more than the per-parameter gradient reads, " +
                "so an activation/op-result/scratch is round-tripping (residency regression).");
        }
        finally { plan.Dispose(); }
    }
}
