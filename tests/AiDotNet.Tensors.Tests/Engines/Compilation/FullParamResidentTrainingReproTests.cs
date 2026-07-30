// REGRESSION (GPU-residency campaign): making the trainable PARAMETERS GPU-resident (Tensor.Gpu()) on the plain
// FP32 fused-Adam compiled-training path used to SILENTLY BREAK TRAINING — the loss went completely flat. The
// TimeSeries family (NBEATS / Informer / DeepAR / ...) makes params resident BY DEFAULT on GPU float+Adam
// (CompiledTapeTrainingStep.TryStepWithFusedOptimizer, AIDOTNET_GPU_RESIDENT_PARAMS != 0), so this was the
// "compiled fused-tape-plan mistrains on GPU" bug. Measured on an RTX 3080, same graph, only diff = params
// resident or not: resident 7.70 -> 7.70 (flat) vs non-resident 7.70 -> 1.31. Now BOTH converge (~1.31).
//
// The bug was a coupled forward/gradient/weight buffer desync in the on-device fused optimizer path, fixed by
// three changes this test guards:
//   1. Tensor.Gpu() now version-tags the uploaded buffer (_gpuBufferVersion = Version) — without it every
//      version-gated resident-buffer consumer immediately judged the fresh buffer stale and detached it.
//   2. CompiledTrainingPlan.ConfigureOptimizerFloat consumes the AUTHORITATIVE gradient: it trusts the resident
//      grad buffer only when it is version-fresh (CUDA-graph capture or the FP16-hetero resident backward);
//      otherwise the eager backward wrote the HOST grad and the resident buffer is a memset-zero leftover, so it
//      uploads the host gradient instead of applying ~zero.
//   3. After each on-device optimizer update it RE-ARMS the device->host deferred download for the parameter, so
//      the eager forward's next GetDataArray() re-downloads the freshly-updated weights instead of the one-shot
//      host download cached at first read (which left the forward training on FROZEN weights).

#nullable disable
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

[Collection(MixedPrecisionTestCollection.Name)]
public class FullParamResidentTrainingReproTests
{
    private readonly ITestOutputHelper _out;
    public FullParamResidentTrainingReproTests(ITestOutputHelper o) { _out = o; }

    private static float[] RandData(int[] shape, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        int n = 1; foreach (var d in shape) n *= d;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return data;
    }

    [SkippableTheory]
    [InlineData(false)] // baseline: non-resident params train correctly
    [InlineData(true)]  // GPU-resident params (the TimeSeries default) must train IDENTICALLY, not flat-line
    public void Fp32FusedPath_ResidentParams_MustTrain(bool residentParams)
    {
        using var gpu = new DirectGpuTensorEngine();
        Skip.If(!gpu.IsGpuAvailable, "needs a DirectGpu backend (CUDA/OpenCL/...).");

        var input = new Tensor<float>(RandData(new[] { 32, 64 }, 1, scale: 0.5f), new[] { 32, 64 });
        var w1 = new Tensor<float>(RandData(new[] { 64, 64 }, 2, scale: 0.1f), new[] { 64, 64 });
        var w2 = new Tensor<float>(RandData(new[] { 64, 64 }, 3, scale: 0.1f), new[] { 64, 64 });

        // Make params GPU-resident BEFORE the trace/compile — exactly what the real TimeSeries fused path does
        // (CompiledTapeTrainingStep.TryStepWithFusedOptimizer calls Tensor.Gpu() on the params before it compiles,
        // AIDOTNET_GPU_RESIDENT_PARAMS on by default). This is the configuration that used to silently mistrain on
        // GPU: the on-device fused Adam moved the resident weight buffer while the eager forward kept reading a
        // frozen one-shot host download of it, so the loss went completely flat.
        if (residentParams) { w1.Gpu(); w2.Gpu(); }

        CompiledTrainingPlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            var h = gpu.TensorMatMul(input, w1);
            var pred = gpu.TensorMatMul(h, w2);
            var sqErr = gpu.TensorMultiply(pred, pred);
            gpu.ReduceSum(sqErr, null);
            plan = scope.CompileTraining(new[] { w1, w2 });
        }
        GraphMode.SetCurrent(null);

        try
        {
            float firstLoss = plan.Step().GetFlat(0);
            Assert.True(!float.IsNaN(firstLoss) && !float.IsInfinity(firstLoss), $"first loss must be finite, got {firstLoss}.");
            plan.ConfigureOptimizer(OptimizerType.Adam, learningRate: 1e-3f);

            float last = firstLoss;
            for (int i = 0; i < 15; i++) last = plan.Step().GetFlat(0);
            _out.WriteLine($"[residentParams={residentParams}] loss: first={firstLoss:G6} last={last:G6}");

            // Whether or not the params are GPU-resident, an identical graph must learn identically — the loss
            // must drop meaningfully. A GPU-resident-param regression re-freezes the forward's weights → flat loss.
            Assert.True(last < firstLoss * 0.5f,
                $"training must reduce the loss regardless of param residency: first={firstLoss}, last={last}.");
        }
        finally { plan.Dispose(); }
    }
}
