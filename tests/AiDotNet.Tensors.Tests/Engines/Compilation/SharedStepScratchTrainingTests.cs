using System;
using System.Linq;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.Gpu;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Two compiled plans on one CUDA engine that step in turn, as a GAN's discriminator and generator do, co-own the step
/// scratch neither keeps past its step (SharedStepScratch). Each plan's loss and weights are read only after the other
/// plan has stepped -- the moment a buffer both plans write would show -- and must equal, bit for bit, the same training
/// with co-ownership off, while the second plan actually took memory from the first.
/// </summary>
[Collection("DirectGpuSerial")]
public class SharedStepScratchTrainingTests : IDisposable
{
    private const int Steps = 9;   // past the plans' eager warm-up, so their graphs are captured and replayed
    private readonly IEngine _prior = AiDotNetEngine.Current;
    private readonly bool _priorEnabled = SharedStepScratch.Enabled;
    private readonly ITestOutputHelper _output;

    public SharedStepScratchTrainingTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        AiDotNetEngine.Current = _prior;
        SharedStepScratch.Enabled = _priorEnabled;
    }

    private static Tensor<float> Rand(int[] shape, int seed, float scale)
    {
        var rng = RandomHelper.CreateSeededRandom(seed);
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++) t[i] = (float)((rng.NextDouble() * 2 - 1) * scale);
        return t;
    }

    private sealed class Run
    {
        public float[] LossesA = new float[Steps];
        public float[] LossesB = new float[Steps];
        public float[][] Weights = Array.Empty<float[]>();
        public int Takes;
        public int Revocations;
    }

    private static Run Train(DirectGpuTensorEngine engine, bool share, PrecisionMode? autocast = null)
    {
        // Mixed precision narrows the convolutions' operands into step scratch of their own: the plans still co-own it.
        using var mixed = autocast is { } precision ? new AutocastScope(precision) : null;
        SharedStepScratch.Enabled = share;
        AiDotNetEngine.Current = engine;
        const int batch = 16;
        // Plan A: a convolutional scorer (a discriminator's shape).
        var xa = Rand([batch, 3, 32, 32], 1, 1f);
        var ta = Rand([batch, 16, 8, 8], 2, 0.5f);
        var ka1 = Rand([8, 3, 4, 4], 3, 0.2f);
        var ka2 = Rand([16, 8, 4, 4], 4, 0.15f);
        var pa = new[] { ka1, ka2 };
        // Plan B: a transposed-convolution generator.
        var zb = Rand([batch, 16, 4, 4], 5, 1f);
        var tb = Rand([batch, 3, 16, 16], 6, 0.5f);
        var kb1 = Rand([16, 8, 4, 4], 7, 0.15f);
        var bb1 = Rand([8], 8, 0.3f);
        var kb2 = Rand([8, 3, 4, 4], 9, 0.2f);
        var bb2 = Rand([3], 10, 0.3f);
        var pb = new[] { kb1, bb1, kb2, bb2 };
        foreach (var p in pa.Concat(pb)) p.Gpu();

        ICompiledTrainingPlan<float> planA, planB;
        using (var scope = GraphMode.EnableTraining(pa))
        {
            var h = engine.ReLU(engine.Conv2D(xa, ka1, 2, 1, 1));
            h = engine.Conv2D(h, ka2, 2, 1, 1);
            // The unscaled loss is a one-element value that dies inside A's step: A offers it, and B's own loss (one
            // element too) takes it, then turns out to keep it past its step -- the case the taker must revoke.
            engine.TensorMultiplyScalar(engine.TensorMSELoss(h, ta), 0.5f);
            planA = scope.CompileTraining(pa);
        }
        using (var scope = GraphMode.EnableTraining(pb))
        {
            var h = engine.ReLU(engine.FusedConvTranspose2D(zb, kb1, bb1, 2, 2, 1, 1, 0, 0, FusedActivationType.None));
            h = engine.FusedConvTranspose2D(h, kb2, bb2, 2, 2, 1, 1, 0, 0, FusedActivationType.None);
            engine.TensorMSELoss(h, tb);
            planB = scope.CompileTraining(pb);
        }

        var shared = engine.SharedStepScratch;
        int takes0 = shared.Takes, revocations0 = shared.Revocations;
        using (planA)
        using (planB)
        {
            planA.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
            planB.ConfigureOptimizer(OptimizerType.Adam, learningRate: 0.01f);
            var run = new Run();
            Tensor<float>? pendingB = null;
            for (int s = 0; s < Steps; s++)
            {
                Rand(xa.Shape.ToArray(), 100 + s, 1f).AsSpan().CopyTo(xa.AsWritableSpan());
                Rand(zb.Shape.ToArray(), 200 + s, 1f).AsSpan().CopyTo(zb.AsWritableSpan());
                var lossA = planA.Step();
                if (pendingB is not null) run.LossesB[s - 1] = pendingB.ToArray()[0];   // B's last loss, read after A stepped
                pendingB = planB.Step();
                run.LossesA[s] = lossA.ToArray()[0];   // A's loss, read after B stepped
            }
            planA.Step();
            run.LossesB[Steps - 1] = pendingB!.ToArray()[0];
            run.Weights = pa.Concat(pb).Select(p => p.ToArray()).ToArray();
            run.Takes = shared.Takes - takes0;
            run.Revocations = shared.Revocations - revocations0;
            return run;
        }
    }

    private static DirectGpuTensorEngine? TryCuda()
    {
        try
        {
            var candidate = new DirectGpuTensorEngine();
            if (candidate.IsGpuAvailable && candidate.GetBackend() is CudaBackend) return candidate;
            candidate.Dispose();
        }
        catch (Exception ex) when (ex is PlatformNotSupportedException or DllNotFoundException) { }
        return null;
    }

    [SkippableTheory]
    [InlineData(null)]
    [InlineData(PrecisionMode.BFloat16)]
    [InlineData(PrecisionMode.Float16)]
    public void Plans_stepping_in_turn_co_own_scratch_and_train_bit_identically_to_private_scratch(PrecisionMode? autocast)
    {
        var gpu = TryCuda();
        Skip.If(gpu is null, "CUDA backend did not resolve (whole-step capture is CUDA's).");
        if (gpu is null) return;
        using (gpu)
        {
            Train(gpu, share: false, autocast);   // tunes the kernels both runs below then pick alike
            var isolated = Train(gpu, share: false, autocast);
            var shared = Train(gpu, share: true, autocast);
            _output.WriteLine($"takes={shared.Takes} revocations={shared.Revocations}");
            Assert.Equal(0, isolated.Takes);
            Assert.True(shared.Takes > 0, "the second plan took none of the first plan's step scratch");
            for (int s = 0; s < Steps; s++)
            {
                Assert.True(BitConverter.SingleToInt32Bits(isolated.LossesA[s]) == BitConverter.SingleToInt32Bits(shared.LossesA[s]),
                    $"plan A step {s}: loss {shared.LossesA[s]:G9} co-owned vs {isolated.LossesA[s]:G9} private");
                Assert.True(BitConverter.SingleToInt32Bits(isolated.LossesB[s]) == BitConverter.SingleToInt32Bits(shared.LossesB[s]),
                    $"plan B step {s}: loss {shared.LossesB[s]:G9} co-owned vs {isolated.LossesB[s]:G9} private");
            }
            for (int p = 0; p < isolated.Weights.Length; p++)
                for (int i = 0; i < isolated.Weights[p].Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(isolated.Weights[p][i]) == BitConverter.SingleToInt32Bits(shared.Weights[p][i]),
                        $"parameter {p}[{i}]: {shared.Weights[p][i]:G9} co-owned vs {isolated.Weights[p][i]:G9} private");
            // B's loss took A's one-element buffer: the taker's validation must have revoked it (the plan keeps its loss
            // tensor across steps, so the step-persistent contract excludes it from co-ownership).
            Assert.True(shared.Revocations > 0, "no plan revoked a buffer it took for a value it keeps (B's loss)");
        }
    }
}
