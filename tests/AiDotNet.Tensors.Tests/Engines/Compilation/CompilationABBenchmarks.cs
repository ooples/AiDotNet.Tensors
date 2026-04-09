using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Optimization;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B benchmarks comparing compiled vs eager vs PyTorch performance.
/// Each test measures real steps/sec and reports speedup ratios.
/// Mark as Skip for CI — run manually with --filter.
/// </summary>
public class CompilationABBenchmarks
{
    private readonly ITestOutputHelper _output;

    public CompilationABBenchmarks(ITestOutputHelper output) => _output = output;

    #region End-to-End MLP Training

    [Fact]
    public void MLP_Training_CompiledVsEager()
    {
        var engine = new CpuEngine();
        int batchSize = 32, inputDim = 128, hiddenDim = 64, outputDim = 10;
        int warmup = 20, measure = 200;

        var input = CreateRandom(new[] { batchSize, inputDim }, 1);
        var target = CreateRandom(new[] { batchSize, outputDim }, 2);
        var w1 = CreateRandom(new[] { inputDim, hiddenDim }, 3);
        var w2 = CreateRandom(new[] { hiddenDim, outputDim }, 4);
        float lr = 0.01f;

        // === Eager training (GradientTape) ===
        AutoTracer.Enabled = false;
        double eagerMs = MeasureTrainingStep(engine, input, target, w1, w2, lr, warmup, measure);
        AutoTracer.Enabled = true;

        // === Compiled training (GraphMode + CompiledTrainingPlan) ===
        var w1c = w1.Clone(); var w2c = w2.Clone();
        double compiledMs = MeasureCompiledTrainingStep(engine, input, target, w1c, w2c, lr, warmup, measure);

        double speedup = eagerMs / compiledMs;
        _output.WriteLine($"MLP Training [32x128->64->10]:");
        _output.WriteLine($"  Eager:    {eagerMs:F3} ms/step ({1000.0 / eagerMs:F0} steps/sec)");
        _output.WriteLine($"  Compiled: {compiledMs:F3} ms/step ({1000.0 / compiledMs:F0} steps/sec)");
        _output.WriteLine($"  Speedup:  {speedup:F2}x");

        Assert.True(speedup > 1.0, $"Compiled should be faster than eager, got {speedup:F2}x");
    }

    #endregion

    #region Flash Attention Scaling

    [Theory]
    [InlineData(64, 32)]
    [InlineData(128, 64)]
    [InlineData(256, 64)]
    [InlineData(512, 64)]
    [InlineData(1024, 64)]
    public void FlashAttention_VsNaive_Scaling(int seqLen, int headDim)
    {
        int warmup = 10, measure = 50;
        float scale = 1f / MathF.Sqrt(headDim);

        var q = CreateRandomArray(seqLen * headDim, 10);
        var k = CreateRandomArray(seqLen * headDim, 11);
        var v = CreateRandomArray(seqLen * headDim, 12);

        // Naive attention
        double naiveMs = Measure(() =>
        {
            var output = new float[seqLen * headDim];
            NaiveAttention(q, k, v, output, seqLen, seqLen, headDim, scale);
        }, warmup, measure);

        // Flash Attention
        double flashMs = Measure(() =>
        {
            var output = new float[seqLen * headDim];
            FusedAttention.FlashAttentionForward(q, k, v, output, seqLen, seqLen, headDim, scale);
        }, warmup, measure);

        double speedup = naiveMs / flashMs;
        _output.WriteLine($"Attention seq={seqLen}, d={headDim}:");
        _output.WriteLine($"  Naive: {naiveMs:F3}ms, Flash: {flashMs:F3}ms, Speedup: {speedup:F2}x");
    }

    #endregion

    #region Optimizer SIMD Benchmarks

    [Fact]
    public unsafe void Optimizer_SIMD_VsScalar()
    {
        int length = 100_000;
        int warmup = 50, measure = 500;

        var param = CreateRandomArray(length, 20);
        var grad = CreateRandomArray(length, 21);
        var m = new float[length];
        var v = new float[length];

        // SGD SIMD
        var paramCopy = (float[])param.Clone();
        double sgdSimdMs = Measure(() =>
        {
            fixed (float* p = paramCopy, g = grad)
                FusedOptimizer.SgdUpdateSimd(p, g, length, 0.01f);
        }, warmup, measure);

        // SGD scalar
        paramCopy = (float[])param.Clone();
        double sgdScalarMs = Measure(() =>
        {
            for (int i = 0; i < length; i++)
                paramCopy[i] -= 0.01f * grad[i];
        }, warmup, measure);

        _output.WriteLine($"SGD [100K]: SIMD={sgdSimdMs:F4}ms, Scalar={sgdScalarMs:F4}ms, Speedup={sgdScalarMs / sgdSimdMs:F2}x");

        // Adam SIMD
        var mCopy = new float[length]; var vCopy = new float[length];
        paramCopy = (float[])param.Clone();
        double adamSimdMs = Measure(() =>
        {
            fixed (float* p = paramCopy, g = grad, pm = mCopy, pv = vCopy)
                FusedOptimizer.AdamUpdateSimd(p, g, pm, pv, length, 0.001f, 0.9f, 0.999f, 1e-8f, 1);
        }, warmup, measure);

        _output.WriteLine($"Adam [100K]: SIMD={adamSimdMs:F4}ms");

        // Lion SIMD
        var mLion = new float[length];
        paramCopy = (float[])param.Clone();
        double lionSimdMs = Measure(() =>
        {
            fixed (float* p = paramCopy, g = grad, pm = mLion)
                FusedOptimizer.LionUpdateSimd(p, g, pm, length, 0.001f, 0.9f, 0.99f, 0.01f);
        }, warmup, measure);

        _output.WriteLine($"Lion [100K]: SIMD={lionSimdMs:F4}ms");
    }

    #endregion

    #region Fused Kernel Benchmarks

    [Fact]
    public unsafe void FusedKernels_VsUnfused()
    {
        int length = 100_000;
        int warmup = 50, measure = 500;
        var input = CreateRandomArray(length, 30);
        var output1 = new float[length];
        var output2 = new float[length];

        // Swish: fused vs separate sigmoid + multiply
        double fusedSwishMs = Measure(() =>
        {
            fixed (float* pIn = input, pOut = output1)
                FusedKernels.SwishUnsafe(pIn, pOut, length);
        }, warmup, measure);

        double unfusedSwishMs = Measure(() =>
        {
            fixed (float* pIn = input, pOut = output1, pTemp = output2)
            {
                SimdKernels.SigmoidUnsafe(pIn, pTemp, length);
                SimdKernels.VectorMultiplyUnsafe(pIn, pTemp, pOut, length);
            }
        }, warmup, measure);

        _output.WriteLine($"Swish [100K]: Fused={fusedSwishMs:F4}ms, Unfused={unfusedSwishMs:F4}ms, Speedup={unfusedSwishMs / fusedSwishMs:F2}x");

        // GELU: fused vs separate
        double fusedGeluMs = Measure(() =>
        {
            fixed (float* pIn = input, pOut = output1)
                FusedKernels.GeluUnsafe(pIn, pOut, length);
        }, warmup, measure);

        double unfusedGeluMs = Measure(() =>
        {
            fixed (float* pIn = input, pOut = output1)
                SimdKernels.GELUUnsafe(pIn, pOut, length);
        }, warmup, measure);

        _output.WriteLine($"GELU [100K]: Fused={fusedGeluMs:F4}ms, Unfused={unfusedGeluMs:F4}ms, Speedup={unfusedGeluMs / fusedGeluMs:F2}x");

        // Mish: fused vs separate
        double fusedMishMs = Measure(() =>
        {
            fixed (float* pIn = input, pOut = output1)
                FusedKernels.MishUnsafe(pIn, pOut, length);
        }, warmup, measure);

        double unfusedMishMs = Measure(() =>
        {
            fixed (float* pIn = input, pOut = output1)
                SimdKernels.MishUnsafe(pIn, pOut, length);
        }, warmup, measure);

        _output.WriteLine($"Mish [100K]: Fused={fusedMishMs:F4}ms, Unfused={unfusedMishMs:F4}ms, Speedup={unfusedMishMs / fusedMishMs:F2}x");
    }

    #endregion

    #region Per-Pass A/B

    [Fact]
    public void OptimizationPasses_IndividualImpact()
    {
        var engine = new CpuEngine();
        int m = 32, k = 128, h = 64, n = 10;
        int warmup = 30, measure = 300;

        var input = CreateRandom(new[] { m, k }, 40);
        var w1 = CreateRandom(new[] { k, h }, 41);
        var w2 = CreateRandom(new[] { h, n }, 42);

        // Baseline: no optimization passes
        var opts = new TensorCodecOptions
        {
            EnableConstantFolding = false,
            EnableForwardCSE = false,
            EnableConvBnFusion = false,
            EnablePointwiseFusion = false,
            EnableAttentionFusion = false,
            EnableBlasBatch = false,
            EnableDataflowFusion = false,
            EnableSpectralDecomposition = false,
            EnableMixedPrecision = false
        };

        TensorCodecOptions.SetCurrent(opts);
        double baselineMs = MeasureInference(engine, input, w1, w2, warmup, measure);

        // Enable each pass individually
        string[] passNames = { "ConstantFolding", "ForwardCSE", "PointwiseFusion", "DataflowFusion" };
        foreach (var passName in passNames)
        {
            var passOpts = new TensorCodecOptions
            {
                EnableConstantFolding = passName == "ConstantFolding",
                EnableForwardCSE = passName == "ForwardCSE",
                EnablePointwiseFusion = passName == "PointwiseFusion",
                EnableDataflowFusion = passName == "DataflowFusion",
            };
            TensorCodecOptions.SetCurrent(passOpts);
            double passMs = MeasureInference(engine, input, w1, w2, warmup, measure);
            double delta = (baselineMs - passMs) / baselineMs * 100;
            _output.WriteLine($"  {passName}: {passMs:F3}ms ({(delta > 0 ? "+" : "")}{delta:F1}% vs baseline)");
        }

        TensorCodecOptions.SetCurrent(null); // Reset
        _output.WriteLine($"  Baseline (no passes): {baselineMs:F3}ms");
    }

    #endregion

    #region Helpers

    private static double Measure(Action action, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private double MeasureTrainingStep(IEngine engine, Tensor<float> input, Tensor<float> target,
        Tensor<float> w1, Tensor<float> w2, float lr, int warmup, int measure)
    {
        var numOps = MathHelper.GetNumericOperations<float>();
        return Measure(() =>
        {
            using var tape = new GradientTape<float>();
            var h = engine.ReLU(engine.TensorMatMul(input, w1));
            var pred = engine.TensorMatMul(h, w2);
            var diff = engine.TensorSubtract(pred, target);
            var loss = engine.ReduceSum(engine.TensorMultiply(diff, diff), null);
            var grads = tape.ComputeGradients(loss, new[] { w1, w2 });
            if (grads.TryGetValue(w1, out var g1))
                engine.TensorSubtractInPlace(w1, engine.TensorMultiplyScalar(g1, lr));
            if (grads.TryGetValue(w2, out var g2))
                engine.TensorSubtractInPlace(w2, engine.TensorMultiplyScalar(g2, lr));
        }, warmup, measure);
    }

    private double MeasureCompiledTrainingStep(IEngine engine, Tensor<float> input, Tensor<float> target,
        Tensor<float> w1, Tensor<float> w2, float lr, int warmup, int measure)
    {
        // Compile once
        using var scope = GraphMode.Enable();
        var h = engine.ReLU(engine.TensorMatMul(input, w1));
        var pred = engine.TensorMatMul(h, w2);
        var diff = engine.TensorSubtract(pred, target);
        var loss = engine.ReduceSum(engine.TensorMultiply(diff, diff), null);
        var plan = scope.CompileTraining(new[] { w1, w2 });

        // Replay
        return Measure(() =>
        {
            plan.Step();
            var grads = plan.Gradients;
            if (grads[0] is not null)
                engine.TensorSubtractInPlace(w1, engine.TensorMultiplyScalar(grads[0], lr));
            if (grads[1] is not null)
                engine.TensorSubtractInPlace(w2, engine.TensorMultiplyScalar(grads[1], lr));
        }, warmup, measure);
    }

    private double MeasureInference(IEngine engine, Tensor<float> input,
        Tensor<float> w1, Tensor<float> w2, int warmup, int measure)
    {
        using var scope = GraphMode.Enable();
        var h = engine.ReLU(engine.TensorMatMul(input, w1));
        engine.TensorMatMul(h, w2);
        var plan = scope.CompileInference<float>();
        return Measure(() => plan.Execute(), warmup, measure);
    }

    private static void NaiveAttention(float[] q, float[] k, float[] v, float[] output,
        int seqQ, int seqK, int headDim, float scale)
    {
        var scores = new float[seqQ * seqK];
        for (int i = 0; i < seqQ; i++)
            for (int j = 0; j < seqK; j++)
            {
                float dot = 0f;
                for (int d = 0; d < headDim; d++)
                    dot += q[i * headDim + d] * k[j * headDim + d];
                scores[i * seqK + j] = dot * scale;
            }
        for (int i = 0; i < seqQ; i++)
        {
            float max = float.NegativeInfinity;
            for (int j = 0; j < seqK; j++) if (scores[i * seqK + j] > max) max = scores[i * seqK + j];
            float sum = 0f;
            for (int j = 0; j < seqK; j++) { scores[i * seqK + j] = MathF.Exp(scores[i * seqK + j] - max); sum += scores[i * seqK + j]; }
            for (int j = 0; j < seqK; j++) scores[i * seqK + j] /= sum;
        }
        for (int i = 0; i < seqQ; i++)
            for (int d = 0; d < headDim; d++)
            {
                float s = 0f;
                for (int j = 0; j < seqK; j++) s += scores[i * seqK + j] * v[j * headDim + d];
                output[i * headDim + d] = s;
            }
    }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int len = 1; for (int i = 0; i < shape.Length; i++) len *= shape[i];
        var data = new float[len];
        for (int i = 0; i < len; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static float[] CreateRandomArray(int length, int seed)
    {
        var rng = new Random(seed);
        var data = new float[length];
        for (int i = 0; i < length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return data;
    }

    #endregion
}
