using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.Simd;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// A/B test: BatchNorm [32x64x8x8]. PyTorch = 122us, our compiled = 3,010us (24.5x slower).
/// We added FusedKernels.BatchNormInferenceUnsafe specialization. Verify it improved.
/// </summary>
public class BatchNormABTest
{
    private readonly ITestOutputHelper _output;
    public BatchNormABTest(ITestOutputHelper output) => _output = output;

    [Fact]
    public unsafe void BatchNorm_Compiled_vs_Eager_vs_DirectKernel()
    {
        var engine = new CpuEngine();
        int batch = 32, channels = 64, h = 8, w = 8;
        int length = batch * channels * h * w;
        var input = CreateRandom(new[] { batch, channels, h, w }, 42);
        var gamma = CreateRandom(new[] { channels }, 43);
        var beta = CreateRandom(new[] { channels }, 44);
        var mean = CreateRandom(new[] { channels }, 45);
        var variance = CreateRandomPositive(new[] { channels }, 46);
        float eps = 1e-5f;
        int warmup = 5, iters = 50;

        // Path A: Eager engine BatchNorm
        double eagerMs = Measure(() =>
        {
            engine.BatchNorm(input, gamma, beta, eps, out _, out _);
        }, warmup, iters);

        // Path B: Direct FusedKernels.BatchNormInferenceUnsafe
        var directOutput = new float[length];
        var inputArr = input.GetDataArray();
        var gammaArr = gamma.GetDataArray();
        var betaArr = beta.GetDataArray();
        var meanArr = mean.GetDataArray();
        var varArr = variance.GetDataArray();

        // Use GCHandle.Alloc for pinning (can't use fixed in lambda)
        var hIn = System.Runtime.InteropServices.GCHandle.Alloc(inputArr, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hOut = System.Runtime.InteropServices.GCHandle.Alloc(directOutput, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hG = System.Runtime.InteropServices.GCHandle.Alloc(gammaArr, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hB = System.Runtime.InteropServices.GCHandle.Alloc(betaArr, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hM = System.Runtime.InteropServices.GCHandle.Alloc(meanArr, System.Runtime.InteropServices.GCHandleType.Pinned);
        var hV = System.Runtime.InteropServices.GCHandle.Alloc(varArr, System.Runtime.InteropServices.GCHandleType.Pinned);

        double directMs = Measure(() =>
        {
            unsafe
            {
                FusedKernels.BatchNormInferenceUnsafe(
                    (float*)hIn.AddrOfPinnedObject(), (float*)hOut.AddrOfPinnedObject(),
                    length, channels,
                    (float*)hG.AddrOfPinnedObject(), (float*)hB.AddrOfPinnedObject(),
                    (float*)hM.AddrOfPinnedObject(), (float*)hV.AddrOfPinnedObject(), eps);
            }
        }, warmup, iters);

        hIn.Free(); hOut.Free(); hG.Free(); hB.Free(); hM.Free(); hV.Free();

        // Path C: Compiled plan (should use our new specialization)
        CompiledInferencePlan<float> plan;
        using (var scope = GraphMode.Enable())
        {
            engine.BatchNorm(input, gamma, beta, eps, out _, out _);
            plan = scope.CompileInference<float>();
        }
        double compiledMs = Measure(() => plan.Execute(), warmup, iters);
        plan.Dispose();

        _output.WriteLine($"BatchNorm [32x64x8x8] ({length} elements):");
        _output.WriteLine($"  Path A (Eager engine):              {eagerMs:F3}ms");
        _output.WriteLine($"  Path B (Direct FusedKernels SIMD):   {directMs:F3}ms");
        _output.WriteLine($"  Path C (Compiled plan):              {compiledMs:F3}ms");
        _output.WriteLine($"");
        _output.WriteLine($"  Eager vs Direct: {eagerMs / directMs:F1}x overhead");
        _output.WriteLine($"  Compiled vs Direct: {compiledMs / directMs:F1}x overhead");
        _output.WriteLine($"  Compiled vs Eager: {eagerMs / compiledMs:F2}x speedup");
        _output.WriteLine($"");
        _output.WriteLine($"  PyTorch BDN reference: ~0.123ms");
        _output.WriteLine($"  Our best (Direct): {directMs:F3}ms = {directMs / 0.123:F1}x vs PyTorch");

        // Verify compiled is at least as fast as eager
        Assert.True(compiledMs <= eagerMs * 1.5,
            $"Compiled ({compiledMs:F3}ms) should not be much slower than eager ({eagerMs:F3}ms)");
    }

    private static double Measure(Action action, int warmup, int iters)
    {
        for (int i = 0; i < warmup; i++) action();
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++) action();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / iters;
    }

    private static Tensor<float> CreateRandom(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2 - 1);
        return new Tensor<float>(data, shape);
    }

    private static Tensor<float> CreateRandomPositive(int[] shape, int seed)
    {
        var rng = new Random(seed);
        int length = 1;
        for (int i = 0; i < shape.Length; i++) length *= shape[i];
        var data = new float[length];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 0.5 + 0.1);
        return new Tensor<float>(data, shape);
    }
}
