using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.Compilation;

/// <summary>
/// Times CompiledTrainingPlan.Step() (the compiled forward+backward+update that is 94-99% of an
/// AiDotNet training step) at matmul-heavy shapes, to track routing the compiled-plan GEMM
/// delegates from single-threaded SimdGemm to the parallel BlasManaged.Gemm.
/// </summary>
public class CompiledTrainingPlanGemmPerfBench
{
    private readonly ITestOutputHelper _o;
    public CompiledTrainingPlanGemmPerfBench(ITestOutputHelper o) => _o = o;

    private static Tensor<float> Rand(int[] shape, int seed)
    {
        var rng = new Random(seed);
        var t = new Tensor<float>(shape);
        var s = t.AsWritableSpan();
        for (int i = 0; i < s.Length; i++) s[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
        return t;
    }

    [Theory(Skip = "Perf benchmark — run manually")]
    [InlineData(512, 256, 256)]
    [InlineData(2048, 128, 128)]
    [InlineData(256, 256, 256)]
    public void Bench_RawGemm(int M, int K, int N)
    {
        var a = new float[M * K]; var b = new float[K * N]; var c = new float[M * N];
        var rng = new Random(7);
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() - 0.5);
        double gflop = 2.0 * M * K * N / 1e9;
        _o.WriteLine($"native BLAS IsAvailable = {AiDotNet.Tensors.Helpers.BlasProvider.IsAvailable}");

        Action simd = () => AiDotNet.Tensors.Engines.Simd.SimdGemm.Sgemm(
            a.AsSpan(0, M * K), K, false, b.AsSpan(0, K * N), N, false, c.AsSpan(0, M * N), M, K, N);
        Action blas = () => AiDotNet.Tensors.Engines.BlasManaged.BlasManaged.Gemm<float>(
            a.AsSpan(0, M * K), K, false, b.AsSpan(0, K * N), N, false, c.AsSpan(0, M * N), N, M, N, K,
            new AiDotNet.Tensors.Engines.BlasManaged.BlasOptions<float> { PackingMode = AiDotNet.Tensors.Engines.BlasManaged.PackingMode.DisableAutotune });
        Action nativeFull = () => AiDotNet.Tensors.Helpers.BlasProvider.TryGemm(
            M, N, K, a, 0, K, b, 0, N, c, 0, N);

        foreach (var (nm, fn) in new[] { ("SimdGemm", simd), ("BlasManaged", blas), ("NativeFull", nativeFull) })
        {
            for (int i = 0; i < 50; i++) fn();
            double best = double.MaxValue;
            var sw = new Stopwatch();
            for (int i = 0; i < 400; i++) { sw.Restart(); fn(); sw.Stop(); best = Math.Min(best, sw.Elapsed.TotalMilliseconds); }
            _o.WriteLine($"raw GEMM [M={M},K={K},N={N}] {nm,-12}: min {best:F4} ms = {gflop / (best / 1000.0):F1} GF/s");
        }
    }

    [Theory(Skip = "Perf benchmark — run manually")]
    [InlineData(256, 256)]
    [InlineData(512, 256)]
    [InlineData(2048, 128)]
    public void Bench_CompiledTrainingStep(int M, int D)
    {
        var engine = new CpuEngine();
        var input = Rand(new[] { M, D }, 1);
        var target = Rand(new[] { M, 10 }, 2);
        var w1 = Rand(new[] { D, D }, 3);
        var w2 = Rand(new[] { D, D }, 4);
        var w3 = Rand(new[] { D, 10 }, 5);

        using var scope = GraphMode.Enable();
        var h1 = engine.ReLU(engine.TensorMatMul(input, w1));
        var h2 = engine.ReLU(engine.TensorMatMul(h1, w2));
        var pred = engine.TensorMatMul(h2, w3);
        var diff = engine.TensorSubtract(pred, target);
        var loss = engine.ReduceSum(engine.TensorMultiply(diff, diff), null);
        var plan = scope.CompileTraining(new[] { w1, w2, w3 });

        for (int i = 0; i < 60; i++) plan.Step();
        CompiledTrainingPlan<float>.ResetProf();
        const int iters = 600;
        double best = double.MaxValue, sum = 0;
        var sw = new Stopwatch();
        for (int i = 0; i < iters; i++)
        {
            sw.Restart();
            plan.Step();
            sw.Stop();
            double ms = sw.Elapsed.TotalMilliseconds;
            best = Math.Min(best, ms); sum += ms;
        }
        _o.WriteLine($"compiled train step [M={M},D={D}]: min {best:F4} ms | avg {sum / iters:F4} ms");
        if (Environment.GetEnvironmentVariable("AIDOTNET_PROFILE_STEP") == "1")
        {
            int c = Math.Max(1, CompiledTrainingPlan<float>.ProfStepCount);
            _o.WriteLine($"  PHASES: fwd={CompiledTrainingPlan<float>.ProfForwardUs / 1000.0 / c:F3}ms " +
                         $"gradZero={CompiledTrainingPlan<float>.ProfGradZeroUs / 1000.0 / c:F3}ms " +
                         $"bwd={CompiledTrainingPlan<float>.ProfBackwardUs / 1000.0 / c:F3}ms " +
                         $"optim={CompiledTrainingPlan<float>.ProfOptimUs / 1000.0 / c:F3}ms");
            var per = CompiledTrainingPlan<float>.ProfPerStepUs;
            var names = CompiledTrainingPlan<float>.ProfBackwardStepNames;
            if (per.Length > 0)
            {
                var idx = new int[per.Length];
                for (int i = 0; i < idx.Length; i++) idx[i] = i;
                Array.Sort(idx, (a, b) => per[b].CompareTo(per[a]));
                for (int r = 0; r < Math.Min(8, idx.Length); r++)
                {
                    int i = idx[r];
                    string nm = (names is not null && i < names.Length) ? names[i] : $"#{i}";
                    _o.WriteLine($"    op {per[i] / 1000.0 / c:F3}ms  {nm}");
                }
            }
        }
    }
}
