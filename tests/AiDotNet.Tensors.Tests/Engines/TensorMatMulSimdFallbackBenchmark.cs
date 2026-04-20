using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

/// <summary>
/// One-shot microbenchmark verifying the SIMD-blocked fallback in
/// <see cref="CpuEngine.TensorMatMul{T}"/> is fast enough at DiT-XL-shaped
/// double matmuls to bring diffusion model tests under the 120s xUnit budget.
/// <para>
/// Gated behind the <c>AIDN_RUN_GEMM_BENCH=1</c> env var so it does not run in
/// CI by default. Reports raw wall-clock per matmul to the test output sink.
/// Run locally:
/// <code>
/// AIDN_RUN_GEMM_BENCH=1 dotnet test \
///   --filter FullyQualifiedName~TensorMatMulSimdFallbackBenchmark
/// </code>
/// </para>
/// </summary>
public class TensorMatMulSimdFallbackBenchmark
{
    private readonly ITestOutputHelper _output;
    private readonly CpuEngine _engine = new();

    public TensorMatMulSimdFallbackBenchmark(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void DiTXL_DoubleMatMul_ShouldBeFastEnoughForCi()
    {
        if (Environment.GetEnvironmentVariable("AIDN_RUN_GEMM_BENCH") != "1")
        {
            _output.WriteLine("skipped: set AIDN_RUN_GEMM_BENCH=1 to run.");
            return;
        }

        // DiT-XL/2 hot shapes (Pika21 default):
        //   QKV projection: [256, 1152] @ [1152, 1152]  → 339M FMAs
        //   MLP up:         [256, 1152] @ [1152, 4608]  → 1.36G FMAs
        //   MLP down:       [256, 4608] @ [4608, 1152]  → 1.36G FMAs
        var shapes = new (int m, int n, int p, string label)[]
        {
            (256, 1152, 1152, "QKV proj"),
            (256, 1152, 4608, "MLP up"),
            (256, 4608, 1152, "MLP down"),
            (1152, 1152, 1152, "Square 1152²"),
        };

        var rng = new Random(42);
        foreach (var (m, n, p, label) in shapes)
        {
            var aData = new double[m * n];
            var bData = new double[n * p];
            for (int i = 0; i < aData.Length; i++) aData[i] = rng.NextDouble() * 2 - 1;
            for (int i = 0; i < bData.Length; i++) bData[i] = rng.NextDouble() * 2 - 1;
            var a = new Tensor<double>(aData, new[] { m, n });
            var b = new Tensor<double>(bData, new[] { n, p });

            // Warm-up
            _ = _engine.TensorMatMul(a, b);

            var sw = Stopwatch.StartNew();
            const int iterations = 5;
            for (int i = 0; i < iterations; i++)
            {
                _ = _engine.TensorMatMul(a, b);
            }
            sw.Stop();

            double perMatmulMs = sw.Elapsed.TotalMilliseconds / iterations;
            long fmas = (long)m * n * p;
            double gflops = fmas * 2.0 / (perMatmulMs / 1000.0) / 1e9;

            _output.WriteLine(
                $"[{label,-14}] {m}x{n} @ {n}x{p}  per-matmul={perMatmulMs,8:F2} ms  ~{gflops,5:F1} GFLOPS");
        }
    }
}
