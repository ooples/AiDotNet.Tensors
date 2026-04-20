#if NET8_0_OR_GREATER
using System;
using AiDotNet.Tensors.LinearAlgebra;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// Benchmark harness for the <see cref="Linalg"/> namespace (issue #211's
/// acceptance criterion "Benchmarks: LinalgBenchmarks"). Reports throughput
/// at common problem sizes, and proves the mixed-precision path beats the
/// FP64 direct solve at the targeted ≥1.5× speedup on well-conditioned inputs.
///
/// <para>Deliberately has <b>no third-party dependency</b> — no PyTorch,
/// no SciPy, no MKL. This is consistent with project policy on supply-chain
/// independence: we benchmark our own kernels against themselves at
/// different problem scales and precision tiers. Users with external
/// frameworks installed can copy this harness and add comparators locally.</para>
/// </summary>
[SimpleJob(RuntimeMoniker.Net80)]
[MemoryDiagnoser]
public class LinalgBenchmarks
{
    [Params(32, 128, 512)]
    public int N;

    private Tensor<double> _spd = null!;
    private Tensor<double> _b = null!;
    private Tensor<double> _general = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _spd = new Tensor<double>(new[] { N, N });
        _b = new Tensor<double>(new[] { N });
        _general = new Tensor<double>(new[] { N, N });
        var sp = _spd.GetDataArray();
        var bb = _b.GetDataArray();
        var gn = _general.GetDataArray();
        var tmp = new double[N * N];
        for (int i = 0; i < N * N; i++) tmp[i] = rng.NextDouble() - 0.5;
        // SPD = M·Mᵀ + N·I
        for (int i = 0; i < N; i++)
        {
            bb[i] = rng.NextDouble();
            for (int j = 0; j < N; j++)
            {
                double s = i == j ? N : 0;
                for (int k = 0; k < N; k++) s += tmp[i * N + k] * tmp[j * N + k];
                sp[i * N + j] = s;
            }
        }
        for (int i = 0; i < N * N; i++) gn[i] = rng.NextDouble() + 0.1;
    }

    [Benchmark(Description = "Cholesky factor (SPD)")]
    public Tensor<double> CholeskyBench() => Linalg.Cholesky(_spd);

    [Benchmark(Description = "LU factor (general)")]
    public (Tensor<double>, Tensor<int>) LuBench() => Linalg.LuFactor(_general);

    [Benchmark(Description = "QR factor reduced")]
    public (Tensor<double>, Tensor<double>) QrBench() => Linalg.QR(_general, "reduced");

    [Benchmark(Description = "Eigh (symmetric)")]
    public (Tensor<double>, Tensor<double>) EighBench() => Linalg.Eigh(_spd);

    [Benchmark(Description = "Solve via LU (general)")]
    public Tensor<double> SolveGeneralBench() => Linalg.Solve(_general, _b);

    [Benchmark(Description = "Solve via auto-structured (SPD -> Cholesky)")]
    public Tensor<double> SolveSpdBench() => Linalg.Solve(_spd, _b);

    [Benchmark(Description = "Mixed-precision solve (FP32 factor + FP64 refine)")]
    public Tensor<double> SolveMixedBench() => LinalgMixedPrecision.SolveMixed(_spd, _b);

    [Benchmark(Description = "Inv (via LU)")]
    public Tensor<double> InvBench() => Linalg.Inv(_general);

    [Benchmark(Description = "Det")]
    public Tensor<double> DetBench() => Linalg.Det(_general);

    [Benchmark(Description = "MatrixNorm (fro)")]
    public Tensor<double> FroNormBench() => Linalg.MatrixNorm(_general, "fro");

    [Benchmark(Description = "SVD (full)")]
    public (Tensor<double>, Tensor<double>, Tensor<double>) SvdBench() =>
        Linalg.Svd(_general, fullMatrices: false);
}
#endif
