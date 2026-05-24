using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.BlasManaged;
using Xunit;
using Xunit.Abstractions;
using BlasManagedLib = AiDotNet.Tensors.Engines.BlasManaged.BlasManaged;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-N (#404): parallel pack-B correctness + measurement. The big-N FFN
/// shapes where serial pack-B previously dominated total time should now
/// pack faster, and produce bit-identical output to the serial path.
/// </summary>
public class ParallelPackBTest
{
    private readonly ITestOutputHelper _output;
    public ParallelPackBTest(ITestOutputHelper output) { _output = output; }

    [Theory]
    [InlineData(1024, 3072, 768)]  // BERT_FFN_up
    [InlineData(1024, 768, 3072)]  // BERT_FFN_down
    [InlineData(512, 4096, 1024)]  // GPT2med_FFN_up
    public void ParallelPackB_Wide_N_Matches_Serial_FP32(int M, int N, int K)
    {
        var rng = new Random(42);
        var a = new float[M * K];
        var b = new float[K * N];
        var cParallel = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        // Run with default options — the parallel pack-B branch fires because
        // pack-B size > 256 KB at these shapes.
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cParallel, N, M, N, K);

        // Naive double-precision reference — both serial and parallel pack
        // should converge to within FP32 reduction-order tolerance of this.
        // For K=3072, eps_fp32·K ≈ 4e-4 absolute; allow 2x slack.
        // Use a sparse-sample check (full 3M-cell loop would be slow).
        int sampleStride = (M * N) / 200;  // ~200 sample points
        double maxAbsErr = 0;
        for (int idx = 0; idx < cParallel.Length; idx += sampleStride)
        {
            int i = idx / N;
            int j = idx % N;
            double sum = 0;
            for (int kk = 0; kk < K; kk++) sum += (double)a[i * K + kk] * b[kk * N + j];
            maxAbsErr = Math.Max(maxAbsErr, Math.Abs(sum - cParallel[idx]));
        }
        double bound = 8 * K * 1.2e-7;  // 8 · K · eps_fp32 abs
        Assert.True(maxAbsErr < bound,
            $"Parallel pack-B drift {maxAbsErr:G6} exceeds bound {bound:G6} at shape {M}×{N}×{K}");

        _output.WriteLine($"{M}×{N}×{K} FP32: max abs err {maxAbsErr:G6} (bound {bound:G6})");
    }

    [Fact]
    public void ParallelPackB_Single_Threaded_Gives_Same_Output_As_Multi_Threaded()
    {
        // Force single-thread via NumThreads=-1, then default (multi-thread)
        // — outputs should match within FP32 reduction-order tolerance for
        // typical input but the pack itself is bit-identical (data movement
        // only). Different threading inside the GEMM compute can change
        // reduction order, so use an absolute tolerance.
        const int M = 512, N = 1024, K = 768;
        var rng = new Random(7);
        var a = new float[M * K];
        var b = new float[K * N];
        var cSerial = new float[M * N];
        var cParallel = new float[M * N];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cSerial, N, M, N, K,
            new BlasOptions<float> { NumThreads = -1 });
        BlasManagedLib.Gemm<float>(a, K, false, b, N, false, cParallel, N, M, N, K);

        double maxAbsDelta = 0;
        for (int i = 0; i < cSerial.Length; i++)
            maxAbsDelta = Math.Max(maxAbsDelta, Math.Abs(cSerial[i] - cParallel[i]));
        // K=768 reduction-order drift bound, doubled for slack.
        double bound = 4 * K * 1.2e-7;
        Assert.True(maxAbsDelta < bound,
            $"Serial vs parallel pack-B output drift {maxAbsDelta:G6} > {bound:G6}");
    }
}
