// Copyright (c) AiDotNet. All rights reserved.
// #226 regression guard. Before the per-thread cuBLAS handle fix, N host threads shared ONE
// non-thread-safe cublasHandle on the process-global GPU engine; concurrent GEMMs corrupted the
// handle's internal state and faulted with a sticky CUDA-700 (illegal address), later misreported at
// the deferred DtoH download as "buffer released before materialization (#226)". This reproduces that
// exact scenario — the Ooples research sweep trains ~12 models in parallel over one engine — and
// asserts every concurrent GEMM both SUCCEEDS (no crash) and MATCHES the CPU reference (no corruption).
// Skipped when no CUDA GPU is present (acts as a compile-time guard on CI runners without GPUs).

using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.LinearAlgebra;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

[Collection("DirectGpuSerial")]
public class CublasConcurrencyReproTests : IDisposable
{
    private readonly DirectGpuTensorEngine? _gpu;
    private readonly bool _available;
    private readonly CpuEngine _cpu = new();

    public CublasConcurrencyReproTests()
    {
        try
        {
            _gpu = new DirectGpuTensorEngine();
            _available = _gpu.IsGpuAvailable;
        }
        catch
        {
            _available = false;
        }
    }

    public void Dispose()
    {
        _gpu?.Dispose();
        GC.SuppressFinalize(this);
    }

    private static Matrix<float> Randn(int seed, int rows, int cols)
    {
        var rng = new Random(seed);
        var m = new Matrix<float>(rows, cols);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                m[i, j] = (float)(rng.NextDouble() * 2 - 1);
        return m;
    }

    private static bool Close(float a, float b, float tol)
        => MathF.Abs(a - b) <= tol * (1 + MathF.Abs(a) + MathF.Abs(b));

    [SkippableFact]
    public void ConcurrentGemm_NoCrash_MatchesCpu()
    {
        Skip.If(!_available, "GPU backend not available");

        const int threads = 12;   // mirrors the parallel research sweep degree
        const int iters = 60;      // sustained churn per thread to surface handle corruption
        const int m = 48, k = 64, n = 32;

        // Independent operands + precomputed CPU reference per thread (read-only across threads).
        var lhs = new Matrix<float>[threads];
        var rhs = new Matrix<float>[threads];
        var reference = new Matrix<float>[threads];
        for (int t = 0; t < threads; t++)
        {
            lhs[t] = Randn(1000 + t, m, k);
            rhs[t] = Randn(2000 + t, k, n);
            reference[t] = _cpu.MatrixMultiply(lhs[t], rhs[t]);
        }

        var failures = new ConcurrentQueue<string>();

        Parallel.For(0, threads, new ParallelOptions { MaxDegreeOfParallelism = threads }, t =>
        {
            try
            {
                for (int it = 0; it < iters; it++)
                {
                    var gpu = _gpu!.MatrixMultiply(lhs[t], rhs[t]);
                    for (int i = 0; i < m; i++)
                        for (int j = 0; j < n; j++)
                        {
                            if (!Close(gpu[i, j], reference[t][i, j], 1e-2f))
                            {
                                failures.Enqueue(
                                    $"thread {t} iter {it} [{i},{j}]: gpu={gpu[i, j]} cpu={reference[t][i, j]}");
                                return;
                            }
                        }
                }
            }
            catch (Exception ex)
            {
                failures.Enqueue($"thread {t} threw {ex.GetType().Name}: {ex.GetBaseException().Message}");
            }
        });

        Assert.True(failures.IsEmpty,
            $"{failures.Count} concurrent-GEMM failures (first few):\n" +
            string.Join("\n", failures.ToArray()[..Math.Min(5, failures.Count)]));
    }
}
