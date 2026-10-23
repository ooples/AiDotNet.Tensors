#if NETCOREAPP
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// The generated FP32 GEMM (#1262): every configuration in the search space either emits or is rejected with a
/// reason, and on a CUDA device each sampled configuration matches an fp64 oracle over layouts, edge shapes,
/// alpha/beta and batches, never reads C when beta is zero, and spills nothing to local memory.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DirectPtxSgemmGeneratorTests
{
    private const double Tolerance = 2e-5;

    private static readonly PtxGemmLayout[] Layouts =
    {
        new(false, false), new(true, false), new(false, true), new(true, true),
    };

    // Small and large tiles, both buffering modes, both vector widths, padded and not.
    private static readonly PtxGemmConfiguration[] Sampled =
    {
        new(32, 32, 8, 4, 4, 1, 0, false),
        new(64, 64, 16, 4, 4, 4, 4, true),
        new(128, 64, 16, 8, 4, 4, 0, true),
        new(128, 128, 8, 8, 8, 4, 4, true),
        new(64, 128, 16, 4, 8, 1, 4, false),
        new(256, 128, 16, 8, 8, 4, 0, true),
        // Split fragments and grouped rasterization, scalar and vector, on grids whose row count no group divides.
        new(128, 128, 8, 8, 8, 4, 4, true, true, 8),
        new(64, 128, 16, 4, 8, 1, 4, false, true, 4),
        new(128, 64, 16, 8, 4, 4, 0, true, false, 16),
        new(32, 32, 8, 8, 4, 1, 0, true, true, 4),
        // cp.async pipelines (SM 8.0+): two, three and four stages, over every layout's mix of 16-byte and 4-byte copies.
        new(64, 64, 16, 4, 4, 4, 4, false, false, 1, 1, 2),
        new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 1, 3),
        new(128, 64, 16, 8, 4, 4, 0, false, true, 4, 1, 4),
        // cp.async for the M/N-contiguous operand, register transposition for the K-contiguous one.
        new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 2, 2, true),
        new(64, 128, 16, 4, 8, 4, 0, false, true, 4, 1, 3, true),
        // Register-pipelined fragments: the default seed, a 16 x 8 thread tile over register transposition, and
        // 4-byte cp.async copies (the affine fast path for tiles inside K, the general path at the K edge).
        new(128, 128, 16, 8, 8, 4, 4, false, true, 8, 2, 2, true, FragmentPipeline: true),
        new(256, 128, 16, 16, 8, 4, 4, false, true, 1, 1, 3, true, FragmentPipeline: true),
        new(128, 256, 8, 8, 16, 4, 4, false, true, 8, 1, 4, false, FragmentPipeline: true),
    };

    // (M, N, K): degenerate, odd (no tile divides), exact tiles, tall / wide / deep, and vector-aligned odd.
    private static readonly (int M, int N, int K)[] Shapes =
    {
        (1, 1, 1), (3, 5, 7), (127, 255, 513), (64, 64, 64), (128, 256, 96), (1, 300, 200), (300, 1, 200),
        (40, 44, 1000), (200, 132, 36),
    };

    [Fact]
    public void SearchSpace_EveryConfigurationEmits()
    {
        var space = PtxGemmConfiguration.SearchSpace().ToList();
        Assert.True(space.Count > 200, $"the search space shrank to {space.Count} configurations");
        foreach (var config in space)
            foreach (var layout in Layouts)
            {
                string ptx = PtxSgemmGenerator.Emit(config, layout, 8, 6);
                Assert.Contains(".entry " + PtxSgemmGenerator.EntryPoint, ptx);
                Assert.Contains(".maxntid " + config.Threads + ", 1, 1", ptx);
            }
    }

    [Fact]
    public void SampledConfigurations_AreValid()
    {
        foreach (var config in Sampled) Assert.True(config.Invalidity() is null, config.Name + ": " + config.Invalidity());
    }

    [Fact]
    public void InvalidConfigurations_AreRejectedWithAReason()
    {
        Assert.NotNull(new PtxGemmConfiguration(64, 64, 8, 3, 4, 1, 0, false).Invalidity());   // TM not a multiple of 4
        Assert.NotNull(new PtxGemmConfiguration(64, 64, 8, 4, 4, 2, 0, false).Invalidity());   // vector width 2
        Assert.NotNull(new PtxGemmConfiguration(256, 256, 32, 8, 8, 4, 4, true).Invalidity()); // > 48 KiB shared
        Assert.NotNull(new PtxGemmConfiguration(64, 64, 8, 4, 4, 1, 0, false, true).Invalidity());       // 4x4 split
        Assert.NotNull(new PtxGemmConfiguration(64, 64, 8, 4, 8, 1, 0, false, false, 3).Invalidity());  // swizzle 3
        Assert.NotNull(new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 0, true, false, 1, 1, 2).Invalidity());  // async + db
        Assert.NotNull(new PtxGemmConfiguration(64, 64, 16, 4, 4, 1, 0, false, false, 1, 1, 2).Invalidity()); // async scalar
        Assert.Throws<PlatformNotSupportedException>(() => PtxSgemmGenerator.Emit(
            new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 0, false, false, 1, 1, 2), new PtxGemmLayout(false, false), 7, 5));
        Assert.Throws<ArgumentException>(() =>
            PtxSgemmGenerator.Emit(new PtxGemmConfiguration(64, 64, 8, 4, 4, 2, 0, false), new PtxGemmLayout(false, false), 8, 6));
    }

    [Fact]
    public void VectorConfigurations_AdmitOnlyAlignedProblems()
    {
        var v4 = new PtxGemmConfiguration(64, 64, 16, 4, 4, 4, 0, true);
        var nn = new PtxGemmLayout(false, false);
        Assert.True(PtxSgemmGenerator.Admits(v4, nn, 64, 64, 64, 64, 64, 64, 0, 0, 0));
        Assert.False(PtxSgemmGenerator.Admits(v4, nn, 64, 63, 64, 64, 63, 63, 0, 0, 0));  // N not a multiple of 4
        Assert.False(PtxSgemmGenerator.Admits(v4, nn, 64, 64, 66, 66, 64, 64, 0, 0, 0));  // K (A contiguous) not
        Assert.True(PtxSgemmGenerator.Admits(v4 with { VectorWidth = 1 }, nn, 7, 9, 11, 11, 9, 9, 0, 0, 0));
    }

    private static DirectPtxRuntime RequireRuntime()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        var runtime = new DirectPtxRuntime();
        if (runtime.ComputeCapabilityMajor < 7)
        {
            runtime.Dispose();
            Skip.If(true, "Needs compute capability 7.0 or newer.");
        }
        return runtime;
    }

    private static float[] Values(int count, int seed, float scale)
    {
        var random = new Random(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)((random.NextDouble() * 2 - 1) * scale);
        return values;
    }

    // C[b] = alpha * op(A[b]) @ op(B[b]) + beta * C[b] in double; op(A) is M x K, op(B) is K x N.
    private static double[] Oracle(float[] a, float[] b, float[] c, PtxGemmLayout layout, int batch, int m, int n, int k,
        float alpha, float beta)
    {
        int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n;
        var result = new double[batch * m * n];
        for (int z = 0; z < batch; z++)
            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        double av = layout.TransposeA ? a[z * m * k + p * lda + i] : a[z * m * k + i * lda + p];
                        double bv = layout.TransposeB ? b[z * k * n + j * ldb + p] : b[z * k * n + p * ldb + j];
                        sum += av * bv;
                    }
                    int index = z * m * n + i * n + j;
                    result[index] = alpha * sum + (beta == 0 ? 0 : beta * (double)c[index]);
                }
        return result;
    }

    private static void Check(DirectPtxRuntime runtime, PtxSgemmKernel kernel, int batch, int m, int n, int k,
        float alpha, float beta, bool poisonC)
    {
        var layout = kernel.Layout;
        int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n, ldc = n;
        float[] a = Values(Math.Max(1, batch * m * k), m * 31 + k, 1f);
        float[] b = Values(Math.Max(1, batch * k * n), n * 17 + k, 1f);
        float[] c = poisonC ? Enumerable.Repeat(float.NaN, batch * m * n).ToArray() : Values(batch * m * n, m + n, 1f);
        double[] expected = Oracle(a, b, c, layout, batch, m, n, k, alpha, beta);
        using var da = runtime.AllocateBytes((nuint)(a.Length * sizeof(float)));
        using var db = runtime.AllocateBytes((nuint)(b.Length * sizeof(float)));
        using var dc = runtime.AllocateBytes((nuint)(c.Length * sizeof(float)));
        da.Upload<float>(a);
        db.Upload<float>(b);
        dc.Upload<float>(c);
        kernel.Launch(da.Pointer, db.Pointer, dc.Pointer, m, n, k, lda, ldb, ldc, alpha, beta, batch,
            (long)m * k, (long)k * n, (long)m * n);
        runtime.Synchronize();
        var actual = new float[c.Length];
        dc.Download<float>(actual);
        // Relative to the terms each output sums: |alpha| * sum |a||b| plus |beta * c|, bounded by k + 1.
        double scale = Math.Abs(alpha) * Math.Max(1, k) + Math.Abs(beta) + 1e-30;
        string what = $"{kernel.Configuration.Name} {layout.Name} b{batch} M{m} N{n} K{k} alpha{alpha} beta{beta}";
        for (int i = 0; i < actual.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= Tolerance * scale,
                $"{what}: [{i}] got {actual[i]}, expected {expected[i]}");
    }

    [SkippableFact]
    public void SampledConfigurations_MatchTheOracle_OverLayoutsShapesAndScalars()
    {
        using var runtime = RequireRuntime();
        foreach (var config in Sampled)
            foreach (var layout in Layouts)
            {
                if (config.IsAsync && runtime.ComputeCapabilityMajor < 8) continue;
                using var kernel = new PtxSgemmKernel(runtime, config, layout);
                Assert.Equal(0, kernel.Audit.Function.LocalBytesPerThread);
                foreach (var (m, n, k) in Shapes)
                {
                    int lda = layout.TransposeA ? m : k, ldb = layout.TransposeB ? k : n;
                    if (!PtxSgemmGenerator.Admits(config, layout, m, n, k, lda, ldb, n, (long)m * k, (long)k * n, (long)m * n))
                        continue;
                    Check(runtime, kernel, 1, m, n, k, 1f, 0f, poisonC: false);
                    Check(runtime, kernel, 1, m, n, k, 0.5f, 2f, poisonC: false);
                }
            }
    }

    [Fact]
    public void EveryLayout_RunsSomeVectorAndSomeScalarShape()
    {
        // Guards the loop above: if admission silently rejected everything, it would pass without running.
        foreach (var layout in Layouts)
        {
            Assert.Contains(Shapes, s => PtxSgemmGenerator.Admits(Sampled[1], layout, s.M, s.N, s.K,
                layout.TransposeA ? s.M : s.K, layout.TransposeB ? s.K : s.N, s.N, 0, 0, 0));
            Assert.All(Shapes, s => Assert.True(PtxSgemmGenerator.Admits(Sampled[0], layout, s.M, s.N, s.K,
                layout.TransposeA ? s.M : s.K, layout.TransposeB ? s.K : s.N, s.N, 0, 0, 0)));
        }
    }

    [SkippableFact]
    public void Batched_MatchesTheOracle()
    {
        using var runtime = RequireRuntime();
        foreach (var config in new[] { Sampled[0], Sampled[2], Sampled[6], Sampled[9], Sampled[10], Sampled[11], Sampled[13] })
            foreach (var layout in Layouts)
            {
                if (config.IsAsync && runtime.ComputeCapabilityMajor < 8) continue;
                using var kernel = new PtxSgemmKernel(runtime, config, layout);
                Check(runtime, kernel, 3, 40, 36, 52, 1f, 0f, poisonC: false);
                Check(runtime, kernel, 3, 40, 36, 52, -1.5f, 0.25f, poisonC: false);
            }
    }

    [SkippableFact]
    public void BetaZero_NeverReadsC_AndKZeroScalesC()
    {
        using var runtime = RequireRuntime();
        foreach (var config in new[] { Sampled[0], Sampled[1], Sampled[11], Sampled[13] })
        {
            if (config.IsAsync && runtime.ComputeCapabilityMajor < 8) continue;
            using var kernel = new PtxSgemmKernel(runtime, config, new PtxGemmLayout(false, false));
            // C holds NaN: any read would poison the output.
            Check(runtime, kernel, 1, 64, 64, 64, 1f, 0f, poisonC: true);
            // K = 0: C = beta * C.
            Check(runtime, kernel, 1, 64, 64, 0, 1f, 3f, poisonC: false);
        }
    }
}
#endif
