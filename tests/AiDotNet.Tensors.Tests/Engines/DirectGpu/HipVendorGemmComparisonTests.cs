#if NET6_0_OR_GREATER
using System;
using System.Diagnostics;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #377 F: measures hipBLAS SGEMM against the GEMM HipBackend runs without it (CLBlast's indirect kernel or the
/// native kernels), on the same buffers, so the decision to drop the vendor path rests on numbers from the device.
/// Needs a HIP device whose rocBLAS has Tensile kernels for it, so it runs only when asked.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class HipVendorGemmComparisonTests
{
    private const string ProofEnvironmentVariable = "AIDOTNET_RUN_HIP_VENDOR_GEMM_COMPARISON";
    private const string GemmImplEnvironmentVariable = "AIDOTNET_GPU_GEMM_IMPL";
    private readonly ITestOutputHelper _output;

    public HipVendorGemmComparisonTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    [Trait("Category", "HardwareProof")]
    public void HipBlas_VersusBuiltInGemm_Measured()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable(ProofEnvironmentVariable), "1", StringComparison.Ordinal),
            $"Set {ProofEnvironmentVariable}=1 to run the hipBLAS comparison.");

        string? previousImpl = Environment.GetEnvironmentVariable(GemmImplEnvironmentVariable);
        try
        {
            using var backend = new HipBackend();
            Skip.IfNot(backend.IsAvailable && backend.IsHipBlasAvailable,
                "A HIP device with a working hipBLAS/rocBLAS for its architecture is required.");

            int[][] shapes = ComparisonShapes;

            _output.WriteLine("M x N x K | hipBLAS ms | built-in ms | built-in / hipBLAS");
            foreach (var s in shapes)
            {
                int m = s[0], n = s[1], k = s[2];
                float[] a = Values(m * k, 11), b = Values(k * n, 23);
                float[] reference = CpuGemm(a, b, m, n, k);
                using var ga = backend.AllocateBuffer(a);
                using var gb = backend.AllocateBuffer(b);
                using var gc = backend.AllocateBuffer(new float[m * n]);

                Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, "hipblas");
                double vendorMs = Time(() => Assert.True(backend.TryExecuteHipBlasGemm(ga, gb, gc, m, n, k, 1f, 0f)));
                AssertClose(reference, backend.DownloadBuffer(gc), k, $"hipBLAS {m}x{n}x{k}");

                Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, "builtin");
                double builtInMs = Time(() => backend.Gemm(ga, gb, gc, m, n, k));
                AssertClose(reference, backend.DownloadBuffer(gc), k, $"built-in {m}x{n}x{k}");

                _output.WriteLine($"{m} x {n} x {k} | {vendorMs:F3} | {builtInMs:F3} | {builtInMs / vendorMs:F2}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, previousImpl);
        }
    }

    /// <summary>
    /// Evolves the built-in CLBlast GEMM for each shape (AIDOTNET_HIP_VENDOR_TUNE_ATTEMPTS evaluations, default 64; shapes
    /// filtered by AIDOTNET_HIP_VENDOR_TUNE_SHAPES, e.g. "1024x1024x1024;2048x2048x2048"), then times the deployed kernel
    /// against hipBLAS: the per-shape ceiling of the built-in route. 128-wide tiles are seeded, since a short random
    /// search never reaches them from the 64-wide default.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "HardwareProof")]
    public async System.Threading.Tasks.Task TunedBuiltInGemm_VersusHipBlas_Measured()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable(ProofEnvironmentVariable), "1", StringComparison.Ordinal),
            $"Set {ProofEnvironmentVariable}=1 to run the hipBLAS comparison.");
        int attempts = int.TryParse(Environment.GetEnvironmentVariable("AIDOTNET_HIP_VENDOR_TUNE_ATTEMPTS"), out int parsed) && parsed >= 3
            ? parsed : 64;
        string? filter = Environment.GetEnvironmentVariable("AIDOTNET_HIP_VENDOR_TUNE_SHAPES");

        string? previousImpl = Environment.GetEnvironmentVariable(GemmImplEnvironmentVariable);
        try
        {
            using var backend = new HipBackend();
            Skip.IfNot(backend.IsAvailable && backend.IsHipBlasAvailable && backend.EnsureClBlastIndirectGemm(),
                "A HIP device with a working hipBLAS/rocBLAS and the CLBlast indirect GEMM is required.");
            var baseline = backend.ClBlastDefaultConfig ?? throw new InvalidOperationException("No CLBlast default configuration.");
            var seeds = new System.Collections.Generic.List<AiDotNet.Tensors.Engines.DirectGpu.OpenCL.GemmConfig>
            {
                baseline with { TileM = 128, TileN = 128, TileK = 16, ThreadTileM = 16, ThreadTileN = 16, VectorWidthM = 4, VectorWidthN = 4, MdimaSize = 16, NdimbSize = 16 },
                baseline with { TileM = 128, TileN = 64, TileK = 16, ThreadTileM = 16, ThreadTileN = 8, VectorWidthM = 4, VectorWidthN = 4, MdimaSize = 16, NdimbSize = 8 },
                baseline with { TileM = 64, TileN = 128, TileK = 16, ThreadTileM = 8, ThreadTileN = 16, VectorWidthM = 4, VectorWidthN = 4, MdimaSize = 8, NdimbSize = 16 },
            };

            _output.WriteLine("M x N x K | hipBLAS ms | tuned ms | tuned / hipBLAS | config");
            foreach (var s in ComparisonShapes)
            {
                int m = s[0], n = s[1], k = s[2];
                if (!string.IsNullOrEmpty(filter) && Array.IndexOf(filter.Split(';'), $"{m}x{n}x{k}") < 0) continue;
                var options = new AiDotNet.Evolution.EvolutionEngineOptions
                {
                    RunId = $"hip-vendor-compare-{m}x{n}x{k}",
                    Seed = 377,
                    MaxEvaluationAttempts = attempts,
                    MaxProposals = attempts * 2,
                    MaxGenerations = Math.Max(3, attempts / 4),
                    ProposalBatchSize = 4,
                    MaxDegreeOfParallelism = 1,
                    IslandCount = 1,
                    MigrationInterval = 0,
                    MigrantsPerIsland = 1
                };
                var tuning = await backend.TuneGemmWithEvolutionAsync(m, n, k, additionalSeeds: seeds, engineOptions: options);

                float[] a = Values(m * k, 11), b = Values(k * n, 23);
                float[] reference = CpuGemm(a, b, m, n, k);
                using var ga = backend.AllocateBuffer(a);
                using var gb = backend.AllocateBuffer(b);
                using var gc = backend.AllocateBuffer(new float[m * n]);

                Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, "hipblas");
                double vendorMs = Time(() => Assert.True(backend.TryExecuteHipBlasGemm(ga, gb, gc, m, n, k, 1f, 0f)));
                Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, "builtin");
                double tunedMs = Time(() => backend.Gemm(ga, gb, gc, m, n, k));
                AssertClose(reference, backend.DownloadBuffer(gc), k, $"tuned {m}x{n}x{k}");

                _output.WriteLine($"{m} x {n} x {k} | {vendorMs:F3} | {tunedMs:F3} | {tunedMs / vendorMs:F2} | {tuning.ActiveDeployment.Configuration}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(GemmImplEnvironmentVariable, previousImpl);
        }
    }

    private static readonly int[][] ComparisonShapes =
    {
        new[] { 128, 128, 128 }, new[] { 256, 256, 256 }, new[] { 512, 512, 512 }, new[] { 1024, 1024, 1024 },
        new[] { 2048, 2048, 2048 }, new[] { 64, 4096, 1024 }, new[] { 4096, 64, 1024 }, new[] { 256, 256, 2304 },
        new[] { 1024, 4096, 1024 }, new[] { 1024, 1024, 4096 }, new[] { 200, 500, 300 }, new[] { 32, 1024, 1024 }
    };

    /// <summary>Times every row-major SGEMM tile (#377) against hipBLAS on the comparison shapes.</summary>
    [SkippableFact]
    [Trait("Category", "HardwareProof")]
    public void RowMajorSgemm_VersusHipBlas_Measured()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable(ProofEnvironmentVariable), "1", StringComparison.Ordinal),
            $"Set {ProofEnvironmentVariable}=1 to run the hipBLAS comparison.");
        using var backend = new HipBackend();
        Skip.IfNot(backend.IsAvailable && backend.IsHipBlasAvailable && backend.EnsureRowMajorGemm(),
            "A HIP device with a working hipBLAS/rocBLAS is required.");
        _output.WriteLine("M x N x K | hipBLAS ms | best row-major ms | ratio | best tile");
        foreach (var s in ComparisonShapes)
        {
            int m = s[0], n = s[1], k = s[2];
            float[] a = Values(m * k, 11), b = Values(k * n, 23);
            float[] reference = CpuGemm(a, b, m, n, k);
            using var ga = backend.AllocateBuffer(a);
            using var gb = backend.AllocateBuffer(b);
            using var gc = backend.AllocateBuffer(new float[m * n]);
            double vendorMs = Time(() => Assert.True(backend.TryExecuteHipBlasGemm(ga, gb, gc, m, n, k, 1f, 0f)));
            double best = double.MaxValue; string bestName = string.Empty;
            foreach (var tile in AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels.HipRowMajorSgemmKernels.Tiles)
            {
                if (!tile.Supports(m, n, k)) continue;
                foreach (int split in new[] { 1, 2, 4, 8 })
                {
                    if (split > 1 && k < 64 * split) continue;
                    double ms = Time(() => Assert.True(backend.TryExecuteRowMajorGemm(tile, ga, gb, gc, m, n, k, 1f, 0f, split)));
                    AssertClose(reference, backend.DownloadBuffer(gc), k, $"{tile.KernelName} split {split} {m}x{n}x{k}");
                    if (ms < best) { best = ms; bestName = $"{tile.KernelName} split {split}"; }
                }
            }
            _output.WriteLine($"{m} x {n} x {k} | {vendorMs:F3} | {best:F3} | {best / vendorMs:F2} | {bestName}");
        }
    }

    private static double Time(Action gemm)
    {
        for (int i = 0; i < 3; i++) gemm();
        var samples = new double[15];
        for (int i = 0; i < samples.Length; i++)
        {
            long start = Stopwatch.GetTimestamp();
            gemm();
            samples[i] = (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
        }
        Array.Sort(samples);
        return samples[samples.Length / 2];
    }

    private static float[] Values(int count, int seed)
    {
        var random = RandomHelper.CreateSeededRandom(seed);
        var values = new float[count];
        for (int i = 0; i < count; i++) values[i] = (float)(random.NextDouble() * 2.0 - 1.0);
        return values;
    }

    private static float[] CpuGemm(float[] a, float[] b, int m, int n, int k)
    {
        var c = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int p = 0; p < k; p++)
            {
                double av = a[i * k + p];
                for (int j = 0; j < n; j++) c[i * n + j] += (float)(av * b[p * n + j]);
            }
        return c;
    }

    private static void AssertClose(float[] expected, float[] actual, int k, string label)
    {
        // Values are uniform in [-1, 1], so a K-term dot product has magnitude ~sqrt(K/3); fp32 accumulation error
        // grows ~sqrt(K) * eps relative to that.
        double tolerance = 1e-4 * Math.Sqrt(k) * Math.Max(1.0, Math.Sqrt(k / 3.0));
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{label}: index {i} expected {expected[i]} got {actual[i]} (tolerance {tolerance:E2})");
    }
}
#endif
