#if NETCOREAPP
using System;
using System.Linq;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// #1263: the generated GEMM evolved through AiDotNet.Evolution and the tuning scaffold. On every host, the seeds admit
/// their problem; on a CUDA device, a small fixed-budget run compiles, validates and times candidates, deploys a winner
/// with measured resources, and that winner is correct against an fp64 oracle.
/// </summary>
[Collection("DirectGpuSerial")]
public sealed class DirectPtxGemmEvolutionTests
{
    private readonly ITestOutputHelper _output;

    public DirectPtxGemmEvolutionTests(ITestOutputHelper output) => _output = output;

    private static readonly PtxGemmLayout[] Layouts =
    {
        new(false, false), new(true, false), new(false, true), new(true, true),
    };

    [Fact]
    public void Seeds_AreValidAndAdmitTheirProblem()
    {
        foreach (var layout in Layouts)
            foreach (var (m, n, k) in new[] { (256, 256, 256), (1, 4096, 4096), (127, 255, 513), (3136, 64, 576) })
            {
                var seeds = PtxGemmEvolution.Seeds(layout, m, n, k);
                Assert.NotEmpty(seeds);
                Assert.All(seeds, s =>
                {
                    Assert.Null(s.Invalidity());
                    Assert.True(PtxGemmEvolution.Admits(s, layout, m, n, k), s.Name);
                });
            }
    }

    [SkippableFact]
    public async Task SmallBudgetRun_DeploysACorrectMeasuredWinner()
    {
        Skip.IfNot(DirectPtxRuntime.IsAvailable, "Direct PTX runtime is unavailable.");
        using var runtime = new DirectPtxRuntime();
        Skip.If(runtime.ComputeCapabilityMajor < 7, "Needs compute capability 7.0 or newer.");
        const int m = 192, n = 160, k = 96;
        var layout = new PtxGemmLayout(false, false);
        var options = new EvolutionEngineOptions
        {
            RunId = "ptx-gemm-evolution-test",
            Seed = 1263,
            MaxEvaluationAttempts = 8,
            MaxProposals = 16,
            MaxGenerations = 2,
            ProposalBatchSize = 4,
            MaxDegreeOfParallelism = 1,
            IslandCount = 1,
            MigrationInterval = 0,
            MigrantsPerIsland = 1,
        };

        EvolutionKernelTuningResult<PtxGemmConfiguration> result =
            await PtxGemmEvolution.TuneAsync(runtime, layout, m, n, k, engineOptions: options,
                onTrial: (configuration, trial) => _output.WriteLine(
                    $"{configuration.Name}: {trial.Status} {trial.Measurement?.ThroughputGflops:F1} GFLOP/s {trial.Diagnostic}"));

        Assert.True(result.Run.Counters.EvaluationAttempts >= 3, $"only {result.Run.Counters.EvaluationAttempts} evaluations");
        var deployed = result.ActiveDeployment;
        Assert.Equal(KernelTuningValidationScope.Output, deployed.Measurement.Correctness.Scope);
        Assert.True(deployed.Measurement.Timing.SampleCount >= 3);
        Assert.True(deployed.Measurement.ThroughputGflops > 0);
        Assert.Equal(KernelTuningResourceMetricStatus.Measured, deployed.Measurement.Resources.RegistersPerThreadMetric.Status);
        Assert.True(deployed.Measurement.Resources.RegistersPerThread > 0);
        Assert.Equal(KernelTuningResourceMetricStatus.Measured, deployed.Measurement.Resources.CompileTimeMetric.Status);
        PtxGemmConfiguration winner = deployed.Configuration;
        Assert.Null(winner.Invalidity());
        Assert.True(PtxGemmEvolution.Admits(winner, layout, m, n, k));

        // The deployed configuration, run independently of the scaffold, against an fp64 oracle.
        var random = new Random(7);
        float[] a = Enumerable.Range(0, m * k).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        float[] b = Enumerable.Range(0, k * n).Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
        using var kernel = new PtxSgemmKernel(runtime, winner, layout);
        using var da = runtime.AllocateBytes((nuint)(a.Length * sizeof(float)));
        using var db = runtime.AllocateBytes((nuint)(b.Length * sizeof(float)));
        using var dc = runtime.AllocateBytes((nuint)(m * n * sizeof(float)));
        da.Upload<float>(a);
        db.Upload<float>(b);
        kernel.Launch(da.Pointer, db.Pointer, dc.Pointer, m, n, k, k, n, n, 1f, 0f);
        runtime.Synchronize();
        var c = new float[m * n];
        dc.Download<float>(c);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int p = 0; p < k; p++) sum += (double)a[i * k + p] * b[p * n + j];
                Assert.True(Math.Abs(c[i * n + j] - sum) <= 2e-5 * k, $"[{i},{j}] got {c[i * n + j]}, expected {sum}");
            }
    }
}
#endif
