using System.Text.Json;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// #1261 (phase A of #377): where CUDA GEMM stands before any kernel work. Every CUDA GEMM runs on cuBLAS today
/// (both direct-PTX GEMM kernels report <c>IsPromotedShape =&gt; false</c>), so this times, per shape:
/// <list type="bullet">
/// <item>cuBLAS as production configures it: TF32 tensor-op math on Ampere+ outside deterministic mode;</item>
/// <item>cuBLAS in strict FP32 (<c>CUBLAS_PEDANTIC_MATH</c>, what deterministic mode selects): the fair bar for
/// an FP32 kernel, and the precision PyTorch's default <c>float32_matmul_precision("highest")</c> uses;</item>
/// <item>each direct-PTX GEMM kernel where its shape contract admits the shape, else an explicit unsupported row;</item>
/// <item>PyTorch (eager, CUDA graph, torch.compile) through the shared Python lane.</item>
/// </list>
/// Opt-in only (<c>--only=gemm-grid</c>): the large shapes make it far longer than the default matrix.
/// </summary>
internal static partial class DirectPtxDenseLinearFullExperiment
{
    internal const string GemmGridOperation = "gemm-grid";

    /// <summary>
    /// #1263: the same grid with the generated GEMM evolved per shape (AiDotNet.Evolution) in place of the fixed PTX
    /// kernels. Tuning runs once per shape (its trials and winner are logged as JSON) and is not timed; the winner is then
    /// measured like every other lane.
    /// </summary>
    internal const string GemmGridEvolvedOperation = "gemm-grid-evolved";

    /// <summary>
    /// #1265: the grid through production routing. Per shape and math mode, <see cref="CudaBackend.TuneGemmRouteAsync"/>
    /// evolves the generated kernel, checks it against cuBLAS's output and times it against cuBLAS in paired rounds
    /// (its verdict is logged as JSON, not persisted); then <see cref="CudaBackend.Gemm"/> itself is timed next to the
    /// cuBLAS lanes, routed or not as decided.
    /// </summary>
    internal const string GemmGridRoutedOperation = "gemm-grid-routed";

    private static readonly Dictionary<string, PtxGemmConfiguration> EvolvedWinners = new(StringComparer.Ordinal);

    private static AiDotNet.Evolution.EvolutionEngineOptions EvolutionOptions(string shape) => new()
    {
        RunId = "gemm-grid-" + shape,
        Seed = 1263,
        MaxEvaluationAttempts = 40,
        MaxProposals = 64,
        MaxGenerations = 24,
        ProposalBatchSize = 8,
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1,
    };

    // (name, M, K, N): square, decode / skinny, LLM projections, vision im2col, and odd sizes no tile divides.
    internal static readonly (string Name, int M, int K, int N)[] GemmGrid =
    {
        ("sq128", 128, 128, 128), ("sq256", 256, 256, 256), ("sq512", 512, 512, 512),
        ("sq1024", 1024, 1024, 1024), ("sq2048", 2048, 2048, 2048), ("sq4096", 4096, 4096, 4096),
        ("decode-m1-4096", 1, 4096, 4096), ("decode-m16-4096", 16, 4096, 4096),
        ("decode-m1-mlp-up", 1, 4096, 11008), ("decode-m16-mlp-up", 16, 4096, 11008),
        ("prefill-m128-qkv", 128, 4096, 4096), ("prefill-m2048-qkv", 2048, 4096, 4096),
        ("prefill-m2048-mlp-up", 2048, 4096, 11008), ("prefill-m2048-mlp-down", 2048, 11008, 4096),
        ("vision-56x56-c64", 3136, 576, 64), ("vision-28x28-c128", 784, 1152, 128),
        ("odd-1000", 1000, 1000, 1000), ("odd-127x513x255", 127, 513, 255),
    };

    // Checked output elements per lane: an fp64 oracle over the whole 4096^3 output would take minutes on the host.
    private const int GridErrorSamples = 512;
    private const double Fp32GridTolerance = 2e-4;
    private const double Tf32GridTolerance = 1e-2;

    private static void RunGemmGrid(CudaBackend backend, int run, bool evolve = false, bool routed = false)
    {
        // The cuBLAS lanes must time cuBLAS even where a persisted or tuned route exists.
        backend.GemmRoutingEnabled = false;
        // AIDOTNET_GEMM_GRID_SHAPES=name,name,... runs only those shapes: resumes a grid a thermal or idle-GPU gate
        // stopped part-way, without re-running the shapes it already measured.
        string? only = Environment.GetEnvironmentVariable("AIDOTNET_GEMM_GRID_SHAPES");
        var selected = string.IsNullOrWhiteSpace(only)
            ? null
            : new HashSet<string>(only!.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries), StringComparer.Ordinal);
        foreach (var (name, m, k, n) in GemmGrid)
        {
            if (selected is not null && !selected.Contains(name)) continue;
            var random = new Random(20261261 + run * 101 + m * 7 + n);
            float[] leftHost = Values(random, m * k);
            float[] rightHost = Values(random, k * n, 0.0625f);
            var samples = SampleOutputs(random, m, n);
            double[] oracle = SampledOracle(leftHost, rightHost, k, n, samples);
            double work = 2d * m * k * n;
            string shape = $"M{m} K{k} N{n}";
            string operation = GemmGridOperation + "-" + name;
            // Enough launches per device sample to span ~1 ms, so small shapes are not timer-bound and the
            // largest do not take minutes. The graph lane replays 1,000 operations per sample, which for large
            // shapes is tens of seconds of sustained load (this 3080 then exceeds the 75 C evidence ceiling) and
            // adds nothing: event timing over several launches already amortizes launch overhead there. So graphs
            // run only for shapes up to 0.2 GFLOP, where launch overhead is a real share of the time.
            int launches = (int)Math.Clamp(Math.Ceiling(2e10 / work), 1, DeviceLaunches);
            bool capture = work <= 2e8;
            double RelativeError(IGpuBuffer output) => SampledRelativeError(backend.DownloadBuffer(output), samples, oracle, n);

            using var left = backend.AllocateBuffer(leftHost);
            using var right = backend.AllocateBuffer(rightHost);
            using var output = backend.AllocateBuffer(m * n);

            DirectPtxFeatureGate.TestOverride = false;
            DirectPtxFeatureGate.FusedLinearExperimentOverride = false;
            AiDotNetEngine.SetDeterministicMode(false);
            GpuBenchmarkEnvironment.WaitForCooldown(operation);
            BenchmarkRoute(backend, run, operation, shape, "cuBLAS TF32 (prod default)", work,
                () => backend.Gemm(left, right, output, m, n, k), () => RelativeError(output),
                Resources.None, capture, tolerance: Tf32GridTolerance, deviceLaunches: launches);

            AiDotNetEngine.SetDeterministicMode(true);
            try
            {
                GpuBenchmarkEnvironment.WaitForCooldown(operation);
                BenchmarkRoute(backend, run, operation, shape, "cuBLAS FP32 strict", work,
                    () => backend.Gemm(left, right, output, m, n, k), () => RelativeError(output),
                    Resources.None, capture, tolerance: Fp32GridTolerance, deviceLaunches: launches);
            }
            finally
            {
                AiDotNetEngine.SetDeterministicMode(false);
            }

            if (evolve)
            {
                RunEvolvedLane(backend, run, name, operation, shape, m, n, k, work, launches, capture,
                    left, right, output, RelativeError);
                continue;
            }

            if (routed)
            {
                foreach (bool strict in new[] { true, false })
                    RunRoutedLane(backend, run, name, operation, shape, m, n, k, work, launches, capture, strict,
                        left, right, output, RelativeError);
                continue;
            }

            EnableDirectExperiment();
            if (PtxFusedLinearTiledKernel.IsSupportedShape(m, k, n) &&
                backend.PrewarmDirectPtxGemmTiled(m, k, n, DirectPtxLinearWeightLayout.InputMajor) &&
                backend.TryGetDirectPtxGemmTiledAudit(m, k, n, DirectPtxLinearWeightLayout.InputMajor, 1, out var tiledAudit))
            {
                GpuBenchmarkEnvironment.WaitForCooldown(operation);
                BenchmarkRoute(backend, run, operation, shape, "Direct PTX tiled 16x16x32", work, () =>
                    {
                        if (!backend.TryDirectPtxGemmTiled(left, right, output, m, k, n, DirectPtxLinearWeightLayout.InputMajor))
                            throw new InvalidOperationException(backend.DirectPtxLastError);
                    }, () => RelativeError(output), Resource(tiledAudit), capture,
                    tolerance: Fp32GridTolerance, deviceLaunches: launches);
            }
            else
            {
                PrintUnsupported(run, operation, shape, "Direct PTX tiled 16x16x32", backend.DirectPtxLastError);
            }

            if (PtxGemmKernel.IsSupportedShape(m, k, n) && backend.PrewarmDirectPtxGemm(m, k, n) &&
                backend.TryGetDirectPtxGemmAudit(m, k, n, out var gemmAudit))
            {
                GpuBenchmarkEnvironment.WaitForCooldown(operation);
                BenchmarkRoute(backend, run, operation, shape, "Direct PTX gemm 64x64x8", work, () =>
                    {
                        if (!backend.TryDirectPtxGemm(left, right, output, m, k, n))
                            throw new InvalidOperationException(backend.DirectPtxLastError);
                    }, () => RelativeError(output), Resource(gemmAudit), capture,
                    tolerance: Fp32GridTolerance, deviceLaunches: launches);
            }
            else
            {
                PrintUnsupported(run, operation, shape, "Direct PTX gemm 64x64x8", backend.DirectPtxLastError);
            }
        }
    }

    private static void RunEvolvedLane(CudaBackend backend, int run, string name, string operation, string shape,
        int m, int n, int k, double work, int launches, bool capture, IGpuBuffer left, IGpuBuffer right, IGpuBuffer output,
        Func<IGpuBuffer, double> relativeError)
    {
        var layout = new PtxGemmLayout(false, false);
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        if (!EvolvedWinners.TryGetValue(name, out PtxGemmConfiguration winner))
        {
            GpuBenchmarkEnvironment.WaitForCooldown(operation + "-tune");
            // A fresh store and registry: each benchmark tunes from scratch instead of hydrating a deployment persisted by an
            // earlier run (which then stands as the incumbent).
            var tuning = PtxGemmEvolution.TuneAsync(runtime, layout, m, n, k, engineOptions: EvolutionOptions(name),
                deploymentRegistry: new KernelTuningDeploymentRegistry<PtxGemmConfiguration>(),
                store: new FreshTuningStore(),
                onTrial: (configuration, trial) => Console.WriteLine("gemm_grid_evolution_trial_json=" +
                    JsonSerializer.Serialize(new
                    {
                        run,
                        operation,
                        configuration = configuration.Name,
                        status = trial.Status.ToString(),
                        gflops = trial.Measurement?.ThroughputGflops,
                        diagnostic = trial.Diagnostic,
                    }))).GetAwaiter().GetResult();
            // Time the search's best (the paired-replay candidate). Whether it would also pass the deployment gate,
            // whose P95 test is noise-bound at microsecond scale, is reported separately as "promoted".
            winner = tuning.ProposedWinner.Configuration;
            EvolvedWinners[name] = winner;
            Console.WriteLine("gemm_grid_evolution_winner_json=" + JsonSerializer.Serialize(new
            {
                operation,
                configuration = winner.Name,
                incumbent = tuning.IncumbentDeployment.Configuration.Name,
                incumbent_gflops = tuning.IncumbentDeployment.Measurement.ThroughputGflops,
                proposed = tuning.ProposedWinner.Configuration.Name,
                proposed_gflops = tuning.ProposedWinner.Measurement.ThroughputGflops,
                promoted = tuning.WasPromoted,
                evaluations = tuning.Run.Counters.EvaluationAttempts,
            }));
        }
        using var kernel = new PtxSgemmKernel(runtime, winner, layout);
        GpuBenchmarkEnvironment.WaitForCooldown(operation);
        BenchmarkRoute(backend, run, operation, shape, "Direct PTX evolved " + winner.Name, work,
            () => kernel.Launch(left.Handle, right.Handle, output.Handle, m, n, k, k, n, n, 1f, 0f),
            () => relativeError(output), Resource(kernel.Audit), capture,
            tolerance: Fp32GridTolerance, deviceLaunches: launches);
    }

    private static void RunRoutedLane(CudaBackend backend, int run, string name, string operation, string shape,
        int m, int n, int k, double work, int launches, bool capture, bool strict, IGpuBuffer left, IGpuBuffer right,
        IGpuBuffer output, Func<IGpuBuffer, double> relativeError)
    {
        AiDotNetEngine.SetDeterministicMode(strict);
        try
        {
            GpuBenchmarkEnvironment.WaitForCooldown(operation + "-route");
            CudaGemmRouteDecision decision = backend.TuneGemmRouteAsync(m, n, k,
                EvolutionOptions(name + (strict ? "-strict" : "-tf32")), persist: false).GetAwaiter().GetResult();
            Console.WriteLine("gemm_grid_route_json=" + JsonSerializer.Serialize(new
            {
                run,
                operation,
                mode = strict ? "strict-fp32" : "tf32",
                kernel = decision.Kernel,
                routed = decision.UsesDirectPtx,
                median_speedup = decision.MedianSpeedup,
                minimum_speedup = decision.MinimumSpeedup,
                cublas_us = decision.CublasMicroseconds,
                ptx_us = decision.DirectPtxMicroseconds,
                max_relative_difference = decision.MaxRelativeDifference,
                reason = decision.Reason,
                graph_routed = decision.UsesDirectPtxInGraph,
                graph_median_speedup = decision.GraphMedianSpeedup,
                graph_minimum_speedup = decision.GraphMinimumSpeedup,
                graph_cublas_us = decision.GraphCublasMicroseconds,
                graph_ptx_us = decision.GraphDirectPtxMicroseconds,
                graph_reason = decision.GraphReason,
            }));
            backend.GemmRoutingEnabled = true;
            GpuBenchmarkEnvironment.WaitForCooldown(operation);
            // Eager and graph launches route separately; the graph row (" graph" suffix) runs the graph verdict.
            string lane = (strict ? "Gemm routed FP32 strict (" : "Gemm routed TF32 (") +
                (decision.UsesDirectPtx ? "PTX " + decision.Kernel : "cuBLAS") + "; graph " +
                (decision.UsesDirectPtxInGraph ? "PTX" : "cuBLAS") + ")";
            BenchmarkRoute(backend, run, operation, shape, lane, work,
                () => backend.Gemm(left, right, output, m, n, k), () => relativeError(output), Resources.None, capture,
                tolerance: strict ? Fp32GridTolerance : Tf32GridTolerance, deviceLaunches: launches);
        }
        catch (Exception exception) when (exception is InvalidOperationException or ArgumentException)
        {
            // A shape no generated kernel admits (e.g. unaligned for the TF32 kernel) stays on cuBLAS: say so.
            PrintUnsupported(run, operation, shape, strict ? "Gemm routed FP32 strict" : "Gemm routed TF32", exception.Message);
        }
        finally
        {
            backend.GemmRoutingEnabled = false;
            backend.ClearGemmRoutes();
            AiDotNetEngine.SetDeterministicMode(false);
        }
    }

    // Loads nothing and keeps winners only in memory, so benchmark tuning neither reads nor writes the shared cache.
    private sealed class FreshTuningStore : IKernelTuningStore<PtxGemmConfiguration>
    {
        public bool TryLoad(KernelTuningIdentity identity, AiDotNet.Evolution.IEvolutionGenomeCodec<PtxGemmConfiguration> codec,
            out KernelTuningDeploymentSnapshot<PtxGemmConfiguration>? snapshot)
        {
            snapshot = null;
            return false;
        }

        public bool TryStore(KernelTuningDeploymentSnapshot<PtxGemmConfiguration> snapshot,
            AiDotNet.Evolution.IEvolutionGenomeCodec<PtxGemmConfiguration> codec) => true;
    }

    private static void PrintUnsupported(int run, string operation, string shape, string method, string? reason)
        => Console.WriteLine("dense_linear_status_json=" + JsonSerializer.Serialize(new
        {
            run,
            operation,
            shape,
            method,
            status = "unsupported",
            reason = string.IsNullOrEmpty(reason) ? "shape outside the kernel's contract" : reason,
        }));

    // Fixed corners plus random interior points, so edge tiles are always checked.
    private static (int Row, int Column)[] SampleOutputs(Random random, int m, int n)
    {
        var points = new List<(int, int)> { (0, 0), (0, n - 1), (m - 1, 0), (m - 1, n - 1) };
        while (points.Count < Math.Min(GridErrorSamples, m * n)) points.Add((random.Next(m), random.Next(n)));
        return points.ToArray();
    }

    private static double[] SampledOracle(float[] left, float[] right, int k, int n, (int Row, int Column)[] samples)
    {
        var expected = new double[samples.Length];
        for (int s = 0; s < samples.Length; s++)
        {
            var (row, column) = samples[s];
            double sum = 0;
            for (int inner = 0; inner < k; inner++) sum += left[row * k + inner] * (double)right[inner * n + column];
            expected[s] = sum;
        }
        return expected;
    }

    // Max absolute error over the samples, relative to the largest sampled magnitude.
    private static double SampledRelativeError(float[] actual, (int Row, int Column)[] samples, double[] expected, int n)
    {
        double scale = 1e-30, error = 0;
        for (int s = 0; s < samples.Length; s++)
        {
            scale = Math.Max(scale, Math.Abs(expected[s]));
            error = Math.Max(error, Math.Abs(actual[samples[s].Row * n + samples[s].Column] - expected[s]));
        }
        return error / scale;
    }
}
