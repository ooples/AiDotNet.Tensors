using System.Collections.Concurrent;
#pragma warning disable AIDEVO005 // AiDotNet.Evolution.Ptx is experimental; this campaign driver is its consumer.
using System.Text.Json;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Ptx;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// LLM program-evolution campaigns over the CUDA kernels where a vendor library still beats ours (ooples/AiDotNet#1159,
/// the "win first, then remove" plan for cuBLAS and cuDNN). A campaign takes our best kernel for its shape as the
/// incumbent, lets a model rewrite its PTX, and keeps a rewrite only when it reproduces a CPU reference on every
/// validation shape and wins the paired, alternating CUDA-event timing against the incumbent
/// (<see cref="PtxKernelFitnessEvaluator"/>). The winner is then timed against the vendor library.
/// </summary>
/// <remarks>
/// <para>Targets (<c>--ptx-campaign &lt;target&gt;</c>):</para>
/// <list type="bullet">
/// <item><c>measure [--shapes=MxNxK,...]</c>: no model calls. Times our tuned strict-FP32 GEMM against cuBLAS per shape,
/// eager and as graph replays, and the 3x3 convolution kernel against cuDNN, so a campaign is aimed where the vendor
/// still wins.</item>
/// <item><c>gemm-large [--m=2048 --n=2048 --k=2048]</c>: the strict-FP32 SGEMM against eager cuBLAS.</item>
/// <item><c>gemm-in-graph [--m --n --k]</c>: the same search, judged against cuBLAS as replays of a captured CUDA graph
/// (launch overhead gone, device time only), the mode compiled training steps run in.</item>
/// <item><c>conv3x3</c>: the exact-shape 3x3 convolution PTX kernel (batch 1, 64 to 64 channels, 16x16, stride 1, pad 1)
/// against cuDNN through the backend's vendor slot, eager and in-graph.</item>
/// </list>
/// <para>Common options: <c>[--generations=8] [--islands=2] [--model=claude-opus-5-5] [--max-usd=50]
/// [--max-usd-per-call=5] [--out=dir]</c>. <c>--max-usd</c> is the campaign's total budget
/// (<see cref="ClaudeCodeChatClientOptions.TotalBudgetUsd"/>: each call reserves its cap out of what remains, and is
/// charged what the CLI reports it cost); each call is capped at <c>--max-usd-per-call</c>. Once the budget is spent,
/// further proposals fail without starting the CLI and the campaign finishes its generations and verdict. Neither limit
/// is a hard cap: the CLI can run a turn past a call's cap, so spend can exceed <c>--max-usd</c> by about a turn per
/// call in flight. The model is reached through the Claude Code CLI (<c>claude -p</c>) on PATH, so the run uses that
/// login. Every proposal, verdict and the final comparison is written as JSON lines to stdout and to the output
/// directory.</para>
/// <para>The comparison is strict FP32 on both sides: the process runs in deterministic mode, so cuBLAS uses
/// <c>CUBLAS_PEDANTIC_MATH</c> (no TF32) like the SGEMM it is compared with.</para>
/// <para>Kernels win or lose per GPU: a campaign's verdict holds for the card it ran on. To confirm on another card,
/// run the same command there; the incumbent is re-tuned for that card first, so the comparison is always against its
/// own best kernel.</para>
/// </remarks>
internal static partial class DirectPtxDenseLinearFullExperiment
{
    internal static void RunPtxCampaign(string[] args)
    {
        string target = args.Length > 1 && !args[1].StartsWith("--", StringComparison.Ordinal) ? args[1] : "gemm-large";
        // Strict FP32 everywhere: cuBLAS in pedantic math, the same precision as the SGEMM and the CPU reference.
        AiDotNetEngine.SetDeterministicMode(true);
        using var session = new CampaignSession(args, target);
        switch (target)
        {
            case "measure":
                RunCampaignMeasurement(session);
                break;
            case "gemm-large":
            case "gemm-in-graph":
                RunGemmCampaign(session, inGraph: target == "gemm-in-graph");
                break;
            case "conv3x3":
                RunConv3x3Campaign(session);
                break;
            default:
                throw new ArgumentException(
                    $"Unknown PTX campaign '{target}'. Implemented: measure, gemm-large, gemm-in-graph, conv3x3.");
        }
    }

    /// <summary>Arguments, output directory and JSON-lines log shared by every target.</summary>
    private sealed class CampaignSession : IDisposable
    {
        private readonly string[] _args;
        private readonly StreamWriter _log;

        internal CampaignSession(string[] args, string target)
        {
            _args = args;
            Target = target;
            Output = Path.GetFullPath(Text("out", Path.Combine(Path.GetTempPath(), "aidotnet-ptx-campaign",
                target + "-" + DateTime.UtcNow.ToString("yyyyMMddHHmmss", System.Globalization.CultureInfo.InvariantCulture))));
            Directory.CreateDirectory(Output);
            _log = new StreamWriter(Path.Combine(Output, "campaign.jsonl"));
        }

        internal string Target { get; }
        internal string Output { get; }

        internal int Arg(string name, int fallback) =>
            _args.FirstOrDefault(a => a.StartsWith("--" + name + "=", StringComparison.Ordinal)) is { } value
                ? int.Parse(value[(name.Length + 3)..], System.Globalization.CultureInfo.InvariantCulture) : fallback;

        internal string Text(string name, string fallback) =>
            _args.FirstOrDefault(a => a.StartsWith("--" + name + "=", StringComparison.Ordinal)) is { } value
                ? value[(name.Length + 3)..] : fallback;

        // A dollar cap is fractional ("--max-usd=0.50"), so it is not parsed as an integer.
        internal decimal Usd(string name, string fallback) =>
            decimal.Parse(Text(name, fallback), System.Globalization.NumberStyles.Number,
                System.Globalization.CultureInfo.InvariantCulture);

        // stdout keeps the benchmarks' "name_json=" prefix that the other experiments' log scrapers key on; the file is
        // plain JSON Lines, one record per line, with the record kind as a field.
        internal void Emit(string kind, object payload)
        {
            string json = JsonSerializer.Serialize(payload);
            Console.WriteLine("ptx_campaign_" + kind + "_json=" + json);
            _log.WriteLine(JsonSerializer.Serialize(new { kind, record = payload }));
            _log.Flush();
        }

        public void Dispose() => _log.Dispose();
    }

    /// <summary>The model search's outcome: the best candidate (null when nothing was found) and what it cost.</summary>
    private sealed record CampaignSearch(EvolutionArchiveEntry<ProgramGenome>? Best, double? BestSpeedupOverIncumbent);

    /// <summary>
    /// Runs the model search from <paramref name="incumbent"/> and emits its "search" record: proposals, tokens, calls by
    /// outcome and dollars spent.
    /// </summary>
    private static CampaignSearch RunCampaignSearch(CampaignSession session, PtxProgramCompiler compiler,
        PtxCorrectnessEvaluator correctness, PtxKernelFitnessEvaluator fitness, string incumbent, string referenceIdentity)
    {
        int generations = session.Arg("generations", 8), islands = session.Arg("islands", 2);
        string model = session.Text("model", "claude-opus-5-5");
        decimal maxUsd = session.Usd("max-usd", "50");
        if (maxUsd <= 0) throw new ArgumentOutOfRangeException(nameof(session), "--max-usd must be positive.");
        decimal maxUsdPerCall = Math.Min(session.Usd("max-usd-per-call", "5"), maxUsd);
        if (maxUsdPerCall <= 0) throw new ArgumentOutOfRangeException(nameof(session), "--max-usd-per-call must be positive.");

        // The incumbent must pass its own contract, or the search has no parent and ends before its first proposal.
        PtxCorrectnessReport seed = correctness.Evaluate(incumbent);
        session.Emit("seed_check", new
        {
            target = session.Target, passed = seed.Passed, verdict = seed.Verdict.ToString(), message = seed.Message,
            max_abs_error = seed.MaxAbsoluteError, max_rel_error = seed.MaxRelativeError,
        });
        if (!seed.Passed) return new CampaignSearch(null, null);

        // Proposals: the model rewrites or patches the PTX; every proposal is JIT-checked before it is evaluated.
        var ptxOptions = new PtxProgramEvolutionOptions
        {
            Id = "ptx-campaign-" + session.Target,
            AuditDirectory = Path.Combine(session.Output, "proposals"),
            ModelVersionIdentity = model,
            AllowPatches = true,
            AllowRewrites = true,
            IncumbentProfile = null,
            // Every editable line is catalogued in the prompt, about 192 bytes with its source, and the parent is
            // abandoned before the first call when its lines exceed MaxInputTokens / 192. The 128x128 incumbents run to
            // about 3,250 lines, but the bound follows the incumbent actually tuned: a BlockK 32, 8x16 thread tile emits
            // 4,096 FMA lines alone. Twice its line count (at least 4,096) leaves room for a rewrite that grows it.
            MaxInputTokens = Math.Max(4_096, 2 * incumbent.Split('\n').Length) * 192,
            // The CLI reports the model's reasoning as output tokens: on the 2048^3 campaign three of four parsed
            // replies ran to 16.5k-21.5k tokens and were discarded at the 16,384 default after they had been paid for.
            // The dollar caps bound the spend; this bound only has to admit a full reply.
            MaxOutputTokens = session.Arg("max-output-tokens", 65_536),
        };
        var ledger = new EvolutionResourceLedger("ptx-campaign-" + session.Target,
            new EvolutionResources(new[] { new KeyValuePair<string, decimal>("cost_units", 1_000_000m) }));
        var resources = new ProgramEvolutionResourceOptions(ledger, 1_000m, ptxOptions.CostUnitVersionHash);
        // MaxBudgetUsd caps one claude -p process, and every call starts another, so the campaign's budget is the client's
        // TotalBudgetUsd across calls. Retries are off: the evolution loop's next request is the retry, charged like any other.
        // Neither is a hard cap: the client admits a call while any budget remains and clamps its cap to the remainder, and
        // the CLI can run one turn past a cap, so spend can exceed --max-usd by about a turn per call in flight.
        var claude = new ClaudeCodeChatClient(new ClaudeCodeChatClientOptions
        {
            Model = model,
            MaxBudgetUsd = maxUsdPerCall,
            TotalBudgetUsd = maxUsd,
            MaxRetries = 0,
            // A reply admitted at 65,536 output tokens runs past the 5-minute default: on the in-graph campaign 9 of 16
            // paid calls ended in "did not finish within 300 seconds".
            Timeout = TimeSpan.FromMinutes(session.Arg("call-timeout-min", 20)),
        });
        var client = new CountingChatClient(claude, maxUsd);
        var variation = PtxProgramVariation.Create(client, compiler, ptxOptions, resources);
        var correctnessGate = new DelegateProgramFitnessEvaluator((genome, _, token) =>
        {
            PtxCorrectnessReport report = correctness.Evaluate(genome.Source, token);
            return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                report.Passed ? EvolutionEvaluationStatus.Completed : EvolutionEvaluationStatus.Failed,
                quality: report.Passed ? 1 : 0, costUnits: 1));
        }, "ptx-" + session.Target + "-correctness", "ptx-" + session.Target + "-correctness-v1:" + referenceIdentity);

        var engine = ProgramEvolution.CreateEngine(variation, correctnessGate, fitness, resources,
            new EvolutionEngineOptions
            {
                RunId = "ptx-campaign-" + session.Target,
                MaxGenerations = generations,
                IslandCount = islands,
                Seed = 1159,
                EnableEvaluationCache = false,
                OutputDirectory = Path.Combine(session.Output, "engine"),
            },
            _ => new MapElitesArchive<ProgramGenome>(new[]
            {
                new EvolutionDescriptorDefinition("registers", 0, 256, 16, EvolutionOutOfRangePolicy.Clamp),
            }));
        EvolutionRunResult<ProgramGenome> run = engine.RunAsync(new[] { new ProgramGenome(incumbent) }).GetAwaiter().GetResult();
        EvolutionArchiveEntry<ProgramGenome>? best = run.Best;
        var usage = variation.GetUsage();
        session.Emit("search", new
        {
            target = session.Target, generations, islands, model,
            best_speedup_over_incumbent = best?.Evaluation.Quality,
            proposals = usage.Proposals, input_tokens = usage.InputTokens, output_tokens = usage.OutputTokens,
            // Attempts include requests refused for budget, which never started the CLI; failures are calls that did.
            call_attempts = client.Attempts, calls_refused_for_budget = client.RefusedForBudget,
            calls_failed = client.Failed,
            call_failures = client.FailureMessages,
            max_usd = maxUsd, max_usd_per_call = maxUsdPerCall,
            // Charged: each call's reported total_cost_usd, or its whole cap when the CLI reported none.
            usd_spent = claude.SpentUsd,
            // Why the run ended, and the failures it kept: a search that made no proposals says why here.
            stop_reason = run.StopReason.ToString(),
            counters = JsonSerializer.Serialize(run.Counters),
            island_statuses = JsonSerializer.Serialize(run.IslandStatuses),
            retained_failures = run.RetainedFailures.Take(10).Select(f => JsonSerializer.Serialize(f)).ToArray(),
        });
        return new CampaignSearch(best, best?.Evaluation.Quality);
    }

    /// <summary>
    /// The winner (when the search beat the incumbent) re-checked for correctness and timed against the incumbent under
    /// the promotion gate. Returns the winner's source, or null after emitting a "not promoted" verdict.
    /// </summary>
    private static string? CheckCampaignWinner(CampaignSession session, CampaignSearch search, PtxProgramCompiler compiler,
        PtxCorrectnessEvaluator correctness, PtxTimingEvaluator timing, string incumbent, out PtxTimingReport? measured)
    {
        measured = null;
        if (search.Best is null || (search.BestSpeedupOverIncumbent ?? 0) <= 1.0)
        {
            session.Emit("verdict", new { target = session.Target, promoted = false,
                reason = "No rewrite beat the incumbent." });
            return null;
        }
        string winner = search.Best.Candidate.CanonicalGenome.Genome.Source;
        File.WriteAllText(Path.Combine(session.Output, "winner.ptx"), winner);
        PtxCorrectnessReport check = correctness.Evaluate(winner);
        if (!check.Passed)
        {
            // A rejected kernel is never launched for timing: it may fault or hang, and the verdict must still be written.
            session.Emit("verdict", new { target = session.Target, promoted = false, correct = false,
                reason = check.Message, output = session.Output });
            return null;
        }
        measured = timing.Measure(winner, incumbent);
        if (measured.QualifiesForPromotion)
            PtxKernelArtifact.Create(compiler, check, measured, incumbent, timing.Identity)
                .Save(Path.Combine(session.Output, "artifact"));
        return winner;
    }

    // ---------------------------------------------------------------- GEMM

    private static void RunGemmCampaign(CampaignSession session, bool inGraph)
    {
        int m = session.Arg("m", 2048), n = session.Arg("n", 2048), k = session.Arg("k", 2048);
        using var backend = new CudaBackend();
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        var layout = new PtxGemmLayout(false, false);
        int sm = runtime.ComputeCapabilityMajor * 10 + runtime.ComputeCapabilityMinor;

        // 1. The incumbent: the best parameterized generated kernel for this shape on this card.
        var tuning = PtxGemmEvolution.TuneAsync(runtime, layout, m, n, k,
                engineOptions: EvolutionOptions("campaign-" + session.Target),
                deploymentRegistry: new KernelTuningDeploymentRegistry<PtxGemmConfiguration>(), store: new FreshTuningStore())
            .GetAwaiter().GetResult();
        PtxGemmConfiguration incumbentConfig = tuning.ProposedWinner.Configuration;
        string incumbent = PtxSgemmGenerator.Emit(incumbentConfig, layout, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        session.Emit("incumbent", new
        {
            target = session.Target, m, n, k, sm, configuration = incumbentConfig.Name,
            gflops = tuning.ProposedWinner.Measurement.ThroughputGflops,
        });

        // --vendor-check-only=1: the incumbent against cuBLAS and nothing else, so the measurement is checked without a
        // model call.
        if (session.Arg("vendor-check-only", 0) == 1)
        {
            session.Emit("vendor_check", new
            {
                target = session.Target, m, n, k, sm, incumbent_kernel = incumbentConfig.Name,
                incumbent_vs_cublas = MeasureGemmAgainstCublas(backend, runtime, incumbentConfig, incumbent, m, n, k),
            });
            return;
        }

        // 2. The contract: the incumbent's exact signature and launch geometry, so a rewrite is a drop-in for it.
        bool vector = incumbentConfig.VectorWidth > 1;
        var contract = SgemmContract(incumbentConfig, sm, m, n, k, vector);
        var compiler = new PtxProgramCompiler(contract);
        IPtxKernelReference reference = PtxKernelReference.Create("cpu-sgemm-row-major-fp64-accumulate", invocation =>
        {
            int mm = (int)invocation.Symbol("M"), nn = (int)invocation.Symbol("N"), kk = (int)invocation.Symbol("K");
            ReadOnlySpan<float> a = invocation.Input<float>("a_ptr");
            ReadOnlySpan<float> b = invocation.Input<float>("b_ptr");
            Span<float> c = invocation.Output<float>("c_ptr");
            for (int i = 0; i < mm; i++)
                for (int j = 0; j < nn; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < kk; p++) sum += (double)a[i * kk + p] * b[p * nn + j];
                    c[i * nn + j] = (float)sum;
                }
        });
        var correctness = new PtxCorrectnessEvaluator(compiler, reference, incumbent);
        var timing = new PtxTimingEvaluator(compiler);
        var fitness = new PtxKernelFitnessEvaluator(correctness, timing, incumbent);

        // 3. The search, then 4. the winner must still be correct and qualify over the incumbent.
        CampaignSearch search = RunCampaignSearch(session, compiler, correctness, fitness, incumbent, reference.Identity);
        string? winner = CheckCampaignWinner(session, search, compiler, correctness, timing, incumbent, out PtxTimingReport? measured);
        if (winner is null || measured is null) return;

        // 5. The campaign's own incumbent and winner, each timed against cuBLAS on the same buffers (strict FP32: the
        // process is in deterministic mode), eager and as graph replays. Production routing re-tunes on its own and can
        // pick another kernel (128x4096x4096: b128x64x32 split against the campaign's b128x64x16), so its verdict does
        // not describe these two kernels.
        var vsIncumbent = MeasureGemmAgainstCublas(backend, runtime, incumbentConfig, incumbent, m, n, k);
        var vsWinner = MeasureGemmAgainstCublas(backend, runtime, incumbentConfig, winner, m, n, k);
        session.Emit("verdict", new
        {
            target = session.Target, m, n, k, sm,
            correct = true,
            qualifies_over_incumbent = measured.QualifiesForPromotion,
            timing = measured.Message,
            incumbent_kernel = incumbentConfig.Name,
            vendor_mode = inGraph ? "graph-replay" : "eager",
            incumbent_vs_cublas = vsIncumbent,
            winner_vs_cublas = vsWinner,
            output = session.Output,
        });
    }

    /// <summary>
    /// <paramref name="ptx"/>, launched with <paramref name="config"/>'s geometry, against strict cuBLAS on the same
    /// buffers after checking both write the same C: paired rounds, eager and as graph replays.
    /// </summary>
    private static unsafe object MeasureGemmAgainstCublas(CudaBackend backend, DirectPtxRuntime runtime,
        PtxGemmConfiguration config, string ptx, int m, int n, int k)
    {
        var random = new Random(1265);
        float[] hostA = Values(random, checked(m * k));
        float[] hostB = Values(random, checked(k * n));
        using IGpuBuffer a = backend.AllocateBuffer(hostA);
        using IGpuBuffer b = backend.AllocateBuffer(hostB);
        using IGpuBuffer c = backend.AllocateBuffer(checked(m * n));
        using IGpuBuffer check = backend.AllocateBuffer(checked(m * n));
        using DirectPtxModule module = runtime.LoadModule(ptx);
        IntPtr function = module.GetFunction(PtxSgemmGenerator.EntryPoint);
        uint gridX = (uint)((n + config.BlockN - 1) / config.BlockN), gridY = (uint)((m + config.BlockM - 1) / config.BlockM);
        uint dynamicShared = config.IsAsync ? (uint)config.SharedBytes : 0;

        Action cublas = () => backend.CublasGemm(a, b, c, m, n, k, 1f, 0f);
        void Launch(IGpuBuffer output)
        {
            IntPtr pa = a.Handle, pb = b.Handle, pc = output.Handle;
            uint mu = (uint)m, nu = (uint)n, ku = (uint)k, lda = (uint)k, ldb = (uint)n, ldc = (uint)n;
            float alpha = 1f, beta = 0f;
            long sa = 0, sb = 0, sc = 0;
            void** arguments = stackalloc void*[14];
            arguments[0] = &pa; arguments[1] = &pb; arguments[2] = &pc;
            arguments[3] = &mu; arguments[4] = &nu; arguments[5] = &ku;
            arguments[6] = &lda; arguments[7] = &ldb; arguments[8] = &ldc;
            arguments[9] = &alpha; arguments[10] = &beta;
            arguments[11] = &sa; arguments[12] = &sb; arguments[13] = &sc;
            module.Launch(function, gridX, gridY, 1, (uint)config.Threads, 1, 1, dynamicShared, arguments);
        }
        Action generated = () => Launch(c);

        // The generated kernel writes a buffer of its own: into cuBLAS's, a kernel that wrote nothing would match.
        cublas();
        float[] expected = backend.DownloadBuffer(c);
        Launch(check);
        float[] actual = backend.DownloadBuffer(check);
        double largest = 1e-30, worst = 0;
        for (int j = 0; j < expected.Length; j++)
        {
            largest = Math.Max(largest, Math.Abs(expected[j]));
            double difference = Math.Abs((double)actual[j] - expected[j]);
            worst = Math.Max(worst, double.IsNaN(difference) ? double.PositiveInfinity : difference);
        }
        double relative = worst / largest;
        if (!(relative <= 1e-4))
            return new { comparable = false, max_relative_difference = relative };

        // About 50 ms of work per round arm at ~20 TFLOP/s.
        int reps = (int)Math.Clamp(1e12 / (2.0 * m * n * k), 10, 500);
        var (eager, graph) = backend.MeasureAgainstVendor(cublas, generated, reps);
        return new
        {
            comparable = true, max_relative_difference = relative, reps,
            eager_median_speedup = eager.MedianSpeedup, eager_minimum_speedup = eager.MinimumSpeedup,
            eager_cublas_us = eager.CublasMicroseconds, eager_ptx_us = eager.DirectPtxMicroseconds,
            graph_median_speedup = graph.MedianSpeedup, graph_minimum_speedup = graph.MinimumSpeedup,
            graph_cublas_us = graph.CublasMicroseconds, graph_ptx_us = graph.DirectPtxMicroseconds,
            graph_note = graph.Rounds == 0 ? graph.Reason : null,
        };
    }

    private static PtxKernelContract SgemmContract(PtxGemmConfiguration config, int sm, int m, int n, int k, bool vector)
    {
        // A 16-byte vector configuration admits only shapes whose contiguous extents are multiples of 4
        // (PtxSgemmGenerator.Admits), so its validation shapes stay on that lattice; a scalar one also gets ragged shapes.
        long[] lattice = vector ? new long[] { 4 } : Array.Empty<long>();
        var validation = new List<IReadOnlyDictionary<string, long>>
        {
            Shape(64, 64, 64), Shape(128, 256, 96), Shape(512, 512, 512), Shape(1000 / 4 * 4, 1000 / 4 * 4, 1000 / 4 * 4),
        };
        if (!vector) validation.Add(Shape(127, 255, 513));
        return new PtxKernelContract("sgemm-nn-f32", PtxSgemmGenerator.EntryPoint,
            "Row-major C[M,N] = alpha * A[M,K] @ B[K,N] + beta * C, alpha = 1, beta = 0 (C is never read), batch 1. " +
            "Keep the parameter list, entry point and launch geometry (grid (ceil(N/" + config.BlockN + "), ceil(M/" +
            config.BlockM + ")), " + config.Threads + " threads per block); every output element must be written.",
            PtxTargetLimits.ForSm(sm),
            new[]
            {
                PtxKernelParameter.Input("a_ptr", PtxElementType.Float32, PtxExtent.Of("M", "K")),
                PtxKernelParameter.Input("b_ptr", PtxElementType.Float32, PtxExtent.Of("K", "N")),
                PtxKernelParameter.Output("c_ptr", PtxElementType.Float32, PtxExtent.Of("M", "N")),
                PtxKernelParameter.Scalar("m", PtxElementType.UInt32, PtxExtent.Of("M")),
                PtxKernelParameter.Scalar("n", PtxElementType.UInt32, PtxExtent.Of("N")),
                PtxKernelParameter.Scalar("k", PtxElementType.UInt32, PtxExtent.Of("K")),
                PtxKernelParameter.Scalar("lda", PtxElementType.UInt32, PtxExtent.Of("K")),
                PtxKernelParameter.Scalar("ldb", PtxElementType.UInt32, PtxExtent.Of("N")),
                PtxKernelParameter.Scalar("ldc", PtxElementType.UInt32, PtxExtent.Of("N")),
                PtxKernelParameter.ScalarConstant("alpha", PtxElementType.Float32, 1.0),
                PtxKernelParameter.ScalarConstant("beta", PtxElementType.Float32, 0.0),
                PtxKernelParameter.ScalarConstant("stride_a", PtxElementType.Int64, 0),
                PtxKernelParameter.ScalarConstant("stride_b", PtxElementType.Int64, 0),
                PtxKernelParameter.ScalarConstant("stride_c", PtxElementType.Int64, 0),
            },
            new[]
            {
                new PtxShapeSymbol("M", 1, 8192, lattice), new PtxShapeSymbol("N", 1, 8192, lattice), new PtxShapeSymbol("K", 1, 8192, lattice),
            },
            // An async-copy configuration stages its tiles in dynamic shared memory, sized as PtxSgemmKernel launches it;
            // launched with none, its first tile store runs past the allocation (CUDA_ERROR_ILLEGAL_ADDRESS).
            new PtxLaunchConfiguration(PtxExtent.Of("N").CeilDiv(config.BlockN), PtxExtent.Of("M").CeilDiv(config.BlockM), null,
                config.Threads, 1, 1, config.IsAsync ? PtxExtent.Constant(config.SharedBytes) : null),
            // Fp32GridTolerance: a different summation order than the fp64-accumulated reference.
            new PtxTolerance(1e-3, Fp32GridTolerance), Shape(m, n, k), validation, fuzzCases: 0, seed: 1159);

        static IReadOnlyDictionary<string, long> Shape(long mm, long nn, long kk) =>
            new Dictionary<string, long> { ["M"] = mm, ["N"] = nn, ["K"] = kk };
    }

    // ---------------------------------------------------------------- conv 3x3

    private const int ConvCi = PtxConv2DNchw3x3ForwardF32Kernel.InputChannels;
    private const int ConvCo = PtxConv2DNchw3x3ForwardF32Kernel.OutputChannels;
    private const int ConvH = PtxConv2DNchw3x3ForwardF32Kernel.Height;
    private const int ConvW = PtxConv2DNchw3x3ForwardF32Kernel.Width;
    private const int ConvBlock = PtxConv2DNchw3x3ForwardF32Kernel.BlockThreads;

    private static void RunConv3x3Campaign(CampaignSession session)
    {
        using var backend = new CudaBackend();
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        int sm = runtime.ComputeCapabilityMajor * 10 + runtime.ComputeCapabilityMinor;
        string incumbent = PtxConv2DNchw3x3ForwardF32Kernel.EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        session.Emit("incumbent", new
        {
            target = session.Target, sm, kernel = PtxConv2DNchw3x3ForwardF32Kernel.EntryPoint,
            batch = 1, in_channels = ConvCi, out_channels = ConvCo, height = ConvH, width = ConvW,
        });

        var contract = Conv3x3Contract(sm);
        var compiler = new PtxProgramCompiler(contract);
        IPtxKernelReference reference = PtxKernelReference.Create("cpu-conv2d-nchw-3x3-s1-p1-fp64-accumulate", invocation =>
        {
            ReadOnlySpan<float> input = invocation.Input<float>("input_ptr");
            ReadOnlySpan<float> weights = invocation.Input<float>("weights_ptr");
            Span<float> output = invocation.Output<float>("output_ptr");
            Conv3x3Reference(input, weights, output);
        });
        var correctness = new PtxCorrectnessEvaluator(compiler, reference, incumbent);
        var timing = new PtxTimingEvaluator(compiler);
        var fitness = new PtxKernelFitnessEvaluator(correctness, timing, incumbent);

        CampaignSearch search = RunCampaignSearch(session, compiler, correctness, fitness, incumbent, reference.Identity);
        string? winner = CheckCampaignWinner(session, search, compiler, correctness, timing, incumbent, out PtxTimingReport? measured);
        if (winner is null || measured is null) return;

        var vsIncumbent = MeasureConv3x3AgainstCudnn(backend, runtime, incumbent);
        var vsWinner = MeasureConv3x3AgainstCudnn(backend, runtime, winner);
        session.Emit("verdict", new
        {
            target = session.Target, sm,
            correct = true,
            qualifies_over_incumbent = measured.QualifiesForPromotion,
            timing = measured.Message,
            incumbent_vs_cudnn = vsIncumbent,
            winner_vs_cudnn = vsWinner,
            output = session.Output,
        });
    }

    private static PtxKernelContract Conv3x3Contract(int sm)
    {
        var shape = new Dictionary<string, long> { ["CI"] = ConvCi, ["CO"] = ConvCo, ["H"] = ConvH, ["W"] = ConvW, ["R"] = 3, ["S"] = 3 };
        return new PtxKernelContract("conv2d-nchw-3x3-s1-p1-f32", PtxConv2DNchw3x3ForwardF32Kernel.EntryPoint,
            "NCHW fp32 convolution, batch 1, " + ConvCi + " input and " + ConvCo + " output channels, " + ConvH + "x" + ConvW +
            ", 3x3 kernel, stride 1, zero padding 1, no bias: output[co,y,x] = sum over ci, ky, kx of " +
            "input[ci, y+ky-1, x+kx-1] * weights[co,ci,ky,kx], out-of-range inputs read as 0. The shape is fixed. " +
            "Keep the parameter list, entry point and launch geometry (grid " + ConvCo * ConvH * ConvW / ConvBlock + " blocks of " +
            ConvBlock + " threads); every output element must be written.",
            PtxTargetLimits.ForSm(sm),
            new[]
            {
                PtxKernelParameter.Input("input_ptr", PtxElementType.Float32, PtxExtent.Of("CI", "H", "W")),
                PtxKernelParameter.Input("weights_ptr", PtxElementType.Float32, PtxExtent.Of("CO", "CI", "R", "S")),
                PtxKernelParameter.Output("output_ptr", PtxElementType.Float32, PtxExtent.Of("CO", "H", "W")),
            },
            new[]
            {
                new PtxShapeSymbol("CI", ConvCi, ConvCi, Array.Empty<long>()),
                new PtxShapeSymbol("CO", ConvCo, ConvCo, Array.Empty<long>()),
                new PtxShapeSymbol("H", ConvH, ConvH, Array.Empty<long>()),
                new PtxShapeSymbol("W", ConvW, ConvW, Array.Empty<long>()),
                new PtxShapeSymbol("R", 3, 3, Array.Empty<long>()),
                new PtxShapeSymbol("S", 3, 3, Array.Empty<long>()),
            },
            new PtxLaunchConfiguration(PtxExtent.Of("CO", "H", "W").CeilDiv(ConvBlock), null, null, ConvBlock, 1, 1, null),
            // 576 products per output summed in fp32 against an fp64-accumulated reference.
            new PtxTolerance(1e-3, Fp32GridTolerance), shape, new[] { shape }, fuzzCases: 0, seed: 1159);
    }

    private static void Conv3x3Reference(ReadOnlySpan<float> input, ReadOnlySpan<float> weights, Span<float> output)
    {
        for (int co = 0; co < ConvCo; co++)
            for (int y = 0; y < ConvH; y++)
                for (int x = 0; x < ConvW; x++)
                {
                    double sum = 0;
                    for (int ci = 0; ci < ConvCi; ci++)
                        for (int ky = 0; ky < 3; ky++)
                        {
                            int iy = y + ky - 1;
                            if ((uint)iy >= ConvH) continue;
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int ix = x + kx - 1;
                                if ((uint)ix >= ConvW) continue;
                                sum += (double)input[(ci * ConvH + iy) * ConvW + ix] * weights[((co * ConvCi + ci) * 3 + ky) * 3 + kx];
                            }
                        }
                    output[(co * ConvH + y) * ConvW + x] = (float)sum;
                }
    }

    /// <summary>
    /// <paramref name="ptx"/> against cuDNN (the backend's vendor slot: cuDNN's default algorithm, or a faster measured
    /// one) on the same buffers, after checking both write the same output: paired rounds, eager and as graph replays.
    /// </summary>
    private static unsafe object MeasureConv3x3AgainstCudnn(CudaBackend backend, DirectPtxRuntime runtime, string ptx)
    {
        var random = new Random(1159);
        float[] inputHost = Values(random, ConvCi * ConvH * ConvW);
        float[] weightHost = Values(random, ConvCo * ConvCi * 9, 0.0625f);
        using IGpuBuffer input = backend.AllocateBuffer(inputHost);
        using IGpuBuffer weights = backend.AllocateBuffer(weightHost);
        using IGpuBuffer output = backend.AllocateBuffer(ConvCo * ConvH * ConvW);
        using IGpuBuffer check = backend.AllocateBuffer(ConvCo * ConvH * ConvW);
        using DirectPtxModule module = runtime.LoadModule(ptx);
        IntPtr function = module.GetFunction(PtxConv2DNchw3x3ForwardF32Kernel.EntryPoint);

        Action cudnn = () => backend.Conv2D(input, weights, output, 1, ConvCi, ConvH, ConvW, ConvCo, ConvH, ConvW,
            3, 3, 1, 1, 1, 1, 1, 1);
        void Launch(IGpuBuffer destination)
        {
            IntPtr i = input.Handle, w = weights.Handle, o = destination.Handle;
            void** arguments = stackalloc void*[3];
            arguments[0] = &i;
            arguments[1] = &w;
            arguments[2] = &o;
            module.Launch(function, (uint)(ConvCo * ConvH * ConvW / ConvBlock), 1, 1, ConvBlock, 1, 1, 0, arguments);
        }
        Action generated = () => Launch(output);

        // The generated kernel writes a buffer of its own: into cuDNN's, a kernel that wrote nothing would match.
        cudnn();
        float[] expected = backend.DownloadBuffer(output);
        Launch(check);
        float[] actual = backend.DownloadBuffer(check);
        double largest = 1e-30, worst = 0;
        for (int j = 0; j < expected.Length; j++)
        {
            largest = Math.Max(largest, Math.Abs(expected[j]));
            double difference = Math.Abs((double)actual[j] - expected[j]);
            worst = Math.Max(worst, double.IsNaN(difference) ? double.PositiveInfinity : difference);
        }
        double relative = worst / largest;
        if (!(relative <= 1e-4))
            return new { comparable = false, max_relative_difference = relative };

        // About 50 ms of work per round arm at ~2.4 MFLOP per launch.
        var (eager, graph) = backend.MeasureAgainstVendor(cudnn, generated, reps: 200);
        return new
        {
            comparable = true, max_relative_difference = relative,
            eager_median_speedup = eager.MedianSpeedup, eager_minimum_speedup = eager.MinimumSpeedup,
            eager_cudnn_us = eager.CublasMicroseconds, eager_ptx_us = eager.DirectPtxMicroseconds,
            graph_median_speedup = graph.MedianSpeedup, graph_minimum_speedup = graph.MinimumSpeedup,
            graph_cudnn_us = graph.CublasMicroseconds, graph_ptx_us = graph.DirectPtxMicroseconds,
            graph_note = graph.Rounds == 0 ? graph.Reason : null,
        };
    }

    // ---------------------------------------------------------------- measure

    /// <summary>
    /// Where the vendor still wins, before any money is spent: per GEMM shape, production routing's strict-FP32 verdict
    /// (our evolved SGEMM against pedantic cuBLAS, eager and in-graph), and the 3x3 convolution kernel against cuDNN.
    /// </summary>
    private static void RunCampaignMeasurement(CampaignSession session)
    {
        string shapes = session.Text("shapes",
            "2048x2048x2048,4096x4096x4096,1024x1024x1024,2048x4096x4096,2048x11008x4096,2048x4096x11008,128x4096x4096,3136x64x576");
        using var backend = new CudaBackend();
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        int sm = runtime.ComputeCapabilityMajor * 10 + runtime.ComputeCapabilityMinor;
        foreach (string shape in shapes.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int[] mnk = shape.Split('x').Select(s => int.Parse(s, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
            CudaGemmRouteDecision d = backend.TuneGemmRouteAsync(mnk[0], mnk[1], mnk[2],
                    EvolutionOptions("campaign-measure-" + shape), persist: false)
                .GetAwaiter().GetResult();
            session.Emit("measure_gemm", new
            {
                sm, m = d.M, n = d.N, k = d.K, strict_fp32 = d.StrictFp32, kernel = d.Kernel,
                max_relative_difference = d.MaxRelativeDifference,
                eager_median_speedup = d.MedianSpeedup, eager_cublas_us = d.CublasMicroseconds, eager_ptx_us = d.DirectPtxMicroseconds,
                graph_median_speedup = d.GraphMedianSpeedup, graph_cublas_us = d.GraphCublasMicroseconds,
                graph_ptx_us = d.GraphDirectPtxMicroseconds,
                eager_reason = d.Reason, graph_reason = d.GraphReason,
            });
        }
        string conv = PtxConv2DNchw3x3ForwardF32Kernel.EmitPtx(runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        session.Emit("measure_conv3x3", new { sm, versus_cudnn = MeasureConv3x3AgainstCudnn(backend, runtime, conv) });
    }

    /// <summary>
    /// Counts the campaign's model requests by outcome. The variation's ChatCalls counts every request, including the
    /// ones the client refuses once the budget is spent, which never start the CLI. The client admits a call while any
    /// budget remains, so a request that fails with nothing left was refused; any other failure is the provider's.
    /// </summary>
    private sealed class CountingChatClient : IProgramChatClient
    {
        private readonly ClaudeCodeChatClient _inner;
        private readonly decimal _totalBudgetUsd;
        private int _attempts, _refused, _failed;
        private readonly ConcurrentQueue<string> _failures = new();

        public CountingChatClient(ClaudeCodeChatClient inner, decimal totalBudgetUsd)
        {
            _inner = inner;
            _totalBudgetUsd = totalBudgetUsd;
        }

        public string ModelId => _inner.ModelId;
        public int Attempts => Volatile.Read(ref _attempts);
        public int RefusedForBudget => Volatile.Read(ref _refused);
        public int Failed => Volatile.Read(ref _failed);
        /// <summary>Type and message of every call that threw, in order, so a campaign record says why a paid call failed.</summary>
        public string[] FailureMessages => _failures.ToArray();

        public async Task<ProgramChatResponse> GetResponseAsync(IReadOnlyList<ProgramChatMessage> messages,
            ProgramChatOptions options, CancellationToken cancellationToken)
        {
            Interlocked.Increment(ref _attempts);
            bool exhausted = _inner.SpentUsd >= _totalBudgetUsd;
            try
            {
                return await _inner.GetResponseAsync(messages, options, cancellationToken).ConfigureAwait(false);
            }
            catch (Exception ex) when (!cancellationToken.IsCancellationRequested)
            {
                Interlocked.Increment(ref exhausted ? ref _refused : ref _failed);
                string message = ex.GetType().Name + ": " + ex.Message;
                _failures.Enqueue(message.Length > 2_000 ? message.Substring(0, 2_000) : message);
                throw;
            }
        }
    }
}
