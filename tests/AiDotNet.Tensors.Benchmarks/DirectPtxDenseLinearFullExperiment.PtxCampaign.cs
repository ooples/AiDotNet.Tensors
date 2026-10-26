#pragma warning disable AIDEVO005 // AiDotNet.Evolution.Ptx is experimental; this campaign driver is its consumer.
using System.Text.Json;
using AiDotNet.Evolution;
using AiDotNet.Evolution.Programs;
using AiDotNet.Evolution.Ptx;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Benchmarks;

/// <summary>
/// LLM program-evolution campaigns over the CUDA kernels where a vendor library still beats ours (ooples/AiDotNet#1159,
/// the "win first, then remove" plan for cuBLAS and cuDNN). A campaign takes the best parameterized generated kernel
/// for its shape as the incumbent, lets a model rewrite its PTX, and keeps a rewrite only when it reproduces a CPU
/// reference on every validation shape and wins the paired, alternating CUDA-event timing against the incumbent
/// (<see cref="PtxKernelFitnessEvaluator"/>). The winner is then timed against cuBLAS the way production routing is.
/// </summary>
/// <remarks>
/// <para>Run: <c>--ptx-campaign gemm-large [--m=2048 --n=2048 --k=2048] [--generations=8] [--islands=2]
/// [--model=claude-opus-5-5] [--max-usd=50] [--max-usd-per-call=5] [--out=dir]</c>. <c>--max-usd</c> caps the whole
/// campaign (<see cref="CampaignUsdBudgetChatClient"/>); each call is capped at <c>--max-usd-per-call</c> and charged
/// that cap. The model is reached through the Claude Code CLI
/// (<c>claude -p</c>) on PATH, so the run uses that login. Every proposal, verdict and the final comparison is written
/// as JSON lines to stdout and to the output directory.</para>
/// <para>Kernels win or lose per GPU: a campaign's verdict holds for the card it ran on. To confirm on another card
/// (the RTX 3080 the vendor wins were measured on), run the same command there; the incumbent is re-tuned for that
/// card first, so the comparison is always against its own best parameterized kernel.</para>
/// </remarks>
internal static partial class DirectPtxDenseLinearFullExperiment
{
    internal static void RunPtxCampaign(string[] args)
    {
        string target = args.Length > 1 ? args[1] : "gemm-large";
        if (target != "gemm-large")
            throw new ArgumentException($"Unknown PTX campaign '{target}'. Implemented: gemm-large.");
        int Arg(string name, int fallback) =>
            args.FirstOrDefault(a => a.StartsWith("--" + name + "=", StringComparison.Ordinal)) is { } value
                ? int.Parse(value[(name.Length + 3)..], System.Globalization.CultureInfo.InvariantCulture) : fallback;
        string Text(string name, string fallback) =>
            args.FirstOrDefault(a => a.StartsWith("--" + name + "=", StringComparison.Ordinal)) is { } value
                ? value[(name.Length + 3)..] : fallback;

        int m = Arg("m", 2048), n = Arg("n", 2048), k = Arg("k", 2048);
        int generations = Arg("generations", 8), islands = Arg("islands", 2);
        string model = Text("model", "claude-opus-5-5");
        // A dollar cap is fractional ("--max-usd=0.50"), so it is not parsed as an integer.
        decimal maxUsd = decimal.Parse(Text("max-usd", "50"), System.Globalization.NumberStyles.Number,
            System.Globalization.CultureInfo.InvariantCulture);
        if (maxUsd <= 0) throw new ArgumentOutOfRangeException(nameof(args), "--max-usd must be positive.");
        decimal maxUsdPerCall = Math.Min(decimal.Parse(Text("max-usd-per-call", "5"), System.Globalization.NumberStyles.Number,
            System.Globalization.CultureInfo.InvariantCulture), maxUsd);
        if (maxUsdPerCall <= 0) throw new ArgumentOutOfRangeException(nameof(args), "--max-usd-per-call must be positive.");
        string output = Path.GetFullPath(Text("out", Path.Combine(Path.GetTempPath(), "aidotnet-ptx-campaign",
            target + "-" + DateTime.UtcNow.ToString("yyyyMMddHHmmss", System.Globalization.CultureInfo.InvariantCulture))));
        Directory.CreateDirectory(output);
        using var log = new StreamWriter(Path.Combine(output, "campaign.jsonl"));
        // stdout keeps the benchmarks' "name_json=" prefix that the other experiments' log scrapers key on; the file is
        // plain JSON Lines, one record per line, with the record kind as a field.
        void Emit(string kind, object payload)
        {
            string json = JsonSerializer.Serialize(payload);
            Console.WriteLine("ptx_campaign_" + kind + "_json=" + json);
            log.WriteLine(JsonSerializer.Serialize(new { kind, record = payload }));
            log.Flush();
        }

        using var backend = new CudaBackend();
        DirectPtxRuntime runtime = backend.GetDirectPtxRuntime();
        var layout = new PtxGemmLayout(false, false);
        int sm = runtime.ComputeCapabilityMajor * 10 + runtime.ComputeCapabilityMinor;

        // 1. The incumbent: the best parameterized generated kernel for this shape on this card.
        var tuning = PtxGemmEvolution.TuneAsync(runtime, layout, m, n, k, engineOptions: EvolutionOptions("campaign-" + target),
            deploymentRegistry: new KernelTuningDeploymentRegistry<PtxGemmConfiguration>(), store: new FreshTuningStore())
            .GetAwaiter().GetResult();
        PtxGemmConfiguration incumbentConfig = tuning.ProposedWinner.Configuration;
        string incumbent = PtxSgemmGenerator.Emit(incumbentConfig, layout, runtime.ComputeCapabilityMajor, runtime.ComputeCapabilityMinor);
        Emit("incumbent", new
        {
            target, m, n, k, sm, configuration = incumbentConfig.Name,
            gflops = tuning.ProposedWinner.Measurement.ThroughputGflops,
        });

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

        // 3. Proposals: the model rewrites or patches the PTX; every proposal is JIT-checked before it is evaluated.
        var ptxOptions = new PtxProgramEvolutionOptions
        {
            Id = "ptx-campaign-" + target,
            AuditDirectory = Path.Combine(output, "proposals"),
            ModelVersionIdentity = model,
            AllowPatches = true,
            AllowRewrites = true,
            IncumbentProfile = null,
        };
        var ledger = new EvolutionResourceLedger("ptx-campaign-" + target,
            new EvolutionResources(new[] { new KeyValuePair<string, decimal>("cost_units", 1_000_000m) }));
        var resources = new ProgramEvolutionResourceOptions(ledger, 1_000m, ptxOptions.CostUnitVersionHash);
        // MaxBudgetUsd caps one claude -p process, and every call and retry starts another, so the campaign's budget is
        // held across calls by the wrapper instead.
        var client = new CampaignUsdBudgetChatClient(model, maxUsd, maxUsdPerCall);
        var variation = PtxProgramVariation.Create(client, compiler, ptxOptions, resources);
        var correctnessGate = new DelegateProgramFitnessEvaluator((genome, _, token) =>
        {
            PtxCorrectnessReport report = correctness.Evaluate(genome.Source, token);
            return new ValueTask<EvolutionTaskResult>(new EvolutionTaskResult(
                report.Passed ? EvolutionEvaluationStatus.Completed : EvolutionEvaluationStatus.Failed,
                quality: report.Passed ? 1 : 0, costUnits: 1));
        }, "ptx-sgemm-correctness", "ptx-sgemm-correctness-v1:" + reference.Identity);

        var engine = ProgramEvolution.CreateEngine(variation, correctnessGate, fitness, resources,
            new EvolutionEngineOptions
            {
                RunId = "ptx-campaign-" + target,
                MaxGenerations = generations,
                IslandCount = islands,
                Seed = 1159,
                EnableEvaluationCache = false,
                OutputDirectory = Path.Combine(output, "engine"),
            },
            _ => new MapElitesArchive<ProgramGenome>(new[]
            {
                new EvolutionDescriptorDefinition("registers", 0, 256, 16, EvolutionOutOfRangePolicy.Clamp),
            }));
        EvolutionRunResult<ProgramGenome> run = engine.RunAsync(new[] { new ProgramGenome(incumbent) }).GetAwaiter().GetResult();
        EvolutionArchiveEntry<ProgramGenome>? best = run.Best;
        var usage = variation.GetUsage();
        Emit("search", new
        {
            target, generations, islands, model,
            best_speedup_over_incumbent = best?.Evaluation.Quality,
            proposals = usage.Proposals, calls = usage.ChatCalls, input_tokens = usage.InputTokens, output_tokens = usage.OutputTokens,
            max_usd = maxUsd, max_usd_per_call = maxUsdPerCall,
            // An upper bound: every call is charged its full cap, since claude -p reports no spend back.
            usd_committed = client.CommittedUsd, calls_refused_for_budget = client.RefusedCalls,
        });
        if (best is null || (best.Evaluation.Quality ?? 0) <= 1.0)
        {
            Emit("verdict", new { target, promoted = false, reason = "No rewrite beat the parameterized incumbent." });
            return;
        }

        // 4. The winner must still be correct, qualify under the promotion gate, and be timed against cuBLAS.
        string winner = best.Candidate.CanonicalGenome.Genome.Source;
        File.WriteAllText(Path.Combine(output, "winner.ptx"), winner);
        PtxCorrectnessReport check = correctness.Evaluate(winner);
        if (!check.Passed)
        {
            // A rejected kernel is never launched for timing: it may fault or hang, and the verdict must still be written.
            Emit("verdict", new { target, m, n, k, sm, promoted = false, correct = false, reason = check.Message, output });
            return;
        }
        PtxTimingReport measured = timing.Measure(winner, incumbent);
        if (measured.QualifiesForPromotion)
            PtxKernelArtifact.Create(compiler, check, measured, incumbent, timing.Identity).Save(Path.Combine(output, "artifact"));
        CudaGemmRouteDecision vendor = backend.TuneGemmRouteAsync(m, n, k, EvolutionOptions("campaign-cublas-" + target), persist: false)
            .GetAwaiter().GetResult();
        // Production routing re-tunes on its own, so its paired cuBLAS measurement describes the campaign's incumbent
        // only when that tune chose the same kernel. Otherwise the comparison is reported as unavailable, never as a
        // product of two measurements of different kernels.
        bool sameKernel = string.Equals(vendor.Kernel, incumbentConfig.Name, StringComparison.Ordinal);
        Emit("verdict", new
        {
            target, m, n, k, sm,
            correct = true,
            qualifies_over_incumbent = measured.QualifiesForPromotion,
            timing = measured.Message,
            route_kernel = vendor.Kernel,
            incumbent_kernel = incumbentConfig.Name,
            incumbent_vs_cublas_median = sameKernel ? vendor.MedianSpeedup : (double?)null,
            cublas_us = vendor.CublasMicroseconds,
            incumbent_us = sameKernel ? vendor.DirectPtxMicroseconds : (double?)null,
            estimated_rewrite_vs_cublas = sameKernel ? vendor.MedianSpeedup * (best.Evaluation.Quality ?? 0) : (double?)null,
            output,
        });
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
            new PtxLaunchConfiguration(PtxExtent.Of("N").CeilDiv(config.BlockN), PtxExtent.Of("M").CeilDiv(config.BlockM), null,
                config.Threads, 1, 1, null),
            // Fp32GridTolerance: a different summation order than the fp64-accumulated reference.
            new PtxTolerance(1e-3, Fp32GridTolerance), Shape(m, n, k), validation, fuzzCases: 0, seed: 1159);

        static IReadOnlyDictionary<string, long> Shape(long mm, long nn, long kk) =>
            new Dictionary<string, long> { ["M"] = mm, ["N"] = nn, ["K"] = kk };
    }
}
