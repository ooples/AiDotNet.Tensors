using System.Globalization;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Helpers.Autotune;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Simd;

/// <summary>
/// Chooses, per exact conv shape and pass, between the engine's existing conv routes and <see cref="DirectConvAvx2"/>
/// (and the direct kernel's task granularity), by timing the real engine pass under AiDotNet.Evolution's kernel tuner.
/// Every candidate is validated against a double-precision reference before its time counts. A winner is activated
/// in <see cref="DirectConvTuning"/> and persisted through the kernel-tuning store, so a later process can activate it
/// with <see cref="TryActivatePersisted"/> instead of re-measuring.
/// </summary>
/// <remarks>
/// Tuning swaps the process-wide configuration for the shape while it measures, so run it before training starts,
/// not alongside conv work on other threads.
/// </remarks>
internal static class DirectConvEvolutionAutotuner
{
    private static readonly KernelId ForwardKernel = new("direct-conv-avx2", "conv2d-forward-fp32");
    private static readonly KernelId BackwardInputKernel = new("direct-conv-avx2", "conv2d-backward-input-fp32");
    private static readonly KernelId BackwardKernelKernel = new("direct-conv-avx2", "conv2d-backward-kernel-fp32");

    /// <summary>Task counts the direct forward and input-gradient kernels may split their output tiles into.</summary>
    internal static readonly int[] TaskTargets = { 32, 64, 128, 256, 512, 1024 };

    private const int DefaultTaskTarget = 128;

    /// <summary>Every configuration that can run <paramref name="shape"/>, the existing route first.</summary>
    public static IReadOnlyList<DirectConvConfiguration> GetSearchSpace(in DirectConvShape shape)
    {
        var space = new List<DirectConvConfiguration> { new(DirectConvRoute.Im2Col, 0) };
        if (!DirectConvAvx2.IsEligible(shape)) return space;
        if (shape.Pass == DirectConvPass.BackwardKernel)
            space.Add(new DirectConvConfiguration(DirectConvRoute.Direct, 0));
        else
            foreach (int tasks in TaskTargets) space.Add(new DirectConvConfiguration(DirectConvRoute.Direct, tasks));
        return space;
    }

    /// <summary>What the shape runs today without a tuned configuration.</summary>
    public static DirectConvConfiguration GetIncumbent(in DirectConvShape shape)
        => DirectConvAvx2.IsEligible(shape) && DirectConvAvx2.DefaultChoosesDirect(shape)
            ? Canonicalize(shape, new DirectConvConfiguration(DirectConvRoute.Direct, DefaultTaskTarget))
            : new DirectConvConfiguration(DirectConvRoute.Im2Col, 0);

    public static KernelTuningExperiment<DirectConvConfiguration> CreateExperiment(
        in DirectConvShape shape, int inputSeed = 1729, int warmupCount = 2, int searchSampleCount = 5, int holdoutSampleCount = 9)
    {
        ValidateShape(shape);
        return new KernelTuningExperiment<DirectConvConfiguration>(
            new DirectConvExperimentBackend(shape, inputSeed),
            new StopwatchKernelTuningTimer(),
            GetIncumbent(shape),
            new KernelTuningWorkload(shape.FloatingPointOperations, KernelTuningWorkUnit.FloatingPointOperations),
            KernelTuningTimingScope.SteadyStateExecution,
            warmupCount,
            searchSampleCount,
            holdoutSampleCount);
    }

    public static EvolutionKernelAutotuner<DirectConvConfiguration> Create(
        DirectConvShape shape,
        Func<DirectConvConfiguration, EvolutionEvaluationContext, CancellationToken, ValueTask<KernelTuningTrialResult>> evaluator,
        IKernelTuningFinalistEvaluator<DirectConvConfiguration> finalistEvaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeploymentRegistry<DirectConvConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<DirectConvConfiguration>? store = null)
    {
        ValidateShape(shape);
        if (evaluator is null) throw new ArgumentNullException(nameof(evaluator));
        if (finalistEvaluator is null) throw new ArgumentNullException(nameof(finalistEvaluator));
        return new EvolutionKernelAutotuner<DirectConvConfiguration>(
            CreateIdentity(shape, searchSpaceVersion, benchmarkProtocolVersion),
            new DirectConvCodec(),
            new DirectConvVariation(shape),
            (configuration, context, cancellationToken) =>
            {
                KernelTuningTrialResult? invalid = ValidateConfiguration(shape, configuration);
                return invalid is null
                    ? evaluator(configuration, context, cancellationToken)
                    : new ValueTask<KernelTuningTrialResult>(invalid);
            },
            finalistEvaluator,
            engineOptions ?? DefaultEngineOptions(GetSearchSpace(shape).Count),
            tuningOptions ?? DefaultTuningOptions(),
            checkpointStore: checkpointStore,
            deploymentRegistry: deploymentRegistry,
            store: store,
            deploymentValidator: configuration => ValidateConfiguration(shape, configuration) is null);
    }

    /// <summary>
    /// Measures every configuration of <paramref name="shape"/> (the space is small enough to be exhaustive) on the
    /// real engine pass, and activates the deployed winner.
    /// </summary>
    public static async Task<EvolutionKernelTuningResult<DirectConvConfiguration>> TuneAsync(
        DirectConvShape shape,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        KernelTuningDeploymentRegistry<DirectConvConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<DirectConvConfiguration>? store = null,
        CancellationToken cancellationToken = default)
    {
        KernelTuningExperiment<DirectConvConfiguration> experiment = CreateExperiment(shape);
        return await TuneAsync(shape, experiment.EvaluateAsync, experiment, searchSpaceVersion, benchmarkProtocolVersion,
            engineOptions, tuningOptions, deploymentRegistry, store, cancellationToken).ConfigureAwait(false);
    }

    public static async Task<EvolutionKernelTuningResult<DirectConvConfiguration>> TuneAsync(
        DirectConvShape shape,
        Func<DirectConvConfiguration, EvolutionEvaluationContext, CancellationToken, ValueTask<KernelTuningTrialResult>> evaluator,
        IKernelTuningFinalistEvaluator<DirectConvConfiguration> finalistEvaluator,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        EvolutionEngineOptions? engineOptions = null,
        KernelTuningOptions? tuningOptions = null,
        KernelTuningDeploymentRegistry<DirectConvConfiguration>? deploymentRegistry = null,
        IKernelTuningStore<DirectConvConfiguration>? store = null,
        CancellationToken cancellationToken = default)
    {
        IReadOnlyList<DirectConvConfiguration> space = GetSearchSpace(shape);
        EvolutionKernelAutotuner<DirectConvConfiguration> tuner = Create(shape, evaluator, finalistEvaluator,
            searchSpaceVersion, benchmarkProtocolVersion, engineOptions, tuningOptions,
            deploymentRegistry: deploymentRegistry, store: store);
        EvolutionKernelTuningResult<DirectConvConfiguration> result =
            await tuner.TuneExhaustiveAsync(space, cancellationToken).ConfigureAwait(false);
        DirectConvConfiguration winner = result.ActiveDeployment.Configuration;
        if (ValidateConfiguration(shape, winner) is null) DirectConvTuning.Activate(shape, winner);
        return result;
    }

    /// <summary>Activates the configuration a previous run persisted for <paramref name="shape"/>, if there is a valid one.</summary>
    public static bool TryActivatePersisted(
        DirectConvShape shape,
        KernelSearchSpaceVersion searchSpaceVersion,
        KernelBenchmarkProtocolVersion benchmarkProtocolVersion,
        IKernelTuningStore<DirectConvConfiguration>? store = null)
    {
        ValidateShape(shape);
        KernelTuningIdentity identity = CreateIdentity(shape, searchSpaceVersion, benchmarkProtocolVersion);
        var codec = new DirectConvCodec();
        IKernelTuningStore<DirectConvConfiguration> resolvedStore =
            store ?? new AutotuneCacheKernelTuningStore<DirectConvConfiguration>();
        KernelTuningDeploymentSnapshot<DirectConvConfiguration>? snapshot;
        try
        {
            if (!resolvedStore.TryLoad(identity, codec, out snapshot)
                || snapshot is null
                || !string.Equals(snapshot.Identity.StableKey, identity.StableKey, StringComparison.Ordinal)
                || ValidateConfiguration(shape, snapshot.Configuration) is not null)
                return false;
            if (!string.Equals(EvolutionHash.Compute(codec.Serialize(snapshot.Configuration)), snapshot.GenomeId, StringComparison.Ordinal))
                return false;
        }
        catch (InvalidDataException ex)
        {
            // A corrupt or foreign cache entry: keep the measured defaults rather than fail the conv.
            System.Diagnostics.Trace.TraceWarning("Ignoring an unreadable direct-conv tuning entry for {0}: {1}", shape, ex);
            return false;
        }
        catch (IOException ex)
        {
            System.Diagnostics.Trace.TraceWarning("Could not read the direct-conv tuning store for {0}: {1}", shape, ex);
            return false;
        }

        DirectConvTuning.Activate(shape, snapshot.Configuration);
        return true;
    }

    internal static KernelTuningIdentity CreateIdentity(
        in DirectConvShape shape, KernelSearchSpaceVersion searchSpaceVersion, KernelBenchmarkProtocolVersion benchmarkProtocolVersion)
        => new KernelTuningIdentity(
            shape.Pass switch
            {
                DirectConvPass.Forward => ForwardKernel,
                DirectConvPass.BackwardInput => BackwardInputKernel,
                _ => BackwardKernelKernel,
            },
            new ShapeProfile(shape.Batch, shape.InChannels, shape.OutChannels, shape.Height, shape.Width,
                shape.KernelHeight, shape.KernelWidth, shape.StrideH, shape.StrideW, shape.PadH, shape.PadW,
                shape.DilationH, shape.DilationW),
            KernelTuningDeviceFingerprint.CurrentCpu(),
            KernelTuningBackend.ManagedCpu,
            searchSpaceVersion,
            benchmarkProtocolVersion);

    internal static KernelTuningTrialResult? ValidateConfiguration(in DirectConvShape shape, DirectConvConfiguration configuration)
    {
        if (!Enum.IsDefined(typeof(DirectConvRoute), configuration.Route))
            return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration, "Unknown direct-conv route.");
        if (configuration.Route == DirectConvRoute.Im2Col)
            return configuration.TargetTasks == 0
                ? null
                : KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration, "The im2col route has no task target.");
        if (!DirectConvAvx2.IsEligible(shape))
            return KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration,
                "The direct kernels cannot run this shape on this CPU.");
        bool tasksValid = shape.Pass == DirectConvPass.BackwardKernel
            ? configuration.TargetTasks == 0
            : Array.IndexOf(TaskTargets, configuration.TargetTasks) >= 0;
        return tasksValid
            ? null
            : KernelTuningTrialResult.Rejected(KernelTuningTrialStatus.InvalidConfiguration,
                "The task target is outside the search space for this pass.");
    }

    private static DirectConvConfiguration Canonicalize(in DirectConvShape shape, DirectConvConfiguration configuration)
    {
        if (configuration.Route != DirectConvRoute.Direct || !DirectConvAvx2.IsEligible(shape))
            return new DirectConvConfiguration(DirectConvRoute.Im2Col, 0);
        if (shape.Pass == DirectConvPass.BackwardKernel) return new DirectConvConfiguration(DirectConvRoute.Direct, 0);
        return Array.IndexOf(TaskTargets, configuration.TargetTasks) >= 0
            ? configuration
            : new DirectConvConfiguration(DirectConvRoute.Direct, DefaultTaskTarget);
    }

    // The archive's resource axis is the scratch each route allocates; the existing routes have no launch count to report.
    private static KernelTuningOptions DefaultTuningOptions() => new()
    {
        ArchiveProfile = KernelTuningArchiveProfile.Custom,
        ArchiveDescriptors = new[] { new KernelTuningDescriptorDefinition(KernelTuningMetric.Log2WorkspaceBytes, 0, 40, 16) },
    };

    private static EvolutionEngineOptions DefaultEngineOptions(int count) => new()
    {
        RunId = "direct-conv-avx2-tuning",
        Seed = 1729,
        MaxEvaluationAttempts = count,
        MaxProposals = count,
        MaxGenerations = 0,
        ProposalBatchSize = Math.Max(1, count),
        // One candidate at a time: each one is a parallel conv that needs the whole pool to time fairly.
        MaxDegreeOfParallelism = 1,
        IslandCount = 1,
        MigrationInterval = 0,
        MigrantsPerIsland = 1,
    };

    private static void ValidateShape(in DirectConvShape shape)
    {
        if (!Enum.IsDefined(typeof(DirectConvPass), shape.Pass)) throw new ArgumentOutOfRangeException(nameof(shape));
        if (shape.Batch <= 0 || shape.InChannels <= 0 || shape.OutChannels <= 0 || shape.Height <= 0 || shape.Width <= 0
            || shape.KernelHeight <= 0 || shape.KernelWidth <= 0 || shape.StrideH <= 0 || shape.StrideW <= 0
            || shape.PadH < 0 || shape.PadW < 0 || shape.DilationH <= 0 || shape.DilationW <= 0
            || shape.OutputHeight <= 0 || shape.OutputWidth <= 0)
            throw new ArgumentException($"Not a valid conv shape: {shape}.", nameof(shape));
    }

    private sealed class DirectConvCodec : IEvolutionGenomeCodec<DirectConvConfiguration>
    {
        public string Id => "direct-conv-avx2-route";
        public string VersionHash => "1";

        public string Serialize(DirectConvConfiguration genome)
        {
            ValidatePayload(genome);
            return ((int)genome.Route).ToString(CultureInfo.InvariantCulture) + "|"
                + genome.TargetTasks.ToString(CultureInfo.InvariantCulture);
        }

        public DirectConvConfiguration Deserialize(string payload)
        {
            if (payload is null) throw new ArgumentNullException(nameof(payload));
            string[] values = payload.Split('|');
            if (values.Length != 2) throw new InvalidDataException("Invalid direct-conv genome field count.");
            var result = new DirectConvConfiguration((DirectConvRoute)Parse(values[0]), Parse(values[1]));
            ValidatePayload(result);
            return result;
        }

        private static void ValidatePayload(DirectConvConfiguration value)
        {
            if (!Enum.IsDefined(typeof(DirectConvRoute), value.Route) || value.TargetTasks < 0
                || (value.Route == DirectConvRoute.Im2Col && value.TargetTasks != 0))
                throw new InvalidDataException("The direct-conv genome contains an invalid typed field.");
        }

        private static int Parse(string value) =>
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                ? parsed
                : throw new InvalidDataException("The direct-conv genome contains an invalid integer.");
    }

    private sealed class DirectConvVariation : IVariationOperator<DirectConvConfiguration>
    {
        private readonly DirectConvShape _shape;

        internal DirectConvVariation(DirectConvShape shape) => _shape = shape;

        public string Id => "direct-conv-avx2-route-variation";
        public string VersionHash => "1";

        public ValueTask<DirectConvConfiguration> ProposeAsync(
            EvolutionVariationContext<DirectConvConfiguration> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            DirectConvConfiguration value = context.Parent.Candidate.CanonicalGenome.Genome;
            value = context.Random.NextInt(2) == 0
                ? value with { Route = value.Route == DirectConvRoute.Direct ? DirectConvRoute.Im2Col : DirectConvRoute.Direct }
                : value with { Route = DirectConvRoute.Direct, TargetTasks = TaskTargets[context.Random.NextInt(TaskTargets.Length)] };
            return new ValueTask<DirectConvConfiguration>(Canonicalize(_shape, value));
        }
    }
}

/// <summary>
/// Runs one conv pass through the engine with a candidate configuration forced for its shape, and checks the result
/// against a double-precision reference computed once.
/// </summary>
internal sealed class DirectConvExperimentBackend : IKernelTuningExperimentBackend<DirectConvConfiguration>
{
    private readonly DirectConvShape _shape;
    private readonly CpuEngine _engine = new();
    private readonly Tensor<float> _input;
    private readonly Tensor<float> _kernel;
    private readonly Tensor<float> _gradOutput;
    private readonly Tensor<float> _result;
    private readonly double[] _reference;
    private readonly int[] _stride;
    private readonly int[] _padding;
    private readonly int[] _dilation;

    internal DirectConvExperimentBackend(DirectConvShape shape, int inputSeed)
    {
        _shape = shape;
        int oh = shape.OutputHeight, ow = shape.OutputWidth;
        _input = Fill(new[] { shape.Batch, shape.InChannels, shape.Height, shape.Width }, unchecked((uint)inputSeed));
        _kernel = Fill(new[] { shape.OutChannels, shape.InChannels, shape.KernelHeight, shape.KernelWidth }, unchecked((uint)inputSeed ^ 0x9e3779b9u));
        _gradOutput = Fill(new[] { shape.Batch, shape.OutChannels, oh, ow }, unchecked((uint)inputSeed ^ 0x85ebca6bu));
        _stride = new[] { shape.StrideH, shape.StrideW };
        _padding = new[] { shape.PadH, shape.PadW };
        _dilation = new[] { shape.DilationH, shape.DilationW };
        _result = new Tensor<float>(shape.Pass switch
        {
            DirectConvPass.Forward => _gradOutput.Shape.ToArray(),
            DirectConvPass.BackwardInput => _input.Shape.ToArray(),
            _ => _kernel.Shape.ToArray(),
        });
        _reference = new double[_result.Length];
        ComputeReference();
    }

    public ValueTask PrepareAsync(DirectConvConfiguration configuration, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return default;
    }

    public ValueTask ExecuteAsync(DirectConvConfiguration configuration, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        bool hadPrevious = DirectConvTuning.TryGet(_shape, out DirectConvConfiguration previous);
        DirectConvTuning.Activate(_shape, configuration);
        try
        {
            switch (_shape.Pass)
            {
                case DirectConvPass.Forward:
                    _engine.Conv2DInto(_result, _input, _kernel, _stride, _padding, _dilation);
                    break;
                case DirectConvPass.BackwardInput:
                    _engine.Conv2DBackwardInputInto(_result, _gradOutput, _kernel, _input.Shape.ToArray(),
                        _stride, _padding, _dilation, accumulate: false);
                    break;
                default:
                    _engine.Conv2DBackwardKernelInto(_result, _gradOutput, _input, _kernel.Shape.ToArray(),
                        _stride, _padding, _dilation, accumulate: false);
                    break;
            }
        }
        finally
        {
            if (hadPrevious) DirectConvTuning.Activate(_shape, previous);
            else DirectConvTuning.Deactivate(_shape);
        }
        return default;
    }

    public ValueTask SynchronizeAsync(CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return default;
    }

    public async ValueTask<KernelTuningCorrectnessEvidence> ValidateAsync(
        DirectConvConfiguration configuration, CancellationToken cancellationToken = default)
    {
        for (int i = 0; i < _result.Length; i++) _result.SetFlat(i, float.NaN);
        await ExecuteAsync(configuration, cancellationToken).ConfigureAwait(false);
        double maximumAbsoluteError = 0d, maximumRelativeError = 0d;
        for (int i = 0; i < _reference.Length; i++)
        {
            double actual = _result.GetFlat(i), expected = _reference[i];
            double absoluteError = Math.Abs(actual - expected);
            if (double.IsNaN(absoluteError)) absoluteError = double.PositiveInfinity;
            maximumAbsoluteError = Math.Max(maximumAbsoluteError, absoluteError);
            maximumRelativeError = Math.Max(maximumRelativeError, absoluteError / Math.Max(Math.Abs(expected), 1e-30d));
        }
        int reduction = _shape.Pass switch
        {
            DirectConvPass.Forward => _shape.InChannels * _shape.KernelHeight * _shape.KernelWidth,
            DirectConvPass.BackwardInput => _shape.OutChannels * _shape.KernelHeight * _shape.KernelWidth,
            _ => _shape.Batch * _shape.OutputHeight * _shape.OutputWidth,
        };
        double absoluteTolerance = 5e-5d * Math.Max(1d, reduction / 64d);
        const double relativeTolerance = 5e-4d;
        if (maximumAbsoluteError > absoluteTolerance && maximumRelativeError > relativeTolerance)
            throw new KernelTuningValidationException(KernelTuningTrialStatus.OutputMismatch,
                "The conv pass differs from the double-precision reference.");
        return new KernelTuningCorrectnessEvidence(KernelTuningValidationScope.Output,
            maximumAbsoluteError, maximumRelativeError, absoluteTolerance, relativeTolerance);
    }

    public KernelTuningResourceUsage GetResourceUsage(DirectConvConfiguration configuration) => new(
        KernelTuningResourceMetric<long>.Measured(WorkspaceBytes(_shape, configuration.Route)),
        KernelTuningResourceMetric<double>.NotApplicable(),
        KernelTuningResourceMetric<int>.NotApplicable(),
        KernelTuningResourceMetric<TimeSpan>.Measured(TimeSpan.Zero),
        KernelTuningResourceMetric<int>.Measured(1));

    /// <summary>
    /// Scratch the pass allocates: the direct kernels' packed (padded, channel-blocked) operands, or for the existing
    /// routes the column matrix of the batch-wide lowered GEMM.
    /// </summary>
    internal static long WorkspaceBytes(in DirectConvShape s, DirectConvRoute route)
    {
        long taps = (long)s.KernelHeight * s.KernelWidth;
        long positions = (long)s.OutputHeight * s.OutputWidth;
        if (route == DirectConvRoute.Im2Col)
            return s.InChannels * taps * s.Batch * positions * sizeof(float);
        long kernel = s.OutChannels * s.InChannels * taps;
        switch (s.Pass)
        {
            case DirectConvPass.Forward:
                return ((long)s.Batch * s.InChannels * (s.Height + 2 * s.PadH) * (s.Width + 2 * s.PadW) + kernel) * sizeof(float);
            case DirectConvPass.BackwardInput:
                bool strided = s.StrideH > 1 || s.StrideW > 1;
                long borderH = strided ? s.KernelHeight : s.KernelHeight - 1 - s.PadH;
                long borderW = strided ? s.KernelWidth : s.KernelWidth - 1 - s.PadW;
                return ((long)s.Batch * s.OutChannels * (s.OutputHeight + 2 * borderH) * (s.OutputWidth + 2 * borderW) + kernel) * sizeof(float);
            default:
                return ((long)s.Batch * s.InChannels * (s.Height + 2 * s.PadH) * (s.Width + 2 * s.PadW)
                    + (long)s.Batch * s.OutChannels * positions) * sizeof(float) + s.Batch * positions * sizeof(int);
        }
    }

    private void ComputeReference()
    {
        var s = _shape;
        int oh = s.OutputHeight, ow = s.OutputWidth;
        float[] x = _input.ToArray(), w = _kernel.ToArray(), g = _gradOutput.ToArray();
        double[] r = _reference;
        // One task per output plane (forward / input gradient) or per output channel (kernel gradient): each writes
        // only its own slice of the reference.
        int tasks = s.Pass == DirectConvPass.BackwardKernel ? s.OutChannels
            : s.Batch * (s.Pass == DirectConvPass.Forward ? s.OutChannels : s.InChannels);
        CpuParallelSettings.ParallelForOrSerial(0, tasks, (long)tasks * 1_000_000, t =>
        {
            // The task fixes (b, o) forward, (b, c) for the input gradient, o for the kernel gradient.
            int b0 = 0, b1 = s.Batch, o0 = 0, o1 = s.OutChannels, c0 = 0, c1 = s.InChannels;
            switch (s.Pass)
            {
                case DirectConvPass.Forward: b0 = t / s.OutChannels; b1 = b0 + 1; o0 = t % s.OutChannels; o1 = o0 + 1; break;
                case DirectConvPass.BackwardInput: b0 = t / s.InChannels; b1 = b0 + 1; c0 = t % s.InChannels; c1 = c0 + 1; break;
                default: o0 = t; o1 = t + 1; break;
            }
            for (int b = b0; b < b1; b++)
            for (int o = o0; o < o1; o++)
            for (int c = c0; c < c1; c++)
            {                for (int y = 0; y < oh; y++)
                for (int z = 0; z < ow; z++)
                for (int i = 0; i < s.KernelHeight; i++)
                for (int j = 0; j < s.KernelWidth; j++)
                {
                    int ih = y * s.StrideH + i * s.DilationH - s.PadH, iw = z * s.StrideW + j * s.DilationW - s.PadW;
                    if (ih < 0 || ih >= s.Height || iw < 0 || iw >= s.Width) continue;
                    int xi = ((b * s.InChannels + c) * s.Height + ih) * s.Width + iw;
                    int wi = ((o * s.InChannels + c) * s.KernelHeight + i) * s.KernelWidth + j;
                    int gi = ((b * s.OutChannels + o) * oh + y) * ow + z;
                    switch (s.Pass)
                    {
                        case DirectConvPass.Forward: r[gi] += (double)w[wi] * x[xi]; break;
                        case DirectConvPass.BackwardInput: r[xi] += (double)w[wi] * g[gi]; break;
                        default: r[wi] += (double)g[gi] * x[xi]; break;
                    }
                }
            }
        }, deterministicSafe: true);
    }

    private static Tensor<float> Fill(int[] shape, uint state)
    {
        var t = new Tensor<float>(shape);
        for (int i = 0; i < t.Length; i++)
        {
            state = unchecked(state * 1664525u + 1013904223u);
            t.SetFlat(i, (float)(((state >> 8) / 16777216d - 0.5d) * 0.25d));
        }
        return t;
    }
}
