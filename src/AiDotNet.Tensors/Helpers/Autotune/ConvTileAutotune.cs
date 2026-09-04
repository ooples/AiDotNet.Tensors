using System;
using System.Collections.Generic;
using System.Globalization;
using AiDotNet.Evolution;

namespace AiDotNet.Tensors.Helpers.Autotune;

/// <summary>
/// Tile-candidate generation and cache-key conventions for the shared-memory
/// tiled 1x1 convolution GEMM (issue #841). A 1x1 NCHW convolution is the
/// batched GEMM <c>O[K, HW] = W[K, C] . X[C, HW]</c>, so a square TILE x TILE
/// block is only launchable without boundary predicates when the tile divides
/// K, C, and the spatial extent HW <b>exactly</b> — and only when the block's
/// TILE*TILE threads fit the device's max-threads-per-block limit.
///
/// <para>This is the candidate half of conv autotuning: dispatch feeds the
/// returned candidates to <see cref="GpuFirstRunAutotuner.Resolve"/>, which
/// benchmarks each on-device and caches the winner keyed per card. It contains
/// no kernel dependency (pure divisibility + geometry), so it is unit-testable
/// without a GPU and lands ahead of the kernel it will drive.</para>
/// </summary>
public static class ConvTileAutotune
{
    /// <summary>Autotune category for convolution kernels.</summary>
    public const string Category = "conv2d";

    /// <summary>Kernel-family name for the tiled 1x1 NCHW FP32 GEMM specialization.</summary>
    public const string TiledOneByOneName = "tiled-gemm-1x1-nchw-fp32";

    /// <summary>Prefix for a tile variant id, e.g. <c>"tile-16"</c>.</summary>
    public const string VariantPrefix = "tile-";

    /// <summary>Structured-parameter key carrying the chosen tile edge.</summary>
    public const string TileParameter = "Tile";

    /// <summary>
    /// Default tile edges to sweep, ordered so the first <b>valid</b> tile is a
    /// sound unmeasured fallback: 16 (256 threads — the drafted emitter's tile,
    /// a balanced reuse/occupancy point), then 32 (max reuse, 1024 threads),
    /// then 8 and 4 (progressively lower reuse, but useful for otherwise
    /// untileable exact shapes).
    /// </summary>
    public static readonly IReadOnlyList<int> DefaultTileEdges = new[] { 16, 32, 8, 4 };

    /// <summary>The kernel identity for the tiled 1x1 conv on a given device (keyed per physical card).</summary>
    public static KernelId KernelId(GpuDeviceFingerprint fingerprint) =>
        GpuFirstRunAutotuner.GpuKernelId(Category, TiledOneByOneName, fingerprint);

    /// <summary>
    /// The shape key for one exact contract. Dimension order is
    /// (batch, outputChannels, inputChannels, spatial) so distinct contracts get
    /// distinct cache entries under the same kernel family.
    /// </summary>
    public static ShapeProfile Shape(int batch, int outputChannels, int inputChannels, int spatial) =>
        new(batch, outputChannels, inputChannels, spatial);

    /// <summary>
    /// Returns the tile candidates that can launch the tiled 1x1 conv for the
    /// given contract, in preference order. A tile <c>t</c> qualifies only when
    /// it divides <paramref name="outputChannels"/>, <paramref name="inputChannels"/>,
    /// and <paramref name="spatial"/> exactly (so no boundary predicate is needed
    /// and a column tile never straddles a batch), and when <c>t*t</c> fits
    /// <paramref name="maxThreadsPerBlock"/>. An empty result means no offered
    /// tile is launchable for this contract (e.g. HW=196=14^2 admits none of
    /// {4,8,16,32}); the caller must fall back rather than call the tiled path.
    /// </summary>
    public static IReadOnlyList<AutotuneCandidate> Candidates(
        int outputChannels,
        int inputChannels,
        int spatial,
        IReadOnlyList<int>? tileEdges = null,
        int maxThreadsPerBlock = 1024)
    {
        IReadOnlyList<ConvTileConfiguration> typed = TypedCandidates(
            outputChannels, inputChannels, spatial, tileEdges, maxThreadsPerBlock);
        var result = new List<AutotuneCandidate>(typed.Count);
        for (int i = 0; i < typed.Count; i++) result.Add(CandidateFor(typed[i]));
        return result;
    }

    /// <summary>
    /// Returns valid tile configurations without encoding launch decisions into strings.
    /// </summary>
    public static IReadOnlyList<ConvTileConfiguration> TypedCandidates(
        int outputChannels,
        int inputChannels,
        int spatial,
        IReadOnlyList<int>? tileEdges = null,
        int maxThreadsPerBlock = 1024)
    {
        ValidateLaunchArguments(outputChannels, inputChannels, spatial, maxThreadsPerBlock);

        IReadOnlyList<int> edges = tileEdges ?? DefaultTileEdges;
        var result = new List<ConvTileConfiguration>(edges.Count);
        var seen = new HashSet<int>();
        foreach (int tileEdge in edges)
        {
            if (tileEdge <= 0 || !seen.Add(tileEdge)) continue;
            if (!IsLaunchableTile(
                    tileEdge, outputChannels, inputChannels, spatial, maxThreadsPerBlock))
                continue;
            result.Add(new ConvTileConfiguration(tileEdge));
        }
        return result;
    }

    /// <summary>
    /// Creates an evolutionary tuner whose benchmark delegate can launch the typed configuration on the real GPU.
    /// </summary>
    /// <remarks>
    /// The returned tuner performs search off the serving path and atomically publishes its winner through
    /// <paramref name="deployment"/>. CPU-only tests may supply a deterministic fake benchmark; production callers
    /// supply the same delegate shape backed by CUDA, PTX, Vulkan, Metal, or another device backend.
    /// </remarks>
    public static EvolutionKernelAutotuner<ConvTileConfiguration> CreateEvolutionTuner(
        GpuDeviceFingerprint fingerprint,
        int batch,
        int outputChannels,
        int inputChannels,
        int spatial,
        Func<ConvTileConfiguration, EvolutionEvaluationContext, CancellationToken,
            ValueTask<KernelTuningMeasurement>> benchmark,
        string searchSpaceVersion,
        string benchmarkVersion,
        EvolutionEngineOptions? options = null,
        IEvolutionCheckpointStore? checkpointStore = null,
        KernelTuningDeployment<ConvTileConfiguration>? deployment = null,
        IReadOnlyList<int>? tileEdges = null,
        int maxThreadsPerBlock = 1024)
    {
        if (batch <= 0) throw new ArgumentOutOfRangeException(nameof(batch));
        if (benchmark is null) throw new ArgumentNullException(nameof(benchmark));
        IReadOnlyList<ConvTileConfiguration> candidates = TypedCandidates(
            outputChannels, inputChannels, spatial, tileEdges, maxThreadsPerBlock);
        if (candidates.Count == 0)
            throw new ArgumentException("The supplied shape and device limit admit no convolution tiles.", nameof(tileEdges));

        var identity = new KernelTuningIdentity(
            new KernelId(Category, TiledOneByOneName),
            Shape(batch, outputChannels, inputChannels, spatial),
            fingerprint,
            searchSpaceVersion,
            benchmarkVersion);
        return new EvolutionKernelAutotuner<ConvTileConfiguration>(
            identity,
            new ConvTileCodec(candidates),
            new ConvTileVariation(candidates),
            benchmark,
            options,
            checkpointStore: checkpointStore,
            deployment: deployment);
    }

    /// <summary>True when at least one offered tile can launch this contract.</summary>
    public static bool HasLaunchableTile(
        int outputChannels, int inputChannels, int spatial,
        IReadOnlyList<int>? tileEdges = null, int maxThreadsPerBlock = 1024)
    {
        ValidateLaunchArguments(outputChannels, inputChannels, spatial, maxThreadsPerBlock);

        IReadOnlyList<int> edges = tileEdges ?? DefaultTileEdges;
        for (int i = 0; i < edges.Count; i++)
            if (IsLaunchableTile(
                    edges[i], outputChannels, inputChannels, spatial, maxThreadsPerBlock))
                return true;
        return false;
    }

    private static void ValidateLaunchArguments(
        int outputChannels, int inputChannels, int spatial, int maxThreadsPerBlock)
    {
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (spatial <= 0) throw new ArgumentOutOfRangeException(nameof(spatial));
        if (maxThreadsPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(maxThreadsPerBlock));
    }

    private static bool IsLaunchableTile(
        int tile, int outputChannels, int inputChannels, int spatial, int maxThreadsPerBlock) =>
        tile > 0 &&
        (long)tile * tile <= maxThreadsPerBlock &&
        outputChannels % tile == 0 &&
        inputChannels % tile == 0 &&
        spatial % tile == 0;

    /// <summary>Builds the candidate for a specific tile edge.</summary>
    public static AutotuneCandidate CandidateFor(int tile)
    {
        return CandidateFor(new ConvTileConfiguration(tile));
    }

    private static AutotuneCandidate CandidateFor(ConvTileConfiguration configuration)
    {
        string edge = configuration.TileEdge.ToString(CultureInfo.InvariantCulture);
        return new AutotuneCandidate(
            VariantPrefix + edge,
            new Dictionary<string, string>(StringComparer.Ordinal) { [TileParameter] = edge });
    }

    /// <summary>Parses the tile edge back out of a variant id (e.g. <c>"tile-16"</c> -&gt; 16).</summary>
    public static bool TryParseTile(string? variant, out int tile)
    {
        tile = 0;
        if (string.IsNullOrEmpty(variant) ||
            !variant!.StartsWith(VariantPrefix, StringComparison.Ordinal))
            return false;
        return int.TryParse(
            variant.Substring(VariantPrefix.Length),
            NumberStyles.None, CultureInfo.InvariantCulture, out tile) && tile > 0;
    }

    /// <summary>Reads the chosen tile edge from a resolved winner (variant preferred, parameter fallback).</summary>
    public static bool TryGetTile(AutotuneResolution resolution, out int tile)
    {
        if (TryParseTile(resolution.Variant, out tile)) return true;
        if (resolution.Parameters is not null &&
            resolution.Parameters.TryGetValue(TileParameter, out string? raw) &&
            int.TryParse(raw, NumberStyles.None, CultureInfo.InvariantCulture, out tile) && tile > 0)
            return true;
        tile = 0;
        return false;
    }

    private sealed class ConvTileCodec : IEvolutionGenomeCodec<ConvTileConfiguration>
    {
        private readonly HashSet<int> _allowedTileEdges;

        public ConvTileCodec(IReadOnlyList<ConvTileConfiguration> candidates)
        {
            _allowedTileEdges = new HashSet<int>(candidates.Select(candidate => candidate.TileEdge));
            VersionHash = CandidateSpaceHash("conv-tile-payload-v1", candidates);
        }

        public string Id => "conv-tile";

        public string VersionHash { get; }

        public string Serialize(ConvTileConfiguration genome) =>
            _allowedTileEdges.Contains(genome.TileEdge)
                ? genome.TileEdge.ToString(CultureInfo.InvariantCulture)
                : throw new ArgumentOutOfRangeException(
                    nameof(genome),
                    "The tile edge is not part of this tuner's validated candidate space.");

        public ConvTileConfiguration Deserialize(string payload)
        {
            if (!int.TryParse(payload, NumberStyles.None, CultureInfo.InvariantCulture, out int tileEdge) ||
                !_allowedTileEdges.Contains(tileEdge))
            {
                throw new InvalidDataException(
                    "The convolution tile payload is not part of this tuner's validated candidate space.");
            }
            return new ConvTileConfiguration(tileEdge);
        }
    }

    private sealed class ConvTileVariation : IVariationOperator<ConvTileConfiguration>
    {
        private readonly ConvTileConfiguration[] _candidates;

        public ConvTileVariation(IReadOnlyList<ConvTileConfiguration> candidates)
        {
            _candidates = candidates.ToArray();
            VersionHash = CandidateSpaceHash("conv-tile-finite-space-v1", candidates);
        }

        public string Id => "conv-tile-finite-space";

        public string VersionHash { get; }

        public ValueTask<ConvTileConfiguration> ProposeAsync(
            EvolutionVariationContext<ConvTileConfiguration> context,
            CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            return new ValueTask<ConvTileConfiguration>(
                _candidates[context.Random.NextInt(_candidates.Length)]);
        }
    }

    private static string CandidateSpaceHash(
        string componentVersion,
        IReadOnlyList<ConvTileConfiguration> candidates)
    {
        var components = new string[candidates.Count + 1];
        components[0] = componentVersion;
        for (int i = 0; i < candidates.Count; i++)
            components[i + 1] = candidates[i].TileEdge.ToString(CultureInfo.InvariantCulture);
        return EvolutionHash.Combine(components);
    }
}
