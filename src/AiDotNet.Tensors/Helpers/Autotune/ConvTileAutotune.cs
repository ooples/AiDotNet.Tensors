using System;
using System.Collections.Generic;
using System.Globalization;

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
    /// then 8 (low occupancy, last resort).
    /// </summary>
    public static readonly IReadOnlyList<int> DefaultTileEdges = new[] { 16, 32, 8 };

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
    /// {8,16,32}); the caller must fall back rather than call the tiled path.
    /// </summary>
    public static IReadOnlyList<AutotuneCandidate> Candidates(
        int outputChannels,
        int inputChannels,
        int spatial,
        IReadOnlyList<int>? tileEdges = null,
        int maxThreadsPerBlock = 1024)
    {
        if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
        if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
        if (spatial <= 0) throw new ArgumentOutOfRangeException(nameof(spatial));
        if (maxThreadsPerBlock <= 0) throw new ArgumentOutOfRangeException(nameof(maxThreadsPerBlock));

        IReadOnlyList<int> edges = tileEdges ?? DefaultTileEdges;
        var result = new List<AutotuneCandidate>(edges.Count);
        var seen = new HashSet<int>();
        foreach (int t in edges)
        {
            if (t <= 0 || !seen.Add(t)) continue;                 // ignore junk / duplicates
            if ((long)t * t > maxThreadsPerBlock) continue;       // block would exceed thread limit
            if (outputChannels % t != 0 || inputChannels % t != 0 || spatial % t != 0) continue; // needs exact division
            result.Add(CandidateFor(t));
        }
        return result;
    }

    /// <summary>True when at least one offered tile can launch this contract.</summary>
    public static bool HasLaunchableTile(
        int outputChannels, int inputChannels, int spatial,
        IReadOnlyList<int>? tileEdges = null, int maxThreadsPerBlock = 1024) =>
        Candidates(outputChannels, inputChannels, spatial, tileEdges, maxThreadsPerBlock).Count > 0;

    /// <summary>Builds the candidate for a specific tile edge.</summary>
    public static AutotuneCandidate CandidateFor(int tile)
    {
        if (tile <= 0) throw new ArgumentOutOfRangeException(nameof(tile));
        string edge = tile.ToString(CultureInfo.InvariantCulture);
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
}
