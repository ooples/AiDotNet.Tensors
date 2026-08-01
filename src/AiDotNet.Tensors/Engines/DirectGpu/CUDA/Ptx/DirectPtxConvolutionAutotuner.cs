using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers.Autotune;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

internal readonly record struct DirectPtxConvolutionVariant(bool IsTiled, int Tile)
{
    internal static DirectPtxConvolutionVariant Direct => new(false, 0);

    internal static DirectPtxConvolutionVariant Tiled(int tile) => new(true, tile);
}

/// <summary>
/// Production resolver for the exact fused 1x1 convolution contract. The
/// shipped unrolled kernel remains candidate zero, so disabled tuning, failed
/// measurements, and cache misses all preserve the established implementation.
/// Tiled kernels can replace it only after stable on-device evidence.
/// </summary>
internal static class DirectPtxConvolutionAutotuner
{
    internal const string ShareableKernelName = "fused-1x1-bias-relu-nchw-fp32-v1";
    internal const string DirectVariant = "unrolled-direct";
    private const string AlgorithmParameter = "Algorithm";
    private const string DirectAlgorithm = "unrolled";
    private const string TiledAlgorithm = "shared-memory-tiled";

    internal static IReadOnlyList<AutotuneCandidate> CandidateConfigurations(
        int outputChannels,
        int inputChannels,
        int spatial,
        int maxThreadsPerBlock = 1024)
    {
        IReadOnlyList<AutotuneCandidate> tiles = ConvTileAutotune.Candidates(
            outputChannels, inputChannels, spatial,
            maxThreadsPerBlock: maxThreadsPerBlock);
        var result = new List<AutotuneCandidate>(tiles.Count + 1)
        {
            new(DirectVariant, new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [AlgorithmParameter] = DirectAlgorithm
            })
        };
        for (int i = 0; i < tiles.Count; i++)
        {
            AutotuneCandidate tile = tiles[i];
            var parameters = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [AlgorithmParameter] = TiledAlgorithm
            };
            foreach (KeyValuePair<string, string> pair in tile.Parameters)
                parameters[pair.Key] = pair.Value;
            result.Add(new AutotuneCandidate(tile.Variant, parameters));
        }
        return result;
    }

    internal static bool TryGetVariant(
        AutotuneCandidate candidate,
        int outputChannels,
        int inputChannels,
        int spatial,
        out DirectPtxConvolutionVariant variant,
        int maxThreadsPerBlock = 1024)
    {
        if (outputChannels <= 0 || inputChannels <= 0 || spatial <= 0 ||
            maxThreadsPerBlock <= 0)
        {
            variant = default;
            return false;
        }

        if (string.Equals(candidate.Variant, DirectVariant, StringComparison.Ordinal))
        {
            variant = DirectPtxConvolutionVariant.Direct;
            return true;
        }

        if (!ConvTileAutotune.TryParseTile(candidate.Variant, out int tile) ||
            (long)tile * tile > maxThreadsPerBlock ||
            outputChannels % tile != 0 || inputChannels % tile != 0 || spatial % tile != 0)
        {
            variant = default;
            return false;
        }

        // Only variants deliberately offered by this build may be loaded from
        // local or community state. This rejects arbitrary divisible tile sizes
        // that the emitter has never entered into the production sweep.
        IReadOnlyList<int> offered = ConvTileAutotune.DefaultTileEdges;
        bool found = false;
        for (int i = 0; i < offered.Count; i++)
        {
            if (offered[i] != tile) continue;
            found = true;
            break;
        }
        if (!found)
        {
            variant = default;
            return false;
        }

        variant = DirectPtxConvolutionVariant.Tiled(tile);
        return true;
    }

    internal static DirectPtxConvolutionVariant Resolve(
        DirectPtxRuntime runtime,
        int batch,
        int outputChannels,
        int inputChannels,
        int spatial,
        Func<DirectPtxConvolutionVariant, double> benchmark,
        bool autotuneEnabled,
        IGpuTuningExchange? exchange = null)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));
        if (benchmark is null) throw new ArgumentNullException(nameof(benchmark));

        IReadOnlyList<AutotuneCandidate> candidates = CandidateConfigurations(
            outputChannels, inputChannels, spatial);
        AutotuneResolution resolution = CommunityAutotune.Resolve(
            exchange ?? GpuTuningExchangeProvider.Current,
            ConvTileAutotune.Category,
            ShareableKernelName,
            runtime.Fingerprint,
            ConvTileAutotune.Shape(batch, outputChannels, inputChannels, spatial),
            candidates,
            candidate =>
            {
                if (!TryGetVariant(
                        candidate, outputChannels, inputChannels, spatial,
                        out DirectPtxConvolutionVariant decoded))
                    throw new InvalidOperationException(
                        $"Unsupported convolution autotune candidate '{candidate.Variant}'.");
                return benchmark(decoded);
            },
            candidate => TryGetVariant(
                candidate, outputChannels, inputChannels, spatial, out _),
            autotuneEnabled);

        if (!TryGetVariant(
                new AutotuneCandidate(resolution.Variant, resolution.Parameters),
                outputChannels, inputChannels, spatial,
                out DirectPtxConvolutionVariant selected))
            throw new InvalidOperationException(
                $"Convolution autotune resolved unsupported variant '{resolution.Variant}'.");
        return selected;
    }

    internal static bool TryLoad(
        DirectPtxRuntime runtime,
        int batch,
        int outputChannels,
        int inputChannels,
        int spatial,
        out DirectPtxConvolutionVariant variant)
    {
        if (runtime is null) throw new ArgumentNullException(nameof(runtime));

        KernelId kernelId = GpuFirstRunAutotuner.GpuKernelId(
            ConvTileAutotune.Category, ShareableKernelName, runtime.Fingerprint);
        KernelChoice? cached = AutotuneCache.Lookup(
            kernelId,
            ConvTileAutotune.Shape(batch, outputChannels, inputChannels, spatial));
        if (cached is null)
        {
            variant = default;
            return false;
        }

        return TryGetVariant(
            new AutotuneCandidate(cached.Variant, cached.Parameters),
            outputChannels, inputChannels, spatial, out variant);
    }
}
