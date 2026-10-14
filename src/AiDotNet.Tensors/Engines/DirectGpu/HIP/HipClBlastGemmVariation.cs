using System;
using System.Threading;
using System.Threading.Tasks;
using AiDotNet.Evolution;
using AiDotNet.Tensors.Engines.DirectGpu.HIP.Kernels;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

/// <summary>
/// Variation for the CLBlast GEMMK 0 kernel on HIP (#1266). It mutates only the fields that kernel reads, over CLBlast's
/// own xgemm tuning ranges, and repairs every proposal to a configuration HIP can build, so no evaluation is spent on a
/// kernel identical to its parent or on one validation would reject. The generic OpenCL operator spends roughly 8 of its
/// 21 mutations on fields the K0 kernel ignores.
/// </summary>
internal sealed class HipClBlastGemmVariation : IVariationOperator<OpenClGemmConfiguration>
{
    private static readonly int[] Tiles = { 16, 32, 64, 128 };
    private static readonly int[] KTiles = { 16, 32 };
    private static readonly int[] ThreadDimensions = { 8, 16, 32 };
    private static readonly int[] VectorWidths = { 1, 2, 4 };
    private static readonly int[] Unrolls = { 1, 2 };
    private const int RepairAttempts = 64;
    private readonly GpuCapabilities _capabilities;

    internal HipClBlastGemmVariation(GpuCapabilities capabilities) => _capabilities = capabilities;

    public string Id => "hip-clblast-k0-variation";
    public string VersionHash => "1";

    public ValueTask<OpenClGemmConfiguration> ProposeAsync(
        EvolutionVariationContext<OpenClGemmConfiguration> context,
        CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        OpenClGemmConfiguration parent = context.Inspirations.Count > 0 && context.Random.NextDouble() < 0.3d
            ? context.Inspirations[context.Random.NextInt(context.Inspirations.Count)].Candidate.CanonicalGenome.Genome
            : context.Parent.Candidate.CanonicalGenome.Genome;
        parent = Canonical(parent);
        for (int attempt = 0; attempt < RepairAttempts; attempt++)
        {
            OpenClGemmConfiguration candidate = parent;
            int mutations = 1 + context.Random.NextInt(3);
            for (int i = 0; i < mutations; i++) candidate = Mutate(candidate, context.Random);
            if (!candidate.Equals(parent) && IsBuildable(candidate))
                return new ValueTask<OpenClGemmConfiguration>(candidate);
        }
        return new ValueTask<OpenClGemmConfiguration>(parent);
    }

    /// <summary>The genome with every field the K0 kernel ignores at its fixed value, so equal kernels are equal genomes.</summary>
    internal static OpenClGemmConfiguration Canonical(OpenClGemmConfiguration value) => value with
    {
        KernelTemplate = GemmKernelTemplate.ClBlastBaselineK0,
        UseDoubleBuffering = false,
        UseVectorizedLoads = false,
        UseSubgroupOps = false,
        UseTrueVectorLds = false,
        UseColumnMajorA = false,
        KReg = 1
    };

    private bool IsBuildable(OpenClGemmConfiguration candidate)
    {
        GemmConfig config = candidate.ToGemmConfig();
        return HipClBlastGemmKernels.Supports(config) &&
            DynamicGemmKernel.ValidateConfig(config, _capabilities.MaxWorkGroupSize, _capabilities.LocalMemoryBytes) is null;
    }

    private static OpenClGemmConfiguration Mutate(OpenClGemmConfiguration value, StableRandom random) => random.NextInt(14) switch
    {
        0 => value with { TileM = Pick(Tiles, random) },
        1 => value with { TileN = Pick(Tiles, random) },
        2 => value with { TileK = Pick(KTiles, random) },
        3 => value with { ThreadTileM = Pick(ThreadDimensions, random) },
        4 => value with { ThreadTileN = Pick(ThreadDimensions, random) },
        5 => value with { VectorWidthM = Pick(VectorWidths, random) },
        6 => value with { VectorWidthN = Pick(VectorWidths, random) },
        7 => value with { StrideM = !value.StrideM },
        8 => value with { StrideN = !value.StrideN },
        9 => value with { MdimaSize = Pick(ThreadDimensions, random) },
        10 => value with { NdimbSize = Pick(ThreadDimensions, random) },
        11 => value with { KUnroll = Pick(Unrolls, random) },
        12 => value with { CacheA = !value.CacheA },
        13 => value with { CacheB = !value.CacheB },
        _ => throw new InvalidOperationException()
    };

    private static int Pick(int[] values, StableRandom random) => values[random.NextInt(values.Length)];
}