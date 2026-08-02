// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// One exact output-partition and shared-storage schedule for the outer-product
/// Winograd lowering.
/// </summary>
/// <remarks>
/// Search, identity, and replay consume this same finite set so a targeted
/// experiment cannot become an implicit dispatch default.
/// </remarks>
public sealed class CodegenOuterProductWinogradSchedule
{
    private const string BaseName = "inline-outer-winograd-conv2d";

    private static readonly IReadOnlyList<CodegenOuterProductWinogradSchedule> _searchSpace =
        Array.AsReadOnly(new[]
        {
            new CodegenOuterProductWinogradSchedule(32, threadTileM: 8, compactShared: false),
            new CodegenOuterProductWinogradSchedule(32, threadTileM: 8, compactShared: true),
            new CodegenOuterProductWinogradSchedule(16, threadTileM: 8, compactShared: false),
            new CodegenOuterProductWinogradSchedule(16, threadTileM: 8, compactShared: true),
            new CodegenOuterProductWinogradSchedule(16, threadTileM: 4, compactShared: false),
            new CodegenOuterProductWinogradSchedule(16, threadTileM: 4, compactShared: true),

            // Keep each eight-tile group in two complete warps. Partition one has
            // only three physical groups, so it retires two whole outer-product
            // warps and omits the corresponding inverse transforms.
            new CodegenOuterProductWinogradSchedule(
                32, threadTileM: 8, compactShared: false, tileGroupWarps: true),
            new CodegenOuterProductWinogradSchedule(
                32, threadTileM: 8, compactShared: true, tileGroupWarps: true),

            // Also pack only physical input tiles into producer lanes. This composes
            // exact input transforms, whole-warp outer products, and exact output
            // transforms under the same true-FP32 contract.
            new CodegenOuterProductWinogradSchedule(
                32, threadTileM: 8, compactShared: false,
                denseVProducers: true, tileGroupWarps: true),
            new CodegenOuterProductWinogradSchedule(
                32, threadTileM: 8, compactShared: true,
                denseVProducers: true, tileGroupWarps: true),
        });

    public CodegenOuterProductWinogradSchedule(
        int blockTiles, bool compactShared, int threadTileM = 8,
        bool denseVProducers = false, bool tileGroupWarps = false)
    {
        if (blockTiles is not (16 or 32))
            throw new ArgumentOutOfRangeException(
                nameof(blockTiles), "The measured Winograd tile count must be 16 or 32.");
        if (threadTileM is not (4 or 8) ||
            16 * (32 / threadTileM) * (blockTiles / 8) > 256)
            throw new ArgumentOutOfRangeException(
                nameof(threadTileM),
                "The measured Winograd M fragment must be 4 or 8 and fit a 256-thread CTA.");
        if ((denseVProducers || tileGroupWarps) &&
            (blockTiles != 32 || threadTileM != 8))
            throw new ArgumentException(
                "Exact physical-tile schedules require the M32 x 32-tile geometry.");
        if (denseVProducers && !tileGroupWarps)
            throw new ArgumentException(
                "Dense V production requires whole-warp physical-tile ownership.");

        BlockTiles = blockTiles;
        ThreadTileM = threadTileM;
        CompactShared = compactShared;
        DenseVProducers = denseVProducers;
        TileGroupWarps = tileGroupWarps;
    }

    public int BlockTiles { get; }
    public int ThreadTileM { get; }
    public bool CompactShared { get; }
    public bool DenseVProducers { get; }
    public bool TileGroupWarps { get; }
    public int TilePartitions => 64 / BlockTiles;

    /// <summary>Stable name stored in autotune evidence and resolved by the conveyor.</summary>
    public string WinnerName => BaseName +
        (BlockTiles == 32 ? string.Empty : FormattableString.Invariant($"-t{BlockTiles}")) +
        (ThreadTileM == 8 ? string.Empty : FormattableString.Invariant($"-tm{ThreadTileM}")) +
        (CompactShared ? "-compact" : string.Empty) +
        (DenseVProducers ? "-dense-v" : string.Empty) +
        (TileGroupWarps ? "-tile-warps" : string.Empty);

    public static CodegenOuterProductWinogradSchedule Default => _searchSpace[0];
    public static IReadOnlyList<CodegenOuterProductWinogradSchedule> SearchSpace => _searchSpace;

    public static CodegenOuterProductWinogradSchedule? Find(string? winner)
    {
        if (string.IsNullOrWhiteSpace(winner)) return null;
        foreach (CodegenOuterProductWinogradSchedule schedule in _searchSpace)
            if (string.Equals(schedule.WinnerName, winner, StringComparison.Ordinal))
                return schedule;
        return null;
    }
}
