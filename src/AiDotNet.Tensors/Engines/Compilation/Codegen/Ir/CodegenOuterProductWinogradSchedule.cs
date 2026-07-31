// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// One exact output-partition and shared-storage schedule for the outer-product
/// Winograd lowering.
/// </summary>
/// <remarks>
/// The smaller tile is a first-class measured candidate because shared-memory
/// occupancy alone does not expose more work when the launch grid has fewer CTAs
/// than the device has SMs. Search, identity, and replay consume this same finite
/// set so a targeted experiment cannot become an implicit dispatch default.
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
        });

    public CodegenOuterProductWinogradSchedule(
        int blockTiles, bool compactShared, int threadTileM = 8)
    {
        if (blockTiles is not (16 or 32))
            throw new ArgumentOutOfRangeException(
                nameof(blockTiles), "The measured Winograd tile count must be 16 or 32.");
        if (threadTileM is not (4 or 8) ||
            16 * (BlockM / threadTileM) * (blockTiles / 8) > 256)
            throw new ArgumentOutOfRangeException(
                nameof(threadTileM),
                "The measured Winograd M fragment must be 4 or 8 and fit one CTA.");
        BlockTiles = blockTiles;
        ThreadTileM = threadTileM;
        CompactShared = compactShared;
    }

    private const int BlockM = 32;
    public int BlockTiles { get; }
    public int ThreadTileM { get; }
    public bool CompactShared { get; }
    public int TilePartitions => 64 / BlockTiles;

    /// <summary>Stable name stored in autotune evidence and resolved by the conveyor.</summary>
    public string WinnerName => BaseName +
        (BlockTiles == 32 ? string.Empty : FormattableString.Invariant($"-t{BlockTiles}")) +
        (ThreadTileM == 8 ? string.Empty : FormattableString.Invariant($"-tm{ThreadTileM}")) +
        (CompactShared ? "-compact" : string.Empty);

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
