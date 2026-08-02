// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>An exact measured output/thread fragment for the split outer-product partial.</summary>
public sealed class CodegenTiledOuterProductSchedule
{
    private static readonly IReadOnlyList<CodegenTiledOuterProductSchedule> _searchSpace =
        Array.AsReadOnly(new[]
        {
            new CodegenTiledOuterProductSchedule(
                maximumTileM: 16, maximumTileN: 16,
                threadTileM: 2, threadTileN: 2,
                suffix: string.Empty, requireExactTile: false),
            new CodegenTiledOuterProductSchedule(
                maximumTileM: 32, maximumTileN: 16,
                threadTileM: 4, threadTileN: 2,
                suffix: "m32n16tm4tn2", requireExactTile: true),
        });

    private CodegenTiledOuterProductSchedule(
        int maximumTileM, int maximumTileN, int threadTileM, int threadTileN,
        string suffix, bool requireExactTile)
    {
        MaximumTileM = maximumTileM;
        MaximumTileN = maximumTileN;
        ThreadTileM = threadTileM;
        ThreadTileN = threadTileN;
        Suffix = suffix;
        RequireExactTile = requireExactTile;
    }

    public int MaximumTileM { get; }
    public int MaximumTileN { get; }
    public int ThreadTileM { get; }
    public int ThreadTileN { get; }
    public string Suffix { get; }
    public bool RequireExactTile { get; }
    public bool IsDefault => Suffix.Length == 0;

    public static CodegenTiledOuterProductSchedule Default => _searchSpace[0];
    public static IReadOnlyList<CodegenTiledOuterProductSchedule> SearchSpace => _searchSpace;

    public static CodegenTiledOuterProductSchedule FindForWinner(string? winner)
    {
        if (winner is not null)
            for (int i = 1; i < _searchSpace.Count; i++)
                if (winner.EndsWith(":" + _searchSpace[i].Suffix,
                        StringComparison.Ordinal))
                    return _searchSpace[i];
        return Default;
    }
}
