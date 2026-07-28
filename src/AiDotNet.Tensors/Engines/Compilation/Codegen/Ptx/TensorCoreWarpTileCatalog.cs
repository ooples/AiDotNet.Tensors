// Copyright (c) AiDotNet. All rights reserved.

using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ptx;

/// <summary>
/// Warp tiles measured to be fastest per GEMM shape, with a fallback for shapes never
/// measured.
/// </summary>
/// <remarks>
/// <para>
/// This exists because the obvious rule is wrong. "Use the largest tile that fits" is right
/// at 2048³ and 4096³ and wrong at 1024³, where the measured best was 4x2 at 71.9 us against
/// 4x4's 75.0 us. A static model picked lowerings four times on branch
/// <c>agent/direct-ptx-conv-promotion-841</c> and lost to the hardware every time it was
/// checked, so the shapes that have been measured are recorded as measurements rather than
/// re-derived from a rule that is known not to hold.
/// </para>
/// <para>
/// KEYED BY SHAPE, and the key is the only thing the lookup uses. An earlier autotune cache
/// on this campaign was written under a catalog name and read under a spec name -- they differ
/// for every depthwise entry -- so those kernels silently ran untuned while the cache reported
/// them tuned. A cache miss is indistinguishable from "the modelled choice already won", which
/// is why the miss path here is a documented ladder rather than silence.
/// </para>
/// <para>
/// Regenerate with:
/// <code>
/// dotnet run --project tests/AiDotNet.Tensors.Benchmarks -c Release -f net10.0 -- --warp-tile-sweep
/// </code>
/// on an idle GPU at locked clocks. Every candidate is verified against the fp64 oracle before
/// it is timed, so an entry here cannot be a tile that computes the wrong answer quickly.
/// </para>
/// </remarks>
public static class TensorCoreWarpTileCatalog
{
    /// <summary>Measured winners, keyed by (M, N, K).</summary>
    /// <remarks>
    /// Measured on an RTX 3080 (sm_86) at 1770 MHz, idle, best of three, fp16 operands with
    /// an fp32 accumulator. TFLOP/s for every candidate is in <c>docs/TENSOR_CORES.md</c>.
    /// </remarks>
    private static readonly Dictionary<(int M, int N, int K), Choice> Measured =
        new()
        {
            // 2x2 + cp.async at 19.5 us (13.8 TF). The SMALLEST tile wins here, and by a
            // wide margin -- at this size there is not enough work to amortise a big tile's
            // occupancy cost. Second best was 4x2 + registers at 28.1 us.
            [(512, 512, 512)] = new Choice(2, 2, AsyncCopy: true),

            // 4x2 + cp.async at 63.2 us (34.0 TF), against 4x4 + cp.async at 74.5 and
            // 4x2 + registers at 71.7. The largest tile LOSES here.
            [(1024, 1024, 1024)] = new Choice(4, 2, AsyncCopy: true),

            // 4x4 + cp.async at 400.2 us (42.9 TF), against 4x2 + cp.async at 422.3.
            [(2048, 2048, 2048)] = new Choice(4, 4, AsyncCopy: true),

            // 4x2 + cp.async at 2922.7 us (47.0 TF). Note 4x4 + cp.async is 41.6 TF while
            // 4x4 + REGISTERS is 45.0 -- the staging form and the tile interact, and not
            // monotonically. No model produced this; the sweep did.
            [(4096, 4096, 4096)] = new Choice(4, 2, AsyncCopy: true),
        };

    /// <summary>A measured lowering: the warp tile and the staging form together.</summary>
    /// <param name="TileM">wmma tiles a warp owns along M.</param>
    /// <param name="TileN">wmma tiles a warp owns along N.</param>
    /// <param name="AsyncCopy">Whether staging uses <c>cp.async</c> rather than registers.</param>
    /// <remarks>
    /// The two are recorded TOGETHER because they are not independent. At 4096^3 the 4x4 tile
    /// is faster with register staging (45.0 TF) than with cp.async (41.6 TF), while the 4x2
    /// tile is much faster with cp.async (47.0 TF) than without (38.4 TF). Choosing them
    /// separately would pick 4x4 + cp.async, which is the worst of those four.
    /// </remarks>
    public readonly record struct Choice(int TileM, int TileN, bool AsyncCopy);

    /// <summary>
    /// The measured winner for a shape, or the fallback ladder's choice when the shape has
    /// not been measured.
    /// </summary>
    /// <param name="m">Output rows.</param>
    /// <param name="n">Output columns.</param>
    /// <param name="k">Contracted extent.</param>
    /// <param name="measured">True when the answer came from a measurement.</param>
    public static Choice Select(int m, int n, int k, out bool measured)
    {
        if (Measured.TryGetValue((m, n, k), out var winner))
        {
            measured = true;
            return winner;
        }

        measured = false;

        // THE FALLBACK LADDER, for shapes nobody has measured: the largest tile whose block
        // tile divides the output, staged with cp.async. cp.async is the right default because
        // it won at 10 of the 12 measured (shape, tile) pairs, and its two losses were small;
        // the tile choice is the one that varies, which is what the entries above record.
        foreach (var (tileM, tileN) in new[] { (4, 2), (4, 4), (2, 4), (2, 2) })
            if (m % (tileM * 32) == 0 && n % (tileN * 32) == 0)
                return new Choice(tileM, tileN, AsyncCopy: true);

        return new Choice(2, 2, AsyncCopy: true);
    }

    /// <summary>Shapes with a measured entry. Exposed so a test can check they are reachable.</summary>
    public static IEnumerable<(int M, int N, int K)> MeasuredShapes => Measured.Keys;
}
