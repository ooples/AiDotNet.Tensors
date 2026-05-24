using System;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Utility for 2D (M, N) parallel partitioning. Given total M and N block
/// counts, flattens the 2D grid into a 1D work-item index that
/// <see cref="Helpers.CpuParallelSettings.ParallelForOrSerial"/> can dispatch.
///
/// <para>
/// Strategy integration deferred to G5 (AxisSelector); G4 lands the
/// arithmetic primitive with unit tests. The 2D grid is preferred over
/// M-axis when M alone has too few blocks to fill all cores (e.g., on a
/// 16-core machine with M-blocks=4, 2D gives 4×N_blocks parallelism).
/// </para>
/// </summary>
internal static class MN2DDriver
{
    /// <summary>
    /// Convert a flat work-item index into (M-block, N-block) coordinates.
    /// Total items = numMBlocks × numNBlocks. Work items are enumerated
    /// row-major: item 0 = (m=0, n=0), item 1 = (m=0, n=1), ...,
    /// item numNBlocks = (m=1, n=0), and so on.
    /// </summary>
    /// <param name="flatIndex">Flat work-item index in [0, numMBlocks × numNBlocks).</param>
    /// <param name="numNBlocks">Total N-block count (stride for the 2D layout).</param>
    /// <returns>(mBlock, nBlock) — block coordinates.</returns>
    public static (int MBlock, int NBlock) UnflattenIndex(int flatIndex, int numNBlocks)
    {
        if (flatIndex < 0) throw new ArgumentOutOfRangeException(nameof(flatIndex));
        if (numNBlocks <= 0) throw new ArgumentOutOfRangeException(nameof(numNBlocks));
        int mBlock = flatIndex / numNBlocks;
        int nBlock = flatIndex % numNBlocks;
        return (mBlock, nBlock);
    }

    /// <summary>
    /// Total work-item count for a 2D (numMBlocks, numNBlocks) grid.
    /// </summary>
    public static int TotalItems(int numMBlocks, int numNBlocks)
    {
        if (numMBlocks < 0) throw new ArgumentOutOfRangeException(nameof(numMBlocks));
        if (numNBlocks < 0) throw new ArgumentOutOfRangeException(nameof(numNBlocks));
        // CodeRabbit #366: int*int can wrap silently and corrupt scheduling.
        // Promote to long, check the int32 ceiling, then cast back.
        long total = (long)numMBlocks * numNBlocks;
        if (total > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(numMBlocks),
                $"TotalItems ({numMBlocks} × {numNBlocks} = {total}) exceeds Int32.MaxValue.");
        return (int)total;
    }

    /// <summary>
    /// Decide whether 2D MN-grid is preferred over 1D M-axis for the current
    /// shape. Heuristic from the spec: 2D when M alone would underutilize
    /// cores (numMBlocks * 2 &lt; procs) AND there are multiple N-blocks.
    /// </summary>
    /// <param name="numMBlocks">Total M-block count.</param>
    /// <param name="numNBlocks">Total N-block count.</param>
    /// <param name="procs">Available processor count.</param>
    public static bool ShouldUse2DGrid(int numMBlocks, int numNBlocks, int procs)
    {
        if (procs <= 1) return false;
        if (numNBlocks <= 1) return false;  // No N parallelism possible.
        // M alone is enough? 1D M-axis is simpler.
        if (numMBlocks * 2 >= procs) return false;
        return true;
    }
}
