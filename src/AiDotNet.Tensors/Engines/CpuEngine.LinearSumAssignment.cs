using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The batched minimum-cost assignment behind <see cref="IEngine.TensorLinearSumAssignment{T}"/>: SciPy's
/// shortest-augmenting-path solver (Crouse, "On implementing 2D rectangular assignment algorithms", IEEE TAES 2016;
/// scipy/optimize/_lsap.c), batch items in parallel. The device kernels
/// (<see cref="DirectGpu.NumericRecurrenceKernelTemplate"/>) run the same steps, one work-group per batch item.
/// </summary>
public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<int> TensorLinearSumAssignment<T>(Tensor<T> cost)
    {
        GraphMode.ThrowIfInferenceUnsupported(GraphCaptureLimitation.MixedElementTypes);
        var (batch, size) = LinearSumAssignmentShape(cost);
        // long, ulong and decimal costs can differ below double's 53-bit mantissa (2^53 + 1 and 2^53 are one double), so
        // they solve in decimal, which holds every long and ulong exactly; every other element type converts to double
        // exactly.
        var result = typeof(T) == typeof(long) || typeof(T) == typeof(ulong) || typeof(T) == typeof(decimal)
            ? LinearSumAssignmentExact(ToDecimals(cost), batch, size)
            : LinearSumAssignmentHost(RoIAlignPlan.ToDoubles(cost), batch, size);
        return new Tensor<int>(result, cost.Rank == 2 ? new[] { size } : new[] { batch, size });
    }

    private static decimal[] ToDecimals<T>(Tensor<T> cost)
    {
        var values = cost.GetFlattenedData();
        var result = new decimal[cost.Length];
        if (values is long[] longs)
            for (int i = 0; i < result.Length; i++) result[i] = longs[i];
        else if (values is ulong[] ulongs)
            for (int i = 0; i < result.Length; i++) result[i] = ulongs[i];
        else
            Array.Copy((decimal[])(object)values, result, result.Length);
        return result;
    }

    /// <summary>
    /// <see cref="LinearSumAssignmentHost"/> in decimal arithmetic, for long, ulong and decimal costs. The solver's duals and
    /// path costs stay within (2S + 2) times the largest cost magnitude, so the costs must satisfy
    /// <c>max|cost| * 4 (S + 1) &lt;= decimal.MaxValue</c> (every long or ulong cost does for S below 10^9).
    /// </summary>
    internal static int[] LinearSumAssignmentExact(decimal[] cost, int batch, int size)
    {
        var result = new int[batch * size];
        if (size == 0) return result;
        decimal maxAbs = 0m;
        foreach (decimal c in cost) maxAbs = Math.Max(maxAbs, Math.Abs(c));
        if (maxAbs > decimal.MaxValue / (4m * (size + 1m)))
            throw new ArgumentOutOfRangeException(nameof(cost),
                $"Assignment costs must satisfy max|cost| * 4 (S + 1) <= decimal.MaxValue; got max|cost| = {maxAbs} for S = {size}.");
        CpuParallelSettings.ParallelForOrSerial(0, batch, (long)batch * size * size * size, b =>
            SolveAssignmentExact(cost, b * size * size, size, result, b * size), deterministicSafe: true);
        return result;
    }

    /// <summary>Validates a [B, S, S] or [S, S] cost; returns (B, S).</summary>
    internal static (int Batch, int Size) LinearSumAssignmentShape<T>(Tensor<T> cost)
    {
        if (cost is null) throw new ArgumentNullException(nameof(cost));
        if (cost.Rank != 2 && cost.Rank != 3)
            throw new ArgumentException($"cost must be [B, S, S] or [S, S]; got rank {cost.Rank}.", nameof(cost));
        int rank = cost.Rank, size = cost._shape[rank - 1];
        if (cost._shape[rank - 2] != size)
            throw new ArgumentException($"cost must be square in its last two dimensions; got [{string.Join(", ", cost._shape)}].", nameof(cost));
        return (rank == 3 ? cost._shape[0] : 1, size);
    }

    /// <summary>The assignment of every batch item: result[b * S + i] is the column of row i.</summary>
    internal static int[] LinearSumAssignmentHost(double[] cost, int batch, int size)
    {
        var result = new int[batch * size];
        if (size == 0) return result;
        CpuParallelSettings.ParallelForOrSerial(0, batch, (long)batch * size * size * size, b =>
            SolveAssignment(cost, b * size * size, size, result, b * size), deterministicSafe: true);
        return result;
    }

    /// <summary>
    /// One S x S problem. A non-finite cost is replaced first: NaN and +inf by <c>big = (S + 1)(2 max|finite| + 1)</c>,
    /// which exceeds the cost difference between any two assignments of the finite entries, so an assignment uses one
    /// only where none can avoid it; -inf by <c>-big</c>. Then, per row, the shortest augmenting path over the reduced
    /// costs <c>minVal + c[i, j] - u[i] - v[j]</c>: the next column scanned is the unscanned one with the smallest path
    /// cost, ties to the lowest unassigned column, then the lowest column.
    /// </summary>
    private static void SolveAssignment(double[] cost, int offset, int size, int[] result, int resultOffset)
    {
        double maxAbs = 0;
        for (int k = 0; k < size * size; k++)
        {
            double c = cost[offset + k];
            if (!double.IsNaN(c) && !double.IsInfinity(c)) maxAbs = Math.Max(maxAbs, Math.Abs(c));
        }
        // Scale by an exact power of two before forming penalties or reduced costs. Otherwise finite
        // costs near MaxValue overflow, leaving an augmenting path without a predecessor.
        double scale = maxAbs > 1.0
            ? BitConverter.Int64BitsToDouble(BitConverter.DoubleToInt64Bits(maxAbs) & 0x7ff0000000000000L)
            : 1.0;
        double big = (size + 1.0) * (2 * (maxAbs / scale) + 1);
        var c2 = new double[size * size];
        for (int k = 0; k < c2.Length; k++)
        {
            double c = cost[offset + k];
            c2[k] = double.IsNaN(c) || double.IsPositiveInfinity(c) ? big : double.IsNegativeInfinity(c) ? -big : c / scale;
        }

        var u = new double[size];
        var v = new double[size];
        var spc = new double[size];
        var path = new int[size];
        var row4col = new int[size];
        var col4row = new int[size];
        var sr = new bool[size];
        var sc = new bool[size];
        row4col.AsSpan().Fill(-1);
        col4row.AsSpan().Fill(-1);
        path.AsSpan().Fill(-1);

        for (int curRow = 0; curRow < size; curRow++)
        {
            spc.AsSpan().Fill(double.PositiveInfinity);
            Array.Clear(sr, 0, size);
            Array.Clear(sc, 0, size);
            double minVal = 0;
            int i = curRow, sink = -1;
            while (sink == -1)
            {
                sr[i] = true;
                double best = double.PositiveInfinity;
                int bestKey = int.MaxValue, bestCol = -1;
                for (int j = 0; j < size; j++)
                {
                    if (sc[j]) continue;
                    double r = minVal + c2[i * size + j] - u[i] - v[j];
                    if (r < spc[j]) { path[j] = i; spc[j] = r; }
                    int key = (row4col[j] < 0 ? 0 : size) + j;
                    if (spc[j] < best || (spc[j] == best && key < bestKey)) { best = spc[j]; bestKey = key; bestCol = j; }
                }
                minVal = best;
                sc[bestCol] = true;
                if (row4col[bestCol] == -1) sink = bestCol;
                else i = row4col[bestCol];
            }

            u[curRow] += minVal;
            for (int r = 0; r < size; r++)
                if (sr[r] && r != curRow) u[r] += minVal - spc[col4row[r]];
            for (int j = 0; j < size; j++)
                if (sc[j]) v[j] -= minVal - spc[j];

            for (int j = sink; ;)
            {
                int row = path[j];
                row4col[j] = row;
                int previous = col4row[row];
                col4row[row] = j;
                j = previous;
                if (row == curRow) break;
            }
        }
        Array.Copy(col4row, 0, result, resultOffset, size);
    }

    /// <summary>
    /// <see cref="SolveAssignment"/>'s steps in decimal, with the same scan order and tie-breaks. Decimal has no
    /// non-finite values to replace or overflow to scale away (the caller bounds the magnitudes), and decimal.MaxValue
    /// stands for an unreached column's path cost.
    /// </summary>
    private static void SolveAssignmentExact(decimal[] cost, int offset, int size, int[] result, int resultOffset)
    {
        var u = new decimal[size];
        var v = new decimal[size];
        var spc = new decimal[size];
        var path = new int[size];
        var row4col = new int[size];
        var col4row = new int[size];
        var sr = new bool[size];
        var sc = new bool[size];
        row4col.AsSpan().Fill(-1);
        col4row.AsSpan().Fill(-1);
        path.AsSpan().Fill(-1);

        for (int curRow = 0; curRow < size; curRow++)
        {
            spc.AsSpan().Fill(decimal.MaxValue);
            Array.Clear(sr, 0, size);
            Array.Clear(sc, 0, size);
            decimal minVal = 0m;
            int i = curRow, sink = -1;
            while (sink == -1)
            {
                sr[i] = true;
                decimal best = decimal.MaxValue;
                int bestKey = int.MaxValue, bestCol = -1;
                for (int j = 0; j < size; j++)
                {
                    if (sc[j]) continue;
                    decimal r = minVal + cost[offset + i * size + j] - u[i] - v[j];
                    if (r < spc[j]) { path[j] = i; spc[j] = r; }
                    int key = (row4col[j] < 0 ? 0 : size) + j;
                    if (spc[j] < best || (spc[j] == best && key < bestKey)) { best = spc[j]; bestKey = key; bestCol = j; }
                }
                minVal = best;
                sc[bestCol] = true;
                if (row4col[bestCol] == -1) sink = bestCol;
                else i = row4col[bestCol];
            }

            u[curRow] += minVal;
            for (int r = 0; r < size; r++)
                if (sr[r] && r != curRow) u[r] += minVal - spc[col4row[r]];
            for (int j = 0; j < size; j++)
                if (sc[j]) v[j] -= minVal - spc[j];

            for (int j = sink; ;)
            {
                int row = path[j];
                row4col[j] = row;
                int previous = col4row[row];
                col4row[row] = j;
                j = previous;
                if (row == curRow) break;
            }
        }
        Array.Copy(col4row, 0, result, resultOffset, size);
    }
}
