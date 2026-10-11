using System;
using AiDotNet.Tensors.Engines.Compilation;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>How the decoding scales of a <see cref="IEngine.TensorScaledMatMul{T}"/> operand are laid out (torch.nn.functional.ScalingType).</summary>
public enum ScalingRecipe
{
    /// <summary>One scale for the whole operand.</summary>
    TensorWise,
    /// <summary>One scale per row of the left operand [M, K] (scale [M] or [M, 1]); per column of the right operand [K, N] (scale [N] or [1, N]).</summary>
    RowWise,
    /// <summary>One scale per 16 consecutive elements along K (NVFP4): left [M, ceil(K/16)], right [N, ceil(K/16)].</summary>
    BlockWise1x16,
    /// <summary>One scale per 32 consecutive elements along K (MXFP8 / MXFP4): left [M, ceil(K/32)], right [N, ceil(K/32)].</summary>
    BlockWise1x32,
    /// <summary>One scale per 128 consecutive elements along K: left [M, ceil(K/128)], right [N, ceil(K/128)].</summary>
    BlockWise1x128,
    /// <summary>One scale per 128x128 tile: left [ceil(M/128), ceil(K/128)], right [ceil(N/128), ceil(K/128)].</summary>
    BlockWise128x128,
}

/// <summary>Physical layout of a block-scale tensor (torch.nn.functional.SwizzleType).</summary>
public enum ScaleSwizzle
{
    /// <summary>Row-major scale matrix.</summary>
    None,
    /// <summary>
    /// The 32x4x4 tiling block-scaled GEMMs read: the scale matrix padded to multiples of 128 rows and 4 columns, cut
    /// into 128x4 tiles stored tile after tile, each tile as [32][4][4] (row within a 32-row slab, slab, column).
    /// </summary>
    Swizzle32x4x4,
}

/// <summary>
/// torch.nn.functional.grouped_mm, scaled_mm and scaled_grouped_mm: the grouped GEMM of mixture-of-experts layers and
/// the block-scaled low-precision GEMM.
/// </summary>
/// <remarks>
/// <para>Grouped products are composed from <see cref="IEngine.TensorNarrow{T}"/>, <see cref="IEngine.TensorMatMul{T}"/>
/// and <see cref="IEngine.TensorConcatenate{T}"/>, so they are differentiable and run on whatever engine executes
/// them.</para>
/// <para>Scaled products take the operands as their decoded values (the library has no 8- or 4-bit element type) with
/// the decoding scales of torch's recipes: <c>mm(a * scale_a, b * scale_b) + bias</c>, which is the value torch
/// computes. Like torch's scaled_mm they are not recorded for autograd.</para>
/// </remarks>
public partial class CpuEngine
{
    /// <inheritdoc/>
    public virtual Tensor<T> TensorGroupedMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<int>? offsets = null, Tensor<T>? bias = null)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Rank is < 2 or > 3 || b.Rank is < 2 or > 3)
            throw new ArgumentException($"grouped_mm takes 2-D or 3-D operands; got ranks {a.Rank} and {b.Rank}.");
        if (offsets is not null) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput, "TensorGroupedMatMul");
        IEngine e = this;
        int[]? ends = offsets is null ? null : offsets.AsSpan().ToArray();

        Tensor<T> result;
        if (a.Rank == 3 && b.Rank == 3)
        {
            if (ends is not null) throw new ArgumentException("Two 3-D operands enumerate the groups themselves; offsets must be null.", nameof(offsets));
            if (a._shape[0] != b._shape[0] || a._shape[2] != b._shape[1])
                throw new ArgumentException($"Batched groups need [G, M, K] x [G, K, N]; got [{string.Join(", ", a._shape)}] x [{string.Join(", ", b._shape)}].");
            result = e.TensorBatchMatMul(a, b);
            if (bias is not null) result = AddGroupBias(e, result, bias);
            return result;
        }

        if (ends is null) throw new ArgumentException("A 2-D operand is jagged: offsets are required.", nameof(offsets));
        ValidateOffsets(ends);
        int groups = ends.Length;

        if (a.Rank == 2 && b.Rank == 3)
        {
            // Jagged rows: rows [ends[i-1], ends[i]) of a times b[i]. Rows past the last offset are zero.
            int k = a._shape[1], n = b._shape[2];
            if (b._shape[0] != groups || b._shape[1] != k) throw new ArgumentException($"Expected b of shape [{groups}, {k}, N]; got [{string.Join(", ", b._shape)}].", nameof(b));
            if (ends[groups - 1] > a._shape[0]) throw new ArgumentException("The last offset passes the rows of a.", nameof(offsets));
            var parts = new List<Tensor<T>>();
            int start = 0;
            for (int g = 0; g < groups; g++)
            {
                int rows = ends[g] - start;
                if (rows > 0)
                {
                    var part = e.TensorMatMul(e.TensorNarrow(a, 0, start, rows), e.Reshape(e.TensorNarrow(b, 0, g, 1), new[] { k, n }));
                    if (bias is not null) part = ((CpuEngine)e).TensorBroadcastAdd(part, GroupBias(e, bias, g, groups, n));
                    parts.Add(part);
                }
                start = ends[g];
            }
            if (start < a._shape[0]) parts.Add(new Tensor<T>(new[] { a._shape[0] - start, n }));
            return parts.Count == 1 ? parts[0] : e.TensorConcatenate(parts.ToArray(), 0);
        }

        if (a.Rank == 2 && b.Rank == 2)
        {
            // Jagged K (the weight gradient of a grouped layer): out[i] = a[:, Ki] @ b[Ki, :].
            int m = a._shape[0], n = b._shape[1];
            if (a._shape[1] != b._shape[0]) throw new ArgumentException("a's columns and b's rows are the shared, jagged dimension and must match.", nameof(b));
            if (ends[groups - 1] > a._shape[1]) throw new ArgumentException("The last offset passes the shared dimension.", nameof(offsets));
            var parts = new Tensor<T>[groups];
            int start = 0;
            for (int g = 0; g < groups; g++)
            {
                int len = ends[g] - start;
                var part = len > 0
                    ? e.TensorMatMul(e.TensorNarrow(a, 1, start, len), e.TensorNarrow(b, 0, start, len))
                    : new Tensor<T>(new[] { m, n });
                if (bias is not null) part = ((CpuEngine)e).TensorBroadcastAdd(part, GroupBias(e, bias, g, groups, n));
                parts[g] = e.Reshape(part, new[] { 1, m, n });
                start = ends[g];
            }
            return groups == 1 ? parts[0] : e.TensorConcatenate(parts, 0);
        }

        // a 3-D, b 2-D: jagged N. Columns [ends[i-1], ends[i]) of the result are a[i] @ b[:, Ni].
        {
            int m = a._shape[1], k = a._shape[2];
            if (a._shape[0] != groups || b._shape[0] != k) throw new ArgumentException($"Expected a [{groups}, M, {k}] and b [{k}, N]; got [{string.Join(", ", a._shape)}] and [{string.Join(", ", b._shape)}].");
            if (ends[groups - 1] > b._shape[1]) throw new ArgumentException("The last offset passes the columns of b.", nameof(offsets));
            var parts = new List<Tensor<T>>();
            int start = 0;
            for (int g = 0; g < groups; g++)
            {
                int cols = ends[g] - start;
                if (cols > 0)
                {
                    var part = e.TensorMatMul(e.Reshape(e.TensorNarrow(a, 0, g, 1), new[] { m, k }), e.TensorNarrow(b, 1, start, cols));
                    if (bias is not null) part = ((CpuEngine)e).TensorBroadcastAdd(part, e.TensorNarrow(GroupBias(e, bias, g, groups, -1), 0, 0, cols));
                    parts.Add(part);
                }
                start = ends[g];
            }
            if (start < b._shape[1]) parts.Add(new Tensor<T>(new[] { m, b._shape[1] - start }));
            return parts.Count == 1 ? parts[0] : e.TensorConcatenate(parts.ToArray(), 1);
        }
    }

    private static void ValidateOffsets(int[] ends)
    {
        if (ends.Length == 0) throw new ArgumentException("offsets must name at least one group.", "offsets");
        int prev = 0;
        foreach (int o in ends)
        {
            if (o < prev) throw new ArgumentException("offsets must be non-decreasing and non-negative.", "offsets");
            prev = o;
        }
    }

    // A grouped bias is [N] (shared by every group) or [G, N] (one row per group). Checked up front so a mismatch
    // names the expected shape instead of failing later inside a narrow or a broadcast. n < 0 means the caller's
    // per-group output width varies (jagged columns) and is enforced by the caller's own narrow.
    private static void ValidateGroupBias<T>(Tensor<T> bias, int groups, int n)
    {
        bool ok = bias.Rank switch
        {
            1 => n < 0 || bias._shape[0] == n,
            2 => bias._shape[0] == groups && (n < 0 || bias._shape[1] == n),
            _ => false,
        };
        if (!ok)
            throw new ArgumentException(
                $"A grouped bias is [{(n < 0 ? "N" : n.ToString())}] or [{groups}, {(n < 0 ? "N" : n.ToString())}]; got [{string.Join(", ", bias._shape)}].",
                nameof(bias));
    }

    // The bias row for group g.
    private static Tensor<T> GroupBias<T>(IEngine e, Tensor<T> bias, int g, int groups, int n)
    {
        ValidateGroupBias(bias, groups, n);
        return bias.Rank == 1 ? bias : e.Reshape(e.TensorNarrow(bias, 0, g, 1), new[] { bias._shape[1] });
    }

    private static Tensor<T> AddGroupBias<T>(IEngine e, Tensor<T> result, Tensor<T> bias)
    {
        // result [G, M, N]; bias [N] broadcasts directly, [G, N] becomes [G, 1, N].
        ValidateGroupBias(bias, result._shape[0], result._shape[2]);
        var b = bias.Rank == 2 ? e.Reshape(bias, new[] { bias._shape[0], 1, bias._shape[1] }) : bias;
        return ((CpuEngine)e).TensorBroadcastAdd(result, b);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorScaledMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> scaleA, ScalingRecipe recipeA,
        Tensor<T> scaleB, ScalingRecipe recipeB, ScaleSwizzle swizzleA = ScaleSwizzle.None, ScaleSwizzle swizzleB = ScaleSwizzle.None,
        Tensor<T>? bias = null)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 2 || b.Rank != 2) throw new ArgumentException("scaled_mm multiplies two matrices.");
        if (a._shape[1] != b._shape[0]) throw new ArgumentException($"Inner dimensions differ: {a._shape[1]} and {b._shape[0]}.");
        IEngine e = this;
        Tensor<T> result;
        using (new Autodiff.NoGradScope<T>())
        {
            var left = Dequantize(e, a, scaleA, recipeA, swizzleA, rightOperand: false);
            var right = Dequantize(e, b, scaleB, recipeB, swizzleB, rightOperand: true);
            result = e.TensorMatMul(left, right);
            if (bias is not null) result = ((CpuEngine)e).TensorBroadcastAdd(result, bias);
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorScaledGroupedMatMul<T>(Tensor<T> a, Tensor<T> b, Tensor<T> scaleA, ScalingRecipe recipeA,
        Tensor<T> scaleB, ScalingRecipe recipeB, ScaleSwizzle swizzleA = ScaleSwizzle.None, ScaleSwizzle swizzleB = ScaleSwizzle.None,
        Tensor<T>? bias = null, Tensor<int>? offsets = null)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput, "TensorScaledGroupedMatMul");
        IEngine e = this;
        using (new Autodiff.NoGradScope<T>())
        {
            bool jaggedK = a.Rank == 2 && b.Rank == 2;
            if (jaggedK)
            {
                // Both operands: DequantizeGrouped decodes every non-tensor-wise jagged-K operand as per-group row-wise
                // and ignores its swizzle, so a block-wise recipeB whose length happened to equal G * N, or any
                // swizzle, would be applied silently wrong rather than rejected.
                foreach (var recipe in new[] { recipeA, recipeB })
                    if (recipe != ScalingRecipe.TensorWise && recipe != ScalingRecipe.RowWise)
                        throw new NotSupportedException("A jagged contraction dimension takes tensor-wise or per-group row-wise scales; block scales would straddle group boundaries.");
                if (swizzleA != ScaleSwizzle.None || swizzleB != ScaleSwizzle.None)
                    throw new NotSupportedException("Swizzled scales apply to block-scaled recipes only, which a jagged contraction dimension does not take.");
            }
            var left = DequantizeGrouped(e, a, scaleA, recipeA, swizzleA, rightOperand: false, jaggedK ? offsets : null);
            var right = DequantizeGrouped(e, b, scaleB, recipeB, swizzleB, rightOperand: true, jaggedK ? offsets : null);
            return e.TensorGroupedMatMul(left, right, offsets, bias);
        }
    }

    // A grouped operand: a 3-D operand is G matrices, each with its slice of the scales; a 2-D one is one matrix, except
    // with a jagged contraction dimension, where row-wise scales are per group ([G, M] for a, [G, N] for b).
    // Everything is engine ops: the scales are never read on the host, so a GPU engine keeps them on the device.
    private static Tensor<T> DequantizeGrouped<T>(IEngine e, Tensor<T> x, Tensor<T> scale, ScalingRecipe recipe, ScaleSwizzle swizzle,
        bool rightOperand, Tensor<int>? jaggedKOffsets)
    {
        if (x.Rank == 2 && jaggedKOffsets is null) return Dequantize(e, x, scale, recipe, swizzle, rightOperand);
        if (x.Rank == 2)
        {
            if (recipe == ScalingRecipe.TensorWise) return Dequantize(e, x, scale, recipe, swizzle, rightOperand);
            // Row-wise per group along the jagged K: a [M, Ktot] with scale [G, M]; b [Ktot, N] with scale [G, N].
            var ends = jaggedKOffsets!.AsSpan().ToArray();
            int rows = x._shape[0], cols = x._shape[1], per = rightOperand ? cols : rows;
            var scales = e.Reshape(scale, new[] { ends.Length, per });
            var parts = new List<Tensor<T>>();
            int start = 0, kTotal = rightOperand ? rows : cols;
            for (int g = 0; g < ends.Length; g++)
            {
                int len = ends[g] - start;
                if (len > 0)
                {
                    var row = e.TensorNarrow(scales, 0, g, 1);                                   // [1, per]
                    var slice = rightOperand ? e.TensorNarrow(x, 0, start, len) : e.TensorNarrow(x, 1, start, len);
                    var s = rightOperand
                        ? e.TensorBroadcastTo(row, new[] { len, per })                              // scale each column of b
                        : e.TensorBroadcastTo(e.Reshape(row, new[] { per, 1 }), new[] { per, len }); // scale each row of a
                    parts.Add(e.TensorMultiply(slice, s));
                }
                start = ends[g];
            }
            if (start < kTotal)
                parts.Add(rightOperand ? e.TensorNarrow(x, 0, start, kTotal - start) : e.TensorNarrow(x, 1, start, kTotal - start));
            return parts.Count == 1 ? parts[0] : e.TensorConcatenate(parts.ToArray(), rightOperand ? 0 : 1);
        }
        // 3-D: G matrices; the scales have a leading G axis (none for tensor-wise).
        int groups = x._shape[0];
        var slices = new Tensor<T>[groups];
        int perGroup = recipe == ScalingRecipe.TensorWise ? 0 : scale.Length / groups;
        var flatScale = recipe == ScalingRecipe.TensorWise ? scale : e.Reshape(scale, new[] { scale.Length });
        for (int g = 0; g < groups; g++)
        {
            var mat = e.Reshape(e.TensorNarrow(x, 0, g, 1), new[] { x._shape[1], x._shape[2] });
            var s = recipe == ScalingRecipe.TensorWise ? scale : e.TensorNarrow(flatScale, 0, g * perGroup, perGroup);
            slices[g] = e.Reshape(Dequantize(e, mat, s, recipe, swizzle, rightOperand), new[] { 1, x._shape[1], x._shape[2] });
        }
        return groups == 1 ? slices[0] : e.TensorConcatenate(slices, 0);
    }

    /// <summary>
    /// <paramref name="x"/> times its decoding scales. The left operand is [M, K]; the right operand is [K, N] and its
    /// block scales are laid out over its transpose [N, K], as torch passes it (column-major). Engine ops only.
    /// </summary>
    private static Tensor<T> Dequantize<T>(IEngine e, Tensor<T> x, Tensor<T> scale, ScalingRecipe recipe, ScaleSwizzle swizzle, bool rightOperand)
    {
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        int rows = x._shape[0], cols = x._shape[1];
        if (recipe == ScalingRecipe.TensorWise)
        {
            if (scale.Length != 1) throw new ArgumentException("A tensor-wise scale is a single value.", nameof(scale));
            return e.TensorMultiply(x, e.TensorBroadcastTo(e.Reshape(scale, new[] { 1, 1 }), x._shape));
        }
        // The operand as [outer, K]: the left operand's rows, the right operand's columns.
        int outer = rightOperand ? cols : rows, k = rightOperand ? rows : cols;
        int blockOuter, blockK;
        switch (recipe)
        {
            case ScalingRecipe.RowWise: blockOuter = 1; blockK = k; break;
            case ScalingRecipe.BlockWise1x16: blockOuter = 1; blockK = 16; break;
            case ScalingRecipe.BlockWise1x32: blockOuter = 1; blockK = 32; break;
            case ScalingRecipe.BlockWise1x128: blockOuter = 1; blockK = 128; break;
            case ScalingRecipe.BlockWise128x128: blockOuter = 128; blockK = 128; break;
            default: throw new ArgumentOutOfRangeException(nameof(recipe));
        }
        int scaleRows = (outer + blockOuter - 1) / blockOuter, scaleCols = (k + blockK - 1) / blockK;
        var logical = LogicalScales(e, scale, scaleRows, scaleCols, swizzle);                 // [scaleRows, scaleCols]
        // Expand every scale over its block: [sr, 1, sc, 1] -> [sr, bo, sc, bk] -> [sr*bo, sc*bk], then cut to [outer, K].
        var expanded = e.Reshape(
            e.TensorBroadcastTo(e.Reshape(logical, new[] { scaleRows, 1, scaleCols, 1 }), new[] { scaleRows, blockOuter, scaleCols, blockK }),
            new[] { scaleRows * blockOuter, scaleCols * blockK });
        if (scaleRows * blockOuter != outer) expanded = e.TensorNarrow(expanded, 0, 0, outer);
        if (scaleCols * blockK != k) expanded = e.TensorNarrow(expanded, 1, 0, k);
        var full = rightOperand ? e.TensorTranspose(expanded) : expanded;                       // to x's own [rows, cols]
        return e.TensorMultiply(x, full);
    }

    /// <summary>The [rows, cols] scale matrix from its stored layout, by engine ops (a gather for the swizzled one).</summary>
    private static Tensor<T> LogicalScales<T>(IEngine e, Tensor<T> scale, int rows, int cols, ScaleSwizzle swizzle)
    {
        if (swizzle == ScaleSwizzle.None)
        {
            if (scale.Length != rows * cols)
                throw new ArgumentException($"Expected {rows * cols} scales ([{rows}, {cols}]); got {scale.Length}.", nameof(scale));
            return e.Reshape(scale, new[] { rows, cols });
        }
        // 32x4x4: tiles of 128 rows x 4 columns over the padded matrix, row-major over tiles; inside a tile the element
        // (r, c) sits at (r % 32) * 16 + (r / 32) * 4 + c. The positions depend only on the shape, never on the data.
        int tileCols = (cols + 3) / 4, tileRows = (rows + 127) / 128;
        if (scale.Length < tileRows * tileCols * 512)
            throw new ArgumentException($"A 32x4x4-swizzled [{rows}, {cols}] scale matrix has {tileRows * tileCols * 512} entries; got {scale.Length}.", nameof(scale));
        var positions = new int[rows * cols];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
            {
                int tile = (r / 128) * tileCols + c / 4;
                int rr = r % 128, cc = c % 4;
                positions[r * cols + c] = tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + cc;
            }
        return e.TensorTake(scale, new Tensor<int>(positions, new[] { rows, cols }));
    }
}
