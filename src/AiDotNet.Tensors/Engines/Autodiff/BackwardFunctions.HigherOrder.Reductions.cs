using System;
using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    /// <summary>
    /// D[i, j] = ‖x_i - y_j‖² (PairwiseDistanceSquared of two point sets): dL/dx = 2(rowsum(G) ⊙ x - G·y) and
    /// dL/dy = 2(colsum(G) ⊙ y - Gᵀ·x), from recorded engine ops, so it is differentiable to any order.
    /// </summary>
    internal static void PairwiseDistanceSquaredTwoSetBackward(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var x = inputs[0];
        var y = inputs[1];
        var two = Helpers.MathHelper.GetNumericOperations<T>().FromDouble(2.0);
        if (DifferentiableOps.IsGradientRequired(x))
        {
            var rowSum = engine.ReduceSum(gradOutput, new[] { 1 }, keepDims: true);
            var gx = engine.TensorSubtract(engine.TensorMultiply(engine.TensorBroadcastTo(rowSum, x._shape), x),
                engine.TensorMatMul(gradOutput, y));
            DifferentiableOps.AccumulateGrad(grads, x, engine.TensorMultiplyScalar(gx, two), engine);
        }
        if (DifferentiableOps.IsGradientRequired(y))
        {
            var colSum = engine.Reshape(engine.ReduceSum(gradOutput, new[] { 0 }, keepDims: false), new[] { y._shape[0], 1 });
            var gy = engine.TensorSubtract(engine.TensorMultiply(engine.TensorBroadcastTo(colSum, y._shape), y),
                engine.TensorMatMul(engine.TensorTranspose(gradOutput), x));
            DifferentiableOps.AccumulateGrad(grads, y, engine.TensorMultiplyScalar(gy, two), engine);
        }
    }

    /// <summary>
    /// An embedding table's gradient, table[ids[k], j] += g[k, j], as a recorded sparse map (whose backward is the
    /// lookup), for create-graph backwards. Out-of-range ids contribute nothing, as in the kernels.
    /// </summary>
    internal static Tensor<T> EmbeddingGradRecorded(Tensor<T> gradOutput, long[] ids, int vocabSize, int embeddingDim)
    {
        var to = new List<int>(ids.Length * embeddingDim);
        var from = new List<int>(ids.Length * embeddingDim);
        for (int k = 0; k < ids.Length; k++)
        {
            long row = ids[k];
            if (row < 0 || row >= vocabSize) continue;
            for (int j = 0; j < embeddingDim; j++)
            {
                to.Add((int)row * embeddingDim + j);
                from.Add(k * embeddingDim + j);
            }
        }
        return SparseMap(gradOutput, new[] { vocabSize, embeddingDim }, to.ToArray(), from.ToArray(), null);
    }

    /// <summary>
    /// A gather's source gradient — source[(outer, indices[k], inner)] += g[(outer, k, inner)], the index flattened —
    /// as a recorded sparse map (whose backward is the gather), for create-graph backwards.
    /// </summary>
    internal static Tensor<T> GatherGradRecorded(Tensor<T> gradOutput, Tensor<int> indices, int axis, int[] sourceShape)
    {
        int ax = axis < 0 ? axis + sourceShape.Length : axis;
        var idx = indices.GetFlattenedData();
        int count = idx.Length, rows = sourceShape[ax];
        int outer = 1, inner = 1;
        for (int d = 0; d < ax; d++) outer *= sourceShape[d];
        for (int d = ax + 1; d < sourceShape.Length; d++) inner *= sourceShape[d];
        var to = new List<int>(outer * count * inner);
        var from = new List<int>(outer * count * inner);
        for (int o = 0; o < outer; o++)
            for (int k = 0; k < count; k++)
            {
                int row = idx[k];
                if (row < 0) row += rows;
                if (row < 0 || row >= rows) continue;
                for (int i = 0; i < inner; i++)
                {
                    to.Add((o * rows + row) * inner + i);
                    from.Add((o * count + k) * inner + i);
                }
            }
        return SparseMap(gradOutput, (int[])sourceShape.Clone(), to.ToArray(), from.ToArray(), null);
    }

    /// <summary>
    /// Under create-graph: the gradient of the population variance (or its square root) of the whole tensor, as
    /// recorded ops: mean((x - mean(x))²), and sqrt of it for std.
    /// </summary>
    internal static bool TryFullVarianceComposite(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, bool std)
    {
        var x = inputs[0];
        var outputShape = gradOutput._shape;
        return TryCompositeBackward(gradOutput, inputs, engine, grads, e =>
        {
            var all = new int[x.Rank];
            for (int d = 0; d < all.Length; d++) all[d] = d;
            var centered = e.TensorSubtract(x, e.TensorBroadcastTo(e.ReduceMean(x, all, keepDims: true), x._shape));
            var variance = e.ReduceMean(e.TensorMultiply(centered, centered), all, keepDims: true);
            return ToShape(e, std ? e.TensorSqrt(variance) : variance, outputShape);
        });
    }

    /// <summary>
    /// A 2-D bicubic grid sample as recorded engine ops, differentiable to any order in both the input and the grid —
    /// PyTorch's grid_sampler_2d bicubic: per axis the (unpadded) fractional pixel coordinate s = slope·g + offset,
    /// t = s - floor(s) (the floor a constant), Catmull-Rom (a = -0.75) weights as polynomials in t at the offsets
    /// {1+t, t, 1-t, 2-t}, and each of the 4x4 integer taps padded as <see cref="CpuEngine"/>'s forward pads it (a zero
    /// pad reads nothing), the tap values a recorded gather of the input.
    /// </summary>
    internal static Tensor<T> GridSampleBicubicRecorded(IEngine e, Tensor<T> input, Tensor<T> grid,
        GridSamplePadding padding, bool alignCorners)
    {
        var ops = Helpers.MathHelper.GetNumericOperations<T>();
        int n = input._shape[0], channels = input._shape[1], height = input._shape[2], width = input._shape[3];
        int outH = grid._shape[1], outW = grid._shape[2];
        int points = outH * outW;
        var g = RoIAlignPlan.ToDoubles(grid);
        var pointShape = new[] { n, outH, outW };
        var outShape = new[] { n, channels, outH, outW };
        var weightShape = new[] { n, 1, outH, outW };
        var size = new[] { width, height };

        // Per axis (x, y): the fraction t = slope·g + offset (constants per point) and the integer floor.
        var slope = new double[2][];
        var offset = new double[2][];
        var floor = new int[2][];
        var valid = new bool[n * points];
        for (int a = 0; a < 2; a++) { slope[a] = new double[n * points]; offset[a] = new double[n * points]; floor[a] = new int[n * points]; }
        for (int p = 0; p < n * points; p++)
        {
            valid[p] = true;
            for (int a = 0; a < 2; a++)
            {
                double normalized = g[p * 2 + a];
                double s = GridSourceIndex(normalized, size[a], GridSamplePadding.Zeros, alignCorners, out double ds);
                if (double.IsNaN(s) || double.IsInfinity(s) || s <= int.MinValue + 2 || s >= int.MaxValue - 2)
                {
                    // A non-finite or out-of-int-range point samples nothing (as in the forward): weights 0, no taps.
                    valid[p] = false;
                    continue;
                }
                int f = (int)Math.Floor(s);
                slope[a][p] = ds;
                offset[a][p] = s - ds * normalized - f;
                floor[a][p] = f;
            }
        }

        // The four Catmull-Rom weights of each axis as polynomials in its recorded fraction t.
        const double A = -0.75;
        Tensor<T> Inner(Tensor<T> d) => e.TensorAddScalar(e.TensorMultiply(
            e.TensorAddScalar(e.TensorMultiplyScalar(d, ops.FromDouble(A + 2)), ops.FromDouble(-(A + 3))),
            e.TensorMultiply(d, d)), ops.One);
        Tensor<T> Outer(Tensor<T> d) => e.TensorAddScalar(e.TensorMultiply(
            e.TensorAddScalar(e.TensorMultiply(
                e.TensorAddScalar(e.TensorMultiplyScalar(d, ops.FromDouble(A)), ops.FromDouble(-5 * A)), d), ops.FromDouble(8 * A)),
            d), ops.FromDouble(-4 * A));
        var weights = new Tensor<T>[2][];
        for (int a = 0; a < 2; a++)
        {
            var axis = e.Reshape(e.TensorSliceAxis(grid, 3, a), pointShape);
            var t = e.TensorAdd(e.TensorMultiply(axis, RoIAlignPlan.FromDoubles<T>(slope[a], pointShape)),
                RoIAlignPlan.FromDoubles<T>(offset[a], pointShape));
            var oneMinusT = OneMinus(e, t);
            weights[a] = new[]
            {
                Outer(e.TensorAddScalar(t, ops.One)),
                Inner(t),
                Inner(oneMinusT),
                Outer(e.TensorAddScalar(oneMinusT, ops.One)),
            };
        }

        int volume = height * width;
        var inputShape = (int[])input._shape.Clone();
        Tensor<T>? result = null;
        for (int ky = 0; ky < 4; ky++)
            for (int kx = 0; kx < 4; kx++)
            {
                var index = new int[n * channels * points];
                for (int b = 0; b < n; b++)
                    for (int q = 0; q < points; q++)
                    {
                        int p = b * points + q;
                        int xi = valid[p] ? CpuEngine.GridSampleBicubicTap(floor[0][p] - 1 + kx, width, padding, alignCorners) : -1;
                        int yi = valid[p] ? CpuEngine.GridSampleBicubicTap(floor[1][p] - 1 + ky, height, padding, alignCorners) : -1;
                        bool inside = xi >= 0 && yi >= 0;
                        for (int ch = 0; ch < channels; ch++)
                            index[(b * channels + ch) * points + q] = inside ? (b * channels + ch) * volume + yi * width + xi : -1;
                    }
                var gatherShape = (int[])outShape.Clone();
                var values = RecordedLinearMap("GridSampleBicubicGather", input,
                    v => GatherFlat(v, index, gatherShape), w => ScatterAddFlat(w, index, inputShape));
                var weight = e.Reshape(e.TensorMultiply(weights[1][ky], weights[0][kx]), weightShape);
                var term = e.TensorMultiply(values, e.TensorBroadcastTo(weight, outShape));
                result = result is null ? term : e.TensorAdd(result, term);
            }
        return result!;
    }

    /// <summary>
    /// Under create-graph: the gradient of a gated linear unit a ⊙ act(b) — a and b the two halves of the input along
    /// <paramref name="dim"/>, as the GeGLU / SwiGLU / ReGLU kernels split it — by differentiating it as recorded ops.
    /// Their engine backwards are host loops, so the second backward saw the gradient as a constant.
    /// </summary>
    internal static bool TryGatedLinearUnitComposite(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, int dim, Func<IEngine, Tensor<T>, Tensor<T>> activation)
    {
        var x = inputs[0];
        return TryCompositeBackward(gradOutput, inputs, engine, grads, e =>
        {
            int ax = dim < 0 ? dim + x.Rank : dim;
            int half = x._shape[ax] / 2;
            return e.TensorMultiply(e.TensorNarrow(x, ax, 0, half), activation(e, e.TensorNarrow(x, ax, half, half)));
        });
    }
}
