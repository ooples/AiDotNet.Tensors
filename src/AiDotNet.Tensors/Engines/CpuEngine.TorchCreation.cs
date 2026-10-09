using AiDotNet.Tensors.Engines.Compilation;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The PyTorch creation, window, index, shape and statistics ops (see the matching <see cref="IEngine"/> members).
/// Shape and statistics ops are composed from recorded ops, so they differentiate through them; quantile records its
/// own backward; creation, random, window and index ops are constants.
/// </summary>
public partial class CpuEngine
{
    private static int NormalizeDim(int dim, int rank)
    {
        int d = dim < 0 ? dim + rank : dim;
        if (d < 0 || d >= rank) throw new ArgumentOutOfRangeException(nameof(dim), $"dim {dim} is out of range for rank {rank}.");
        return d;
    }

    private static int[] AllAxes(int rank) => Enumerable.Range(0, rank).ToArray();

    private static Random RandomSource(int? seed)
        => seed.HasValue ? RandomHelper.CreateSeededRandom(seed.Value) : RandomHelper.CreateSecureRandom();

    private static Tensor<T> FromDoubles<T>(int[] shape, Func<int, double> value)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        var result = new Tensor<T>(shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < dst.Length; i++) dst[i] = ops.FromDouble(value(i));
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAmin<T>(Tensor<T> tensor, int[] axes, bool keepDims = false)
        => TensorNegate(ReduceMax(TensorNegate(tensor), axes, keepDims));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAll<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return Truthiness(tensor, axes, keepDims, all: true);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAny<T>(Tensor<T> tensor, int[]? axes = null, bool keepDims = false)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return Truthiness(tensor, axes, keepDims, all: false);
    }

    // all = min over the 0/1 indicator, any = max over it; constants, so no gradient is recorded.
    private Tensor<T> Truthiness<T>(Tensor<T> tensor, int[]? axes, bool keepDims, bool all)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        var source = tensor.IsContiguous ? tensor : tensor.Contiguous();
        var values = source.AsSpan().ToArray();
        var indicator = FromDoubles<T>((int[])source._shape.Clone(), i => ops.Equals(values[i], ops.Zero) ? 0 : 1);
        var reduceAxes = axes ?? AllAxes(tensor.Rank);
        using (new NoGradScope<T>())
            return all ? TensorAmin(indicator, reduceAxes, keepDims) : ReduceMax(indicator, reduceAxes, keepDims);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorArange<T>(double start, double end, double step = 1)
    {
        int count = RangeCount(start, end, step, Math.Ceiling((end - start) / step));
        return FromDoubles<T>(new[] { count }, i => start + i * step);
    }

    private static int RangeCount(double start, double end, double step, double size)
    {
        if (double.IsNaN(start) || double.IsInfinity(start)) throw new ArgumentOutOfRangeException(nameof(start), "start must be finite.");
        if (double.IsNaN(end) || double.IsInfinity(end)) throw new ArgumentOutOfRangeException(nameof(end), "end must be finite.");
        if (step == 0 || double.IsNaN(step) || double.IsInfinity(step)) throw new ArgumentOutOfRangeException(nameof(step), "step must be finite and non-zero.");
        if ((end - start) * step < 0) throw new ArgumentException("end must lie in the direction of step from start.");
        if (size > int.MaxValue) throw new ArgumentOutOfRangeException(nameof(step), $"the range would hold {size} elements, more than a tensor axis can.");
        return (int)size;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRange<T>(double start, double end, double step = 1)
    {
        // PyTorch: size = (int64)((end - start) / step + 1), truncating with no slack, so range(0, 0.3, 0.1) has 3
        // elements because 0.3 / 0.1 rounds to 2.9999999999999996.
        int count = RangeCount(start, end, step, Math.Floor((end - start) / step + 1));
        return FromDoubles<T>(new[] { count }, i => start + i * step);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogspace<T>(double start, double end, int steps, double logBase = 10)
    {
        if (steps < 0) throw new ArgumentOutOfRangeException(nameof(steps));
        return FromDoubles<T>(new[] { steps }, i => Math.Pow(logBase, steps == 1 ? start : start + i * (end - start) / (steps - 1)));
    }

    private static int[] LikeShape<T>(Tensor<T> tensor)
        => (int[])(tensor ?? throw new ArgumentNullException(nameof(tensor)))._shape.Clone();

    /// <inheritdoc/>
    public virtual Tensor<T> TensorZerosLike<T>(Tensor<T> tensor) => new Tensor<T>(LikeShape(tensor));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorOnesLike<T>(Tensor<T> tensor) => TensorFullLike(tensor, MathHelper.GetNumericOperations<T>().One);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFullLike<T>(Tensor<T> tensor, T value)
    {
        var result = new Tensor<T>(LikeShape(tensor));
        result.AsWritableSpan().Fill(value);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorEmptyLike<T>(Tensor<T> tensor) => TensorZerosLike(tensor);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRandLike<T>(Tensor<T> tensor, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        var rng = RandomSource(seed);
        return FromDoubles<T>(LikeShape(tensor), _ => rng.NextDouble());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRandnLike<T>(Tensor<T> tensor, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        var rng = RandomSource(seed);
        // Box–Muller; 1 - U keeps the logarithm's argument in (0, 1].
        return FromDoubles<T>(LikeShape(tensor),
            _ => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble()));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRandint<T>(long low, long high, int[] shape, int? seed = null)
    {
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        if (high <= low) throw new ArgumentException("high must be greater than low.", nameof(high));
        var rng = RandomSource(seed);
        double span = high - (double)low;
        return FromDoubles<T>((int[])shape.Clone(), _ => low + Math.Min(Math.Floor(rng.NextDouble() * span), span - 1));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRandintLike<T>(Tensor<T> tensor, long low, long high, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return TensorRandint<T>(low, high, LikeShape(tensor), seed);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRandperm<T>(int n, int? seed = null)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
        var rng = RandomSource(seed);
        var order = Enumerable.Range(0, n).ToArray();
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }
        return FromDoubles<T>(new[] { n }, i => order[i]);
    }

    // A symmetric window of M points, or its periodic form (the symmetric one of N + 1 points, last point dropped).
    private static Tensor<T> Window<T>(int length, bool periodic, Func<double, double> shape)
    {
        if (length < 0) throw new ArgumentOutOfRangeException(nameof(length));
        if (length == 1) return FromDoubles<T>(new[] { 1 }, _ => 1);
        int m = periodic ? length + 1 : length;
        // shape takes the position t = n / (M - 1) in [0, 1].
        return FromDoubles<T>(new[] { length }, n => shape(n / (double)(m - 1)));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHannWindow<T>(int length, bool periodic = true)
        => Window<T>(length, periodic, t => 0.5 - 0.5 * Math.Cos(2 * Math.PI * t));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHammingWindow<T>(int length, bool periodic = true, double alpha = 0.54, double beta = 0.46)
        => Window<T>(length, periodic, t => alpha - beta * Math.Cos(2 * Math.PI * t));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBlackmanWindow<T>(int length, bool periodic = true)
        => Window<T>(length, periodic, t => 0.42 - 0.5 * Math.Cos(2 * Math.PI * t) + 0.08 * Math.Cos(4 * Math.PI * t));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBartlettWindow<T>(int length, bool periodic = true)
        => Window<T>(length, periodic, t => 1 - Math.Abs(2 * t - 1));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorKaiserWindow<T>(int length, bool periodic = true, double beta = 12)
    {
        double denominator = SpecialFunctions.BesselI(0, beta);
        return Window<T>(length, periodic, t =>
        {
            double r = 2 * t - 1;
            return SpecialFunctions.BesselI(0, beta * Math.Sqrt(Math.Max(0, 1 - r * r))) / denominator;
        });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTrilIndices<T>(int row, int col, int offset = 0)
        => TriangleIndices<T>(row, col, (i, j) => j - i <= offset);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTriuIndices<T>(int row, int col, int offset = 0)
        => TriangleIndices<T>(row, col, (i, j) => j - i >= offset);

    private static Tensor<T> TriangleIndices<T>(int row, int col, Func<int, int, bool> keep)
    {
        if (row < 0 || col < 0) throw new ArgumentOutOfRangeException(nameof(row), "row and col must be non-negative.");
        var rows = new List<int>();
        var cols = new List<int>();
        for (int i = 0; i < row; i++)
            for (int j = 0; j < col; j++)
                if (keep(i, j)) { rows.Add(i); cols.Add(j); }
        int n = rows.Count;
        return FromDoubles<T>(new[] { 2, n }, k => k < n ? rows[k] : cols[k - n]);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorUnravelIndex<T>(Tensor<T> indices, int[] shape)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        var ops = MathHelper.GetNumericOperations<T>();
        long total = shape.Aggregate(1L, (a, d) => a * d);
        var source = indices.IsContiguous ? indices : indices.Contiguous();
        var flat = source.AsSpan().ToArray().Select(v => (long)ops.ToDouble(v)).ToArray();
        foreach (var f in flat)
            if (f < -total || f >= total)
                throw new ArgumentOutOfRangeException(nameof(indices), $"index {f} is out of range for shape [{string.Join(", ", shape)}].");
        var result = new Tensor<T>[shape.Length];
        long stride = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            long s = stride, extent = shape[d];
            result[d] = FromDoubles<T>((int[])source._shape.Clone(), i => ((flat[i] + total) % total) / s % extent);
            stride *= shape[d];
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCombinations<T>(Tensor<T> tensor, int r = 2, bool withReplacement = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank != 1) throw new ArgumentException("TensorCombinations expects a 1-D tensor.", nameof(tensor));
        if (r < 0) throw new ArgumentOutOfRangeException(nameof(r));
        int n = tensor._shape[0];
        var picks = new List<int>();
        var current = new int[r];
        void Visit(int position, int from)
        {
            if (position == r) { picks.AddRange(current); return; }
            for (int i = from; i < n; i++) { current[position] = i; Visit(position + 1, withReplacement ? i : i + 1); }
        }
        Visit(0, 0);
        int combos = r == 0 ? 0 : picks.Count / r;
        if (combos == 0) return new Tensor<T>(new[] { 0, r });
        // Gathered through the recorded take, so a gradient reaches the picked elements.
        var index = new Tensor<int>(picks.ToArray(), new[] { combos, r });
        return TensorTake(tensor, index);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorChunk<T>(Tensor<T> tensor, int chunks, int dim = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (chunks <= 0) throw new ArgumentOutOfRangeException(nameof(chunks), "chunks must be positive.");
        int d = NormalizeDim(dim, tensor.Rank), extent = tensor._shape[d];
        int size = Math.Max(1, (extent + chunks - 1) / chunks);
        var sizes = new List<int>();
        for (int start = 0; start < extent; start += size) sizes.Add(Math.Min(size, extent - start));
        return TensorSplitWithSizes(tensor, sizes.ToArray(), d);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorSplitWithSizes<T>(Tensor<T> tensor, int[] sizes, int dim = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (sizes == null) throw new ArgumentNullException(nameof(sizes));
        int d = NormalizeDim(dim, tensor.Rank);
        if (sizes.Sum() != tensor._shape[d])
            throw new ArgumentException($"sizes sum to {sizes.Sum()} but dim {d} has {tensor._shape[d]} elements.", nameof(sizes));
        var pieces = new Tensor<T>[sizes.Length];
        int start = 0;
        for (int i = 0; i < sizes.Length; i++) { pieces[i] = TensorNarrow(tensor, d, start, sizes[i]); start += sizes[i]; }
        return pieces;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorUnflatten<T>(Tensor<T> tensor, int dim, int[] sizes)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (sizes == null) throw new ArgumentNullException(nameof(sizes));
        int d = NormalizeDim(dim, tensor.Rank);
        var resolved = (int[])sizes.Clone();
        int inferred = Array.IndexOf(resolved, -1);
        int known = resolved.Where(s => s != -1).Aggregate(1, (a, b) => a * b);
        if (inferred >= 0) resolved[inferred] = tensor._shape[d] / known;
        if (resolved.Aggregate(1, (a, b) => a * b) != tensor._shape[d])
            throw new ArgumentException($"sizes [{string.Join(", ", sizes)}] do not multiply to {tensor._shape[d]}.", nameof(sizes));
        var shape = tensor._shape.Take(d).Concat(resolved).Concat(tensor._shape.Skip(d + 1)).ToArray();
        return Reshape(tensor, shape);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSelect<T>(Tensor<T> tensor, int dim, int index)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int d = NormalizeDim(dim, tensor.Rank), extent = tensor._shape[d];
        int i = index < 0 ? index + extent : index;
        if (i < 0 || i >= extent) throw new ArgumentOutOfRangeException(nameof(index));
        var shape = tensor._shape.Where((_, k) => k != d).ToArray();
        return Reshape(TensorNarrow(tensor, d, i, 1), shape);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorViewAs<T>(Tensor<T> tensor, Tensor<T> other)
    {
        if (other == null) throw new ArgumentNullException(nameof(other));
        return Reshape(tensor, (int[])other._shape.Clone());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSumToSize<T>(Tensor<T> tensor, int[] size)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (size == null) throw new ArgumentNullException(nameof(size));
        int lead = tensor.Rank - size.Length;
        if (lead < 0) throw new ArgumentException("size has more dimensions than the tensor.", nameof(size));
        var axes = new List<int>();
        for (int k = 0; k < tensor.Rank; k++)
        {
            if (k < lead) { axes.Add(k); continue; }
            int target = size[k - lead];
            if (target == 1 && tensor._shape[k] != 1) axes.Add(k);
            else if (target != tensor._shape[k])
                throw new ArgumentException($"size [{string.Join(", ", size)}] is not broadcast-compatible.", nameof(size));
        }
        var summed = axes.Count == 0 ? tensor : ReduceSum(tensor, axes.ToArray(), keepDims: true);
        return Reshape(summed, (int[])size.Clone());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false,
        QuantileInterpolation interpolation = QuantileInterpolation.Linear)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return QuantileCore("TensorQuantile", tensor, q, dim, keepDim, interpolation, ignoreNan: false);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNanQuantile<T>(Tensor<T> tensor, double q, int? dim = null, bool keepDim = false,
        QuantileInterpolation interpolation = QuantileInterpolation.Linear)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return QuantileCore("TensorNanQuantile", tensor, q, dim, keepDim, interpolation, ignoreNan: true);
    }

    // Per output: sort the slice, read the two order statistics around q·(n-1) and blend them. The backward sends
    // (1 - w)·dy to the lower one and w·dy to the upper one, as PyTorch's does.
    private Tensor<T> QuantileCore<T>(string opName, Tensor<T> tensor, double q, int? dim, bool keepDim,
        QuantileInterpolation interpolation, bool ignoreNan)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (double.IsNaN(q) || q < 0 || q > 1) throw new ArgumentOutOfRangeException(nameof(q), "q must be in [0, 1].");
        var x = dim is null ? Reshape(tensor, new[] { tensor.Length }) : tensor;
        int d = dim is null ? 0 : NormalizeDim(dim.Value, tensor.Rank);
        var source = x.IsContiguous ? x : x.Contiguous();
        var ops = MathHelper.GetNumericOperations<T>();
        var values = source.AsSpan();
        int extent = source._shape[d];
        int outer = source._shape.Take(d).Aggregate(1, (a, b) => a * b);
        int inner = source._shape.Skip(d + 1).Aggregate(1, (a, b) => a * b);
        int outputs = outer * inner;
        var lo = new int[outputs];
        var hi = new int[outputs];
        var weight = new double[outputs];
        var result = new double[outputs];
        var slice = new List<(double Value, int Index)>(extent);
        for (int o = 0; o < outer; o++)
        {
            for (int i = 0; i < inner; i++)
            {
                int output = o * inner + i;
                slice.Clear();
                bool sawNan = false;
                for (int k = 0; k < extent; k++)
                {
                    int flat = (o * extent + k) * inner + i;
                    double v = ops.ToDouble(values[flat]);
                    if (double.IsNaN(v)) { sawNan = true; if (ignoreNan) continue; }
                    slice.Add((v, flat));
                }
                if (slice.Count == 0 || (sawNan && !ignoreNan))
                {
                    result[output] = double.NaN;
                    lo[output] = hi[output] = -1;
                    continue;
                }
                slice.Sort((a, b) => a.Value.CompareTo(b.Value));
                double position = q * (slice.Count - 1);
                int below = (int)Math.Floor(position), above = (int)Math.Ceiling(position);
                double w = position - below;
                switch (interpolation)
                {
                    case QuantileInterpolation.Lower: above = below; w = 0; break;
                    case QuantileInterpolation.Higher: below = above; w = 0; break;
                    case QuantileInterpolation.Nearest:
                        below = above = (int)Math.Round(position, MidpointRounding.ToEven);
                        w = 0;
                        break;
                    case QuantileInterpolation.Midpoint: w = below == above ? 0 : 0.5; break;
                }
                lo[output] = slice[below].Index;
                hi[output] = slice[above].Index;
                weight[output] = w;
                result[output] = (1 - w) * slice[below].Value + w * slice[above].Value;
            }
        }
        var shape = keepDim
            ? (dim is null ? Enumerable.Repeat(1, tensor.Rank).ToArray() : source._shape.Select((s, k) => k == d ? 1 : s).ToArray())
            : source._shape.Where((_, k) => k != d).ToArray();
        var output2 = FromDoubles<T>(shape, k => result[k]);
        DifferentiableOps.RecordUnary(opName, output2, x, QuantileBackward<T>, new object[] { lo, hi, weight });
        return output2;
    }

    private static void QuantileBackward<T>(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var lo = (int[])savedState[0];
        var hi = (int[])savedState[1];
        var weight = (double[])savedState[2];
        var ops = MathHelper.GetNumericOperations<T>();
        var dy = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
        var g = dy.AsSpan();
        var dx = new double[inputs[0].Length];
        for (int k = 0; k < lo.Length; k++)
        {
            if (lo[k] < 0) continue;
            double gk = ops.ToDouble(g[k]);
            dx[lo[k]] += (1 - weight[k]) * gk;
            dx[hi[k]] += weight[k] * gk;
        }
        DifferentiableOps.AccumulateGrad(grads, inputs[0], FromDoubles<T>((int[])inputs[0]._shape.Clone(), i => dx[i]), engine);
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Std, Tensor<T> Mean) TensorStdMean<T>(Tensor<T> tensor, int[]? axes = null, int correction = 1, bool keepDims = false)
    {
        var (variance, mean) = TensorVarMean(tensor, axes, correction, keepDims);
        return (TensorSqrt(variance), mean);
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Var, Tensor<T> Mean) TensorVarMean<T>(Tensor<T> tensor, int[]? axes = null, int correction = 1, bool keepDims = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        var reduce = (axes ?? AllAxes(tensor.Rank)).Select(a => NormalizeDim(a, tensor.Rank)).ToArray();
        int count = reduce.Aggregate(1, (a, k) => a * tensor._shape[k]);
        var meanKept = ReduceMean(tensor, reduce, keepDims: true);
        var centered = TensorSubtract(tensor, TensorBroadcastTo(meanKept, LikeShape(tensor)));
        var variance = TensorMultiplyScalar(ReduceSum(TensorSquare(centered), reduce, keepDims), ops.FromDouble(1.0 / (count - correction)));
        var mean = keepDims ? meanKept : ReduceMean(tensor, reduce, keepDims: false);
        return (variance, mean);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCov<T>(Tensor<T> tensor, int correction = 1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank > 2) throw new ArgumentException("TensorCov expects a 1-D or 2-D [variables, observations] tensor.", nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        var x = tensor.Rank == 1 ? Reshape(tensor, new[] { 1, tensor.Length }) : tensor;
        int observations = x._shape[1];
        var centered = TensorSubtract(x, TensorBroadcastTo(ReduceMean(x, new[] { 1 }, keepDims: true), (int[])x._shape.Clone()));
        var cov = TensorMultiplyScalar(TensorMatMul(centered, TensorTranspose(centered)), ops.FromDouble(1.0 / (observations - correction)));
        return tensor.Rank == 1 ? Reshape(cov, new[] { 1 }) : cov;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCorrcoef<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var x = tensor.Rank == 1 ? Reshape(tensor, new[] { 1, tensor.Length }) : tensor;
        var cov = TensorCov(x);
        int n = cov._shape[0];
        var stddev = TensorSqrt(Reshape(TensorDiagonal(cov), new[] { n, 1 }));
        var ops = MathHelper.GetNumericOperations<T>();
        // PyTorch clips to [-1, 1] against rounding.
        var corr = TensorClamp(TensorDivide(cov, TensorMatMul(stddev, TensorTranspose(stddev))), ops.FromDouble(-1), ops.One);
        return tensor.Rank == 1 ? Reshape(corr, new[] { 1 }) : corr;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDiff<T>(Tensor<T> tensor, int n = 1, int dim = -1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n));
        int d = NormalizeDim(dim, tensor.Rank);
        var x = tensor;
        for (int i = 0; i < n && x._shape[d] > 0; i++)
        {
            int length = x._shape[d] - 1;
            x = TensorSubtract(TensorNarrow(x, d, 1, length), TensorNarrow(x, d, 0, length));
        }
        return x;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTrapezoid<T>(Tensor<T> y, double dx = 1, int dim = -1)
    {
        if (y == null) throw new ArgumentNullException(nameof(y));
        int d = NormalizeDim(dim, y.Rank);
        var ops = MathHelper.GetNumericOperations<T>();
        if (y._shape[d] < 2) return new Tensor<T>(y._shape.Where((_, k) => k != d).ToArray());
        return TensorMultiplyScalar(ReduceSum(TrapezoidPairs(y, d), new[] { d }, keepDims: false), ops.FromDouble(dx / 2));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumulativeTrapezoid<T>(Tensor<T> y, double dx = 1, int dim = -1)
    {
        if (y == null) throw new ArgumentNullException(nameof(y));
        int d = NormalizeDim(dim, y.Rank);
        var ops = MathHelper.GetNumericOperations<T>();
        if (y._shape[d] < 2) return new Tensor<T>(y._shape.Select((s, k) => k == d ? 0 : s).ToArray());
        return TensorMultiplyScalar(TensorCumSum(TrapezoidPairs(y, d), d), ops.FromDouble(dx / 2));
    }

    // y[1:] + y[:-1] along dim d.
    private Tensor<T> TrapezoidPairs<T>(Tensor<T> y, int d)
    {
        int length = y._shape[d] - 1;
        return TensorAdd(TensorNarrow(y, d, 1, length), TensorNarrow(y, d, 0, length));
    }
}
