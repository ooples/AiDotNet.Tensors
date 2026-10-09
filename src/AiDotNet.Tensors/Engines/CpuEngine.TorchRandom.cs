using AiDotNet.Tensors.Engines.Compilation;
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.NN.Losses;

namespace AiDotNet.Tensors.Engines;

/// <summary>Random sampling, dropout variants and composed ops (see the matching <see cref="IEngine"/> members).</summary>
public partial class CpuEngine
{
    private static double StandardNormal(Random rng)
        => Math.Sqrt(-2 * Math.Log(1 - rng.NextDouble())) * Math.Cos(2 * Math.PI * rng.NextDouble());

    // Knuth's product method for small rates; Hörmann's transformed rejection (PTRS) above 10.
    private static double SamplePoisson(double rate, Random rng)
    {
        if (rate < 0 || double.IsNaN(rate)) throw new ArgumentOutOfRangeException(nameof(rate), "Poisson rates must be non-negative.");
        if (rate == 0) return 0;
        if (rate < 10)
        {
            double limit = Math.Exp(-rate), product = rng.NextDouble();
            int k = 0;
            while (product > limit) { k++; product *= rng.NextDouble(); }
            return k;
        }
        double slam = Math.Sqrt(rate), logLam = Math.Log(rate);
        double b = 0.931 + 2.53 * slam, a = -0.059 + 0.02483 * b, invAlpha = 1.1239 + 1.1328 / (b - 3.4), vr = 0.9277 - 3.6224 / (b - 2);
        while (true)
        {
            double u = rng.NextDouble() - 0.5, v = rng.NextDouble(), us = 0.5 - Math.Abs(u);
            double k = Math.Floor((2 * a / us + b) * u + rate + 0.43);
            if (us >= 0.07 && v <= vr) return k;
            if (k < 0 || (us < 0.013 && v > us)) continue;
            if (Math.Log(v) + Math.Log(invAlpha) - Math.Log(a / (us * us) + b) <= -rate + k * logLam - SpecialFunctions.LogGamma(k + 1))
                return k;
        }
    }

    // Exact binomial draw by geometric waiting times between successes: O(n·min(p, 1-p)) expected steps.
    private static double SampleBinomial(double count, double p, Random rng)
    {
        // torch.binomial does not validate this and returns fractional draws for a fractional count; a count of
        // trials must be a whole number, so it is rejected rather than truncated.
        if (double.IsNaN(count) || count != Math.Floor(count))
            throw new ArgumentOutOfRangeException(nameof(count), $"count must be a whole number of trials, got {count}.");
        long n = (long)count;
        if (n < 0 || p < 0 || p > 1 || double.IsNaN(p)) throw new ArgumentOutOfRangeException(nameof(p), "need count ≥ 0 and 0 ≤ p ≤ 1.");
        if (n == 0 || p == 0) return 0;
        if (p == 1) return n;
        bool flip = p > 0.5;
        double q = flip ? 1 - p : p, logQ = Math.Log(1 - q);
        long successes = 0, position = 0;
        while (true)
        {
            position += (long)Math.Floor(Math.Log(1 - rng.NextDouble()) / logQ) + 1;
            if (position > n) break;
            successes++;
        }
        return flip ? n - successes : successes;
    }

    private Tensor<T> Sample<T>(int[] shape, int? seed, Func<Random, double> draw)
    {
        var rng = RandomSource(seed);
        return FromDoubles<T>((int[])shape.Clone(), _ => draw(rng));
    }

    private Tensor<T> SampleElementwise<T>(Tensor<T> parameters, int? seed, Func<double, Random, double> draw)
    {
        if (parameters == null) throw new ArgumentNullException(nameof(parameters));
        var ops = MathHelper.GetNumericOperations<T>();
        var values = (parameters.IsContiguous ? parameters : parameters.Contiguous()).AsSpan().ToArray();
        var rng = RandomSource(seed);
        return FromDoubles<T>((int[])parameters._shape.Clone(), i => draw(ops.ToDouble(values[i]), rng));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBernoulli<T>(Tensor<T> probabilities, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return SampleElementwise(probabilities, seed, (p, rng) =>
        {
            if (p < 0 || p > 1 || double.IsNaN(p)) throw new ArgumentOutOfRangeException(nameof(probabilities), "probabilities must lie in [0, 1].");
            return rng.NextDouble() < p ? 1 : 0;
        });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBinomial<T>(Tensor<T> count, Tensor<T> probability, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (count == null) throw new ArgumentNullException(nameof(count));
        if (probability == null) throw new ArgumentNullException(nameof(probability));
        if (!count._shape.SequenceEqual(probability._shape)) throw new ArgumentException("count and probability must have the same shape.");
        var ops = MathHelper.GetNumericOperations<T>();
        var p = (probability.IsContiguous ? probability : probability.Contiguous()).AsSpan().ToArray();
        var n = (count.IsContiguous ? count : count.Contiguous()).AsSpan().ToArray();
        var rng = RandomSource(seed);
        return FromDoubles<T>((int[])count._shape.Clone(), i => SampleBinomial(ops.ToDouble(n[i]), ops.ToDouble(p[i]), rng));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPoisson<T>(Tensor<T> rates, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return SampleElementwise(rates, seed, SamplePoisson);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNormal<T>(Tensor<T> mean, Tensor<T> std, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (mean == null) throw new ArgumentNullException(nameof(mean));
        if (std == null) throw new ArgumentNullException(nameof(std));
        // mean + std·ε with ε a constant draw: the reparameterized form, so gradients reach mean and std as in PyTorch.
        var epsilon = Sample<T>(mean._shape, seed, StandardNormal);
        return TensorAdd(mean, TensorMultiply(std, epsilon));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorUniform<T>(int[] shape, double low = 0, double high = 1, int? seed = null)
        => Sample<T>(shape, seed, rng => low + (high - low) * rng.NextDouble());

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCauchy<T>(int[] shape, double median = 0, double sigma = 1, int? seed = null)
        => Sample<T>(shape, seed, rng => median + sigma * Math.Tan(Math.PI * (rng.NextDouble() - 0.5)));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorExponential<T>(int[] shape, double rate = 1, int? seed = null)
    {
        if (rate <= 0) throw new ArgumentOutOfRangeException(nameof(rate), "rate must be positive.");
        return Sample<T>(shape, seed, rng => -Math.Log(1 - rng.NextDouble()) / rate);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorGeometric<T>(int[] shape, double p, int? seed = null)
    {
        if (p <= 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p), "p must be in (0, 1].");
        return Sample<T>(shape, seed, rng => p == 1 ? 1 : Math.Max(1, Math.Ceiling(Math.Log(1 - rng.NextDouble()) / Math.Log(1 - p))));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogNormal<T>(int[] shape, double mean = 1, double std = 2, int? seed = null)
        => Sample<T>(shape, seed, rng => Math.Exp(mean + std * StandardNormal(rng)));

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMultinomial<T>(Tensor<T> probabilities, int numSamples, bool replacement = false, int? seed = null)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (probabilities == null) throw new ArgumentNullException(nameof(probabilities));
        if (probabilities.Rank < 1 || probabilities.Rank > 2) throw new ArgumentException("multinomial expects 1-D or 2-D weights.");
        int categories = probabilities._shape[probabilities.Rank - 1], rows = probabilities.Length / Math.Max(1, categories);
        if (numSamples <= 0) throw new ArgumentOutOfRangeException(nameof(numSamples));
        if (!replacement && numSamples > categories)
            throw new ArgumentException("without replacement, numSamples cannot exceed the number of categories.", nameof(numSamples));
        var ops = MathHelper.GetNumericOperations<T>();
        var weights = (probabilities.IsContiguous ? probabilities : probabilities.Contiguous()).AsSpan().ToArray().Select(v => ops.ToDouble(v)).ToArray();
        if (weights.Any(w => w < 0 || double.IsNaN(w) || double.IsInfinity(w)))
            throw new ArgumentException("weights must be finite and non-negative.", nameof(probabilities));
        var rng = RandomSource(seed);
        var picks = new double[rows * numSamples];
        var row = new double[categories];
        for (int r = 0; r < rows; r++)
        {
            Array.Copy(weights, r * categories, row, 0, categories);
            for (int s = 0; s < numSamples; s++)
            {
                double total = row.Sum();
                if (total <= 0) throw new ArgumentException($"row {r} has no positive weight left to sample.", nameof(probabilities));
                double u = rng.NextDouble() * total, running = 0;
                int chosen = Array.FindLastIndex(row, v => v > 0);
                for (int c = 0; c < categories; c++)
                {
                    running += row[c];
                    if (u < running && row[c] > 0) { chosen = c; break; }
                }
                picks[r * numSamples + s] = chosen;
                if (!replacement) row[chosen] = 0;
            }
        }
        var shape = probabilities.Rank == 1 ? new[] { numSamples } : new[] { rows, numSamples };
        return FromDoubles<T>(shape, i => picks[i]);
    }

    // SELU's negative saturation value, -λ·α.
    private const double SeluSaturation = -1.7580993408473766;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return AlphaDropout(tensor, p, training, seed, channelWise: false);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFeatureAlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        return AlphaDropout(tensor, p, training, seed, channelWise: true);
    }

    // out = a·(x·m + α'(1 - m)) + b with a, b keeping mean 0 and variance 1 under SELU; recorded as x·(a·m) + const.
    private Tensor<T> AlphaDropout<T>(Tensor<T> tensor, double p, bool training, int? seed, bool channelWise)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
        if (!training || p == 0) return tensor;
        double a = 1 / Math.Sqrt((1 - p) * (1 + p * SeluSaturation * SeluSaturation)), b = -a * SeluSaturation * p;
        var keep = KeepMask(tensor._shape, p, seed, channelWise ? Math.Max(0, tensor.Rank - 2) : -1);
        var scale = FromDoubles<T>((int[])tensor._shape.Clone(), i => a * keep[i]);
        var offset = FromDoubles<T>((int[])tensor._shape.Clone(), i => a * SeluSaturation * (1 - keep[i]) + b);
        return TensorAdd(TensorMultiply(tensor, scale), offset);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorChannelDropout<T>(Tensor<T> tensor, double p, bool training, int spatialDims, int? seed = null)
    {
        if (training) GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p));
        if (spatialDims < 0 || spatialDims >= tensor.Rank) throw new ArgumentOutOfRangeException(nameof(spatialDims));
        if (!training || p == 0) return tensor;
        var keep = KeepMask(tensor._shape, p, seed, spatialDims);
        double scale = p == 1 ? 0 : 1 / (1 - p);
        return TensorMultiply(tensor, FromDoubles<T>((int[])tensor._shape.Clone(), i => keep[i] * scale));
    }

    // A 0/1 keep mask (1 with probability 1 - p) over shape; with spatialDims ≥ 0 one draw per channel, shared over the
    // trailing spatialDims axes.
    private double[] KeepMask(int[] shape, double p, int? seed, int spatialDims)
    {
        var rng = RandomSource(seed);
        int total = shape.Aggregate(1, (x, y) => x * y);
        int block = spatialDims < 0 ? 1 : shape.Skip(shape.Length - spatialDims).Aggregate(1, (x, y) => x * y);
        var keep = new double[total];
        for (int start = 0; start < total; start += block)
        {
            double value = rng.NextDouble() >= p ? 1 : 0;
            for (int i = 0; i < block; i++) keep[start + i] = value;
        }
        return keep;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBilinear<T>(Tensor<T> input1, Tensor<T> input2, Tensor<T> weight, Tensor<T>? bias = null)
    {
        if (input1 == null) throw new ArgumentNullException(nameof(input1));
        if (input2 == null) throw new ArgumentNullException(nameof(input2));
        if (weight == null) throw new ArgumentNullException(nameof(weight));
        if (weight.Rank != 3) throw new ArgumentException($"weight must be [out, in1, in2], got rank {weight.Rank}.", nameof(weight));
        int outFeatures = weight._shape[0], in1 = weight._shape[1], in2 = weight._shape[2];
        if (input1.Rank == 0 || input1._shape[input1.Rank - 1] != in1)
            throw new ArgumentException($"input1's last axis must be in1 = {in1}.", nameof(input1));
        if (input2.Rank == 0 || input2._shape[input2.Rank - 1] != in2)
            throw new ArgumentException($"input2's last axis must be in2 = {in2}.", nameof(input2));
        if (!input1._shape.Take(input1.Rank - 1).SequenceEqual(input2._shape.Take(input2.Rank - 1)))
            throw new ArgumentException("input1 and input2 must share their leading dimensions.", nameof(input2));
        if (bias != null && bias.Length != outFeatures)
            throw new ArgumentException($"bias must hold out = {outFeatures} values, got {bias.Length}.", nameof(bias));
        int rows = input1.Length / in1;
        var lead = input1._shape.Take(input1.Rank - 1).ToArray();
        // x1·W as [rows, out·in2], then multiplied by x2 and summed over in2.
        var w = Reshape(TensorPermute(weight, new[] { 1, 0, 2 }), new[] { in1, outFeatures * in2 });
        var left = Reshape(TensorMatMul(Reshape(input1, new[] { rows, in1 }), w), new[] { rows, outFeatures, in2 });
        var right = TensorBroadcastTo(Reshape(input2, new[] { rows, 1, in2 }), new[] { rows, outFeatures, in2 });
        var y = ReduceSum(TensorMultiply(left, right), new[] { 2 }, keepDims: false);
        if (bias != null) y = TensorAdd(y, TensorBroadcastTo(Reshape(bias, new[] { 1, outFeatures }), new[] { rows, outFeatures }));
        return Reshape(y, lead.Concat(new[] { outFeatures }).ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorChannelShuffle<T>(Tensor<T> tensor, int groups)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int channels = tensor._shape[1];
        if (groups <= 0 || channels % groups != 0) throw new ArgumentException($"{channels} channels are not divisible into {groups} groups.", nameof(groups));
        var rest = tensor._shape.Skip(2).ToArray();
        var split = new[] { tensor._shape[0], groups, channels / groups }.Concat(rest).ToArray();
        var perm = new[] { 0, 2, 1 }.Concat(Enumerable.Range(3, rest.Length)).ToArray();
        return Reshape(TensorPermute(Reshape(tensor, split), perm), (int[])tensor._shape.Clone());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPixelUnshuffle<T>(Tensor<T> tensor, int downscaleFactor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int r = downscaleFactor, rank = tensor.Rank;
        if (r <= 0 || rank < 3) throw new ArgumentException("pixel_unshuffle expects [..., C, H, W] and a positive factor.");
        int c = tensor._shape[rank - 3], h = tensor._shape[rank - 2], w = tensor._shape[rank - 1];
        if (h % r != 0 || w % r != 0) throw new ArgumentException($"H={h} and W={w} must be divisible by {r}.");
        int batch = tensor.Length / (c * h * w);
        var x = Reshape(tensor, new[] { batch, c, h / r, r, w / r, r });
        var y = Reshape(TensorPermute(x, new[] { 0, 1, 3, 5, 2, 4 }), new[] { batch, c * r * r, h / r, w / r });
        return Reshape(y, tensor._shape.Take(rank - 3).Concat(new[] { c * r * r, h / r, w / r }).ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLocalResponseNorm<T>(Tensor<T> tensor, int size, double alpha = 1e-4, double beta = 0.75, double k = 1)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size));
        if (tensor.Rank < 3) throw new ArgumentException($"local response norm needs [batch, channels, ...] with rank ≥ 3, got {tensor.Rank}.", nameof(tensor));
        int n = tensor._shape[0], c = tensor._shape[1], m = tensor.Length / (n * c);
        var x = Reshape(tensor, new[] { n, c, m });
        // Window sums of x² over channels by a cumulative sum with a leading zero: S[c] = cs[c + size] - cs[c].
        var ops = MathHelper.GetNumericOperations<T>();
        var padded = PadNd(TensorSquare(x), new[] { 0, 0, size / 2 + 1, (size - 1) / 2 }, PadMode.Constant, ops.Zero);
        var cs = TensorCumSum(padded, 1);
        var window = TensorSubtract(TensorNarrow(cs, 1, size, c), TensorNarrow(cs, 1, 0, c));
        var denominator = TensorAddScalar(TensorMultiplyScalar(window, ops.FromDouble(alpha / size)), ops.FromDouble(k));
        var scaled = TensorPow(denominator, ops.FromDouble(beta));
        return Reshape(TensorDivide(x, scaled), (int[])tensor._shape.Clone());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSoftMarginLoss<T>(Tensor<T> input, Tensor<T> target, LossReduction reduction = LossReduction.Mean)
    {
        // log(1 + exp(-y·x)) = -logsigmoid(y·x), stable for large |x|.
        var loss = TensorNegate(TensorLogSigmoid(TensorMultiply(input, target)));
        return reduction switch
        {
            LossReduction.None => loss,
            LossReduction.Sum => ReduceSum(loss, null, false),
            _ => ReduceMean(loss, AllAxes(loss.Rank), false),
        };
    }

    // (Σ |x|^p over axes)^(1/p), recorded. |x|^p is a power, not exp(p·log|x|), so a zero element has a zero
    // derivative (p > 1) instead of 0·∞. The root of a zero sum has an infinite derivative, so zero sums are
    // shifted to 1 inside the root and back out after it (a constant mask): the norm stays 0 and, as in PyTorch's
    // norm backward, its gradient is 0 rather than NaN.
    private Tensor<T> PNorm<T>(Tensor<T> x, double p, int[] axes, bool keepDims)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (p == 1) return ReduceSum(TensorAbs(x), axes, keepDims);
        var sum = ReduceSum(p == 2 ? TensorSquare(x) : TensorPow(TensorAbs(x), ops.FromDouble(p)), axes, keepDims);
        var sums = (sum.IsContiguous ? sum : sum.Contiguous()).AsSpan().ToArray();
        var zeroMask = FromDoubles<T>((int[])sum._shape.Clone(), i => ops.ToDouble(sums[i]) == 0 ? 1.0 : 0.0);
        var shifted = TensorAdd(sum, zeroMask);
        var root = p == 2 ? TensorSqrt(shifted) : TensorPow(shifted, ops.FromDouble(1 / p));
        return TensorSubtract(root, zeroMask);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRenorm<T>(Tensor<T> tensor, double p, int dim, double maxNorm)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (p <= 0) throw new ArgumentOutOfRangeException(nameof(p));
        int d = NormalizeDim(dim, tensor.Rank);
        var ops = MathHelper.GetNumericOperations<T>();
        var others = Enumerable.Range(0, tensor.Rank).Where(k => k != d).ToArray();
        var norm = others.Length == 0 ? TensorAbs(tensor) : PNorm(tensor, p, others, keepDims: true);
        // Slices over maxNorm are scaled by maxNorm / (norm + 1e-7) (PyTorch's epsilon), the rest by 1.
        var normValues = norm.AsSpan().ToArray();
        var over = FromDoubles<T>((int[])norm._shape.Clone(), i => ops.ToDouble(normValues[i]) > maxNorm ? 1 : 0);
        var limited = TensorDivide(TensorFullLike(norm, ops.FromDouble(maxNorm)), TensorAddScalar(norm, ops.FromDouble(1e-7)));
        var factor = TensorWhere(over, limited, TensorOnesLike(norm));
        return TensorMultiply(tensor, TensorBroadcastTo(factor, (int[])tensor._shape.Clone()));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNormExceptDim<T>(Tensor<T> tensor, double pow = 2, int dim = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (dim == -1) return PNorm(tensor, pow, AllAxes(tensor.Rank), keepDims: false);
        int d = NormalizeDim(dim, tensor.Rank);
        var others = Enumerable.Range(0, tensor.Rank).Where(k => k != d).ToArray();
        return others.Length == 0 ? TensorAbs(tensor) : PNorm(tensor, pow, others, keepDims: true);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorGradient<T>(Tensor<T> tensor, double spacing = 1, int[]? dims = null)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        var axes = (dims ?? AllAxes(tensor.Rank)).Select(a => NormalizeDim(a, tensor.Rank)).ToArray();
        var result = new Tensor<T>[axes.Length];
        for (int i = 0; i < axes.Length; i++)
        {
            int d = axes[i], n = tensor._shape[d];
            if (n < 2) throw new ArgumentException($"torch.gradient needs at least 2 samples along dim {d}.");
            var first = TensorMultiplyScalar(TensorSubtract(TensorNarrow(tensor, d, 1, 1), TensorNarrow(tensor, d, 0, 1)), ops.FromDouble(1 / spacing));
            var last = TensorMultiplyScalar(TensorSubtract(TensorNarrow(tensor, d, n - 1, 1), TensorNarrow(tensor, d, n - 2, 1)), ops.FromDouble(1 / spacing));
            if (n == 2) { result[i] = TensorConcatenate(new[] { first, last }, d); continue; }
            var inner = TensorMultiplyScalar(TensorSubtract(TensorNarrow(tensor, d, 2, n - 2), TensorNarrow(tensor, d, 0, n - 2)), ops.FromDouble(1 / (2 * spacing)));
            result[i] = TensorConcatenate(new[] { first, inner, last }, d);
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPadSequence<T>(Tensor<T>[] sequences, bool batchFirst = false, double paddingValue = 0)
    {
        if (sequences == null || sequences.Length == 0) throw new ArgumentException("pad_sequence needs at least one sequence.", nameof(sequences));
        var ops = MathHelper.GetNumericOperations<T>();
        int longest = sequences.Max(s => s._shape[0]);
        var padded = sequences.Select(s =>
        {
            var pad = new int[2 * s.Rank];
            pad[2 * s.Rank - 1] = longest - s._shape[0];   // after-padding of axis 0 (PadNd lists the innermost axis first)
            return longest == s._shape[0] ? s : PadNd(s, pad, PadMode.Constant, ops.FromDouble(paddingValue));
        }).ToArray();
        return TensorStack(padded, batchFirst ? 0 : 1);
    }

    /// <inheritdoc/>
    public virtual Tensor<int> TensorNonzeroStatic<T>(Tensor<T> tensor, int size, int fillValue = -1)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (size < 0) throw new ArgumentOutOfRangeException(nameof(size));
        var ops = MathHelper.GetNumericOperations<T>();
        var values = (tensor.IsContiguous ? tensor : tensor.Contiguous()).AsSpan().ToArray();
        int rank = Math.Max(1, tensor.Rank);
        var result = Enumerable.Repeat(fillValue, size * rank).ToArray();
        int found = 0;
        for (int flat = 0; flat < values.Length && found < size; flat++)
        {
            if (ops.Equals(values[flat], ops.Zero)) continue;
            int rest = flat;
            for (int k = tensor.Rank - 1; k >= 0; k--) { result[found * rank + k] = rest % tensor._shape[k]; rest /= tensor._shape[k]; }
            found++;
        }
        return new Tensor<int>(result, new[] { size, rank });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorUniqueDim<T>(Tensor<T> tensor, int dim)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.DataDependentOutputShape);
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int d = NormalizeDim(dim, tensor.Rank);
        var ops = MathHelper.GetNumericOperations<T>();
        var slices = Enumerable.Range(0, tensor._shape[d])
            .Select(i => (Index: i, Key: TensorSelect(tensor, d, i).AsSpan().ToArray().Select(v => ops.ToDouble(v)).ToArray()))
            .ToList();
        int Compare(double[] a, double[] b)
        {
            for (int i = 0; i < a.Length; i++) { int c = a[i].CompareTo(b[i]); if (c != 0) return c; }
            return 0;
        }
        slices.Sort((a, b) => Compare(a.Key, b.Key));
        var keep = new List<int>();
        for (int i = 0; i < slices.Count; i++)
            if (i == 0 || Compare(slices[i].Key, slices[i - 1].Key) != 0) keep.Add(slices[i].Index);
        using (new NoGradScope<T>())
            return TensorConcatenate(keep.Select(i => TensorNarrow(tensor, d, i, 1)).ToArray(), d);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexReduce<T>(Tensor<T> tensor, int dim, Tensor<int> index, Tensor<T> source, ScatterReduceMode reduce, bool includeSelf = true)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (index == null) throw new ArgumentNullException(nameof(index));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int d = NormalizeDim(dim, tensor.Rank);
        if (index.Length != source._shape[d]) throw new ArgumentException("index must have one entry per slice of source along dim.", nameof(index));
        // index_reduce's 1-D index, expanded to source's shape, is scatter_reduce's index.
        var idx = index.AsSpan().ToArray();
        int inner = source._shape.Skip(d + 1).Aggregate(1, (a, b) => a * b), extent = source._shape[d];
        var expanded = new int[source.Length];
        for (int i = 0; i < expanded.Length; i++) expanded[i] = idx[i / inner % extent];
        return TensorScatterReduce(tensor, d, new Tensor<int>(expanded, (int[])source._shape.Clone()), source, reduce, includeSelf);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorConvTranspose1D<T>(Tensor<T> input, Tensor<T> kernel, int stride = 1, int padding = 0, int outputPadding = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (kernel == null) throw new ArgumentNullException(nameof(kernel));
        int n = input._shape[0], cin = input._shape[1], length = input._shape[2], cout = kernel._shape[1], k = kernel._shape[2];
        var y = ConvTranspose2D(Reshape(input, new[] { n, cin, 1, length }), Reshape(kernel, new[] { cin, cout, 1, k }),
            new[] { 1, stride }, new[] { 0, padding }, new[] { 0, outputPadding });
        return Reshape(y, new[] { n, cout, y._shape[3] });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorConvTbc<T>(Tensor<T> input, Tensor<T> weight, Tensor<T> bias, int pad = 0)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (weight == null) throw new ArgumentNullException(nameof(weight));
        if (bias == null) throw new ArgumentNullException(nameof(bias));
        var x = TensorPermute(input, new[] { 1, 2, 0 });              // [T, B, Cin] -> [B, Cin, T]
        var w = TensorPermute(weight, new[] { 2, 1, 0 });             // [K, Cin, Cout] -> [Cout, Cin, K]
        var y = Conv1D(x, w, 1, pad);                                 // [B, Cout, T']
        y = TensorAdd(y, TensorBroadcastTo(Reshape(bias, new[] { 1, bias.Length, 1 }), (int[])y._shape.Clone()));
        return TensorPermute(y, new[] { 2, 0, 1 });                   // [T', B, Cout]
    }
}
