using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

// Distribution draws from the caller's System.Random: normal (a sampler delegate, or Box-Muller in a chosen
// convention), clamped normal, gamma (Marsaglia-Tsang), Bernoulli and Rademacher. Each consumes the stream exactly as
// the scalar loop it replaces, element by element in row-major order, so a seeded caller's values and later draws are
// unchanged. The draws are host-side by necessity (System.Random is one sequential stream); DirectGpuTensorEngine runs
// the per-element transform (Box-Muller, the threshold) as a device kernel over the uploaded uniforms. None has a
// differentiable input, so none records a tape node.
public partial class CpuEngine
{
    // ------------------------------------------------------------------ uniforms

    /// <summary>One Box-Muller pair under <paramref name="variant"/>'s rules, <c>u1</c> drawn first, floor applied.</summary>
    internal static void DrawBoxMullerPair(Random generator, BoxMullerVariant variant, out double u1, out double u2)
    {
        switch (variant.FirstUniform)
        {
            case BoxMullerFirstUniform.OneMinus:
                u1 = 1.0 - generator.NextDouble();
                u2 = generator.NextDouble();
                break;
            case BoxMullerFirstUniform.RedrawWhileZero:
                do { u1 = generator.NextDouble(); } while (u1 == 0.0);
                u2 = generator.NextDouble();
                break;
            case BoxMullerFirstUniform.RedrawPairWhileZero:
                do
                {
                    u1 = generator.NextDouble();
                    u2 = generator.NextDouble();
                } while (u1 <= double.Epsilon);
                break;
            case BoxMullerFirstUniform.Raw:
                u1 = generator.NextDouble();
                u2 = generator.NextDouble();
                break;
            default:
                throw new ArgumentOutOfRangeException(nameof(variant), variant.FirstUniform, "Unknown Box-Muller first-uniform rule.");
        }
        if (variant.SecondUniform == BoxMullerSecondUniform.OneMinus) u2 = 1.0 - u2;
        else if (variant.SecondUniform != BoxMullerSecondUniform.Raw)
            throw new ArgumentOutOfRangeException(nameof(variant), variant.SecondUniform, "Unknown Box-Muller second-uniform rule.");
        u1 = Math.Max(u1, variant.FirstUniformFloor);
    }

    /// <summary><c>sqrt(-2 ln u1) cos(2 pi u2)</c> (or sin), in the scalar loops' evaluation order.</summary>
    internal static double BoxMullerNormal(double u1, double u2, BoxMullerBranch branch)
    {
        double r = Math.Sqrt(-2.0 * Math.Log(u1));
        return branch == BoxMullerBranch.Sine ? r * Math.Sin(2.0 * Math.PI * u2) : r * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>Every rule of <paramref name="variant"/> is a defined value: checked before the first draw, so an invalid
    /// convention throws with the caller's Random untouched (and even when there is nothing to draw).</summary>
    internal static void ValidateVariant(BoxMullerVariant variant)
    {
        if (variant.FirstUniform < BoxMullerFirstUniform.Raw || variant.FirstUniform > BoxMullerFirstUniform.RedrawPairWhileZero)
            throw new ArgumentOutOfRangeException(nameof(variant), variant.FirstUniform, "Unknown Box-Muller first-uniform rule.");
        if (variant.SecondUniform != BoxMullerSecondUniform.Raw && variant.SecondUniform != BoxMullerSecondUniform.OneMinus)
            throw new ArgumentOutOfRangeException(nameof(variant), variant.SecondUniform, "Unknown Box-Muller second-uniform rule.");
        if (variant.Branch != BoxMullerBranch.Cosine && variant.Branch != BoxMullerBranch.Sine)
            throw new ArgumentOutOfRangeException(nameof(variant), variant.Branch, "Unknown Box-Muller branch.");
    }

    /// <summary>
    /// The uniforms of <paramref name="count"/> Box-Muller draws, interleaved <c>[u1, u2]</c> per element, after the
    /// variant's rules: what the device transform kernel consumes.
    /// </summary>
    [ScalarKernel("System.Random is a sequential stream: each pair is drawn in order, and a redraw rule decides how many draws an element takes.")]
    internal static double[] DrawBoxMullerUniforms(Random generator, BoxMullerVariant variant, int count)
    {
        var pairs = new double[2 * count];
        for (int i = 0; i < count; i++)
        {
            DrawBoxMullerPair(generator, variant, out double u1, out double u2);
            pairs[2 * i] = u1;
            pairs[2 * i + 1] = u2;
        }
        return pairs;
    }

    // ------------------------------------------------------------------ TensorNormalFromRng (sampler delegate)

    /// <summary>One <paramref name="sampler"/>(generator) per element, row-major; see
    /// <see cref="TensorDataEngineExtensions.TensorNormalFromRng{T}"/>.</summary>
    public virtual Tensor<T> TensorNormalFromRng<T>(Random generator, int[] shape, Func<Random, double> sampler)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (sampler is null) throw new ArgumentNullException(nameof(sampler));
        int count = (int)ElementCountOf(shape ?? throw new ArgumentNullException(nameof(shape)));
        var values = DrawFromSampler(generator, sampler, count);
        var result = new Tensor<T>((int[])shape.Clone());
        StoreDoubles(values, result.GetDataArray());
        return result;
    }

    [ScalarKernel("The sampler is the caller's delegate over one sequential System.Random stream: one call per element, in order.")]
    internal static double[] DrawFromSampler(Random generator, Func<Random, double> sampler, int count)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++) values[i] = sampler(generator);
        return values;
    }

    /// <summary>The doubles converted to <typeparamref name="T"/> into <paramref name="destination"/>[0, count).</summary>
    internal static void StoreDoubles<T>(double[] values, T[] destination)
    {
        if (destination is double[] d)
        {
            Array.Copy(values, d, values.Length);
            return;
        }
        if (destination is float[] f)
        {
            CpuParallelSettings.ParallelForOrSerial(0, (values.Length + DataOpChunk - 1) / DataOpChunk, values.Length, chunk =>
            {
                int from = chunk * DataOpChunk, to = Math.Min(values.Length, from + DataOpChunk);
                for (int i = from; i < to; i++) f[i] = (float)values[i];
            }, deterministicSafe: true);
            return;
        }
        ConvertFromDoubles(values, destination);
    }

    // ------------------------------------------------------------------ Box-Muller normal / clamped normal

    /// <summary><c>mean + stdDev * z</c> per element, z a Box-Muller normal under <paramref name="variant"/>; see
    /// <see cref="TensorDataEngineExtensions.TensorNormalBoxMullerFromRng{T}"/>.</summary>
    public virtual Tensor<T> TensorNormalBoxMullerFromRng<T>(Random generator, int[] shape, BoxMullerVariant variant,
        double mean = 0.0, double stdDev = 1.0)
        => TensorTruncatedNormalFromRng<T>(generator, shape, variant, mean, stdDev, double.NegativeInfinity, double.PositiveInfinity);

    /// <summary><c>clamp(mean + stdDev * z, low, high)</c> per element; see
    /// <see cref="TensorDataEngineExtensions.TensorTruncatedNormalFromRng{T}"/>.</summary>
    public virtual Tensor<T> TensorTruncatedNormalFromRng<T>(Random generator, int[] shape, BoxMullerVariant variant,
        double mean, double stdDev, double low, double high)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        int count = (int)ElementCountOf(shape ?? throw new ArgumentNullException(nameof(shape)));
        ValidateVariant(variant);
        ValidateClampBounds(low, high);
        var result = new Tensor<T>((int[])shape.Clone());
        if (count == 0) return result;
        var pairs = DrawBoxMullerUniforms(generator, variant, count);
        StoreDoubles(TransformBoxMullerPairs(pairs, variant.Branch, mean, stdDev, low, high), result.GetDataArray());
        return result;
    }

    /// <summary><c>clamp(mean + stdDev * z, low, high)</c> for each drawn pair (interleaved u1, u2), in parallel.</summary>
    internal static double[] TransformBoxMullerPairs(double[] pairs, BoxMullerBranch branch, double mean, double stdDev,
        double low, double high)
    {
        int count = pairs.Length / 2;
        var values = new double[count];
        CpuParallelSettings.ParallelForOrSerial(0, (count + DataOpChunk - 1) / DataOpChunk, (long)count * 16, chunk =>
        {
            int from = chunk * DataOpChunk, to = Math.Min(count, from + DataOpChunk);
            for (int i = from; i < to; i++)
                values[i] = ClampNormal(mean + stdDev * BoxMullerNormal(pairs[2 * i], pairs[2 * i + 1], branch), low, high);
        }, deterministicSafe: true);
        return values;
    }

    internal static void ValidateClampBounds(double low, double high)
    {
        if (double.IsNaN(low) || double.IsNaN(high) || low > high)
            throw new ArgumentException($"The clamp bounds must satisfy low <= high; got [{low}, {high}].", nameof(low));
    }

    /// <summary>The value clamped to [low, high]; a NaN passes through unchanged.</summary>
    internal static double ClampNormal(double value, double low, double high)
        => value < low ? low : value > high ? high : value;

    // ------------------------------------------------------------------ gamma (Marsaglia-Tsang)

    /// <summary>Gamma(shape, scale[i]) draws, row-major; see
    /// <see cref="TensorDataEngineExtensions.TensorGammaFromRng{T}(IEngine, Random, double, Tensor{T}, BoxMullerVariant)"/>.</summary>
    public virtual Tensor<T> TensorGammaFromRng<T>(Random generator, double shape, Tensor<T> scale, BoxMullerVariant normal)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        ValidateGammaShape(shape);
        ValidateVariant(normal);
        var scales = RoIAlignPlan.ToDoubles(scale);
        var values = DrawGammas(generator, null, shape, scales, normal);
        var result = new Tensor<T>((int[])scale._shape.Clone());
        StoreDoubles(values, result.GetDataArray());
        return result;
    }

    /// <summary>Gamma(shape[i], scale[i]) draws, row-major; see
    /// <see cref="TensorDataEngineExtensions.TensorGammaFromRng{T}(IEngine, Random, Tensor{T}, Tensor{T}, BoxMullerVariant)"/>.</summary>
    public virtual Tensor<T> TensorGammaFromRng<T>(Random generator, Tensor<T> shape, Tensor<T> scale, BoxMullerVariant normal)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        if (scale is null) throw new ArgumentNullException(nameof(scale));
        if (!SameShape(shape._shape, scale._shape))
            throw new ArgumentException($"shape and scale must have the same shape; got [{string.Join(", ", shape._shape)}] and [{string.Join(", ", scale._shape)}].", nameof(scale));
        ValidateVariant(normal);
        var shapes = RoIAlignPlan.ToDoubles(shape);
        foreach (double s in shapes) ValidateGammaShape(s);
        var values = DrawGammas(generator, shapes, 0.0, RoIAlignPlan.ToDoubles(scale), normal);
        var result = new Tensor<T>((int[])scale._shape.Clone());
        StoreDoubles(values, result.GetDataArray());
        return result;
    }

    private static void ValidateGammaShape(double shape)
    {
        if (!(shape > 0.0) || double.IsInfinity(shape))
            throw new ArgumentOutOfRangeException(nameof(shape), shape, "A gamma shape must be finite and positive.");
    }

    [ScalarKernel("Marsaglia-Tsang is a rejection sampler on one sequential System.Random stream: how many draws an element takes depends on the draws.")]
    private static double[] DrawGammas(Random generator, double[]? shapes, double sharedShape, double[] scales, BoxMullerVariant normal)
    {
        var values = new double[scales.Length];
        for (int i = 0; i < values.Length; i++)
            values[i] = SampleGamma(generator, shapes is null ? sharedShape : shapes[i], scales[i], normal);
        return values;
    }

    /// <summary>
    /// Marsaglia and Tsang (2000): for shape below one the boost uniform is drawn first, then
    /// <c>Gamma(shape + 1, scale) * u^(1 / shape)</c>; otherwise <c>d = shape - 1/3</c>, <c>c = 1 / sqrt(9 d)</c>, normals
    /// x redrawn while <c>v = 1 + c x &lt;= 0</c>, <c>v = v^3</c>, then one uniform u accepts on the squeeze
    /// <c>u &lt; 1 - 0.0331 x^4</c> or on <c>ln u &lt; x^2 / 2 + d (1 - v + ln v)</c>, returning <c>d v scale</c>.
    /// </summary>
    internal static double SampleGamma(Random generator, double shape, double scale, BoxMullerVariant normal)
    {
        if (shape < 1.0)
        {
            double u = generator.NextDouble();
            return SampleGamma(generator, shape + 1.0, scale, normal) * Math.Pow(u, 1.0 / shape);
        }
        double d = shape - 1.0 / 3.0;
        double c = 1.0 / Math.Sqrt(9.0 * d);
        while (true)
        {
            double x, v;
            do
            {
                DrawBoxMullerPair(generator, normal, out double u1, out double u2);
                x = BoxMullerNormal(u1, u2, normal.Branch);
                v = 1.0 + c * x;
            } while (v <= 0.0);
            v = v * v * v;
            double u = generator.NextDouble();
            if (u < 1.0 - 0.0331 * (x * x) * (x * x)) return d * v * scale;
            if (Math.Log(u) < 0.5 * x * x + d * (1.0 - v + Math.Log(v))) return d * v * scale;
        }
    }

    // ------------------------------------------------------------------ Bernoulli / Rademacher

    /// <summary>1 where the element's draw is on <paramref name="successWhen"/>'s side of its probability, else 0; see
    /// <see cref="TensorDataEngineExtensions.TensorBernoulliFromRng{T}(IEngine, Random, Tensor{T}, UniformDrawComparison)"/>.</summary>
    public virtual Tensor<T> TensorBernoulliFromRng<T>(Random generator, Tensor<T> probabilities,
        UniformDrawComparison successWhen)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        if (probabilities is null) throw new ArgumentNullException(nameof(probabilities));
        ValidateComparison(successWhen);
        var thresholds = RoIAlignPlan.ToDoubles(probabilities);
        var draws = DrawUniforms(generator, thresholds.Length);
        var result = new Tensor<T>((int[])probabilities._shape.Clone());
        StoreDoubles(CompareDraws(draws, thresholds, 0.0, successWhen, 1.0, 0.0), result.GetDataArray());
        return result;
    }

    /// <summary>Per draw: <paramref name="onSuccess"/> when it is on the comparison's side of its threshold
    /// (<paramref name="thresholds"/>[i], or <paramref name="threshold"/> when that is null), else
    /// <paramref name="otherwise"/>.</summary>
    internal static double[] CompareDraws(double[] draws, double[]? thresholds, double threshold,
        UniformDrawComparison comparison, double onSuccess, double otherwise)
    {
        var values = new double[draws.Length];
        for (int i = 0; i < values.Length; i++)
            values[i] = Exceeds(draws[i], thresholds is null ? threshold : thresholds[i], comparison) ? onSuccess : otherwise;
        return values;
    }

    /// <summary>1 where the element's draw is on <paramref name="successWhen"/>'s side of <paramref name="threshold"/>,
    /// else 0; see <see cref="TensorDataEngineExtensions.TensorBernoulliFromRng{T}(IEngine, Random, int[], double, UniformDrawComparison)"/>.</summary>
    public virtual Tensor<T> TensorBernoulliFromRng<T>(Random generator, int[] shape, double threshold,
        UniformDrawComparison successWhen)
        => ThresholdDraws<T>(generator, shape, threshold, successWhen, 1.0, 0.0);

    /// <summary>+/-1 per element under <paramref name="rule"/>; see
    /// <see cref="TensorDataEngineExtensions.TensorRademacherFromRng{T}"/>.</summary>
    public virtual Tensor<T> TensorRademacherFromRng<T>(Random generator, int[] shape,
        RademacherRule rule = RademacherRule.NegativeWhenDrawBelowHalf)
    {
        var (comparison, onSuccess, otherwise) = RademacherThreshold(rule);
        return ThresholdDraws<T>(generator, shape, 0.5, comparison, onSuccess, otherwise);
    }

    /// <summary>The rule as a threshold draw at 0.5: (comparison, value on success, value otherwise).</summary>
    internal static (UniformDrawComparison Comparison, double OnSuccess, double Otherwise) RademacherThreshold(RademacherRule rule)
        => rule switch
        {
            RademacherRule.NegativeWhenDrawBelowHalf => (UniformDrawComparison.DrawBelowThreshold, -1.0, 1.0),
            RademacherRule.PositiveWhenDrawAboveHalf => (UniformDrawComparison.DrawAboveThreshold, 1.0, -1.0),
            _ => throw new ArgumentOutOfRangeException(nameof(rule), rule, "Unknown Rademacher rule."),
        };

    /// <summary>One draw per element compared with one threshold: <paramref name="onSuccess"/> or
    /// <paramref name="otherwise"/> (the Bernoulli and Rademacher draws).</summary>
    private protected virtual Tensor<T> ThresholdDraws<T>(Random generator, int[] shape, double threshold, UniformDrawComparison comparison,
        double onSuccess, double otherwise)
    {
        if (generator is null) throw new ArgumentNullException(nameof(generator));
        int count = (int)ElementCountOf(shape ?? throw new ArgumentNullException(nameof(shape)));
        ValidateComparison(comparison);
        var draws = DrawUniforms(generator, count);
        var result = new Tensor<T>((int[])shape.Clone());
        StoreDoubles(CompareDraws(draws, null, threshold, comparison, onSuccess, otherwise), result.GetDataArray());
        return result;
    }

    internal static void ValidateComparison(UniformDrawComparison comparison)
    {
        if (comparison != UniformDrawComparison.DrawBelowThreshold && comparison != UniformDrawComparison.DrawAboveThreshold)
            throw new ArgumentOutOfRangeException(nameof(comparison), comparison, "Unknown uniform-draw comparison.");
    }

    /// <summary>Whether the draw is on the comparison's side of the threshold (false for a NaN threshold).</summary>
    internal static bool Exceeds(double draw, double threshold, UniformDrawComparison comparison)
        => comparison == UniformDrawComparison.DrawAboveThreshold ? draw > threshold : draw < threshold;

    [ScalarKernel("System.Random is a sequential stream: one NextDouble per element, in order.")]
    internal static double[] DrawUniforms(Random generator, int count)
    {
        var draws = new double[count];
        for (int i = 0; i < count; i++) draws[i] = generator.NextDouble();
        return draws;
    }
}
