using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The IIR / FIR filter along the last axis (scipy.signal.lfilter): one tape node whose backward is the same op on the
/// time-reversed gradient, so a GPU engine runs both directions on the device (<see cref="DirectGpuTensorEngine"/>
/// overrides <see cref="TensorLinearFilter{T}"/> with the backend's <see cref="DirectGpu.IDataTransformBackend"/>, one
/// thread per independent signal).
/// </summary>
public partial class CpuEngine
{
    /// <summary>scipy.signal.lfilter along the last axis; see
    /// <see cref="TensorDataEngineExtensions.TensorLinearFilter{T}"/>.</summary>
    public virtual Tensor<T> TensorLinearFilter<T>(Tensor<T> b, Tensor<T> a, Tensor<T> x, Tensor<T>? zi = null)
    {
        var (signals, length, order) = LinearFilterShape(b, a, x, zi);
        var y = RoIAlignPlan.FromDoubles<T>(
            LinearFilterHost(RoIAlignPlan.ToDoubles(b), RoIAlignPlan.ToDoubles(a), RoIAlignPlan.ToDoubles(x),
                zi is null ? null : RoIAlignPlan.ToDoubles(zi), signals, length, order, double.NaN),
            (int[])x._shape.Clone());
        RecordLinearFilter(y, b, a, x, zi);
        return y;
    }

    /// <summary>lfilter with the forecast guard on every output before it feeds the recursion; see
    /// <see cref="TensorDataEngineExtensions.TensorLinearFilterGuarded{T}"/>.</summary>
    public virtual Tensor<T> TensorLinearFilterGuarded<T>(Tensor<T> b, Tensor<T> a, Tensor<T> x, Tensor<T>? zi, double maxAbs)
    {
        ValidateGuard(maxAbs);
        var (signals, length, order) = LinearFilterShape(b, a, x, zi);
        return RoIAlignPlan.FromDoubles<T>(
            LinearFilterHost(RoIAlignPlan.ToDoubles(b), RoIAlignPlan.ToDoubles(a), RoIAlignPlan.ToDoubles(x),
                zi is null ? null : RoIAlignPlan.ToDoubles(zi), signals, length, order, maxAbs),
            (int[])x._shape.Clone());
    }

    internal static void ValidateGuard(double maxAbs)
    {
        if (double.IsNaN(maxAbs) || maxAbs < 0.0)
            throw new ArgumentOutOfRangeException(nameof(maxAbs), maxAbs, "The guard bound must be non-negative (positive infinity: no clamp).");
    }

    /// <summary>Validates the operands; returns (independent signals, samples per signal, K = max(len b, len a)).</summary>
    internal static (int Signals, int Length, int Order) LinearFilterShape<T>(Tensor<T> b, Tensor<T> a, Tensor<T> x, Tensor<T>? zi)
    {
        if (b is null) throw new ArgumentNullException(nameof(b));
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (b.Rank != 1 || b.Length < 1) throw new ArgumentException("b must be a non-empty 1-D coefficient vector.", nameof(b));
        if (a.Rank != 1 || a.Length < 1) throw new ArgumentException("a must be a non-empty 1-D coefficient vector.", nameof(a));
        if (x.Rank < 1) throw new ArgumentException("x needs at least one dimension (time is the last axis).", nameof(x));
        int length = x._shape[x.Rank - 1];
        int signals = length == 0 ? 0 : x.Length / length;
        if (length == 0)
        {
            signals = 1;
            for (int d = 0; d < x.Rank - 1; d++) signals *= x._shape[d];
        }
        int order = Math.Max(a.Length, b.Length);
        if (zi is not null)
        {
            bool ok = zi.Rank == x.Rank && zi._shape[zi.Rank - 1] == order - 1;
            for (int d = 0; ok && d < x.Rank - 1; d++) ok = zi._shape[d] == x._shape[d];
            if (!ok)
            {
                var expected = (int[])x._shape.Clone();
                expected[expected.Length - 1] = order - 1;
                throw new ArgumentException($"zi must have shape [{string.Join(", ", expected)}] (x's leading dimensions and max(len(a), len(b)) - 1); got [{string.Join(", ", zi._shape)}].", nameof(zi));
            }
        }
        return (signals, length, order);
    }

    /// <summary>
    /// scipy's direct form II transposed, every signal in parallel: with <c>b</c>, <c>a</c> divided by <c>a[0]</c> and
    /// padded with zeros to K, <c>y[n] = b[0] x[n] + z[0]</c>, then <c>z[k-1] = b[k] x[n] - a[k] y[n] + z[k]</c> for
    /// <c>k = 1 .. K-1</c> (<c>z[K-1] = 0</c>), z starting at <c>zi</c> (zero when absent). A finite
    /// <paramref name="guard"/> (not NaN) passes each <c>y[n]</c> through NaN -> 0 and a clamp to <c>+-guard</c> before it
    /// is stored and fed back. The device kernels run the same steps in float.
    /// </summary>
    internal static double[] LinearFilterHost(double[] b, double[] a, double[] x, double[]? zi, int signals, int length,
        int order, double guard)
    {
        double a0 = a[0];
        if (a0 == 0.0 || double.IsNaN(a0))
            throw new ArgumentException("a[0] must be nonzero (scipy.signal.lfilter normalizes by it).", nameof(a));
        var bn = new double[order];
        var an = new double[order];
        for (int k = 0; k < b.Length; k++) bn[k] = b[k] / a0;
        for (int k = 0; k < a.Length; k++) an[k] = a[k] / a0;
        bool guarded = !double.IsNaN(guard);
        int nb = b.Length, na = a.Length;
        var y = new double[x.Length];
        CpuParallelSettings.ParallelForOrSerial(0, signals, (long)signals * length * order * 2, s =>
        {
            var z = new double[Math.Max(1, order)];
            if (zi is not null) Array.Copy(zi, s * (order - 1), z, 0, order - 1);
            int o = s * length;
            for (int n = 0; n < length; n++)
            {
                double xn = x[o + n];
                double yn = bn[0] * xn + z[0];
                if (guarded) yn = double.IsNaN(yn) ? 0.0 : yn > guard ? guard : yn < -guard ? -guard : yn;
                // Only the coefficients the caller gave contribute: a shorter b or a must not turn a NaN input (or a
                // guarded-away NaN output) into a NaN state through 0 * NaN.
                for (int k = 1; k < order; k++)
                {
                    double t = 0.0;
                    if (k < nb) t = bn[k] * xn;
                    if (k < na) t -= an[k] * yn;
                    z[k - 1] = t + z[k];
                }
                y[o + n] = yn;
            }
        }, deterministicSafe: true);
        return y;
    }

    /// <summary>Records the filter's tape node (b, a, x and zi when present).</summary>
    private protected static void RecordLinearFilter<T>(Tensor<T> y, Tensor<T> b, Tensor<T> a, Tensor<T> x, Tensor<T>? zi) =>
        DifferentiableOps.RecordIfActive("LinearFilter", y, zi is null ? new[] { b, a, x } : new[] { b, a, x, zi },
            LinearFilterBackward<T>);

    /// <summary>
    /// Backward of <see cref="TensorLinearFilter{T}"/>. In unnormalized form the filter is
    /// <c>sum_k a[k] y[n-k] = sum_k b[k] x[n-k] + a[0] zi[n]</c> (zi[n] = 0 past K - 2), so with
    /// <c>r = A^-T g = flip(lfilter([1], a, flip(g)))</c>: <c>dx = flip(lfilter(b, a, flip(g)))</c>,
    /// <c>db[k] = sum_n r[n] x[n-k]</c>, <c>da[k] = -sum_n r[n] y[n-k] (+ sum_n r[n] zi[n] for k = 0)</c> and
    /// <c>dzi[n] = a[0] r[n] = flip(lfilter([a[0]], a, flip(g)))[n]</c>, sums over every signal. Every step is an engine
    /// op, so a createGraph backward stays differentiable.
    /// </summary>
    internal static void LinearFilterBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        Tensor<T> b = inputs[0], a = inputs[1], x = inputs[2];
        Tensor<T>? zi = inputs.Length > 3 ? inputs[3] : null;
        int axis = x.Rank - 1, length = x._shape[axis];
        var time = new[] { axis };
        var flipped = engine.TensorFlip(gradOutput, time);
        if (DifferentiableOps.IsGradientRequired(x))
            DifferentiableOps.AccumulateGrad(grads, x, engine.TensorFlip(engine.TensorLinearFilter(b, a, flipped), time), engine);
        bool needB = DifferentiableOps.IsGradientRequired(b), needA = DifferentiableOps.IsGradientRequired(a);
        var ziNeedingGrad = zi is not null && DifferentiableOps.IsGradientRequired(zi) ? zi : null;
        bool needZi = ziNeedingGrad is not null;
        if (!needB && !needA && !needZi) return;
        var ops = MathHelper.GetNumericOperations<T>();
        var one = new Tensor<T>(new[] { ops.One }, new[] { 1 });
        var r = engine.TensorFlip(engine.TensorLinearFilter(one, a, flipped), time);
        if (needB) DifferentiableOps.AccumulateGrad(grads, b, LagCorrelations(engine, r, x, b.Length, length, axis, false), engine);
        if (needA)
        {
            var da = LagCorrelations(engine, r, output, a.Length, length, axis, true);
            if (zi is not null)
            {
                int m = Math.Min(zi._shape[axis], length);
                if (m > 0)
                {
                    var c = SumAll(engine, engine.TensorMultiply(engine.TensorNarrow(r, axis, 0, m), engine.TensorNarrow(zi, axis, 0, m)));
                    var rest = a.Length > 1 ? new Tensor<T>(new[] { a.Length - 1 }) : null;
                    var bump = rest is null ? c : engine.TensorConcatenate(new[] { c, rest }, 0);
                    da = engine.TensorAdd(da, bump);
                }
            }
            DifferentiableOps.AccumulateGrad(grads, a, da, engine);
        }
        if (ziNeedingGrad is not null)
        {
            int states = ziNeedingGrad._shape[axis];
            int m = Math.Min(states, length);
            var q = engine.TensorFlip(engine.TensorLinearFilter(engine.TensorNarrow(a, 0, 0, 1), a, flipped), time);
            Tensor<T> dzi;
            if (m == states) dzi = engine.TensorNarrow(q, axis, 0, m);
            else
            {
                var padShape = (int[])ziNeedingGrad._shape.Clone();
                padShape[axis] = states - m;
                var pad = new Tensor<T>(padShape);
                dzi = m == 0 ? pad : engine.TensorConcatenate(new[] { engine.TensorNarrow(q, axis, 0, m), pad }, axis);
            }
            DifferentiableOps.AccumulateGrad(grads, ziNeedingGrad, dzi, engine);
        }
    }

    // [taps]: entry k = (negate ? -1 : 1) * sum over every signal of r[n] * s[n - k], zero for lags at or past the length.
    private static Tensor<T> LagCorrelations<T>(IEngine engine, Tensor<T> r, Tensor<T> s, int taps, int length, int axis, bool negate)
    {
        var parts = new Tensor<T>[taps];
        for (int k = 0; k < taps; k++)
        {
            if (k >= length)
            {
                parts[k] = new Tensor<T>(new[] { 1 });
                continue;
            }
            var sum = SumAll(engine, engine.TensorMultiply(engine.TensorNarrow(r, axis, k, length - k), engine.TensorNarrow(s, axis, 0, length - k)));
            parts[k] = negate ? engine.TensorNegate(sum) : sum;
        }
        return taps == 1 ? parts[0] : engine.TensorConcatenate(parts, 0);
    }

    private static Tensor<T> SumAll<T>(IEngine engine, Tensor<T> t) => engine.Reshape(engine.ReduceSum(t, null, false), new[] { 1 });
}
