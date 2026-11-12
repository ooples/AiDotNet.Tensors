using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The sequential Fritsch-Carlson (1980) step-2 limiter behind <see cref="IEngine.TensorFritschCarlsonLimit{T}"/>,
/// series in parallel and the recurrence in order inside each (the device kernels run one series per thread).
/// </summary>
public partial class CpuEngine
{
    /// <inheritdoc />
    public virtual Tensor<T> TensorFritschCarlsonLimit<T>(Tensor<T> slopes, Tensor<T> secants)
    {
        var (batch, n) = FritschCarlsonShape(slopes, secants);
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var cm = slopes; var cd = secants;
            Action<IEngine, Tensor<T>> execute = (eng, result) =>
                DirectGpuTensorEngine.CopyResultInto(eng, eng.TensorFritschCarlsonLimit(cm, cd), result);
            return scope.RecordBinary(LazyNodeType.Custom, "FritschCarlsonLimit", slopes, secants,
                (int[])slopes._shape.Clone(), execute);
        }
        return RoIAlignPlan.FromDoubles<T>(
            FritschCarlsonHost(RoIAlignPlan.ToDoubles(slopes), RoIAlignPlan.ToDoubles(secants), batch, n),
            (int[])slopes._shape.Clone());
    }

    /// <summary>Validates slopes [..., n] and secants [..., n - 1]; returns (number of series, n).</summary>
    internal static (int Batch, int N) FritschCarlsonShape<T>(Tensor<T> slopes, Tensor<T> secants)
    {
        if (slopes is null) throw new ArgumentNullException(nameof(slopes));
        if (secants is null) throw new ArgumentNullException(nameof(secants));
        if (slopes.Rank < 1) throw new ArgumentException("slopes must have at least one dimension [..., n].", nameof(slopes));
        int rank = slopes.Rank, n = slopes._shape[rank - 1];
        if (n < 1) throw new ArgumentException("Each series needs at least one slope.", nameof(slopes));
        CheckOffDiagonal(secants, slopes, nameof(secants));
        return (slopes.Length / n, n);
    }

    /// <summary>
    /// The limiter on every series: for i = 0 .. n - 2, a zero secant flattens both end slopes; otherwise, when
    /// <c>a^2 + b^2 &gt; 9</c> (a = m[i] / d[i], b = m[i+1] / d[i]), both are scaled by <c>tau = 3 / sqrt(a^2 + b^2)</c>
    /// as <c>tau * (a * d[i])</c> and <c>tau * (b * d[i])</c>. Step i reads the m[i] step i - 1 wrote.
    /// <para>Evaluated in slope space without forming a, b or their squares, which overflow for finite inputs (slopes
    /// 1e200 over a secant 1e-200 made tau 0 and the result 0 * inf = NaN): with s = max(|m[i]|, |m[i+1]|),
    /// u = m / s and r = sqrt(u0^2 + u1^2) in [1, sqrt 2], the test a^2 + b^2 &gt; 9 is s / |d| &gt; 3 / r and the
    /// limited slopes are u * (3 / r) * |d|. The device kernels use the same steps, so a NaN slope leaves both unchanged
    /// on every backend.</para>
    /// </summary>
    internal static double[] FritschCarlsonHost(double[] slopes, double[] secants, int batch, int n)
    {
        var m = (double[])slopes.Clone();
        CpuParallelSettings.ParallelForOrSerial(0, batch, (long)batch * n * 8, b =>
        {
            int o = b * n, od = b * (n - 1);
            for (int i = 0; i < n - 1; i++)
            {
                double d = secants[od + i];
                if (d == 0.0)
                {
                    m[o + i] = 0.0;
                    m[o + i + 1] = 0.0;
                    continue;
                }
                double m0 = m[o + i], m1 = m[o + i + 1];
                double ad = d < 0.0 ? -d : d;
                double a0 = m0 < 0.0 ? -m0 : m0, a1 = m1 < 0.0 ? -m1 : m1;
                double s = a0 > a1 ? a0 : a1;
                if (s > 0.0)
                {
                    double u0 = m0 / s, u1 = m1 / s;
                    double k = 3.0 / Math.Sqrt(u0 * u0 + u1 * u1);
                    if (s / ad > k)
                    {
                        m[o + i] = u0 * k * ad;
                        m[o + i + 1] = u1 * k * ad;
                    }
                }
            }
        }, deterministicSafe: true);
        return m;
    }
}
