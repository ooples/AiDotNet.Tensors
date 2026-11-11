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
                double a = m[o + i] / d, c = m[o + i + 1] / d;
                double sum = a * a + c * c;
                if (sum > 9.0)
                {
                    double tau = 3.0 / Math.Sqrt(sum);
                    m[o + i] = tau * (a * d);
                    m[o + i + 1] = tau * (c * d);
                }
            }
        }, deterministicSafe: true);
        return m;
    }
}
