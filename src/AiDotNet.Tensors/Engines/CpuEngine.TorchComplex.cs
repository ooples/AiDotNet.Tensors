using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    private static Complex<T>[] ComplexValues<T>(Tensor<Complex<T>> input, string name)
    {
        if (input == null) throw new ArgumentNullException(name);
        return (input.IsContiguous ? input : input.Contiguous()).AsSpan().ToArray();
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorReal<T>(Tensor<Complex<T>> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        return new Tensor<T>(ComplexValues(input, nameof(input)).Select(z => z.Real).ToArray(), input._shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorImag<T>(Tensor<Complex<T>> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        return new Tensor<T>(ComplexValues(input, nameof(input)).Select(z => z.Imaginary).ToArray(), input._shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<Complex<T>> TensorComplex<T>(Tensor<T> real, Tensor<T> imag)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (real == null) throw new ArgumentNullException(nameof(real));
        if (imag == null) throw new ArgumentNullException(nameof(imag));
        if (!real._shape.SequenceEqual(imag._shape)) throw new ArgumentException("real and imag must have the same shape.", nameof(imag));
        var re = (real.IsContiguous ? real : real.Contiguous()).AsSpan().ToArray();
        var im = (imag.IsContiguous ? imag : imag.Contiguous()).AsSpan().ToArray();
        return new Tensor<Complex<T>>(re.Select((r, i) => new Complex<T>(r, im[i])).ToArray(), real._shape.ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorViewAsReal<T>(Tensor<Complex<T>> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput);
        var z = ComplexValues(input, nameof(input));
        var pairs = new T[z.Length * 2];
        for (int i = 0; i < z.Length; i++) { pairs[2 * i] = z[i].Real; pairs[2 * i + 1] = z[i].Imaginary; }
        return new Tensor<T>(pairs, input._shape.Concat(new[] { 2 }).ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<Complex<T>> TensorViewAsComplex<T>(Tensor<T> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousOutput);
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank == 0 || input._shape[input.Rank - 1] != 2)
            throw new ArgumentException("view_as_complex needs a trailing axis of size 2.", nameof(input));
        var v = (input.IsContiguous ? input : input.Contiguous()).AsSpan().ToArray();
        var z = new Complex<T>[v.Length / 2];
        for (int i = 0; i < z.Length; i++) z[i] = new Complex<T>(v[2 * i], v[2 * i + 1]);
        return new Tensor<Complex<T>>(z, input._shape.Take(input.Rank - 1).ToArray());
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAngle<T>(Tensor<T> input)
    {
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HostBoundary);
        if (input == null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        var v = (input.IsContiguous ? input : input.Contiguous()).AsSpan().ToArray();
        // PyTorch: NaN propagates, negatives (including -inf) give π, everything else 0; -0.0 gives 0.
        return new Tensor<T>(v.Select(x =>
        {
            double d = ops.ToDouble(x);
            return ops.FromDouble(double.IsNaN(d) ? double.NaN : d < 0 ? Math.PI : 0.0);
        }).ToArray(), input._shape.ToArray());
    }
}
