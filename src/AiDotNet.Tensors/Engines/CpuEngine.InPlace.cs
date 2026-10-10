using System;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The in-place forms of the engine's ops (torch's <c>op_</c> methods): the mechanism the generated ones share, and
/// the ones that need more than "compute the out-of-place result and write it back".
/// </summary>
/// <remarks>
/// <para><b>Mechanism.</b> <see cref="ApplyInPlace{U}"/> runs the out-of-place op through <see cref="IEngine"/> (so a
/// GPU engine runs its kernels) and writes the result over the tensor: through <see cref="IEngine.TensorCopy{T}"/>
/// for a contiguous tensor, element by element through the view otherwise, so every alias of the storage sees the
/// write. It records like the hand-written in-place ops (<see cref="ReLUInPlace{T}"/> ...): under graph capture as
/// an in-place node; under a gradient tape the tensor's history first moves to a snapshot of its value
/// (<see cref="DifferentiableOps.RebaseForInPlace{T}"/>), the op reads the snapshot, and the write is recorded
/// snapshot -> tensor.</para>
/// <para><b>True in place.</b> Where the result is bit-identical to the out-of-place op, and nothing records, the
/// elementwise ops write the tensor's own buffer with no temporary (<see cref="TryUnaryInPlaceFast{U}"/>).</para>
/// <para><b>Layout ops</b> (<c>squeeze_</c>, <c>t_</c>, <c>as_strided_</c>, <c>set_</c>, <c>resize_</c>) change the
/// tensor's shape and strides rather than its values: the tensor adopts the layout of the out-of-place view of its
/// own storage (<see cref="TensorBase{T}.AdoptFrom"/>).</para>
/// </remarks>
public partial class CpuEngine
{
    internal unsafe delegate void FloatInPlaceKernel(float* input, float* output, int length);
    internal unsafe delegate void DoubleInPlaceKernel(double* input, double* output, int length);

    /// <summary>
    /// Overwrites <paramref name="target"/> with <paramref name="op"/>(<paramref name="target"/>), where
    /// <paramref name="op"/> is an out-of-place engine op that returns a tensor of the same shape.
    /// </summary>
    /// <param name="others">Same-typed tensors the op reads besides the target: graph capture orders the write after
    /// their producers.</param>
    private void ApplyInPlace<U>(Tensor<U> target, string opName, Func<IEngine, Tensor<U>, Tensor<U>> op, params Tensor<U>[] others)
    {
        if (target is null) throw new ArgumentNullException(nameof(target));
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current; scope?.BindEngineIfUnset(this);
            if (scope != null)
            {
                var captured = op;
                var name = opName;
                scope.RecordInPlace(LazyNodeType.Custom, opName, target, others, (eng, t) => InPlaceEager(eng, t, name, captured));
                return;
            }
        }
        InPlaceEager(this, target, opName, op);
    }

    private static void InPlaceEager<U>(IEngine engine, Tensor<U> target, string opName, Func<IEngine, Tensor<U>, Tensor<U>> op)
    {
        // Under a tape the target's history moves to a snapshot of its current value (the op's recorded input keeps
        // the pre-write value its backward needs, and the producer's gradient stays the old value's), and the write
        // is recorded snapshot -> target.
        var old = DifferentiableOps.RebaseForInPlace(target);
        bool record = old is not null;
        var result = op(engine, old ?? target);
        if (!ShapesEqual(result._shape, target._shape))
            throw new InvalidOperationException(
                $"{opName}: the result has shape [{string.Join(", ", result._shape)}] but the tensor written in place has " +
                $"shape [{string.Join(", ", target._shape)}]; an in-place op cannot change the shape (torch would broadcast " +
                "the other operands to the tensor, never the tensor to them).");
        WriteInPlaceResult(engine, target, result);
        if (record) DifferentiableOps.RecordUnary(opName, target, result, BackwardFunctions<U>.InPlaceAssignBackward);
    }

    // True when a gradient tape records in-place writes on this thread: in-place ops then take the recorded path.
    internal static bool IsRecordingInPlace<U>()
    {
        if (DifferentiableOps._anyTapeActive == 0) return false;
        var tape = GradientTape<U>.Current;
        return tape is not null && tape.Options.RecordInPlace && !NoGradScope<U>.IsSuppressed;
    }

    private static bool ShapesEqual(int[] a, int[] b)
    {
        if (a.Length != b.Length) return false;
        for (int i = 0; i < a.Length; i++) if (a[i] != b[i]) return false;
        return true;
    }

    /// <summary>Writes <paramref name="result"/> over <paramref name="target"/>, through the view when it is one.</summary>
    private static void WriteInPlaceResult<U>(IEngine engine, Tensor<U> target, Tensor<U> result)
    {
        if (ReferenceEquals(target, result)) { target.IncrementVersion(); return; }
        if (target.IsContiguous)
        {
            engine.TensorCopy(result, target);
            return;
        }
        var source = result.IsContiguous ? result : result.Contiguous();
        for (int i = 0; i < source.Length; i++) target[i] = source[i];
        target.IncrementVersion();
    }

    /// <summary>
    /// The zero-temporary path for an elementwise unary op: applies the op to the tensor's own buffer when nothing
    /// records it, the data is on the host and contiguous, and the element type is float or double. Returns false to
    /// send the caller down <see cref="ApplyInPlace{U}"/>. <paramref name="scalar"/> must be the exact function the
    /// out-of-place op applies (or the kernels an exact one), so the two give the same bits.
    /// </summary>
    private unsafe bool TryUnaryInPlaceFast<U>(Tensor<U> tensor, Func<double, double>? scalar,
        FloatInPlaceKernel? floatKernel = null, DoubleInPlaceKernel? doubleKernel = null)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (typeof(U) != typeof(float) && typeof(U) != typeof(double)) return false;
        if (GraphMode.IsActive && GraphMode.Current is not null) return false;
        if (IsRecordingInPlace<U>()) return false;
        // A GPU engine keeps its own tensors on the device: its in-place ops run as kernels there.
        if (this is DirectGpuTensorEngine gpu && gpu.SupportsGpu) return false;
        if (!tensor.IsContiguous || tensor.IsGpuResident) return false;
        if (scalar is null && (typeof(U) == typeof(float) ? floatKernel is null : doubleKernel is null)) return false;

        int length = tensor.Length;
        if (typeof(U) == typeof(float))
        {
            Span<float> span = ((Tensor<float>)(object)tensor).AsWritableSpan();
            fixed (float* p = span)
            {
                var ptr = (IntPtr)p;
                if (floatKernel is not null)
                    ParallelElementwiseChunks(length, (start, count) => floatKernel((float*)ptr + start, (float*)ptr + start, count));
                else
                    ParallelElementwiseChunks(length, (start, count) =>
                    {
                        float* q = (float*)ptr + start;
                        for (int i = 0; i < count; i++) q[i] = (float)scalar!(q[i]);
                    });
            }
        }
        else
        {
            Span<double> span = ((Tensor<double>)(object)tensor).AsWritableSpan();
            fixed (double* p = span)
            {
                var ptr = (IntPtr)p;
                if (doubleKernel is not null)
                    ParallelElementwiseChunks(length, (start, count) => doubleKernel((double*)ptr + start, (double*)ptr + start, count));
                else
                    ParallelElementwiseChunks(length, (start, count) =>
                    {
                        double* q = (double*)ptr + start;
                        for (int i = 0; i < count; i++) q[i] = scalar!(q[i]);
                    });
            }
        }
        tensor.IncrementVersion();
        return true;
    }

    /// <summary>
    /// Makes <paramref name="tensor"/> adopt the layout of <paramref name="viewOp"/>(<paramref name="tensor"/>), an
    /// out-of-place view of its storage: the in-place view ops (<c>squeeze_</c>, <c>t_</c> ...).
    /// </summary>
    private void ApplyLayoutInPlace<U>(Tensor<U> tensor, string opName, Func<IEngine, Tensor<U>, Tensor<U>> viewOp)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        // A lazy graph fixes every node's shape at capture; a node that changes its own target's shape cannot replay.
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.Stateful, opName);
        IEngine engine = this;
        Tensor<U>? recorded = null;
        var old = DifferentiableOps.RebaseForInPlace(tensor);
        if (old is not null) recorded = viewOp(engine, old);
        Tensor<U> layout;
        using (new NoGradScope<U>()) layout = viewOp(engine, tensor);
        tensor.AdoptFrom(layout);
        if (recorded is not null) DifferentiableOps.RecordUnary(opName, tensor, recorded, BackwardFunctions<U>.InPlaceAssignBackward);
    }

    // ---- values -------------------------------------------------------------------------------------------------

    /// <inheritdoc/>
    public virtual void TensorZeroInPlace<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        ApplyInPlace(tensor, "TensorZeroInPlace", static (e, x) => e.TensorZerosLike(x));
    }

    /// <inheritdoc/>
    public virtual void TensorCopyInPlace<T>(Tensor<T> tensor, Tensor<T> source)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (source is null) throw new ArgumentNullException(nameof(source));
        if (ReferenceEquals(source, tensor)) return;   // t.copy_(t) changes nothing, and must not record t as its own input
        // torch broadcasts the source to the tensor written; the gradient flows to the source only.
        ApplyInPlace(tensor, "TensorCopyInPlace",
            (e, x) => ShapesEqual(source._shape, x._shape) ? source : e.TensorBroadcastTo(source, x._shape), source);
    }

    /// <inheritdoc/>
    public virtual void TensorDetachInPlace<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        // The history moves to a snapshot that every use recorded so far now reads, so those uses still reach the
        // producer; the tensor itself is left with none, and gradients of later uses stop at it.
        DifferentiableOps.RebaseForInPlace(tensor);
        tensor.GradFn = null;
    }

    /// <inheritdoc/>
    public virtual void TensorRequiresGradInPlace<T>(Tensor<T> tensor, bool requiresGrad = true)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        // The tape records every op and computes gradients for whichever tensors are asked for, so every tensor
        // already "requires grad"; turning it off cuts the tensor from its history, as detach_ does.
        if (!requiresGrad) TensorDetachInPlace(tensor);
    }

    /// <inheritdoc/>
    public virtual void DropoutInPlace<T>(Tensor<T> input, double dropoutRate = 0.5, bool training = true)
        => ApplyInPlace(input, "DropoutInPlace", (e, x) => e.Dropout(x, dropoutRate, training, out _));

    /// <inheritdoc/>
    public virtual void TensorFillDiagonalInPlace<T>(Tensor<T> tensor, T value, bool wrap = false)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank < 2) throw new ArgumentException("fill_diagonal_ needs a tensor of at least 2 dimensions.", nameof(tensor));
        if (tensor.Rank > 2)
            for (int d = 1; d < tensor.Rank; d++)
                if (tensor._shape[d] != tensor._shape[0])
                    throw new ArgumentException("fill_diagonal_ on more than 2 dimensions needs every dimension the same length.", nameof(tensor));
        ApplyInPlace(tensor, "TensorFillDiagonalInPlace", (e, x) =>
        {
            var r = x.Clone();
            var data = r.AsWritableSpan();
            int rank = r.Rank;
            // Row-major step between consecutive diagonal elements: the sum of the strides.
            int step = 0, stride = 1;
            for (int d = rank - 1; d >= 0; d--) { step += stride; stride *= r._shape[d]; }
            int n = r._shape[0];
            for (int d = 1; d < rank; d++) n = Math.Min(n, r._shape[d]);
            if (rank == 2 && wrap)
            {
                // torch: a tall matrix continues the diagonal after every (cols + 1) rows, through the whole buffer.
                for (int i = 0; i < data.Length; i += step) data[i] = value;
            }
            else
            {
                for (int i = 0; i < n; i++) data[i * step] = value;
            }
            return r;
        });
    }

    /// <inheritdoc/>
    public virtual void TensorApplyInPlace<T>(Tensor<T> tensor, Func<T, T> func)
    {
        if (func is null) throw new ArgumentNullException(nameof(func));
        ApplyInPlace(tensor, "TensorApplyInPlace", (e, x) => e.TensorMap(x, func));
    }

    /// <inheritdoc/>
    public virtual void TensorMapInPlace<T>(Tensor<T> tensor, Tensor<T> other, Func<T, T, T> func)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        if (func is null) throw new ArgumentNullException(nameof(func));
        ApplyInPlace(tensor, "TensorMapInPlace", (e, x) =>
        {
            var b = ShapesEqual(other._shape, x._shape) ? other : e.TensorBroadcastTo(other, x._shape);
            var xa = x.IsContiguous ? x : x.Contiguous();
            var ba = b.IsContiguous ? b : b.Contiguous();
            var r = new Tensor<T>(x._shape);
            var xs = xa.AsSpan(); var bs = ba.AsSpan(); var rs = r.AsWritableSpan();
            for (int i = 0; i < rs.Length; i++) rs[i] = func(xs[i], bs[i]);
            return r;
        }, other);
    }

    /// <inheritdoc/>
    public virtual void TensorMap2InPlace<T>(Tensor<T> tensor, Tensor<T> x, Tensor<T> y, Func<T, T, T, T> func)
    {
        if (x is null) throw new ArgumentNullException(nameof(x));
        if (y is null) throw new ArgumentNullException(nameof(y));
        if (func is null) throw new ArgumentNullException(nameof(func));
        ApplyInPlace(tensor, "TensorMap2InPlace", (e, self) =>
        {
            var a = ShapesEqual(x._shape, self._shape) ? x : e.TensorBroadcastTo(x, self._shape);
            var b = ShapesEqual(y._shape, self._shape) ? y : e.TensorBroadcastTo(y, self._shape);
            var sc = self.IsContiguous ? self : self.Contiguous();
            var ac = a.IsContiguous ? a : a.Contiguous();
            var bc = b.IsContiguous ? b : b.Contiguous();
            var r = new Tensor<T>(self._shape);
            var ss = sc.AsSpan(); var xs = ac.AsSpan(); var ys = bc.AsSpan(); var rs = r.AsWritableSpan();
            for (int i = 0; i < rs.Length; i++) rs[i] = func(ss[i], xs[i], ys[i]);
            return r;
        }, x, y);
    }

    /// <inheritdoc/>
    public virtual void TensorEmbeddingRenormInPlace<T>(Tensor<T> weight, Tensor<int> indices, double maxNorm, double normType = 2.0)
    {
        if (weight is null) throw new ArgumentNullException(nameof(weight));
        if (indices is null) throw new ArgumentNullException(nameof(indices));
        if (weight.Rank != 2) throw new ArgumentException("embedding_renorm_ needs a 2-D weight [num_embeddings, dim].", nameof(weight));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.HeterogeneousInput, "TensorEmbeddingRenormInPlace");
        if (normType <= 0) throw new ArgumentOutOfRangeException(nameof(normType), "The norm order must be positive.");
        // torch runs this under no_grad: it rescales the rows a lookup touched, it is not part of the graph. Composed from
        // engine ops (gather the rows, p-norm, scale, scatter back) so a GPU engine runs it as device kernels.
        var numOps = MathHelper.GetNumericOperations<T>();
        int rows = weight._shape[0], dim = weight._shape[1];
        var unique = new System.Collections.Generic.SortedSet<int>();
        foreach (int row in indices.AsSpan().ToArray())
        {
            if (row < 0 || row >= rows) throw new ArgumentOutOfRangeException(nameof(indices), $"Index {row} is outside [0, {rows}).");
            unique.Add(row);
        }
        if (unique.Count == 0) return;
        IEngine e = this;
        using (new NoGradScope<T>())
        {
            var idx = new Tensor<int>(new System.Collections.Generic.List<int>(unique).ToArray(), new[] { unique.Count });
            var selected = e.TensorIndexSelect(weight, idx, 0);                                     // [k, dim]
            var powered = e.TensorPow(e.TensorAbs(selected), numOps.FromDouble(normType));
            var norms = e.TensorPow(e.ReduceSum(powered, new[] { 1 }, keepDims: true), numOps.FromDouble(1.0 / normType));   // [k, 1]
            // scale = 1 where norm <= maxNorm, else maxNorm / (norm + 1e-7) (torch's exact rule)
            var over = e.TensorGreaterThan(norms, numOps.FromDouble(maxNorm));                      // 1 / 0
            var shrink = e.TensorMultiplyScalar(e.TensorReciprocal(e.TensorAddScalar(norms, numOps.FromDouble(1e-7))), numOps.FromDouble(maxNorm));
            var scale = e.TensorAddScalar(e.TensorMultiply(over, e.TensorAddScalar(shrink, numOps.FromDouble(-1.0))), numOps.One);
            var rescaled = e.TensorMultiply(selected, e.TensorBroadcastTo(scale, selected._shape));
            e.TensorIndexCopyInPlace(weight, 0, idx, rescaled);
        }
    }

    // ---- random fills -------------------------------------------------------------------------------------------

    /// <inheritdoc/>
    public virtual void TensorBernoulliInPlace<T>(Tensor<T> tensor, double p = 0.5, int? seed = null)
    {
        if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p), "A probability must lie in [0, 1].");
        var numOps = MathHelper.GetNumericOperations<T>();
        ApplyInPlace(tensor, "TensorBernoulliInPlace", (e, x) =>
        {
            var probabilities = new Tensor<T>(x._shape);
            probabilities.Fill(numOps.FromDouble(p));
            return e.TensorBernoulli(probabilities, seed);
        });
    }

    /// <inheritdoc/>
    public virtual void TensorBernoulliInPlace<T>(Tensor<T> tensor, Tensor<T> p, int? seed = null)
    {
        if (p is null) throw new ArgumentNullException(nameof(p));
        ApplyInPlace(tensor, "TensorBernoulliInPlace",
            (e, x) => e.TensorBernoulli(ShapesEqual(p._shape, x._shape) ? p : e.TensorBroadcastTo(p, x._shape), seed));
    }

    /// <inheritdoc/>
    public virtual void TensorCauchyInPlace<T>(Tensor<T> tensor, double median = 0, double sigma = 1, int? seed = null)
        => ApplyInPlace(tensor, "TensorCauchyInPlace", (e, x) => e.TensorCauchy<T>(x._shape, median, sigma, seed));

    /// <inheritdoc/>
    public virtual void TensorExponentialInPlace<T>(Tensor<T> tensor, double rate = 1, int? seed = null)
        => ApplyInPlace(tensor, "TensorExponentialInPlace", (e, x) => e.TensorExponential<T>(x._shape, rate, seed));

    /// <inheritdoc/>
    public virtual void TensorGeometricInPlace<T>(Tensor<T> tensor, double p, int? seed = null)
        => ApplyInPlace(tensor, "TensorGeometricInPlace", (e, x) => e.TensorGeometric<T>(x._shape, p, seed));

    /// <inheritdoc/>
    public virtual void TensorLogNormalInPlace<T>(Tensor<T> tensor, double mean = 1, double std = 2, int? seed = null)
        => ApplyInPlace(tensor, "TensorLogNormalInPlace", (e, x) => e.TensorLogNormal<T>(x._shape, mean, std, seed));

    /// <inheritdoc/>
    public virtual void TensorNormalInPlace<T>(Tensor<T> tensor, double mean = 0, double std = 1, int? seed = null)
    {
        var numOps = MathHelper.GetNumericOperations<T>();
        ApplyInPlace(tensor, "TensorNormalInPlace", (e, x) =>
            e.TensorAddScalar(e.TensorMultiplyScalar(e.TensorRandnLike(x, seed), numOps.FromDouble(std)), numOps.FromDouble(mean)));
    }

    /// <inheritdoc/>
    public virtual void TensorUniformInPlace<T>(Tensor<T> tensor, double low = 0, double high = 1, int? seed = null)
        => ApplyInPlace(tensor, "TensorUniformInPlace", (e, x) => e.TensorUniform<T>(x._shape, low, high, seed));

    /// <inheritdoc/>
    public virtual void TensorRandomIntegersInPlace<T>(Tensor<T> tensor, long low, long high, int? seed = null)
        => ApplyInPlace(tensor, "TensorRandomIntegersInPlace", (e, x) => e.TensorRandint<T>(low, high, x._shape, seed));

    /// <inheritdoc/>
    public virtual void TensorRandomIntegersInPlace<T>(Tensor<T> tensor, long high, int? seed = null)
        => TensorRandomIntegersInPlace(tensor, 0, high, seed);

    // ---- layout ---------------------------------------------------------------------------------------------------

    /// <inheritdoc/>
    public virtual void TensorSqueezeInPlace<T>(Tensor<T> tensor, int axis = -1)
        => ApplyLayoutInPlace(tensor, "TensorSqueezeInPlace", (e, x) => e.TensorSqueeze(x, axis));

    /// <inheritdoc/>
    public virtual void TensorUnsqueezeInPlace<T>(Tensor<T> tensor, int axis)
        => ApplyLayoutInPlace(tensor, "TensorUnsqueezeInPlace", (e, x) => e.TensorExpandDims(x, axis));

    /// <inheritdoc/>
    public virtual void TensorSwapAxesInPlace<T>(Tensor<T> tensor, int axis1, int axis2)
        => ApplyLayoutInPlace(tensor, "TensorSwapAxesInPlace", (e, x) => e.TensorSwapAxes(x, axis1, axis2));

    /// <inheritdoc/>
    public virtual void TensorTransposeInPlace<T>(Tensor<T> tensor, int dim0, int dim1)
        => ApplyLayoutInPlace(tensor, "TensorTransposeInPlace", (e, x) => e.TensorSwapAxes(x, dim0, dim1));

    /// <inheritdoc/>
    public virtual void TensorTInPlace<T>(Tensor<T> tensor)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank > 2) throw new ArgumentException("t_ expects a tensor of at most 2 dimensions.", nameof(tensor));
        if (tensor.Rank < 2) return;
        ApplyLayoutInPlace(tensor, "TensorTInPlace", (e, x) => e.TensorSwapAxes(x, 0, 1));
    }

    /// <inheritdoc/>
    public virtual void TensorAsStridedInPlace<T>(Tensor<T> tensor, int[] size, int[] stride, int? storageOffset = null)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (size is null) throw new ArgumentNullException(nameof(size));
        if (stride is null) throw new ArgumentNullException(nameof(stride));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.Stateful, "TensorAsStridedInPlace");
        // A view of the tensor's own storage; its constructor checks the layout stays inside the storage.
        var view = new Tensor<T>(tensor.DataVector, size, stride, storageOffset ?? tensor._storageOffset, tensor._storage);
        tensor.AdoptFrom(view);
    }

    /// <inheritdoc/>
    public virtual void TensorSetInPlace<T>(Tensor<T> tensor, Tensor<T> source)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (source is null) throw new ArgumentNullException(nameof(source));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.Stateful, "TensorSetInPlace");
        tensor.AdoptFrom(source);
    }

    /// <inheritdoc/>
    public virtual void TensorSetInPlace<T>(Tensor<T> tensor, Tensor<T> source, int storageOffset, int[] size, int[] stride)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (source is null) throw new ArgumentNullException(nameof(source));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.Stateful, "TensorSetInPlace");
        tensor.AdoptFrom(new Tensor<T>(source.DataVector, size, stride, storageOffset, source._storage));
    }

    /// <inheritdoc/>
    public virtual void TensorResizeInPlace<T>(Tensor<T> tensor, int[] shape)
    {
        if (tensor is null) throw new ArgumentNullException(nameof(tensor));
        if (shape is null) throw new ArgumentNullException(nameof(shape));
        GraphMode.ThrowIfActiveUnsupported(GraphCaptureLimitation.Stateful, "TensorResizeInPlace");
        long count = 1;
        foreach (int d in shape)
        {
            if (d < 0) throw new ArgumentException("A dimension cannot be negative.", nameof(shape));
            count *= d;
        }
        // torch keeps the storage when the new size fits in it from the current offset, and grows it otherwise; the
        // elements already there keep their values in row-major order. (New elements are zero here, uninitialized in
        // torch.)
        var strides = new int[shape.Length];
        for (int d = shape.Length - 1, s = 1; d >= 0; d--) { strides[d] = s; s *= Math.Max(shape[d], 1); }
        if (tensor.IsContiguous && tensor._storageOffset + count <= tensor.DataVector.Length)
        {
            tensor.AdoptFrom(new Tensor<T>(tensor.DataVector, shape, strides, tensor._storageOffset, tensor._storage));
            return;
        }
        var grown = new Tensor<T>(shape);
        var from = tensor.IsContiguous ? tensor : tensor.Contiguous();
        int keep = (int)Math.Min(from.Length, count);
        from.AsSpan().Slice(0, keep).CopyTo(grown.AsWritableSpan());
        tensor.AdoptFrom(grown);
    }

    /// <inheritdoc/>
    public virtual void TensorResizeAsInPlace<T>(Tensor<T> tensor, Tensor<T> other)
    {
        if (other is null) throw new ArgumentNullException(nameof(other));
        TensorResizeInPlace(tensor, other._shape);
    }
}
