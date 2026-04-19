using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Interfaces;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// Reduction modes for <see cref="IEngine.TensorScatterReduce{T}"/>.
/// Mirrors the PyTorch string arg set of <c>torch.scatter_reduce</c>:
/// "sum", "prod", "mean", "amin", "amax".
/// </summary>
public enum ScatterReduceMode
{
    /// <summary>Sum values mapped to the same target slot.</summary>
    Sum,
    /// <summary>Multiply values mapped to the same target slot.</summary>
    Prod,
    /// <summary>Arithmetic mean over values mapped to the same target slot.</summary>
    Mean,
    /// <summary>Minimum over values mapped to the same target slot.</summary>
    AMin,
    /// <summary>Maximum over values mapped to the same target slot.</summary>
    AMax
}

/// <summary>
/// Parity-210 op additions — movement, cumulative, comparison, clamp,
/// special math, element-wise binary, indexing completeness. Kept in a
/// separate partial to avoid bloating CpuEngine.cs.
/// </summary>
public partial class CpuEngine
{
    // ==================================================================
    // Movement ops
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRoll<T>(Tensor<T> tensor, int[] shifts, int[] axes)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (shifts == null) throw new ArgumentNullException(nameof(shifts));
        if (axes == null) throw new ArgumentNullException(nameof(axes));
        if (shifts.Length != axes.Length)
            throw new ArgumentException("shifts and axes must have the same length");

        int rank = tensor.Rank;
        var shape = tensor._shape;

        // Per-axis effective shift, normalised into [0, dim).
        var effShift = new int[rank];
        foreach (var a in axes) if (a < 0 || a >= rank)
            throw new ArgumentOutOfRangeException(nameof(axes), $"axis {a} out of range [0, {rank})");
        for (int i = 0; i < axes.Length; i++)
        {
            int a = axes[i];
            int d = shape[a];
            if (d == 0) continue;
            int s = shifts[i] % d;
            if (s < 0) s += d;
            effShift[a] = (effShift[a] + s) % d;
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        // Iterate every index; destination position = (i_k + shift_k) mod d_k.
        var idx = new int[rank];
        int total = tensor.Length;
        var strides = ComputeRowMajorStrides(shape);
        for (int linear = 0; linear < total; linear++)
        {
            // Compute destination linear index applying rolls.
            int dstLinear = 0;
            for (int k = 0; k < rank; k++)
            {
                int newIdx = idx[k] + effShift[k];
                if (newIdx >= shape[k]) newIdx -= shape[k];
                dstLinear += newIdx * strides[k];
            }
            dst[dstLinear] = src[linear];
            // increment idx
            for (int k = rank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorRoll", result, tensor,
            BackwardFunctions<T>.RollBackward,
            savedState: new object[] { (int[])shifts.Clone(), (int[])axes.Clone() });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFlip<T>(Tensor<T> tensor, int[] axes)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (axes == null) throw new ArgumentNullException(nameof(axes));

        int rank = tensor.Rank;
        foreach (var a in axes) if (a < 0 || a >= rank)
            throw new ArgumentOutOfRangeException(nameof(axes), $"axis {a} out of range");
        var flipSet = new System.Collections.Generic.HashSet<int>(axes);

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var shape = tensor._shape;
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        var idx = new int[rank];
        int total = tensor.Length;
        var strides = ComputeRowMajorStrides(shape);
        for (int linear = 0; linear < total; linear++)
        {
            int dstLinear = 0;
            for (int k = 0; k < rank; k++)
            {
                int newIdx = flipSet.Contains(k) ? shape[k] - 1 - idx[k] : idx[k];
                dstLinear += newIdx * strides[k];
            }
            dst[dstLinear] = src[linear];
            for (int k = rank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorFlip", result, tensor,
            BackwardFunctions<T>.FlipBackward,
            savedState: new object[] { (int[])axes.Clone() });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRepeatInterleave<T>(Tensor<T> tensor, int repeats, int dim)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (repeats < 1) throw new ArgumentOutOfRangeException(nameof(repeats), "repeats must be >= 1");
        int rank = tensor.Rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var inShape = tensor._shape;
        var outShape = (int[])inShape.Clone();
        outShape[dim] = inShape[dim] * repeats;

        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(outShape);
        var dst = result.AsWritableSpan();

        // Iterate over the output's index space, map each to the source.
        var idx = new int[rank];
        int total = result.Length;
        var srcStrides = ComputeRowMajorStrides(inShape);
        for (int linear = 0; linear < total; linear++)
        {
            // Source index = same, except dim is idx[dim] / repeats.
            int srcLinear = 0;
            for (int k = 0; k < rank; k++)
            {
                int i = (k == dim) ? idx[k] / repeats : idx[k];
                srcLinear += i * srcStrides[k];
            }
            dst[linear] = src[srcLinear];
            for (int k = rank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < outShape[k]) break;
                idx[k] = 0;
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorRepeatInterleave", result, tensor,
            BackwardFunctions<T>.RepeatInterleaveBackward,
            savedState: new object[] { repeats, dim });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFliplr<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank < 2) throw new ArgumentException("Fliplr requires a tensor with at least 2 dimensions");
        // Flip along the last axis ("left/right" in matrix-convention).
        return TensorFlip(tensor, new[] { tensor.Rank - 1 });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFlipud<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank < 1) throw new ArgumentException("Flipud requires a tensor with at least 1 dimension");
        // Flip along the first axis ("up/down").
        return TensorFlip(tensor, new[] { 0 });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRot90<T>(Tensor<T> tensor, int k = 1, int[]? axes = null)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        axes ??= new[] { 0, 1 };
        if (axes.Length != 2) throw new ArgumentException("Rot90 requires exactly 2 axes");
        if (axes[0] == axes[1]) throw new ArgumentException("Rot90 axes must be different");

        // k mod 4; if k<0, adjust to positive rotation count.
        int steps = ((k % 4) + 4) % 4;
        if (steps == 0) return (Tensor<T>)tensor.Clone();

        // Canonical rotation: k=1 ≡ transpose(axes) then flip(axes[0]);
        //                    k=2 ≡ flip(axes[0]) then flip(axes[1]);
        //                    k=3 ≡ flip(axes[0]) then transpose(axes).
        var result = tensor;
        int a0 = axes[0], a1 = axes[1];
        if (a0 < 0) a0 += tensor.Rank;
        if (a1 < 0) a1 += tensor.Rank;

        for (int s = 0; s < steps; s++)
        {
            // One 90° rotation = swap the two axes + flip the (new) axes[1].
            var perm = new int[result.Rank];
            for (int i = 0; i < result.Rank; i++) perm[i] = i;
            perm[a0] = a1;
            perm[a1] = a0;
            result = result.Transpose(perm);
            result = TensorFlip(result, new[] { a0 });
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSwapAxes<T>(Tensor<T> tensor, int axis1, int axis2)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (axis1 < 0) axis1 += rank;
        if (axis2 < 0) axis2 += rank;
        if (axis1 < 0 || axis1 >= rank) throw new ArgumentOutOfRangeException(nameof(axis1));
        if (axis2 < 0 || axis2 >= rank) throw new ArgumentOutOfRangeException(nameof(axis2));
        var perm = new int[rank];
        for (int i = 0; i < rank; i++) perm[i] = i;
        perm[axis1] = axis2;
        perm[axis2] = axis1;
        return tensor.Transpose(perm);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMoveDim<T>(Tensor<T> tensor, int source, int destination)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (source < 0) source += rank;
        if (destination < 0) destination += rank;
        if (source < 0 || source >= rank) throw new ArgumentOutOfRangeException(nameof(source));
        if (destination < 0 || destination >= rank) throw new ArgumentOutOfRangeException(nameof(destination));

        // Build permutation that "pulls out" source and inserts at destination.
        var perm = new int[rank];
        int w = 0;
        for (int i = 0; i < rank; i++)
        {
            if (w == destination) { perm[w++] = source; }
            if (i != source)
            {
                if (w == destination) { perm[w++] = source; }
                perm[w++] = i;
            }
        }
        if (w < rank) perm[w++] = source;
        return tensor.Transpose(perm);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAtLeast1D<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        return tensor.Rank >= 1 ? tensor : tensor.Reshape(new[] { 1 });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAtLeast2D<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank >= 2) return tensor;
        if (tensor.Rank == 0) return tensor.Reshape(new[] { 1, 1 });
        // Rank 1: make it a row vector [1, N].
        return tensor.Reshape(new[] { 1, tensor._shape[0] });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAtLeast3D<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank >= 3) return tensor;
        if (tensor.Rank == 0) return tensor.Reshape(new[] { 1, 1, 1 });
        if (tensor.Rank == 1) return tensor.Reshape(new[] { 1, tensor._shape[0], 1 });
        // Rank 2: insert leading 1.
        return tensor.Reshape(new[] { 1, tensor._shape[0], tensor._shape[1] });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHStack<T>(Tensor<T>[] tensors)
    {
        if (tensors == null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));
        // torch.hstack: concat along axis 0 for 1-D, axis 1 for higher ranks.
        int axis = tensors[0].Rank == 1 ? 0 : 1;
        return TensorConcatenate(tensors, axis);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorVStack<T>(Tensor<T>[] tensors)
    {
        if (tensors == null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));
        // torch.vstack: promote 1-D to 2-D rows and concat along axis 0.
        var promoted = new Tensor<T>[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
            promoted[i] = tensors[i].Rank == 1
                ? tensors[i].Reshape(new[] { 1, tensors[i]._shape[0] })
                : tensors[i];
        return TensorConcatenate(promoted, 0);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDStack<T>(Tensor<T>[] tensors)
    {
        if (tensors == null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));
        // torch.dstack: concat along axis 2 after promoting each tensor to at-least-3D.
        var promoted = new Tensor<T>[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
            promoted[i] = TensorAtLeast3D(tensors[i]);
        return TensorConcatenate(promoted, 2);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorColumnStack<T>(Tensor<T>[] tensors)
    {
        if (tensors == null || tensors.Length == 0) throw new ArgumentNullException(nameof(tensors));
        // torch.column_stack: 1-D tensors become columns of a 2-D matrix;
        // ≥2-D tensors concat along axis 1.
        var promoted = new Tensor<T>[tensors.Length];
        for (int i = 0; i < tensors.Length; i++)
            promoted[i] = tensors[i].Rank == 1
                ? tensors[i].Reshape(new[] { tensors[i]._shape[0], 1 })
                : tensors[i];
        return TensorConcatenate(promoted, 1);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRowStack<T>(Tensor<T>[] tensors)
        => TensorVStack(tensors);  // torch.row_stack is an alias for vstack.

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorHSplit<T>(Tensor<T> tensor, int sections)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int axis = tensor.Rank == 1 ? 0 : 1;
        return TensorSplit(tensor, sections, axis);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorVSplit<T>(Tensor<T> tensor, int sections)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank < 2) throw new ArgumentException("VSplit requires rank >= 2");
        return TensorSplit(tensor, sections, 0);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorDSplit<T>(Tensor<T> tensor, int sections)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank < 3) throw new ArgumentException("DSplit requires rank >= 3");
        return TensorSplit(tensor, sections, 2);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBroadcastTo<T>(Tensor<T> tensor, int[] shape)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (shape == null) throw new ArgumentNullException(nameof(shape));
        int srcRank = tensor.Rank;
        int dstRank = shape.Length;
        if (dstRank < srcRank)
            throw new ArgumentException("Broadcast target rank must be >= source rank");

        // Right-align source shape against target. A source dim must either
        // equal the target dim or be 1 (broadcast).
        var padded = new int[dstRank];
        int offset = dstRank - srcRank;
        for (int i = 0; i < dstRank; i++)
        {
            int srcDim = i < offset ? 1 : tensor._shape[i - offset];
            if (srcDim != shape[i] && srcDim != 1)
                throw new ArgumentException(
                    $"Cannot broadcast dim {srcDim} (source idx {i - offset}) to {shape[i]}");
            padded[i] = srcDim;
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();
        var src = tensor.AsSpan();

        int outTotal = result.Length;
        var srcStrides = new int[dstRank];
        // Compute source strides treating broadcast (size-1) dims as stride 0.
        int stride = 1;
        for (int i = dstRank - 1; i >= 0; i--)
        {
            srcStrides[i] = padded[i] == 1 ? 0 : stride;
            stride *= padded[i];
        }
        var idx = new int[dstRank];
        for (int linear = 0; linear < outTotal; linear++)
        {
            int srcPos = 0;
            for (int k = 0; k < dstRank; k++) srcPos += idx[k] * srcStrides[k];
            dst[linear] = src[srcPos];
            for (int k = dstRank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < shape[k]) break;
                idx[k] = 0;
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTake<T>(Tensor<T> tensor, Tensor<int> indices)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var idx = indices.AsSpan();
        var result = new Tensor<T>(indices._shape);
        var dst = result.AsWritableSpan();
        int total = src.Length;
        for (int i = 0; i < idx.Length; i++)
        {
            int pos = idx[i];
            if (pos < 0 || pos >= total)
                throw new IndexOutOfRangeException(
                    $"indices[{i}]={pos} out of range for flattened tensor length {total}");
            dst[i] = src[pos];
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTakeAlongDim<T>(Tensor<T> tensor, Tensor<int> indices, int dim)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        if (indices.Rank != rank)
            throw new ArgumentException("indices must have the same rank as tensor");
        for (int k = 0; k < rank; k++)
        {
            if (k != dim && indices._shape[k] != tensor._shape[k])
                throw new ArgumentException(
                    $"indices.shape[{k}]={indices._shape[k]} must match tensor.shape[{k}]={tensor._shape[k]}");
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var idx = indices.AsSpan();
        var result = new Tensor<T>(indices._shape);
        var dst = result.AsWritableSpan();

        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int srcAxis = tensor._shape[dim];
        int idxAxis = indices._shape[dim];

        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < idxAxis; i++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int idxPos = outer * idxAxis * innerSize + i * innerSize + inner;
                    int target = idx[idxPos];
                    if (target < 0 || target >= srcAxis)
                        throw new IndexOutOfRangeException(
                            $"indices out of range at linear {idxPos}");
                    int srcPos = outer * srcAxis * innerSize + target * innerSize + inner;
                    dst[idxPos] = src[srcPos];
                }
        return result;
    }

    // ==================================================================
    // Cumulative ops
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumProd<T>(Tensor<T> tensor, int axis)
        => CumulativeAlongAxis(tensor, axis, MathHelper.GetNumericOperations<T>().One,
            (a, b) => MathHelper.GetNumericOperations<T>().Multiply(a, b),
            "TensorCumProd", BackwardFunctions<T>.CumProdBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumMax<T>(Tensor<T> tensor, int axis)
        => CumulativeAlongAxis(tensor, axis,
            CumulativeInitial<T>(max: true),
            (a, b) => {
                var ops = MathHelper.GetNumericOperations<T>();
                return ops.GreaterThan(a, b) ? a : b;
            },
            "TensorCumMax", BackwardFunctions<T>.CumMaxBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumMin<T>(Tensor<T> tensor, int axis)
        => CumulativeAlongAxis(tensor, axis,
            CumulativeInitial<T>(max: false),
            (a, b) => {
                var ops = MathHelper.GetNumericOperations<T>();
                return ops.LessThan(a, b) ? a : b;
            },
            "TensorCumMin", BackwardFunctions<T>.CumMinBackward);

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogCumSumExp<T>(Tensor<T> tensor, int axis)
    {
        // log-sum-exp cumulative scan: running m = max, running s = sum exp(x-m)
        // result[i] = m + log(s)
        // Numerically stable across wide dynamic ranges.
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var shape = tensor._shape;
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        var strides = ComputeRowMajorStrides(shape);
        int axisLen = shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= shape[k];
        int axisStride = strides[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int basePos = outer * axisLen * innerSize;
            for (int inner = 0; inner < innerSize; inner++)
            {
                T runningMax = default!;
                T runningSum = ops.Zero;
                bool first = true;
                for (int i = 0; i < axisLen; i++)
                {
                    int pos = basePos + i * axisStride + inner;
                    T x = src[pos];
                    if (first)
                    {
                        runningMax = x;
                        runningSum = ops.One; // exp(x - x) = 1
                        first = false;
                    }
                    else if (ops.GreaterThan(x, runningMax))
                    {
                        // scale existing sum by exp(old_max - new_max)
                        var scale = ops.Exp(ops.Subtract(runningMax, x));
                        runningSum = ops.Add(ops.Multiply(runningSum, scale), ops.One);
                        runningMax = x;
                    }
                    else
                    {
                        runningSum = ops.Add(runningSum, ops.Exp(ops.Subtract(x, runningMax)));
                    }
                    dst[pos] = ops.Add(runningMax, ops.Log(runningSum));
                }
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorLogCumSumExp", result, tensor,
            BackwardFunctions<T>.LogCumSumExpBackward,
            savedState: new object[] { axis });
        return result;
    }

    // ==================================================================
    // Comparison / predicate ops
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorIsClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException($"IsClose requires matching shapes");

        var ops = MathHelper.GetNumericOperations<T>();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        var result = new Bit[a.Length];
        for (int i = 0; i < a.Length; i++)
        {
            bool aNan = ops.IsNaN(av[i]);
            bool bNan = ops.IsNaN(bv[i]);
            if (aNan || bNan)
            {
                result[i] = (equalNan && aNan && bNan) ? Bit.True : Bit.False;
                continue;
            }
            var diff = ops.Abs(ops.Subtract(av[i], bv[i]));
            var tol = ops.Add(atol, ops.Multiply(rtol, ops.Abs(bv[i])));
            result[i] = ops.LessThanOrEquals(diff, tol) ? Bit.True : Bit.False;
        }
        return new Tensor<Bit>(result, a._shape);
    }

    /// <inheritdoc/>
    public virtual bool TensorAllClose<T>(Tensor<T> a, Tensor<T> b, T rtol, T atol, bool equalNan = false)
    {
        var mask = TensorIsClose(a, b, rtol, atol, equalNan);
        var span = mask.AsSpan();
        for (int i = 0; i < span.Length; i++)
            if (!(bool)span[i]) return false;
        return true;
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorIsIn<T>(Tensor<T> elements, Tensor<T> testElements, bool invert = false)
    {
        if (elements == null) throw new ArgumentNullException(nameof(elements));
        if (testElements == null) throw new ArgumentNullException(nameof(testElements));

        var set = new System.Collections.Generic.HashSet<T>(testElements.AsSpan().ToArray());
        if (!elements.IsContiguous) elements = elements.Contiguous();
        var src = elements.AsSpan();
        var result = new Bit[elements.Length];
        for (int i = 0; i < src.Length; i++)
        {
            bool contained = set.Contains(src[i]);
            result[i] = (contained ^ invert) ? Bit.True : Bit.False;
        }
        return new Tensor<Bit>(result, elements._shape);
    }

    // ==================================================================
    // Clamp family
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorClampMin<T>(Tensor<T> tensor, T min)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.LessThan(src[i], min) ? min : src[i];

        // Box min as object; T can be any numeric here so a direct cast is
        // safe. The backward unboxes via savedState[0].
        var minBox = (object?)min ?? throw new InvalidOperationException(
            "Clamp min must not be null");
        DifferentiableOps.RecordUnary(
            "TensorClampMin", result, tensor,
            BackwardFunctions<T>.ClampMinBackward,
            savedState: new[] { minBox });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorClampMax<T>(Tensor<T> tensor, T max)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
            dst[i] = ops.GreaterThan(src[i], max) ? max : src[i];

        var maxBox = (object?)max ?? throw new InvalidOperationException(
            "Clamp max must not be null");
        DifferentiableOps.RecordUnary(
            "TensorClampMax", result, tensor,
            BackwardFunctions<T>.ClampMaxBackward,
            savedState: new[] { maxBox });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorClampTensor<T>(Tensor<T> tensor, Tensor<T>? min, Tensor<T>? max)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (min is null && max is null)
            throw new ArgumentException("At least one of min / max must be supplied");

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        // For now, require exact-shape bounds (no broadcasting). A broadcasting
        // variant can layer on top via BroadcastTensors.
        if (min is not null && !min._shape.SequenceEqual(tensor._shape))
            throw new ArgumentException("min shape must match tensor shape (broadcasting TBD)");
        if (max is not null && !max._shape.SequenceEqual(tensor._shape))
            throw new ArgumentException("max shape must match tensor shape (broadcasting TBD)");

        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        var minSpan = min is null ? default : (min.IsContiguous ? min : min.Contiguous()).AsSpan();
        var maxSpan = max is null ? default : (max.IsContiguous ? max : max.Contiguous()).AsSpan();

        for (int i = 0; i < src.Length; i++)
        {
            var v = src[i];
            if (min is not null && ops.LessThan(v, minSpan[i])) v = minSpan[i];
            if (max is not null && ops.GreaterThan(v, maxSpan[i])) v = maxSpan[i];
            dst[i] = v;
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSelectScatter<T>(
        Tensor<T> tensor, Tensor<T> source, int dim, int index)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        int axisSize = tensor._shape[dim];
        if (index < 0) index += axisSize;
        if (index < 0 || index >= axisSize) throw new ArgumentOutOfRangeException(nameof(index));

        // Source should have rank = tensor.rank - 1, matching tensor shape with
        // dim dropped.
        if (source.Rank != rank - 1)
            throw new ArgumentException($"source rank {source.Rank} must be tensor rank - 1 ({rank - 1})");
        int srcDim = 0;
        for (int k = 0; k < rank; k++)
        {
            if (k == dim) continue;
            if (source._shape[srcDim] != tensor._shape[k])
                throw new ArgumentException(
                    $"source.shape[{srcDim}]={source._shape[srcDim]} must match tensor.shape[{k}]={tensor._shape[k]}");
            srcDim++;
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var src = source.AsSpan();

        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= tensor._shape[k];

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                int dstPos = outer * axisSize * innerSize + index * innerSize + inner;
                int srcPos = outer * innerSize + inner;
                dst[dstPos] = src[srcPos];
            }
        return result;
    }

    /// <inheritdoc/>
    public virtual (T Min, T Max) TensorAminmax<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Length == 0) throw new ArgumentException("Aminmax requires a non-empty tensor");
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        T min = src[0], max = src[0];
        for (int i = 1; i < src.Length; i++)
        {
            if (ops.LessThan(src[i], min)) min = src[i];
            if (ops.GreaterThan(src[i], max)) max = src[i];
        }
        return (min, max);
    }

    // ==================================================================
    // Indexing family
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexAdd<T>(
        Tensor<T> tensor, int axis, Tensor<int> indices, Tensor<T> source)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (indices.Rank != 1) throw new ArgumentException("indices must be 1-D");
        if (source._shape[axis] != indices._shape[0])
            throw new ArgumentException(
                $"source.shape[{axis}]={source._shape[axis]} must match indices.length={indices._shape[0]}");
        for (int k = 0; k < rank; k++)
        {
            if (k != axis && source._shape[k] != tensor._shape[k])
                throw new ArgumentException(
                    $"source.shape[{k}]={source._shape[k]} must match tensor.shape[{k}]={tensor._shape[k]}");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var srcData = source.AsSpan();
        var idxData = indices.AsSpan();

        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int dstAxis = tensor._shape[axis];
        int srcAxis = source._shape[axis];
        int dstAxisStride = innerSize;
        int srcAxisStride = innerSize;

        for (int outer = 0; outer < outerSize; outer++)
        {
            int dstOuter = outer * dstAxis * dstAxisStride;
            int srcOuter = outer * srcAxis * srcAxisStride;
            for (int i = 0; i < idxData.Length; i++)
            {
                int target = idxData[i];
                if (target < 0 || target >= dstAxis)
                    throw new IndexOutOfRangeException(
                        $"indices[{i}]={target} out of range for axis size {dstAxis}");
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int dstPos = dstOuter + target * dstAxisStride + inner;
                    int srcPos = srcOuter + i * srcAxisStride + inner;
                    dst[dstPos] = ops.Add(dst[dstPos], srcData[srcPos]);
                }
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexPut<T>(
        Tensor<T> tensor, Tensor<int>[] indices, Tensor<T> source, bool accumulate = false)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        if (indices.Length != rank)
            throw new ArgumentException(
                $"IndexPut expects one index tensor per axis ({rank}); got {indices.Length}");

        // Every index tensor must be 1-D and the same length. Source length
        // must match the index length.
        int n = indices[0]?.Length ?? 0;
        for (int k = 0; k < rank; k++)
        {
            if (indices[k] == null) throw new ArgumentNullException(nameof(indices));
            if (indices[k].Rank != 1)
                throw new ArgumentException($"indices[{k}] must be 1-D");
            if (indices[k].Length != n)
                throw new ArgumentException($"indices[{k}].Length={indices[k].Length} must match indices[0].Length={n}");
        }
        if (source.Length != n)
            throw new ArgumentException($"source length {source.Length} must match index length {n}");

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var srcData = source.AsSpan();

        // Contiguous row-major strides over tensor.Shape.
        var strides = ComputeRowMajorStrides(tensor._shape);
        for (int i = 0; i < n; i++)
        {
            int pos = 0;
            for (int k = 0; k < rank; k++)
            {
                int idx = indices[k][i];
                if (idx < 0 || idx >= tensor._shape[k])
                    throw new IndexOutOfRangeException(
                        $"indices[{k}][{i}]={idx} out of range for axis size {tensor._shape[k]}");
                pos += idx * strides[k];
            }
            dst[pos] = accumulate ? ops.Add(dst[pos], srcData[i]) : srcData[i];
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexCopy<T>(
        Tensor<T> tensor, int axis, Tensor<int> indices, Tensor<T> source)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (indices.Rank != 1) throw new ArgumentException("indices must be 1-D");
        if (source._shape[axis] != indices._shape[0])
            throw new ArgumentException(
                $"source.shape[{axis}]={source._shape[axis]} must match indices.length={indices._shape[0]}");
        for (int k = 0; k < rank; k++)
        {
            if (k != axis && source._shape[k] != tensor._shape[k])
                throw new ArgumentException(
                    $"source.shape[{k}]={source._shape[k]} must match tensor.shape[{k}]={tensor._shape[k]}");
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var srcData = source.AsSpan();
        var idxData = indices.AsSpan();

        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int dstAxis = tensor._shape[axis];
        int srcAxis = source._shape[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int dstOuter = outer * dstAxis * innerSize;
            int srcOuter = outer * srcAxis * innerSize;
            for (int i = 0; i < idxData.Length; i++)
            {
                int target = idxData[i];
                if (target < 0 || target >= dstAxis)
                    throw new IndexOutOfRangeException(
                        $"indices[{i}]={target} out of range for axis size {dstAxis}");
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int dstPos = dstOuter + target * innerSize + inner;
                    int srcPos = srcOuter + i * innerSize + inner;
                    dst[dstPos] = srcData[srcPos];
                }
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorExpandAs<T>(Tensor<T> tensor, Tensor<T> other)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (other == null) throw new ArgumentNullException(nameof(other));
        return TensorBroadcastTo(tensor, other._shape);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorBroadcastTensors<T>(Tensor<T>[] tensors)
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length == 0) return System.Array.Empty<Tensor<T>>();

        // Compute the common broadcast shape: right-align, size-1 ↔ any,
        // mismatch otherwise.
        int maxRank = 0;
        foreach (var t in tensors)
        {
            if (t == null) throw new ArgumentNullException(nameof(tensors));
            if (t.Rank > maxRank) maxRank = t.Rank;
        }
        var broadcast = new int[maxRank];
        for (int i = 0; i < maxRank; i++) broadcast[i] = 1;
        foreach (var t in tensors)
        {
            int offset = maxRank - t.Rank;
            for (int i = 0; i < t.Rank; i++)
            {
                int d = t._shape[i];
                int target = broadcast[offset + i];
                if (target == 1) broadcast[offset + i] = d;
                else if (d != 1 && d != target)
                    throw new ArgumentException(
                        $"Cannot broadcast dim {d} against {target} at position {offset + i}");
            }
        }

        // Broadcast every input to `broadcast`.
        var result = new Tensor<T>[tensors.Length];
        for (int k = 0; k < tensors.Length; k++)
            result[k] = TensorBroadcastTo(tensors[k], broadcast);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorUniqueConsecutive<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        if (src.Length == 0) return new Tensor<T>(new T[0], new[] { 0 });

        var ops = MathHelper.GetNumericOperations<T>();
        var keep = new System.Collections.Generic.List<T>();
        keep.Add(src[0]);
        for (int i = 1; i < src.Length; i++)
        {
            if (!ops.Equals(src[i], src[i - 1])) keep.Add(src[i]);
        }
        var arr = keep.ToArray();
        return new Tensor<T>(arr, new[] { arr.Length });
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorBlockDiag<T>(Tensor<T>[] matrices)
    {
        if (matrices == null) throw new ArgumentNullException(nameof(matrices));
        if (matrices.Length == 0) throw new ArgumentException("BlockDiag requires at least one matrix");

        var ops = MathHelper.GetNumericOperations<T>();
        int totalRows = 0, totalCols = 0;
        foreach (var m in matrices)
        {
            if (m == null) throw new ArgumentNullException(nameof(matrices));
            if (m.Rank != 2) throw new ArgumentException("BlockDiag requires 2-D matrices");
            totalRows += m._shape[0];
            totalCols += m._shape[1];
        }

        var result = AutoTensorCache.RentOrAllocate<T>(new[] { totalRows, totalCols });
        var dst = result.AsWritableSpan();
        var zero = ops.Zero;
        for (int i = 0; i < dst.Length; i++) dst[i] = zero;

        int rowOffset = 0, colOffset = 0;
        foreach (var m in matrices)
        {
            var contig = m.IsContiguous ? m : m.Contiguous();
            var src = contig.AsSpan();
            int r = m._shape[0], c = m._shape[1];
            for (int i = 0; i < r; i++)
                for (int j = 0; j < c; j++)
                    dst[(rowOffset + i) * totalCols + (colOffset + j)] = src[i * c + j];
            rowOffset += r;
            colOffset += c;
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorSliceScatter<T>(
        Tensor<T> tensor, Tensor<T> source, int dim, int start, int length)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        if (start < 0 || start + length > tensor._shape[dim])
            throw new ArgumentOutOfRangeException(
                $"slice[{dim}] start={start} length={length} out of range for axis size {tensor._shape[dim]}");
        if (source._shape[dim] != length)
            throw new ArgumentException(
                $"source.shape[{dim}]={source._shape[dim]} must match length={length}");
        for (int k = 0; k < rank; k++)
        {
            if (k != dim && source._shape[k] != tensor._shape[k])
                throw new ArgumentException(
                    $"source.shape[{k}]={source._shape[k]} must match tensor.shape[{k}]={tensor._shape[k]}");
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var src = source.AsSpan();

        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int dstAxis = tensor._shape[dim];

        for (int outer = 0; outer < outerSize; outer++)
            for (int i = 0; i < length; i++)
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int dstPos = outer * dstAxis * innerSize + (start + i) * innerSize + inner;
                    int srcPos = outer * length * innerSize + i * innerSize + inner;
                    dst[dstPos] = src[srcPos];
                }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorIndexFill<T>(
        Tensor<T> tensor, int axis, Tensor<int> indices, T value)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));
        if (indices.Rank != 1) throw new ArgumentException("indices must be 1-D");

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var idxData = indices.AsSpan();

        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int axisSize = tensor._shape[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int outerBase = outer * axisSize * innerSize;
            for (int i = 0; i < idxData.Length; i++)
            {
                int target = idxData[i];
                if (target < 0 || target >= axisSize)
                    throw new IndexOutOfRangeException(
                        $"indices[{i}]={target} out of range for axis size {axisSize}");
                for (int inner = 0; inner < innerSize; inner++)
                    dst[outerBase + target * innerSize + inner] = value;
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorScatterReduce<T>(
        Tensor<T> tensor, int dim, Tensor<int> indices, Tensor<T> source,
        ScatterReduceMode mode, bool includeSelf = true)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        if (!indices._shape.SequenceEqual(source._shape))
            throw new ArgumentException("indices and source must have the same shape");
        if (indices.Rank != rank)
            throw new ArgumentException("indices/source must match tensor rank");
        for (int k = 0; k < rank; k++)
        {
            if (k != dim && indices._shape[k] != tensor._shape[k])
                throw new ArgumentException(
                    $"indices.shape[{k}]={indices._shape[k]} must match tensor.shape[{k}]={tensor._shape[k]}");
        }

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var srcData = source.AsSpan();
        var idxData = indices.AsSpan();

        // For mean mode, track per-target counts including (optionally) self.
        int[]? counts = mode == ScatterReduceMode.Mean ? new int[result.Length] : null;
        if (counts is not null && includeSelf)
            for (int i = 0; i < counts.Length; i++) counts[i] = 1;

        // If !includeSelf, wipe target positions that any index touches so
        // reduction starts from "no observations yet" at those slots.
        bool[]? touched = !includeSelf ? new bool[result.Length] : null;

        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= tensor._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= tensor._shape[k];
        int dstAxis = tensor._shape[dim];
        int srcAxis = source._shape[dim];

        // First pass: when !includeSelf, reset touched positions to identity.
        if (!includeSelf)
        {
            T identity = mode switch
            {
                ScatterReduceMode.Sum or ScatterReduceMode.Mean => ops.Zero,
                ScatterReduceMode.Prod => ops.One,
                ScatterReduceMode.AMin => ops.MaxValue,
                ScatterReduceMode.AMax => ops.MinValue,
                _ => ops.Zero
            };
            for (int outer = 0; outer < outerSize; outer++)
            {
                int outerBase = outer * dstAxis * innerSize;
                for (int i = 0; i < srcAxis; i++)
                {
                    int baseIdx = outer * srcAxis * innerSize + i * innerSize;
                    for (int inner = 0; inner < innerSize; inner++)
                    {
                        int target = idxData[baseIdx + inner];
                        if (target < 0 || target >= dstAxis) continue;
                        int dstPos = outerBase + target * innerSize + inner;
                        if (!touched![dstPos])
                        {
                            dst[dstPos] = identity;
                            touched[dstPos] = true;
                            if (counts is not null) counts[dstPos] = 0;
                        }
                    }
                }
            }
        }

        // Main pass: apply reduction.
        for (int outer = 0; outer < outerSize; outer++)
        {
            int outerBase = outer * dstAxis * innerSize;
            for (int i = 0; i < srcAxis; i++)
            {
                int srcBase = outer * srcAxis * innerSize + i * innerSize;
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int target = idxData[srcBase + inner];
                    if (target < 0 || target >= dstAxis)
                        throw new IndexOutOfRangeException(
                            $"indices out of range at linear {srcBase + inner}");
                    int dstPos = outerBase + target * innerSize + inner;
                    T s = srcData[srcBase + inner];
                    switch (mode)
                    {
                        case ScatterReduceMode.Sum:
                        case ScatterReduceMode.Mean:
                            dst[dstPos] = ops.Add(dst[dstPos], s);
                            if (counts is not null) counts[dstPos]++;
                            break;
                        case ScatterReduceMode.Prod:
                            dst[dstPos] = ops.Multiply(dst[dstPos], s);
                            break;
                        case ScatterReduceMode.AMin:
                            if (ops.LessThan(s, dst[dstPos])) dst[dstPos] = s;
                            break;
                        case ScatterReduceMode.AMax:
                            if (ops.GreaterThan(s, dst[dstPos])) dst[dstPos] = s;
                            break;
                    }
                }
            }
        }

        if (mode == ScatterReduceMode.Mean && counts is not null)
        {
            for (int i = 0; i < dst.Length; i++)
                if (counts[i] > 0)
                    dst[i] = ops.Divide(dst[i], ops.FromDouble(counts[i]));
        }

        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMaskedScatter<T>(
        Tensor<T> tensor, Tensor<Bit> mask, Tensor<T> source)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (mask == null) throw new ArgumentNullException(nameof(mask));
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (!tensor._shape.SequenceEqual(mask._shape))
            throw new ArgumentException(
                $"mask shape [{string.Join(", ", mask._shape)}] must match tensor shape [{string.Join(", ", tensor._shape)}]");

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        if (!source.IsContiguous) source = source.Contiguous();

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var maskData = mask.AsSpan();
        var srcData = source.AsSpan();

        int sourceCursor = 0;
        for (int i = 0; i < maskData.Length; i++)
        {
            if ((bool)maskData[i])
            {
                if (sourceCursor >= srcData.Length)
                    throw new ArgumentException("source has fewer elements than mask-true count");
                dst[i] = srcData[sourceCursor++];
            }
        }
        return result;
    }

    // ==================================================================
    // Sort / order statistics
    // ==================================================================

    /// <inheritdoc/>
    public virtual (Tensor<T> Values, Tensor<int> Indices) TensorSort<T>(
        Tensor<T> input, int axis = -1, bool descending = false)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        int rank = input.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        var numOps = MathHelper.GetNumericOperations<T>();
        var valuesOut = AutoTensorCache.RentOrAllocate<T>(input._shape);
        var indicesOut = new Tensor<int>(input._shape);

        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        var valDst = valuesOut.AsWritableSpan();
        var idxDst = indicesOut.AsWritableSpan();

        int axisSize = input._shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= input._shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= input._shape[k];

        var buffer = new (T value, int index)[axisSize];
        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                int baseIdx = outer * axisSize * innerSize + inner;
                for (int a = 0; a < axisSize; a++)
                    buffer[a] = (src[baseIdx + a * innerSize], a);

                Array.Sort(buffer, descending
                    ? (x, y) => numOps.Compare(y.value, x.value)
                    : (Comparison<(T value, int index)>)((x, y) => numOps.Compare(x.value, y.value)));

                for (int a = 0; a < axisSize; a++)
                {
                    int pos = baseIdx + a * innerSize;
                    valDst[pos] = buffer[a].value;
                    idxDst[pos] = buffer[a].index;
                }
            }
        return (valuesOut, indicesOut);
    }

    /// <inheritdoc/>
    public virtual (T Value, int Index) TensorKthvalue<T>(Tensor<T> input, int k)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (k < 1 || k > input.Length) throw new ArgumentOutOfRangeException(nameof(k));

        var numOps = MathHelper.GetNumericOperations<T>();
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        var pairs = new (T value, int index)[src.Length];
        for (int i = 0; i < src.Length; i++) pairs[i] = (src[i], i);
        Array.Sort(pairs, (x, y) => numOps.Compare(x.value, y.value));
        return pairs[k - 1];
    }

    /// <inheritdoc/>
    public virtual T TensorMedian<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0) throw new ArgumentException("Median requires a non-empty tensor");
        // PyTorch returns the lower median for even-length; keep the same behaviour.
        int k = (input.Length + 1) / 2;
        var (value, _) = TensorKthvalue(input, k);
        return value;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorUnique<T>(Tensor<T> input, bool sorted = true)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();

        // HashSet<T> preserves first-insertion order in .NET; we sort afterwards if requested.
        var seen = new System.Collections.Generic.HashSet<T>();
        var order = new System.Collections.Generic.List<T>();
        for (int i = 0; i < src.Length; i++)
        {
            if (seen.Add(src[i])) order.Add(src[i]);
        }

        if (sorted)
        {
            var numOps = MathHelper.GetNumericOperations<T>();
            order.Sort((a, b) => numOps.Compare(a, b));
        }

        var outArr = order.ToArray();
        return new Tensor<T>(outArr, new[] { outArr.Length });
    }

    /// <inheritdoc/>
    public virtual Tensor<int> TensorSearchSorted<T>(
        Tensor<T> sortedSequence, Tensor<T> values, bool right = false)
    {
        if (sortedSequence == null) throw new ArgumentNullException(nameof(sortedSequence));
        if (values == null) throw new ArgumentNullException(nameof(values));
        if (sortedSequence.Rank != 1)
            throw new ArgumentException("SearchSorted expects a 1-D sorted sequence");

        var numOps = MathHelper.GetNumericOperations<T>();
        if (!sortedSequence.IsContiguous) sortedSequence = sortedSequence.Contiguous();
        if (!values.IsContiguous) values = values.Contiguous();
        var seq = sortedSequence.AsSpan();
        var vs = values.AsSpan();
        var result = new Tensor<int>(values._shape);
        var dst = result.AsWritableSpan();

        for (int i = 0; i < vs.Length; i++)
        {
            // Branchless-ready binary search; returns insertion index.
            int lo = 0, hi = seq.Length;
            var v = vs[i];
            while (lo < hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                // right=false → lower bound: split at seq[mid] >= v (v <= seq[mid]).
                // right=true  → upper bound: split at seq[mid] >  v (v <  seq[mid]).
                bool beforeMid = right
                    ? numOps.LessThan(v, seq[mid])
                    : numOps.LessThanOrEquals(v, seq[mid]);
                if (beforeMid) hi = mid; else lo = mid + 1;
            }
            dst[i] = lo;
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual T TensorNanMedian<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        // Filter NaNs; take lower-median of the remaining values.
        var kept = new System.Collections.Generic.List<T>(src.Length);
        for (int i = 0; i < src.Length; i++)
            if (!ops.IsNaN(src[i])) kept.Add(src[i]);
        if (kept.Count == 0) return ops.FromDouble(double.NaN);
        kept.Sort((a, b) => ops.Compare(a, b));
        int k = (kept.Count + 1) / 2;
        return kept[k - 1];
    }

    /// <inheritdoc/>
    public virtual (T Value, int Count) TensorMode<T>(Tensor<T> input)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Length == 0) throw new ArgumentException("Mode requires a non-empty tensor");
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        // Count occurrences; return the most frequent. Ties broken by smallest
        // value. Using a list of (value, count) pairs with linear search
        // avoids the Dictionary<T, …> notnull constraint on generic T — Mode
        // is typically called on small tensors so the O(N·U) cost (U = unique
        // values) is acceptable.
        var numOps = MathHelper.GetNumericOperations<T>();
        var counts = new System.Collections.Generic.List<(T value, int count)>();
        for (int i = 0; i < src.Length; i++)
        {
            int found = -1;
            for (int j = 0; j < counts.Count; j++)
            {
                if (numOps.Equals(counts[j].value, src[i])) { found = j; break; }
            }
            if (found < 0) counts.Add((src[i], 1));
            else counts[found] = (counts[found].value, counts[found].count + 1);
        }

        T bestValue = numOps.Zero;  // guaranteed overwritten below (input is non-empty)
        int bestCount = -1;
        foreach (var kv in counts)
        {
            if (bestCount < 0
                || kv.count > bestCount
                || (kv.count == bestCount && numOps.LessThan(kv.value, bestValue)))
            {
                bestValue = kv.value;
                bestCount = kv.count;
            }
        }
        return (bestValue, bestCount);
    }

    /// <inheritdoc/>
    public virtual Tensor<int> TensorBucketize<T>(Tensor<T> input, Tensor<T> boundaries, bool right = false)
        => TensorSearchSorted(boundaries, input, right);

    /// <inheritdoc/>
    public virtual Tensor<int> TensorHistogram<T>(Tensor<T> input, int bins, T min, T max)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (bins < 1) throw new ArgumentOutOfRangeException(nameof(bins));

        var numOps = MathHelper.GetNumericOperations<T>();
        if (numOps.GreaterThanOrEquals(min, max))
            throw new ArgumentException("Histogram requires min < max");
        if (!input.IsContiguous) input = input.Contiguous();

        var result = new Tensor<int>(new[] { bins });
        var dst = result.AsWritableSpan();
        var src = input.AsSpan();

        var width = numOps.Divide(numOps.Subtract(max, min), numOps.FromDouble(bins));
        for (int i = 0; i < src.Length; i++)
        {
            var v = src[i];
            if (numOps.LessThan(v, min) || numOps.GreaterThan(v, max)) continue;
            // Bin index: floor((v - min) / width). Clamp the last-bin edge so
            // the upper boundary maps into the final bin.
            int idx;
            if (numOps.Equals(v, max)) idx = bins - 1;
            else
            {
                var f = numOps.Divide(numOps.Subtract(v, min), width);
                idx = numOps.ToInt32(numOps.Floor(f));
                if (idx >= bins) idx = bins - 1;
                if (idx < 0) idx = 0;
            }
            dst[idx]++;
        }
        return result;
    }

    // ==================================================================
    // Element-wise binary math
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHypot<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Sqrt(ops.Add(ops.Multiply(ax, ax), ops.Multiply(bx, bx)));
        }, "TensorHypot");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCopysign<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            var sign = ops.SignOrZero(bx);
            var mag = ops.Abs(ax);
            // Treat sign==0 (zero b) as positive, matching IEEE copysign on +0.
            return ops.LessThan(sign, ops.Zero) ? ops.Negate(mag) : mag;
        }, "TensorCopysign");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFmod<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            // IEEE: result has same sign as dividend (a).
            var ops = MathHelper.GetNumericOperations<T>();
            if (ops.Equals(bx, ops.Zero)) return ops.Zero;
            var q = TruncateTowardZero(ops.Divide(ax, bx));
            return ops.Subtract(ax, ops.Multiply(q, bx));
        }, "TensorFmod");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRemainder<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) => {
            // Python-style: result has same sign as divisor (b).
            var ops = MathHelper.GetNumericOperations<T>();
            if (ops.Equals(bx, ops.Zero)) return ops.Zero;
            var q = ops.Floor(ops.Divide(ax, bx));
            return ops.Subtract(ax, ops.Multiply(q, bx));
        }, "TensorRemainder");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFloatPower<T>(Tensor<T> a, Tensor<T> b)
        => ElementwiseBinary(a, b, (ax, bx) =>
            MathHelper.GetNumericOperations<T>().Power(ax, bx), "TensorFloatPower");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLdexp<T>(Tensor<T> x, Tensor<int> exp)
    {
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (exp == null) throw new ArgumentNullException(nameof(exp));
        if (!x._shape.SequenceEqual(exp._shape))
            throw new ArgumentException("Ldexp requires matching shapes");
        var ops = MathHelper.GetNumericOperations<T>();
        if (!x.IsContiguous) x = x.Contiguous();
        if (!exp.IsContiguous) exp = exp.Contiguous();
        var src = x.AsSpan();
        var e = exp.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(x._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            // x * 2^e. Compute via Power(2.0, e) then multiply.
            double scale = System.Math.Pow(2.0, e[i]);
            dst[i] = ops.Multiply(src[i], ops.FromDouble(scale));
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorNextAfter<T>(Tensor<T> a, Tensor<T> b)
    {
        // Bit-level next-after. We dispatch on typeof(T) so fp32 stays in fp32
        // (avoids the trap where "next after 1.0 toward 2.0" in fp64 rounds
        // straight back to 1.0f when cast through T=float).
        return ElementwiseBinary(a, b, (av, bv) => NextAfterDispatch(av, bv), "TensorNextAfter");
    }

    private static T NextAfterDispatch<T>(T av, T bv)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        if (typeof(T) == typeof(float))
        {
            // Convert.ToSingle handles the generic-to-numeric path via
            // IConvertible without tripping nullable-reference warnings.
            float af = System.Convert.ToSingle(av);
            float bf = System.Convert.ToSingle(bv);
            return ops.FromDouble(NextAfterFloat(af, bf));
        }
        if (typeof(T) == typeof(double))
        {
            double ad = System.Convert.ToDouble(av);
            double bd = System.Convert.ToDouble(bv);
            return ops.FromDouble(NextAfterDouble(ad, bd));
        }
        // Integer / decimal / complex: no meaningful ulp; return b.
        return bv;
    }

    private static float NextAfterFloat(float a, float b)
    {
        if (a == b) return b;
        if (float.IsNaN(a) || float.IsNaN(b)) return float.NaN;
        // Reinterpret as int32 via unsafe cast (net471-compatible without
        // BitConverter.SingleToInt32Bits).
        unsafe
        {
            int bits = *(int*)&a;
            bits += (a < b) ? (a >= 0 ? 1 : -1) : (a >= 0 ? -1 : 1);
            return *(float*)&bits;
        }
    }

    private static double NextAfterDouble(double a, double b)
    {
        if (a == b) return b;
        if (double.IsNaN(a) || double.IsNaN(b)) return double.NaN;
        long bits = System.BitConverter.DoubleToInt64Bits(a);
        bits += (a < b) ? (a >= 0 ? 1 : -1) : (a >= 0 ? -1 : 1);
        return System.BitConverter.Int64BitsToDouble(bits);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPut<T>(Tensor<T> tensor, Tensor<int> indices, Tensor<T> source)
    {
        // Flat-indexed scatter. Inverse of Take: tensor[indices.flatten()] = source.
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        if (source == null) throw new ArgumentNullException(nameof(source));
        if (indices.Length != source.Length)
            throw new ArgumentException("indices and source must have the same element count");

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var idx = indices.AsSpan();
        var src = source.AsSpan();
        int total = dst.Length;
        for (int i = 0; i < idx.Length; i++)
        {
            int pos = idx[i];
            if (pos < 0 || pos >= total)
                throw new IndexOutOfRangeException(
                    $"indices[{i}]={pos} out of range for flattened length {total}");
            dst[pos] = src[i];
        }
        return result;
    }

    // ==================================================================
    // Special math
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorErfc<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Subtract(ops.One, MathHelper.Erf(x));
        }, "TensorErfc");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorXlogy<T>(Tensor<T> x, Tensor<T> y)
        => ElementwiseBinary(x, y, (xv, yv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            // x * log(y) with the convention 0 * log(0) = 0.
            return ops.Equals(xv, ops.Zero) ? ops.Zero : ops.Multiply(xv, ops.Log(yv));
        }, "TensorXlogy");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorXlog1py<T>(Tensor<T> x, Tensor<T> y)
        => ElementwiseBinary(x, y, (xv, yv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            // x * log(1 + y) with 0 * log(...) = 0.
            return ops.Equals(xv, ops.Zero)
                ? ops.Zero
                : ops.Multiply(xv, ops.Log(ops.Add(ops.One, yv)));
        }, "TensorXlog1py");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLgamma<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            // log|Γ(x)| via existing Lanczos-backed Gamma helper. Negative x is
            // where |·| matters; the Gamma helper's reflection formula returns
            // a signed value, so we take the absolute before logging.
            return ops.Log(ops.Abs(MathHelper.Gamma(x)));
        }, "TensorLgamma");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorErfinv<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double y = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
            if (y >= 1.0) return ops.FromDouble(double.PositiveInfinity);
            if (y <= -1.0) return ops.FromDouble(double.NegativeInfinity);
            // Seed: Winitzki's approximation, good to ~5 digits.
            double ln = System.Math.Log(1.0 - y * y);
            double a = 0.147;
            double t = 2.0 / (System.Math.PI * a) + ln / 2.0;
            double xs = System.Math.Sign(y) * System.Math.Sqrt(System.Math.Sqrt(t * t - ln / a) - t);
            // Refine with 2 Newton steps: f(x) = erf(x) - y, f'(x) = 2/√π · e^(-x²).
            // Using MathHelper.Erf (Abramowitz-Stegun, 6-digit) — .NET doesn't
            // expose Math.Erf until very recent versions.
            for (int it = 0; it < 2; it++)
            {
                double e = MathHelper.Erf(xs);
                double df = 2.0 / System.Math.Sqrt(System.Math.PI) * System.Math.Exp(-xs * xs);
                xs -= (e - y) / df;
            }
            return ops.FromDouble(xs);
        }, "TensorErfinv");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI0<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
            // Series: I₀(x) = Σ_{k=0}^∞ (x/2)^(2k) / (k!)².
            // 25 terms cover |x| up to ~15 with fp64 precision.
            double halfX = xd / 2.0;
            double halfSq = halfX * halfX;
            double term = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 25; k++)
            {
                term *= halfSq / (k * k);
                sum += term;
                if (term < 1e-16) break;
            }
            return ops.FromDouble(sum);
        }, "TensorI0");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI1<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
            // Series: I₁(x) = (x/2) · Σ_{k=0}^∞ (x/2)^(2k) / (k! · (k+1)!).
            double halfX = xd / 2.0;
            double halfSq = halfX * halfX;
            double term = 1.0;   // k=0: 1 / (0! · 1!) = 1
            double sum = 1.0;
            for (int k = 1; k < 25; k++)
            {
                term *= halfSq / (k * (k + 1));
                sum += term;
                if (term < 1e-16) break;
            }
            return ops.FromDouble(halfX * sum);
        }, "TensorI1");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI0e<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            // I₀ with exponential scaling: e^(-|x|) · I₀(x). Safe for large x
            // where I₀ overflows.
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
            double absX = System.Math.Abs(xd);
            double halfX = xd / 2.0;
            double halfSq = halfX * halfX;
            double term = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 25; k++)
            {
                term *= halfSq / (k * k);   // I₀ has (k!)² denominator ⇒ k·k per step
                sum += term;
                if (term < 1e-16 * sum) break;
            }
            return ops.FromDouble(System.Math.Exp(-absX) * sum);
        }, "TensorI0e");

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI1e<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
            double absX = System.Math.Abs(xd);
            double halfX = xd / 2.0;
            double halfSq = halfX * halfX;
            double term = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 25; k++)
            {
                term *= halfSq / (k * (k + 1));
                sum += term;
                if (term < 1e-16 * sum) break;
            }
            return ops.FromDouble(System.Math.Exp(-absX) * halfX * sum);
        }, "TensorI1e");

    /// <inheritdoc/>
    public virtual (Tensor<T> Mantissa, Tensor<int> Exponent) TensorFrexp<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var mant = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var exp = new Tensor<int>(tensor._shape);
        var mdst = mant.AsWritableSpan();
        var edst = exp.AsWritableSpan();
        for (int i = 0; i < src.Length; i++)
        {
            double xd = System.Convert.ToDouble(src[i], System.Globalization.CultureInfo.InvariantCulture);
            if (xd == 0.0)
            {
                mdst[i] = ops.Zero;
                edst[i] = 0;
                continue;
            }
            int e = (int)System.Math.Floor(System.Math.Log(System.Math.Abs(xd), 2.0)) + 1;
            double m = xd * System.Math.Pow(2.0, -e);
            // Normalise into [0.5, 1) — adjust by one step if floating error
            // pushes us to the edge.
            while (System.Math.Abs(m) >= 1.0) { m *= 0.5; e++; }
            while (System.Math.Abs(m) < 0.5) { m *= 2.0; e--; }
            mdst[i] = ops.FromDouble(m);
            edst[i] = e;
        }
        return (mant, exp);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDigamma<T>(Tensor<T> tensor)
        => ElementwiseUnary(tensor, x => {
            // Asymptotic series with recurrence shift. Good enough for fp32;
            // matches PyTorch within ~1e-5 on x in [0.1, 100].
            var ops = MathHelper.GetNumericOperations<T>();
            // Recurrence: ψ(x+1) = ψ(x) + 1/x; shift x up until large enough
            // for the asymptotic series to converge quickly.
            double xd = ToDoubleSafe(x);
            double result = 0;
            while (xd < 6.0)
            {
                result -= 1.0 / xd;
                xd += 1.0;
            }
            // ψ(x) ≈ log(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) …
            double xinv = 1.0 / xd;
            double xinv2 = xinv * xinv;
            result += System.Math.Log(xd) - 0.5 * xinv
                     - xinv2 * (1.0/12.0 - xinv2 * (1.0/120.0 - xinv2 / 252.0));
            return ops.FromDouble(result);
        }, "TensorDigamma");

    // Safe T→double via FromDouble + ToInt32 pair; for float/double/complex
    // INumericOperations exposes ToDouble via FromDouble's inverse path. We
    // use a small indirection to avoid referencing a specific type.
    private static double ToDoubleSafe<T>(T value)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // Common numeric types (float, double, decimal) go cleanly through
        // Convert. For complex this is best-effort.
        return System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
    }

    // ==================================================================
    // Helpers
    // ==================================================================

    private static int[] ComputeRowMajorStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int acc = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = acc;
            acc *= shape[i];
        }
        return strides;
    }

    private Tensor<T> CumulativeAlongAxis<T>(
        Tensor<T> tensor,
        int axis,
        T initial,
        Func<T, T, T> combine,
        string opName,
        BackwardFunction<T> backward)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        int rank = tensor.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank) throw new ArgumentOutOfRangeException(nameof(axis));

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var shape = tensor._shape;
        var result = AutoTensorCache.RentOrAllocate<T>(shape);
        var dst = result.AsWritableSpan();

        var strides = ComputeRowMajorStrides(shape);
        int axisLen = shape[axis];
        int outerSize = 1; for (int k = 0; k < axis; k++) outerSize *= shape[k];
        int innerSize = 1; for (int k = axis + 1; k < rank; k++) innerSize *= shape[k];
        int axisStride = strides[axis];

        for (int outer = 0; outer < outerSize; outer++)
        {
            int basePos = outer * axisLen * innerSize;
            for (int inner = 0; inner < innerSize; inner++)
            {
                T acc = initial;
                for (int i = 0; i < axisLen; i++)
                {
                    int pos = basePos + i * axisStride + inner;
                    acc = (i == 0) ? src[pos] : combine(acc, src[pos]);
                    dst[pos] = acc;
                }
            }
        }

        DifferentiableOps.RecordUnary(
            opName, result, tensor,
            backward,
            savedState: new object[] { axis });
        return result;
    }

    private static T CumulativeInitial<T>(bool max)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        // Not actually used — cumulative fns handle the first element specially.
        return ops.Zero;
    }

    private static Tensor<T> ElementwiseUnary<T>(
        Tensor<T> tensor, Func<T, T> f, string opName)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dst[i] = f(src[i]);
        return result;
    }

    private static Tensor<T> ElementwiseBinary<T>(
        Tensor<T> a, Tensor<T> b, Func<T, T, T> f, string opName)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException(
                $"{opName}: shape mismatch [{string.Join(", ", a._shape)}] vs [{string.Join(", ", b._shape)}]");
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(a._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < av.Length; i++) dst[i] = f(av[i], bv[i]);
        return result;
    }

    private static T TruncateTowardZero<T>(T value)
    {
        // Truncate toward zero: floor for positive, ceil for negative.
        var ops = MathHelper.GetNumericOperations<T>();
        return ops.LessThan(value, ops.Zero) ? ops.Ceiling(value) : ops.Floor(value);
    }
}
