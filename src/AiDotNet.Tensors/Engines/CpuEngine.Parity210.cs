using System;
using System.Linq;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
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

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor;
                var cs = (int[])shifts.Clone();
                var ca = (int[])axes.Clone();
                return scope.RecordUnary(LazyNodeType.Custom, "TensorRoll", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorRoll(ct, cs, ca); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.RollBackward, new object[] { cs, ca });
            }
        }

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

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor;
                var ca = (int[])axes.Clone();
                return scope.RecordUnary(LazyNodeType.Custom, "TensorFlip", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorFlip(ct, ca); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.FlipBackward, new object[] { ca });
            }
        }

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

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor;
                var cr = repeats;
                var cd = dim;
                var graphShape = (int[])tensor._shape.Clone();
                graphShape[dim] = tensor._shape[dim] * repeats;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorRepeatInterleave", tensor, graphShape,
                    (eng, output) => { var r = eng.TensorRepeatInterleave(ct, cr, cd); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.RepeatInterleaveBackward, new object[] { cr, cd });
            }
        }

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
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var ci = indices;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorTake", tensor, (int[])indices._shape.Clone(),
                    (eng, output) => { var r = eng.TensorTake(ct, ci); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.TakeBackward, new object[] { ci, (int[])ct._shape.Clone() });
            }
        }
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

        DifferentiableOps.RecordUnary(
            "TensorTake", result, tensor,
            BackwardFunctions<T>.TakeBackward,
            savedState: new object[] { indices, tensor._shape });
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

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var ci = indices; var cd = dim;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorTakeAlongDim", tensor, (int[])indices._shape.Clone(),
                    (eng, output) => { var r = eng.TensorTakeAlongDim(ct, ci, cd); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.TakeAlongDimBackward, new object[] { ci, cd });
            }
        }
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
        DifferentiableOps.RecordUnary(
            "TensorTakeAlongDim", result, tensor,
            BackwardFunctions<T>.TakeAlongDimBackward,
            savedState: new object[] { indices, dim });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddMM<T>(Tensor<T> input, Tensor<T> a, Tensor<T> b)
    {
        // Default alpha=beta=1; the compiler can't express that cleanly for
        // unconstrained generic T, so we route through the explicit overload.
        var ops = MathHelper.GetNumericOperations<T>();
        return TensorAddMM(input, a, b, ops.One, ops.One);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorAddMM<T>(
        Tensor<T> input, Tensor<T> a, Tensor<T> b, T alpha, T beta)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 2 || b.Rank != 2)
            throw new ArgumentException("AddMM requires 2-D matrices");
        if (a._shape[1] != b._shape[0])
            throw new ArgumentException("Inner dims must match");
        int m = a._shape[0], n = b._shape[1];
        if (input._shape[0] != m || input._shape[1] != n)
            throw new ArgumentException(
                $"input shape must be [{m}, {n}]; got [{input._shape[0]}, {input._shape[1]}]");

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ci = input; var ca = a; var cb = b; var al = alpha; var be = beta;
                var opsHelper = MathHelper.GetNumericOperations<T>();
                return scope.RecordVariadic(LazyNodeType.Custom, "TensorAddMM", new[] { input, a, b }, new[] { m, n },
                    (eng, output) => { var r = eng.TensorAddMM(ci, ca, cb, al, be); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.AddMMBackward, new object[] { opsHelper.ToDouble(al), opsHelper.ToDouble(be) });
            }
        }

        var ops = MathHelper.GetNumericOperations<T>();
        var matmul = TensorMatMul(a, b);
        var result = AutoTensorCache.RentOrAllocate<T>(new[] { m, n });
        var mmSrc = matmul.AsSpan();
        var inSrc = (input.IsContiguous ? input : input.Contiguous()).AsSpan();
        var dst = result.AsWritableSpan();
        for (int i = 0; i < dst.Length; i++)
            dst[i] = ops.Add(ops.Multiply(alpha, mmSrc[i]), ops.Multiply(beta, inSrc[i]));
        // Stash alpha/beta as double so we're robust to T being a reference
        // type — AddMMBackward reconstructs them via ops.FromDouble.
        DifferentiableOps.RecordIfActive<T>(
            "TensorAddMM", result, new[] { input, a, b },
            BackwardFunctions<T>.AddMMBackward,
            savedState: new object[] { ops.ToDouble(alpha), ops.ToDouble(beta) });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDot<T>(Tensor<T> a, Tensor<T> b, int[] axesA, int[] axesB)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (axesA == null) throw new ArgumentNullException(nameof(axesA));
        if (axesB == null) throw new ArgumentNullException(nameof(axesB));
        if (axesA.Length != axesB.Length)
            throw new ArgumentException("axesA and axesB must have the same length");

        // Build einsum labels for the two operands. Use distinct chars for
        // each dim, but match labels between a and b on the contraction axes.
        int aRank = a.Rank;
        int bRank = b.Rank;
        if (axesA.Length > aRank || axesB.Length > bRank)
            throw new ArgumentException("axes exceed operand rank");

        var aLabels = new char[aRank];
        var bLabels = new char[bRank];

        // Normalise negative axes and check for duplicates.
        var contractedA = new int[axesA.Length];
        var contractedB = new int[axesB.Length];
        for (int i = 0; i < axesA.Length; i++)
        {
            int ax = axesA[i] < 0 ? axesA[i] + aRank : axesA[i];
            if (ax < 0 || ax >= aRank) throw new ArgumentOutOfRangeException(nameof(axesA));
            contractedA[i] = ax;
            int bx = axesB[i] < 0 ? axesB[i] + bRank : axesB[i];
            if (bx < 0 || bx >= bRank) throw new ArgumentOutOfRangeException(nameof(axesB));
            contractedB[i] = bx;
        }

        // Assign a contraction label (lowercase starting at 'a') for each
        // contracted dim; use uppercase starting at 'A' for free dims.
        char contractCursor = 'a';
        char freeCursor = 'A';
        var freeA = new System.Collections.Generic.List<char>();
        var freeB = new System.Collections.Generic.List<char>();

        // Map contracted a-axis to its label, same label goes to matching b-axis.
        var aIsContracted = new bool[aRank];
        var bIsContracted = new bool[bRank];
        for (int i = 0; i < contractedA.Length; i++)
        {
            char label = contractCursor++;
            aLabels[contractedA[i]] = label;
            bLabels[contractedB[i]] = label;
            aIsContracted[contractedA[i]] = true;
            bIsContracted[contractedB[i]] = true;
        }
        for (int i = 0; i < aRank; i++)
        {
            if (!aIsContracted[i])
            {
                char label = freeCursor++;
                aLabels[i] = label;
                freeA.Add(label);
            }
        }
        for (int i = 0; i < bRank; i++)
        {
            if (!bIsContracted[i])
            {
                char label = freeCursor++;
                bLabels[i] = label;
                freeB.Add(label);
            }
        }

        string aStr = new string(aLabels);
        string bStr = new string(bLabels);
        string outStr = new string(freeA.Concat(freeB).ToArray());
        string eq = $"{aStr},{bStr}->{outStr}";
        return TensorEinsum(eq, a, b);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCosineSimilarity<T>(
        Tensor<T> x1, Tensor<T> x2, int dim = -1, double eps = 1e-8)
    {
        if (x1 == null) throw new ArgumentNullException(nameof(x1));
        if (x2 == null) throw new ArgumentNullException(nameof(x2));
        if (!x1._shape.SequenceEqual(x2._shape))
            throw new ArgumentException("CosineSimilarity: shapes must match");

        int rank = x1.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));

        var ops = MathHelper.GetNumericOperations<T>();
        if (!x1.IsContiguous) x1 = x1.Contiguous();
        if (!x2.IsContiguous) x2 = x2.Contiguous();
        var a = x1.AsSpan();
        var b = x2.AsSpan();

        // Output shape drops dim.
        var outShape = new int[rank - 1];
        int w = 0;
        for (int i = 0; i < rank; i++) if (i != dim) outShape[w++] = x1._shape[i];
        var result = new Tensor<T>(outShape.Length == 0 ? new[] { 1 } : outShape);
        // 0-rank result for e.g. 1-D input: treat specially — single scalar.
        var dst = result.AsWritableSpan();

        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= x1._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= x1._shape[k];
        int axisLen = x1._shape[dim];
        var epsV = ops.FromDouble(eps);

        int resCursor = 0;
        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                T dot = ops.Zero;
                T na = ops.Zero;
                T nb = ops.Zero;
                for (int i = 0; i < axisLen; i++)
                {
                    int pos = outer * axisLen * innerSize + i * innerSize + inner;
                    T av = a[pos];
                    T bv = b[pos];
                    dot = ops.Add(dot, ops.Multiply(av, bv));
                    na = ops.Add(na, ops.Multiply(av, av));
                    nb = ops.Add(nb, ops.Multiply(bv, bv));
                }
                var denom = ops.Multiply(
                    MaxScalar(ops, ops.Sqrt(na), epsV),
                    MaxScalar(ops, ops.Sqrt(nb), epsV));
                dst[resCursor++] = ops.Divide(dot, denom);
            }

        // For 1-D inputs we produced a length-1 result; flatten to scalar.
        if (rank == 1) return result.Reshape(new int[0]);
        return result;
    }

    private static T MaxScalar<T>(Interfaces.INumericOperations<T> ops, T a, T b)
        => ops.GreaterThan(a, b) ? a : b;

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPDist<T>(Tensor<T> input, double p = 2.0)
    {
        // Pairwise p-norm distance over the N rows of a 2-D [N, D] input.
        // Output shape: 1-D of length N·(N-1)/2, ordered (0,1),(0,2),...,
        // matching torch.pdist.
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 2) throw new ArgumentException("PDist requires rank-2 input");
        int n = input._shape[0];
        int d = input._shape[1];
        if (n == 0) return new Tensor<T>(new[] { 0 });

        var ops = MathHelper.GetNumericOperations<T>();
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        int pairs = n * (n - 1) / 2;
        var result = new Tensor<T>(new[] { pairs });
        var dst = result.AsWritableSpan();
        int cursor = 0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
            {
                dst[cursor++] = PNorm(ops, src, i * d, j * d, d, p);
            }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCDist<T>(Tensor<T> x1, Tensor<T> x2, double p = 2.0)
    {
        // Cross pairwise p-norm: output[i, j] = ‖x1[i] − x2[j]‖_p.
        if (x1 == null) throw new ArgumentNullException(nameof(x1));
        if (x2 == null) throw new ArgumentNullException(nameof(x2));
        if (x1.Rank != 2 || x2.Rank != 2)
            throw new ArgumentException("CDist requires rank-2 inputs");
        if (x1._shape[1] != x2._shape[1])
            throw new ArgumentException("CDist: feature dim must match");

        var ops = MathHelper.GetNumericOperations<T>();
        if (!x1.IsContiguous) x1 = x1.Contiguous();
        if (!x2.IsContiguous) x2 = x2.Contiguous();
        var a = x1.AsSpan();
        var b = x2.AsSpan();
        int m = x1._shape[0], n = x2._shape[0], d = x1._shape[1];

        var result = new Tensor<T>(new[] { m, n });
        var dst = result.AsWritableSpan();
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                dst[i * n + j] = PNormCross(ops, a, i * d, b, j * d, d, p);
        return result;
    }

    private static T PNorm<T>(Interfaces.INumericOperations<T> ops, System.ReadOnlySpan<T> s,
        int offA, int offB, int d, double p)
    {
        // ‖a − b‖_p with inputs pulled from the same span (pdist case).
        double acc = 0;
        for (int k = 0; k < d; k++)
        {
            double diff = System.Convert.ToDouble(s[offA + k]) - System.Convert.ToDouble(s[offB + k]);
            acc += System.Math.Pow(System.Math.Abs(diff), p);
        }
        return ops.FromDouble(System.Math.Pow(acc, 1.0 / p));
    }

    private static T PNormCross<T>(Interfaces.INumericOperations<T> ops,
        System.ReadOnlySpan<T> a, int offA, System.ReadOnlySpan<T> b, int offB, int d, double p)
    {
        double acc = 0;
        for (int k = 0; k < d; k++)
        {
            double diff = System.Convert.ToDouble(a[offA + k]) - System.Convert.ToDouble(b[offB + k]);
            acc += System.Math.Pow(System.Math.Abs(diff), p);
        }
        return ops.FromDouble(System.Math.Pow(acc, 1.0 / p));
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorKron<T>(Tensor<T> a, Tensor<T> b)
    {
        // Kronecker product torch.kron — generalises to arbitrary rank.
        //
        //   output[i0*b0 + j0, i1*b1 + j1, ..., iN*bN + jN]
        //     = a[i0, i1, ..., iN] * b[j0, j1, ..., jN]
        //
        // Implementation: right-align shapes by padding the shorter-rank input
        // with leading ones, then iterate the cartesian product of (a index,
        // b index) pairs and write each into the flat output position derived
        // from i*b_dim + j along each axis.
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));

        int rankA = a.Rank;
        int rankB = b.Rank;
        int rank = System.Math.Max(rankA, rankB);

        // Pad the smaller-rank input with leading 1s.
        var aShape = new int[rank];
        var bShape = new int[rank];
        for (int k = 0; k < rank; k++)
        {
            aShape[k] = (k >= rank - rankA) ? a._shape[k - (rank - rankA)] : 1;
            bShape[k] = (k >= rank - rankB) ? b._shape[k - (rank - rankB)] : 1;
        }

        var outShape = new int[rank];
        int outTotal = 1;
        for (int k = 0; k < rank; k++)
        {
            outShape[k] = aShape[k] * bShape[k];
            outTotal *= outShape[k];
        }

        var ops = MathHelper.GetNumericOperations<T>();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();

        var result = AutoTensorCache.RentOrAllocate<T>(outShape);
        var dst = result.AsWritableSpan();
        var aSrc = a.AsSpan();
        var bSrc = b.AsSpan();

        // Row-major strides for a (reshaped), b (reshaped), and output.
        var aStrides = new int[rank];
        var bStrides = new int[rank];
        var outStrides = new int[rank];
        aStrides[rank - 1] = 1; bStrides[rank - 1] = 1; outStrides[rank - 1] = 1;
        for (int k = rank - 2; k >= 0; k--)
        {
            aStrides[k] = aStrides[k + 1] * aShape[k + 1];
            bStrides[k] = bStrides[k + 1] * bShape[k + 1];
            outStrides[k] = outStrides[k + 1] * outShape[k + 1];
        }

        // Walk every output element: for position idx, decompose into
        // (i, j) per axis via idx / outStrides[k], then map back to
        // aIdx and bIdx.
        var idx = new int[rank];
        for (int o = 0; o < outTotal; o++)
        {
            // Decompose flat o into per-axis indices.
            int rem = o;
            for (int k = 0; k < rank; k++)
            {
                idx[k] = rem / outStrides[k];
                rem -= idx[k] * outStrides[k];
            }
            int aFlat = 0, bFlat = 0;
            for (int k = 0; k < rank; k++)
            {
                int i = idx[k] / bShape[k];   // a's index along axis k
                int j = idx[k] % bShape[k];   // b's index along axis k
                aFlat += i * aStrides[k];
                bFlat += j * bStrides[k];
            }
            dst[o] = ops.Multiply(aSrc[aFlat], bSrc[bFlat]);
        }

        DifferentiableOps.RecordBinary(
            "TensorKron", result, a, b,
            BackwardFunctions<T>.KronBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorInner<T>(Tensor<T> a, Tensor<T> b)
    {
        // Inner product over the last axis (torch.inner). Output shape is
        // a.shape[:-1] + b.shape[:-1]; contracts matching last-axis sizes.
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a._shape[a.Rank - 1] != b._shape[b.Rank - 1])
            throw new ArgumentException(
                $"Inner: last-axis sizes must match ({a._shape[a.Rank - 1]} vs {b._shape[b.Rank - 1]})");

        // Build einsum equation where a's last-dim label and b's last-dim
        // label are equal (contracted). Use two disjoint label alphabets for
        // a's and b's free dims.
        char cursor = 'a';
        var aLabels = new char[a.Rank];
        var bLabels = new char[b.Rank];
        var outLabels = new System.Text.StringBuilder();
        for (int i = 0; i < a.Rank - 1; i++)
        {
            aLabels[i] = cursor;
            outLabels.Append(cursor);
            cursor++;
        }
        char contract = cursor++;
        aLabels[a.Rank - 1] = contract;
        for (int i = 0; i < b.Rank - 1; i++)
        {
            bLabels[i] = cursor;
            outLabels.Append(cursor);
            cursor++;
        }
        bLabels[b.Rank - 1] = contract;

        string eq = $"{new string(aLabels)},{new string(bLabels)}->{outLabels}";
        return TensorEinsum(eq, a, b);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCartesianProd<T>(Tensor<T>[] tensors)
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length == 0)
            throw new ArgumentException("CartesianProd requires at least one tensor");
        foreach (var t in tensors)
        {
            if (t == null) throw new ArgumentNullException(nameof(tensors));
            if (t.Rank != 1) throw new ArgumentException("CartesianProd requires 1-D inputs");
        }

        int d = tensors.Length;
        long totalLong = 1;
        for (int k = 0; k < d; k++) totalLong *= tensors[k]._shape[0];
        if (totalLong > int.MaxValue)
            throw new OverflowException("CartesianProd output size overflows Int32");
        int total = (int)totalLong;

        var result = AutoTensorCache.RentOrAllocate<T>(new[] { total, d });
        var dst = result.AsWritableSpan();

        var sizes = new int[d];
        for (int k = 0; k < d; k++) sizes[k] = tensors[k]._shape[0];

        // Walk the multi-index row-major.
        var idx = new int[d];
        for (int row = 0; row < total; row++)
        {
            for (int k = 0; k < d; k++)
            {
                dst[row * d + k] = tensors[k][idx[k]];
            }
            // increment
            for (int k = d - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < sizes[k]) break;
                idx[k] = 0;
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorMeshgrid<T>(Tensor<T>[] tensors, string indexing = "ij")
    {
        if (tensors == null) throw new ArgumentNullException(nameof(tensors));
        if (tensors.Length == 0) return System.Array.Empty<Tensor<T>>();
        foreach (var t in tensors)
        {
            if (t == null) throw new ArgumentNullException(nameof(tensors));
            if (t.Rank != 1) throw new ArgumentException("Meshgrid requires 1-D inputs");
        }
        if (indexing != "ij" && indexing != "xy")
            throw new ArgumentException("indexing must be 'ij' or 'xy'");

        int d = tensors.Length;
        // Output shape by indexing rule:
        //   ij: (len(t0), len(t1), ..., len(t{d-1}))
        //   xy: applies only when d >= 2 — first two axes are swapped vs ij.
        var outShape = new int[d];
        for (int i = 0; i < d; i++) outShape[i] = tensors[i]._shape[0];
        if (indexing == "xy" && d >= 2)
        {
            (outShape[0], outShape[1]) = (outShape[1], outShape[0]);
        }

        var results = new Tensor<T>[d];
        var strides = ComputeRowMajorStrides(outShape);
        int total = 1; foreach (var s in outShape) total *= s;

        for (int k = 0; k < d; k++)
        {
            // Figure out which axis in the output corresponds to tensor k.
            int axisK = k;
            if (indexing == "xy" && d >= 2)
            {
                if (k == 0) axisK = 1;
                else if (k == 1) axisK = 0;
            }

            results[k] = AutoTensorCache.RentOrAllocate<T>(outShape);
            var dst = results[k].AsWritableSpan();
            var src = (tensors[k].IsContiguous ? tensors[k] : tensors[k].Contiguous()).AsSpan();

            // For each linear index in the output, pick the coordinate along
            // axisK and use src[that coord].
            for (int linear = 0; linear < total; linear++)
            {
                int coord = (linear / strides[axisK]) % outShape[axisK];
                dst[linear] = src[coord];
            }
        }
        return results;
    }

    /// <inheritdoc/>
    public virtual T TensorTrace<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank != 2) throw new ArgumentException("Trace requires a 2-D tensor");
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        int rows = tensor._shape[0];
        int cols = tensor._shape[1];
        int n = System.Math.Min(rows, cols);
        T acc = ops.Zero;
        for (int i = 0; i < n; i++) acc = ops.Add(acc, src[i * cols + i]);
        return acc;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorDiagEmbed<T>(Tensor<T> tensor, int offset = 0)
    {
        // Take a rank-R tensor whose last dim has length L, and embed it as
        // the diagonal of a square R+1-rank tensor whose last two dims are
        // (L + |offset|). Mirrors torch.diag_embed for offset=0.
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank < 1) throw new ArgumentException("DiagEmbed requires rank >= 1");

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var co = offset;
                int r = tensor.Rank;
                int dl = tensor._shape[r - 1];
                int ms = dl + System.Math.Abs(co);
                var oShape = new int[r + 1];
                for (int i = 0; i < r - 1; i++) oShape[i] = tensor._shape[i];
                oShape[r - 1] = ms;
                oShape[r] = ms;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorDiagEmbed", tensor, oShape,
                    (eng, output) => { var res = eng.TensorDiagEmbed(ct, co); res.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.DiagEmbedBackward, new object[] { co });
            }
        }

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        int rank = tensor.Rank;
        int diagLen = tensor._shape[rank - 1];
        int matSize = diagLen + System.Math.Abs(offset);

        var outShape = new int[rank + 1];
        for (int i = 0; i < rank - 1; i++) outShape[i] = tensor._shape[i];
        outShape[rank - 1] = matSize;
        outShape[rank] = matSize;

        var result = AutoTensorCache.RentOrAllocate<T>(outShape);
        var dst = result.AsWritableSpan();
        var ops = MathHelper.GetNumericOperations<T>();
        var zero = ops.Zero;
        for (int i = 0; i < dst.Length; i++) dst[i] = zero;

        var src = tensor.AsSpan();
        int batchSize = 1; for (int k = 0; k < rank - 1; k++) batchSize *= tensor._shape[k];
        for (int b = 0; b < batchSize; b++)
            for (int i = 0; i < diagLen; i++)
            {
                int row = offset >= 0 ? i : i - offset;
                int col = offset >= 0 ? i + offset : i;
                int dstPos = b * matSize * matSize + row * matSize + col;
                int srcPos = b * diagLen + i;
                dst[dstPos] = src[srcPos];
            }
        DifferentiableOps.RecordUnary(
            "TensorDiagEmbed", result, tensor, BackwardFunctions<T>.DiagEmbedBackward,
            savedState: new object[] { offset });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCross<T>(Tensor<T> a, Tensor<T> b, int dim = -1)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException("Cross: shapes must match");
        int rank = a.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        if (a._shape[dim] != 3)
            throw new ArgumentException($"Cross requires size 3 along dim {dim}; got {a._shape[dim]}");

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ca = a; var cb = b; var cd = dim;
                return scope.RecordBinary(LazyNodeType.Custom, "TensorCross", a, b, (int[])a._shape.Clone(),
                    (eng, output) => { var r = eng.TensorCross(ca, cb, cd); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.CrossBackward, new object[] { cd });
            }
        }

        var ops = MathHelper.GetNumericOperations<T>();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();

        var result = AutoTensorCache.RentOrAllocate<T>(a._shape);
        var av = a.AsSpan();
        var bv = b.AsSpan();
        var dst = result.AsWritableSpan();

        int outerSize = 1; for (int k = 0; k < dim; k++) outerSize *= a._shape[k];
        int innerSize = 1; for (int k = dim + 1; k < rank; k++) innerSize *= a._shape[k];

        for (int outer = 0; outer < outerSize; outer++)
            for (int inner = 0; inner < innerSize; inner++)
            {
                int b0 = outer * 3 * innerSize + 0 * innerSize + inner;
                int b1 = outer * 3 * innerSize + 1 * innerSize + inner;
                int b2 = outer * 3 * innerSize + 2 * innerSize + inner;
                T ax = av[b0], ay = av[b1], az = av[b2];
                T bx = bv[b0], by = bv[b1], bz = bv[b2];
                // c = a × b = (ay·bz - az·by, az·bx - ax·bz, ax·by - ay·bx)
                dst[b0] = ops.Subtract(ops.Multiply(ay, bz), ops.Multiply(az, by));
                dst[b1] = ops.Subtract(ops.Multiply(az, bx), ops.Multiply(ax, bz));
                dst[b2] = ops.Subtract(ops.Multiply(ax, by), ops.Multiply(ay, bx));
            }
        DifferentiableOps.RecordBinary(
            "TensorCross", result, a, b, BackwardFunctions<T>.CrossBackward,
            savedState: new object[] { dim });
        return result;
    }

    /// <inheritdoc/>
    public virtual T TensorVecDot<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (a.Rank != 1 || b.Rank != 1) throw new ArgumentException("VecDot requires 1-D tensors");
        if (a.Length != b.Length) throw new ArgumentException("vectors must have the same length");
        var ops = MathHelper.GetNumericOperations<T>();
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        T acc = ops.Zero;
        for (int i = 0; i < av.Length; i++)
            acc = ops.Add(acc, ops.Multiply(av[i], bv[i]));
        return acc;
    }

    // ==================================================================
    // Cumulative ops
    // ==================================================================

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumProd<T>(Tensor<T> tensor, int axis)
    {
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var ca = axis;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorCumProd", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorCumProd(ct, ca); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.CumProdBackward, new object[] { ca });
            }
        }
        return CumulativeAlongAxis(tensor, axis, MathHelper.GetNumericOperations<T>().One,
            (a, b) => MathHelper.GetNumericOperations<T>().Multiply(a, b),
            "TensorCumProd", BackwardFunctions<T>.CumProdBackward);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumMax<T>(Tensor<T> tensor, int axis)
    {
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var ca = axis;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorCumMax", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorCumMax(ct, ca); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.CumMaxBackward, new object[] { ca });
            }
        }
        // Fp32 contiguous inner-most axis fast path: SIMD-assisted running max.
        if (typeof(T) == typeof(float))
        {
            int r = tensor.Rank;
            int ax = axis < 0 ? axis + r : axis;
            if (ax == r - 1 && tensor.IsContiguous)
            {
                var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
                var src = (float[])(object)tensor.GetDataArray();
                var dst = (float[])(object)result.GetDataArray();
                int axisLen = tensor._shape[ax];
                int outer = tensor.Length / axisLen;
                for (int o = 0; o < outer; o++)
                {
                    Simd.ScanKernels.RunningMaxFloat(
                        new ReadOnlySpan<float>(src, o * axisLen, axisLen),
                        new Span<float>(dst, o * axisLen, axisLen));
                }
                DifferentiableOps.RecordUnary("TensorCumMax", result, tensor,
                    BackwardFunctions<T>.CumMaxBackward, new object[] { axis });
                return result;
            }
        }
        return CumulativeAlongAxis(tensor, axis,
            CumulativeInitial<T>(max: true),
            (a, b) => {
                var ops = MathHelper.GetNumericOperations<T>();
                return ops.GreaterThan(a, b) ? a : b;
            },
            "TensorCumMax", BackwardFunctions<T>.CumMaxBackward);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCumMin<T>(Tensor<T> tensor, int axis)
    {
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var ca = axis;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorCumMin", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorCumMin(ct, ca); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.CumMinBackward, new object[] { ca });
            }
        }
        if (typeof(T) == typeof(float))
        {
            int r = tensor.Rank;
            int ax = axis < 0 ? axis + r : axis;
            if (ax == r - 1 && tensor.IsContiguous)
            {
                var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
                var src = (float[])(object)tensor.GetDataArray();
                var dst = (float[])(object)result.GetDataArray();
                int axisLen = tensor._shape[ax];
                int outer = tensor.Length / axisLen;
                for (int o = 0; o < outer; o++)
                {
                    Simd.ScanKernels.RunningMinFloat(
                        new ReadOnlySpan<float>(src, o * axisLen, axisLen),
                        new Span<float>(dst, o * axisLen, axisLen));
                }
                DifferentiableOps.RecordUnary("TensorCumMin", result, tensor,
                    BackwardFunctions<T>.CumMinBackward, new object[] { axis });
                return result;
            }
        }
        return CumulativeAlongAxis(tensor, axis,
            CumulativeInitial<T>(max: false),
            (a, b) => {
                var ops = MathHelper.GetNumericOperations<T>();
                return ops.LessThan(a, b) ? a : b;
            },
            "TensorCumMin", BackwardFunctions<T>.CumMinBackward);
    }

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
    public virtual Tensor<T> TensorNanToNum<T>(
        Tensor<T> tensor, double? nan = null, double? posinf = null, double? neginf = null)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var cn = nan; var cpi = posinf; var cni = neginf;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorNanToNum", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorNanToNum(ct, cn, cpi, cni); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.NanToNumBackward);
            }
        }
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();

        double defaultNan = nan ?? 0.0;
        // PyTorch defaults posinf / neginf to the max/min finite of the dtype
        // when not supplied. We use float.MaxValue / -float.MaxValue which is
        // finite both in fp32 and fp64 (and saturates cleanly for narrower
        // types via FromDouble).
        double defaultPosInf = posinf ?? (double)float.MaxValue;
        double defaultNegInf = neginf ?? -(double)float.MaxValue;

        for (int i = 0; i < src.Length; i++)
        {
            double d = System.Convert.ToDouble(src[i], System.Globalization.CultureInfo.InvariantCulture);
            if (double.IsNaN(d)) dst[i] = ops.FromDouble(defaultNan);
            else if (double.IsPositiveInfinity(d)) dst[i] = ops.FromDouble(defaultPosInf);
            else if (double.IsNegativeInfinity(d)) dst[i] = ops.FromDouble(defaultNegInf);
            else dst[i] = src[i];
        }
        DifferentiableOps.RecordUnary(
            "TensorNanToNum", result, tensor, BackwardFunctions<T>.NanToNumBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorIsFinite<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = new Bit[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            bool finite = !ops.IsNaN(src[i]);
            // Also check for ±∞ via round-trip to double (when T is fp).
            if (finite)
            {
                double d = System.Convert.ToDouble(src[i], System.Globalization.CultureInfo.InvariantCulture);
                finite = !double.IsInfinity(d);
            }
            result[i] = finite ? Bit.True : Bit.False;
        }
        return new Tensor<Bit>(result, tensor._shape);
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorIsNan<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = new Bit[src.Length];
        for (int i = 0; i < src.Length; i++)
            result[i] = ops.IsNaN(src[i]) ? Bit.True : Bit.False;
        return new Tensor<Bit>(result, tensor._shape);
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorIsInf<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = new Bit[src.Length];
        for (int i = 0; i < src.Length; i++)
        {
            double d = System.Convert.ToDouble(src[i], System.Globalization.CultureInfo.InvariantCulture);
            result[i] = double.IsInfinity(d) ? Bit.True : Bit.False;
        }
        return new Tensor<Bit>(result, tensor._shape);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTriu<T>(Tensor<T> tensor, int diagonal = 0)
    {
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var cd = diagonal;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorTriu", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorTriu(ct, cd); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.TriuBackward, new object[] { cd });
            }
        }
        var result = TriangularFill(tensor, diagonal, keepUpper: true);
        DifferentiableOps.RecordUnary(
            "TensorTriu", result, tensor, BackwardFunctions<T>.TriuBackward,
            savedState: new object[] { diagonal });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorTril<T>(Tensor<T> tensor, int diagonal = 0)
    {
        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var cd = diagonal;
                return scope.RecordUnary(LazyNodeType.Custom, "TensorTril", tensor, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorTril(ct, cd); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.TrilBackward, new object[] { cd });
            }
        }
        var result = TriangularFill(tensor, diagonal, keepUpper: false);
        DifferentiableOps.RecordUnary(
            "TensorTril", result, tensor, BackwardFunctions<T>.TrilBackward,
            savedState: new object[] { diagonal });
        return result;
    }

    private Tensor<T> TriangularFill<T>(Tensor<T> tensor, int diagonal, bool keepUpper)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (tensor.Rank < 2) throw new ArgumentException("Triu/Tril requires rank >= 2");

        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        int rank = tensor.Rank;
        int rows = tensor._shape[rank - 2];
        int cols = tensor._shape[rank - 1];
        int batchSize = 1; for (int k = 0; k < rank - 2; k++) batchSize *= tensor._shape[k];

        var result = (Tensor<T>)tensor.Clone();
        var dst = result.AsWritableSpan();
        var zero = ops.Zero;

        for (int b = 0; b < batchSize; b++)
        {
            int baseIdx = b * rows * cols;
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                {
                    // keep cell iff:
                    //   Triu: j - i >= diagonal
                    //   Tril: j - i <= diagonal
                    int offset = j - i;
                    bool keep = keepUpper ? offset >= diagonal : offset <= diagonal;
                    if (!keep) dst[baseIdx + i * cols + j] = zero;
                }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<int> TensorNonzero<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        int rank = tensor.Rank;
        var strides = ComputeRowMajorStrides(tensor._shape);

        // First pass: count.
        int n = 0;
        for (int i = 0; i < src.Length; i++) if (!ops.Equals(src[i], ops.Zero)) n++;

        // Second pass: coordinates.
        var result = new Tensor<int>(new[] { n, rank });
        var dst = result.AsWritableSpan();
        int row = 0;
        for (int i = 0; i < src.Length; i++)
        {
            if (!ops.Equals(src[i], ops.Zero))
            {
                int rem = i;
                for (int k = 0; k < rank; k++)
                {
                    dst[row * rank + k] = rem / strides[k];
                    rem %= strides[k];
                }
                row++;
            }
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual int TensorCountNonzero<T>(Tensor<T> tensor)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        var ops = MathHelper.GetNumericOperations<T>();
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        int n = 0;
        for (int i = 0; i < src.Length; i++) if (!ops.Equals(src[i], ops.Zero)) n++;
        return n;
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorLogicalAnd(Tensor<Bit> a, Tensor<Bit> b)
        => BitBinary(a, b, (av, bv) => (bool)av && (bool)bv);

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorLogicalOr(Tensor<Bit> a, Tensor<Bit> b)
        => BitBinary(a, b, (av, bv) => (bool)av || (bool)bv);

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorLogicalXor(Tensor<Bit> a, Tensor<Bit> b)
        => BitBinary(a, b, (av, bv) => (bool)av ^ (bool)bv);

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorLogicalNot(Tensor<Bit> a)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (!a.IsContiguous) a = a.Contiguous();
        var src = a.AsSpan();
        var dst = new Bit[src.Length];
        for (int i = 0; i < src.Length; i++) dst[i] = (bool)src[i] ? Bit.False : Bit.True;
        return new Tensor<Bit>(dst, a._shape);
    }

    private static Tensor<Bit> BitBinary(
        Tensor<Bit> a, Tensor<Bit> b, Func<Bit, Bit, bool> f)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException("logical op: shape mismatch");
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        var dst = new Bit[av.Length];
        for (int i = 0; i < av.Length; i++) dst[i] = f(av[i], bv[i]) ? Bit.True : Bit.False;
        return new Tensor<Bit>(dst, a._shape);
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

        // Broadcast min / max against the tensor shape using NumPy / PyTorch
        // rules: right-align, dims of 1 or missing broadcast freely.
        // ValidateAndComputeBroadcastStride returns null when the shape is
        // exactly compatible (no broadcasting needed) or a stride vector in
        // the tensor's row-major order when broadcasting is active.
        int[]? minStrides = ValidateAndComputeClampBroadcastStrides(tensor._shape, min?._shape);
        int[]? maxStrides = ValidateAndComputeClampBroadcastStrides(tensor._shape, max?._shape);

        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(tensor._shape);
        var dst = result.AsWritableSpan();
        var minSpan = min is null ? default : (min.IsContiguous ? min : min.Contiguous()).AsSpan();
        var maxSpan = max is null ? default : (max.IsContiguous ? max : max.Contiguous()).AsSpan();

        int rank = tensor.Rank;
        var idx = new int[rank];
        for (int i = 0; i < src.Length; i++)
        {
            var v = src[i];
            if (min is not null)
            {
                int mIdx = minStrides == null ? i : BroadcastLookup(idx, minStrides);
                if (ops.LessThan(v, minSpan[mIdx])) v = minSpan[mIdx];
            }
            if (max is not null)
            {
                int xIdx = maxStrides == null ? i : BroadcastLookup(idx, maxStrides);
                if (ops.GreaterThan(v, maxSpan[xIdx])) v = maxSpan[xIdx];
            }
            dst[i] = v;
            // Advance row-major index.
            for (int k = rank - 1; k >= 0; k--)
            {
                idx[k]++;
                if (idx[k] < tensor._shape[k]) break;
                idx[k] = 0;
            }
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

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var ca = axis; var ci = indices; var cs = source;
                return scope.RecordBinary(LazyNodeType.Custom, "TensorIndexAdd", tensor, source, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorIndexAdd(ct, ca, ci, cs); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.IndexAddBackward, new object[] { ca, ci });
            }
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
        DifferentiableOps.RecordUnary(
            "TensorIndexAdd", result, tensor,
            BackwardFunctions<T>.IndexAddBackward,
            savedState: new object[] { axis, indices });
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

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var ca = axis; var ci = indices; var cs = source;
                return scope.RecordBinary(LazyNodeType.Custom, "TensorIndexCopy", tensor, source, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorIndexCopy(ct, ca, ci, cs); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.IndexCopyBackward, new object[] { ca, ci });
            }
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
        DifferentiableOps.RecordUnary(
            "TensorIndexCopy", result, tensor,
            BackwardFunctions<T>.IndexCopyBackward,
            savedState: new object[] { axis, indices });
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
        DifferentiableOps.RecordUnary(
            "TensorIndexFill", result, tensor,
            BackwardFunctions<T>.IndexFillBackward,
            savedState: new object[] { axis, indices });
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

        if (GraphMode.IsActive)
        {
            var scope = GraphMode.Current;
            if (scope != null)
            {
                var ct = tensor; var cm = mask; var cs = source;
                return scope.RecordBinary(LazyNodeType.Custom, "TensorMaskedScatter", tensor, source, (int[])tensor._shape.Clone(),
                    (eng, output) => { var r = eng.TensorMaskedScatter(ct, cm, cs); r.AsSpan().CopyTo(output.AsWritableSpan()); },
                    BackwardFunctions<T>.MaskedScatterBackward, new object[] { cm });
            }
        }

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
        DifferentiableOps.RecordUnary(
            "TensorMaskedScatter", result, tensor,
            BackwardFunctions<T>.MaskedScatterBackward,
            savedState: new object[] { mask });
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

        // Float32 inner-most-axis fast path: route through SortKernels' SIMD
        // bitonic (short axes) or Array.Sort pair-sort (longer axes) without
        // boxing into a (T,int) struct array.
        if (typeof(T) == typeof(float) && innerSize == 1 && !descending)
        {
            float[] srcFloatArr = (float[])(object)input.GetDataArray();
            float[] valFloatArr = (float[])(object)valuesOut.GetDataArray();
            int[] idxArr = indicesOut.GetDataArray();
            for (int outer = 0; outer < outerSize; outer++)
            {
                int baseIdx = outer * axisSize;
                var vSlice = valFloatArr.AsSpan(baseIdx, axisSize);
                var iSlice = idxArr.AsSpan(baseIdx, axisSize);
                srcFloatArr.AsSpan(baseIdx, axisSize).CopyTo(vSlice);
                for (int a = 0; a < axisSize; a++) iSlice[a] = a;
                Simd.SortKernels.SortFloatWithIndicesAscending(vSlice, iSlice);
            }
            return (valuesOut, indicesOut);
        }

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
    public virtual Tensor<int> TensorArgsort<T>(Tensor<T> input, int axis = -1, bool descending = false)
    {
        // Argsort is just Sort discarding the values tensor. We share the
        // same kernel path so any future SIMD Sort lands here too.
        var (_, indices) = TensorSort(input, axis, descending);
        return indices;
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Values, Tensor<int>? Inverse, Tensor<int>? Counts) TensorUniqueWithInfo<T>(
        Tensor<T> input, bool sorted = true, bool returnInverse = false, bool returnCounts = false)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        var ops = MathHelper.GetNumericOperations<T>();

        // Collect distinct values preserving first-seen order, track bucket index per element.
        // We can't use Dictionary<T, int> because T lacks a `notnull` constraint;
        // linear scan over the growing list is O(n·k) but fine for typical unique counts.
        var list = new System.Collections.Generic.List<T>();
        var bucketIdx = new int[src.Length];
        var counts = new System.Collections.Generic.List<int>();
        for (int i = 0; i < src.Length; i++)
        {
            int b = -1;
            for (int j = 0; j < list.Count; j++)
            {
                if (ops.Equals(list[j], src[i])) { b = j; break; }
            }
            if (b < 0)
            {
                b = list.Count;
                list.Add(src[i]);
                counts.Add(0);
            }
            bucketIdx[i] = b;
            counts[b] = counts[b] + 1;
        }

        int n = list.Count;
        Tensor<T> values;
        int[] remap = null!;
        if (sorted)
        {
            // Sort unique values and build a remap from old bucket index -> new sorted position.
            var pairs = new (T value, int oldIdx)[n];
            for (int i = 0; i < n; i++) pairs[i] = (list[i], i);
            Array.Sort(pairs, (a, b) => ops.Compare(a.value, b.value));
            var sortedArr = new T[n];
            remap = new int[n];
            for (int i = 0; i < n; i++)
            {
                sortedArr[i] = pairs[i].value;
                remap[pairs[i].oldIdx] = i;
            }
            values = new Tensor<T>(sortedArr, new[] { n });
        }
        else
        {
            values = new Tensor<T>(list.ToArray(), new[] { n });
        }

        Tensor<int>? inverse = null;
        if (returnInverse)
        {
            var invArr = new int[src.Length];
            for (int i = 0; i < src.Length; i++)
                invArr[i] = sorted ? remap[bucketIdx[i]] : bucketIdx[i];
            inverse = new Tensor<int>(invArr, (int[])input._shape.Clone());
        }

        Tensor<int>? countsOut = null;
        if (returnCounts)
        {
            var countsArr = new int[n];
            if (sorted)
                for (int i = 0; i < n; i++) countsArr[remap[i]] = counts[i];
            else
                for (int i = 0; i < n; i++) countsArr[i] = counts[i];
            countsOut = new Tensor<int>(countsArr, new[] { n });
        }

        return (values, inverse, countsOut);
    }

    /// <inheritdoc/>
    public virtual (Tensor<T> Values, Tensor<int>? Inverse, Tensor<int>? Counts) TensorUniqueConsecutiveWithInfo<T>(
        Tensor<T> input, bool returnInverse = false, bool returnCounts = false)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();
        var ops = MathHelper.GetNumericOperations<T>();

        var values = new System.Collections.Generic.List<T>();
        var counts = new System.Collections.Generic.List<int>();
        var inv = returnInverse ? new int[src.Length] : null;

        if (src.Length > 0)
        {
            T last = src[0];
            values.Add(last);
            counts.Add(1);
            if (inv != null) inv[0] = 0;
            for (int i = 1; i < src.Length; i++)
            {
                if (!ops.Equals(src[i], last))
                {
                    last = src[i];
                    values.Add(last);
                    counts.Add(1);
                }
                else
                {
                    counts[counts.Count - 1] = counts[counts.Count - 1] + 1;
                }
                if (inv != null) inv[i] = values.Count - 1;
            }
        }

        var valuesOut = new Tensor<T>(values.ToArray(), new[] { values.Count });
        Tensor<int>? inverseOut = inv != null ? new Tensor<int>(inv, (int[])input._shape.Clone()) : null;
        Tensor<int>? countsOut = returnCounts ? new Tensor<int>(counts.ToArray(), new[] { counts.Count }) : null;
        return (valuesOut, inverseOut, countsOut);
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorHistc<T>(Tensor<T> input, int bins, T min, T max)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (bins < 1) throw new ArgumentOutOfRangeException(nameof(bins));

        var ops = MathHelper.GetNumericOperations<T>();
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();

        // If min == max, auto-detect from input (matches torch.histc).
        T loBound = min, hiBound = max;
        if (ops.Equals(min, max))
        {
            if (src.Length == 0)
            {
                loBound = ops.Zero;
                hiBound = ops.Zero;
            }
            else
            {
                loBound = src[0];
                hiBound = src[0];
                for (int i = 1; i < src.Length; i++)
                {
                    if (ops.LessThan(src[i], loBound)) loBound = src[i];
                    if (ops.GreaterThan(src[i], hiBound)) hiBound = src[i];
                }
            }
            // Degenerate-range guard: torch returns a histogram where all values
            // go into the single bin when min == max and input is non-empty.
            if (ops.Equals(loBound, hiBound))
            {
                var r = new Tensor<T>(new[] { bins });
                var d = r.AsWritableSpan();
                d[0] = ops.FromDouble(src.Length);
                for (int b = 1; b < bins; b++) d[b] = ops.Zero;
                return r;
            }
        }
        else if (ops.GreaterThanOrEquals(loBound, hiBound))
        {
            throw new ArgumentException("Histc requires min < max when both are explicit");
        }

        var result = new Tensor<T>(new[] { bins });
        var dst = result.AsWritableSpan();
        var width = ops.Divide(ops.Subtract(hiBound, loBound), ops.FromDouble(bins));
        for (int i = 0; i < src.Length; i++)
        {
            var v = src[i];
            if (ops.LessThan(v, loBound) || ops.GreaterThan(v, hiBound)) continue;
            int idx;
            if (ops.Equals(v, hiBound)) idx = bins - 1;
            else
            {
                var f = ops.Divide(ops.Subtract(v, loBound), width);
                idx = ops.ToInt32(ops.Floor(f));
                if (idx >= bins) idx = bins - 1;
                if (idx < 0) idx = 0;
            }
            dst[idx] = ops.Add(dst[idx], ops.One);
        }
        return result;
    }

    /// <inheritdoc/>
    public virtual bool TensorEqual<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape)) return false;
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var ops = MathHelper.GetNumericOperations<T>();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        for (int i = 0; i < av.Length; i++)
        {
            // NaN != NaN per PyTorch semantics; Equals returns false for NaN.
            if (!ops.Equals(av[i], bv[i])) return false;
        }
        return true;
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorEq<T>(Tensor<T> a, Tensor<T> b)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (b == null) throw new ArgumentNullException(nameof(b));
        if (!a._shape.SequenceEqual(b._shape))
            throw new ArgumentException("Eq: tensors must have the same shape (broadcast TBD)");
        if (!a.IsContiguous) a = a.Contiguous();
        if (!b.IsContiguous) b = b.Contiguous();
        var ops = MathHelper.GetNumericOperations<T>();
        var av = a.AsSpan();
        var bv = b.AsSpan();
        var result = new Tensor<Bit>(a._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < av.Length; i++)
            dst[i] = ops.Equals(av[i], bv[i]) ? Bit.True : Bit.False;
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<Bit> TensorEqScalar<T>(Tensor<T> a, T scalar)
    {
        if (a == null) throw new ArgumentNullException(nameof(a));
        if (!a.IsContiguous) a = a.Contiguous();
        var ops = MathHelper.GetNumericOperations<T>();
        var av = a.AsSpan();
        var result = new Tensor<Bit>(a._shape);
        var dst = result.AsWritableSpan();
        for (int i = 0; i < av.Length; i++)
            dst[i] = ops.Equals(av[i], scalar) ? Bit.True : Bit.False;
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorTensorSplit<T>(Tensor<T> tensor, int sections, int dim = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (sections <= 0) throw new ArgumentOutOfRangeException(nameof(sections));
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));

        int dimSize = tensor._shape[dim];
        // PyTorch rule: first (dimSize % sections) chunks have ceil(dimSize/sections),
        // remaining chunks have floor(dimSize/sections).
        int baseSize = dimSize / sections;
        int extra = dimSize % sections;

        var indices = new int[sections - 1];
        int cursor = 0;
        for (int i = 0; i < sections - 1; i++)
        {
            cursor += baseSize + (i < extra ? 1 : 0);
            indices[i] = cursor;
        }
        return TensorTensorSplit(tensor, indices, dim);
    }

    /// <inheritdoc/>
    public virtual Tensor<T>[] TensorTensorSplit<T>(Tensor<T> tensor, int[] indices, int dim = 0)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (indices == null) throw new ArgumentNullException(nameof(indices));
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        int dimSize = tensor._shape[dim];

        var result = new Tensor<T>[indices.Length + 1];
        int prev = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            int cur = System.Math.Min(System.Math.Max(indices[i], prev), dimSize);
            result[i] = SliceAlongAxis(tensor, dim, prev, cur);
            prev = cur;
        }
        result[indices.Length] = SliceAlongAxis(tensor, dim, prev, dimSize);
        return result;
    }

    /// <summary>
    /// Slice along <paramref name="dim"/> from start (inclusive) to end (exclusive).
    /// Produces a contiguous tensor; empty-slice (end <= start) returns an
    /// empty tensor with the sliced dim zero.
    /// </summary>
    private static Tensor<T> SliceAlongAxis<T>(Tensor<T> tensor, int dim, int start, int end)
    {
        int rank = tensor.Rank;
        int len = System.Math.Max(0, end - start);
        var outShape = new int[rank];
        for (int k = 0; k < rank; k++) outShape[k] = tensor._shape[k];
        outShape[dim] = len;

        if (len == 0) return new Tensor<T>(outShape);
        if (!tensor.IsContiguous) tensor = tensor.Contiguous();

        var src = tensor.AsSpan();
        var result = new Tensor<T>(outShape);
        var dst = result.AsWritableSpan();
        int outer = 1; for (int k = 0; k < dim; k++) outer *= tensor._shape[k];
        int inner = 1; for (int k = dim + 1; k < rank; k++) inner *= tensor._shape[k];
        int srcStride = tensor._shape[dim] * inner;
        int dstStride = len * inner;
        for (int o = 0; o < outer; o++)
            for (int i = 0; i < len; i++)
                for (int n = 0; n < inner; n++)
                    dst[o * dstStride + i * inner + n] =
                        src[o * srcStride + (start + i) * inner + n];
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorUnfold<T>(Tensor<T> tensor, int dim, int size, int step)
    {
        if (tensor == null) throw new ArgumentNullException(nameof(tensor));
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "size must be positive");
        if (step <= 0) throw new ArgumentOutOfRangeException(nameof(step), "step must be positive");
        int rank = tensor.Rank;
        if (dim < 0) dim += rank;
        if (dim < 0 || dim >= rank) throw new ArgumentOutOfRangeException(nameof(dim));
        int dimSize = tensor._shape[dim];
        if (size > dimSize)
            throw new ArgumentException(
                $"Unfold size {size} exceeds dimension size {dimSize} along axis {dim}");

        int nWindows = (dimSize - size) / step + 1;

        // Output shape: tensor.Shape with shape[dim] replaced by nWindows and
        // a new trailing axis of length 'size'.
        var outShape = new int[rank + 1];
        for (int i = 0; i < rank; i++) outShape[i] = tensor._shape[i];
        outShape[dim] = nWindows;
        outShape[rank] = size;

        if (!tensor.IsContiguous) tensor = tensor.Contiguous();
        var src = tensor.AsSpan();
        var result = AutoTensorCache.RentOrAllocate<T>(outShape);
        var dst = result.AsWritableSpan();

        int outerSize = 1;
        for (int k = 0; k < dim; k++) outerSize *= tensor._shape[k];
        int innerSize = 1;
        for (int k = dim + 1; k < rank; k++) innerSize *= tensor._shape[k];

        // Source strides for the original tensor.
        int srcDimStride = innerSize;                  // stride of `dim` in source
        int srcOuterStride = dimSize * innerSize;      // stride of axes < dim

        // Destination strides: out shape = outer × nWindows × inner × size
        int dstSizeStride = 1;
        int dstInnerStride = size;
        int dstWindowStride = innerSize * size;
        int dstOuterStride = nWindows * dstWindowStride;

        for (int outer = 0; outer < outerSize; outer++)
        {
            int srcOuterBase = outer * srcOuterStride;
            int dstOuterBase = outer * dstOuterStride;
            for (int w = 0; w < nWindows; w++)
            {
                int windowStart = w * step;
                int dstWindowBase = dstOuterBase + w * dstWindowStride;
                for (int inner = 0; inner < innerSize; inner++)
                {
                    int dstInnerBase = dstWindowBase + inner * dstInnerStride;
                    for (int s = 0; s < size; s++)
                    {
                        int srcPos = srcOuterBase
                                   + (windowStart + s) * srcDimStride
                                   + inner;
                        dst[dstInnerBase + s * dstSizeStride] = src[srcPos];
                    }
                }
            }
        }

        DifferentiableOps.RecordUnary(
            "TensorUnfold", result, tensor, BackwardFunctions<T>.UnfoldParity210Backward,
            savedState: new object[] { dim, size, step, (int[])tensor._shape.Clone() });
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorZeta<T>(Tensor<T> x, Tensor<T> q)
    {
        // ζ(x, q) = Σ_{k=0}^∞ (k + q)^{-x}.
        //
        // Strategy: accelerated Euler-Maclaurin summation. For x ≤ 1 the
        // series diverges outside the Re(x) > 1 region; PyTorch returns +Inf
        // at x = 1 and NaN for x < 1 with non-positive-integer q. We mirror
        // that by emitting +Inf at poles and leaving the series to diverge
        // visibly elsewhere (the caller should stay in x > 1).
        //
        // Convergence: sum the first N direct terms, then apply the
        // Euler-Maclaurin correction:
        //   ζ(x, q) ≈ Σ_{k=0}^{N-1} (k+q)^{-x}
        //           + (N+q)^{1-x} / (x-1)
        //           + 0.5 · (N+q)^{-x}
        //           + Σ_{j=1}^M  B_{2j}/(2j)! · (x)(x+1)…(x+2j-2) · (N+q)^{-x-2j+1}
        // with N = 10, M = 8 — single-precision accurate for x ≥ 1 + 1e-3
        // (the gray zone just above x = 1 loses a couple digits, same as
        // torch).
        if (x == null) throw new ArgumentNullException(nameof(x));
        if (q == null) throw new ArgumentNullException(nameof(q));
        if (!x._shape.SequenceEqual(q._shape))
            throw new ArgumentException("Zeta: x and q must have the same shape");

        var result = ElementwiseBinary(x, q, (xv, qv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(xv, System.Globalization.CultureInfo.InvariantCulture);
            double qd = System.Convert.ToDouble(qv, System.Globalization.CultureInfo.InvariantCulture);
            return ops.FromDouble(ZetaScalar(xd, qd));
        }, "TensorZeta");
        DifferentiableOps.RecordBinary("TensorZeta", result, x, q, BackwardFunctions<T>.ZetaBackward);
        return result;
    }

    /// <summary>
    /// Scalar Hurwitz zeta via Euler-Maclaurin with eight Bernoulli corrections.
    /// </summary>
    private static double ZetaScalar(double x, double q)
    {
        // Pole at x = 1.
        if (x == 1.0) return double.PositiveInfinity;
        // ζ(x, q) for q <= 0 hits a pole at every non-positive integer.
        if (q <= 0.0 && q == System.Math.Floor(q)) return double.PositiveInfinity;

        // Direct sum of first N terms.
        const int N = 12;
        double sum = 0.0;
        for (int k = 0; k < N; k++)
            sum += System.Math.Pow(k + q, -x);

        double Nq = N + q;
        double lnNq = System.Math.Log(Nq);
        // Integral + half-correction.
        double cont = System.Math.Exp((1.0 - x) * lnNq) / (x - 1.0);
        double half = 0.5 * System.Math.Exp(-x * lnNq);

        // Euler-Maclaurin Bernoulli corrections.
        // Coefficient for the 2j-th term is B_{2j}/(2j)! · (x)(x+1)…(x+2j-2)
        // multiplied by (N+q)^{-x-2j+1}.
        double[] b2k = { 1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0,
                         5.0/66.0, -691.0/2730.0, 7.0/6.0, -3617.0/510.0 };
        double[] fact2k = { 2.0, 24.0, 720.0, 40320.0,
                            3628800.0, 479001600.0, 87178291200.0, 20922789888000.0 };
        double corr = 0.0;
        double xPow = System.Math.Exp(-x * lnNq) / Nq;  // (N+q)^(-x-1)
        double invNq2 = 1.0 / (Nq * Nq);
        double rising = x;              // (x)_0 -> (x), will extend to (x)(x+1)…
        for (int j = 1; j <= 8; j++)
        {
            // (x)(x+1)...(x + 2j-2) — 2j-1 terms.
            if (j > 1)
            {
                rising *= (x + 2 * (j - 1) - 2) * (x + 2 * (j - 1) - 1);
                xPow *= invNq2;
            }
            double term = (b2k[j - 1] / fact2k[j - 1]) * rising * xPow;
            corr += term;
            if (System.Math.Abs(term) < 1e-16 * System.Math.Abs(sum + cont + half)) break;
        }

        return sum + cont + half + corr;
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

        if (!sortedSequence.IsContiguous) sortedSequence = sortedSequence.Contiguous();
        if (!values.IsContiguous) values = values.Contiguous();
        var result = new Tensor<int>(values._shape);
        var dst = result.AsWritableSpan();

        // Float32 fast path: route through SortKernels which uses AVX2
        // popcount-masked comparison for short sequences and a branchless
        // binary search for longer ones.
        if (typeof(T) == typeof(float))
        {
            float[] seqArr = (float[])(object)sortedSequence.GetDataArray();
            float[] vsArr = (float[])(object)values.GetDataArray();
            ReadOnlySpan<float> seqSpan = seqArr;
            if (right)
                for (int i = 0; i < vsArr.Length; i++) dst[i] = Simd.SortKernels.UpperBoundFloat(seqSpan, vsArr[i]);
            else
                for (int i = 0; i < vsArr.Length; i++) dst[i] = Simd.SortKernels.LowerBoundFloat(seqSpan, vsArr[i]);
            return result;
        }

        var numOps = MathHelper.GetNumericOperations<T>();
        var seqT = sortedSequence.AsSpan();
        var vsT = values.AsSpan();
        for (int i = 0; i < vsT.Length; i++)
        {
            // Branchless-ready binary search; returns insertion index.
            int lo = 0, hi = seqT.Length;
            var v = vsT[i];
            while (lo < hi)
            {
                int mid = lo + ((hi - lo) >> 1);
                // right=false → lower bound: split at seq[mid] >= v (v <= seq[mid]).
                // right=true  → upper bound: split at seq[mid] >  v (v <  seq[mid]).
                bool beforeMid = right
                    ? numOps.LessThan(v, seqT[mid])
                    : numOps.LessThanOrEquals(v, seqT[mid]);
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
    public virtual Tensor<int> TensorBinCount(Tensor<int> input, int? minLength = null)
    {
        if (input == null) throw new ArgumentNullException(nameof(input));
        if (input.Rank != 1) throw new ArgumentException("BinCount requires a 1-D int tensor");
        if (!input.IsContiguous) input = input.Contiguous();
        var src = input.AsSpan();

        // Find max; reject negatives (torch.bincount rejects them too).
        int maxV = minLength ?? 0;
        for (int i = 0; i < src.Length; i++)
        {
            if (src[i] < 0)
                throw new ArgumentException($"BinCount requires non-negative ints; got {src[i]}");
            if (src[i] >= maxV) maxV = src[i] + 1;
        }
        if (maxV == 0) return new Tensor<int>(new[] { 0 });

        var result = new Tensor<int>(new[] { maxV });
        var dst = result.AsWritableSpan();
        for (int i = 0; i < src.Length; i++) dst[src[i]]++;
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorMultiDot<T>(Tensor<T>[] matrices)
    {
        if (matrices == null) throw new ArgumentNullException(nameof(matrices));
        if (matrices.Length == 0) throw new ArgumentException("MultiDot requires at least one matrix");
        if (matrices.Length == 1) return matrices[0];

        // Build einsum equation: "ab,bc,cd,...->a?" — one distinct char per
        // contraction axis. The greedy path optimizer inside TensorEinsum
        // picks an efficient contraction order automatically.
        var labels = new System.Collections.Generic.List<string>(matrices.Length);
        char cursor = 'a';
        string prev = new string(cursor++, 1) + new string(cursor++, 1);
        labels.Add(prev);
        for (int i = 1; i < matrices.Length; i++)
        {
            // The right-side label of prev becomes the left-side label of this.
            char left = prev[1];
            char right = cursor++;
            string cur = new string(new[] { left, right });
            labels.Add(cur);
            prev = cur;
        }
        string outLabels = new string(new[] { labels[0][0], prev[1] });
        string eq = string.Join(",", labels) + "->" + outLabels;
        return TensorEinsum(eq, matrices);
    }

    /// <inheritdoc/>
    public virtual Tensor<int> TensorHistogramDD<T>(
        Tensor<T> samples, int[] bins, T[] mins, T[] maxs)
    {
        if (samples == null) throw new ArgumentNullException(nameof(samples));
        if (bins == null) throw new ArgumentNullException(nameof(bins));
        if (mins == null) throw new ArgumentNullException(nameof(mins));
        if (maxs == null) throw new ArgumentNullException(nameof(maxs));
        if (samples.Rank != 2)
            throw new ArgumentException("HistogramDD expects samples of shape [N, D]");
        int n = samples._shape[0];
        int d = samples._shape[1];
        if (bins.Length != d || mins.Length != d || maxs.Length != d)
            throw new ArgumentException($"bins / mins / maxs must have length D={d}");

        var ops = MathHelper.GetNumericOperations<T>();
        for (int k = 0; k < d; k++)
        {
            if (bins[k] < 1) throw new ArgumentOutOfRangeException(nameof(bins));
            if (ops.GreaterThanOrEquals(mins[k], maxs[k]))
                throw new ArgumentException($"bins[{k}]: mins must be < maxs");
        }

        if (!samples.IsContiguous) samples = samples.Contiguous();
        var src = samples.AsSpan();

        // Output shape is bins[0] × bins[1] × ... × bins[d-1].
        var result = new Tensor<int>(bins);
        var dst = result.AsWritableSpan();

        // Precompute widths and row-major strides of the output.
        var widths = new T[d];
        for (int k = 0; k < d; k++)
            widths[k] = ops.Divide(ops.Subtract(maxs[k], mins[k]), ops.FromDouble(bins[k]));
        var strides = ComputeRowMajorStrides(bins);

        // Walk samples; compute bin index per dim; drop out-of-range samples.
        for (int i = 0; i < n; i++)
        {
            int binIdx = 0;
            bool inRange = true;
            for (int k = 0; k < d; k++)
            {
                var v = src[i * d + k];
                if (ops.LessThan(v, mins[k]) || ops.GreaterThan(v, maxs[k]))
                {
                    inRange = false; break;
                }
                int kIdx;
                if (ops.Equals(v, maxs[k])) kIdx = bins[k] - 1;
                else
                {
                    var f = ops.Divide(ops.Subtract(v, mins[k]), widths[k]);
                    kIdx = ops.ToInt32(ops.Floor(f));
                    if (kIdx >= bins[k]) kIdx = bins[k] - 1;
                    if (kIdx < 0) kIdx = 0;
                }
                binIdx += kIdx * strides[k];
            }
            if (inRange) dst[binIdx]++;
        }
        return result;
    }

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
    {
        var lazy = TryRecordLazyBinary("TensorHypot", a, b,
            eng => eng.TensorHypot(a, b), BackwardFunctions<T>.HypotBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Sqrt(ops.Add(ops.Multiply(ax, ax), ops.Multiply(bx, bx)));
        }, "TensorHypot");
        DifferentiableOps.RecordBinary("TensorHypot", result, a, b, BackwardFunctions<T>.HypotBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorCopysign<T>(Tensor<T> a, Tensor<T> b)
    {
        var lazy = TryRecordLazyBinary("TensorCopysign", a, b,
            eng => eng.TensorCopysign(a, b), BackwardFunctions<T>.CopysignBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            var sign = ops.SignOrZero(bx);
            var mag = ops.Abs(ax);
            return ops.LessThan(sign, ops.Zero) ? ops.Negate(mag) : mag;
        }, "TensorCopysign");
        DifferentiableOps.RecordBinary("TensorCopysign", result, a, b, BackwardFunctions<T>.CopysignBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFmod<T>(Tensor<T> a, Tensor<T> b)
    {
        var lazy = TryRecordLazyBinary("TensorFmod", a, b,
            eng => eng.TensorFmod(a, b), BackwardFunctions<T>.FmodBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            if (ops.Equals(bx, ops.Zero)) return ops.Zero;
            var q = TruncateTowardZero(ops.Divide(ax, bx));
            return ops.Subtract(ax, ops.Multiply(q, bx));
        }, "TensorFmod");
        DifferentiableOps.RecordBinary("TensorFmod", result, a, b, BackwardFunctions<T>.FmodBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorRemainder<T>(Tensor<T> a, Tensor<T> b)
    {
        var lazy = TryRecordLazyBinary("TensorRemainder", a, b,
            eng => eng.TensorRemainder(a, b), BackwardFunctions<T>.RemainderBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(a, b, (ax, bx) => {
            var ops = MathHelper.GetNumericOperations<T>();
            if (ops.Equals(bx, ops.Zero)) return ops.Zero;
            var q = ops.Floor(ops.Divide(ax, bx));
            return ops.Subtract(ax, ops.Multiply(q, bx));
        }, "TensorRemainder");
        DifferentiableOps.RecordBinary("TensorRemainder", result, a, b, BackwardFunctions<T>.RemainderBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorFloatPower<T>(Tensor<T> a, Tensor<T> b)
    {
        var lazy = TryRecordLazyBinary("TensorFloatPower", a, b,
            eng => eng.TensorFloatPower(a, b), BackwardFunctions<T>.FloatPowerBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(a, b, (ax, bx) =>
            MathHelper.GetNumericOperations<T>().Power(ax, bx), "TensorFloatPower");
        DifferentiableOps.RecordBinary("TensorFloatPower", result, a, b, BackwardFunctions<T>.FloatPowerBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogAddExp<T>(Tensor<T> a, Tensor<T> b)
    {
        var lazy = TryRecordLazyBinary("TensorLogAddExp", a, b,
            eng => eng.TensorLogAddExp(a, b), BackwardFunctions<T>.LogAddExpBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(a, b, (av, bv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            var larger = ops.GreaterThan(av, bv) ? av : bv;
            var smaller = ops.GreaterThan(av, bv) ? bv : av;
            var diff = ops.Subtract(smaller, larger);
            var expDiff = ops.Exp(diff);
            return ops.Add(larger, ops.Log(ops.Add(ops.One, expDiff)));
        }, "TensorLogAddExp");
        DifferentiableOps.RecordBinary("TensorLogAddExp", result, a, b, BackwardFunctions<T>.LogAddExpBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLogAddExp2<T>(Tensor<T> a, Tensor<T> b)
    {
        var lazy = TryRecordLazyBinary("TensorLogAddExp2", a, b,
            eng => eng.TensorLogAddExp2(a, b), BackwardFunctions<T>.LogAddExp2Backward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(a, b, (av, bv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            var larger = ops.GreaterThan(av, bv) ? av : bv;
            var smaller = ops.GreaterThan(av, bv) ? bv : av;
            var diff = ops.Subtract(smaller, larger);
            var ln2 = ops.FromDouble(System.Math.Log(2.0));
            var pow2 = ops.Exp(ops.Multiply(diff, ln2));
            var log1p = ops.Log(ops.Add(ops.One, pow2));
            return ops.Add(larger, ops.Divide(log1p, ln2));
        }, "TensorLogAddExp2");
        DifferentiableOps.RecordBinary("TensorLogAddExp2", result, a, b, BackwardFunctions<T>.LogAddExp2Backward);
        return result;
    }

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
            double scale = System.Math.Pow(2.0, e[i]);
            dst[i] = ops.Multiply(src[i], ops.FromDouble(scale));
        }
        DifferentiableOps.RecordUnary("TensorLdexp", result, x,
            BackwardFunctions<T>.LdexpBackward, new object[] { exp });
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
    {
        var lazy = TryRecordLazyUnary("TensorErfc", tensor,
            eng => eng.TensorErfc(tensor), BackwardFunctions<T>.ErfcBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Subtract(ops.One, MathHelper.Erf(x));
        }, "TensorErfc");
        DifferentiableOps.RecordUnary("TensorErfc", result, tensor, BackwardFunctions<T>.ErfcBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorXlogy<T>(Tensor<T> x, Tensor<T> y)
    {
        var lazy = TryRecordLazyBinary("TensorXlogy", x, y,
            eng => eng.TensorXlogy(x, y), BackwardFunctions<T>.XlogyBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(x, y, (xv, yv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Equals(xv, ops.Zero) ? ops.Zero : ops.Multiply(xv, ops.Log(yv));
        }, "TensorXlogy");
        DifferentiableOps.RecordBinary("TensorXlogy", result, x, y, BackwardFunctions<T>.XlogyBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorXlog1py<T>(Tensor<T> x, Tensor<T> y)
    {
        var lazy = TryRecordLazyBinary("TensorXlog1py", x, y,
            eng => eng.TensorXlog1py(x, y), BackwardFunctions<T>.Xlog1pyBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseBinary(x, y, (xv, yv) => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Equals(xv, ops.Zero)
                ? ops.Zero
                : ops.Multiply(xv, ops.Log(ops.Add(ops.One, yv)));
        }, "TensorXlog1py");
        DifferentiableOps.RecordBinary("TensorXlog1py", result, x, y, BackwardFunctions<T>.Xlog1pyBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorLgamma<T>(Tensor<T> tensor)
    {
        var lazy = TryRecordLazyUnary("TensorLgamma", tensor,
            eng => eng.TensorLgamma(tensor), BackwardFunctions<T>.LgammaBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            return ops.Log(ops.Abs(MathHelper.Gamma(x)));
        }, "TensorLgamma");
        DifferentiableOps.RecordUnary("TensorLgamma", result, tensor, BackwardFunctions<T>.LgammaBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorPolygamma<T>(int n, Tensor<T> tensor)
    {
        if (n < 0) throw new ArgumentOutOfRangeException(nameof(n), "Polygamma order must be >= 0");
        if (n == 0) return TensorDigamma(tensor);

        var result = ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
            return ops.FromDouble(PolygammaScalar(n, xd));
        }, "TensorPolygamma");

        DifferentiableOps.RecordUnary(
            "TensorPolygamma", result, tensor, BackwardFunctions<T>.PolygammaBackward,
            savedState: new object[] { n });
        return result;
    }

    /// <summary>
    /// Scalar polygamma for arbitrary order n ≥ 1. Uses shift-up recurrence to
    /// push the argument into the stable asymptotic regime, then sums the
    /// general asymptotic series in terms of Bernoulli numbers:
    /// <para>
    ///   ψ^(n)(x) = (-1)^(n+1) · [ (n-1)!/x^n + n!/(2 x^(n+1))
    ///                             + Σ_{k≥1} B_{2k} · (2k+n-1)!/(2k)! / x^(2k+n) ]
    /// </para>
    /// </summary>
    private static double PolygammaScalar(int n, double x)
    {
        // For non-positive integer arguments polygamma has poles at x = 0, -1, -2, ...
        if (x <= 0.0 && x == System.Math.Floor(x))
            return double.PositiveInfinity;

        // Reflection for x < 0.5 to put x in the forward-recurrence-friendly region.
        // PyTorch uses the same split; the reflection uses higher-order cotangent
        // derivatives, which cost more than a handful of recurrence steps, so we
        // prefer to just recurrence-up for x < shift threshold.
        double recurrence = 0.0;
        double xd = x;
        int signN = (n & 1) == 1 ? 1 : -1;  // (-1)^(n+1)

        // Shift up until xd ≥ 10 so the asymptotic tail converges in ≤ 8 terms.
        while (xd < 10.0)
        {
            // ψ^(n)(x) = ψ^(n)(x+1) + (-1)^n · n! / x^(n+1)
            // Accumulate -(-1)^n · n! / x^(n+1) in `recurrence` so that
            // result = recurrence + asymptotic(xd_shifted).
            recurrence += signN * System.Math.Exp(FactorialLogD(n) - (n + 1) * System.Math.Log(xd));
            xd += 1.0;
        }

        // Asymptotic:  ψ^(n)(xd) = (-1)^(n+1) · [ (n-1)!/xd^n + n!/(2·xd^(n+1)) + Σ ... ]
        double lnX = System.Math.Log(xd);
        double leading = System.Math.Exp(FactorialLogD(n - 1) - n * lnX);
        double half    = 0.5 * System.Math.Exp(FactorialLogD(n) - (n + 1) * lnX);
        double asympt = leading + half;

        // B_{2k} (Bernoulli), k = 1..8 — sufficient for xd ≥ 10, n ≤ ~20.
        // B_2 = 1/6, B_4 = -1/30, B_6 = 1/42, B_8 = -1/30, B_10 = 5/66,
        // B_12 = -691/2730, B_14 = 7/6, B_16 = -3617/510.
        double[] b2k = { 1.0/6.0, -1.0/30.0, 1.0/42.0, -1.0/30.0,
                         5.0/66.0, -691.0/2730.0, 7.0/6.0, -3617.0/510.0 };
        double invX2 = 1.0 / (xd * xd);
        double xPow = 1.0 / System.Math.Pow(xd, n);  // xd^-n
        for (int k = 1; k <= 8; k++)
        {
            // term = B_{2k} · (2k+n-1)! / (2k)! · xd^-(2k+n)
            xPow *= invX2;  // xd^-(n+2k)
            double logTerm = FactorialLogD(2 * k + n - 1) - FactorialLogD(2 * k);
            double term = b2k[k - 1] * System.Math.Exp(logTerm) * xPow;
            asympt += term;
            if (System.Math.Abs(term) < 1e-16 * System.Math.Abs(asympt)) break;
        }

        return recurrence + signN * asympt;
    }

    /// <summary>Natural log of k! via lgamma(k+1). Safe for k up to 170.</summary>
    private static double FactorialLogD(int k)
    {
        if (k <= 1) return 0.0;
        // Direct log-sum for small k; Stirling for larger. Use MathHelper.Gamma
        // would overflow for k > 170 so we take logs all the way through.
        double s = 0.0;
        for (int i = 2; i <= k; i++) s += System.Math.Log(i);
        return s;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorErfinv<T>(Tensor<T> tensor)
    {
        var lazy = TryRecordLazyUnary("TensorErfinv", tensor,
            eng => eng.TensorErfinv(tensor), BackwardFunctions<T>.ErfinvBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseUnary(tensor, x => {
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
        DifferentiableOps.RecordUnary("TensorErfinv", result, tensor, BackwardFunctions<T>.ErfinvBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI0<T>(Tensor<T> tensor)
    {
        var lazy = TryRecordLazyUnary("TensorI0", tensor,
            eng => eng.TensorI0(tensor), BackwardFunctions<T>.I0Backward);
        if (lazy != null) return lazy;
        var result = ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
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
        DifferentiableOps.RecordUnary("TensorI0", result, tensor, BackwardFunctions<T>.I0Backward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI1<T>(Tensor<T> tensor)
    {
        var lazy = TryRecordLazyUnary("TensorI1", tensor,
            eng => eng.TensorI1(tensor), BackwardFunctions<T>.I1Backward);
        if (lazy != null) return lazy;
        var result = ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = System.Convert.ToDouble(x, System.Globalization.CultureInfo.InvariantCulture);
            double halfX = xd / 2.0;
            double halfSq = halfX * halfX;
            double term = 1.0;
            double sum = 1.0;
            for (int k = 1; k < 25; k++)
            {
                term *= halfSq / (k * (k + 1));
                sum += term;
                if (term < 1e-16) break;
            }
            return ops.FromDouble(halfX * sum);
        }, "TensorI1");
        DifferentiableOps.RecordUnary("TensorI1", result, tensor, BackwardFunctions<T>.I1Backward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI0e<T>(Tensor<T> tensor)
    {
        var lazy = TryRecordLazyUnary("TensorI0e", tensor,
            eng => eng.TensorI0e(tensor), BackwardFunctions<T>.I0eBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseUnary(tensor, x => {
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
        DifferentiableOps.RecordUnary("TensorI0e", result, tensor, BackwardFunctions<T>.I0eBackward);
        return result;
    }

    /// <inheritdoc/>
    public virtual Tensor<T> TensorI1e<T>(Tensor<T> tensor)
    {
        var lazy = TryRecordLazyUnary("TensorI1e", tensor,
            eng => eng.TensorI1e(tensor), BackwardFunctions<T>.I1eBackward);
        if (lazy != null) return lazy;
        var result = ElementwiseUnary(tensor, x => {
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
        DifferentiableOps.RecordUnary("TensorI1e", result, tensor, BackwardFunctions<T>.I1eBackward);
        return result;
    }

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
    {
        var result = ElementwiseUnary(tensor, x => {
            var ops = MathHelper.GetNumericOperations<T>();
            double xd = ToDoubleSafe(x);
            double acc = 0;
            while (xd < 6.0)
            {
                acc -= 1.0 / xd;
                xd += 1.0;
            }
            double xinv = 1.0 / xd;
            double xinv2 = xinv * xinv;
            acc += System.Math.Log(xd) - 0.5 * xinv
                     - xinv2 * (1.0 / 12.0 - xinv2 * (1.0 / 120.0 - xinv2 / 252.0));
            return ops.FromDouble(acc);
        }, "TensorDigamma");
        DifferentiableOps.RecordUnary("TensorDigamma", result, tensor, BackwardFunctions<T>.DigammaBackward);
        return result;
    }

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

    /// <summary>
    /// Validates that <paramref name="bounds"/> can broadcast against
    /// <paramref name="target"/> (NumPy / PyTorch semantics) and returns the
    /// per-axis stride vector (in <paramref name="target"/>'s row-major
    /// order) to convert a target index to a bounds flat index. Returns null
    /// when the shapes match exactly (linear index works verbatim).
    /// Throws when they can't broadcast.
    /// </summary>
    private static int[]? ValidateAndComputeClampBroadcastStrides(int[] target, int[]? bounds)
    {
        if (bounds == null) return null;
        if (target.SequenceEqual(bounds)) return null;

        int tr = target.Length, br = bounds.Length;
        if (br > tr)
            throw new ArgumentException(
                $"clamp bounds shape [{string.Join(",", bounds)}] has higher rank than tensor [{string.Join(",", target)}]");

        // Right-align shapes: walk from the trailing axis back.  Each axis
        // must be 1 or equal to the target axis size.
        var strides = new int[tr];
        int bStride = 1;
        for (int k = tr - 1; k >= 0; k--)
        {
            int boundsAxis = k - (tr - br);        // index into bounds, or -1 if padded
            int boundsDim = boundsAxis >= 0 ? bounds[boundsAxis] : 1;
            if (boundsDim != 1 && boundsDim != target[k])
                throw new ArgumentException(
                    $"clamp bounds shape [{string.Join(",", bounds)}] not broadcastable against tensor [{string.Join(",", target)}]");
            // Stride is 0 when broadcasting (dim size 1 in bounds), else the
            // physical stride of that axis in the bounds' contiguous layout.
            strides[k] = boundsDim == 1 ? 0 : bStride;
            if (boundsDim != 1) bStride *= boundsDim;
        }
        return strides;
    }

    /// <summary>Flat-index lookup into a broadcast-strided bounds tensor.</summary>
    private static int BroadcastLookup(int[] idx, int[] strides)
    {
        int pos = 0;
        for (int k = 0; k < idx.Length; k++) pos += idx[k] * strides[k];
        return pos;
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

    /// <summary>
    /// LazyTensorScope capture helper for unary element-wise parity-210 ops.
    /// Returns null if graph mode is not active; callers fall through to the
    /// eager implementation when null is returned.
    /// </summary>
    private static Tensor<T>? TryRecordLazyUnary<T>(
        string opName, Tensor<T> input, Func<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backward = null, object[]? savedState = null)
    {
        if (!GraphMode.IsActive) return null;
        var scope = GraphMode.Current;
        if (scope == null) return null;
        return scope.RecordUnary(LazyNodeType.Custom, opName, input, (int[])input._shape.Clone(),
            (eng, output) => { var r = execute(eng); r.AsSpan().CopyTo(output.AsWritableSpan()); },
            backward, savedState);
    }

    /// <summary>
    /// LazyTensorScope capture helper for binary element-wise parity-210 ops.
    /// Output shape is inferred from operand a (callers ensure a and b broadcast
    /// to a.Shape).  Returns null when graph mode is not active.
    /// </summary>
    private static Tensor<T>? TryRecordLazyBinary<T>(
        string opName, Tensor<T> a, Tensor<T> b, Func<IEngine, Tensor<T>> execute,
        BackwardFunction<T>? backward = null, object[]? savedState = null)
    {
        if (!GraphMode.IsActive) return null;
        var scope = GraphMode.Current;
        if (scope == null) return null;
        return scope.RecordBinary(LazyNodeType.Custom, opName, a, b, (int[])a._shape.Clone(),
            (eng, output) => { var r = execute(eng); r.AsSpan().CopyTo(output.AsWritableSpan()); },
            backward, savedState);
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
