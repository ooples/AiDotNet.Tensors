// Copyright (c) AiDotNet. All rights reserved.

using System;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Tensors.LinearAlgebra.Sparse;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Tape-aware wrappers around <see cref="SparseOps"/>. Mirrors the
/// <c>torch.sparse</c> autograd surface so callers can chain sparse ops
/// inside a <see cref="GradientTape{T}"/> without the gradients
/// disappearing at the sparse boundary — that's the symptom PyTorch
/// users hit on <c>torch.sparse.mm</c> with non-COO inputs.
///
/// <para>Backward semantics match PyTorch's defaults: for <c>Y = A_sparse · B_dense</c>
/// we return a dense <c>dA</c> and dense <c>dB</c>. Pattern-preserving
/// sparse gradients are an opt-in via <see cref="SparseSampledAddMMRecord{T}"/>
/// (the only natural pattern-preserving op).</para>
///
/// <para>Coverage is the same as <see cref="SparseOps"/>: sparse·dense
/// matmul, sampled-addmm, sparse softmax / log-softmax. Higher-order
/// ops (sparse·sparse SpGEMM, sum/mean reductions) are pure
/// rearrangements; their backward routes back through the dense path
/// and is tracked in the parent issue's deferred list.</para>
/// </summary>
public static class SparseAutograd
{
    /// <summary>Tape-aware sparse · dense matmul. The backward returns
    /// dense gradients for both sides. <paramref name="aDense"/> is the
    /// caller-visible <see cref="Tensor{T}"/> the gradient for A is
    /// accumulated against — the same Tensor instance must be the one
    /// passed to <c>ComputeGradients(loss, new[] { aDense, b })</c>,
    /// otherwise the lookup misses and dA stays zero. The internal
    /// dense snapshot used to compute <c>A^T · grad</c> still comes from
    /// <c>a.ToDense()</c> at record time and is captured in savedState
    /// so backward sees the values <em>at forward time</em> — matches
    /// PyTorch's <c>sparse_mm_backward</c> which densifies before
    /// transposing.</summary>
    public static Tensor<T> SparseMatMulRecord<T>(SparseTensor<T> a, Tensor<T> aDense, Tensor<T> b)
    {
        if (aDense is null) throw new ArgumentNullException(nameof(aDense));
        var output = SparseOps.SparseMatMul(a, b);
        if (DifferentiableOps._anyTapeActive == 0) return output;

        // Snapshot A's values at forward time. Storing on savedState
        // ensures backward uses the values that produced this output
        // even if A was mutated downstream.
        var aDenseSnapshot = a.ToDense();
        DifferentiableOps.RecordBinary(
            "SparseMatMul",
            output,
            aDense,
            b,
            SparseMatMulBackward,
            savedState: new object[] { aDenseSnapshot });
        return output;
    }

    /// <summary>Tape-aware <c>α · (A_sparse · B) + β · C</c>.
    /// <paramref name="aDense"/> is the caller-visible Tensor that
    /// receives dA — see <see cref="SparseMatMulRecord{T}"/> for the
    /// rationale.</summary>
    public static Tensor<T> SparseAddMMRecord<T>(Tensor<T> c, SparseTensor<T> a, Tensor<T> aDense, Tensor<T> b, T alpha, T beta)
    {
        if (aDense is null) throw new ArgumentNullException(nameof(aDense));
        var output = SparseOps.SparseAddMM(c, a, b, alpha, beta);
        if (DifferentiableOps._anyTapeActive == 0) return output;

        var aDenseSnapshot = a.ToDense();
        var savedState = new object[] { alpha!, beta!, aDenseSnapshot };
        DifferentiableOps.RecordIfActive(
            "SparseAddMM",
            output,
            new[] { c, aDense, b },
            SparseAddMMBackward,
            savedState);
        return output;
    }

    /// <summary>Tape-aware <c>sampled_addmm</c>. Returns a dense
    /// materialisation of the sparse output so downstream tape ops
    /// (sum, log, etc.) can compose without a sparse-aware reducer;
    /// the backward preserves the pattern by routing dA through
    /// <see cref="SparseOps.SparseMask{T}"/> before the dense product.
    /// Use <see cref="SparseOps.SparseSampledAddMM{T}"/> directly if
    /// you want the sparse output object instead.</summary>
    public static Tensor<T> SparseSampledAddMMRecord<T>(SparseTensor<T> pattern,
        Tensor<T> a, Tensor<T> b, Tensor<T> c, T alpha, T beta)
    {
        var sparseOut = SparseOps.SparseSampledAddMM(pattern, a, b, c, alpha, beta);
        var output = sparseOut.ToDense();
        if (DifferentiableOps._anyTapeActive == 0) return output;

        var savedState = new object[] { pattern, alpha!, beta! };
        DifferentiableOps.RecordIfActive(
            "SparseSampledAddMM",
            output,
            new[] { a, b, c },
            SparseSampledAddMMBackward,
            savedState);
        return output;
    }

    /// <summary>Tape-aware sparse · dense matmul that produces a
    /// PATTERN-PRESERVING sparse gradient on the sparse side. Use this
    /// when the sparse tensor is a registered trainable parameter
    /// (typical: <c>SparseLinearLayer&lt;T&gt;</c> + sparse-aware
    /// <see cref="ParameterBuffer{T}"/>) — the produced gradient's
    /// <see cref="SparseTensor{T}.Values"/> length matches the original
    /// pattern's <c>NonZeroCount</c>, so the parameter buffer can store
    /// it as a flat values vector without materialising the dense matrix.
    /// Avoids the O(rows·columns) memory allocation that
    /// <see cref="SparseMatMulRecord{T}"/> incurs in its backward.
    /// </summary>
    /// <param name="a">Sparse left operand. Must be the registered
    /// trainable-parameter SparseTensor — the gradient is keyed against
    /// this exact instance, so reusing a different SparseTensor with the
    /// same pattern won't accumulate gradients correctly.</param>
    /// <param name="b">Dense right operand.</param>
    public static Tensor<T> SparsePatternPreservingMatMulRecord<T>(
        SparseTensor<T> a, Tensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        var output = SparseOps.SparseMatMul(a, b);
        if (DifferentiableOps._anyTapeActive == 0) return output;

        // Snapshot BOTH the COO pattern AND the values aligned with that
        // pattern. Backward computes dB by iterating non-zeros of A, and
        // the values must match the index order — but if A is stored in
        // CSR/BSR/etc. its DataVector ordering is NOT aligned with the
        // COO indices we save. ToCoo() materialises both in the right
        // order; snapshot the values once via ToArray() (single copy)
        // so the rest is allocation-free at backward time.
        var coo = a.ToCoo();
        var aValuesSnapshot = coo.DataVector.ToArray();
        DifferentiableOps.RecordBinary(
            "SparsePatternPreservingMatMul",
            output,
            a,    // sparse parameter — gradient accumulator keyed here
            b,
            SparsePatternPreservingMatMulBackward,
            savedState: new object[]
            {
                coo.RowIndices,
                coo.ColumnIndices,
                a.Rows,
                a.Columns,
                aValuesSnapshot,
            });
        return output;
    }

    private static void SparsePatternPreservingMatMulBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        // Y = A_sparse · B (pattern P fixed)
        // dB = A^T · dY  (dense form: standard SpMM with transposed A)
        // dA[i,j] for non-zero (i,j) ∈ P:
        //     dA[i,j] = sum_k B[j,k] · dY[i,k]
        // Pack the dA values in COO order matching the saved pattern.
        var aSparse = (SparseTensor<T>)inputs[0];
        var b = inputs[1];
        var rowIndices = (int[])savedState[0];
        var colIndices = (int[])savedState[1];
        int rows = (int)savedState[2];
        int columns = (int)savedState[3];
        var aValues = (T[])savedState[4];

        var ops = MathHelper.GetNumericOperations<T>();
        int nnz = rowIndices.Length;
        int innerK = b.Shape[1];

        // Compute dA over the pattern only — never materialises the
        // dense O(rows × columns) gradient.
        var gradValues = new T[nnz];
        for (int idx = 0; idx < nnz; idx++)
        {
            int i = rowIndices[idx];
            int j = colIndices[idx];
            T sum = ops.Zero;
            for (int k = 0; k < innerK; k++)
            {
                sum = ops.Add(sum, ops.Multiply(b[j, k], gradOutput[i, k]));
            }
            gradValues[idx] = sum;
        }
        var sparseGradA = new SparseTensor<T>(rows, columns, rowIndices, colIndices, gradValues);
        AccumulateGrad(aSparse, sparseGradA, gradAccumulator, engine);

        // dB = A^T · dY computed as a sparse-transpose · dense matmul,
        // i.e. iterate the sparse pattern of A and accumulate into a
        // dense [columns × innerK] gradient. This avoids the O(rows × columns)
        // densification that materialising A and transposing it would
        // incur — the whole memory-efficiency point of the
        // pattern-preserving op. The values used here MUST be the
        // forward-time snapshot (`aValues`, captured during record in
        // COO order matching `rowIndices` / `colIndices`); reading
        // aSparse.DataVector directly would assume COO ordering, but
        // CSR/BSR-stored sparse tensors return values in a different
        // order. The snapshot fixes that misalignment.
        var gradB = new Tensor<T>(new[] { columns, innerK });
        var gradBSpan = gradB.AsWritableSpan();
        // gradB starts zero (Tensor ctor zero-fills); accumulate
        //   gradB[j, k] += A_value[idx] · dY[i, k]    where (i, j) ∈ pattern_A
        // Equivalent to A^T · dY without materialising the dense A.
        for (int idx = 0; idx < nnz; idx++)
        {
            int i = rowIndices[idx];
            int j = colIndices[idx];
            T aVal = aValues[idx];
            int rowStart = j * innerK;
            for (int k = 0; k < innerK; k++)
            {
                gradBSpan[rowStart + k] = ops.Add(
                    gradBSpan[rowStart + k],
                    ops.Multiply(aVal, gradOutput[i, k]));
            }
        }
        AccumulateGrad(b, gradB, gradAccumulator, engine);
        _ = output;
    }

    private static void SparseMatMulBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        // Y = A · B (A is the caller-visible dense Tensor; the values
        // used in the matmul are the snapshot in savedState[0])
        // dA = grad_Y · B^T  → accumulated against `aDense` (caller-visible)
        // dB = A^T · grad_Y  → uses the snapshot for the transpose
        var aDense = inputs[0];     // caller-visible target (gradient destination)
        var b = inputs[1];
        var aSnapshot = (Tensor<T>)savedState[0];

        var bT = engine.TensorTranspose(b);
        var gradA = engine.TensorMatMul(gradOutput, bT);
        AccumulateGrad(aDense, gradA, gradAccumulator, engine);

        var aT = engine.TensorTranspose(aSnapshot);
        var gradB = engine.TensorMatMul(aT, gradOutput);
        AccumulateGrad(b, gradB, gradAccumulator, engine);
        _ = output;
    }

    private static void SparseAddMMBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        // out = α · (A · B) + β · C
        // dC = β · grad_out
        // d(A·B) = α · grad_out  ⇒  dA = α · grad_out · B^T,  dB = α · A^T · grad_out
        T alpha = (T)savedState[0];
        T beta = (T)savedState[1];
        var aSnapshot = (Tensor<T>)savedState[2];
        var c = inputs[0];
        var aDense = inputs[1];     // caller-visible target
        var b = inputs[2];

        var gradC = engine.TensorMultiplyScalar(gradOutput, beta);
        AccumulateGrad(c, gradC, gradAccumulator, engine);

        var alphaGradOut = engine.TensorMultiplyScalar(gradOutput, alpha);
        var bT = engine.TensorTranspose(b);
        var gradA = engine.TensorMatMul(alphaGradOut, bT);
        AccumulateGrad(aDense, gradA, gradAccumulator, engine);

        var aT = engine.TensorTranspose(aSnapshot);
        var gradB = engine.TensorMatMul(aT, alphaGradOut);
        AccumulateGrad(b, gradB, gradAccumulator, engine);
        _ = output;
    }

    private static void SparseSampledAddMMBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        // out (sparse pattern P) = α · (A · B) | P + β · C | P
        // grad arrives as a Tensor<T> sized like the pattern's dense shape;
        // backward preserves P by masking the densified product.
        var pattern = (SparseTensor<T>)savedState[0];
        T alpha = (T)savedState[1];
        T beta = (T)savedState[2];
        var a = inputs[0];
        var b = inputs[1];
        var c = inputs[2];

        // C only contributes at pattern positions — grad through C is
        // β · grad masked by P. The earlier dense fall-through leaked
        // gradient onto structural-zero positions, which the autograd
        // pattern-preservation test rightly catches.
        var maskedGradOut = SparseOps.SparseMask(gradOutput, pattern).ToDense();
        var gradC = engine.TensorMultiplyScalar(maskedGradOut, beta);
        AccumulateGrad(c, gradC, gradAccumulator, engine);

        // dA = α · (grad ⊙ pattern_mask) · B^T
        // dB = α · A^T · (grad ⊙ pattern_mask)
        var alphaGrad = engine.TensorMultiplyScalar(maskedGradOut, alpha);

        var bT = engine.TensorTranspose(b);
        var gradA = engine.TensorMatMul(alphaGrad, bT);
        AccumulateGrad(a, gradA, gradAccumulator, engine);

        var aT = engine.TensorTranspose(a);
        var gradB = engine.TensorMatMul(aT, alphaGrad);
        AccumulateGrad(b, gradB, gradAccumulator, engine);
        _ = output;
    }

    /// <summary>Tape-aware <c>sum</c> over a sparse tensor's stored
    /// non-zeros. Returns a dense result so downstream tape ops can
    /// compose; gradient is keyed on <paramref name="aDense"/> (caller
    /// supplies the dense view it wants gradients on, since
    /// <see cref="SparseTensor{T}"/> doesn't accumulate gradients in
    /// its sparse storage form). The dense view is what the rest of
    /// autograd hooks against.</summary>
    public static Tensor<T> SparseSumRecord<T>(SparseTensor<T> a, Tensor<T> aDense, int? axis = null)
    {
        var output = SparseOps.SparseSum(a, axis);
        if (DifferentiableOps._anyTapeActive == 0) return output;

        var savedState = new object[] { axis as object ?? -1 };
        DifferentiableOps.RecordIfActive(
            "SparseSum",
            output,
            new[] { aDense },
            SparseSumBackward<T>,
            savedState);
        return output;
    }

    /// <summary>Tape-aware <c>mean</c> companion of <see cref="SparseSumRecord{T}"/>.
    /// Divisor matches PyTorch's <c>torch.sparse.mean</c> (dense element
    /// count along the reduction axis, including structural zeros).</summary>
    public static Tensor<T> SparseMeanRecord<T>(SparseTensor<T> a, Tensor<T> aDense, int? axis = null)
    {
        var output = SparseOps.SparseMean(a, axis);
        if (DifferentiableOps._anyTapeActive == 0) return output;

        var savedState = new object[] { axis as object ?? -1, a.Rows, a.Columns };
        DifferentiableOps.RecordIfActive(
            "SparseMean",
            output,
            new[] { aDense },
            SparseMeanBackward<T>,
            savedState);
        return output;
    }

    /// <summary>Tape-aware sparse softmax. Output is dense (same shape as
    /// the densified sparse input), pattern preserved through backward
    /// via the standard softmax-Jacobian
    /// <c>dx = (dy − sum(dy ⊙ y)) ⊙ y</c>, applied per-row over the stored
    /// non-zeros. Gradient is keyed on <paramref name="aDense"/>.</summary>
    public static Tensor<T> SparseSoftmaxRecord<T>(SparseTensor<T> a, Tensor<T> aDense)
    {
        var sparseOut = SparseOps.SparseSoftmax(a);
        var output = sparseOut.ToDense();
        if (DifferentiableOps._anyTapeActive == 0) return output;

        var savedState = new object[] { sparseOut, false /* takeLog */ };
        DifferentiableOps.RecordIfActive(
            "SparseSoftmax",
            output,
            new[] { aDense },
            SparseSoftmaxBackward<T>,
            savedState);
        return output;
    }

    /// <summary>Tape-aware sparse log-softmax. Same pattern as
    /// <see cref="SparseSoftmaxRecord{T}"/>; the backward uses
    /// <c>dx = dy − sum(dy) · y</c> where <c>y = softmax(x)</c>.</summary>
    public static Tensor<T> SparseLogSoftmaxRecord<T>(SparseTensor<T> a, Tensor<T> aDense)
    {
        var sparseLog = SparseOps.SparseLogSoftmax(a);
        var output = sparseLog.ToDense();
        if (DifferentiableOps._anyTapeActive == 0) return output;

        // Backward needs softmax (not log-softmax) values.
        var sparseSoftmax = SparseOps.SparseSoftmax(a);
        var savedState = new object[] { sparseSoftmax, true /* takeLog */ };
        DifferentiableOps.RecordIfActive(
            "SparseLogSoftmax",
            output,
            new[] { aDense },
            SparseSoftmaxBackward<T>,
            savedState);
        return output;
    }

    /// <summary>Tape-aware sparse · sparse matmul. Returns the dense
    /// materialisation of the CSR product so downstream tape ops compose;
    /// backward routes through the dense matmul Jacobian and is keyed on
    /// the dense views supplied by the caller.</summary>
    /// <summary>Pattern-preserving sparse · sparse matmul. Backward
    /// produces SPARSE gradients on both sides — values computed only at
    /// each operand's own non-zero pattern, not as a full dense matrix.
    /// Use this when both A and B are registered trainable parameters
    /// (typical: a sparse-sparse projection chain). The output is
    /// densified for downstream tape composition (matches the dense-grad
    /// SpGeMM variant); the gradient flowing back into A and B is
    /// pattern-preserving sparse and integrates with sparse-aware
    /// <see cref="ParameterBuffer{T}"/> slots without densification.
    /// </summary>
    /// <param name="a">Sparse left operand (also the gradient target —
    /// must be the same SparseTensor instance the caller registered).</param>
    /// <param name="b">Sparse right operand (same identity rule).</param>
    public static Tensor<T> SparsePatternPreservingSpGeMMRecord<T>(
        SparseTensor<T> a, SparseTensor<T> b)
    {
        if (a is null) throw new ArgumentNullException(nameof(a));
        if (b is null) throw new ArgumentNullException(nameof(b));
        var sparseOut = SparseOps.SparseSpGeMM(a, b);
        var output = sparseOut.ToDense();
        if (DifferentiableOps._anyTapeActive == 0) return output;

        // Snapshot patterns AND values for both operands at forward time.
        // The sparse-aware backward needs B's values to compute dA and
        // A's values to compute dB. Snapshot prevents in-place weight
        // updates between forward + backward from corrupting gradients.
        // ToArray() already allocates a fresh copy — the previous
        // ToArray().Clone() was a redundant second copy.
        var aCoo = a.ToCoo();
        var bCoo = b.ToCoo();
        var aValuesSnapshot = aCoo.DataVector.ToArray();
        var bValuesSnapshot = bCoo.DataVector.ToArray();

        DifferentiableOps.RecordBinary(
            "SparsePatternPreservingSpGeMM",
            output,
            a,
            b,
            SparsePatternPreservingSpGeMMBackward,
            savedState: new object[]
            {
                aCoo.RowIndices, aCoo.ColumnIndices, a.Rows, a.Columns, aValuesSnapshot,
                bCoo.RowIndices, bCoo.ColumnIndices, b.Rows, b.Columns, bValuesSnapshot,
            });
        return output;
    }

    private static void SparsePatternPreservingSpGeMMBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        // Y_dense = A_sparse · B_sparse  (then densified for tape composition)
        // dA at non-zero (i,j) ∈ pattern_A:
        //     dA[i,j] = sum_k B[j,k] · gradOut[i,k]
        //   where B[j,k] is read via the sparse indexer (zero for
        //   structural zeros, snapshot value for non-zero positions).
        //   Equivalently: gather only over the k positions where (j,k) is
        //   in pattern_B, then sum sparse_B(j,k) · gradOut(i,k).
        // dB at non-zero (j,k) ∈ pattern_B:
        //     dB[j,k] = sum_i A[i,j] · gradOut[i,k]
        var aSparse = (SparseTensor<T>)inputs[0];
        var bSparse = (SparseTensor<T>)inputs[1];
        var aRowIdx = (int[])savedState[0];
        var aColIdx = (int[])savedState[1];
        int aRows = (int)savedState[2];
        int aCols = (int)savedState[3];
        var aValues = (T[])savedState[4];
        var bRowIdx = (int[])savedState[5];
        var bColIdx = (int[])savedState[6];
        int bRows = (int)savedState[7];
        int bCols = (int)savedState[8];
        var bValues = (T[])savedState[9];

        var ops = MathHelper.GetNumericOperations<T>();

        // Build a B-row → list-of-(col, value) lookup so dA's per-(i,j)
        // sum only scans the actual non-zeros in row j of B (not the
        // structural zeros). O(nnz_B) total work to build.
        var bByRow = new System.Collections.Generic.List<(int col, T val)>[bRows];
        for (int idx = 0; idx < bRowIdx.Length; idx++)
        {
            int row = bRowIdx[idx];
            (bByRow[row] ??= new System.Collections.Generic.List<(int, T)>())
                .Add((bColIdx[idx], bValues[idx]));
        }

        int nnzA = aRowIdx.Length;
        var gradAValues = new T[nnzA];
        for (int idx = 0; idx < nnzA; idx++)
        {
            int i = aRowIdx[idx];
            int j = aColIdx[idx];
            T sum = ops.Zero;
            var bRow = bByRow[j];
            if (bRow is not null)
            {
                foreach (var (k, bVal) in bRow)
                {
                    sum = ops.Add(sum, ops.Multiply(bVal, gradOutput[i, k]));
                }
            }
            gradAValues[idx] = sum;
        }
        var sparseGradA = new SparseTensor<T>(aRows, aCols, aRowIdx, aColIdx, gradAValues);
        AccumulateGrad(aSparse, sparseGradA, gradAccumulator, engine);

        // dB symmetric: build A-col → list-of-(row, value).
        var aByCol = new System.Collections.Generic.List<(int row, T val)>[aCols];
        for (int idx = 0; idx < aRowIdx.Length; idx++)
        {
            int col = aColIdx[idx];
            (aByCol[col] ??= new System.Collections.Generic.List<(int, T)>())
                .Add((aRowIdx[idx], aValues[idx]));
        }

        int nnzB = bRowIdx.Length;
        var gradBValues = new T[nnzB];
        for (int idx = 0; idx < nnzB; idx++)
        {
            int j = bRowIdx[idx];
            int k = bColIdx[idx];
            T sum = ops.Zero;
            var aCol = aByCol[j];
            if (aCol is not null)
            {
                foreach (var (i, aVal) in aCol)
                {
                    sum = ops.Add(sum, ops.Multiply(aVal, gradOutput[i, k]));
                }
            }
            gradBValues[idx] = sum;
        }
        var sparseGradB = new SparseTensor<T>(bRows, bCols, bRowIdx, bColIdx, gradBValues);
        AccumulateGrad(bSparse, sparseGradB, gradAccumulator, engine);
        _ = output;
    }

    public static Tensor<T> SparseSpGeMMRecord<T>(SparseTensor<T> a, SparseTensor<T> b,
        Tensor<T> aDense, Tensor<T> bDense)
    {
        if (aDense is null) throw new ArgumentNullException(nameof(aDense));
        if (bDense is null) throw new ArgumentNullException(nameof(bDense));
        var sparseOut = SparseOps.SparseSpGeMM(a, b);
        var output = sparseOut.ToDense();
        if (DifferentiableOps._anyTapeActive == 0) return output;

        // Snapshot both dense views at forward time. Without these,
        // backward reads inputs[0]/inputs[1] which still point at the
        // caller's tensors — if those are mutated between forward and
        // backward (a real pattern in optimisers that modify weights
        // in-place between tape recording and ComputeGradients), the
        // matmul transposes use the wrong values and gradients are
        // wrong. Matches the snapshot pattern used by SparseMatMul.
        var aSnapshot = aDense.Clone();
        var bSnapshot = bDense.Clone();
        DifferentiableOps.RecordBinary(
            "SparseSpGeMM",
            output,
            aDense,
            bDense,
            SpGeMMBackward<T>,
            savedState: new object[] { aSnapshot, bSnapshot });
        return output;
    }

    private static void SparseSumBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        var aDense = inputs[0];
        int axis = (int)savedState[0];
        var ops = MathHelper.GetNumericOperations<T>();
        // Broadcast grad_output back over the reduction axis.
        var gradA = new Tensor<T>((int[])aDense._shape.Clone());
        var gradSpan = gradA.AsWritableSpan();
        var gradOutSpan = gradOutput.AsSpan();
        int rows = aDense._shape[0], cols = aDense._shape[1];
        if (axis < 0)
        {
            T fill = gradOutSpan[0];
            for (int i = 0; i < gradSpan.Length; i++) gradSpan[i] = fill;
        }
        else if (axis == 0)
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    gradSpan[r * cols + c] = gradOutSpan[c];
        }
        else // axis == 1
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    gradSpan[r * cols + c] = gradOutSpan[r];
        }
        AccumulateGrad(aDense, gradA, gradAccumulator, engine);
        _ = output; _ = ops;
    }

    private static void SparseMeanBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        var aDense = inputs[0];
        int axis = (int)savedState[0];
        int rows = (int)savedState[1], cols = (int)savedState[2];
        var ops = MathHelper.GetNumericOperations<T>();
        int divisor = axis < 0 ? rows * cols : axis == 0 ? rows : cols;
        T scale = ops.Divide(ops.One, ops.FromDouble(divisor));
        var scaledGrad = engine.TensorMultiplyScalar(gradOutput, scale);
        // Reuse the sum-broadcast logic.
        var fakeSum = new Tensor<T>[] { aDense };
        SparseSumBackward<T>(scaledGrad, fakeSum, output, savedState, engine, gradAccumulator);
    }

    private static void SparseSoftmaxBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        var aDense = inputs[0];
        var softmax = (SparseTensor<T>)savedState[0];
        bool takeLog = (bool)savedState[1];
        var ops = MathHelper.GetNumericOperations<T>();

        // Per-row softmax Jacobian on the stored pattern only.
        var csr = softmax.Format == SparseStorageFormat.Csr ? softmax : softmax.ToCsr();
        var rowPtr = csr.RowPointers;
        var colIdx = csr.ColumnIndices;
        var smVals = csr.DataVector;
        var gradOutSpan = gradOutput.AsSpan();
        int n = aDense._shape[1];
        var gradA = new Tensor<T>((int[])aDense._shape.Clone());
        var gradASpan = gradA.AsWritableSpan();

        for (int r = 0; r < aDense._shape[0]; r++)
        {
            int rs = rowPtr[r], re = rowPtr[r + 1];
            if (re == rs) continue;
            T weighted = ops.Zero;
            for (int p = rs; p < re; p++)
            {
                int col = colIdx[p];
                if (takeLog)
                    weighted = ops.Add(weighted, gradOutSpan[r * n + col]);
                else
                    weighted = ops.Add(weighted, ops.Multiply(gradOutSpan[r * n + col], smVals[p]));
            }
            for (int p = rs; p < re; p++)
            {
                int col = colIdx[p];
                T dy = gradOutSpan[r * n + col];
                T y = smVals[p];
                T dx = takeLog
                    ? ops.Subtract(dy, ops.Multiply(weighted, y))
                    : ops.Multiply(y, ops.Subtract(dy, weighted));
                gradASpan[r * n + col] = dx;
            }
        }
        AccumulateGrad(aDense, gradA, gradAccumulator, engine);
        _ = output;
    }

    private static void SpGeMMBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        // dA = grad · B^T,  dB = A^T · grad  — same Jacobian as dense matmul.
        // inputs[0]/inputs[1] are the caller-visible gradient targets;
        // savedState carries the forward-time snapshots so the matmul
        // transposes use the values that produced this output even if
        // the caller mutated their tensors in-place between forward
        // and backward.
        var aDense = inputs[0];      // caller-visible gradient target
        var bDense = inputs[1];      // caller-visible gradient target
        var aSnapshot = (Tensor<T>)savedState[0];
        var bSnapshot = (Tensor<T>)savedState[1];

        var bT = engine.TensorTranspose(bSnapshot);
        var gradA = engine.TensorMatMul(gradOutput, bT);
        AccumulateGrad(aDense, gradA, gradAccumulator, engine);

        var aT = engine.TensorTranspose(aSnapshot);
        var gradB = engine.TensorMatMul(aT, gradOutput);
        AccumulateGrad(bDense, gradB, gradAccumulator, engine);
        _ = output;
    }

    private static void AccumulateGrad<T>(
        Tensor<T> target,
        Tensor<T> grad,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator,
        IEngine engine)
    {
        if (gradAccumulator.TryGetValue(target, out var existing))
            gradAccumulator[target] = engine.TensorAdd(existing, grad);
        else
            gradAccumulator[target] = grad;
    }
}
