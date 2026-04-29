// Copyright (c) AiDotNet. All rights reserved.

using System;
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
    /// dense gradients for both sides.</summary>
    public static Tensor<T> SparseMatMulRecord<T>(SparseTensor<T> a, Tensor<T> b)
    {
        var output = SparseOps.SparseMatMul(a, b);
        if (DifferentiableOps._anyTapeActive == 0) return output;

        // Capture A as a dense reference so backward can compute A^T · grad
        // via the standard dense matmul path. This matches PyTorch's
        // sparse_mm_backward, which densifies A before transposing.
        var aDense = a.ToDense();
        DifferentiableOps.RecordBinary(
            "SparseMatMul",
            output,
            aDense,
            b,
            SparseMatMulBackward,
            savedState: null);
        return output;
    }

    /// <summary>Tape-aware <c>α · (A_sparse · B) + β · C</c>.</summary>
    public static Tensor<T> SparseAddMMRecord<T>(Tensor<T> c, SparseTensor<T> a, Tensor<T> b, T alpha, T beta)
    {
        var output = SparseOps.SparseAddMM(c, a, b, alpha, beta);
        if (DifferentiableOps._anyTapeActive == 0) return output;

        var aDense = a.ToDense();
        var savedState = new object[] { alpha!, beta! };
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

    private static void SparseMatMulBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T>[] inputs,
        Tensor<T> output,
        object[] savedState,
        IEngine engine,
        System.Collections.Generic.Dictionary<Tensor<T>, Tensor<T>> gradAccumulator)
    {
        // Y = A · B (A dense-projected from sparse, B dense)
        // dA = grad_Y · B^T
        // dB = A^T · grad_Y
        var aDense = inputs[0];
        var b = inputs[1];

        var bT = engine.TensorTranspose(b);
        var gradA = engine.TensorMatMul(gradOutput, bT);
        AccumulateGrad(aDense, gradA, gradAccumulator, engine);

        var aT = engine.TensorTranspose(aDense);
        var gradB = engine.TensorMatMul(aT, gradOutput);
        AccumulateGrad(b, gradB, gradAccumulator, engine);
        _ = output; _ = savedState;
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
        var c = inputs[0];
        var aDense = inputs[1];
        var b = inputs[2];

        var gradC = engine.TensorMultiplyScalar(gradOutput, beta);
        AccumulateGrad(c, gradC, gradAccumulator, engine);

        var alphaGradOut = engine.TensorMultiplyScalar(gradOutput, alpha);
        var bT = engine.TensorTranspose(b);
        var gradA = engine.TensorMatMul(alphaGradOut, bT);
        AccumulateGrad(aDense, gradA, gradAccumulator, engine);

        var aT = engine.TensorTranspose(aDense);
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
