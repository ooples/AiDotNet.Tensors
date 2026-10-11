using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// Algebraic simplification rules for the backward pass:
/// 1. Double-transpose elimination: A^T^T = A (skip redundant transpose)
/// 2. Transposed GEMM: use TryGemmEx(transA/transB) instead of materializing transposes
/// 3. Associative regrouping: choose (A^T @ B) @ C vs A^T @ (B @ C) based on dimension analysis
///
/// These are applied at compile time when building OptimizedBackwardPlan.
/// </summary>
internal static class AlgebraicSimplificationPass
{
    /// <summary>
    /// Optimized backward for a linear layer: computes gradInput, gradWeight, gradBias
    /// using transposed BLAS GEMM and cached transposes.
    ///
    /// For y = x @ W + b:
    ///   dL/dW = x^T @ gradOutput     (transposed GEMM, no transpose alloc)
    ///   dL/dx = gradOutput @ W^T      (transposed GEMM, no transpose alloc)
    ///   dL/db = sum(gradOutput, axis=0)  (inline reduction)
    ///
    /// Uses BackwardCSEPass to cache transposes across layers.
    /// </summary>
    internal static unsafe void OptimizedLinearBackward<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> weight,
        Tensor<T>? bias,
        BackwardCSEPass<T> cse,
        IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        // dL/dx = gradOutput @ W^T — uses transposed BLAS or cached transpose
        var gradInput = cse.MatMulTransposeRight(gradOutput, weight);
        DifferentiableOps.AccumulateGrad(grads, input, gradInput, engine);

        // dL/dW = x^T @ gradOutput — uses transposed BLAS or cached transpose
        var gradWeight = cse.MatMulTransposeLeft(input, gradOutput);
        DifferentiableOps.AccumulateGrad(grads, weight, gradWeight, engine);

        // dL/db = sum(gradOutput, axis=0)
        if (bias != null && typeof(T) == typeof(float) && gradOutput.Rank == 2)
        {
            int m = gradOutput._shape[0];
            int n = gradOutput._shape[1];
            var biasGradArr = new float[n];
            var gArr = (float[])(object)gradOutput.GetDataArray();
            fixed (float* pG = gArr, pB = biasGradArr)
            {
                for (int row = 0; row < m; row++)
                {
                    float* gRow = pG + row * n;
                    for (int j = 0; j < n; j++)
                        pB[j] += gRow[j];
                }
            }
            var biasGrad = new Tensor<T>((T[])(object)biasGradArr, bias._shape);
            DifferentiableOps.AccumulateGrad(grads, bias, biasGrad, engine);
        }
        else if (bias != null)
        {
            var biasGrad = engine.ReduceSum(gradOutput, new[] { 0 }, keepDims: false);
            DifferentiableOps.AccumulateGrad(grads, bias, biasGrad, engine);
        }
    }

    /// <summary>
    /// Estimates the FLOP cost of a matmul with given dimensions.
    /// Used for associative regrouping decisions.
    /// </summary>
    internal static long EstimateMatMulFlops(int m, int k, int n)
    {
        return 2L * m * k * n; // 2 * M * K * N (multiply + add per element)
    }

    /// <summary>
    /// For (A @ B) @ C vs A @ (B @ C), choose the grouping with fewer total FLOPs.
    /// Returns true if left-grouping (A@B)@C is cheaper.
    /// </summary>
    internal static bool ShouldGroupLeft(int m, int k1, int k2, int n)
    {
        // (A[m,k1] @ B[k1,k2]) @ C[k2,n]: cost = 2*m*k1*k2 + 2*m*k2*n
        long leftCost = EstimateMatMulFlops(m, k1, k2) + EstimateMatMulFlops(m, k2, n);

        // A[m,k1] @ (B[k1,k2] @ C[k2,n]): cost = 2*k1*k2*n + 2*m*k1*n
        long rightCost = EstimateMatMulFlops(k1, k2, n) + EstimateMatMulFlops(m, k1, n);

        return leftCost <= rightCost;
    }
}
