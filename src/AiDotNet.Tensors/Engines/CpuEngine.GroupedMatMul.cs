using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The native jagged-rows grouped GEMM behind <see cref="CpuEngine.TensorGroupedMatMul{T}"/> (torch grouped_mm with a
/// 2-D <c>a</c> and a 3-D <c>b</c>, with or without a bias): one tape node instead of a narrow, a matmul, a bias add and a
/// concatenation per group.
/// </summary>
/// <remarks>
/// <para>
/// Rows <c>[ends[g-1], ends[g])</c> of <c>a</c> <c>[M, K]</c> are multiplied by <c>b[g]</c> (<c>[G, K, N]</c>, or
/// <c>[G, N, K]</c> read transposed), each group at its own row count, so an unbalanced mixture-of-experts router costs
/// exactly its routed rows. A bias (<c>[N]</c>, or <c>[G, N]</c> per group) is added to every routed row. Rows past the
/// last offset are zero, as torch defines them. The backward is the same op on the gradient
/// (<c>da_g = dout_g · op(b_g)^T</c>), the per-group outer product <c>db_g = a_g^T · dout_g</c> (zero for an empty
/// group) and the per-group column sum of <c>dout</c> for the bias.
/// </para>
/// <para>
/// Every engine runs it natively: <see cref="DirectGpuTensorEngine"/> overrides <see cref="GroupedRowsMatMul{T}"/> and
/// <see cref="GroupedRowsWeightGrad{T}"/> with the backend's <see cref="DirectGpu.IGroupedGemmBackend"/>, which all six
/// GPU backends implement (CUDA and HIP with vendor BLAS on buffer views, OpenCL, Vulkan, Metal and WebGPU with
/// <see cref="DirectGpu.GroupedGemmKernelTemplate"/>'s kernels).
/// </para>
/// </remarks>
public partial class CpuEngine
{
    /// <summary>
    /// Jagged-rows grouped matmul: [M, K] x [G, K, N] (or [G, N, K] with <paramref name="transposeB"/>) -> [M, N], plus
    /// <paramref name="bias"/> ([N] or [G, N]) on every routed row.
    /// </summary>
    internal virtual Tensor<T> GroupedRowsMatMul<T>(Tensor<T> a, Tensor<T> b, int[] ends, bool transposeB, Tensor<T>? bias = null)
    {
        var (g, m, k, n) = GroupedRowsShape(a, b, ends, transposeB);
        if (bias is not null) ValidateGroupBias(bias, g, n);
        if (GraphMode.IsActive && GraphMode.Current is { } scope && scope.BindEngine(this))
        {
            var ca = a; var cb = b; var cbias = bias;
            Action<IEngine, Tensor<T>> execute = (eng, result) =>
            {
                var host = eng.As<CpuEngine>() ?? new CpuEngine();
                DirectGpuTensorEngine.CopyResultInto(eng, host.GroupedRowsMatMul(ca, cb, ends, transposeB, cbias), result);
            };
            var state = new object[] { ends, transposeB };
            return bias is null
                ? scope.RecordBinary(LazyNodeType.Custom, "GroupedMatMul", a, b, new[] { m, n }, execute, GroupedRowsMatMulBackward<T>, state)
                : scope.RecordVariadic(LazyNodeType.Custom, "GroupedMatMul", new[] { a, b, bias }, new[] { m, n }, execute,
                    GroupedRowsMatMulBackward<T>, state);
        }

        var output = new Tensor<T>(new[] { m, n });   // zero-initialized: rows past the last offset stay zero
        if (output.Length > 0)
        {
            var y = output.GetDataArray();
            if (k > 0)
            {
                var x = HostArray(a, out int aOff);
                var w = HostArray(b, out int bOff);
                GroupedRowsHost(x, aOff, w, bOff, y, ends, k, n, transposeB);
            }
            if (bias is not null)
            {
                var bv = HostArray(bias, out int biasOff);
                GroupedBiasHost(bv, biasOff, bias.Rank == 2, y, ends, n);
            }
        }
        RecordGroupedRows(output, a, b, bias, ends, transposeB);
        return output;
    }

    /// <summary>Records the native op's tape node (two inputs, or three with a bias).</summary>
    private protected static void RecordGroupedRows<T>(Tensor<T> output, Tensor<T> a, Tensor<T> b, Tensor<T>? bias, int[] ends, bool transposeB)
    {
        var state = new object[] { ends, transposeB };
        if (bias is null) DifferentiableOps.RecordBinary("GroupedMatMul", output, a, b, GroupedRowsMatMulBackward<T>, state);
        else DifferentiableOps.RecordIfActive("GroupedMatMul", output, new[] { a, b, bias }, GroupedRowsMatMulBackward<T>, state);
    }

    /// <summary>
    /// Per-group outer product <c>w[g] = a_g^T · c_g</c>: a [M, K], c [M, N] -> [G, K, N]. Recorded on an active tape
    /// (the backward of the grouped matmul calls it, so a createGraph backward stays differentiable).
    /// </summary>
    internal virtual Tensor<T> GroupedRowsWeightGrad<T>(Tensor<T> a, Tensor<T> c, int[] ends)
    {
        int g = ends.Length, k = a._shape[1], n = c._shape[1];
        var output = new Tensor<T>(new[] { g, k, n });   // zero-initialized: an empty group's gradient is zero
        if (output.Length > 0 && a._shape[0] > 0)
        {
            var x = HostArray(a, out int aOff);
            var y = HostArray(c, out int cOff);
            GroupedWeightGradHost(x, aOff, y, cOff, output.GetDataArray(), ends, k, n);
        }
        RecordGroupedWeightGrad(output, a, c, ends);
        return output;
    }

    /// <summary>Records the per-group outer product's tape node.</summary>
    private protected static void RecordGroupedWeightGrad<T>(Tensor<T> output, Tensor<T> a, Tensor<T> c, int[] ends) =>
        DifferentiableOps.RecordBinary("GroupedRowsWeightGrad", output, a, c, GroupedRowsWeightGradBackward<T>, new object[] { ends });

    /// <summary>
    /// Backward of <see cref="GroupedRowsWeightGrad{T}"/> (<c>w_g = a_g^T c_g</c>): <c>da_g = c_g dw_g^T</c>,
    /// <c>dc_g = a_g dw_g</c>; rows past the last offset get zero. Both are grouped matmuls, so they record in turn.
    /// </summary>
    internal static void GroupedRowsWeightGradBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var ends = (int[])savedState[0];
        var grouped = engine.As<CpuEngine>() ?? new CpuEngine();
        if (DifferentiableOps.IsGradientRequired(inputs[0]))
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grouped.GroupedRowsMatMul(inputs[1], gradOutput, ends, true), engine);
        if (DifferentiableOps.IsGradientRequired(inputs[1]))
            DifferentiableOps.AccumulateGrad(grads, inputs[1], grouped.GroupedRowsMatMul(inputs[0], gradOutput, ends, false), engine);
    }

    /// <summary>Validates the jagged-rows form and returns (G, M, K, N).</summary>
    internal static (int G, int M, int K, int N) GroupedRowsShape<T>(Tensor<T> a, Tensor<T> b, int[] ends, bool transposeB)
    {
        if (a.Rank != 2 || b.Rank != 3) throw new ArgumentException("Jagged-rows grouped matmul takes a [M, K] and b [G, K, N].");
        int g = b._shape[0], m = a._shape[0], k = a._shape[1];
        int bk = transposeB ? b._shape[2] : b._shape[1], n = transposeB ? b._shape[1] : b._shape[2];
        if (bk != k || ends.Length != g)
            throw new ArgumentException($"Expected b of shape [{ends.Length}, {k}, N]; got [{string.Join(", ", b._shape)}]" + (transposeB ? " (transposed)." : "."), nameof(b));
        int prev = 0;
        foreach (int e in ends)
        {
            if (e < prev) throw new ArgumentException("offsets must be non-decreasing and non-negative.", "offsets");
            prev = e;
        }
        if (prev > m) throw new ArgumentException("The last offset passes the rows of a.", "offsets");
        return (g, m, k, n);
    }

    /// <summary>
    /// Backward of <see cref="GroupedRowsMatMul{T}"/>. Without transposeB (<c>out_g = a_g b_g</c>): <c>da = grouped(dout, b, T)</c>,
    /// <c>db_g = a_g^T dout_g</c>. With it (<c>out_g = a_g b_g^T</c>): <c>da = grouped(dout, b)</c>, <c>db_g = dout_g^T a_g</c>.
    /// The bias gets the per-group column sums of <c>dout</c> (summed over the groups for a shared [N] bias). Rows past the
    /// last offset get a zero gradient.
    /// </summary>
    internal static void GroupedRowsMatMulBackward<T>(
        Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var ends = (int[])savedState[0];
        bool transposeB = (bool)savedState[1];
        var grouped = engine.As<CpuEngine>() ?? new CpuEngine();
        if (DifferentiableOps.IsGradientRequired(inputs[0]))
            DifferentiableOps.AccumulateGrad(grads, inputs[0], grouped.GroupedRowsMatMul(gradOutput, inputs[1], ends, !transposeB), engine);
        if (DifferentiableOps.IsGradientRequired(inputs[1]))
            DifferentiableOps.AccumulateGrad(grads, inputs[1],
                transposeB
                    ? grouped.GroupedRowsWeightGrad(gradOutput, inputs[0], ends)
                    : grouped.GroupedRowsWeightGrad(inputs[0], gradOutput, ends),
                engine);
        if (inputs.Length > 2 && DifferentiableOps.IsGradientRequired(inputs[2]))
        {
            // Column sums per group: the weight gradient of a ones column, [G, 1, N].
            var ops = MathHelper.GetNumericOperations<T>();
            int m = gradOutput._shape[0], n = gradOutput._shape[1];
            var ones = new Tensor<T>(new[] { m, 1 });
            ones.Fill(ops.One);
            var perGroup = engine.Reshape(grouped.GroupedRowsWeightGrad(ones, gradOutput, ends), new[] { ends.Length, n });
            var biasGrad = inputs[2].Rank == 2 ? perGroup : engine.ReduceSum(perGroup, new[] { 0 }, false);
            DifferentiableOps.AccumulateGrad(grads, inputs[2], biasGrad, engine);
        }
    }

    // out rows of group i = a rows · b[i] (or b[i]^T), each through the managed GEMM at its exact size.
    private static void GroupedRowsHost<T>(T[] a, int aOff, T[] b, int bOff, T[] y, int[] ends, int k, int n, bool transposeB)
    {
        int start = 0, ldb = transposeB ? k : n;
        for (int i = 0; i < ends.Length; i++)
        {
            int rows = ends[i] - start;
            if (rows > 0)
            {
                int ao = aOff + start * k, bo = bOff + i * k * n, yo = start * n;
                if (a is float[] aF && b is float[] bF && y is float[] yF)
                    BlasManaged.BlasManaged.Gemm<float>(aF.AsSpan(ao, rows * k), k, false, bF.AsSpan(bo, k * n), ldb, transposeB,
                        yF.AsSpan(yo, rows * n), n, rows, n, k);
                else if (a is double[] aD && b is double[] bD && y is double[] yD)
                    BlasManaged.BlasManaged.Gemm<double>(aD.AsSpan(ao, rows * k), k, false, bD.AsSpan(bo, k * n), ldb, transposeB,
                        yD.AsSpan(yo, rows * n), n, rows, n, k);
                else
                    GroupedBlockGeneric(a, ao, b, bo, y, yo, rows, k, n, transposeB);
            }
            start = ends[i];
        }
    }

    // bias ([N], or [G, N] with perGroup) added to every routed row; rows past the last offset stay zero.
    private static void GroupedBiasHost<T>(T[] bias, int biasOff, bool perGroup, T[] y, int[] ends, int n)
    {
        int start = 0;
        for (int i = 0; i < ends.Length; i++)
        {
            int rows = ends[i] - start, bo = biasOff + (perGroup ? i * n : 0);
            if (rows > 0)
            {
                if (y is float[] yF && bias is float[] bF)
                {
                    var row = bF.AsSpan(bo, n);
                    for (int r = start; r < ends[i]; r++)
                        Simd.SimdKernels.VectorAdd(yF.AsSpan(r * n, n), row, yF.AsSpan(r * n, n));
                }
                else if (y is double[] yD && bias is double[] bD)
                {
                    var row = bD.AsSpan(bo, n);
                    for (int r = start; r < ends[i]; r++)
                        Simd.SimdKernels.VectorAdd(yD.AsSpan(r * n, n), row, yD.AsSpan(r * n, n));
                }
                else
                    GroupedBiasGeneric(bias, bo, y, start, ends[i], n);
            }
            start = ends[i];
        }
    }

    // w[i] = a_i^T · c_i for each group's rows; empty groups keep the zero-initialized block.
    private static void GroupedWeightGradHost<T>(T[] a, int aOff, T[] c, int cOff, T[] w, int[] ends, int k, int n)
    {
        int start = 0;
        for (int i = 0; i < ends.Length; i++)
        {
            int rows = ends[i] - start;
            if (rows > 0)
            {
                int ao = aOff + start * k, co = cOff + start * n, wo = i * k * n;
                if (a is float[] aF && c is float[] cF && w is float[] wF)
                    BlasManaged.BlasManaged.Gemm<float>(aF.AsSpan(ao, rows * k), k, true, cF.AsSpan(co, rows * n), n, false,
                        wF.AsSpan(wo, k * n), n, k, n, rows);
                else if (a is double[] aD && c is double[] cD && w is double[] wD)
                    BlasManaged.BlasManaged.Gemm<double>(aD.AsSpan(ao, rows * k), k, true, cD.AsSpan(co, rows * n), n, false,
                        wD.AsSpan(wo, k * n), n, k, n, rows);
                else
                    GroupedWeightBlockGeneric(a, ao, c, co, w, wo, rows, k, n);
            }
            start = ends[i];
        }
    }

    [ScalarKernel("Element types other than float and double have no typed GEMM; float and double take BlasManaged above.")]
    private static void GroupedBlockGeneric<T>(T[] a, int ao, T[] b, int bo, T[] y, int yo, int rows, int k, int n, bool transposeB)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < n; j++)
            {
                T sum = ops.Zero;
                for (int q = 0; q < k; q++)
                    sum = ops.Add(sum, ops.Multiply(a[ao + r * k + q], b[bo + (transposeB ? j * k + q : q * n + j)]));
                y[yo + r * n + j] = sum;
            }
    }

    [ScalarKernel("Element types other than float and double have no typed add; float and double take SimdKernels above.")]
    private static void GroupedBiasGeneric<T>(T[] bias, int bo, T[] y, int rowStart, int rowEnd, int n)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int r = rowStart; r < rowEnd; r++)
            for (int j = 0; j < n; j++)
                y[r * n + j] = ops.Add(y[r * n + j], bias[bo + j]);
    }

    [ScalarKernel("Element types other than float and double have no typed GEMM; float and double take BlasManaged above.")]
    private static void GroupedWeightBlockGeneric<T>(T[] a, int ao, T[] c, int co, T[] w, int wo, int rows, int k, int n)
    {
        var ops = MathHelper.GetNumericOperations<T>();
        for (int p = 0; p < k; p++)
            for (int j = 0; j < n; j++)
            {
                T sum = ops.Zero;
                for (int r = 0; r < rows; r++)
                    sum = ops.Add(sum, ops.Multiply(a[ao + r * k + p], c[co + r * n + j]));
                w[wo + p * n + j] = sum;
            }
    }
}
