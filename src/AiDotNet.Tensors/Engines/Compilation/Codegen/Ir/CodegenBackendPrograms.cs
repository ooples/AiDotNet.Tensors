// Copyright (c) AiDotNet. All rights reserved.
// The declared normalisation programs in the EXACT contract of the hand-written backend kernels they replace
// (#890 phase 4a), so a backend can route an existing call to them with no change to its callers.
//
// The programs keep their own internal statistics (a negated mean, rstd); these adapters add the row passes that
// produce what the hand-written kernels save and read:
//
//   LayerNorm / GroupNorm / BatchNorm   saveMean = mean, saveInvVar = rstd
//   RmsNorm                             saveRms  = rms = 1 / rstd
//   BatchNorm (training)                runningMean += momentum * (mean - runningMean),
//                                       runningVar  += momentum * (unbiased var - runningVar)
//
// and, in front of each backward program, the passes that turn the saved statistics back into what it reads. A row
// pass inside a fused kernel is a few instructions per row, so the contract costs nothing measurable.
//
// Every tensor a contract names is one the backend binds to a caller's buffer; any other name in the program is an
// intermediate the backend owns.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>Declared programs shaped as the hand-written backend normalisation kernels.</summary>
public static class CodegenBackendPrograms
{
    private static CodegenExpr Op(int input) => CodegenExpr.Operand(input);

    private static CodegenTensorBinding Bind(int parameter, string name, int[] shape, bool isOutput = false)
    {
        var map = new CodegenAffineExpr[shape.Length];
        for (int d = 0; d < map.Length; d++) map[d] = CodegenAffineExpr.Axis(d);
        return new CodegenTensorBinding(parameter, name, (int[])shape.Clone(), map, isOutput);
    }

    private static CodegenIterationSpace RowSpace(int[] shape)
    {
        var axes = new CodegenAxis[shape.Length];
        for (int d = 0; d < axes.Length; d++) axes[d] = CodegenAxis.Parallel("r" + d, shape[d]);
        return new CodegenIterationSpace(axes);
    }

    /// <summary>A per-row pass computing <paramref name="body"/> over the named per-row inputs.</summary>
    private static CodegenKernelSpec RowPass(string name, int[] shape, string[] inputs, string output, CodegenExpr body,
        CodegenActivationKind activation = CodegenActivationKind.None)
    {
        var bindings = new CodegenTensorBinding[inputs.Length];
        for (int i = 0; i < inputs.Length; i++) bindings[i] = Bind(i, inputs[i], shape);
        return new CodegenKernelSpec(name, RowSpace(shape), bindings,
            Bind(inputs.Length, output, shape, isOutput: true), body.Operands(), CodegenReduceKind.None,
            activation: activation, body: body);
    }

    private static CodegenProgram Append(CodegenProgram program, params CodegenKernelSpec[] extra)
    {
        var passes = new List<CodegenKernelSpec>(program.Passes);
        var temps = new List<long>(program.TempElements);
        foreach (var pass in extra) { passes.Add(pass); temps.Add(pass.Output.ElementCount); }
        return new CodegenProgram(passes, temps, program.Description + " (backend contract)");
    }

    private static CodegenProgram Prepend(CodegenProgram program, params CodegenKernelSpec[] first)
    {
        var passes = new List<CodegenKernelSpec>(first);
        var temps = new List<long>();
        foreach (var pass in first) temps.Add(pass.Output.ElementCount);
        passes.AddRange(program.Passes);
        temps.AddRange(program.TempElements);
        return new CodegenProgram(passes, temps, program.Description + " (backend contract)");
    }

    /// <summary>
    /// CudaBackend.LayerNorm: sources x, gamma, beta; outputs y, mean (saveMean), rstd (saveInvVar).
    /// </summary>
    // ---- linear layers (#890 phase 4c), in CudaBackend.FusedLinear*'s contract: x [batch, in], W [in, out] -------

    /// <summary><c>y[b, j] = act(sum_k x[b, k] W[k, j] + bias[j])</c>: one contraction with a bias and activation epilogue.</summary>
    public static CodegenProgram FusedLinear(int batch, int inFeatures, int outFeatures, CodegenActivationKind activation)
    {
        var space = new CodegenIterationSpace(
            CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("j", outFeatures), CodegenAxis.Reduce("k", inFeatures));
        const int B = 0, J = 1, K = 2;
        var x = new CodegenTensorBinding(0, "x", new[] { batch, inFeatures }, new[] { A(B), A(K) });
        var w = new CodegenTensorBinding(1, "w", new[] { inFeatures, outFeatures }, new[] { A(K), A(J) });
        var bias = new CodegenTensorBinding(2, "bias", new[] { outFeatures }, new[] { A(J) });
        var y = new CodegenTensorBinding(3, "y", new[] { batch, outFeatures }, new[] { A(B), A(J) }, isOutput: true);
        var pass = new CodegenKernelSpec("fused_linear", space, new[] { x, w, bias }, y, new[] { 0, 1 },
            CodegenReduceKind.Sum, biasInput: 2, activation: activation);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "fused linear (backend contract)");
    }

    /// <summary>
    /// The backward of <see cref="FusedLinear"/>: <c>dpre = derivative(dy, saved)</c> -- the activation's backward,
    /// <paramref name="derivative"/> called with the gradient and the saved pre-activation or output -- then
    /// <c>dx = dpre W^T</c>, <c>dW = x^T dpre</c> and <c>dbias = sum_b dpre</c>, each overwriting its gradient.
    /// </summary>
    public static CodegenProgram FusedLinearBackward(int batch, int inFeatures, int outFeatures,
        CodegenLibraryFunction derivative)
    {
        if (derivative is null) throw new ArgumentNullException(nameof(derivative));
        var activationGrad = RowPass("fused_linear_dpre", new[] { batch, outFeatures }, new[] { "dy", "saved" }, "dpre",
            CodegenExpr.Call(derivative, Op(0), Op(1)));

        var dxSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("b", batch), CodegenAxis.Parallel("i", inFeatures), CodegenAxis.Reduce("j", outFeatures));
        var dx = new CodegenKernelSpec("fused_linear_dx", dxSpace,
            new[]
            {
                new CodegenTensorBinding(0, "dpre", new[] { batch, outFeatures }, new[] { A(0), A(2) }),
                new CodegenTensorBinding(1, "w", new[] { inFeatures, outFeatures }, new[] { A(1), A(2) }),
            },
            new CodegenTensorBinding(2, "dx", new[] { batch, inFeatures }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        var dwSpace = new CodegenIterationSpace(
            CodegenAxis.Parallel("i", inFeatures), CodegenAxis.Parallel("j", outFeatures), CodegenAxis.Reduce("b", batch));
        var dw = new CodegenKernelSpec("fused_linear_dw", dwSpace,
            new[]
            {
                new CodegenTensorBinding(0, "x", new[] { batch, inFeatures }, new[] { A(2), A(0) }),
                new CodegenTensorBinding(1, "dpre", new[] { batch, outFeatures }, new[] { A(2), A(1) }),
            },
            new CodegenTensorBinding(2, "dw", new[] { inFeatures, outFeatures }, new[] { A(0), A(1) }, isOutput: true),
            new[] { 0, 1 }, CodegenReduceKind.Sum);

        var dbSpace = new CodegenIterationSpace(CodegenAxis.Parallel("j", outFeatures), CodegenAxis.Reduce("b", batch));
        var db = new CodegenKernelSpec("fused_linear_db", dbSpace,
            new[] { new CodegenTensorBinding(0, "dpre", new[] { batch, outFeatures }, new[] { A(1), A(0) }) },
            new CodegenTensorBinding(1, "dbias", new[] { outFeatures }, new[] { A(0) }, isOutput: true),
            new[] { 0 }, CodegenReduceKind.Sum);

        return new CodegenProgram(new[] { activationGrad, dx, dw, db }, new long[] { (long)batch * outFeatures },
            "fused linear backward (backend contract)");
    }

    /// <summary>
    /// <c>C[m, n] = sum_k op(A)[m, k] op(B)[k, n]</c>, row-major, in CudaBackend.Gemm's contract (alpha 1, beta 0):
    /// op(A) is <c>A[M, K]</c>, or <c>A[K, M]</c> transposed when <paramref name="transA"/>; op(B) is <c>B[K, N]</c>, or
    /// <c>B[N, K]</c> transposed when <paramref name="transB"/>. A <paramref name="batch"/> above 1 adds a leading
    /// batch axis to all three tensors (CudaBackend.BatchedGemm's packed strides).
    /// </summary>
    public static CodegenProgram Gemm(int batch, int m, int n, int k, bool transA, bool transB)
    {
        if (batch <= 0 || m <= 0 || n <= 0 || k <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch), "GEMM extents must be positive.");
        bool batched = batch > 1;
        int lead = batched ? 1 : 0;
        var axes = new List<CodegenAxis>();
        if (batched) axes.Add(CodegenAxis.Parallel("g", batch));
        axes.Add(CodegenAxis.Parallel("m", m));
        axes.Add(CodegenAxis.Parallel("n", n));
        axes.Add(CodegenAxis.Reduce("k", k));
        int M = lead, N = lead + 1, K = lead + 2;

        CodegenTensorBinding Matrix(int parameter, string name, int rows, int columns, int rowAxis, int columnAxis,
            bool isOutput = false)
        {
            var shape = batched ? new[] { batch, rows, columns } : new[] { rows, columns };
            var map = batched ? new[] { A(0), A(rowAxis), A(columnAxis) } : new[] { A(rowAxis), A(columnAxis) };
            return new CodegenTensorBinding(parameter, name, shape, map, isOutput);
        }

        var a = transA ? Matrix(0, "a", k, m, K, M) : Matrix(0, "a", m, k, M, K);
        var b = transB ? Matrix(1, "b", n, k, N, K) : Matrix(1, "b", k, n, K, N);
        var c = Matrix(2, "c", m, n, M, N, isOutput: true);
        var pass = new CodegenKernelSpec(batched ? "batched_gemm" : "gemm", new CodegenIterationSpace(axes.ToArray()),
            new[] { a, b }, c, new[] { 0, 1 }, CodegenReduceKind.Sum);
        return new CodegenProgram(new[] { pass }, Array.Empty<long>(), "gemm (backend contract)");
    }

    private static CodegenAffineExpr A(int axis) => CodegenAffineExpr.Axis(axis);

    public static CodegenProgram LayerNorm(int rows, int columns, double epsilon) =>
        Append(CodegenFusedStatistics.LayerNorm(rows, columns, epsilon),
            RowPass("layernorm_save_mean", new[] { rows }, new[] { "negmean" }, "mean", CodegenExpr.Negate(Op(0))));

    /// <summary>CudaBackend.RmsNorm: sources x, gamma; outputs y, rms (saveRms = 1/rstd).</summary>
    public static CodegenProgram RmsNorm(int rows, int columns, double epsilon) =>
        Append(CodegenFusedStatistics.RmsNorm(rows, columns, epsilon),
            RowPass("rmsnorm_save_rms", new[] { rows }, new[] { "rstd" }, "rms", Op(0), CodegenActivationKind.Reciprocal));

    /// <summary>
    /// CudaBackend.GroupNorm over [batch, channels, spatial]: sources x, gamma, beta; outputs y, mean, rstd, each
    /// statistic [batch, groups].
    /// </summary>
    public static CodegenProgram GroupNorm(int batch, int channels, int groups, int spatial, double epsilon) =>
        Append(CodegenFusedStatistics.GroupNorm(batch, channels, groups, spatial, epsilon),
            RowPass("groupnorm_save_mean", new[] { batch, groups }, new[] { "negmean" }, "mean", CodegenExpr.Negate(Op(0))));

    /// <summary>
    /// CudaBackend.BatchNorm in training mode: sources x, gamma, beta, running_mean, running_var; outputs y, mean,
    /// rstd, running_mean_out and running_var_out. The backend binds running_mean and running_mean_out (and the
    /// variances) to the SAME buffer: each element is read and then written by one thread, as the hand-written kernel
    /// updates them in place.
    /// </summary>
    public static CodegenProgram BatchNorm2DTraining(int batch, int channels, int height, int width,
        double epsilon, double momentum)
    {
        if (!(momentum >= 0 && momentum <= 1)) throw new ArgumentOutOfRangeException(nameof(momentum));
        long count = (long)batch * height * width;
        // The running variance takes the UNBIASED variance, as the hand-written kernel (and PyTorch) do; a single
        // sample has none, so it is left unchanged by folding in the biased one.
        double unbias = count > 1 ? count / (double)(count - 1) : 1.0;
        var c = new[] { channels };
        var program = CodegenFusedStatistics.BatchNorm2D(batch, channels, height, width, epsilon);
        return Append(program,
            RowPass("batchnorm_save_mean", c, new[] { "negmean" }, "mean", CodegenExpr.Negate(Op(0))),
            // running_mean*(1 - m) + mean*m
            RowPass("batchnorm_running_mean", c, new[] { "running_mean", "mean" }, "running_mean_out",
                CodegenExpr.Add(CodegenExpr.Product(Op(0), CodegenExpr.Constant(1 - momentum)),
                                CodegenExpr.Product(Op(1), CodegenExpr.Constant(momentum)))),
            // var = m2s - dmean^2 (the program's own shifted variance), unbiased, folded in.
            RowPass("batchnorm_running_var", c, new[] { "running_var", "m2s", "dmean" }, "running_var_out",
                CodegenExpr.Add(
                    CodegenExpr.Product(Op(0), CodegenExpr.Constant(1 - momentum)),
                    CodegenExpr.Product(CodegenExpr.Constant(momentum * unbias),
                        CodegenExpr.Subtract(Op(1), CodegenExpr.Product(Op(2), Op(2)))))));
    }

    /// <summary>CudaBackend.LayerNormBackward: sources dy, x, gamma, mean, rstd; outputs dx, dgamma, dbeta.</summary>
    public static CodegenProgram LayerNormBackward(int rows, int columns) =>
        Prepend(CodegenFusedStatistics.LayerNormBackward(rows, columns),
            RowPass("layernorm_bwd_negmean", new[] { rows }, new[] { "mean" }, "negmean", CodegenExpr.Negate(Op(0))));

    /// <summary>CudaBackend.RmsNormBackward: sources dy, x, gamma, rms; outputs dx, dgamma.</summary>
    public static CodegenProgram RmsNormBackward(int rows, int columns) =>
        Prepend(CodegenFusedStatistics.RmsNormBackward(rows, columns),
            RowPass("rmsnorm_bwd_rstd", new[] { rows }, new[] { "rms" }, "rstd", Op(0), CodegenActivationKind.Reciprocal));

    /// <summary>CudaBackend.BatchNormBackward: sources dy, x, gamma, mean, rstd; outputs dx, dgamma, dbeta.</summary>
    public static CodegenProgram BatchNorm2DBackward(int batch, int channels, int height, int width) =>
        Prepend(CodegenFusedStatistics.BatchNorm2DBackward(batch, channels, height, width),
            RowPass("batchnorm_bwd_negmean", new[] { channels }, new[] { "mean" }, "negmean", CodegenExpr.Negate(Op(0))));
}
