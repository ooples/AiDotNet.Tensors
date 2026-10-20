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
