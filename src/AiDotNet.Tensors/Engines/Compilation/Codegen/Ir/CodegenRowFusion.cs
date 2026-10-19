// Copyright (c) AiDotNet. All rights reserved.
// Fusing a row-wise program -- softmax, LayerNorm, RMSNorm -- into ONE kernel.
//
// As separate passes each statistic re-reads the whole input: LayerNorm at 8192x1024 is three
// reads and a write of 32 MB, and on a GTX 1660 Ti it ran at 0.62x PyTorch's single fused
// kernel even though every pass was near DRAM bandwidth. Bandwidth was not the limit; bytes
// were. A warp per row that keeps the row in registers across every pass reads x once and
// writes y once, and turns each statistic into a register-to-register shuffle reduction.
//
// The plan is the IR's half: it decides whether a program has the row shape and classifies
// every binding. Emission is PtxAffineEmitter.EmitRowProgram.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>How a binding of a row-fused program is indexed.</summary>
public enum CodegenRowOperandKind
{
    /// <summary><c>[rows, width]</c> indexed (row, j): one value per element.</summary>
    Matrix,

    /// <summary><c>[rows]</c> indexed (row): a per-row statistic.</summary>
    Row,

    /// <summary><c>[width]</c> indexed (j): a per-feature parameter such as gamma.</summary>
    Feature,
}

/// <summary>A program accepted for single-kernel row fusion.</summary>
public sealed class CodegenRowFusionPlan
{
    internal CodegenRowFusionPlan(
        CodegenProgram program, int rows, int width,
        IReadOnlyList<string> parameters, IReadOnlyDictionary<string, CodegenRowOperandKind> kinds)
    {
        Program = program;
        Rows = rows;
        Width = width;
        Parameters = parameters;
        Kinds = kinds;
    }

    /// <summary>The program being fused.</summary>
    public CodegenProgram Program { get; }

    /// <summary>Rows, one warp each.</summary>
    public int Rows { get; }

    /// <summary>Elements per row; a multiple of 32, so each lane holds Width/32 of them.</summary>
    public int Width { get; }

    /// <summary>
    /// The fused kernel's parameters, by name and in order: every program source, then every
    /// pass output. A launcher binds buffers by these names.
    /// </summary>
    public IReadOnlyList<string> Parameters { get; }

    /// <summary>The kind of every named tensor.</summary>
    public IReadOnlyDictionary<string, CodegenRowOperandKind> Kinds { get; }

    /// <summary>Elements held per lane.</summary>
    public int PerLane => Width / CodegenWarpReduction.Lanes;
}

/// <summary>Decides whether a program can run as one warp-per-row kernel.</summary>
public static class CodegenRowFusion
{
    /// <summary>
    /// Widest row held in registers: 32 values per lane per matrix source. Past this a row
    /// would spill, and the multi-pass program is the better lowering.
    /// </summary>
    public const int MaxWidth = 32 * CodegenWarpReduction.Lanes;

    /// <summary>Returns the plan, or null with the reason the program does not fit.</summary>
    public static CodegenRowFusionPlan? TryPlan(CodegenProgram program, out string reason)
    {
        if (program is null) throw new ArgumentNullException(nameof(program));
        reason = "";
        if (program.Passes.Count == 0) { reason = "no passes"; return null; }

        var first = program.Passes[0].Space.Axes;
        if (first.Count != 2) { reason = "a pass is not (row | j)"; return null; }
        int rows = first[0].Extent, width = first[1].Extent;
        if (width % CodegenWarpReduction.Lanes != 0 || width > MaxWidth)
        {
            reason = "width " + width + " is not a multiple of 32 up to " + MaxWidth;
            return null;
        }

        var kinds = new Dictionary<string, CodegenRowOperandKind>(StringComparer.Ordinal);
        var produced = new HashSet<string>(StringComparer.Ordinal);
        foreach (var pass in program.Passes)
        {
            var axes = pass.Space.Axes;
            if (axes.Count != 2 || axes[0].IsReduction || axes[0].Extent != rows || axes[1].Extent != width)
            {
                reason = pass.Name + " is not (row | j) over " + rows + "x" + width;
                return null;
            }
            bool reduces = axes[1].IsReduction;
            if (pass.Algebra != CodegenAlgebra.Real || pass.SecondaryOutput is not null ||
                pass.ExtraOutputs.Count != 0)
            {
                reason = pass.Name + " has an argmax, extra outputs or a non-real algebra";
                return null;
            }
            if (reduces != (pass.Reduce != CodegenReduceKind.None))
            {
                reason = pass.Name + " reduces inconsistently";
                return null;
            }

            foreach (var input in pass.Inputs)
            {
                if (!Classify(input, rows, width, out var kind))
                {
                    reason = pass.Name + " reads '" + input.Name + "' with an index map the row kernel does not hold";
                    return null;
                }
                if (kinds.TryGetValue(input.Name, out var known) && known != kind)
                {
                    reason = "'" + input.Name + "' is read two different ways";
                    return null;
                }
                kinds[input.Name] = kind;
                // A matrix the kernel PRODUCED is not held in registers; only sources are.
                if (kind == CodegenRowOperandKind.Matrix && produced.Contains(input.Name))
                {
                    reason = pass.Name + " reads the matrix intermediate '" + input.Name + "'";
                    return null;
                }
            }

            // The reduction's epilogue acts on a per-row value, so it can only read
            // per-row operands; a feature parameter there would have no single value.
            if (reduces)
                foreach (int? operand in new[] { pass.BiasInput, pass.ScaleInput, pass.PostScaleInput, pass.PostBiasInput })
                    if (operand is { } at && kinds[pass.Inputs[at].Name] != CodegenRowOperandKind.Row)
                    {
                        reason = pass.Name + "'s epilogue reads a non-row operand";
                        return null;
                    }

            if (!Classify(pass.Output, rows, width, out var outKind) ||
                outKind != (reduces ? CodegenRowOperandKind.Row : CodegenRowOperandKind.Matrix))
            {
                reason = pass.Name + "'s output is not " + (reduces ? "[rows]" : "[rows, width]");
                return null;
            }
            kinds[pass.Output.Name] = outKind;
            produced.Add(pass.Output.Name);
        }

        var parameters = new List<string>();
        foreach (var source in program.Sources()) parameters.Add(source.Name);
        foreach (var pass in program.Passes) parameters.Add(pass.Output.Name);
        return new CodegenRowFusionPlan(program, rows, width, parameters, kinds);
    }

    private static bool Classify(CodegenTensorBinding binding, int rows, int width, out CodegenRowOperandKind kind)
    {
        kind = CodegenRowOperandKind.Matrix;
        if (binding.HasIndirection || binding.ElementType != CodegenElementType.Float32) return false;
        var shape = binding.Shape;
        var map = binding.Map;
        if (shape.Count == 2 && shape[0] == rows && shape[1] == width && IsAxis(map[0], 0) && IsAxis(map[1], 1))
        {
            kind = CodegenRowOperandKind.Matrix;
            return true;
        }
        if (shape.Count == 1 && shape[0] == rows && IsAxis(map[0], 0))
        {
            kind = CodegenRowOperandKind.Row;
            return true;
        }
        if (shape.Count == 1 && shape[0] == width && IsAxis(map[0], 1))
        {
            kind = CodegenRowOperandKind.Feature;
            return true;
        }
        return false;
    }

    private static bool IsAxis(CodegenAffineExpr expr, int axis) =>
        expr.Terms.Count == 1 && expr.Terms[0].Axis == axis && expr.Terms[0].Coefficient == 1 &&
        expr.Constant == 0 && expr.Divisor == 1;
}
