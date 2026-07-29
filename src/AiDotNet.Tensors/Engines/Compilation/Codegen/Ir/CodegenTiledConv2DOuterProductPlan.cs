// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>A cooperative KxC tile recovered from a split dense 3x3 weight gradient.</summary>
/// <remarks>
/// The split partial has the semantic form
/// <c>partial[K,C,kh,kw,oh] = sum(n,ow) dY[n,K,oh,ow] * X[n,C,oh+kh-1,ow+kw-1]</c>.
/// A CTA fixes <c>(kh,kw,oh)</c>, stages complete contiguous rows for a KxC tile, and
/// reuses them for the outer product. The existing affine combine then sums the promoted
/// <c>oh</c> axis deterministically.
/// </remarks>
public sealed class CodegenTiledConv2DOuterProductPlan
{
    private CodegenTiledConv2DOuterProductPlan(
        int directInput, int windowInput,
        int mAxis, int nAxis, int tapRowAxis, int tapColumnAxis, int batchAxis,
        int outerReductionAxis, int innerReductionAxis,
        int m, int n, int tapRows, int tapColumns, int batch,
        int outerReduction, int innerReduction, int inputHeight, int inputWidth,
        int tileM, int tileN, int threadTileM, int threadTileN, int stages)
    {
        DirectInput = directInput;
        WindowInput = windowInput;
        MAxis = mAxis;
        NAxis = nAxis;
        TapRowAxis = tapRowAxis;
        TapColumnAxis = tapColumnAxis;
        BatchAxis = batchAxis;
        OuterReductionAxis = outerReductionAxis;
        InnerReductionAxis = innerReductionAxis;
        M = m;
        N = n;
        TapRows = tapRows;
        TapColumns = tapColumns;
        Batch = batch;
        OuterReduction = outerReduction;
        InnerReduction = innerReduction;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        TileM = tileM;
        TileN = tileN;
        ThreadTileM = threadTileM;
        ThreadTileN = threadTileN;
        Stages = stages;
    }

    public int DirectInput { get; }
    public int WindowInput { get; }
    public int MAxis { get; }
    public int NAxis { get; }
    public int TapRowAxis { get; }
    public int TapColumnAxis { get; }
    public int BatchAxis { get; }
    public int OuterReductionAxis { get; }
    public int InnerReductionAxis { get; }
    public int M { get; }
    public int N { get; }
    public int TapRows { get; }
    public int TapColumns { get; }
    public int Batch { get; }
    public int OuterReduction { get; }
    public int InnerReduction { get; }
    public int InputHeight { get; }
    public int InputWidth { get; }
    public int TileM { get; }
    public int TileN { get; }
    public int ThreadTileM { get; }
    public int ThreadTileN { get; }
    public int Stages { get; }
    public int ThreadsM => TileM / ThreadTileM;
    public int ThreadsN => TileN / ThreadTileN;
    public int BlockThreads => ThreadsM * ThreadsN;
    public int Steps => OuterReduction;
    public int Blocks => Batch * TapRows * TapColumns * (M / TileM) * (N / TileN);
    public int DirectStageElements => TileM * InnerReduction;
    public int WindowStageElements => TileN * InputWidth;
    public int StageBytes => (DirectStageElements + WindowStageElements) * sizeof(float);
    public int SharedMemoryBytes => Stages * StageBytes;

    public static bool TryCreate(
        CodegenKernelSpec spec,
        out CodegenTiledConv2DOuterProductPlan? plan,
        out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (spec.Reduce != CodegenReduceKind.Sum ||
            spec.PreReduce != CodegenPreReduceOp.None ||
            spec.ProductInputs.Count != 2)
        {
            reason = "a tiled dense weight partial needs an untransformed sum of two operands";
            return false;
        }
        if (spec.PreBiasInput.HasValue || spec.BiasInput.HasValue ||
            spec.ScaleInput.HasValue || spec.ReduceScale != 1.0 ||
            spec.Activation != CodegenActivationKind.None ||
            spec.SecondaryOutput is not null || spec.ExtraOutputs.Count != 0)
        {
            reason = "the split dense weight partial must not contain an epilogue or side output";
            return false;
        }
        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "the tiled dense weight partial accumulates and stores fp32";
            return false;
        }
        foreach (int input in spec.ProductInputs)
            if (spec.Inputs[input].ElementType != CodegenElementType.Float32)
            {
                reason = "the tiled dense weight partial currently stages fp32 operands";
                return false;
            }

        int[] reductions = spec.Space.ReductionAxes;
        if (reductions.Length != 2)
        {
            reason = "the split dense weight partial needs batch and column reductions";
            return false;
        }
        if (!TryIdentityOutput(spec, out int[] outputAxes) || outputAxes.Length != 5 ||
            spec.Output.Shape[2] != 3 || spec.Output.Shape[3] != 3)
        {
            reason = "the split dense weight output must be identity [K,C,3,3,row]";
            return false;
        }

        int first = spec.ProductInputs[0], second = spec.ProductInputs[1];
        if (!MatchesOperands(spec, first, second, outputAxes, reductions,
                out int outerReductionAxis, out int innerReductionAxis))
        {
            if (!MatchesOperands(spec, second, first, outputAxes, reductions,
                    out outerReductionAxis, out innerReductionAxis))
            {
                reason = "operands are not direct-output and padded-input NCHW rows";
                return false;
            }
            (first, second) = (second, first);
        }

        int m = spec.Output.Shape[0];
        int n = spec.Output.Shape[1];
        int batch = spec.Output.Shape[4];
        int outerReduction = spec.Space.Axes[outerReductionAxis].Extent;
        int innerReduction = spec.Space.Axes[innerReductionAxis].Extent;
        int inputHeight = spec.Inputs[second].Shape[2];
        int inputWidth = spec.Inputs[second].Shape[3];
        if (inputHeight != batch || inputWidth != innerReduction ||
            inputWidth % 4 != 0)
        {
            reason = "the split rows must be same-size and vectorizable by four";
            return false;
        }

        int tileM = LargestDivisorAtMost(m, 32, 4);
        int tileN = LargestDivisorAtMost(n, 16, 4);
        if (tileM == 0 || tileN == 0)
        {
            reason = "K or C has no supported whole tile";
            return false;
        }
        int threadTileM = tileM >= 16 ? 2 : 1;
        int threadTileN = tileN >= 16 ? 2 : 1;
        int threads = (tileM / threadTileM) * (tileN / threadTileN);
        if (threads < 32 || threads > 256)
        {
            reason = "the selected dense weight tile needs " + threads +
                " threads, outside [32,256]";
            return false;
        }

        plan = new CodegenTiledConv2DOuterProductPlan(
            first, second,
            outputAxes[0], outputAxes[1], outputAxes[2], outputAxes[3], outputAxes[4],
            outerReductionAxis, innerReductionAxis,
            m, n, 3, 3, batch, outerReduction, innerReduction,
            inputHeight, inputWidth,
            tileM, tileN, threadTileM, threadTileN, stages: 2);
        reason = "eligible";
        return true;
    }

    private static bool MatchesOperands(
        CodegenKernelSpec spec, int directInput, int windowInput,
        int[] outputAxes, int[] reductions,
        out int outerReductionAxis, out int innerReductionAxis)
    {
        outerReductionAxis = innerReductionAxis = -1;
        CodegenTensorBinding direct = spec.Inputs[directInput];
        CodegenTensorBinding window = spec.Inputs[windowInput];
        if (direct.Shape.Count != 4 || direct.Map.Count != 4 ||
            window.Shape.Count != 4 || window.Map.Count != 4)
            return false;

        if (!TryPlainAxis(direct.Map[0], out int outer) ||
            !Contains(reductions, outer) ||
            !TryPlainAxis(direct.Map[1], out int m) || m != outputAxes[0] ||
            !TryPlainAxis(direct.Map[2], out int batch) || batch != outputAxes[4] ||
            !TryPlainAxis(direct.Map[3], out int inner) ||
            !Contains(reductions, inner) || inner == outer)
            return false;

        int[] directAxes = { outer, m, batch, inner };
        for (int d = 0; d < directAxes.Length; d++)
            if (direct.Shape[d] != spec.Space.Axes[directAxes[d]].Extent)
                return false;

        if (!TryPlainAxis(window.Map[0], out int windowOuter) || windowOuter != outer ||
            !TryPlainAxis(window.Map[1], out int n) || n != outputAxes[1] ||
            !TryWindow(window.Map[2], outputAxes[4], outputAxes[2]) ||
            !TryWindow(window.Map[3], inner, outputAxes[3]) ||
            window.Shape[0] != spec.Space.Axes[outer].Extent ||
            window.Shape[1] != spec.Space.Axes[n].Extent)
            return false;

        outerReductionAxis = outer;
        innerReductionAxis = inner;
        return true;
    }

    private static bool TryWindow(
        CodegenAffineExpr expression, int spatialAxis, int tapAxis)
    {
        if (expression.Terms.Count != 2 || expression.Constant != -1 ||
            expression.Divisor != 1 || expression.RequiresExactDivision)
            return false;
        bool spatial = false, tap = false;
        foreach (var term in expression.Terms)
        {
            if (term.Axis == spatialAxis && term.Coefficient == 1) spatial = true;
            else if (term.Axis == tapAxis && term.Coefficient == 1) tap = true;
            else return false;
        }
        return spatial && tap;
    }

    private static bool TryIdentityOutput(CodegenKernelSpec spec, out int[] axes)
    {
        axes = new int[spec.Output.Map.Count];
        if (spec.Output.Shape.Count != spec.Output.Map.Count) return false;
        var seen = new HashSet<int>();
        for (int d = 0; d < axes.Length; d++)
        {
            if (!TryPlainAxis(spec.Output.Map[d], out axes[d])) return false;
            if (spec.Space.Axes[axes[d]].IsReduction || !seen.Add(axes[d])) return false;
            if (spec.Output.Shape[d] != spec.Space.Axes[axes[d]].Extent) return false;
        }
        return seen.Count == spec.Space.ParallelAxes.Length;
    }

    private static bool TryPlainAxis(CodegenAffineExpr expression, out int axis)
    {
        axis = -1;
        if (expression.Terms.Count != 1 || expression.Terms[0].Coefficient != 1 ||
            expression.Constant != 0 || expression.Divisor != 1 ||
            expression.RequiresExactDivision)
            return false;
        axis = expression.Terms[0].Axis;
        return true;
    }

    private static bool Contains(int[] values, int value)
    {
        for (int i = 0; i < values.Length; i++) if (values[i] == value) return true;
        return false;
    }

    private static int LargestDivisorAtMost(int extent, int maximum, int quantum)
    {
        for (int candidate = Math.Min(extent, maximum); candidate >= quantum; candidate--)
            if (candidate % quantum == 0 && extent % candidate == 0) return candidate;
        return 0;
    }
}
