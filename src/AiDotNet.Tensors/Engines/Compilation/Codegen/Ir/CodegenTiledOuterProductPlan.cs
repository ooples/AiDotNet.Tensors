// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// A cooperative FP32 output tile for a split NCHW pointwise weight gradient.
/// </summary>
/// <remarks>
/// The semantic form is
/// <c>partial[m,n,b] = sum(r0,r1) left[r0,m,b,r1] * right[r0,n,b,r1]</c>.
/// It is intentionally matched from maps rather than kernel names. The split reduction
/// promotes one spatial axis to <c>b</c>; the remaining batch and contiguous spatial axes
/// stay reductions. Tiling M and N lets every staged activation serve an entire output row
/// or column without changing the deterministic two-pass combine.
/// </remarks>
public sealed class CodegenTiledOuterProductPlan
{
    private CodegenTiledOuterProductPlan(
        int leftInput, int rightInput, int mAxis, int nAxis, int batchAxis,
        int outerReductionAxis, int innerReductionAxis,
        int m, int n, int batch, int outerReduction, int innerReduction,
        int tileM, int tileN, int threadTileM, int threadTileN, int stages)
    {
        LeftInput = leftInput;
        RightInput = rightInput;
        MAxis = mAxis;
        NAxis = nAxis;
        BatchAxis = batchAxis;
        OuterReductionAxis = outerReductionAxis;
        InnerReductionAxis = innerReductionAxis;
        M = m;
        N = n;
        Batch = batch;
        OuterReduction = outerReduction;
        InnerReduction = innerReduction;
        TileM = tileM;
        TileN = tileN;
        ThreadTileM = threadTileM;
        ThreadTileN = threadTileN;
        Stages = stages;
    }

    public int LeftInput { get; }
    public int RightInput { get; }
    public int MAxis { get; }
    public int NAxis { get; }
    public int BatchAxis { get; }
    public int OuterReductionAxis { get; }
    public int InnerReductionAxis { get; }
    public int M { get; }
    public int N { get; }
    public int Batch { get; }
    public int OuterReduction { get; }
    public int InnerReduction { get; }
    public int TileM { get; }
    public int TileN { get; }
    public int ThreadTileM { get; }
    public int ThreadTileN { get; }
    public int Stages { get; }
    public int ThreadsM => TileM / ThreadTileM;
    public int ThreadsN => TileN / ThreadTileN;
    public int BlockThreads => ThreadsM * ThreadsN;
    public int Steps => OuterReduction;
    public int Blocks => Batch * (M / TileM) * (N / TileN);
    public int StageBytes => InnerReduction * (TileM + TileN) * sizeof(float);
    public int SharedMemoryBytes => Stages * StageBytes;

    /// <summary>Recovers the exact split outer-product form or reports why it was refused.</summary>
    public static bool TryCreate(
        CodegenKernelSpec spec, out CodegenTiledOuterProductPlan? plan, out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (spec.Reduce != CodegenReduceKind.Sum ||
            spec.PreReduce != CodegenPreReduceOp.None ||
            spec.ProductInputs.Count != 2)
        {
            reason = "a tiled outer product needs an untransformed sum of two operands";
            return false;
        }
        if (spec.PreBiasInput.HasValue || spec.BiasInput.HasValue ||
            spec.ScaleInput.HasValue || spec.ReduceScale != 1.0 ||
            spec.Activation != CodegenActivationKind.None ||
            spec.SecondaryOutput is not null || spec.ExtraOutputs.Count != 0)
        {
            reason = "the split partial must not contain an epilogue or side output";
            return false;
        }
        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "the tiled outer product accumulates and stores fp32";
            return false;
        }
        foreach (int input in spec.ProductInputs)
            if (spec.Inputs[input].ElementType != CodegenElementType.Float32)
            {
                reason = "the tiled outer product currently stages fp32 operands";
                return false;
            }

        int[] reductions = spec.Space.ReductionAxes;
        if (reductions.Length != 2)
        {
            reason = "the split outer-product form needs two surviving reduction axes";
            return false;
        }
        if (!TryIdentityOutput(spec, out int[] outputAxes) || outputAxes.Length != 3)
        {
            reason = "the split outer-product output must be identity [M,N,batch]";
            return false;
        }

        int first = spec.ProductInputs[0], second = spec.ProductInputs[1];
        if (!MatchesOperands(spec, first, second, outputAxes, reductions,
                out int outerReductionAxis, out int innerReductionAxis))
        {
            if (!MatchesOperands(spec, second, first, outputAxes, reductions,
                    out outerReductionAxis, out innerReductionAxis))
            {
                reason = "operands are not matching NCHW [r0,channel,batch,r1] bindings";
                return false;
            }
            (first, second) = (second, first);
        }

        int m = spec.Output.Shape[0];
        int n = spec.Output.Shape[1];
        int batch = spec.Output.Shape[2];
        int outerReduction = spec.Space.Axes[outerReductionAxis].Extent;
        int innerReduction = spec.Space.Axes[innerReductionAxis].Extent;
        if (innerReduction < 4 || innerReduction > 64 || innerReduction % 4 != 0)
        {
            reason = "the contiguous reduction row must contain 4..64 values in groups of four";
            return false;
        }

        int tileM = LargestDivisorAtMost(m, 16, 4);
        int tileN = LargestDivisorAtMost(n, 16, 4);
        if (tileM == 0 || tileN == 0)
        {
            reason = "the M or N extent has no supported whole tile";
            return false;
        }
        int threadTileM = tileM >= 8 ? 2 : 1;
        int threadTileN = tileN >= 16 ? 2 : 1;
        int threads = (tileM / threadTileM) * (tileN / threadTileN);
        if (threads < 32 || threads > 256)
        {
            reason = "the selected whole tile needs " + threads + " threads, outside [32,256]";
            return false;
        }

        plan = new CodegenTiledOuterProductPlan(
            first, second, outputAxes[0], outputAxes[1], outputAxes[2],
            outerReductionAxis, innerReductionAxis,
            m, n, batch, outerReduction, innerReduction,
            tileM, tileN, threadTileM, threadTileN, stages: 2);
        reason = "eligible";
        return true;
    }

    private static bool MatchesOperands(
        CodegenKernelSpec spec, int leftInput, int rightInput,
        int[] outputAxes, int[] reductions,
        out int outerReductionAxis, out int innerReductionAxis)
    {
        outerReductionAxis = reductions[0];
        innerReductionAxis = reductions[1];
        return MatchesOperand(spec, spec.Inputs[leftInput], outerReductionAxis, outputAxes[0],
                outputAxes[2], innerReductionAxis) &&
            MatchesOperand(spec, spec.Inputs[rightInput], outerReductionAxis, outputAxes[1],
                outputAxes[2], innerReductionAxis);
    }

    private static bool MatchesOperand(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int outerReduction, int channel,
        int batch, int innerReduction)
    {
        if (binding.Shape.Count != 4 || binding.Map.Count != 4) return false;
        int[] expected = { outerReduction, channel, batch, innerReduction };
        for (int d = 0; d < expected.Length; d++)
            if (!TryPlainAxis(binding.Map[d], out int actual) || actual != expected[d] ||
                binding.Shape[d] != spec.Space.Axes[expected[d]].Extent)
                return false;
        return true;
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

    private static int LargestDivisorAtMost(int extent, int maximum, int quantum)
    {
        for (int candidate = Math.Min(extent, maximum); candidate >= quantum; candidate--)
            if (candidate % quantum == 0 && extent % candidate == 0) return candidate;
        return 0;
    }
}
