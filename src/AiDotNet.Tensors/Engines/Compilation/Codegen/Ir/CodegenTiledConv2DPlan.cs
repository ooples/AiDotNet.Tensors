// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>A cooperative same-row tile recovered from a dense 3x3 convolution spec.</summary>
/// <remarks>
/// One CTA owns <c>[TileM, outputWidth]</c> for a fixed batch and output row. Each
/// reduction step stages several channels of all three input rows and all nine weights,
/// so input values are reused across output channels and adjacent taps while weights are
/// reused across the complete output row. Both direct and adjoint windows are recognized
/// from their affine maps.
/// </remarks>
public sealed class CodegenTiledConv2DPlan
{
    private CodegenTiledConv2DPlan(
        int matrixInput, int streamInput, int? biasInput,
        int batchAxis, int mAxis, int rowAxis, int columnAxis,
        int reductionChannelAxis, int tapRowAxis, int tapColumnAxis,
        bool matrixReductionMajor, int tapSign, int windowConstant,
        int batch, int m, int outputHeight, int outputWidth,
        int reductionChannels, int inputHeight, int inputWidth,
        int tileM, int tileChannels, int threadTileM, int threadTileWidth,
        int stages)
    {
        MatrixInput = matrixInput;
        StreamInput = streamInput;
        BiasInput = biasInput;
        BatchAxis = batchAxis;
        MAxis = mAxis;
        RowAxis = rowAxis;
        ColumnAxis = columnAxis;
        ReductionChannelAxis = reductionChannelAxis;
        TapRowAxis = tapRowAxis;
        TapColumnAxis = tapColumnAxis;
        MatrixReductionMajor = matrixReductionMajor;
        TapSign = tapSign;
        WindowConstant = windowConstant;
        Batch = batch;
        M = m;
        OutputHeight = outputHeight;
        OutputWidth = outputWidth;
        ReductionChannels = reductionChannels;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        TileM = tileM;
        TileChannels = tileChannels;
        ThreadTileM = threadTileM;
        ThreadTileWidth = threadTileWidth;
        Stages = stages;
    }

    public int MatrixInput { get; }
    public int StreamInput { get; }
    public int? BiasInput { get; }
    public int BatchAxis { get; }
    public int MAxis { get; }
    public int RowAxis { get; }
    public int ColumnAxis { get; }
    public int ReductionChannelAxis { get; }
    public int TapRowAxis { get; }
    public int TapColumnAxis { get; }
    public bool MatrixReductionMajor { get; }
    public int TapSign { get; }
    public int WindowConstant { get; }
    public int Batch { get; }
    public int M { get; }
    public int OutputHeight { get; }
    public int OutputWidth { get; }
    public int ReductionChannels { get; }
    public int InputHeight { get; }
    public int InputWidth { get; }
    public int TileM { get; }
    public int TileChannels { get; }
    public int ThreadTileM { get; }
    public int ThreadTileWidth { get; }
    public int Stages { get; }
    public int TapRows => 3;
    public int TapColumns => 3;
    public int ThreadsM => TileM / ThreadTileM;
    public int ThreadsWidth => OutputWidth / ThreadTileWidth;
    public int BlockThreads => ThreadsM * ThreadsWidth;
    public int Steps => ReductionChannels / TileChannels;
    public int Blocks => Batch * OutputHeight * (M / TileM);
    public int MatrixStageElements => TileM * TileChannels * TapRows * TapColumns;
    public int StreamStageElements => TileChannels * TapRows * InputWidth;
    public int StageBytes => (MatrixStageElements + StreamStageElements) * sizeof(float);
    public int SharedMemoryBytes => Stages * StageBytes;

    public static bool TryCreate(
        CodegenKernelSpec spec, out CodegenTiledConv2DPlan? plan, out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (spec.Reduce != CodegenReduceKind.Sum ||
            spec.PreReduce != CodegenPreReduceOp.None ||
            spec.ProductInputs.Count != 2)
        {
            reason = "a tiled dense convolution needs an untransformed sum of two operands";
            return false;
        }
        if (spec.PreBiasInput.HasValue || spec.ReduceScale != 1.0 ||
            spec.ScaleInput.HasValue ||
            (spec.Activation != CodegenActivationKind.None &&
             spec.Activation != CodegenActivationKind.ReLU) ||
            spec.SecondaryOutput is not null || spec.ExtraOutputs.Count != 0)
        {
            reason = "the tiled dense convolution accepts only an optional M bias and ReLU";
            return false;
        }
        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "the tiled dense convolution accumulates and stores fp32";
            return false;
        }
        foreach (int input in spec.ProductInputs)
            if (spec.Inputs[input].ElementType != CodegenElementType.Float32)
            {
                reason = "the tiled dense convolution currently stages fp32 operands";
                return false;
            }

        if (!TryIdentityOutput(spec, out int[] outputAxes) || outputAxes.Length != 4)
        {
            reason = "the tiled dense-convolution output must be identity [batch,M,H,W]";
            return false;
        }
        int[] reductions = spec.Space.ReductionAxes;
        if (reductions.Length != 3)
        {
            reason = "a dense 3x3 tile needs channel and two tap reductions";
            return false;
        }

        int matrixInput = -1, reductionChannel = -1, tapRow = -1, tapColumn = -1;
        bool reductionMajor = false;
        foreach (int input in spec.ProductInputs)
        {
            if (!TryMatrix(spec, spec.Inputs[input], outputAxes[1], reductions,
                    out int candidateReduction, out int candidateTapRow,
                    out int candidateTapColumn, out bool candidateReductionMajor))
                continue;
            if (matrixInput >= 0)
            {
                reason = "both product operands look like the dense 3x3 weight matrix";
                return false;
            }
            matrixInput = input;
            reductionChannel = candidateReduction;
            tapRow = candidateTapRow;
            tapColumn = candidateTapColumn;
            reductionMajor = candidateReductionMajor;
        }
        if (matrixInput < 0)
        {
            reason = "no product operand is [M,C,3,3] or [C,M,3,3]";
            return false;
        }

        int streamInput = spec.ProductInputs[0] == matrixInput
            ? spec.ProductInputs[1]
            : spec.ProductInputs[0];
        if (!TryStream(spec, spec.Inputs[streamInput], outputAxes,
                reductionChannel, tapRow, tapColumn,
                out int tapSign, out int windowConstant))
        {
            reason = "the activation operand is not [batch,C,window(H),window(W)]";
            return false;
        }

        int m = spec.Output.Shape[1];
        if (spec.BiasInput.HasValue &&
            !IsMBias(spec, spec.Inputs[spec.BiasInput.Value], outputAxes[1], m))
        {
            reason = "the tiled dense-convolution bias must be a one-dimensional fp32 M broadcast";
            return false;
        }

        int inputHeight = spec.Inputs[streamInput].Shape[2];
        int inputWidth = spec.Inputs[streamInput].Shape[3];
        int outputHeight = spec.Output.Shape[2];
        int outputWidth = spec.Output.Shape[3];
        if (inputHeight != outputHeight || inputWidth != outputWidth ||
            !((tapSign == 1 && windowConstant == -1) ||
              (tapSign == -1 && windowConstant == 1)))
        {
            reason = "the row tile currently requires same-size padding-one direct or adjoint windows";
            return false;
        }
        if (inputWidth % 4 != 0 || outputWidth % 4 != 0)
        {
            reason = "input and output rows must vectorize by four";
            return false;
        }
        int tileM = LargestDivisorAtMost(m, 32, 4);
        int channels = spec.Space.Axes[reductionChannel].Extent;
        int tileChannels = LargestDivisorAtMost(channels, 4, 4);
        if (tileM == 0 || tileChannels == 0)
        {
            reason = "M and reduction channels need supported whole tiles";
            return false;
        }
        int threadTileM = tileM >= 16 ? 2 : 1;
        const int threadTileWidth = 4;
        long sharedBytes = CodegenSharedMemoryBudget.DoubleBufferStages *
            (tileM * (long)tileChannels * 9 +
             tileChannels * 3L * inputWidth) * sizeof(float);
        if (!CodegenSharedMemoryBudget.Fits(sharedBytes, out reason))
            return false;

        int threads = (tileM / threadTileM) * (outputWidth / threadTileWidth);
        if (threads < 32 || threads > 256)
        {
            reason = "the selected row tile needs " + threads + " threads, outside [32,256]";
            return false;
        }

        plan = new CodegenTiledConv2DPlan(
            matrixInput, streamInput, spec.BiasInput,
            outputAxes[0], outputAxes[1], outputAxes[2], outputAxes[3],
            reductionChannel, tapRow, tapColumn,
            reductionMajor, tapSign, windowConstant,
            spec.Output.Shape[0], m, outputHeight, outputWidth,
            channels, inputHeight, inputWidth,
            tileM, tileChannels, threadTileM, threadTileWidth,
            stages: CodegenSharedMemoryBudget.DoubleBufferStages);
        reason = "eligible";
        return true;
    }

    private static bool TryMatrix(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int mAxis, int[] reductions,
        out int reductionChannel, out int tapRow, out int tapColumn,
        out bool reductionMajor)
    {
        reductionChannel = tapRow = tapColumn = -1;
        reductionMajor = false;
        if (binding.Shape.Count != 4 || binding.Map.Count != 4) return false;
        var axes = new int[4];
        for (int d = 0; d < axes.Length; d++)
            if (!TryPlainAxis(binding.Map[d], out axes[d]) ||
                binding.Shape[d] != spec.Space.Axes[axes[d]].Extent)
                return false;
        if (axes[0] == mAxis && Contains(reductions, axes[1]))
        {
            reductionChannel = axes[1];
            reductionMajor = false;
        }
        else if (Contains(reductions, axes[0]) && axes[1] == mAxis)
        {
            reductionChannel = axes[0];
            reductionMajor = true;
        }
        else return false;

        if (!Contains(reductions, axes[2]) || !Contains(reductions, axes[3]) ||
            axes[2] == reductionChannel || axes[3] == reductionChannel ||
            axes[2] == axes[3] || binding.Shape[2] != 3 || binding.Shape[3] != 3)
            return false;
        tapRow = axes[2];
        tapColumn = axes[3];
        return true;
    }

    private static bool TryStream(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int[] outputAxes,
        int reductionChannel, int tapRow, int tapColumn,
        out int tapSign, out int windowConstant)
    {
        tapSign = 0;
        windowConstant = 0;
        if (binding.Shape.Count != 4 || binding.Map.Count != 4) return false;
        if (!TryPlainAxis(binding.Map[0], out int batch) || batch != outputAxes[0] ||
            !TryPlainAxis(binding.Map[1], out int channel) || channel != reductionChannel)
            return false;
        if (!TryUnitWindow(binding.Map[2], outputAxes[2], tapRow,
                out int rowSign, out int rowConstant) ||
            !TryUnitWindow(binding.Map[3], outputAxes[3], tapColumn,
                out int columnSign, out int columnConstant) ||
            rowSign != columnSign || rowConstant != columnConstant)
            return false;
        if (binding.Shape[0] != spec.Space.Axes[outputAxes[0]].Extent ||
            binding.Shape[1] != spec.Space.Axes[reductionChannel].Extent)
            return false;
        tapSign = rowSign;
        windowConstant = rowConstant;
        return true;
    }

    private static bool TryUnitWindow(
        CodegenAffineExpr expression, int spatialAxis, int tapAxis,
        out int tapSign, out int constant)
    {
        tapSign = 0;
        constant = 0;
        if (expression.Terms.Count != 2 || expression.Divisor != 1 ||
            expression.RequiresExactDivision)
            return false;
        int spatialCoefficient = 0;
        foreach (var term in expression.Terms)
        {
            if (term.Axis == spatialAxis) spatialCoefficient = term.Coefficient;
            else if (term.Axis == tapAxis) tapSign = term.Coefficient;
            else return false;
        }
        if (spatialCoefficient != 1 || (tapSign != 1 && tapSign != -1)) return false;
        constant = expression.Constant;
        return true;
    }

    private static bool IsMBias(
        CodegenKernelSpec spec, CodegenTensorBinding binding, int mAxis, int m)
    {
        return binding.ElementType == CodegenElementType.Float32 &&
            binding.Shape.Count == 1 && binding.Map.Count == 1 &&
            binding.Shape[0] == m && binding.Shape[0] == spec.Space.Axes[mAxis].Extent &&
            TryPlainAxis(binding.Map[0], out int axis) && axis == mAxis;
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
