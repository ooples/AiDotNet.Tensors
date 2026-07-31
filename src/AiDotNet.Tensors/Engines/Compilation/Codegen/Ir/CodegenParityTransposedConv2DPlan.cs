// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>A deterministic 2x2 output tile for depthwise 3x3 stride-2 transpose.</summary>
/// <remarks>
/// For output coordinates <c>(2r,2c)</c>, <c>(2r,2c+1)</c>,
/// <c>(2r+1,2c)</c>, and <c>(2r+1,2c+1)</c>, the exact-division windows admit
/// one, two, two, and four taps respectively. One thread owns that complete tile,
/// removing all run-time remainder guards while keeping each output a deterministic
/// assignment.
/// </remarks>
public sealed class CodegenParityTransposedConv2DPlan
{
    public const int KernelSize = 3;
    public const int Stride = 2;
    public const int Padding = 1;
    public const int BlockThreads = 256;

    private CodegenParityTransposedConv2DPlan(
        int input, int weights,
        int batchAxis, int channelAxis, int rowAxis, int columnAxis,
        int tapRowAxis, int tapColumnAxis,
        int batch, int channels, int inputHeight, int inputWidth,
        int outputHeight, int outputWidth, int inputElements)
    {
        Input = input;
        Weights = weights;
        BatchAxis = batchAxis;
        ChannelAxis = channelAxis;
        RowAxis = rowAxis;
        ColumnAxis = columnAxis;
        TapRowAxis = tapRowAxis;
        TapColumnAxis = tapColumnAxis;
        Batch = batch;
        Channels = channels;
        InputHeight = inputHeight;
        InputWidth = inputWidth;
        OutputHeight = outputHeight;
        OutputWidth = outputWidth;
        InputElements = inputElements;
    }

    public int Input { get; }
    public int Weights { get; }
    public int BatchAxis { get; }
    public int ChannelAxis { get; }
    public int RowAxis { get; }
    public int ColumnAxis { get; }
    public int TapRowAxis { get; }
    public int TapColumnAxis { get; }
    public int Batch { get; }
    public int Channels { get; }
    public int InputHeight { get; }
    public int InputWidth { get; }
    public int OutputHeight { get; }
    public int OutputWidth { get; }
    public int InputElements { get; }
    public int Blocks => checked((int)(
        (InputElements + (long)BlockThreads - 1) / BlockThreads));

    public static bool TryCreate(
        CodegenKernelSpec spec,
        out CodegenParityTransposedConv2DPlan? plan,
        out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (spec.Algebra != CodegenAlgebra.Real ||
            spec.Reduce != CodegenReduceKind.Sum ||
            spec.PreReduce != CodegenPreReduceOp.None ||
            spec.ProductInputs.Count != 2)
        {
            reason = "a parity-transposed tile needs a real untransformed sum of two operands";
            return false;
        }
        if (spec.PreBiasInput.HasValue || spec.BiasInput.HasValue ||
            spec.ScaleInput.HasValue || spec.ReduceScale != 1.0 ||
            spec.Activation != CodegenActivationKind.None ||
            spec.SecondaryOutput is not null || spec.ExtraOutputs.Count != 0)
        {
            reason = "the parity-transposed tile does not accept an epilogue or side output";
            return false;
        }
        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "the parity-transposed tile stores fp32";
            return false;
        }
        foreach (int input in spec.ProductInputs)
            if (spec.Inputs[input].ElementType != CodegenElementType.Float32)
            {
                reason = "the parity-transposed tile currently reads fp32 operands";
                return false;
            }

        if (!TryIdentityOutput(spec, out int[] outputAxes) || outputAxes.Length != 4)
        {
            reason = "the parity-transposed output must be identity [batch,channel,H,W]";
            return false;
        }
        int[] reductions = spec.Space.ReductionAxes;
        if (reductions.Length != 2 ||
            spec.Space.Axes[reductions[0]].Extent != KernelSize ||
            spec.Space.Axes[reductions[1]].Extent != KernelSize)
        {
            reason = "the parity-transposed tile requires two 3-tap reduction axes";
            return false;
        }

        int first = spec.ProductInputs[0], second = spec.ProductInputs[1];
        if (!TryWeights(spec, spec.Inputs[second], outputAxes[1], reductions,
                out int tapRow, out int tapColumn))
        {
            if (!TryWeights(spec, spec.Inputs[first], outputAxes[1], reductions,
                    out tapRow, out tapColumn))
            {
                reason = "no product operand is an identity [channel,3,3] weight tensor";
                return false;
            }
            (first, second) = (second, first);
        }

        CodegenTensorBinding inputBinding = spec.Inputs[first];
        if (inputBinding.Shape.Count != 4 || inputBinding.Map.Count != 4 ||
            !TryPlainAxis(inputBinding.Map[0], out int batchAxis) ||
            batchAxis != outputAxes[0] ||
            !TryPlainAxis(inputBinding.Map[1], out int channelAxis) ||
            channelAxis != outputAxes[1] ||
            !MatchesWindow(inputBinding.Map[2], outputAxes[2], tapRow) ||
            !MatchesWindow(inputBinding.Map[3], outputAxes[3], tapColumn))
        {
            reason = "the activation operand is not a depthwise stride-2 transposed window";
            return false;
        }

        int batch = inputBinding.Shape[0];
        int channels = inputBinding.Shape[1];
        int inputHeight = inputBinding.Shape[2];
        int inputWidth = inputBinding.Shape[3];
        int outputHeight = spec.Output.Shape[2];
        int outputWidth = spec.Output.Shape[3];
        if (batch != spec.Output.Shape[0] || channels != spec.Output.Shape[1] ||
            batch != spec.Space.Axes[batchAxis].Extent ||
            channels != spec.Space.Axes[channelAxis].Extent ||
            outputHeight != Stride * (long)inputHeight - 1 ||
            outputWidth != Stride * (long)inputWidth - 1)
        {
            reason = "the parity-transposed tile requires output extent 2*input-1";
            return false;
        }

        long inputElements = inputBinding.ElementCount;
        if (inputElements > int.MaxValue || spec.Output.ElementCount > int.MaxValue ||
            spec.Inputs[second].ElementCount > int.MaxValue)
        {
            reason = "the parity-transposed tile currently uses int32 element offsets";
            return false;
        }

        plan = new CodegenParityTransposedConv2DPlan(
            first, second,
            outputAxes[0], outputAxes[1], outputAxes[2], outputAxes[3],
            tapRow, tapColumn,
            batch, channels, inputHeight, inputWidth, outputHeight, outputWidth,
            checked((int)inputElements));
        reason = "eligible";
        return true;
    }

    private static bool TryWeights(
        CodegenKernelSpec spec, CodegenTensorBinding binding,
        int channelAxis, int[] reductions,
        out int tapRow, out int tapColumn)
    {
        tapRow = tapColumn = -1;
        if (binding.Shape.Count != 3 || binding.Map.Count != 3 ||
            !TryPlainAxis(binding.Map[0], out int channel) || channel != channelAxis ||
            !TryPlainAxis(binding.Map[1], out tapRow) ||
            !TryPlainAxis(binding.Map[2], out tapColumn) || tapRow == tapColumn ||
            !Contains(reductions, tapRow) || !Contains(reductions, tapColumn))
            return false;

        return binding.Shape[0] == spec.Space.Axes[channel].Extent &&
            binding.Shape[1] == KernelSize && binding.Shape[2] == KernelSize;
    }

    private static bool MatchesWindow(
        CodegenAffineExpr expression, int spatialAxis, int tapAxis)
    {
        if (expression.Terms.Count != 2 || expression.Constant != Padding ||
            expression.Divisor != Stride || !expression.RequiresExactDivision)
            return false;

        bool spatial = false, tap = false;
        foreach (CodegenAffineTerm term in expression.Terms)
        {
            if (term.Axis == spatialAxis && term.Coefficient == 1) spatial = true;
            else if (term.Axis == tapAxis && term.Coefficient == -1) tap = true;
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
            if (!TryPlainAxis(spec.Output.Map[d], out axes[d]) ||
                spec.Space.Axes[axes[d]].IsReduction || !seen.Add(axes[d]) ||
                spec.Output.Shape[d] != spec.Space.Axes[axes[d]].Extent)
                return false;
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
}
