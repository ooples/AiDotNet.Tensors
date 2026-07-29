// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation.Codegen.Ir;

/// <summary>
/// A cooperative reduction for a depthwise 3x3 weight gradient.
/// </summary>
/// <remarks>
/// The semantic form is
/// <c>dW[c,kh,kw] = sum(n,oh,ow) dOut[n,c,oh,ow] * input[n,c,oh+kh-1,ow+kw-1]</c>.
/// One block owns a <c>(c,kh)</c> row and computes all three <c>kw</c> values, so a
/// gradient load feeds three products and consecutive lanes read consecutive spatial
/// positions. The form is recovered from affine maps rather than a kernel name.
/// </remarks>
public sealed class CodegenDepthwiseConv2DWeightGradientPlan
{
    private CodegenDepthwiseConv2DWeightGradientPlan(
        int gradOutputInput, int dataInput, int channelAxis, int kernelRowAxis,
        int kernelColumnAxis, int batchAxis, int outputRowAxis, int outputColumnAxis,
        int channels, int batch, int height, int width)
    {
        GradOutputInput = gradOutputInput;
        DataInput = dataInput;
        ChannelAxis = channelAxis;
        KernelRowAxis = kernelRowAxis;
        KernelColumnAxis = kernelColumnAxis;
        BatchAxis = batchAxis;
        OutputRowAxis = outputRowAxis;
        OutputColumnAxis = outputColumnAxis;
        Channels = channels;
        Batch = batch;
        Height = height;
        Width = width;
    }

    public const int BlockThreads = 256;
    public const int KernelSize = 3;
    public const int Padding = 1;
    public const int WarpSize = 32;

    public int GradOutputInput { get; }
    public int DataInput { get; }
    public int ChannelAxis { get; }
    public int KernelRowAxis { get; }
    public int KernelColumnAxis { get; }
    public int BatchAxis { get; }
    public int OutputRowAxis { get; }
    public int OutputColumnAxis { get; }
    public int Channels { get; }
    public int Batch { get; }
    public int Height { get; }
    public int Width { get; }
    public int Blocks => checked(Channels * KernelSize);
    public int ReductionElements => checked(Batch * Height * Width);
    public int WarpsPerBlock => BlockThreads / WarpSize;
    public int SharedMemoryBytes => KernelSize * WarpsPerBlock * sizeof(float);

    /// <summary>Recovers the exact depthwise weight-gradient form or explains refusal.</summary>
    public static bool TryCreate(
        CodegenKernelSpec spec,
        out CodegenDepthwiseConv2DWeightGradientPlan? plan,
        out string reason)
    {
        if (spec is null) throw new ArgumentNullException(nameof(spec));
        plan = null;

        if (spec.Reduce != CodegenReduceKind.Sum ||
            spec.PreReduce != CodegenPreReduceOp.None ||
            spec.ProductInputs.Count != 2)
        {
            reason = "a depthwise weight-gradient needs an untransformed sum of two operands";
            return false;
        }
        if (spec.PreBiasInput.HasValue || spec.BiasInput.HasValue ||
            spec.ScaleInput.HasValue || spec.ReduceScale != 1.0 ||
            spec.Activation != CodegenActivationKind.None ||
            spec.SecondaryOutput is not null || spec.ExtraOutputs.Count != 0)
        {
            reason = "a depthwise weight-gradient must not contain an epilogue or side output";
            return false;
        }
        if (spec.Output.ElementType != CodegenElementType.Float32)
        {
            reason = "the cooperative reduction accumulates and stores fp32";
            return false;
        }
        if (spec.Algebra != CodegenAlgebra.Real || spec.Output.NeedsAtomicStore)
        {
            reason = "the cooperative reduction requires real arithmetic and one owner per output";
            return false;
        }
        foreach (int input in spec.ProductInputs)
            if (spec.Inputs[input].ElementType != CodegenElementType.Float32)
            {
                reason = "the cooperative reduction currently reads fp32 operands";
                return false;
            }

        if (!TryIdentityOutput(spec, out int[] outputAxes) || outputAxes.Length != 3 ||
            spec.Space.ParallelAxes.Length != 3 || spec.Space.ReductionAxes.Length != 3)
        {
            reason = "the output must be identity [channel,kh,kw] with three reduction axes";
            return false;
        }

        int channel = outputAxes[0], kernelRow = outputAxes[1], kernelColumn = outputAxes[2];
        if (spec.Space.Axes[kernelRow].Extent != KernelSize ||
            spec.Space.Axes[kernelColumn].Extent != KernelSize)
        {
            reason = "only an exact 3x3 depthwise weight gradient is supported";
            return false;
        }

        int first = spec.ProductInputs[0], second = spec.ProductInputs[1];
        if (!TryMatchOperands(spec, first, second, channel, kernelRow, kernelColumn,
                out int batch, out int outputRow, out int outputColumn))
        {
            if (!TryMatchOperands(spec, second, first, channel, kernelRow, kernelColumn,
                    out batch, out outputRow, out outputColumn))
            {
                reason = "operands do not match NCHW dOut and same-padded depthwise input maps";
                return false;
            }
            (first, second) = (second, first);
        }

        int channels = spec.Space.Axes[channel].Extent;
        long elements, tensorElements, outputElements, blocks;
        try
        {
            elements = checked((long)spec.Space.Axes[batch].Extent *
                spec.Space.Axes[outputRow].Extent * spec.Space.Axes[outputColumn].Extent);
            tensorElements = checked(elements * channels);
            outputElements = checked((long)channels * KernelSize * KernelSize);
            blocks = checked((long)channels * KernelSize);
        }
        catch (OverflowException)
        {
            reason = "the depthwise weight-gradient extents overflow a 64-bit element count";
            return false;
        }
        if (elements > int.MaxValue || tensorElements > int.MaxValue ||
            outputElements > int.MaxValue || blocks > int.MaxValue)
        {
            reason = "the cooperative reduction uses signed 32-bit element constants";
            return false;
        }

        plan = new CodegenDepthwiseConv2DWeightGradientPlan(
            first, second, channel, kernelRow, kernelColumn, batch, outputRow, outputColumn,
            channels,
            spec.Space.Axes[batch].Extent,
            spec.Space.Axes[outputRow].Extent,
            spec.Space.Axes[outputColumn].Extent);
        reason = "eligible";
        return true;
    }

    private static bool TryMatchOperands(
        CodegenKernelSpec spec, int gradOutputInput, int dataInput,
        int channel, int kernelRow, int kernelColumn,
        out int batch, out int outputRow, out int outputColumn)
    {
        batch = outputRow = outputColumn = -1;
        var grad = spec.Inputs[gradOutputInput];
        var data = spec.Inputs[dataInput];
        if (grad.Shape.Count != 4 || grad.Map.Count != 4 ||
            data.Shape.Count != 4 || data.Map.Count != 4)
            return false;

        if (!TryPlainAxis(grad.Map[0], out batch) ||
            !TryPlainAxis(grad.Map[1], out int gradChannel) || gradChannel != channel ||
            !TryPlainAxis(grad.Map[2], out outputRow) ||
            !TryPlainAxis(grad.Map[3], out outputColumn))
            return false;

        var reductions = new HashSet<int>(spec.Space.ReductionAxes);
        if (!reductions.SetEquals(new[] { batch, outputRow, outputColumn })) return false;
        if (!ShapeMatchesAxes(spec, grad, batch, channel, outputRow, outputColumn)) return false;
        if (!ShapeMatchesAxes(spec, data, batch, channel, outputRow, outputColumn)) return false;

        return IsPlainAxis(data.Map[0], batch) &&
            IsPlainAxis(data.Map[1], channel) &&
            IsWindow(data.Map[2], outputRow, kernelRow) &&
            IsWindow(data.Map[3], outputColumn, kernelColumn);
    }

    private static bool ShapeMatchesAxes(
        CodegenKernelSpec spec, CodegenTensorBinding binding,
        int batch, int channel, int row, int column)
    {
        int[] expected = { batch, channel, row, column };
        for (int d = 0; d < expected.Length; d++)
            if (binding.Shape[d] != spec.Space.Axes[expected[d]].Extent) return false;
        return true;
    }

    private static bool IsWindow(CodegenAffineExpr expression, int spatial, int tap)
    {
        if (expression.Terms.Count != 2 || expression.Constant != -Padding ||
            expression.Divisor != 1 || expression.RequiresExactDivision)
            return false;

        bool sawSpatial = false, sawTap = false;
        foreach (var term in expression.Terms)
        {
            if (term.Coefficient != 1) return false;
            if (term.Axis == spatial) sawSpatial = true;
            else if (term.Axis == tap) sawTap = true;
            else return false;
        }
        return sawSpatial && sawTap;
    }

    private static bool IsPlainAxis(CodegenAffineExpr expression, int expected) =>
        TryPlainAxis(expression, out int actual) && actual == expected;

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
}
