using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class CpuEngine
{
    /// <summary>
    /// Compiled-graph mean over one axis of a contiguous float32 host tensor (e.g. sequence pooling before a
    /// classifier head): one node whose forward writes the means straight into its output and whose backward writes
    /// dY / axisSize, broadcast along the axis, straight into the input's gradient accumulator. The generic node ran an
    /// eager ReduceMean into a fresh tensor and copied it every step, and its backward built and copied a broadcast
    /// gradient. Returns null (the caller records the generic node) for any other layout.
    /// </summary>
    private static Tensor<float>? TryRecordReduceMeanFloat(
        LazyTensorScope scope, Tensor<float> input, int[] axes, bool keepDims, int[] outShape)
    {
        if (axes.Length != 1 || !IsZeroOffsetContiguous(input)) return null;
        int axis = axes[0] < 0 ? input.Rank + axes[0] : axes[0];
        if (axis < 0 || axis >= input.Rank) return null;
        int outer = 1, inner = 1, axisSize = input._shape[axis];
        for (int d = 0; d < axis; d++) outer *= input._shape[d];
        for (int d = axis + 1; d < input.Rank; d++) inner *= input._shape[d];
        if (axisSize == 0) return null;
        var shape = new[] { outer, axisSize, inner };

        var result = scope.RecordUnary(LazyNodeType.ReduceMean, "ReduceMean", input, outShape,
            (eng, output) =>
            {
                if (output._gpuBuffer is null && !output.HasPendingGpuData && output.IsContiguous)
                    ReduceMeanForwardFloat(input, output, shape);
                else
                {
                    var staged = new Tensor<float>(outShape);
                    ReduceMeanForwardFloat(input, staged, shape);
                    DirectGpuTensorEngine.CopyResultInto(eng, staged, output);
                }
            },
            ReduceMeanBackwardCompiledFloat, new object[] { shape });
        ReduceMeanForwardFloat(input, result, shape);   // values at trace time, as the generic node had
        return result;
    }

    private static void ReduceMeanForwardFloat(Tensor<float> input, Tensor<float> output, int[] shape)
    {
        int outer = shape[0], axisSize = shape[1], inner = shape[2];
        var x = input.GetCpuBackingForStridedRead(out int xOff)!;
        var o = output.GetCpuBackingForContiguousWrite(out int oOff)!;
        float inv = 1f / axisSize;
        CpuParallelSettings.ParallelForOrSerial(0, outer, (long)outer * axisSize * inner, b =>
        {
            int dst = oOff + b * inner, src = xOff + b * axisSize * inner;
            ScaleCopy(x, src, o, dst, inner, 1f);
            for (int a = 1; a < axisSize; a++) Axpy(1f, x, src + a * inner, o, dst, inner);
            Scale(o, dst, inner, inv);
        }, deterministicSafe: true);
        output.IncrementVersion();
    }

    private static void ReduceMeanBackwardCompiledFloat(
        Tensor<float> gradOutput, Tensor<float>[] inputs, Tensor<float> output, object[] savedState, IEngine engine,
        Dictionary<Tensor<float>, Tensor<float>> grads)
    {
        var input = inputs[0];
        if (!DifferentiableOps.IsGradientRequired(input)) return;
        var shape = (int[])savedState[0];
        int outer = shape[0], axisSize = shape[1], inner = shape[2];
        var dy = ReadableBacking(gradOutput, out int dyOff);
        var target = GradTarget(true, input, grads, engine, distinct: true);
        var dx = target.Array!;
        int dxOff = target.Offset;
        bool overwrite = target.Overwrite;
        float inv = 1f / axisSize;
        CpuParallelSettings.ParallelForOrSerial(0, outer, (long)outer * axisSize * inner, b =>
        {
            int src = dyOff + b * inner, dst = dxOff + b * axisSize * inner;
            for (int a = 0; a < axisSize; a++)
            {
                if (overwrite) ScaleCopy(dy, src, dx, dst + a * inner, inner, inv);
                else Axpy(inv, dy, src, dx, dst + a * inner, inner);
            }
        }, deterministicSafe: true);
        target.Commit(grads, input, engine);
    }
}
