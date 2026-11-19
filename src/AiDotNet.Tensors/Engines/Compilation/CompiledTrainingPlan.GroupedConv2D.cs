using System.Collections.Generic;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Compilation;

internal sealed partial class CompiledTrainingPlan<T>
{
    private static Action<IEngine>? BuildGroupedConv2DBackward(CompiledStep<T> step,
        Dictionary<Tensor<T>, Tensor<T>> gradMap, Dictionary<Tensor<T>, int> writers, HashSet<Tensor<T>>? required)
    {
        if (step.OpType != OpType.GroupedConv2D || step.Inputs.Length != 2
            || step.SavedState is not { Length: 4 } saved || saved[0] is not int[] stride
            || saved[1] is not int[] padding || saved[2] is not int[] dilation || saved[3] is not int groups
            || !gradMap.TryGetValue(step.OutputBuffer, out var gradOutput)) return null;
        var input = step.Inputs[0]; var kernel = step.Inputs[1];
        var geometry = GroupedConvolution.Validate(input, kernel, stride, padding, dilation, groups);
        Tensor<T>? dx = null, dw = null;
        if ((required is null || required.Contains(input)) && !gradMap.TryGetValue(input, out dx)) return null;
        if ((required is null || required.Contains(kernel)) && !gradMap.TryGetValue(kernel, out dw)) return null;
        bool tied = ReferenceEquals(input, kernel);
        bool sharedInput = tied || writers.TryGetValue(input, out int ix) && ix > 1;
        bool sharedKernel = tied || writers.TryGetValue(kernel, out int iw) && iw > 1;
        return engine =>
        {
            // All binding happens after pooling/liveness decisions and at replay.
            MaterializeIfGradPlaceholder(gradOutput);
            if (dx is not null) MaterializeIfGradPlaceholder(dx);
            if (dw is not null) MaterializeIfGradPlaceholder(dw);
            bool addInput = dx is not null && AccumulatesInto(dx, sharedInput);
            bool addKernel = dw is not null && (tied && dx is not null || AccumulatesInto(dw, sharedKernel));
            GroupedConvolution.BackwardInto(engine, gradOutput, input, kernel, dx, dw, geometry, addInput, addKernel);
            if (dx is not null) input.Grad = dx;
            if (dw is not null) kernel.Grad = dw;
        };
    }
}
