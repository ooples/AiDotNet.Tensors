using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    // The create-graph path in its own method: its lambda captures the operands, and a closure declared in the backward
    // itself would be allocated on every first-order call too.
    private static bool GroupedConv2DBackwardCreateGraph(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, DirectGpu.GroupedConv2DGeometry geometry)
    {
        var input = inputs[0]; var kernel = inputs[1];
        return TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstTwoInputs,
            current => GroupedConvolution.SplitForward(current, input, kernel, geometry));
    }

    internal static void GroupedConv2DBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var input = inputs[0]; var kernel = inputs[1];
        var geometry = GroupedConvolution.Validate(input, kernel, (int[])savedState[0], (int[])savedState[1],
            (int[])savedState[2], (int)savedState[3]);
        // Higher-order derivatives retain the original differentiable split expression.
        if (CreatesGraph && GroupedConv2DBackwardCreateGraph(gradOutput, inputs, engine, grads, geometry)) return;
        bool needInput = DifferentiableOps.IsGradientRequired(input), needKernel = DifferentiableOps.IsGradientRequired(kernel);
        if (!needInput && !needKernel) return;
        engine.TryGet<DirectGpuTensorEngine>(out var gpu);
        Tensor<T> Target(Tensor<T> source, out bool add)
        {
            var target = DifferentiableOps.PeekDirectDeviceGradTarget(grads, source, engine, out add);
            if (target is not null) return target;
            if (grads.TryGetValue(source, out target))
            {
                add = DifferentiableOps.GradWriteGeneration == 0 || DifferentiableOps.IsWrittenThisStep(target);
                return target;
            }
            add = false;
            // Same-shape deferred gradient creation; no source payload is read.
            return gpu is not null ? gpu.CreateWeightNormGradient(source) : TensorAllocator.RentUninitialized<T>(source._shape);
        }
        bool addInput = false, addKernel = false;
        var dx = needInput ? Target(input, out addInput) : null;
        var dw = needKernel ? needInput && ReferenceEquals(input, kernel) ? dx : Target(kernel, out addKernel) : null;
        if (needKernel && needInput && ReferenceEquals(input, kernel)) addKernel = true;
        GroupedConvolution.BackwardInto(engine, gradOutput, input, kernel, dx, dw, geometry, addInput, addKernel);
        if (dx is not null) DifferentiableOps.CommitDirectGradWrite(grads, input, dx);
        if (dw is not null) DifferentiableOps.CommitDirectGradWrite(grads, kernel, dw);
    }
}
