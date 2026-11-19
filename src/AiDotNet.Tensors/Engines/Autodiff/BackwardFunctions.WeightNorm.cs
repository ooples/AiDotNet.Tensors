// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System.Collections.Generic;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

internal static partial class BackwardFunctions<T>
{
    internal static void WeightNormBackward(Tensor<T> gradOutput, Tensor<T>[] inputs, Tensor<T> output,
        object[] savedState, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads)
    {
        var direction = inputs[0];
        var gain = inputs[1];
        double epsilon = (double)savedState[0];
        // The native VJP is first order. A higher-order request retains the exact differentiable primitive formula,
        // including epsilon inside the squared sum. In its own method, so first-order calls allocate no closure.
        if (CreatesGraph && WeightNormCreateGraph(gradOutput, inputs, engine, grads, direction, gain, epsilon))
            return;

        bool needDirection = DifferentiableOps.IsGradientRequired(direction);
        bool needGain = DifferentiableOps.IsGradientRequired(gain);
        if (!needDirection && !needGain) return;
        engine.TryGet<DirectGpuTensorEngine>(out var gpu);

        static Tensor<T> Target(Tensor<T> input, IEngine engine, Dictionary<Tensor<T>, Tensor<T>> grads,
            DirectGpuTensorEngine? gpu, out bool add)
        {
            // A compiled resident action has indexed gradient slots; the ordinary
            // tape has only its dictionary. Do not allocate a contribution beside
            // a reusable accumulator in either case.
            var target = DifferentiableOps.PeekDirectDeviceGradTarget(grads, input, engine, out add);
            if (target is not null) return target;
            if (grads.TryGetValue(input, out target))
            {
                add = DifferentiableOps.GradWriteGeneration == 0 || DifferentiableOps.IsWrittenThisStep(target);
                return target;
            }
            add = false;
            return gpu is not null ? gpu.CreateWeightNormGradient(input)
                : TensorAllocator.RentUninitialized<T>(input._shape);
        }

        bool addDirection = false, addGain = false;
        var gradDirection = needDirection ? Target(direction, engine, grads, gpu, out addDirection) : null;
        Tensor<T>? gradGain;
        if (needGain && needDirection && ReferenceEquals(direction, gain))
        {
            gradGain = gradDirection;
            addGain = true; // complete dV is the tied accumulator's first contribution
        }
        else gradGain = needGain ? Target(gain, engine, grads, gpu, out addGain) : null;
        engine.TensorWeightNormBackwardInto(gradOutput, direction, gain, gradDirection, gradGain, epsilon, addDirection, addGain);
        if (gradDirection is not null) DifferentiableOps.CommitDirectGradWrite(grads, direction, gradDirection);
        if (gradGain is not null) DifferentiableOps.CommitDirectGradWrite(grads, gain, gradGain);
    }

    private static bool WeightNormCreateGraph(Tensor<T> gradOutput, Tensor<T>[] inputs, IEngine engine,
        Dictionary<Tensor<T>, Tensor<T>> grads, Tensor<T> direction, Tensor<T> gain, double epsilon)
        => TryCompositeBackwardFor(gradOutput, inputs, engine, grads, FirstTwoInputs, e =>
        {
            var (rows, columns) = CpuEngine.ValidateWeightNorm(direction, gain, epsilon);
            var v = e.Reshape(direction, new[] { rows, columns });
            var g = e.Reshape(gain, new[] { rows, 1 });
            var norm = e.TensorSqrt(e.TensorAddScalar(
                e.ReduceSum(e.TensorMultiply(v, v), new[] { 1 }, keepDims: true), Num(epsilon)));
            var scale = e.TensorDivide(g, norm);
            var weights = e.TensorMultiply(v, Bcast(e, scale, new[] { rows, columns }));
            return e.Reshape(weights, direction._shape);
        });

}
