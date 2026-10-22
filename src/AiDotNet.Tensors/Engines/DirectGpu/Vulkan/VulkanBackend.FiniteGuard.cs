// Copyright (c) AiDotNet. All rights reserved.
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

// #1521: the multi-tensor sum of squares and the guarded optimizer updates on the device, so a compiled step's
// finiteness verdict needs no host read (VulkanFiniteGuardKernels).
public sealed unsafe partial class VulkanBackend
{
    private void MultiTensorSumOfSquaresOnDevice(IReadOnlyList<IGpuBuffer> tensors, IReadOnlyList<int> sizes,
        IGpuBuffer sumOfSquares)
    {
        MultiTensorArgs.ValidateSumBuffer(sumOfSquares);
        MultiTensorArgs.Validate(tensors, sizes);
        Fill(sumOfSquares, 0f, 2);
        for (int t = 0; t < tensors.Count; t++)
            GlslNaryOp(VulkanFiniteGuardKernels.Accumulate, new[] { tensors[t], sumOfSquares },
                MultiTensorArgs.ReductionGroupSize, new[] { (uint)sizes[t] });
        FinishSumOfSquaresDouble(sumOfSquares);
    }

    // Turns the float sum and non-finite mark in the buffer's two slots into the double they stand for.
    internal void FinishSumOfSquaresDouble(IGpuBuffer accumulator)
        => GlslNaryOp(VulkanFiniteGuardKernels.Finish, new[] { accumulator }, 1, System.Array.Empty<uint>());

    private void AdamGuardedOnDevice(bool decoupledWeightDecay,
        IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<IGpuBuffer> firstMoments, IReadOnlyList<IGpuBuffer> secondMoments,
        IReadOnlyList<int> sizes, float learningRate, float beta1, float beta2, float epsilon,
        float weightDecay, int step, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateAdam(parameters, gradients, firstMoments, secondMoments, sizes, epsilon, step);
        // The push constants AdamUpdate / AdamWUpdate pass (operation 0 = Adam, 1 = AdamW).
        for (int t = 0; t < parameters.Count; t++)
            DispatchOptimizer(VulkanFiniteGuardKernels.AdamGuarded, sizes[t],
                new[] { parameters[t], gradients[t], firstMoments[t], secondMoments[t], finiteGuard },
                (uint)sizes[t], decoupledWeightDecay ? 1u : 0u, (uint)step, FloatBits(learningRate), FloatBits(beta1),
                FloatBits(beta2), FloatBits(epsilon), FloatBits(weightDecay));
    }

    private void SgdGuardedOnDevice(IReadOnlyList<IGpuBuffer> parameters, IReadOnlyList<IGpuBuffer> gradients,
        IReadOnlyList<int> sizes, float learningRate, float weightDecay, IGpuBuffer finiteGuard)
    {
        PerTensorMultiTensorOptimizer.ValidateGuard(finiteGuard);
        PerTensorMultiTensorOptimizer.ValidateSgd(parameters, gradients, sizes);
        for (int t = 0; t < parameters.Count; t++)
            DispatchOptimizer(VulkanFiniteGuardKernels.SgdGuarded, sizes[t], new[] { parameters[t], gradients[t], finiteGuard },
                (uint)sizes[t], 0, FloatBits(learningRate), FloatBits(weightDecay));
    }
}
