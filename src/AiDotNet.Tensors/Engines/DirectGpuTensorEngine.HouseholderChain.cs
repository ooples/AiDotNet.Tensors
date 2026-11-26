// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;
using AiDotNet.Tensors.Engines.Autodiff;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

public partial class DirectGpuTensorEngine
{
    public override Tensor<T> HouseholderChain<T>(
        Tensor<T> input,
        Tensor<T> reflectors,
        bool transpose = false,
        double normFloor = 0.0)
    {
        using var __gpuOp = EnterGpuOp();
        if (input is null) throw new ArgumentNullException(nameof(input));
        if (reflectors is null) throw new ArgumentNullException(nameof(reflectors));
        if (reflectors.Rank != 2)
            throw new ArgumentException($"Reflectors must be rank-2 [K, D], got rank {reflectors.Rank}.", nameof(reflectors));

        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        if (k <= 0 || d <= 0)
            throw new ArgumentException("Reflectors shape dimensions must be positive.", nameof(reflectors));
        if (input.Rank < 1 || input.Shape[input.Rank - 1] != d)
            throw new ArgumentException($"Expected last dimension {d}, got [{string.Join(",", input._shape)}].", nameof(input));

        double effFloor = normFloor > 0.0 ? normFloor : DefaultHouseholderNormFloor<T>();
        int rows = input.Length / d;
        if (rows == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !TryGetBackend(out var backend) || backend is not IFusedHouseholderChain fused)
        {
            return base.HouseholderChain(input, reflectors, transpose, effFloor);
        }

        float floorF32 = (float)Math.Max(effFloor, 1.1754943508222875e-38);
        try
        {
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcRef = reflectors.IsContiguous ? reflectors : reflectors.Contiguous();
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var refB = GetOrAllocateBuffer(backend, srcRef);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, checked(rows * d));

            fused.HouseholderChainForward(inB.Buffer, refB.Buffer, outB.Buffer, rows, k, d, transpose, floorF32);

            var result = DeferTensorResult<T>(backend, outB.Buffer, rows * d, (int[])input._shape.Clone());
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "HouseholderChain",
                result,
                new[] { input, reflectors },
                static (gradOutput, inputs, output, savedState, engine, grads) =>
                {
                    bool tr = (bool)savedState[0];
                    double fl = (double)savedState[1];
                    if (!DifferentiableOps._isBackwardCreateGraph
                        && engine.TryGet<DirectGpuTensorEngine>(out var gpu)
                        && gpu.TryHouseholderChainBackwardGpu(gradOutput, inputs[0], inputs[1], tr, fl) is { } gpuGrads)
                    {
                        DifferentiableOps.AccumulateGrad(grads, inputs[0], gpuGrads.GradInput, engine);
                        DifferentiableOps.AccumulateGrad(grads, inputs[1], gpuGrads.GradReflectors, engine);
                        return;
                    }
                    CpuEngine.HouseholderChainBackward(gradOutput, inputs, output, savedState, engine, grads);
                },
                savedState: new object[] { transpose, effFloor });

            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("HouseholderChain", capabilityGap);
            return base.HouseholderChain(input, reflectors, transpose, effFloor);
        }
    }

    private (Tensor<T> GradInput, Tensor<T> GradReflectors)? TryHouseholderChainBackwardGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> reflectors,
        bool transpose,
        double normFloor)
    {
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedHouseholderChain fused)
            return null;

        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        int rows = input.Length / d;
        if (rows <= 0 || k <= 0 || d <= 0 || gradOutput.Length != rows * d)
            return null;

        float floorF32 = (float)Math.Max(normFloor, 1.1754943508222875e-38);
        try
        {
            var srcGrad = gradOutput.IsContiguous ? gradOutput : gradOutput.Contiguous();
            var srcIn = input.IsContiguous ? input : input.Contiguous();
            var srcRef = reflectors.IsContiguous ? reflectors : reflectors.Contiguous();

            using var dyB = GetOrAllocateBuffer(backend, srcGrad);
            using var inB = GetOrAllocateBuffer(backend, srcIn);
            using var refB = GetOrAllocateBuffer(backend, srcRef);
            using var dxB = AllocateFullyWrittenOutputBuffer(backend, checked(rows * d));
            using var dRefB = AllocateFullyWrittenOutputBuffer(backend, checked(k * d));

            fused.HouseholderChainBackward(
                dyB.Buffer, inB.Buffer, refB.Buffer, dxB.Buffer, dRefB.Buffer,
                rows, k, d, transpose, floorF32);

            var gradInput = DeferTensorResult<T>(backend, dxB.Buffer, rows * d, (int[])input._shape.Clone());
            dxB.RelinquishOwnership();
            var gradReflectors = DeferTensorResult<T>(backend, dRefB.Buffer, k * d, (int[])reflectors._shape.Clone());
            dRefB.RelinquishOwnership();
            return (gradInput, gradReflectors);
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("HouseholderChainBackward", capabilityGap);
            return null;
        }
    }
}
