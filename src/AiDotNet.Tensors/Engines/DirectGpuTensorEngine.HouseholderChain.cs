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
        if (input.Rank < 1 || input.Shape[input.Rank - 1] != d)
            throw new ArgumentException($"Expected last dimension {d}, got [{string.Join(",", input._shape)}].", nameof(input));

        double effFloor = normFloor > 0.0 ? normFloor : DefaultHouseholderNormFloor<T>();
        double sqrtFloor = Math.Sqrt(effFloor);
        int rows = d == 0 ? 0 : input.Length / d; // [..., 0] holds no rows
        if (rows == 0 || k == 0 || GraphMode.IsActive || typeof(T) != typeof(float)
            || !(sqrtFloor >= 1.1754943508222875e-38 && sqrtFloor <= float.MaxValue)
            || !TryGetBackend(out var backend) || backend is not IFusedHouseholderChain fused)
        {
            return base.HouseholderChain(input, reflectors, transpose, effFloor);
        }

        try
        {
            using var inB = GetOrAllocateBuffer(backend, input);
            using var refB = GetOrAllocateBuffer(backend, reflectors);
            using var outB = AllocateFullyWrittenOutputBuffer(backend, checked(rows * d));

            fused.HouseholderChainForward(inB.Buffer, refB.Buffer, outB.Buffer, rows, k, d, transpose, effFloor);

            var result = DeferTensorResult<T>(backend, outB.Buffer, rows * d, (int[])input._shape.Clone());
            outB.RelinquishOwnership();

            DifferentiableOps.RecordIfActive<T>(
                "HouseholderChain",
                result,
                new[] { input, reflectors },
                CpuEngine.HouseholderChainBackward<T>,
                savedState: new object[] { transpose, effFloor });

            return result;
        }
        catch (NotSupportedException capabilityGap)
        {
            GpuLaunchProbe.OnFallback("HouseholderChain", capabilityGap);
            return base.HouseholderChain(input, reflectors, transpose, effFloor);
        }
    }

    internal (Tensor<T> GradInput, Tensor<T> GradReflectors)? TryHouseholderChainBackwardGpu<T>(
        Tensor<T> gradOutput,
        Tensor<T> input,
        Tensor<T> reflectors,
        bool transpose,
        double normFloor)
    {
        using var __gpuOp = EnterGpuOp();
        if (typeof(T) != typeof(float) || !TryGetBackend(out var backend) || backend is not IFusedHouseholderChain fused)
            return null;

        int k = reflectors.Shape[0];
        int d = reflectors.Shape[1];
        int rows = input.Length / d;
        double sqrtFloor = Math.Sqrt(normFloor);
        if (rows <= 0 || k <= 0 || d <= 0 || gradOutput.Length != rows * d
            || !(sqrtFloor >= 1.1754943508222875e-38 && sqrtFloor <= float.MaxValue))
        {
            return null;
        }

        try
        {
            using var dyB = GetOrAllocateBuffer(backend, gradOutput);
            using var inB = GetOrAllocateBuffer(backend, input);
            using var refB = GetOrAllocateBuffer(backend, reflectors);
            using var dxB = AllocateFullyWrittenOutputBuffer(backend, checked(rows * d));
            using var dRefB = AllocateFullyWrittenOutputBuffer(backend, checked(k * d));

            fused.HouseholderChainBackward(
                dyB.Buffer, inB.Buffer, refB.Buffer, dxB.Buffer, dRefB.Buffer,
                rows, k, d, transpose, normFloor);

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
