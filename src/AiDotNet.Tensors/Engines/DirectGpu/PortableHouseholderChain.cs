// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using K = AiDotNet.Tensors.Engines.DirectGpu.BatchExecutionCudaKernels;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Portable device implementation of <see cref="IFusedHouseholderChain"/> over the shared <c>hh_p_*</c> batch kernels,
/// executed via <see cref="IFusedLstmSequenceTraining.StatefulLstmLauncher"/> on HIP, Metal, OpenCL, Vulkan, WebGPU,
/// and as CUDA's fallback when <c>numReflectors * dim</c> exceeds single-block shared SRAM.
/// </summary>
internal static class PortableHouseholderChain
{
    private static void Validate(int rows, int numReflectors, int dim, float normFloor)
    {
        if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
        if (numReflectors <= 0) throw new ArgumentOutOfRangeException(nameof(numReflectors));
        if (dim <= 0) throw new ArgumentOutOfRangeException(nameof(dim));
        if (!(normFloor > 0f)) throw new ArgumentOutOfRangeException(nameof(normFloor));
        _ = checked((long)rows * dim + (long)numReflectors * dim);
    }

    public static void Forward(
        IDirectGpuBackend backend,
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer output,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        float normFloor)
    {
        Validate(rows, numReflectors, dim, normFloor);
        if (rows == 0) return;
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        float sqrtFloor = MathF.Sqrt(normFloor);
        int sqrtFloorBits = BitConverter.SingleToInt32Bits(sqrtFloor);
        using var uHat = backend.AllocateBuffer(checked(numReflectors * dim));
        using var meta = backend.AllocateBuffer(checked(2 * numReflectors));
        launch(K.HouseholderNorm, numReflectors, reflectors, uHat, meta, numReflectors, dim, sqrtFloorBits);
        launch(K.HouseholderForward, rows, input, uHat, output, rows, numReflectors, dim, transpose ? 1 : 0);
    }

    public static void Backward(
        IDirectGpuBackend backend,
        IGpuBuffer gradOutput,
        IGpuBuffer input,
        IGpuBuffer reflectors,
        IGpuBuffer gradInput,
        IGpuBuffer gradReflectors,
        int rows,
        int numReflectors,
        int dim,
        bool transpose,
        float normFloor)
    {
        Validate(rows, numReflectors, dim, normFloor);
        if (rows == 0)
        {
            backend.Fill(gradReflectors, 0f, checked(numReflectors * dim));
            return;
        }
        var launch = ((IFusedLstmSequenceTraining)backend).StatefulLstmLauncher;
        float sqrtFloor = MathF.Sqrt(normFloor);
        int sqrtFloorBits = BitConverter.SingleToInt32Bits(sqrtFloor);
        int trans = transpose ? 1 : 0;
        using var uHat = backend.AllocateBuffer(checked(numReflectors * dim));
        using var meta = backend.AllocateBuffer(checked(2 * numReflectors));
        using var pqBuf = backend.AllocateBuffer(checked(rows * 2 * numReflectors));
        using var duHat = backend.AllocateBuffer(checked(numReflectors * dim));
        launch(K.HouseholderNorm, numReflectors, reflectors, uHat, meta, numReflectors, dim, sqrtFloorBits);
        launch(K.HouseholderBackwardInput, rows, gradOutput, input, uHat, gradInput, pqBuf, rows, numReflectors, dim, trans);
        launch(K.HouseholderBackwardReflectorsRaw, dim, gradOutput, input, uHat, pqBuf, duHat, rows, numReflectors, dim, trans);
        launch(K.HouseholderBackwardReflectorsNorm, numReflectors, duHat, uHat, meta, gradReflectors, numReflectors, dim);
    }
}
