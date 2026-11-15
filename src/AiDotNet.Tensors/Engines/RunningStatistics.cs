// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines.Compilation;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines;

/// <summary>
/// The exponential-moving-average update of a normalization layer's running statistic:
/// <c>running = keep * running + batchScale * batch</c>, in place.
/// </summary>
/// <remarks>
/// <para>One launch on a DirectGpu backend (<see cref="DirectGpu.IDirectGpuBackend.AddScaled"/> into the running
/// buffer) where the scale / scale / add composition took three, and allocated the scaled batch statistic. A BatchNorm
/// updates two statistics per training step, so a DenseNet-121 step issued 726 of these launches (8.5 ms on an
/// RTX 1000 Ada at batch 64); fused it issues 242.</para>
/// <para>Under a compiled plan's trace (<see cref="GraphMode"/>) the update records one in-place node, replayed every
/// step, exactly as the composed in-place ops were recorded.</para>
/// </remarks>
internal static class RunningStatistics
{
    internal static void UpdateInPlace<T>(IEngine engine, Tensor<T> running, Tensor<T> batch, T keep, T batchScale)
    {
        if (engine is null) throw new ArgumentNullException(nameof(engine));
        if (running is null) throw new ArgumentNullException(nameof(running));
        if (batch is null) throw new ArgumentNullException(nameof(batch));
        if (GraphMode.IsActive && GraphMode.Current is { } scope)
        {
            scope.BindEngineIfUnset(engine);
            scope.RecordInPlace(LazyNodeType.Custom, "RunningStatisticUpdate", running, new[] { batch },
                (eng, _) => UpdateInPlace(eng, running, batch, keep, batchScale));
            return;
        }
        if (engine.TryGet<DirectGpuTensorEngine>(out var gpu) && gpu.TryRunningStatisticUpdate(running, batch, keep, batchScale))
            return;
        engine.TensorMultiplyScalarInPlace(running, keep);
        engine.TensorAddInPlace(running, engine.TensorMultiplyScalar(batch, batchScale));
    }
}
