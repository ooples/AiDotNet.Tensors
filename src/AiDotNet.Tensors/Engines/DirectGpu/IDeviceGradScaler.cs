// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>The settings of a device loss scaler (see <see cref="Compilation.LossScalingOptions"/>).</summary>
internal readonly record struct DeviceGradScalerSettings(
    float GrowthFactor, float BackoffFactor, int GrowthInterval, float MinScale, float MaxScale, bool Dynamic);

/// <summary>
/// A loss scaler whose state lives on the device: unscales a fixed set of gradient buffers, detects non-finite values and
/// updates the scale with no host synchronization (<see cref="CUDA.Kernels.CudaGradScalerKernels"/>).
/// </summary>
internal interface IDeviceGradScaler : IDisposable
{
    /// <summary>Whether this scaler's table holds exactly these gradient buffers.</summary>
    bool Covers(IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes);

    /// <summary>Unscales every gradient, records a non-finite result, updates the scale and writes it into
    /// <paramref name="seed"/>. Two launches; capture-safe.</summary>
    void UnscaleAndUpdate(IGpuBuffer seed, int seedLength);

    /// <summary>Writes the current scale into <paramref name="seed"/>. One launch; capture-safe.</summary>
    void WriteSeed(IGpuBuffer seed, int seedLength);

    /// <summary>Reads the state (one synchronizing download).</summary>
    Compilation.LossScaleState Read();

    /// <summary>Uploads <paramref name="state"/> (not during capture).</summary>
    void Write(Compilation.LossScaleState state);
}

/// <summary>A backend that can host a <see cref="IDeviceGradScaler"/>.</summary>
[BackendSpecificRoute("CUDA device loss scaling (unscale, non-finite check and scale update as two launches with no host read); "
    + "every backend unscales natively with Scale and the plan's finiteness guard decides the step",
    nameof(IDirectGpuBackend.Scale))]
internal interface IDeviceGradScalerFactory
{
    /// <summary>A scaler over these gradient buffers, or null when the backend cannot run its kernels.</summary>
    IDeviceGradScaler? CreateGradScaler(IReadOnlyList<IGpuBuffer> gradients, IReadOnlyList<int> sizes,
        DeviceGradScalerSettings settings, Compilation.LossScaleState initial);
}

/// <summary>
/// A backend whose FP32 GEMMs run under matmul autocast: every backend. CUDA multiplies on cuBLAS's 16-bit tensor-core
/// kernels; the others round the operands on the device (<see cref="GemmOperandAutocast"/>) and run their FP32 GEMM.
/// </summary>
internal interface IGemmAutocastBackend
{
    /// <summary>
    /// Until the scope is disposed, FP32 GEMMs take operands rounded to <paramref name="precision"/> and accumulate FP32.
    /// </summary>
    IDisposable EnterGemmAutocast(Gpu.PrecisionMode precision, Gpu.MatMulAutocastMode mode = Gpu.MatMulAutocastMode.CastOperands);
}
