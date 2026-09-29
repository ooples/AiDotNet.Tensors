// Copyright (c) AiDotNet. All rights reserved.

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// The device side of one data holder (a <see cref="VectorBase{T}"/>), shared by every tensor view of it -- the
/// PyTorch storage model: views differ in shape, strides and offset, never in where their bytes live.
/// </summary>
/// <remarks>
/// Before this, each tensor OBJECT carried its own device binding, so two views of one vector could hold different
/// device copies (or one could be bound while the other re-uploaded stale host data). Tensors that cover the whole
/// vector (contiguous, offset 0, full length) read and write this shared state; a strided view keeps a private
/// binding of its own contiguous copy (see <see cref="TensorBase{T}"/>).
/// </remarks>
internal sealed class VectorDeviceState
{
    /// <summary>The device buffer holding this vector's elements, or null when it has no device copy.</summary>
    internal Engines.DirectGpu.IGpuBuffer? Buffer;

    /// <summary>The backend that owns <see cref="Buffer"/>.</summary>
    internal Engines.DirectGpu.IDirectGpuBackend? Backend;

    /// <summary>True while <see cref="Buffer"/> holds the vector's current values.</summary>
    internal bool DeviceValid;

    /// <summary>Complex data stored as split real/imaginary float planes.</summary>
    internal bool IsSplitComplex;

    /// <summary>Int data stored as raw int32 rather than numeric floats.</summary>
    internal bool ContainsRawInt32;
}
