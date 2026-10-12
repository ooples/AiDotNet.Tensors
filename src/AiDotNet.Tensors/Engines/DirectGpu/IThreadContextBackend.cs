// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Making the backend's device current on the calling thread before it issues work (#1154). CUDA contexts and HIP
/// devices are bound per thread, so an op driven from a thread other than the one that created the backend would
/// otherwise run against the wrong (or no) device.
/// </summary>
internal interface IThreadContextBackend
{
    /// <summary>Binds this backend's device to the calling thread; cheap when it already is.</summary>
    void EnsureContextCurrent();
}