// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Releasing a buffer that work already queued on the backend may still read (#1154). Each backend knows whether its
/// queue retains submitted resources, so the engine no longer type-tests for CUDA and HIP to decide.
/// </summary>
internal interface IQueuedBufferRelease
{
    /// <summary>Disposes <paramref name="buffer"/> once work queued before this call can no longer read it.</summary>
    void ReleaseAfterQueuedWork(IGpuBuffer buffer);
}