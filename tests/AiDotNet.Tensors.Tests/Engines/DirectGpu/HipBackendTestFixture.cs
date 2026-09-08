// Copyright (c) AiDotNet. All rights reserved.

#if NET6_0_OR_GREATER

using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Owns one HIP backend per test class. xUnit creates a new test-class instance for every
/// theory row, while HIP backend construction compiles the complete kernel inventory.
/// </summary>
public sealed class HipBackendTestFixture : IDisposable
{
    public HipBackend Backend { get; }
    public bool IsAvailable => Backend.IsAvailable;

    public HipBackendTestFixture()
    {
        var backend = new HipBackend();
        if (backend.InitializationState == GpuBackendInitializationState.Failed)
        {
            Exception cause = backend.InitializationException ?? new InvalidOperationException(
                "HIP reported failed initialization without preserving its cause.");
            backend.Dispose();
            throw new InvalidOperationException("HIP was detected, but backend initialization failed.", cause);
        }

        if (backend.IsAvailable !=
            (backend.InitializationState == GpuBackendInitializationState.Succeeded))
        {
            backend.Dispose();
            throw new InvalidOperationException(
                "HIP availability and typed initialization state are inconsistent.");
        }

        Backend = backend;
    }

    public void Dispose()
    {
        Backend.Dispose();
    }
}

#endif
