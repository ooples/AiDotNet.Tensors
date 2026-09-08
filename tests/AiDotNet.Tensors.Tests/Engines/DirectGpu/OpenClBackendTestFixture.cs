// Copyright (c) AiDotNet. All rights reserved.

#if NET6_0_OR_GREATER
#nullable enable

using System;
using AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Owns one compiled OpenCL backend per test class. xUnit creates a new test-class instance for
/// every theory row, so constructor-owned backends otherwise multiply native contexts and kernel
/// compilation by the number of cases. Reusing only within one class preserves family isolation.
/// </summary>
public sealed class OpenClBackendTestFixture : IDisposable
{
    public OpenClBackendTestFixture()
    {
        var backend = new OpenClBackend();
        if (backend.InitializationError is string initializationError)
        {
            backend.Dispose();
            throw new InvalidOperationException(
                "OpenCL was detected, but backend initialization failed.",
                new InvalidOperationException(initializationError));
        }

        Backend = backend;
    }

    public OpenClBackend Backend { get; }
    public bool IsAvailable => Backend.IsAvailable;

    public void Dispose()
    {
        Backend.Dispose();
    }
}

#endif
