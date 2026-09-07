// Copyright (c) AiDotNet. All rights reserved.

#if NET6_0_OR_GREATER

using AiDotNet.Tensors.Engines.DirectGpu.HIP;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Owns one HIP backend per test class. xUnit creates a new test-class instance for every
/// theory row, while HIP backend construction compiles the complete kernel inventory.
/// </summary>
public sealed class HipBackendTestFixture : IDisposable
{
    public HipBackend? Backend { get; }
    public Exception? InitializationException { get; }
    public bool IsAvailable => Backend?.IsAvailable == true;

    public HipBackendTestFixture()
    {
        try
        {
            Backend = new HipBackend();
        }
        catch (Exception ex)
        {
            InitializationException = ex;
        }
    }

    public void Dispose()
    {
        Backend?.Dispose();
    }
}

#endif
