// Copyright (c) AiDotNet. All rights reserved.

using AiDotNet.Tensors.Engines;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Owns one direct-GPU engine per test class so theory rows share the expensive native backend
/// initialization while retaining isolation between unrelated test families.
/// </summary>
public sealed class DirectGpuTensorEngineTestFixture : IDisposable
{
    public DirectGpuTensorEngine? Engine { get; }
    public Exception? InitializationException { get; }
    public bool IsAvailable => Engine?.IsGpuAvailable == true;

    public DirectGpuTensorEngineTestFixture()
    {
        try
        {
            Engine = new DirectGpuTensorEngine();
        }
        catch (PlatformNotSupportedException ex)
        {
            InitializationException = ex;
        }
        catch (DllNotFoundException ex)
        {
            InitializationException = ex;
        }
    }

    public void Dispose()
    {
        Engine?.Dispose();
    }
}
