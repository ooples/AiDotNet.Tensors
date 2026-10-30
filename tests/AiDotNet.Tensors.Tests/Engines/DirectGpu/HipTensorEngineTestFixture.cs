// Copyright (c) CrowdTrainer LLC. All rights reserved.
using System;
using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Engines.DirectGpu.HIP;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// A direct-GPU engine on the HIP backend, which the default backend order selects only after OpenCL, so the parity
/// device tests run against HIP's own kernels (#1124).
/// </summary>
public sealed class HipTensorEngineTestFixture : IDirectGpuTensorEngineFixture, IDisposable
{
    private readonly HipBackend? _backend;

    public DirectGpuTensorEngine? Engine { get; }
    public bool IsAvailable => Engine?.IsGpuAvailable == true;

    public HipTensorEngineTestFixture()
    {
        try
        {
            var backend = new HipBackend();
            if (!backend.IsAvailable) { backend.Dispose(); return; }
            _backend = backend;
            Engine = new DirectGpuTensorEngine(new DirectGpuEngine(backend));
        }
        catch (Exception ex) when (ex is DllNotFoundException or PlatformNotSupportedException or TypeInitializationException
            or InvalidOperationException)
        {
            Engine = null;
        }
    }

    public void Dispose()
    {
        Engine?.Dispose();
        _backend?.Dispose();
    }
}