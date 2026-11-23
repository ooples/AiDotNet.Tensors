// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Threading.Tasks;

namespace AiDotNet.Tensors.Engines;

/// <summary>Creates the best <see cref="IAsyncEngine"/> this process can run.</summary>
public static class AsyncEngine
{
    /// <summary>
    /// Returns an engine on WebGPU when the platform offers it (a browser with WebGPU, or a WebGPU host set up for this
    /// process), otherwise a <see cref="CpuEngine"/>. Starting the device is itself awaited, so this never blocks.
    /// </summary>
    /// <remarks>Dispose the result when it is <see cref="IDisposable"/>: a WebGPU engine owns its backend.</remarks>
    public static async Task<IAsyncEngine> CreateAsync()
    {
#if NET7_0_OR_GREATER
        if (DirectGpu.WebGpu.WebGpuNativeBindings.IsWebGpuSupported())
        {
            var backend = new DirectGpu.WebGpu.WebGpuAwaitableBackend();
            bool started;
            try
            {
                started = await backend.InitializeAsync().ConfigureAwait(false);
            }
            catch
            {
                backend.Dispose();
                throw;
            }
            if (started) return new AwaitableGpuTensorEngine(backend, ownsBackend: true);
            backend.Dispose();
        }
#endif
        await Task.CompletedTask.ConfigureAwait(false);
        return new CpuEngine();
    }
}
