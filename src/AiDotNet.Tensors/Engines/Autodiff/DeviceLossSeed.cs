using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Autodiff;

/// <summary>
/// The dL/dL = 1 seed of a backward pass, created where the loss lives.
/// </summary>
/// <remarks>
/// <para>Every backward path (the tape walk, the compiled graph, the compiled walker and the delegate chain) seeded
/// with a host ones tensor. For a loss on the GPU that host seed crossed the host/device boundary on every step,
/// even when cached: step-end release frees the device copy the backward makes of it. A device-resident loss now
/// gets a device-filled seed; anything else keeps the host seed exactly as before.</para>
/// </remarks>
internal static class DeviceLossSeed
{
    internal static bool TryCreate<T>(IEngine? engine, Tensor<T> loss, out Tensor<T> seed)
    {
        if (engine is DirectGpuTensorEngine gpu && gpu.TryCreateDeviceSeed(loss, out seed)) return true;
        seed = null!;
        return false;
    }
}
