namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Optional capability: refresh a device buffer from the host without the copy waiting its turn on the compute stream.
/// The host data is copied on a separate copy stream into a staging buffer while earlier work still runs, and the
/// compute stream only copies device-to-device from it. A compiled training step refreshes its input this way between
/// graph replays, where an in-stream host copy left the device idle.
/// </summary>
[BackendSpecificRoute("CUDA copy-stream upload overlapped with the running step; every backend refreshes a buffer from the host natively",
    nameof(IDirectGpuBackend.UploadBufferInPlace))]
internal interface IOverlappedUploadBackend
{
    /// <summary>
    /// Makes <paramref name="destination"/> hold <paramref name="data"/> for all work queued on the compute stream
    /// after this call, exactly as <see cref="IDirectGpuBackend.UploadBufferInPlace"/> would. False -- nothing done --
    /// when the backend cannot overlap the copy now (e.g. during stream capture); the caller then uploads in place.
    /// </summary>
    bool TryUploadOverlapped(float[] data, IGpuBuffer destination);
}
