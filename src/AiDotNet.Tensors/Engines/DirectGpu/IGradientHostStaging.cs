namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A page-locked host area for moving a step's gradients between the device and the host in one batch: every
/// download is queued on the compute stream (after the backward that wrote the gradient) and the host waits once,
/// and every upload is queued on the compute stream (before the update that reads it) with no host wait. Copies
/// from page-locked memory run at the bus rate; per-buffer synchronous copies from pageable arrays (one stream wait
/// and one staging copy per gradient) cost a data-parallel step most of its exchange.
/// </summary>
/// <remarks>The host area must not be written while uploads from it may still be in flight: the next
/// <see cref="WaitForTransfers"/> is the point after which it is free again.</remarks>
internal interface IGradientHostStaging : System.IDisposable
{
    /// <summary>The page-locked host floats (exactly the requested length).</summary>
    System.Span<float> Host { get; }

    /// <summary>Queues a copy of the first <paramref name="count"/> floats of <paramref name="buffer"/> into
    /// <see cref="Host"/> at <paramref name="offset"/>, after the work already queued on the compute stream.</summary>
    void EnqueueDownload(IGpuBuffer buffer, int offset, int count);

    /// <summary>Queues a copy of <paramref name="count"/> floats of <see cref="Host"/> at <paramref name="offset"/>
    /// into the start of <paramref name="buffer"/>, ordered before the work queued on the compute stream after it.</summary>
    void EnqueueUpload(IGpuBuffer buffer, int offset, int count);

    /// <summary>Waits until every queued transfer has finished (downloads are readable, the area is free to write).</summary>
    void WaitForTransfers();
}

/// <summary>A backend that can create <see cref="IGradientHostStaging"/> areas.</summary>
internal interface IGradientHostStagingSource
{
    /// <summary>A page-locked staging area for <paramref name="floats"/> floats, or null where the backend has none
    /// (callers then use the per-buffer download and upload).</summary>
    IGradientHostStaging? TryCreateGradientHostStaging(int floats);
}
