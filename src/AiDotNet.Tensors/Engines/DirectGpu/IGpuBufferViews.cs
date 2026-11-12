namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A view that names the buffer it addresses. A backend may build a new view object for every request, so a cache that
/// must recognize the same memory again keys on the root buffer (see <see cref="GpuBufferViewRoot"/>), not the view.
/// </summary>
internal interface IGpuBufferViewSource
{
    /// <summary>The buffer this view addresses (itself possibly a view).</summary>
    IGpuBuffer ViewParent { get; }
}

internal static class GpuBufferViewRoot
{
    /// <summary>The buffer that owns <paramref name="buffer"/>'s memory: itself, or the root of its view chain.</summary>
    public static IGpuBuffer Of(IGpuBuffer buffer)
    {
        while (buffer is IGpuBufferViewSource view) buffer = view.ViewParent;
        return buffer;
    }
}

/// <summary>
/// Optional backend capability: a non-owning view of a contiguous sub-range of a device buffer.
/// </summary>
/// <remarks>
/// Model parameters are views (shared storage + offset) into one flat <c>ParameterBuffer</c>. Placing that flat
/// array on the device as ONE buffer and addressing each parameter through a view lets on-device optimizers update
/// parameters in place with a single upload for the whole model, instead of refusing every offset view.
/// A view never owns memory: disposing it is a no-op and it keeps its parent alive.
/// </remarks>
internal interface IGpuBufferViews
{
    /// <summary>A view of <paramref name="elementCount"/> floats starting <paramref name="elementOffset"/> floats into <paramref name="parent"/>, or null.</summary>
    IGpuBuffer? TryCreateView(IGpuBuffer parent, int elementOffset, int elementCount);

    /// <summary>True when a view at any float offset is always granted and costs nothing to make (a device-pointer offset),
    /// so a caller may make views per launch, even inside a captured graph; false when views can be refused for their
    /// offset's alignment or allocate.</summary>
    bool ViewsAreFreeAtAnyOffset { get; }
}
