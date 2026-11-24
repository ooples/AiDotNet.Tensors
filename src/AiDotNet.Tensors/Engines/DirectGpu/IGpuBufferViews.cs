namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// A view that names the buffer it addresses. A backend may build a new view object for every request, so a cache that
/// must recognize the same memory again keys on the root buffer (see <see cref="GpuBufferViewRoot"/>), not the view.
/// </summary>
internal interface IGpuBufferViewSource
{
    /// <summary>The buffer this view addresses (itself possibly a view), or null for an allocation owner.</summary>
    IGpuBuffer? ViewParent { get; }
}

internal static class GpuBufferViewRoot
{
    /// <summary>The buffer that owns <paramref name="buffer"/>'s memory: itself, or the root of its view chain.</summary>
    public static IGpuBuffer Of(IGpuBuffer buffer)
    {
        while (buffer is IGpuBufferViewSource { ViewParent: { } parent }) buffer = parent;
        return buffer;
    }

    /// <summary>
    /// Whether <paramref name="root"/> still holds device memory. A native handle says so on CUDA, HIP, OpenCL,
    /// Vulkan and Metal; a WebGPU buffer lives on the JavaScript side and its handle is always zero, so its own
    /// validity is asked instead.
    /// </summary>
    public static bool IsLive(IGpuBuffer root)
    {
#if NET7_0_OR_GREATER
        if (root is WebGpu.WebGpuBuffer web) return web.IsValid;
#endif
        return root.Handle != System.IntPtr.Zero;
    }

    /// <summary>Whether two roots are the same device memory: one object, or one non-zero native handle.</summary>
    public static bool Same(IGpuBuffer left, IGpuBuffer right)
        => ReferenceEquals(left, right) || (left.Handle != System.IntPtr.Zero && left.Handle == right.Handle);

    /// <summary>
    /// Whether <paramref name="buffer"/>'s memory is live: its root has a native handle, or is a WebGPU buffer (whose
    /// handles live on the JavaScript side, so <see cref="IGpuBuffer.Handle"/> is always zero) that is still valid.
    /// </summary>
    public static bool IsBacked(IGpuBuffer buffer)
    {
        var root = Of(buffer);
        if (root.Handle != System.IntPtr.Zero) return true;
#if NET7_0_OR_GREATER
        return root is WebGpu.WebGpuBuffer { IsValid: true };
#else
        return false;
#endif
    }

    /// <summary>
    /// Whether two buffers address the same allocation: the same root, the same native handle, or (WebGPU, where every
    /// handle is zero) the same device buffer id. Two handle-less buffers are not the same merely for sharing zero.
    /// </summary>
    public static bool SameAllocation(IGpuBuffer left, IGpuBuffer right)
    {
        var a = Of(left);
        var b = Of(right);
        if (ReferenceEquals(a, b)) return true;
        if (a.Handle != System.IntPtr.Zero) return a.Handle == b.Handle;
#if NET7_0_OR_GREATER
        return a is WebGpu.WebGpuBuffer wa && b is WebGpu.WebGpuBuffer wb && wa.BufferId == wb.BufferId;
#else
        return false;
#endif
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
