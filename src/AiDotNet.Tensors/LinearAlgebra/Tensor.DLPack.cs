using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu;
using AiDotNet.Tensors.Interop;

namespace AiDotNet.Tensors.LinearAlgebra;

/// <summary>
/// DLPack zero-copy exchange (torch <c>from_dlpack</c> / <c>__dlpack__</c>). See <see cref="DLPack"/> for the ownership,
/// device, stream and element-type rules.
/// </summary>
public partial class Tensor<T>
{
    /// <summary>
    /// Exports this tensor as a DLPack <c>DLManagedTensorVersioned*</c> over its own memory (no copy): what
    /// <c>__dlpack__()</c> returns. The consumer owns the pointer and must call its deleter exactly once.
    /// </summary>
    /// <param name="consumerStream">The stream the consumer will use the data on (CUDA only; ignored for CPU).</param>
    /// <returns>The <c>DLManagedTensorVersioned*</c>.</returns>
    /// <remarks>
    /// <para>The exported memory stays valid until the deleter runs, even if this tensor is disposed first. Strided
    /// views export with their strides. A tensor over a read-only mapping exports with the read-only flag.</para>
    /// <para>A GPU-resident <c>Tensor&lt;float&gt;</c> on the CUDA backend exports its device pointer. Any other
    /// GPU-resident tensor (another backend, whose memory has no foreign-addressable pointer, or an element type the
    /// device stores float-encoded) exports a host copy, flagged <c>DLPACK_FLAG_BITMASK_IS_COPIED</c>; this tensor
    /// stays on its device and does not see writes to the copy.</para>
    /// </remarks>
    /// <exception cref="NotSupportedException">The element type has no DLPack dtype, or the tensor is sparse.</exception>
    /// <exception cref="InvalidOperationException">The tensor is in a packed (NCHWc) layout.</exception>
    public IntPtr ToDLPack(DLPackStream consumerStream = default) => ExportDLPack(versioned: true, consumerStream);

    /// <summary>
    /// Exports this tensor as a legacy, unversioned DLPack <c>DLManagedTensor*</c>, for consumers that predate the
    /// versioned layout. Same ownership and lifetime as <see cref="ToDLPack"/>; the legacy layout has no flags, so a
    /// read-only tensor cannot be exported this way.
    /// </summary>
    public IntPtr ToDLPackLegacy(DLPackStream consumerStream = default) => ExportDLPack(versioned: false, consumerStream);

    /// <summary>
    /// Wraps a DLPack <c>DLManagedTensorVersioned*</c> as a tensor over the producer's memory (no copy): torch's
    /// <c>from_dlpack</c>. Takes ownership of the pointer: the producer's deleter runs exactly once, when this tensor's
    /// storage is released, or before the exception if the tensor cannot be imported.
    /// </summary>
    /// <param name="managedTensorVersioned">The producer's <c>DLManagedTensorVersioned*</c>.</param>
    /// <exception cref="NotSupportedException">A different DLPack major version, an unsupported device, or an element
    /// type that is not <typeparamref name="T"/>'s.</exception>
    /// <exception cref="ArgumentException">A malformed tensor (for example a dtype that is not <typeparamref name="T"/>).</exception>
    public static Tensor<T> FromDLPack(IntPtr managedTensorVersioned) => DLPack.Import<T>(managedTensorVersioned, versioned: true);

    /// <summary>
    /// Wraps a legacy, unversioned DLPack <c>DLManagedTensor*</c> (for example from
    /// <c>torch.utils.dlpack.to_dlpack</c>) as a tensor over the producer's memory. Same ownership as
    /// <see cref="FromDLPack"/>.
    /// </summary>
    public static Tensor<T> FromDLPackLegacy(IntPtr managedTensor) => DLPack.Import<T>(managedTensor, versioned: false);

    private unsafe IntPtr ExportDLPack(bool versioned, DLPackStream consumerStream)
    {
        using var keepThis = new KeepAliveScope(this);   // holds this while its storage is in use
        if (!DLPack.TryGetDataType<T>(out var dtype))
            throw new NotSupportedException($"Tensor<{typeof(T).Name}> has no DLPack element type.");
        if (IsSparse) throw new NotSupportedException("A sparse tensor has no dense storage to export through DLPack.");
        if (Layout != TensorLayout.Nchw)
            throw new InvalidOperationException(
                $"A tensor in the packed {Layout} layout has no DLPack form; reorder it to NCHW (IEngine.ReorderToNchw) first.");

        EnsureMaterialized();
        if (IsGpuResident)
            return TryExportDLPackDevice(versioned, consumerStream, dtype, out IntPtr device)
                ? device
                : ExportDLPackHostCopy(versioned, dtype);

        bool readOnly = _storage.IsReadOnlyMapped;
        if (readOnly && !versioned)
            throw new NotSupportedException(
                "This tensor aliases a read-only mapping; the legacy DLPack layout cannot mark it read-only. Use ToDLPack.");
        // A copy-on-write clone gets its own storage now: the consumer may write, and must not write into a peer.
        if (!readOnly) EnsureOwnedForWrite();
        // A deferred device download lands in the host vector here, before the memory is pinned and handed out.
        _ = _data.AsSpan();

        // The view shares (and references) the storage, so the memory outlives a Dispose of this tensor until the
        // consumer's deleter releases the view.
        var view = CreateStorageView((int[])_shape.Clone(), (int[])_strides.Clone(), _storageOffset);
        MemoryHandle pin = default;
        try
        {
            Memory<T> memory = readOnly
                ? MemoryMarshal.AsMemory(view._storage.AsReadOnlyMemory())   // mapped pages: nothing to escape
                : view._storage.AsMemory();   // marks the array escaped: the result recycler never reuses it
            pin = memory.Pin();
            var data = (IntPtr)((byte*)pin.Pointer + (long)_storageOffset * Unsafe.SizeOf<T>());
            return DLPack.CreateExport(
                versioned, data, DLPackNative.DeviceCpu, 0, dtype, view._shape, view._strides, readOnly, pin, view);
        }
        catch
        {
            pin.Dispose();
            view.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The device-pointer export: a GPU-resident <c>Tensor&lt;float&gt;</c> in plain float32 layout on a backend with
    /// <see cref="IDLPackDeviceBackend"/>. False when the tensor has no such form.
    /// </summary>
    private bool TryExportDLPackDevice(bool versioned, DLPackStream consumerStream, DLPackNative.DLDataType dtype, out IntPtr exported)
    {
        exported = IntPtr.Zero;
        // Device copies of non-float element types are float-encoded, so only float has a device form DLPack describes.
        if (typeof(T) != typeof(float)) return false;
        if (_gpuBackend is not IDLPackDeviceBackend device) return false;
        if (_gpuBufferIsSplitComplex || _gpuBufferContainsRawInt32 || _gpuBufferContainsRawInt64) return false;
        // A strided or offset view has no device buffer in its own layout.
        if (!IsContiguous || _storageOffset != 0) return false;
        var buffer = TryGetGpuBuffer();
        if (buffer is null || buffer.Handle == IntPtr.Zero || buffer.Size < Length) return false;

        var view = CreateStorageView((int[])_shape.Clone(), (int[])_strides.Clone(), _storageOffset);
        try
        {
            // The producer's half of the stream contract: the consumer's stream waits for the work that wrote the data.
            if (!consumerStream.IsNoSync) device.OrderStreamAfterPendingWork(consumerStream.CudaStreamHandle);
            exported = DLPack.CreateExport(
                versioned, buffer.Handle, device.DLPackDeviceType, device.DeviceOrdinal, dtype,
                view._shape, view._strides, readOnly: false, pin: default, keepAlive: view);
            return true;
        }
        catch
        {
            view.Dispose();
            throw;
        }
    }

    /// <summary>
    /// A host copy of a GPU-resident tensor with no device-pointer form, exported with the IS_COPIED flag (the consumer
    /// owns it outright). The legacy layout has no flags; the copy is still the consumer's alone.
    /// </summary>
    private unsafe IntPtr ExportDLPackHostCopy(bool versioned, DLPackNative.DLDataType dtype)
    {
        // ToArray reads the device data back without moving this tensor off its device.
        var copy = new Tensor<T>(ToArray(), (int[])_shape.Clone());
        MemoryHandle pin = default;
        try
        {
            pin = copy._storage.AsMemory().Pin();
            return DLPack.CreateExport(
                versioned, (IntPtr)pin.Pointer, DLPackNative.DeviceCpu, 0, dtype, copy._shape, copy._strides,
                readOnly: false, pin, copy, isCopied: true);
        }
        catch
        {
            pin.Dispose();
            copy.Dispose();
            throw;
        }
    }

    /// <summary>A contiguous tensor over imported DLPack host memory; the storage owns (and finally disposes) it.</summary>
    internal static Tensor<T> AliasDLPackMemory(DLPackHostMemory<T> memory, int[] shape, bool writable)
    {
        try
        {
            var tensor = new Tensor<T>((int[])shape.Clone(), Vector<T>.WrapMemory(memory.Memory));
            tensor.AliasStorageFromMmap(Vector<T>.WrapMemory(memory.Memory), memory, writable);
            return tensor;
        }
        catch
        {
            ((IDisposable)memory).Dispose();   // idempotent if the storage already took it
            throw;
        }
    }

    /// <summary>A strided view over imported DLPack host memory covering <paramref name="covered"/> elements.</summary>
    internal static Tensor<T> AliasDLPackMemoryStrided(
        DLPackHostMemory<T> covering, int covered, int[] shape, int[] strides, int storageOffset, bool writable)
    {
        var flat = AliasDLPackMemory(covering, new[] { covered }, writable);
        try
        {
            // The view takes its own storage reference; disposing the flat alias below leaves the view as the owner.
            return new Tensor<T>(flat._data, shape, strides, storageOffset, flat._storage);
        }
        finally
        {
            flat.Dispose();
        }
    }
}
