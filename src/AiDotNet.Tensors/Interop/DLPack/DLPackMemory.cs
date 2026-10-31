using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading;
using AiDotNet.Tensors.Engines.DirectGpu;

namespace AiDotNet.Tensors.Interop;

/// <summary>
/// The producer's half of an imported DLPack tensor: calls its deleter exactly once, when the last AiDotNet.Tensors
/// object using the memory lets go of it.
/// </summary>
/// <remarks>
/// <para>The deleter runs on <see cref="Release"/>, which the owner of the memory calls once nothing uses it any
/// more: the tensor storage over host memory when its last tensor or view is disposed, and the device buffer over
/// device memory when it is disposed, after it has waited for the kernels still queued on the memory.</para>
/// <para><b>A tensor that is never disposed.</b> Host memory is then never released: the producer's memory stays
/// allocated for the rest of the process, and <see cref="DLPack.LeakedHostImports"/> counts it. Running the deleter
/// from the finalizer would be unsound. A <see cref="Span{T}"/> over the memory is a raw pointer the GC does not
/// track, so code still reading through one (an engine kernel whose tensor argument the JIT already considers dead)
/// does not keep this object alive, and the deleter would free the memory under it.</para>
/// <para>Device memory has no span: kernels address it by the pointer captured at launch. A device import registers
/// its backend (<see cref="SetDeviceOwner"/>), and the finalizer hands the deleter to
/// <see cref="IDLPackDeviceBackend.ReleaseAfterPendingWorkDeferred"/>. That call never touches the driver on the
/// finalizer thread. It queues the release, and the backend's next operation waits for the work queued on its
/// stream before it runs the deleter. That is how the backend already frees its own buffers on finalization.</para>
/// <para>The interlocked exchange makes the deleter run once however release and finalization interleave.</para>
/// </remarks>
internal sealed class DLPackDeleterHandle
{
    private IntPtr _deleter;
    private readonly IntPtr _self;
    private IDLPackDeviceBackend? _device;

    /// <param name="deleter">The producer's <c>deleter</c> function pointer (may be NULL: nothing to call).</param>
    /// <param name="self">The managed-tensor pointer the deleter is called with.</param>
    internal DLPackDeleterHandle(IntPtr deleter, IntPtr self)
    {
        _deleter = deleter;
        _self = self;
    }

    /// <summary>Whether the deleter has run (or there was none to run), or has been queued to run.</summary>
    internal bool IsReleased => Volatile.Read(ref _deleter) == IntPtr.Zero;

    /// <summary>
    /// Marks the memory as device memory used by <paramref name="device"/>'s kernels, so finalization releases it
    /// once the work queued on that backend has finished.
    /// </summary>
    internal void SetDeviceOwner(IDLPackDeviceBackend device) => Volatile.Write(ref _device, device);

    /// <summary>Runs the deleter if it has not run yet. The caller has finished every use of the memory.</summary>
    internal void Release()
    {
        var deleter = Interlocked.Exchange(ref _deleter, IntPtr.Zero);
        if (deleter == IntPtr.Zero) return;
        GC.SuppressFinalize(this);
        DLPackNative.InvokeDeleter(deleter, _self);
    }

    ~DLPackDeleterHandle()
    {
        var deleter = Interlocked.Exchange(ref _deleter, IntPtr.Zero);
        if (deleter == IntPtr.Zero) return;
        var device = Volatile.Read(ref _device);
        if (device is null)
        {
            // Host memory: a span over it may still be in use (see the remarks), so it is leaked, not freed.
            DLPack.CountLeakedHostImport();
            return;
        }
        IntPtr self = _self;
        try
        {
            device.ReleaseAfterPendingWorkDeferred(() => DLPackNative.InvokeDeleter(deleter, self));
        }
        catch
        {
            // A finalizer must not throw. Queuing only fails when out of memory; the producer's memory then leaks.
        }
    }
}

/// <summary>
/// Host memory owned by a DLPack producer, exposed as <see cref="Memory{T}"/> so a tensor can alias it without a
/// copy. Disposing it (the storage does, with its last reference) runs the producer's deleter once.
/// </summary>
internal sealed unsafe class DLPackHostMemory<T> : MemoryManager<T>
{
    private readonly byte* _pointer;
    private readonly int _length;
    private readonly DLPackDeleterHandle _owner;
    private int _disposed;

    /// <param name="pointer">The first element (data pointer plus byte_offset, plus any negative-stride shift).</param>
    /// <param name="length">Element count covered.</param>
    /// <param name="owner">The deleter to run when the memory is released.</param>
    internal DLPackHostMemory(IntPtr pointer, int length, DLPackDeleterHandle owner)
    {
        _pointer = (byte*)pointer;
        _length = length;
        _owner = owner;
    }

    internal DLPackDeleterHandle Owner => _owner;

    public override Span<T> GetSpan()
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(DLPackHostMemory<T>));
        // Span<T>(void*, int) rather than MemoryMarshal.CreateSpan so this compiles on .NET Framework. Only the
        // blittable element types DLPack maps reach here (see DLPack.TryGetDataType).
        return new Span<T>(_pointer, _length);
    }

    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if (Volatile.Read(ref _disposed) != 0) throw new ObjectDisposedException(nameof(DLPackHostMemory<T>));
        if ((uint)elementIndex > (uint)_length) throw new ArgumentOutOfRangeException(nameof(elementIndex));
        // Native memory never moves; the handle only keeps this manager (and so the producer's memory) alive.
        return new MemoryHandle(_pointer + (long)elementIndex * Unsafe.SizeOf<T>(), default, this);
    }

    public override void Unpin() { }

    protected override void Dispose(bool disposing)
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        // Not on the finalizer path: a span over the memory can outlive this manager's last reference, so a manager
        // that was never disposed leaves the memory allocated (the deleter handle counts it as a leak).
        if (disposing) _owner.Release();
    }
}
