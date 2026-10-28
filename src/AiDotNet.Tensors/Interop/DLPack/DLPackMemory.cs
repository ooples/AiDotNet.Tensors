using System;
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Threading;

namespace AiDotNet.Tensors.Interop;

/// <summary>
/// The producer's half of an imported DLPack tensor: calls its deleter exactly once, when the last AiDotNet.Tensors
/// object using the memory lets go of it.
/// </summary>
/// <remarks>
/// <para>The deleter runs on <see cref="Release"/>, which the owning storage calls when its last tensor or view is
/// disposed. A tensor that is never disposed still releases the producer's memory: the finalizer runs the deleter
/// once nothing references this object, and nothing can read the memory by then (every span, vector and device
/// buffer over it holds this object). The interlocked exchange makes the deleter run once however release and
/// finalization interleave.</para>
/// </remarks>
internal sealed class DLPackDeleterHandle
{
    private IntPtr _deleter;
    private readonly IntPtr _self;
    private readonly Action? _beforeDelete;

    /// <param name="deleter">The producer's <c>deleter</c> function pointer (may be NULL: nothing to call).</param>
    /// <param name="self">The managed-tensor pointer the deleter is called with.</param>
    /// <param name="beforeDelete">Runs before the deleter: for device memory, the wait for work still queued on it.</param>
    internal DLPackDeleterHandle(IntPtr deleter, IntPtr self, Action? beforeDelete = null)
    {
        _deleter = deleter;
        _self = self;
        _beforeDelete = beforeDelete;
    }

    /// <summary>Whether the deleter has run (or there was none to run).</summary>
    internal bool IsReleased => Volatile.Read(ref _deleter) == IntPtr.Zero;

    /// <summary>Runs the deleter if it has not run yet.</summary>
    internal void Release()
    {
        var deleter = Interlocked.Exchange(ref _deleter, IntPtr.Zero);
        if (deleter == IntPtr.Zero) return;
        GC.SuppressFinalize(this);
        try
        {
            _beforeDelete?.Invoke();
        }
        finally
        {
            DLPackNative.InvokeDeleter(deleter, _self);
        }
    }

    ~DLPackDeleterHandle()
    {
        var deleter = Interlocked.Exchange(ref _deleter, IntPtr.Zero);
        if (deleter == IntPtr.Zero) return;
        try { _beforeDelete?.Invoke(); }
        catch { /* a finalizer must not throw; the producer's memory is still released below */ }
        DLPackNative.InvokeDeleter(deleter, _self);
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
        // Not on the finalizer path: the deleter handle has its own finalizer, which covers a manager that was
        // never disposed, and running a C callback from here while disposing=false would be the same call twice.
        if (disposing) _owner.Release();
    }
}
