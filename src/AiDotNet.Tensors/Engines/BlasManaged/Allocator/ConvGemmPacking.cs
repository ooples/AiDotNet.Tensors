using System;
using System.Buffers;
using System.Threading;
using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.BlasManaged;

// A channel-block convolution prepares one immutable row-major filter for all of its spatial
// tasks. Share native MR-interleaved stripes for the life of that joined operation only. K-panel
// and row-block sizes may vary between GEMMs because the stripes span the full reduction axis.
internal sealed class ConvGemmPacking<T> : IDisposable where T : unmanaged
{
    private readonly T[] _filter;
    private readonly int _rows, _reduction;
    private readonly ArrayPool<T> _pool;
    private readonly WorkerSlot[] _slots;
    // Guards the one-time packing; a private object, so no caller can contend on this instance.
    private readonly object _packGate = new();
    private T[]? _packed;
    private int _mr;

    private struct WorkerSlot
    {
        internal byte[]? Buffer;
        internal int Active;
    }

    internal ConvGemmPacking(T[] filter, int rows, int reduction, int workers = 1, ArrayPool<T>? pool = null)
    {
        _filter = filter;
        _rows = rows;
        _reduction = reduction;
        _pool = pool ?? ArrayPool<T>.Shared;
        _slots = new WorkerSlot[workers];
    }

    internal unsafe T[]? ForRows(T* source, int length, int lda, int rows, int k, int mc, int mr, out int firstRow)
    {
        firstRow = 0;
        if (lda != _reduction || k != _reduction || rows < mr || (long)rows * k > length
            || (rows > mc && mc % mr != 0))
            return null;
        fixed (T* filter = _filter)
        {
            long offset = source - filter;
            if (offset < 0 || offset % k != 0 || offset / k > _rows - rows || (offset / k) % mr != 0)
                return null;
            firstRow = (int)(offset / k);
        }
        int alignedRows = _rows / mr * mr;
        long elements = (long)alignedRows * k;
        if (elements * sizeof(T) > 2 * 1024 * 1024) return null;
        var existing = Volatile.Read(ref _packed);
        if (existing is not null) return _mr == mr ? existing : null;
        lock (_packGate)
        {
            if (_packed is not null) return _mr == mr ? _packed : null;
            var buffer = _pool.Rent((int)elements);
            try
            {
                CpuStorageAllocationDiagnostics.Record<T>("workspace.conv1d_shared_gemm_filter", (int)elements);
                Avx2Pack.PackA<T>(_filter, k, false, buffer.AsSpan(0, (int)elements), alignedRows, k, mr);
                _mr = mr;
                Volatile.Write(ref _packed, buffer);
                return buffer;
            }
            catch
            {
                _pool.Return(buffer);
                throw;
            }
        }
    }

    public void Dispose()
    {
        var buffer = Interlocked.Exchange(ref _packed, null);
        if (buffer is not null) _pool.Return(buffer);
    }

    // A single joined spatial task processes several different im2col panels. Keep its
    // private B staging until that task ends. Concurrent or nested use gets an independent
    // temporary from the GEMM caller; the active buffer can never be overwritten.
    internal readonly struct Worker : IDisposable
    {
        internal readonly ConvGemmPacking<T> Filter;
        private readonly ArrayPool<byte> _pool;
        private readonly int _index;

        internal Worker(ConvGemmPacking<T> filter, int index = 0, ArrayPool<byte>? pool = null)
        {
            Filter = filter;
            _pool = pool ?? ArrayPool<byte>.Shared;
            _index = index;
        }

        internal byte[]? TryUseB(int bytes)
        {
            ref var slot = ref Filter._slots[_index];
            if (bytes > 2 * 1024 * 1024 || Interlocked.CompareExchange(ref slot.Active, 1, 0) != 0) return null;
            try
            {
                if (slot.Buffer is null || slot.Buffer.Length < bytes)
                {
                    var previous = slot.Buffer;
                    slot.Buffer = null;
                    if (previous is not null) _pool.Return(previous);
                    slot.Buffer = _pool.Rent(bytes);
                    CpuStorageAllocationDiagnostics.Record<byte>("workspace.conv1d_worker_gemm_columns", bytes);
                }
                return slot.Buffer;
            }
            catch
            {
                ReleaseB();
                throw;
            }
        }

        internal void ReleaseB() => Volatile.Write(ref Filter._slots[_index].Active, 0);

        public void Dispose()
        {
            var buffer = Interlocked.Exchange(ref Filter._slots[_index].Buffer, null);
            if (buffer is not null) _pool.Return(buffer);
        }
    }
}
