using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.CpuJit;

/// <summary>
/// Allocates 64-byte aligned memory for SIMD operations.
/// Unlike .NET GC arrays (8-16 byte aligned), this guarantees alignment for:
/// - AVX2 VMOVNTPS (requires 32-byte alignment) — non-temporal stores
/// - AVX-512 VMOVNTPS (requires 64-byte alignment)
/// - Optimal cache line utilization (64-byte cache lines on x86)
///
/// This is the same approach libtorch uses via posix_memalign/aligned_alloc.
/// </summary>
internal sealed class AlignedBuffer : IDisposable
{
    private IntPtr _raw;      // Original allocation (for freeing)
    private IntPtr _aligned;  // Aligned pointer (for use)
    private readonly int _sizeBytes;
    private bool _disposed;

    /// <summary>
    /// Pointer to the aligned memory region.
    /// </summary>
    public IntPtr Pointer => _aligned;

    /// <summary>
    /// Size of the usable buffer in bytes.
    /// </summary>
    public int SizeBytes => _sizeBytes;

    /// <summary>
    /// Number of float elements that fit in this buffer.
    /// </summary>
    public int FloatCount => _sizeBytes / sizeof(float);

    /// <summary>
    /// Gets a typed pointer to the aligned memory.
    /// </summary>
    public unsafe float* FloatPtr => (float*)_aligned;

    /// <summary>
    /// Allocates aligned memory for the specified number of floats.
    /// </summary>
    /// <param name="floatCount">Number of float elements.</param>
    /// <param name="alignment">Alignment in bytes (default 64 for cache line / AVX-512).</param>
    public AlignedBuffer(int floatCount, int alignment = 64)
    {
        if (floatCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(floatCount));

        _sizeBytes = floatCount * sizeof(float);

        // Over-allocate by (alignment - 1) bytes to guarantee we can align within it
        int allocSize = _sizeBytes + alignment - 1;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            _raw = Marshal.AllocHGlobal(allocSize);
        }
        else
        {
            _raw = Marshal.AllocHGlobal(allocSize);
        }

        if (_raw == IntPtr.Zero)
            throw new OutOfMemoryException($"Failed to allocate {allocSize} bytes for aligned buffer.");

        // Align the pointer: round up to next alignment boundary
        long addr = _raw.ToInt64();
        long alignedAddr = (addr + alignment - 1) & ~((long)alignment - 1);
        _aligned = new IntPtr(alignedAddr);
    }

    /// <summary>
    /// Creates a Span over the aligned buffer.
    /// </summary>
    public unsafe Span<float> AsSpan()
    {
        return new Span<float>((void*)_aligned, FloatCount);
    }

    /// <summary>
    /// Copies data from a managed array into this aligned buffer.
    /// </summary>
    public unsafe void CopyFrom(ReadOnlySpan<float> source)
    {
        if (source.Length > FloatCount)
            throw new ArgumentException($"Source ({source.Length}) exceeds buffer capacity ({FloatCount}).");

        fixed (float* srcPtr = source)
        {
            Buffer.MemoryCopy(srcPtr, (void*)_aligned, _sizeBytes, source.Length * sizeof(float));
        }
    }

    /// <summary>
    /// Copies count elements from a managed array starting at offset into this aligned buffer.
    /// </summary>
    public unsafe void CopyFrom(float[] source, int offset, int count)
    {
        if (count > FloatCount)
            throw new ArgumentException($"Count ({count}) exceeds buffer capacity ({FloatCount}).");
        if (offset < 0 || offset + count > source.Length)
            throw new ArgumentOutOfRangeException(nameof(offset), $"Offset ({offset}) + count ({count}) exceeds source length ({source.Length}).");

        fixed (float* srcPtr = source)
        {
            Buffer.MemoryCopy(srcPtr + offset, (void*)_aligned, _sizeBytes, count * sizeof(float));
        }
    }

    /// <summary>
    /// Copies data from this aligned buffer into a managed span.
    /// </summary>
    public unsafe void CopyTo(Span<float> destination)
    {
        if (destination.Length < FloatCount)
            throw new ArgumentException($"Destination ({destination.Length}) is smaller than buffer ({FloatCount}).");

        fixed (float* dstPtr = destination)
        {
            Buffer.MemoryCopy((void*)_aligned, dstPtr, destination.Length * sizeof(float), _sizeBytes);
        }
    }

    /// <summary>
    /// Copies count elements from this aligned buffer into a managed array starting at offset.
    /// </summary>
    public unsafe void CopyTo(float[] destination, int offset, int count)
    {
        if (count > FloatCount)
            throw new ArgumentException($"Count ({count}) exceeds buffer capacity ({FloatCount}).");
        if (offset < 0 || offset + count > destination.Length)
            throw new ArgumentOutOfRangeException(nameof(offset), $"Offset ({offset}) + count ({count}) exceeds destination length ({destination.Length}).");

        fixed (float* dstPtr = destination)
        {
            Buffer.MemoryCopy((void*)_aligned, dstPtr + offset, (destination.Length - offset) * sizeof(float), count * sizeof(float));
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);

        if (_raw != IntPtr.Zero)
        {
            Marshal.FreeHGlobal(_raw);
            _raw = IntPtr.Zero;
            _aligned = IntPtr.Zero;
        }
    }

    ~AlignedBuffer()
    {
        Dispose();
    }
}
