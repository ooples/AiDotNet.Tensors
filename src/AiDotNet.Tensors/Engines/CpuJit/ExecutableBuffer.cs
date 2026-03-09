using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.CpuJit;

/// <summary>
/// Allocates read-write-execute memory for JIT-compiled machine code.
/// Uses VirtualAlloc on Windows, mmap on Linux/macOS.
/// All allocations are page-aligned (4096+ bytes), guaranteeing 64-byte alignment
/// which is required for AVX-512 and optimal for AVX2 non-temporal stores.
/// </summary>
internal sealed class ExecutableBuffer : IDisposable
{
    private IntPtr _buffer;
    private readonly int _size;
    private bool _disposed;

    /// <summary>
    /// Pointer to the executable memory region.
    /// </summary>
    public IntPtr Pointer => _buffer;

    /// <summary>
    /// Size of the allocated buffer in bytes.
    /// </summary>
    public int Size => _size;

    /// <summary>
    /// Allocates executable memory of the specified size.
    /// </summary>
    /// <param name="size">Number of bytes to allocate.</param>
    public ExecutableBuffer(int size)
    {
        if (size <= 0)
            throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");

        _size = size;

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            // Allocate as RW first, switch to RX after code is written (W^X best practice)
            _buffer = VirtualAlloc(IntPtr.Zero, (nuint)size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (_buffer == IntPtr.Zero)
                throw new OutOfMemoryException($"VirtualAlloc failed to allocate {size} bytes of executable memory.");
        }
        else
        {
            // Linux/macOS: mmap with PROT_READ | PROT_WRITE (will mprotect to RX after writing)
            int mapAnon = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? MAP_ANONYMOUS_MACOS : MAP_ANONYMOUS_LINUX;
            _buffer = mmap(IntPtr.Zero, (nuint)size, PROT_READ | PROT_WRITE,
                MAP_PRIVATE | mapAnon, -1, 0);
            if (_buffer == (IntPtr)(-1))
            {
                _buffer = IntPtr.Zero;
                throw new OutOfMemoryException($"mmap failed to allocate {size} bytes of executable memory.");
            }
        }
    }

    /// <summary>
    /// Switches the buffer from writable to executable (W^X compliance).
    /// Must be called after all code has been written and before execution.
    /// </summary>
    public void MakeExecutable()
    {
        if (_buffer == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(ExecutableBuffer));

        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
        {
            if (!VirtualProtect(_buffer, (nuint)_size, PAGE_EXECUTE_READ, out _))
                throw new InvalidOperationException("VirtualProtect failed to set executable permissions.");
        }
        else
        {
            int result = mprotect(_buffer, (nuint)_size, PROT_READ | PROT_EXEC);
            if (result != 0)
                throw new InvalidOperationException("mprotect failed to set executable permissions.");
        }
    }

    /// <summary>
    /// Creates a typed function pointer from the executable buffer.
    /// </summary>
    public unsafe TDelegate CreateDelegate<TDelegate>() where TDelegate : Delegate
    {
        if (_buffer == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(ExecutableBuffer));

        return Marshal.GetDelegateForFunctionPointer<TDelegate>(_buffer);
    }

    /// <summary>
    /// Gets a raw function pointer for use with calli or delegate* unmanaged.
    /// </summary>
    public unsafe void* GetFunctionPointer()
    {
        if (_buffer == IntPtr.Zero)
            throw new ObjectDisposedException(nameof(ExecutableBuffer));

        return (void*)_buffer;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        GC.SuppressFinalize(this);

        if (_buffer != IntPtr.Zero)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                VirtualFree(_buffer, 0, MEM_RELEASE);
            }
            else
            {
                munmap(_buffer, (nuint)_size);
            }
            _buffer = IntPtr.Zero;
        }
    }

    ~ExecutableBuffer()
    {
        Dispose();
    }

    // Windows
    private const uint MEM_COMMIT = 0x1000;
    private const uint MEM_RESERVE = 0x2000;
    private const uint MEM_RELEASE = 0x8000;
    private const uint PAGE_READWRITE = 0x04;
    private const uint PAGE_EXECUTE_READ = 0x20;

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr VirtualAlloc(IntPtr lpAddress, nuint dwSize, uint flAllocationType, uint flProtect);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool VirtualFree(IntPtr lpAddress, nuint dwSize, uint dwFreeType);

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool VirtualProtect(IntPtr lpAddress, nuint dwSize, uint flNewProtect, out uint lpflOldProtect);

    // Linux/macOS
    private const int PROT_READ = 0x1;
    private const int PROT_WRITE = 0x2;
    private const int PROT_EXEC = 0x4;
    private const int MAP_PRIVATE = 0x02;
    private const int MAP_ANONYMOUS_LINUX = 0x20;
    private const int MAP_ANONYMOUS_MACOS = 0x1000;

    [DllImport("libc", SetLastError = true)]
    private static extern IntPtr mmap(IntPtr addr, nuint length, int prot, int flags, int fd, long offset);

    [DllImport("libc", SetLastError = true)]
    private static extern int munmap(IntPtr addr, nuint length);

    [DllImport("libc", SetLastError = true)]
    private static extern int mprotect(IntPtr addr, nuint length, int prot);
}
