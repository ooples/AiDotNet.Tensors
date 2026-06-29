using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-S (#409): allocates RW→RX executable memory and writes machine-code bytes
/// into it, for the hand-emitted GEMM microkernels (<see cref="X64Assembler"/>).
/// This is FIRST-PARTY native code we emit — it introduces no third-party
/// library dependency; the only external calls are to the OS allocator
/// (VirtualAlloc / mmap), which is not a supply-chain dependency.
///
/// <para>
/// W^X discipline: pages are mapped RW, the code is written, then flipped to RX
/// (read+execute, not writable) before use. Some hardened/AOT environments forbid
/// dynamic executable memory; callers must keep a managed fallback
/// (<see cref="NativeAotDetector.IsDynamicCodeSupported"/>) and handle a null/zero
/// pointer from <see cref="TryAllocate"/>.
/// </para>
/// </summary>
internal sealed class ExecutableMemory : IDisposable
{
    private IntPtr _ptr;
    private readonly nuint _size;
    private readonly bool _isWindows;

    /// <summary>Base address of the executable region (call target). IntPtr.Zero if allocation failed.</summary>
    internal IntPtr Pointer => _ptr;

    /// <summary>Size in bytes of the emitted code region (used by callers to bound their kernel caches).</summary>
    internal long Size => (long)_size;

    private ExecutableMemory(IntPtr ptr, nuint size, bool isWindows)
    {
        _ptr = ptr;
        _size = size;
        _isWindows = isWindows;
    }

    /// <summary>
    /// Allocate executable memory and copy <paramref name="code"/> into it,
    /// leaving the region read+execute. Returns null if the platform/runtime
    /// disallows dynamic executable memory (caller falls back to managed).
    /// </summary>
    internal static ExecutableMemory? TryAllocate(ReadOnlySpan<byte> code)
    {
        if (!NativeAotDetector.IsDynamicCodeSupported) return null;
        try
        {
            nuint size = (nuint)code.Length;
            bool win = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
            if (win)
            {
                // VirtualAlloc RW, write, VirtualProtect -> RX, FlushInstructionCache.
                IntPtr p = VirtualAlloc(IntPtr.Zero, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
                if (p == IntPtr.Zero) return null;
                unsafe { fixed (byte* src = code) Buffer.MemoryCopy(src, (void*)p, code.Length, code.Length); }
                if (!VirtualProtect(p, size, PAGE_EXECUTE_READ, out _)) { VirtualFree(p, 0, MEM_RELEASE); return null; }
                FlushInstructionCache(GetCurrentProcess(), p, size);
                return new ExecutableMemory(p, size, true);
            }
            else
            {
                // mmap RW (anon, private), write, mprotect -> RX.
                IntPtr p = mmap(IntPtr.Zero, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
                if (p == IntPtr.Zero || p == new IntPtr(-1)) return null;
                unsafe { fixed (byte* src = code) Buffer.MemoryCopy(src, (void*)p, code.Length, code.Length); }
                if (mprotect(p, size, PROT_READ | PROT_EXEC) != 0) { munmap(p, size); return null; }
                return new ExecutableMemory(p, size, false);
            }
        }
        catch
        {
            return null;
        }
    }

    public void Dispose()
    {
        if (_ptr == IntPtr.Zero) return;
        try
        {
            if (_isWindows) VirtualFree(_ptr, 0, MEM_RELEASE);
            else munmap(_ptr, _size);
        }
        catch { /* best-effort free */ }
        _ptr = IntPtr.Zero;
    }

    // ── Windows ────────────────────────────────────────────────────────────
    private const uint MEM_COMMIT = 0x1000, MEM_RESERVE = 0x2000, MEM_RELEASE = 0x8000;
    private const uint PAGE_READWRITE = 0x04, PAGE_EXECUTE_READ = 0x20;

    [DllImport("kernel32", SetLastError = true)]
    private static extern IntPtr VirtualAlloc(IntPtr addr, nuint size, uint allocType, uint protect);
    [DllImport("kernel32", SetLastError = true)]
    private static extern bool VirtualProtect(IntPtr addr, nuint size, uint newProtect, out uint oldProtect);
    [DllImport("kernel32", SetLastError = true)]
    private static extern bool VirtualFree(IntPtr addr, nuint size, uint freeType);
    [DllImport("kernel32")]
    private static extern bool FlushInstructionCache(IntPtr process, IntPtr addr, nuint size);
    [DllImport("kernel32")]
    private static extern IntPtr GetCurrentProcess();

    // ── Unix ───────────────────────────────────────────────────────────────
    private const int PROT_READ = 0x1, PROT_WRITE = 0x2, PROT_EXEC = 0x4;
    private const int MAP_PRIVATE = 0x2;
    // MAP_ANONYMOUS differs: Linux 0x20, macOS 0x1000. Detect at use.
    private static int MAP_ANONYMOUS => RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? 0x1000 : 0x20;

    [DllImport("libc", SetLastError = true)]
    private static extern IntPtr mmap(IntPtr addr, nuint length, int prot, int flags, int fd, nint offset);
    [DllImport("libc", SetLastError = true)]
    private static extern int mprotect(IntPtr addr, nuint len, int prot);
    [DllImport("libc", SetLastError = true)]
    private static extern int munmap(IntPtr addr, nuint length);
}
