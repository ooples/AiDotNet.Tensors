// Copyright (c) AiDotNet. All rights reserved.

using System;
using System.Runtime.InteropServices;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// NVTX (NVIDIA Tools Extension) range / mark surface. Pairs every
/// <see cref="Range"/> open with a close so Nsight Systems / Nsight
/// Compute can attribute kernel time back to user-named regions.
///
/// <para>Surfaced for #219's "How we beat PyTorch" point #6: PyTorch
/// has scattered NVTX coverage gated by <c>--enable-cuda-nvtx</c>;
/// we autoinstrument from <c>LazyTensorScope</c> + <c>TapeStepContext</c>
/// so every op that runs under a scope/tape lands in the timeline
/// without user code changes. The auto-instrumentation plumbing lives
/// in the per-op dispatchers and calls into <see cref="Push"/> /
/// <see cref="Pop"/> here.</para>
///
/// <para>Falls through to a no-op when <c>nvToolsExt</c> isn't loadable
/// (no NVIDIA driver / no CUDA install). Cached <see cref="IsAvailable"/>
/// keeps the hot-path overhead to a single static-bool read.</para>
/// </summary>
public static class Nvtx
{
    private const string Lib = "nvToolsExt64_1";

    [DllImport(Lib, CharSet = CharSet.Ansi)]
    private static extern int nvtxRangePushA(string message);

    [DllImport(Lib)]
    private static extern int nvtxRangePop();

    [DllImport(Lib, CharSet = CharSet.Ansi)]
    private static extern void nvtxMarkA(string message);

    /// <summary>Whether <c>nvToolsExt</c> is loadable on this host.</summary>
    public static readonly bool IsAvailable = Probe();

    private static bool Probe()
    {
        try { nvtxMarkA("__probe__"); return true; }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
        catch (BadImageFormatException) { return false; }
        catch { return false; }
    }

    /// <summary>Pushes a named range onto NVTX's per-thread stack.
    /// Returns <c>true</c> when the push actually reached the native
    /// side; the <see cref="Range"/> RAII helper uses that to skip the
    /// matching <see cref="Pop"/> on failure (an unconditional pop
    /// would otherwise close an unrelated outer range from the caller's
    /// stack). Pair with <see cref="Pop"/>.</summary>
    public static bool Push(string name)
    {
        if (!IsAvailable) return false;
        try { nvtxRangePushA(name); return true; }
        catch { /* swallow — instrumentation must never break the run */ }
        return false;
    }

    /// <summary>Pops the most-recently pushed range. Idempotent on
    /// platforms without NVTX.</summary>
    public static void Pop()
    {
        if (!IsAvailable) return;
        try { nvtxRangePop(); }
        catch { /* swallow */ }
    }

    /// <summary>Emits a point-in-time NVTX marker.</summary>
    public static void Mark(string name)
    {
        if (!IsAvailable) return;
        try { nvtxMarkA(name); }
        catch { /* swallow */ }
    }

    /// <summary>RAII helper — push on construction, pop on dispose.
    /// <c>using (Nvtx.Range("matmul")) { ... }</c>. If the push fails
    /// (instrumentation should never break the run), the dispose
    /// becomes a no-op rather than closing an unrelated outer range.</summary>
    public static IDisposable Range(string name)
    {
        bool pushed = Push(name);
        return new RangeScope(pushed);
    }

    private sealed class RangeScope : IDisposable
    {
        private readonly bool _pushed;
        private bool _disposed;
        public RangeScope(bool pushed) { _pushed = pushed; }
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            if (_pushed) Pop();
        }
    }
}
