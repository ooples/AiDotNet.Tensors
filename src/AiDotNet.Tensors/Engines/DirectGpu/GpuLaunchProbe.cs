using System.Threading;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// Process-wide GPU work counter. Its only purpose is to let the op-parity backend-completeness
/// guard EMPIRICALLY tell which ops actually execute on the GPU from those that silently fall back
/// to the CPU: every real GPU dispatch — a compute kernel OR a device-to-device buffer copy (the
/// resident data movement behind concat/narrow/gather) — calls <see cref="OnLaunch"/>. Host I/O
/// (write/read buffer) is deliberately NOT counted, since a CPU fallback op that never touches the
/// device makes zero of ALL these calls. A test can <see cref="Reset"/> the counter, run one op on
/// the GPU engine, and read <see cref="Count"/> back — an op that does ZERO GPU work ran entirely on
/// the host, a genuine coverage gap no amount of reflection guesswork can hide.
/// </summary>
/// <remarks>
/// Deliberately backend-agnostic: each GPU backend increments it at its single kernel-dispatch choke
/// point (OpenCL: <c>DirectOpenClKernel.Execute*</c>). The cost is one interlocked add per launch,
/// dwarfed by the launch itself, and the counter is inert unless a test reads it. It is NOT a
/// correctness or scheduling signal — do not gate kernel behaviour on it.
/// </remarks>
internal static class GpuLaunchProbe
{
    private static long _count;

    /// <summary>Total kernel launches observed since the last <see cref="Reset"/> (lock-free read).</summary>
    public static long Count => Interlocked.Read(ref _count);

    /// <summary>Called once per GPU kernel dispatch. Cheap; safe to call from any thread.</summary>
    public static void OnLaunch() => Interlocked.Increment(ref _count);

    /// <summary>Zeroes the counter before a measured region. Returns the pre-reset value.</summary>
    public static long Reset() => Interlocked.Exchange(ref _count, 0);
}
