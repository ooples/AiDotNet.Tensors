using AiDotNet.Tensors.Helpers;

namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>
/// GPU-side surface for the global deterministic-reduction mode flag.
/// </summary>
/// <remarks>
/// <para>
/// Mirrors the contract of <see cref="AiDotNetEngine.DeterministicMode"/> on the GPU
/// dispatch path. When <see cref="IsActive"/> is <c>true</c>, GPU backends must select
/// kernel variants that produce bit-identical results across runs at the same seed
/// (no <c>atomicAdd</c>/<c>atomic_add_float</c> for floating-point accumulation, since
/// atomic ordering is scheduler-dependent and FP addition is not associative).
/// </para>
/// <para>
/// The single source of truth is <see cref="BlasProvider.IsDeterministicMode"/>;
/// this type exists so GPU dispatch code does not have to import the CPU BLAS namespace
/// at every kernel-selection site. Reads are lock-free; the flag is volatile inside
/// <see cref="BlasProvider"/>.
/// </para>
/// <para>
/// Issue #382: extends the existing <see cref="AiDotNetEngine.SetDeterministicMode(bool)"/>
/// contract to cover GPU floating-point reductions. Before this work, the flag governed
/// only CPU GEMM dispatch — GPU kernels using <c>atomicAdd</c> for gradient scatter
/// continued to produce non-deterministic FP accumulation order even when the flag was
/// set, breaking the documented "bit-identical across runs" guarantee.
/// </para>
/// </remarks>
internal static class GpuDeterminism
{
    /// <summary>
    /// Returns <c>true</c> when deterministic-reduction mode is currently in effect for
    /// the calling thread. Reads the per-thread override if installed, otherwise falls
    /// back to the process-wide setting maintained by
    /// <see cref="AiDotNetEngine.SetDeterministicMode(bool)"/>.
    /// </summary>
    public static bool IsActive => BlasProvider.IsDeterministicMode;
}
