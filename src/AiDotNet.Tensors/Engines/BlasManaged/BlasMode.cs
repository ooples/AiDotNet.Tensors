namespace AiDotNet.Tensors.Engines.BlasManaged;

/// <summary>
/// Sub-issue B (#370): execution-mode toggle for <see cref="BlasManaged.Gemm{T}"/>.
///
/// <para>
/// <see cref="Deterministic"/> is the default — every call produces bit-exact
/// identical output across thread counts and across repeated invocations on the
/// same hardware. Allowed transforms: deterministic pairwise reduction trees
/// only; FMA disabled for FP32 (OpenBLAS-compatible); no instruction reorders
/// that affect rounding.
/// </para>
///
/// <para>
/// <see cref="Fast"/> is an opt-in escape hatch. Output may differ by ±1-2 ULP
/// across thread counts due to non-associative reduction order (the K-axis
/// driver's pairwise sum collapses in a thread-id-dependent way). FMA enabled
/// for FP32 (single rounding per fused-multiply-add instead of two — actually
/// <i>more</i> numerically accurate than the non-FMA path, just different).
/// Instruction reordering by the JIT is allowed.
/// </para>
///
/// <para>
/// The perf bar (<see cref="Tests.Engines.BlasManaged.PerfBar"/>) is measured
/// against <see cref="Fast"/>; correctness regression tests (Gate 3, the
/// <c>DeterminismTests</c> suite) stay in <see cref="Deterministic"/>.
/// </para>
/// </summary>
public enum BlasMode
{
    /// <summary>
    /// Default. Bit-exact reproducibility across thread counts on the same hardware.
    /// </summary>
    Deterministic = 0,

    /// <summary>
    /// Opt-in. Allows non-associative reduction, FMA-on-FP32, instruction reorder.
    /// Numerical accuracy is at least as good as <see cref="Deterministic"/>; only
    /// bit-exact reproducibility differs.
    /// </summary>
    Fast = 1,
}
