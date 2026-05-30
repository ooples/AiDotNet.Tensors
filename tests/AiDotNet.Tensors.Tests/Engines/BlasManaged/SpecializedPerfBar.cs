namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Frozen per-variant perf bars for #379. Values are placeholders until the first
/// authoritative bench run on the self-hosted runner (AIDOTNET_PERF_RUNNER=1) lands,
/// at which point the project owner sets them in a single gating commit — same
/// discipline as <see cref="PerfBar"/> for dense GEMM (#368).
/// </summary>
public static class SpecializedPerfBar
{
    // TRSM vs OpenBLAS strsm/dtrsm on the authoritative runner.
    public const int    TrsmMinWinRatePercent = 0;     // TO BE SET after first bench
    public const double TrsmMaxLossMultiple    = 99.0; // TO BE SET after first bench
    public const string TargetHardwareFingerprint = ""; // captured from runner

    /// <summary>True once the owner has frozen the TRSM bar (non-zero win rate).</summary>
    public static bool TrsmBarFrozen => TrsmMinWinRatePercent > 0;
}
