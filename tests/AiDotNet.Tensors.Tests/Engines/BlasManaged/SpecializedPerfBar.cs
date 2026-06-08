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

    // SYRK vs OpenBLAS ssyrk/dsyrk on the authoritative runner.
    public const int    SyrkMinWinRatePercent = 0;     // TO BE SET after first bench
    public const double SyrkMaxLossMultiple    = 99.0; // TO BE SET after first bench

    /// <summary>True once the owner has frozen the SYRK bar (non-zero win rate).</summary>
    public static bool SyrkBarFrozen => SyrkMinWinRatePercent > 0;

    // SYMM vs OpenBLAS ssymm/dsymm on the authoritative runner.
    public const int    SymmMinWinRatePercent = 0;     // TO BE SET after first bench
    public const double SymmMaxLossMultiple    = 99.0; // TO BE SET after first bench

    /// <summary>True once the owner has frozen the SYMM bar (non-zero win rate).</summary>
    public static bool SymmBarFrozen => SymmMinWinRatePercent > 0;

    // SpMM vs MKL Sparse BLAS mkl_sparse_*_mm (else vs the naive managed loop floor).
    public const int    SpMMMinWinRatePercent = 0;     // TO BE SET after first bench
    public const double SpMMMaxLossMultiple    = 99.0; // TO BE SET after first bench

    /// <summary>True once the owner has frozen the SpMM bar (non-zero win rate).</summary>
    public static bool SpMMBarFrozen => SpMMMinWinRatePercent > 0;

    // GPU SpMM (#515): AiDotNet's managed CUDA CSR kernels (csr_spmm / _warp / _vec4 /
    // _double) vs native cuSPARSE cusparseSpMM on the authoritative NVIDIA runner.
    // Freeze from GpuSpMMBenchHarness output (AIDOTNET_PERF_RUNNER=1) in a gating
    // commit; P7 (cuSPARSE/rocSPARSE removal) stays blocked until this is green.
    public const int    GpuSpMMMinWinRatePercent = 0;     // TO BE SET after first runner bench
    public const double GpuSpMMMaxLossMultiple    = 99.0; // TO BE SET after first runner bench

    /// <summary>True once the owner has frozen the GPU SpMM bar (non-zero win rate).</summary>
    public static bool GpuSpMMBarFrozen => GpuSpMMMinWinRatePercent > 0;

    // GBMV vs OpenBLAS sgbmv/dgbmv on the authoritative runner.
    public const int    GbmvMinWinRatePercent = 0;     // TO BE SET after first bench
    public const double GbmvMaxLossMultiple    = 99.0; // TO BE SET after first bench

    /// <summary>True once the owner has frozen the GBMV bar (non-zero win rate).</summary>
    public static bool GbmvBarFrozen => GbmvMinWinRatePercent > 0;
}
