namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue A (#369) task A.7: the codified perf bar for the BlasManaged
/// supply-chain-removal sprint (#368). These constants are set ONCE after
/// <c>artifacts/perf/baseline.json</c> lands and do not move without an
/// explicit user-approved commit (see spec section 7 escape hatches).
///
/// <para>
/// <b>Current state:</b> <see cref="CatalogShapeCount"/> reflects today's
/// <see cref="Catalog.ShapeCatalog.All"/> size. <see cref="MinWinRatePercent"/>
/// and <see cref="MaxLossMultiple"/> are <b>illustrative placeholders</b> that
/// the project owner replaces with real numbers after inspecting baseline.json
/// on the authoritative self-hosted runner (#369 task A.6).
/// </para>
///
/// <para>
/// <b>Workflow:</b>
/// </para>
/// <list type="number">
/// <item>Project owner runs <c>dotnet test --filter PerfHarnessTest</c> on the
/// authoritative runner to produce <c>artifacts/perf/baseline.json</c>.</item>
/// <item>Project owner inspects baseline.json, picks the final perf bar values,
/// and commits this file with <see cref="MinWinRatePercent"/> and
/// <see cref="MaxLossMultiple"/> filled in. Also captures the runner's
/// <c>HardwareFingerprint</c> string into <see cref="TargetHardwareFingerprint"/>.</item>
/// <item>From that point forward, <see cref="PerfBarTest"/> asserts against
/// these constants on every CI run pinned to the authoritative runner.</item>
/// </list>
/// </summary>
public static class PerfBar
{
    /// <summary>
    /// Target win rate: BlasManaged must beat OpenBLAS on at least this percentage
    /// of catalog shapes. Set against the 2026-05-18 baseline that showed 0% wins
    /// (median 23.5× slower than OpenBLAS, worst 381× on Tiny_8x6x4). The 80% bar
    /// is what closes Sub-issue G (#375 — native BLAS removal).
    /// </summary>
    public const int MinWinRatePercent = 80;

    /// <summary>
    /// Ceiling for the worst remaining loss. No shape may be more than this
    /// multiple of OpenBLAS slower. 1.20 = within the noise floor of repeated
    /// measurement. Sub-issue G can only merge when every shape clears this.
    /// </summary>
    public const double MaxLossMultiple = 1.20;

    /// <summary>
    /// <see cref="Catalog.ShapeCatalog.All"/> size as of A.4 commit (37 workload +
    /// 19 instrumented - 2 dedups = 54 shapes). <see cref="PerfBarTest.Catalog_Size_Matches_PerfBar"/>
    /// fails if a new shape is added without updating this constant — by design, the
    /// bar must be re-evaluated when the catalog grows.
    /// </summary>
    public const int CatalogShapeCount = 54;

    /// <summary>
    /// Hardware fingerprint of the authoritative runner where the bar was set.
    /// The perf test skips when running on a different host because perf numbers
    /// aren't comparable across hardware. Captured from
    /// <c>HardwareFingerprint.Current.ToString()</c> on 2026-05-18.
    /// </summary>
    public const string TargetHardwareFingerprint = "x64-amd-avx2-cpu16";
}
