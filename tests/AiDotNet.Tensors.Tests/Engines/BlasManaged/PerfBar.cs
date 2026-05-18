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
    /// of catalog shapes. <b>Placeholder value</b> — actual bar set by project owner
    /// after baseline.json inspection.
    /// </summary>
    public const int MinWinRatePercent = 80;  // PLACEHOLDER — SET BY PROJECT OWNER (#369 A.7)

    /// <summary>
    /// Ceiling for the worst remaining loss. No shape may be more than this
    /// multiple of OpenBLAS slower. <b>Placeholder value</b> — actual bar set by
    /// project owner after baseline.json inspection.
    /// </summary>
    public const double MaxLossMultiple = 1.20;  // PLACEHOLDER — SET BY PROJECT OWNER (#369 A.7)

    /// <summary>
    /// <see cref="Catalog.ShapeCatalog.All"/> size as of A.4 commit (37 workload +
    /// 19 instrumented - 2 dedups = 54 shapes). <see cref="PerfBarTest.Catalog_Size_Matches_PerfBar"/>
    /// fails if a new shape is added without updating this constant — by design, the
    /// bar must be re-evaluated when the catalog grows.
    /// </summary>
    public const int CatalogShapeCount = 54;

    /// <summary>
    /// Hardware fingerprint of the authoritative runner. The perf test skips
    /// when running on a different host (numbers aren't comparable across hardware).
    /// <b>Placeholder value</b> — actual fingerprint captured from
    /// <c>HardwareFingerprint.Current.ToString()</c> on the runner.
    /// </summary>
    public const string TargetHardwareFingerprint = "<runner-fingerprint>";  // PLACEHOLDER — SET BY PROJECT OWNER (#369 A.7)
}
