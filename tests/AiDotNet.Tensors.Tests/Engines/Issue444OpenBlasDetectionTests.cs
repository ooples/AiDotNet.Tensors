// Issue #444 — the native CPU BLAS diagnostic must reflect the ACTUAL load
// state, not a hardcoded `false`.
//
// Background: NativeLibraryDetector.DetectOpenBlas()/DetectMkl() previously
// returned a hardcoded `false` (a stale assumption from when matmul ran only
// through SimdGemm). After BLAS-default-on (#319), the engine's GEMM dispatch
// routes through Helpers.BlasProvider, which loads libopenblas by default.
// The hardcoded false made AccelerationDiagnostics / PlatformDetector report
// "OpenBLAS: Not found" even when libopenblas was deployed, loaded, and
// actively servicing matmul — a diagnostic that contradicted reality.
//
// These tests lock the diagnostic to the real provider state so the divergence
// cannot regress. The Tensors test project does NOT reference
// AiDotNet.Native.OpenBLAS (the library is supply-chain-independent), so these
// run in the "DLL absent" regime — but the asserted invariants hold in BOTH
// regimes by construction (they tie the diagnostic to BlasProvider rather than
// to a fixed expected value).

using AiDotNet.Tensors.Engines;
using AiDotNet.Tensors.Helpers;
using Xunit;
using Xunit.Abstractions;

namespace AiDotNet.Tensors.Tests.Engines;

public class Issue444OpenBlasDetectionTests
{
    private readonly ITestOutputHelper _output;
    public Issue444OpenBlasDetectionTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void Detector_HasCpuBlas_MatchesActualProviderLoadState()
    {
        // Core #444 regression. BlasProvider.IsAvailable is exactly what the
        // engine's GEMM dispatch gates on; HasCpuBlas must equal it — no false
        // "Not found" while OpenBLAS is loaded and servicing matmul, and no
        // false "Available" while it isn't. The pre-fix code hardcoded
        // HasCpuBlas=false and so violated this whenever IsAvailable was true.
        _output.WriteLine($"BlasProvider.IsAvailable = {BlasProvider.IsAvailable}");
        _output.WriteLine($"NativeLibraryDetector.HasCpuBlas = {NativeLibraryDetector.HasCpuBlas}");
        Assert.Equal(BlasProvider.IsAvailable, NativeLibraryDetector.HasCpuBlas);
    }

    [Fact]
    public void Detector_HasOpenBlas_MatchesActiveOpenBlasProvider()
    {
        Assert.Equal(BlasProvider.IsOpenBlasActive, NativeLibraryDetector.HasOpenBlas);
    }

    [Fact]
    public void Detector_HasMkl_MatchesActiveMklProvider()
    {
        Assert.Equal(BlasProvider.IsMklActive, NativeLibraryDetector.HasMkl);
    }

    [Fact]
    public void LoadDiagnostic_PresentWhenUnavailable_AbsentWhenLoaded()
    {
        // When CPU BLAS is unavailable, the diagnostic must explain WHY (so a
        // deployed-but-not-loaded libopenblas is diagnosable in place — the
        // exact gap #444 reports). When it loaded, there is no error to report.
        var diagnostic = NativeLibraryDetector.OpenBlasLoadDiagnostic;
        _output.WriteLine($"HasCpuBlas={NativeLibraryDetector.HasCpuBlas}, diagnostic='{diagnostic ?? "(null)"}'");

        // Available  -> diagnostic null/empty;  Unavailable -> diagnostic present.
        Assert.Equal(NativeLibraryDetector.HasCpuBlas, string.IsNullOrEmpty(diagnostic));
    }

    [Fact]
    public void StatusSummary_ExplainsWhy_WhenOpenBlasNotFound()
    {
        var summary = NativeLibraryDetector.GetStatusSummary();
        _output.WriteLine(summary);
        Assert.Contains("OpenBLAS:", summary);

        if (!NativeLibraryDetector.HasOpenBlas)
        {
            var diagnostic = NativeLibraryDetector.OpenBlasLoadDiagnostic;
            if (!string.IsNullOrEmpty(diagnostic))
            {
                // The summary's OpenBLAS line embeds the captured reason rather
                // than a bare "Not found".
                Assert.Contains(diagnostic, summary);
            }
        }
    }

    [Fact]
    public void PlatformDetector_Capabilities_AgreeWithDetector()
    {
        // PlatformDetector copies NativeLibraryDetector.Status into its
        // capability struct, so the two diagnostic surfaces must stay
        // consistent — fixing the detector fixes both.
        var caps = PlatformDetector.Capabilities;
        Assert.Equal(NativeLibraryDetector.HasOpenBlas, caps.HasOpenBlas);
        Assert.Equal(NativeLibraryDetector.HasMkl, caps.HasMkl);
    }
}
