using System;
using System.Linq;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;
// Merge resolution (#402 + main #412): both branches added a `ShapeInstrumenter` type.
// This file targets the test-side `Catalog.ShapeInstrumenter` (DType-aware, harvest
// mode, deterministic JSON dump on exit). Main's `AiDotNet.Tensors.Helpers.ShapeInstrumenter`
// is a simpler 5-arg variant used by production benchmarks; we reach it explicitly via
// fully-qualified `BlasProvider` access. Dropping the `Helpers` using directive resolves
// the CS0104 ambiguity without renaming either class.
using BlasProvider = AiDotNet.Tensors.Helpers.BlasProvider;

namespace AiDotNet.Tensors.Tests.Engines.BlasManaged;

/// <summary>
/// Sub-issue A (#369) task A.2: verifies the test-only shape instrumentation hook
/// in BlasProvider.cs is invoked from every public Try* GEMM entry point and that
/// <see cref="ShapeInstrumenter"/> deduplicates + frequency-counts shapes correctly.
///
/// <para>
/// Lives in the BlasManaged-Stats-Serial collection because ShapeInstrumenter's
/// state (Enabled flag, counts dictionary, BlasProvider.ShapeLogHook) is process-wide.
/// </para>
///
/// <para>
/// <b>Harvest-mode skip:</b> when <c>AIDOTNET_INSTRUMENT_SHAPES=1</c> is set,
/// <see cref="ShapeInstrumenterBootstrap"/> has globally installed the hook and
/// started recording. These mechanical tests would clobber that state (via their
/// per-test reset + null-out), so they no-op during the harvest pass. The instrumenter
/// mechanics are still verified on every normal test run.
/// </para>
/// </summary>
[Collection("BlasManaged-Stats-Serial")]
public class ShapeInstrumenterTest
{
    private static bool HarvestModeActive
        => Environment.GetEnvironmentVariable("AIDOTNET_INSTRUMENT_SHAPES") == "1";

    public ShapeInstrumenterTest()
    {
        if (HarvestModeActive) return;

        // Each test starts from a clean slate. The hook is wired by the test bodies
        // that need it (so other tests in the same serial collection don't pay for
        // a recorded null-check on every BlasProvider call).
        ShapeInstrumenter.Reset();
        ShapeInstrumenter.Enabled = false;
        BlasProvider.ShapeLogHook = null;
    }

    [Fact]
    public void Instrumenter_Captures_Shape_From_TryGemm()
    {
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = true;
        try
        {
            var a = new float[64 * 32];
            var b = new float[32 * 16];
            var c = new float[64 * 16];
            BlasProvider.TryGemm(64, 16, 32, a, 0, 32, b, 0, 16, c, 0, 16);

            var shapes = ShapeInstrumenter.Snapshot();
            Assert.Contains(
                shapes,
                s => s.M == 64 && s.N == 16 && s.K == 32 && s.Dtype == DType.Single
                  && !s.TransA && !s.TransB);
        }
        finally
        {
            RestoreCleanState();
        }
    }

    [Fact]
    public void Instrumenter_Captures_Trans_Flags_From_TryGemmEx()
    {
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = true;
        try
        {
            var a = new float[32 * 64];
            var b = new float[32 * 16];
            var c = new float[64 * 16];
            // transA=true, transB=false; lda/ldb reflect transposed/non-transposed views.
            BlasProvider.TryGemmEx(64, 16, 32, a, 0, 64, true, b, 0, 16, false, c, 0, 16);

            var shapes = ShapeInstrumenter.Snapshot();
            Assert.Contains(
                shapes,
                s => s.M == 64 && s.N == 16 && s.K == 32 && s.Dtype == DType.Single
                  && s.TransA && !s.TransB);
        }
        finally
        {
            RestoreCleanState();
        }
    }

    [Fact]
    public void Instrumenter_Deduplicates_Identical_Shapes_And_Counts_Frequency()
    {
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = true;
        try
        {
            var a = new float[64 * 32];
            var b = new float[32 * 16];
            var c = new float[64 * 16];
            for (int i = 0; i < 3; i++)
                BlasProvider.TryGemm(64, 16, 32, a, 0, 32, b, 0, 16, c, 0, 16);

            var shapes = ShapeInstrumenter.Snapshot();
            var match = shapes.Single(s => s.M == 64 && s.N == 16 && s.K == 32 && s.Dtype == DType.Single);
            Assert.Equal(3, match.Frequency);
        }
        finally
        {
            RestoreCleanState();
        }
    }

    [Fact]
    public void Instrumenter_Records_Nothing_When_Disabled()
    {
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = false;
        try
        {
            var a = new float[64 * 32];
            var b = new float[32 * 16];
            var c = new float[64 * 16];
            BlasProvider.TryGemm(64, 16, 32, a, 0, 32, b, 0, 16, c, 0, 16);

            Assert.Empty(ShapeInstrumenter.Snapshot());
        }
        finally
        {
            RestoreCleanState();
        }
    }

    [Fact]
    public void Instrumenter_CollapsesSameShape_AcrossDtypes()
    {
        // Merge resolution (#402 + main #412): the merged BlasProvider.ShapeLogHook
        // is the 5-arg variant from main (no DType payload). The bootstrap +
        // WireHook record under DType.Single as a sentinel, so two TryGemm calls
        // with the same M/N/K but different element types DEDUPE to a single
        // catalog entry. Pre-merge the test asserted two separate buckets
        // (DType.Single + DType.Double); we now pin the new contract: one
        // sentinel-dtype bucket whose call count covers both precisions.
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = true;
        try
        {
            var af = new float[64 * 32];
            var bf = new float[32 * 16];
            var cf = new float[64 * 16];
            BlasProvider.TryGemm(64, 16, 32, af, 0, 32, bf, 0, 16, cf, 0, 16);

            var ad = new double[64 * 32];
            var bd = new double[32 * 16];
            var cd = new double[64 * 16];
            BlasProvider.TryGemm(64, 16, 32, ad, 0, 32, bd, 0, 16, cd, 0, 16);

            var shapes = ShapeInstrumenter.Snapshot();
            // Sentinel-bucket dedup: both calls land in the same row.
            // Bucket key is (M, N, K, transA, transB, Dtype) — with Dtype always
            // pinned to Single by the merged 5-arg hook, the FP32 and FP64
            // observations collapse to one Shape entry with Frequency = 2.
            var matching = shapes.Where(s => s.M == 64 && s.N == 16 && s.K == 32).ToList();
            Assert.Single(matching);
            Assert.Equal(DType.Single, matching[0].Dtype);
            Assert.Equal(2, matching[0].Frequency);
        }
        finally
        {
            RestoreCleanState();
        }
    }

    private static void WireHook()
    {
        // Match the 5-arg shape of the merged BlasProvider.ShapeLogHook contract
        // (main #412 baseline). DType is no longer carried by the hook payload —
        // record under DType.Single as a sentinel. See ShapeInstrumenterBootstrap
        // for the matching rationale.
        BlasProvider.ShapeLogHook = (m, n, k, transA, transB) =>
            ShapeInstrumenter.Record(m, n, k, transA, transB, DType.Single);
    }

    private static void RestoreCleanState()
    {
        // In harvest mode we never reached here (guard at top of each test), but
        // double-check defensively to make absolutely sure we never nuke the
        // bootstrap-installed hook.
        if (HarvestModeActive) return;
        ShapeInstrumenter.Enabled = false;
        BlasProvider.ShapeLogHook = null;
        ShapeInstrumenter.Reset();
    }
}
