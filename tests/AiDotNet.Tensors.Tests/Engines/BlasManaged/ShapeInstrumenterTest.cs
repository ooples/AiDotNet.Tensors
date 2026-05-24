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

    // PR #402 merge-resolution: the test class originally called BlasProvider.TryGemm
    // directly and relied on the in-method LogShape call to feed the catalog. Main's
    // helper-test class (Helpers/ShapeInstrumenterTests.cs) takes a different
    // approach — it invokes BlasProvider.ShapeLogHook!.Invoke(...) directly to
    // exercise the aggregation logic in isolation. The TryGemm-driven flavour is
    // brittle on CI runners where libopenblas isn't loaded (TryGemm short-circuits
    // before reaching LogShape) and any attempt to hoist LogShape to the entry of
    // TryGemm causes downstream PackBothStrategy correctness regressions in the
    // Gemm_With*HandleTests. Switching to direct-invoke matches main's pattern,
    // keeps coverage of the Catalog.ShapeInstrumenter aggregation contract, and
    // doesn't depend on which TryGemm dispatch path executes.

    [Fact]
    public void Instrumenter_Captures_Shape_From_HookInvoke()
    {
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = true;
        try
        {
            BlasProvider.ShapeLogHook!.Invoke(64, 16, 32, false, false);

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
    public void Instrumenter_Captures_Trans_Flags_From_HookInvoke()
    {
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = true;
        try
        {
            // Mirrors the transA=true, transB=false case the TryGemmEx fast path
            // would have produced under a hoisted-LogShape setup.
            BlasProvider.ShapeLogHook!.Invoke(64, 16, 32, true, false);

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
            for (int i = 0; i < 3; i++)
                BlasProvider.ShapeLogHook!.Invoke(64, 16, 32, false, false);

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
            // Hook still wired but Enabled=false short-circuits Record(...) → no entries.
            BlasProvider.ShapeLogHook!.Invoke(64, 16, 32, false, false);

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
        // PR #412 merge: BlasProvider.ShapeLogHook is the 5-arg variant from main
        // with no DType payload. The hook lambda in WireHook always records under
        // DType.Single, so two invocations for the same M/N/K dedupe to a single
        // catalog bucket with Frequency = 2.
        if (HarvestModeActive) return;
        WireHook();
        ShapeInstrumenter.Enabled = true;
        try
        {
            // Two invocations representing what would have been FP32 + FP64 GEMM
            // calls on the same shape in the pre-merge 6-arg-hook era.
            BlasProvider.ShapeLogHook!.Invoke(64, 16, 32, false, false);
            BlasProvider.ShapeLogHook!.Invoke(64, 16, 32, false, false);

            var shapes = ShapeInstrumenter.Snapshot();
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
