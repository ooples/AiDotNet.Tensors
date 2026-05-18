using System;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.Tests.Engines.BlasManaged.Catalog;
using Xunit;

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
    public void Instrumenter_Distinguishes_Single_From_Double_Dtype()
    {
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
            Assert.Equal(2, shapes.Count(s => s.M == 64 && s.N == 16 && s.K == 32));
            Assert.Contains(shapes, s => s.Dtype == DType.Single && s.M == 64 && s.N == 16 && s.K == 32);
            Assert.Contains(shapes, s => s.Dtype == DType.Double && s.M == 64 && s.N == 16 && s.K == 32);
        }
        finally
        {
            RestoreCleanState();
        }
    }

    private static void WireHook()
    {
        BlasProvider.ShapeLogHook = (m, n, k, transA, transB, dtype) =>
            ShapeInstrumenter.Record(m, n, k, transA, transB,
                dtype == typeof(float) ? DType.Single : DType.Double);
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
