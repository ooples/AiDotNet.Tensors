using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Drives <see cref="TapeGradientParityHarness.CheckTapeGradient"/> for every declared tape case.
///
/// This is the third parity phase. CheckForward verifies the forward kernel; CheckBackward verifies the
/// backward kernel; neither verifies the WIRING — that the forward records a tape node whose saved state
/// makes ComputeGradients produce the right gradient. GPU Scatter passed both existing phases while
/// silently corrupting d(input) by 3.14e-01, which is what this phase exists to catch.
///
/// One test per op rather than a Theory over the registry: a failure then names the op directly, and an op
/// whose GPU kernel poisons the host cannot take the rest of the phase down with it.
/// </summary>
[Collection("OpParity")]
public class TapeGradientParityTests
{
    private readonly OpParityFixture _fx;

    public TapeGradientParityTests(OpParityFixture fx)
    {
        _fx = fx;
        TapeGradientCases.Register();
    }

    private void Run(string opName) => TapeGradientParityHarness.CheckTapeGradient(opName, _fx);

    [SkippableFact] public void TapeGrad_TensorCosh() => Run("TensorCosh");
    [SkippableFact] public void TapeGrad_TensorSinh() => Run("TensorSinh");
    [SkippableFact] public void TapeGrad_RFFT() => Run("RFFT");
    [SkippableFact] public void TapeGrad_RBFKernel() => Run("RBFKernel");
    [SkippableFact] public void TapeGrad_Upsample3D() => Run("Upsample3D");
    [SkippableFact] public void TapeGrad_AffineGrid() => Run("AffineGrid");
    [SkippableFact] public void TapeGrad_MaxPool3DWithIndices() => Run("MaxPool3DWithIndices");

    /// <summary>
    /// Baseline entries must stay SKIPPED, never silently start passing unnoticed. If one of these begins
    /// to pass, the underlying defect was fixed and the entry should be removed so the ledger keeps
    /// reflecting real debt.
    /// </summary>
    [SkippableFact] public void TapeGrad_Scatter_isBaselined() => Run("Scatter");
    [SkippableFact] public void TapeGrad_Sparsemax_isBaselined() => Run("Sparsemax");
}
