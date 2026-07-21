#if !NETFRAMEWORK

using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.OpParity;

/// <summary>
/// Third parity phase, driven over the WHOLE registry.
///
/// CheckForward verifies the forward kernel; CheckBackward verifies the backward kernel; neither verifies
/// the WIRING — that the forward records a tape node whose saved state makes ComputeGradients produce the
/// right gradient. GPU Scatter passed both existing phases while silently corrupting d(input) by 3.14e-01.
///
/// Sources are DISCOVERED from the tape (leaves = inputs no recorded op produced), so this needs no
/// per-op declaration and covers all 475 OpCase definitions without editing any of them.
/// </summary>
[Collection("OpParity")]
public class TapeGradientParityTests
{
    private readonly OpParityFixture _fx;
    public TapeGradientParityTests(OpParityFixture fx) => _fx = fx;

    public static IEnumerable<object[]> AllOps() =>
        OpParityRegistry.All().Select(op => new object[] { op.Name });

    [SkippableTheory]
    [MemberData(nameof(AllOps))]
    public void TapeGradient_matches_cpu(string opName)
    {
        var op = OpParityRegistry.All().First(o => o.Name == opName);
        TapeGradientParityHarness.CheckTapeGradient(op, _fx);
    }
}
#endif
