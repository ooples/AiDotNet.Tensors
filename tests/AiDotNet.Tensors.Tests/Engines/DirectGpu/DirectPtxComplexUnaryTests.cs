using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the interleaved-complex unary maps (issue #850):
/// conjugate and magnitude. The two share a memory schedule but differ in
/// output extent, so these assertions pin both the arithmetic and the stride.
/// </summary>
public class DirectPtxComplexUnaryTests
{
    [Fact]
    public void ConjugateEmitter_FlipsOnlyTheImaginarySignAndKeepsThePairStride()
    {
        string ptx = PtxFusedComplexUnaryF32Kernel.EmitPtx(
            8, 6, DirectPtxComplexUnaryOp.Conjugate, 262_144);
        Assert.Contains(".visible .entry aidotnet_fused_complex_conjugate_f32(", ptx);
        Assert.Contains("op=complex-conjugate", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.Equal(1, Count(ptx, "ld.global.nc.v2.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v2.f32"));

        // Exactly one negate, applied to the imaginary lane, and the real lane
        // is stored untouched. neg.f32 flips the sign bit, so NaN payloads and
        // signed zeros survive as the reference's unary minus leaves them.
        Assert.Equal(1, Count(ptx, "neg.f32"));
        Assert.Contains("neg.f32 %f2, %f1;", ptx);
        Assert.Contains("st.global.v2.f32 [%rd6], {%f0, %f2};", ptx);
        // Pair in, pair out: both sides use the 8-byte stride.
        Assert.Contains("mul.wide.u32 %rd2, %r2, 8;", ptx);
        Assert.Contains("add.u64 %rd6, %rd1, %rd2;", ptx);
        Assert.DoesNotContain("sqrt", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void MagnitudeEmitter_MatchesTheReferenceRoundingAndHalvesTheOutputStride()
    {
        string ptx = PtxFusedComplexUnaryF32Kernel.EmitPtx(
            8, 6, DirectPtxComplexUnaryOp.Magnitude, 262_144);
        Assert.Contains(".visible .entry aidotnet_fused_complex_magnitude_f32(", ptx);
        Assert.Contains("op=complex-magnitude", ptx);

        // sqrtf(re*re + im*im) with the multiplies UNFUSED. An fma would be both
        // faster and more accurate, and would therefore disagree with the
        // kernel this replaces - so its absence is the assertion.
        Assert.Contains("mul.rn.f32 %f2, %f0, %f0;", ptx);
        Assert.Contains("mul.rn.f32 %f3, %f1, %f1;", ptx);
        Assert.Contains("add.rn.f32 %f2, %f2, %f3;", ptx);
        Assert.DoesNotContain("fma.", ptx, StringComparison.Ordinal);
        // IEEE square root, not the fast approximation.
        Assert.Contains("sqrt.rn.f32 %f2, %f2;", ptx);
        Assert.DoesNotContain("sqrt.approx", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("rsqrt", ptx, StringComparison.Ordinal);

        // Pair in (8-byte stride), scalar out (4-byte stride).
        Assert.Contains("mul.wide.u32 %rd2, %r2, 8;", ptx);
        Assert.Contains("mul.wide.u32 %rd5, %r2, 4;", ptx);
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.DoesNotContain("st.global.v2.f32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("neg.f32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void BothOperators_AreBranchFreeAndRegisterResident()
    {
        foreach (var op in new[]
                 {
                     DirectPtxComplexUnaryOp.Conjugate,
                     DirectPtxComplexUnaryOp.Magnitude
                 })
        {
            string ptx = PtxFusedComplexUnaryF32Kernel.EmitPtx(8, 6, op, 65_536);
            // The launch covers exactly the pair count, so the reference's
            // "if (i >= numPairs) return" guard is unnecessary.
            Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain("setp.", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
            Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
            Assert.Equal(2, Count(ptx, ".param .u64"));
        }
    }

    [Fact]
    public void EachOperatorGetsItsOwnModule()
    {
        Assert.Equal("aidotnet_fused_complex_conjugate_f32",
            PtxFusedComplexUnaryF32Kernel.EntryPointFor(DirectPtxComplexUnaryOp.Conjugate));
        Assert.Equal("aidotnet_fused_complex_magnitude_f32",
            PtxFusedComplexUnaryF32Kernel.EntryPointFor(DirectPtxComplexUnaryOp.Magnitude));
        Assert.NotEqual(
            PtxFusedComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxComplexUnaryOp.Conjugate, 65_536),
            PtxFusedComplexUnaryF32Kernel.EmitPtx(8, 6, DirectPtxComplexUnaryOp.Magnitude, 65_536));
    }

    [Fact]
    public void ShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedComplexUnaryF32Kernel.IsSupportedShape(65_536));
        Assert.True(PtxFusedComplexUnaryF32Kernel.IsSupportedShape(4_194_304));
        Assert.False(PtxFusedComplexUnaryF32Kernel.IsSupportedShape(65_535));
        Assert.False(PtxFusedComplexUnaryF32Kernel.IsPromotedShape(65_536));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedComplexUnaryF32Kernel.EmitPtx(
                8, 6, DirectPtxComplexUnaryOp.Conjugate, 1_000));
    }

    [Fact]
    public void ArchitectureGate_FailsClosedOutsideSm86()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedComplexUnary(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexUnary(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexUnary(8, 7));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexUnary(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedComplexUnary(9, 0));
    }

    private static int Count(string text, string value)
    {
        int count = 0, offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }
}
