using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape FP32 index-select (issue #844).
///
/// The assertions here mostly exist to pin one thing: index-select and gather
/// are NOT the same op. The established index_select takes float indices and
/// applies a C (int) cast, while the embedding_forward kernel behind Gather
/// takes int32 indices. Reusing one kernel for both would reinterpret the index
/// buffer - the exact bug embedding_forward's own source documents having been
/// bitten by - so these tests assert the float path stays float.
/// </summary>
public class DirectPtxIndexSelectTests
{
    [Fact]
    public void Emitter_ConvertsFloatIndicesNumericallyNotByReinterpretation()
    {
        string ptx = PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, 4096, 128);
        Assert.Contains(".visible .entry aidotnet_fused_index_select_f32(", ptx);
        Assert.Contains("op=index-select-f32", ptx);

        // The index is LOADED as a float and converted with a numeric truncation
        // toward zero, matching the reference's (int) cast.
        Assert.Contains("ld.global.nc.f32 %f0, [%rd4];", ptx);
        Assert.Contains("cvt.rzi.s32.f32 %r5, %f0;", ptx);
        // A bit reinterpretation would use mov.b32 on the index - it must not.
        Assert.DoesNotContain("mov.b32 %r5, %f0;", ptx);
        // Nor may the index be read as an integer directly.
        Assert.DoesNotContain("ld.global.nc.s32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("ld.global.nc.u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_ReplacesTheDivideAndRemainderWithAShiftAndMask()
    {
        string ptx = PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, 4096, 128);
        // innerSize 128 = 1 << 7, so i = idx >> 7 and j = idx & 127.
        Assert.Contains("shr.u32 %r3, %r2, 7;", ptx);
        Assert.Contains("and.b32 %r4, %r2, 127;", ptx);
        Assert.DoesNotContain("div.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("rem.", ptx, StringComparison.Ordinal);

        // A different inner size shifts by a different amount.
        string wide = PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, 4096, 512);
        Assert.Contains("shr.u32 %r3, %r2, 9;", wide);
        Assert.Contains("and.b32 %r4, %r2, 511;", wide);
    }

    [Fact]
    public void Emitter_IsPointerOnlyAndBranchFree()
    {
        string ptx = PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, 1024, 64);
        Assert.Equal(3, Count(ptx, "ld.param.u64"));
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
        // The launch covers exactly numIndices * innerSize elements, so the
        // reference's "if (idx >= total) return" guard is unnecessary.
        Assert.DoesNotContain("bra", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("setp.", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_ScalesTheTableOffsetAsSignedSoNegativeIndicesStayNegative()
    {
        string ptx = PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, 4096, 128);
        // The index is signed after the cast. Widening it as unsigned would turn
        // a negative index into a huge positive offset instead of the
        // out-of-range read the reference performs.
        Assert.Contains("mul.wide.s32 %rd5, %r7, 4;", ptx);
        Assert.DoesNotContain("mul.wide.u32 %rd5,", ptx);
    }

    [Fact]
    public void ShapeDomain_IsClosedAndUnpromotedWithoutEvidence()
    {
        Assert.True(PtxFusedIndexSelectF32Kernel.IsSupportedShape(1024, 32));
        Assert.True(PtxFusedIndexSelectF32Kernel.IsSupportedShape(65_536, 512));
        Assert.False(PtxFusedIndexSelectF32Kernel.IsSupportedShape(1000, 128));
        // 96 is not a power of two, so the shift-and-mask split would be wrong.
        Assert.False(PtxFusedIndexSelectF32Kernel.IsSupportedShape(1024, 96));
        Assert.False(PtxFusedIndexSelectF32Kernel.IsPromotedShape(1024, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, 1024, 96));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedIndexSelectF32Kernel.EmitPtx(8, 6, 1000, 128));
    }

    [Fact]
    public void ArchitectureGate_IsSeparateFromGatherAndFailsClosedOutsideSm86()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedIndexSelect(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedIndexSelect(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedIndexSelect(8, 7));
        Assert.False(DirectPtxArchitecture.HasValidatedIndexSelect(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedIndexSelect(9, 0));
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
