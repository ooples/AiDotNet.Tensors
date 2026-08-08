using System;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

/// <summary>
/// Focused coverage for the exact-shape warp-per-plane global max pool
/// (issue #842). The established global_maxpool2d kernel walks each HxW plane
/// serially on a single thread; this specialization gives each plane a warp.
/// Only the value path is ported - the arg-max index path stays on the
/// established kernel, and these assertions pin that boundary.
/// </summary>
public class DirectPtxGlobalMaxPoolTests
{
    [Fact]
    public void Emitter_ReducesOnePlanePerWarpAndIsPointerOnly()
    {
        string ptx = PtxFusedGlobalMaxPoolF32Kernel.EmitPtx(8, 6, 2048, 128);
        Assert.Contains(".visible .entry aidotnet_fused_global_maxpool_f32(", ptx);
        Assert.Contains("op=global-maxpool", ptx);
        Assert.Contains("exact-shape rows=2048 spatial=128", ptx);
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        // 128 spatial / 32 lanes = one v4 load per lane.
        Assert.Equal(1, Count(ptx, "ld.global.nc.v4.f32"));
        // Lane zero writes the single value for its plane.
        Assert.Equal(1, Count(ptx, "st.global.f32"));
        Assert.Contains("setp.eq.u32 %p0, %r3, 0;", ptx);
        Assert.DoesNotContain(".shared", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".local", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain(".param .u32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_SeedsNegativeInfinitySoNaNPlanesMatchTheReference()
    {
        string ptx = PtxFusedGlobalMaxPoolF32Kernel.EmitPtx(8, 6, 2048, 128);
        // The reference seeds maxVal = -INFINITY and updates only on a strict
        // val > maxVal, so NaN never wins. max.f32 returns the non-NaN operand,
        // so an all-NaN plane reduces to -inf in both.
        Assert.Contains("mov.f32 %f4, 0fFF800000;", ptx);
        Assert.DoesNotContain("0f7F800000", ptx);   // must not seed +inf
        // 4 lane folds + 5 warp folds.
        Assert.Equal(9, Count(ptx, "max.f32 %f4, %f4, %f"));
        Assert.DoesNotContain("min.f32", ptx, StringComparison.Ordinal);
    }

    [Fact]
    public void Emitter_ShufflesThroughB32RegistersNotFloatRegisters()
    {
        string ptx = PtxFusedGlobalMaxPoolF32Kernel.EmitPtx(8, 6, 2048, 64);
        Assert.Equal(5, Count(ptx, "shfl.sync.bfly.b32 %r6, %r5,"));
        Assert.DoesNotContain("shfl.sync.bfly.b32 %f", ptx);
    }

    [Fact]
    public void Emitter_ProducesNoIndexOutput()
    {
        string ptx = PtxFusedGlobalMaxPoolF32Kernel.EmitPtx(8, 6, 2048, 128);
        // The arg-max path needs a (value, index) pair shuffle and is NOT ported.
        // Two pointers and one float store are the whole contract.
        Assert.Equal(2, Count(ptx, "ld.param.u64"));
        Assert.DoesNotContain("st.global.u32", ptx, StringComparison.Ordinal);
        Assert.DoesNotContain("st.global.s32", ptx, StringComparison.Ordinal);
        // Exactly two parameter declarations: no indices pointer is declared.
        // ".param .u64" (with the space) matches declarations only, not the
        // "ld.param.u64" loads.
        Assert.Equal(2, Count(ptx, ".param .u64"));
    }

    [Fact]
    public void ShapeDomain_MirrorsTheAveragePoolBucketsAndFailsClosed()
    {
        Assert.True(PtxFusedGlobalMaxPoolF32Kernel.IsSupportedShape(256, 128));
        Assert.True(PtxFusedGlobalMaxPoolF32Kernel.IsSupportedShape(2048, 64));
        Assert.True(PtxFusedGlobalMaxPoolF32Kernel.IsSupportedShape(2048, 128));
        Assert.True(PtxFusedGlobalMaxPoolF32Kernel.IsSupportedShape(8192, 128));
        Assert.False(PtxFusedGlobalMaxPoolF32Kernel.IsSupportedShape(256, 64));
        Assert.False(PtxFusedGlobalMaxPoolF32Kernel.IsSupportedShape(2048, 96));
        Assert.False(PtxFusedGlobalMaxPoolF32Kernel.IsPromotedShape(2048, 128));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            PtxFusedGlobalMaxPoolF32Kernel.EmitPtx(8, 6, 2048, 96));
    }

    [Fact]
    public void ArchitectureGate_FailsClosedOutsideSm86()
    {
        Assert.True(DirectPtxArchitecture.HasValidatedGlobalMaxPool(8, 6));
        Assert.False(DirectPtxArchitecture.HasValidatedGlobalMaxPool(8, 0));
        Assert.False(DirectPtxArchitecture.HasValidatedGlobalMaxPool(8, 7));
        Assert.False(DirectPtxArchitecture.HasValidatedGlobalMaxPool(8, 9));
        Assert.False(DirectPtxArchitecture.HasValidatedGlobalMaxPool(9, 0));
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
