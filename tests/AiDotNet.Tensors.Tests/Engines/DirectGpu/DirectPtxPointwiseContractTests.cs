#if NET5_0_OR_GREATER
using System;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxPointwiseContractTests
{
    [Fact]
    public void CoverageManifest_IsCompleteUniqueAndFailClosed()
    {
        Assert.Equal(74, DirectPtxPointwiseCoverageManifest.All.Count);
        Assert.Equal(
            DirectPtxPointwiseCoverageManifest.All.Count,
            DirectPtxPointwiseCoverageManifest.All
                .Select(cell => cell.Api)
                .Distinct(StringComparer.Ordinal)
                .Count());
        Assert.Equal(
            3,
            DirectPtxPointwiseCoverageManifest.All.Count(
                cell => cell.Status == DirectPtxPointwiseCoverageStatus.ExperimentalDirectPtx));
        Assert.DoesNotContain(
            DirectPtxPointwiseCoverageManifest.All,
            cell => cell.Status == DirectPtxPointwiseCoverageStatus.PromotedDirectPtx);

        string api = DirectPtxPointwiseCoverageManifest.All[0].Api;
        Assert.Equal(api, DirectPtxPointwiseCoverageManifest.Get(api).Api);
        Assert.Throws<ArgumentException>(() => DirectPtxPointwiseCoverageManifest.Get(" "));
        Assert.Throws<KeyNotFoundException>(() =>
            DirectPtxPointwiseCoverageManifest.Get("CudaBackend.NotARealPointwiseRoute"));
    }

    [Fact]
    public void SwiGluEmitter_HasPointerOnlyVectorizedAbiAndNoPromotedShape()
    {
        string ptx = PtxFusedSwiGluF32Kernel.EmitPtx(8, 6, 1, 4096);

        Assert.Contains(".target sm_86", ptx);
        Assert.Contains($".visible .entry {PtxFusedSwiGluF32Kernel.EntryPoint}", ptx);
        Assert.Equal(2, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "ex2.approx.f32"));
        Assert.False(PtxFusedSwiGluF32Kernel.IsPromotedShape(1, 4096));
    }

    [Fact]
    public void GeGluEmitter_HasPointerOnlyVectorizedAbiAndNoPromotedShape()
    {
        string ptx = PtxFusedGeGluF32Kernel.EmitPtx(8, 6, 1, 4096);

        Assert.Contains(".target sm_86", ptx);
        Assert.Contains($".visible .entry {PtxFusedGeGluF32Kernel.EntryPoint}", ptx);
        Assert.Equal(2, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx);
        Assert.Equal(2, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(1, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "tanh.approx.f32"));
        Assert.False(PtxFusedGeGluF32Kernel.IsPromotedShape(1, 4096));
    }

    [Fact]
    public void GeGluBackwardEmitter_HasPointerOnlyVectorizedAbiAndNoPromotedShape()
    {
        string ptx = PtxFusedGeGluBackwardF32Kernel.EmitPtx(8, 6, 1, 4096);

        Assert.Contains(".target sm_86", ptx);
        Assert.Contains($".visible .entry {PtxFusedGeGluBackwardF32Kernel.EntryPoint}", ptx);
        Assert.Equal(3, Count(ptx, ".param .u64"));
        Assert.DoesNotContain(".param .u32", ptx);
        Assert.Equal(3, Count(ptx, "ld.global.nc.v4.f32"));
        Assert.Equal(2, Count(ptx, "st.global.v4.f32"));
        Assert.Equal(4, Count(ptx, "tanh.approx.f32"));
        Assert.False(PtxFusedGeGluBackwardF32Kernel.IsPromotedShape(1, 4096));
    }

    private static int Count(string text, string value) =>
        (text.Length - text.Replace(value, string.Empty, StringComparison.Ordinal).Length) /
        value.Length;
}
#endif
