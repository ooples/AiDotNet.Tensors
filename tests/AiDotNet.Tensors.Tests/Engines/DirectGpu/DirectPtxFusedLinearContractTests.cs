#if NET5_0_OR_GREATER
using System;
using System.Linq;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;
using Xunit;

namespace AiDotNet.Tensors.Tests.Engines.DirectGpu;

public sealed class DirectPtxFusedLinearContractTests
{
    [Theory]
    [InlineData(512)]
    [InlineData(1024)]
    public void M16WeightStaging_CoversEveryTileExactlyOnce(int inputFeatures)
    {
        int[] offsets = PtxFusedLinearGeluFp16M16Kernel
            .GetWeightStagingDestinationOffsets(inputFeatures);
        int weightPanelBytes = PtxFusedLinearGeluFp16M16Kernel.OutputsPerBlock *
            PtxFusedLinearGeluFp16M16Kernel.GetKPerPanel(inputFeatures) * sizeof(ushort);
        int expectedTiles = weightPanelBytes / 16;

        Assert.Equal(expectedTiles, offsets.Length);
        Assert.Equal(expectedTiles, offsets.Distinct().Count());
        Assert.Equal(
            Enumerable.Range(0, expectedTiles).Select(tile => tile * 16),
            offsets.OrderBy(offset => offset));
    }

    [Fact]
    public void Int8LayoutContract_AcceptsCanonicalDecodeProjection()
    {
        CuBlasLtMatmul.ValidateInt8LayoutArguments(
            new IntPtr(4), 4096, 1024, false,
            new IntPtr(8), 1, false,
            new IntPtr(12));
    }

    [Fact]
    public void Int8LayoutContract_RejectsMisalignedPointers()
    {
        Assert.Throws<ArgumentException>(() =>
            CuBlasLtMatmul.ValidateInt8LayoutArguments(
                new IntPtr(3), 4096, 1024, false,
                new IntPtr(8), 1, false,
                new IntPtr(12)));
    }

    [Fact]
    public void Int8LayoutContract_RejectsInvalidTransposeLeadingDimension()
    {
        Assert.Throws<ArgumentException>(() =>
            CuBlasLtMatmul.ValidateInt8LayoutArguments(
                new IntPtr(4), 4096, 1024, false,
                new IntPtr(8), 1, true,
                new IntPtr(12)));
    }

    [Fact]
    public void PhysicalTypeElementSize_IsExhaustive()
    {
        (DirectPtxPhysicalType Type, int Bytes)[] cases =
        [
            (DirectPtxPhysicalType.Int8, 1),
            (DirectPtxPhysicalType.UInt8, 1),
            (DirectPtxPhysicalType.Float16, 2),
            (DirectPtxPhysicalType.BFloat16, 2),
            (DirectPtxPhysicalType.Float32, 4),
            (DirectPtxPhysicalType.Int32, 4)
        ];
        foreach ((DirectPtxPhysicalType physicalType, int expectedBytes) in cases)
            Assert.Equal(expectedBytes, DirectPtxTensorView.GetElementSizeInBytes(physicalType));
    }

    [Fact]
    public void CoverageManifests_HaveUniqueApiKeysAndValidatedLookups()
    {
        Assert.Equal(
            DirectPtxDenseLinearCoverageManifest.All.Count,
            DirectPtxDenseLinearCoverageManifest.All.Select(cell => cell.Api).Distinct().Count());
        Assert.Equal(
            DirectPtxQuantizedMixedSparseCoverageManifest.All.Count,
            DirectPtxQuantizedMixedSparseCoverageManifest.All.Select(cell => cell.Api).Distinct().Count());

        string denseApi = DirectPtxDenseLinearCoverageManifest.All[0].Api;
        string mixedApi = DirectPtxQuantizedMixedSparseCoverageManifest.All[0].Api;
        Assert.Equal(denseApi, DirectPtxDenseLinearCoverageManifest.Get(denseApi).Api);
        Assert.Equal(mixedApi, DirectPtxQuantizedMixedSparseCoverageManifest.Get(mixedApi).Api);
        Assert.Throws<ArgumentException>(() => DirectPtxDenseLinearCoverageManifest.Get(" "));
        Assert.Throws<ArgumentException>(() => DirectPtxQuantizedMixedSparseCoverageManifest.Get(" "));
    }
}
#endif
