using AiDotNet.Tensors.Engines.DirectGpu.CUDA;
using Xunit;

/// <summary>
/// CPU-executable smoke checks for the NCCL / cuBLASLt binding surface.
/// Full multi-GPU tests require CUDA hardware and run in a hardware-
/// gated lane; these ensure the static-constructor / P/Invoke shim
/// path at least loads without throwing when the native lib is
/// absent (availability probe).
/// </summary>
public class NcclAvailabilityTests
{
    [Fact]
    public void NcclComm_IsAvailable_DoesNotThrow()
    {
        // libnccl.so likely missing on CI — the probe must return false
        // rather than crash.
        bool available = NcclComm.IsAvailable;
        // Value doesn't matter; exception freedom does.
        Assert.True(available || !available);
    }

    [Fact]
    public void CuBlasLtMatmul_IsAvailable_DoesNotThrow()
    {
        bool available = CuBlasLtMatmul.IsAvailable;
        Assert.True(available || !available);
    }

    [Fact]
    public void Nccl_ResultEnum_ContainsSuccessAndAllCanonicalCodes()
    {
        // Guards against accidentally renumbering the enum out of sync
        // with nccl.h — silent binding breakage.
        Assert.Equal(0, (int)NcclResult.Success);
        Assert.Equal(4, (int)NcclResult.InvalidArgument);
        Assert.Equal(5, (int)NcclResult.InvalidUsage);
    }

    [Fact]
    public void Nccl_DataType_CoversCommonElementTypes()
    {
        // Float32 = 7, Float16 = 6, BFloat16 = 9 per nccl.h.
        Assert.Equal(7, (int)NcclDataType.Float32);
        Assert.Equal(6, (int)NcclDataType.Float16);
        Assert.Equal(9, (int)NcclDataType.BFloat16);
    }

    [Fact]
    public void Nccl_ReductionOps_CoversSumProdMaxMinAvg()
    {
        Assert.Equal(0, (int)NcclRedOp.Sum);
        Assert.Equal(1, (int)NcclRedOp.Prod);
        Assert.Equal(2, (int)NcclRedOp.Max);
        Assert.Equal(3, (int)NcclRedOp.Min);
        Assert.Equal(4, (int)NcclRedOp.Avg);
    }

    [Fact]
    public void CuBlasLtEpilogue_CoversFusedOps()
    {
        Assert.Equal(1, (int)CublasLtEpilogue.Default);
        Assert.Equal(2, (int)CublasLtEpilogue.ReLU);
        Assert.Equal(4, (int)CublasLtEpilogue.Bias);
        Assert.Equal(32, (int)CublasLtEpilogue.GELU);
        Assert.Equal(36, (int)CublasLtEpilogue.GELUBias);
    }
}
