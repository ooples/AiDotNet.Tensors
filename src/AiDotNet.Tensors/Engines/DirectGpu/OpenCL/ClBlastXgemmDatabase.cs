// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

internal static class ClBlastXgemmDatabase
{
    public static bool TryGetConfig(ClBlastDeviceInfo device, out GemmConfig config)
    {
        config = default;
        if (!ClBlastDatabaseSearch.TryGetParameters(ClBlastXgemmDatabaseData.Vendors, device, out var parameters))
            return false;

        if (parameters.Length < 16)
            return false;

        int gemmK = parameters[0];
        GemmKernelTemplate kernelTemplate;
        string kernelName;
        switch (gemmK)
        {
            case 0:
                kernelTemplate = GemmKernelTemplate.ClBlastBaselineK0;
                kernelName = "clblast_baseline_k0";
                break;
            case 1:
                kernelTemplate = GemmKernelTemplate.ClBlastBaselineK1;
                kernelName = "clblast_baseline_k1";
                break;
            default:
                return false;
        }

        config = new GemmConfig
        {
            KernelTemplate = kernelTemplate,
            KernelName = kernelName,
            TileM = parameters[6],
            TileN = parameters[9],
            TileK = parameters[2],
            ThreadTileM = parameters[5],
            ThreadTileN = parameters[8],
            MdimaSize = parameters[4],
            NdimbSize = parameters[7],
            KReg = parameters[1],
            KUnroll = parameters[3],
            VectorWidthM = parameters[14],
            VectorWidthN = parameters[15],
            CacheA = parameters[10] == 1,
            CacheB = parameters[11] == 1,
            StrideM = parameters[12] == 1,
            StrideN = parameters[13] == 1,
            UseDoubleBuffering = false,
            UseVectorizedLoads = false,
            UseSubgroupOps = false,
            UseTrueVectorLDS = false,
            UseColumnMajorA = false
        };

        return true;
    }
}
