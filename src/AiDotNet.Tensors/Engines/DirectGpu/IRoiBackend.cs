// Copyright (c) AiDotNet. All rights reserved.
// Optional backend capability for the RoI family added by Issue #217.
namespace AiDotNet.Tensors.Engines.DirectGpu
{
    public interface IRoiBackend
    {
        void RoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW,
            float spatialScale, int samplingRatio, bool aligned);

        void RoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW,
            float spatialScale);

        /// <summary>Position-sensitive RoIAlign (R-FCN).</summary>
        void PsRoIAlign(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
            float spatialScale, int samplingRatio);

        /// <summary>Position-sensitive RoI-pool (R-FCN).</summary>
        void PsRoIPool(IGpuBuffer input, IGpuBuffer boxes, IGpuBuffer output,
            int N, int C, int H, int W, int K, int outH, int outW, int outputChannels,
            float spatialScale);
    }
}
