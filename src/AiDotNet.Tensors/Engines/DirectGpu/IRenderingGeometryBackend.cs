// Copyright (c) AiDotNet. All rights reserved.
namespace AiDotNet.Tensors.Engines.DirectGpu;

/// <summary>Optional native kernels for fixed-shape rendering and ray-sampling geometry.</summary>
public interface IRenderingGeometryBackend
{
    /// <summary>Projects camera-space Gaussian covariance and world-space means to image space.</summary>
    void ProjectGaussians3DTo2D(
        IGpuBuffer means3D,
        IGpuBuffer covariances3D,
        IGpuBuffer means2D,
        IGpuBuffer covariances2D,
        IGpuBuffer depths,
        IGpuBuffer visible,
        int numGaussians,
        int covarianceStride,
        int imageWidth,
        int imageHeight,
        float v00, float v01, float v02, float v03,
        float v10, float v11, float v12, float v13,
        float v20, float v21, float v22, float v23,
        float p00, float p11);

    /// <summary>Samples fixed ray slots and marks those whose packed occupancy bit is set.</summary>
    void SampleRaysWithOccupancy(
        IGpuBuffer rayOrigins,
        IGpuBuffer rayDirections,
        IGpuBuffer occupancyBitfield,
        IGpuBuffer positions,
        IGpuBuffer directions,
        IGpuBuffer validMask,
        IGpuBuffer tValues,
        int numRays,
        int occupancyWordCount,
        int gridSize,
        int maxSamples,
        float minX, float minY, float minZ,
        float maxX, float maxY, float maxZ,
        float nearBound, float farBound);
}
