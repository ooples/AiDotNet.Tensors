#if !NET462
namespace AiDotNet.Tensors.Engines.DirectGpu.OpenCL;

public sealed partial class OpenClBackend : IInstantNgpBackend, IFrexpBackend, IRenderingGeometryBackend, IUniqueConsecutiveBackend, INonzeroBackend, IModeBackend, IResidentIndexBackend, ICtcLossBackend, IImportanceSamplingBackend, INmsBackend, ISpiralIndicesBackend
{
    private DirectOpenClKernel GetInstantNgpKernel(string name)
    {
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"OpenCL Instant-NGP kernel not found: {name}.");
        return kernel;
    }

    private static IntPtr InstantNgpHandle(IGpuBuffer buffer) =>
        ((DirectOpenClGpuBuffer)buffer).Buffer.Handle;

    public void HashGridEncodeLevel(
        IGpuBuffer positions, IGpuBuffer hashTable, IGpuBuffer output,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(numPoints * featuresPerLevel);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("instant_ngp_hash_encode_level");
        kernel.SetArg(0, InstantNgpHandle(positions));
        kernel.SetArg(1, InstantNgpHandle(hashTable));
        kernel.SetArg(2, InstantNgpHandle(output));
        kernel.SetArg(3, numPoints); kernel.SetArg(4, resolution); kernel.SetArg(5, tableSize);
        kernel.SetArg(6, featuresPerLevel); kernel.SetArg(7, levelOffset); kernel.SetArg(8, outputStride);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void HashGridEncodeLevelBackward(
        IGpuBuffer positions, IGpuBuffer outputGradient, IGpuBuffer tableGradient,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(tableSize * featuresPerLevel);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("instant_ngp_hash_encode_level_backward");
        kernel.SetArg(0, InstantNgpHandle(positions));
        kernel.SetArg(1, InstantNgpHandle(outputGradient));
        kernel.SetArg(2, InstantNgpHandle(tableGradient));
        kernel.SetArg(3, numPoints); kernel.SetArg(4, resolution); kernel.SetArg(5, tableSize);
        kernel.SetArg(6, featuresPerLevel); kernel.SetArg(7, levelOffset); kernel.SetArg(8, outputStride);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void UniqueConsecutive(
        IGpuBuffer input, IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("unique_consecutive_compact");
        kernel.SetArg(0, InstantNgpHandle(input));
        kernel.SetArg(1, InstantNgpHandle(outputCapacity));
        kernel.SetArg(2, InstantNgpHandle(outputCount));
        kernel.SetArg(3, length);
        kernel.Execute1D(1, 1);
    }

    public void Frexp(
        IGpuBuffer input, IGpuBuffer mantissa, IGpuBuffer exponent, int length)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("frexp_decompose");
        kernel.SetArg(0, InstantNgpHandle(input));
        kernel.SetArg(1, InstantNgpHandle(mantissa));
        kernel.SetArg(2, InstantNgpHandle(exponent));
        kernel.SetArg(3, length);
        kernel.Execute1D(length, CalculateOptimalWorkGroupSize1D(length));
    }

    public void UniqueConsecutiveWithInfo(
        IGpuBuffer input, IGpuBuffer outputValues, IGpuBuffer outputInverse,
        IGpuBuffer outputCounts, IGpuBuffer outputCount, int length,
        bool returnInverse, bool returnCounts)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("unique_consecutive_with_info");
        kernel.SetArg(0, InstantNgpHandle(input));
        kernel.SetArg(1, InstantNgpHandle(outputValues));
        kernel.SetArg(2, InstantNgpHandle(outputInverse));
        kernel.SetArg(3, InstantNgpHandle(outputCounts));
        kernel.SetArg(4, InstantNgpHandle(outputCount));
        kernel.SetArg(5, length); kernel.SetArg(6, returnInverse ? 1 : 0);
        kernel.SetArg(7, returnCounts ? 1 : 0);
        kernel.Execute1D(1, 1);
    }

    public void ProjectGaussians3DTo2D(
        IGpuBuffer means3D, IGpuBuffer covariances3D, IGpuBuffer means2D,
        IGpuBuffer covariances2D, IGpuBuffer depths, IGpuBuffer visible,
        int numGaussians, int covarianceStride, int imageWidth, int imageHeight,
        float v00, float v01, float v02, float v03,
        float v10, float v11, float v12, float v13,
        float v20, float v21, float v22, float v23, float p00, float p11)
    {
        if (numGaussians <= 0) return;
        var kernel = GetInstantNgpKernel("project_gaussians_3d_to_2d");
        kernel.SetArg(0, InstantNgpHandle(means3D));
        kernel.SetArg(1, InstantNgpHandle(covariances3D));
        kernel.SetArg(2, InstantNgpHandle(means2D));
        kernel.SetArg(3, InstantNgpHandle(covariances2D));
        kernel.SetArg(4, InstantNgpHandle(depths)); kernel.SetArg(5, InstantNgpHandle(visible));
        kernel.SetArg(6, numGaussians); kernel.SetArg(7, covarianceStride);
        kernel.SetArg(8, imageWidth); kernel.SetArg(9, imageHeight);
        kernel.SetArg(10, v00); kernel.SetArg(11, v01); kernel.SetArg(12, v02); kernel.SetArg(13, v03);
        kernel.SetArg(14, v10); kernel.SetArg(15, v11); kernel.SetArg(16, v12); kernel.SetArg(17, v13);
        kernel.SetArg(18, v20); kernel.SetArg(19, v21); kernel.SetArg(20, v22); kernel.SetArg(21, v23);
        kernel.SetArg(22, p00); kernel.SetArg(23, p11);
        kernel.Execute1D(numGaussians, CalculateOptimalWorkGroupSize1D(numGaussians));
    }

    public void SampleRaysWithOccupancy(
        IGpuBuffer rayOrigins, IGpuBuffer rayDirections, IGpuBuffer occupancyBitfield,
        IGpuBuffer positions, IGpuBuffer directions, IGpuBuffer validMask, IGpuBuffer tValues,
        int numRays, int occupancyWordCount, int gridSize, int maxSamples,
        float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
        float nearBound, float farBound)
    {
        int total = checked(numRays * maxSamples);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("sample_rays_with_occupancy");
        kernel.SetArg(0, InstantNgpHandle(rayOrigins)); kernel.SetArg(1, InstantNgpHandle(rayDirections));
        kernel.SetArg(2, InstantNgpHandle(occupancyBitfield)); kernel.SetArg(3, InstantNgpHandle(positions));
        kernel.SetArg(4, InstantNgpHandle(directions)); kernel.SetArg(5, InstantNgpHandle(validMask));
        kernel.SetArg(6, InstantNgpHandle(tValues)); kernel.SetArg(7, numRays);
        kernel.SetArg(8, occupancyWordCount); kernel.SetArg(9, gridSize); kernel.SetArg(10, maxSamples);
        kernel.SetArg(11, minX); kernel.SetArg(12, minY); kernel.SetArg(13, minZ);
        kernel.SetArg(14, maxX); kernel.SetArg(15, maxY); kernel.SetArg(16, maxZ);
        kernel.SetArg(17, nearBound); kernel.SetArg(18, farBound);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void UniqueSortedWithInfo(
        IGpuBuffer sortedInput, IGpuBuffer sortedOriginalIndices, IGpuBuffer outputValues,
        IGpuBuffer outputInverse, IGpuBuffer outputCounts, IGpuBuffer outputCount,
        int length, bool returnInverse, bool returnCounts)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("unique_sorted_with_info");
        kernel.SetArg(0, InstantNgpHandle(sortedInput));
        kernel.SetArg(1, InstantNgpHandle(sortedOriginalIndices));
        kernel.SetArg(2, InstantNgpHandle(outputValues));
        kernel.SetArg(3, InstantNgpHandle(outputInverse));
        kernel.SetArg(4, InstantNgpHandle(outputCounts));
        kernel.SetArg(5, InstantNgpHandle(outputCount));
        kernel.SetArg(6, length); kernel.SetArg(7, returnInverse ? 1 : 0);
        kernel.SetArg(8, returnCounts ? 1 : 0);
        kernel.Execute1D(1, 1);
    }

    public void Nonzero(IGpuBuffer input, IGpuBuffer strides, IGpuBuffer outputCapacity,
        IGpuBuffer outputCount, int length, int rank)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("nonzero_compact");
        kernel.SetArg(0, InstantNgpHandle(input));
        kernel.SetArg(1, InstantNgpHandle(strides));
        kernel.SetArg(2, InstantNgpHandle(outputCapacity));
        kernel.SetArg(3, InstantNgpHandle(outputCount));
        kernel.SetArg(4, length); kernel.SetArg(5, rank);
        kernel.Execute1D(1, 1);
    }

    public void Mode(IGpuBuffer input, IGpuBuffer output, int length)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("resident_mode");
        kernel.SetArg(0, InstantNgpHandle(input));
        kernel.SetArg(1, InstantNgpHandle(output));
        kernel.SetArg(2, length);
        kernel.Execute1D(1, 1);
    }

    public void ConvertIndicesToInt32(
        IGpuBuffer numericIndices, IGpuBuffer int32Indices, int length)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("resident_indices_to_int32");
        kernel.SetArg(0, InstantNgpHandle(numericIndices));
        kernel.SetArg(1, InstantNgpHandle(int32Indices));
        kernel.SetArg(2, length);
        kernel.Execute1D(length, CalculateOptimalWorkGroupSize1D(length));
    }

    public void IndexAdd(
        IGpuBuffer destination, IGpuBuffer indices, IGpuBuffer source, IGpuBuffer output,
        int outerSize, int sourceAxis, int destinationAxis, int innerSize)
    {
        int total = checked(outerSize * destinationAxis * innerSize);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("resident_index_add");
        kernel.SetArg(0, InstantNgpHandle(destination));
        kernel.SetArg(1, InstantNgpHandle(indices));
        kernel.SetArg(2, InstantNgpHandle(source));
        kernel.SetArg(3, InstantNgpHandle(output));
        kernel.SetArg(4, outerSize); kernel.SetArg(5, sourceAxis);
        kernel.SetArg(6, destinationAxis); kernel.SetArg(7, innerSize);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void IndexSelect(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int outerSize, int sourceAxis, int indexAxis, int innerSize)
    {
        int total = checked(outerSize * indexAxis * innerSize);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("resident_index_select");
        kernel.SetArg(0, InstantNgpHandle(source));
        kernel.SetArg(1, InstantNgpHandle(indices));
        kernel.SetArg(2, InstantNgpHandle(output));
        kernel.SetArg(3, outerSize); kernel.SetArg(4, sourceAxis);
        kernel.SetArg(5, indexAxis); kernel.SetArg(6, innerSize);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void ScatterMaxWithArgmaxRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer argmax,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(outputRows * innerSize);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("resident_scatter_max_argmax_rows");
        kernel.SetArg(0, InstantNgpHandle(source));
        kernel.SetArg(1, InstantNgpHandle(indices));
        kernel.SetArg(2, InstantNgpHandle(output));
        kernel.SetArg(3, InstantNgpHandle(argmax));
        kernel.SetArg(4, sourceRows); kernel.SetArg(5, innerSize); kernel.SetArg(6, outputRows);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void UniformMeshLaplacian(
        IGpuBuffer faces, IGpuBuffer output, int numFaces, int numVertices)
    {
        int total = checked(numVertices * numVertices);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("resident_uniform_mesh_laplacian");
        kernel.SetArg(0, InstantNgpHandle(faces));
        kernel.SetArg(1, InstantNgpHandle(output));
        kernel.SetArg(2, numFaces); kernel.SetArg(3, numVertices);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void ScatterMeanRowsWithCounts(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = Math.Max(checked(outputRows * innerSize), outputRows);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("resident_scatter_mean_rows_counts");
        kernel.SetArg(0, InstantNgpHandle(source));
        kernel.SetArg(1, InstantNgpHandle(indices));
        kernel.SetArg(2, InstantNgpHandle(output));
        kernel.SetArg(3, InstantNgpHandle(counts));
        kernel.SetArg(4, sourceRows); kernel.SetArg(5, innerSize); kernel.SetArg(6, outputRows);
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void CtcLoss(
        IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer inputLengths,
        IGpuBuffer targetLengths, IGpuBuffer workspace, IGpuBuffer losses,
        int maxTime, int batchSize, int numClasses, int maxTargetLength, int blank)
    {
        if (batchSize <= 0) return;
        var kernel = GetInstantNgpKernel("ctc_loss_forward");
        kernel.SetArg(0, InstantNgpHandle(logProbs));
        kernel.SetArg(1, InstantNgpHandle(targets));
        kernel.SetArg(2, InstantNgpHandle(inputLengths));
        kernel.SetArg(3, InstantNgpHandle(targetLengths));
        kernel.SetArg(4, InstantNgpHandle(workspace));
        kernel.SetArg(5, InstantNgpHandle(losses));
        kernel.SetArg(6, maxTime); kernel.SetArg(7, batchSize); kernel.SetArg(8, numClasses);
        kernel.SetArg(9, maxTargetLength); kernel.SetArg(10, blank);
        kernel.Execute1D(batchSize, CalculateOptimalWorkGroupSize1D(batchSize));
    }

    public void ImportanceSampling(
        IGpuBuffer tValuesCoarse, IGpuBuffer weightsCoarse, IGpuBuffer fineTValues,
        int numRays, int numCoarseSamples, int numFineSamples, uint seed)
    {
        int total = checked(numRays * numFineSamples);
        if (total <= 0) return;
        var kernel = GetInstantNgpKernel("importance_sampling");
        kernel.SetArg(0, InstantNgpHandle(tValuesCoarse));
        kernel.SetArg(1, InstantNgpHandle(weightsCoarse));
        kernel.SetArg(2, InstantNgpHandle(fineTValues));
        kernel.SetArg(3, numRays); kernel.SetArg(4, numCoarseSamples);
        kernel.SetArg(5, numFineSamples); kernel.SetArg(6, unchecked((int)seed));
        kernel.Execute1D(total, CalculateOptimalWorkGroupSize1D(total));
    }

    public void Nms(
        IGpuBuffer boxes, IGpuBuffer scores, IGpuBuffer classIds, IGpuBuffer suppressed,
        IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length, float iouThreshold,
        int batched)
    {
        if (length <= 0) return;
        var kernel = GetInstantNgpKernel("resident_nms");
        kernel.SetArg(0, InstantNgpHandle(boxes)); kernel.SetArg(1, InstantNgpHandle(scores));
        kernel.SetArg(2, InstantNgpHandle(classIds)); kernel.SetArg(3, InstantNgpHandle(suppressed));
        kernel.SetArg(4, InstantNgpHandle(outputCapacity)); kernel.SetArg(5, InstantNgpHandle(outputCount));
        kernel.SetArg(6, length); kernel.SetArg(7, iouThreshold); kernel.SetArg(8, batched);
        kernel.Execute1D(1, 1);
    }

    public void GenerateSpiralIndices(
        IGpuBuffer vertices, IGpuBuffer faces, IGpuBuffer visited, IGpuBuffer currentRing,
        IGpuBuffer nextRing, IGpuBuffer output, int numVertices, int numFaces, int spiralLength)
    {
        if (numVertices <= 0 || spiralLength <= 0) return;
        var kernel = GetInstantNgpKernel("generate_spiral_indices");
        kernel.SetArg(0, InstantNgpHandle(vertices)); kernel.SetArg(1, InstantNgpHandle(faces));
        kernel.SetArg(2, InstantNgpHandle(visited)); kernel.SetArg(3, InstantNgpHandle(currentRing));
        kernel.SetArg(4, InstantNgpHandle(nextRing)); kernel.SetArg(5, InstantNgpHandle(output));
        kernel.SetArg(6, numVertices); kernel.SetArg(7, numFaces); kernel.SetArg(8, spiralLength);
        kernel.Execute1D(1, 1);
    }
}
#endif
