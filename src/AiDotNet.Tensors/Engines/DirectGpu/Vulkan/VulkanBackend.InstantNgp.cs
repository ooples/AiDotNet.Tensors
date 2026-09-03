namespace AiDotNet.Tensors.Engines.DirectGpu.Vulkan;

public sealed unsafe partial class VulkanBackend : IInstantNgpBackend, IFrexpBackend, IRenderingGeometryBackend, IUniqueConsecutiveBackend, INonzeroBackend, IModeBackend, IResidentIndexBackend, ICtcLossBackend, IImportanceSamplingBackend, INmsBackend, ISpiralIndicesBackend
{
    public void HashGridEncodeLevel(
        IGpuBuffer positions, IGpuBuffer hashTable, IGpuBuffer output,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(numPoints * featuresPerLevel);
        GlslDispatchN(VulkanInstantNgpKernels.Forward, total,
            [positions, hashTable, output],
            [(uint)numPoints, (uint)resolution, (uint)tableSize, (uint)featuresPerLevel,
                (uint)levelOffset, (uint)outputStride]);
    }

    public void HashGridEncodeLevelBackward(
        IGpuBuffer positions, IGpuBuffer outputGradient, IGpuBuffer tableGradient,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(tableSize * featuresPerLevel);
        GlslDispatchN(VulkanInstantNgpKernels.Backward, total,
            [positions, outputGradient, tableGradient],
            [(uint)numPoints, (uint)resolution, (uint)tableSize, (uint)featuresPerLevel,
                (uint)levelOffset, (uint)outputStride]);
    }

    public void UniqueConsecutive(
        IGpuBuffer input, IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.UniqueConsecutive, 1,
            [input, outputCapacity, outputCount], [(uint)length]);
    }

    public void Frexp(
        IGpuBuffer input, IGpuBuffer mantissa, IGpuBuffer exponent, int length)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.Frexp, length,
            [input, mantissa, exponent], [(uint)length]);
    }

    public void UniqueConsecutiveWithInfo(
        IGpuBuffer input, IGpuBuffer outputValues, IGpuBuffer outputInverse,
        IGpuBuffer outputCounts, IGpuBuffer outputCount, int length,
        bool returnInverse, bool returnCounts)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.UniqueConsecutiveWithInfo, 1,
            [input, outputValues, outputInverse, outputCounts, outputCount],
            [(uint)length, returnInverse ? 1u : 0u, returnCounts ? 1u : 0u]);
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
        GlslDispatchN(VulkanInstantNgpKernels.ProjectGaussians3DTo2D, numGaussians,
            [means3D, covariances3D, means2D, covariances2D, depths, visible],
            [
                (uint)numGaussians, (uint)covarianceStride, (uint)imageWidth, (uint)imageHeight,
                FloatBits(v00), FloatBits(v01), FloatBits(v02), FloatBits(v03),
                FloatBits(v10), FloatBits(v11), FloatBits(v12), FloatBits(v13),
                FloatBits(v20), FloatBits(v21), FloatBits(v22), FloatBits(v23),
                FloatBits(p00), FloatBits(p11),
            ]);
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
        GlslDispatchN(VulkanInstantNgpKernels.SampleRaysWithOccupancy, total,
            [rayOrigins, rayDirections, occupancyBitfield, positions, directions, validMask, tValues],
            [
                (uint)numRays, (uint)occupancyWordCount, (uint)gridSize, (uint)maxSamples,
                FloatBits(minX), FloatBits(minY), FloatBits(minZ),
                FloatBits(maxX), FloatBits(maxY), FloatBits(maxZ),
                FloatBits(nearBound), FloatBits(farBound),
            ]);
    }

    public void UniqueSortedWithInfo(
        IGpuBuffer sortedInput, IGpuBuffer sortedOriginalIndices, IGpuBuffer outputValues,
        IGpuBuffer outputInverse, IGpuBuffer outputCounts, IGpuBuffer outputCount,
        int length, bool returnInverse, bool returnCounts)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.UniqueSortedWithInfo, 1,
            [sortedInput, sortedOriginalIndices, outputValues, outputInverse, outputCounts, outputCount],
            [(uint)length, returnInverse ? 1u : 0u, returnCounts ? 1u : 0u]);
    }

    public void Nonzero(IGpuBuffer input, IGpuBuffer strides, IGpuBuffer outputCapacity,
        IGpuBuffer outputCount, int length, int rank)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.Nonzero, 1,
            [input, strides, outputCapacity, outputCount], [(uint)length, (uint)rank]);
    }

    public void Mode(IGpuBuffer input, IGpuBuffer output, int length)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.Mode, 1,
            [input, output], [(uint)length]);
    }

    public void ConvertIndicesToInt32(
        IGpuBuffer numericIndices, IGpuBuffer int32Indices, int length)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ConvertIndicesToInt32, length,
            [numericIndices, int32Indices], [(uint)length]);
    }

    public void IndexAdd(
        IGpuBuffer destination, IGpuBuffer indices, IGpuBuffer source, IGpuBuffer output,
        int outerSize, int sourceAxis, int destinationAxis, int innerSize)
    {
        int total = checked(outerSize * destinationAxis * innerSize);
        if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.IndexAdd, total,
            [destination, indices, source, output],
            [(uint)outerSize, (uint)sourceAxis, (uint)destinationAxis, (uint)innerSize]);
    }

    public void IndexSelect(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int outerSize, int sourceAxis, int indexAxis, int innerSize)
    {
        int total = checked(outerSize * indexAxis * innerSize);
        if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.IndexSelect, total,
            [source, indices, output],
            [(uint)outerSize, (uint)sourceAxis, (uint)indexAxis, (uint)innerSize]);
    }

    public void ScatterMaxWithArgmaxRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer argmax,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(outputRows * innerSize);
        if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterMaxWithArgmaxRows, total,
            [source, indices, output, argmax],
            [(uint)sourceRows, (uint)innerSize, (uint)outputRows]);
    }

    public void UniformMeshLaplacian(
        IGpuBuffer faces, IGpuBuffer output, int numFaces, int numVertices)
    {
        int total = checked(numVertices * numVertices);
        if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.UniformMeshLaplacian, total,
            [faces, output], [(uint)numFaces, (uint)numVertices]);
    }

    public void ScatterAddRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(outputRows * innerSize); if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterAddRows, total,
            [source, indices, output], [(uint)sourceRows, (uint)innerSize, (uint)outputRows]);
    }

    public void ScatterMeanRowsWithCounts(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = Math.Max(checked(outputRows * innerSize), outputRows); if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterMeanRowsWithCounts, total,
            [source, indices, output, counts], [(uint)sourceRows, (uint)innerSize, (uint)outputRows]);
    }

    public void ScatterSoftmaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int numGroups)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterSoftmaxRows, total,
            [source, indices, output], [(uint)sourceRows, (uint)innerSize, (uint)numGroups]);
    }

    public void ScatterAddBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterAddBackwardRows, total,
            [gradOutput, indices, gradSource], [(uint)sourceRows, (uint)innerSize, (uint)outputRows]);
    }

    public void ScatterMeanBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer counts, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterMeanBackwardRows, total,
            [gradOutput, indices, counts, gradSource], [(uint)sourceRows, (uint)innerSize, (uint)outputRows]);
    }

    public void ScatterMaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer argmax, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterMaxBackwardRows, total,
            [gradOutput, argmax, gradSource], [(uint)sourceRows, (uint)innerSize, (uint)outputRows]);
    }

    public void ScatterSoftmaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int numGroups)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ScatterSoftmaxBackwardRows, total,
            [gradOutput, output, indices, gradSource], [(uint)sourceRows, (uint)innerSize, (uint)numGroups]);
    }

    public void CtcLoss(
        IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer inputLengths,
        IGpuBuffer targetLengths, IGpuBuffer workspace, IGpuBuffer losses,
        int maxTime, int batchSize, int numClasses, int maxTargetLength, int blank)
    {
        if (batchSize <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.CtcLoss, batchSize,
            [logProbs, targets, inputLengths, targetLengths, workspace, losses],
            [(uint)maxTime, (uint)batchSize, (uint)numClasses,
                (uint)maxTargetLength, (uint)blank]);
    }

    public void ImportanceSampling(
        IGpuBuffer tValuesCoarse, IGpuBuffer weightsCoarse, IGpuBuffer fineTValues,
        int numRays, int numCoarseSamples, int numFineSamples, uint seed)
    {
        int total = checked(numRays * numFineSamples);
        if (total <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.ImportanceSampling, total,
            [tValuesCoarse, weightsCoarse, fineTValues],
            [(uint)numRays, (uint)numCoarseSamples, (uint)numFineSamples, seed]);
    }

    public void Nms(
        IGpuBuffer boxes, IGpuBuffer scores, IGpuBuffer classIds, IGpuBuffer suppressed,
        IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length, float iouThreshold,
        int batched)
    {
        if (length <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.Nms, 1,
            [boxes, scores, classIds, suppressed, outputCapacity, outputCount],
            [(uint)length, unchecked((uint)SingleToInt32BitsCompat(iouThreshold)), (uint)batched]);
    }

    public void GenerateSpiralIndices(
        IGpuBuffer vertices, IGpuBuffer faces, IGpuBuffer visited, IGpuBuffer currentRing,
        IGpuBuffer nextRing, IGpuBuffer output, int numVertices, int numFaces, int spiralLength)
    {
        if (numVertices <= 0 || spiralLength <= 0) return;
        GlslDispatchN(VulkanInstantNgpKernels.SpiralIndices, 1,
            [vertices, faces, visited, currentRing, nextRing, output],
            [(uint)numVertices, (uint)numFaces, (uint)spiralLength]);
    }
}
