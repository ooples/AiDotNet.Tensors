#if NET7_0_OR_GREATER
namespace AiDotNet.Tensors.Engines.DirectGpu.WebGpu;

public sealed partial class WebGpuBackend : IInstantNgpBackend, IUniqueConsecutiveBackend, INonzeroBackend, IModeBackend, IResidentIndexBackend, ICtcLossBackend, IImportanceSamplingBackend, INmsBackend, ISpiralIndicesBackend
{
    private static float[] InstantNgpUniforms(
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride) =>
        [
            BitConverter.Int32BitsToSingle(numPoints),
            BitConverter.Int32BitsToSingle(resolution),
            BitConverter.Int32BitsToSingle(tableSize),
            BitConverter.Int32BitsToSingle(featuresPerLevel),
            BitConverter.Int32BitsToSingle(levelOffset),
            BitConverter.Int32BitsToSingle(outputStride),
            0f,
            0f,
        ];

    public void HashGridEncodeLevel(
        IGpuBuffer positions, IGpuBuffer hashTable, IGpuBuffer output,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(numPoints * featuresPerLevel);
        if (total <= 0) return;
        Dispatch3BufferAsync("InstantNgp:Forward", WebGpuInstantNgpKernels.Forward, "main",
            positions, hashTable, output,
            InstantNgpUniforms(numPoints, resolution, tableSize, featuresPerLevel,
                levelOffset, outputStride), total).GetAwaiter().GetResult();
    }

    public void HashGridEncodeLevelBackward(
        IGpuBuffer positions, IGpuBuffer outputGradient, IGpuBuffer tableGradient,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(tableSize * featuresPerLevel);
        if (total <= 0) return;
        Dispatch3BufferAsync("InstantNgp:Backward", WebGpuInstantNgpKernels.Backward, "main",
            positions, outputGradient, tableGradient,
            InstantNgpUniforms(numPoints, resolution, tableSize, featuresPerLevel,
                levelOffset, outputStride), total).GetAwaiter().GetResult();
    }

    public void UniqueConsecutive(
        IGpuBuffer input, IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length)
    {
        if (length <= 0) return;
        Dispatch3BufferAsync("UniqueConsecutive", WebGpuInstantNgpKernels.UniqueConsecutive,
            "main", input, outputCapacity, outputCount,
            [BitConverter.Int32BitsToSingle(length), 0f, 0f, 0f], 1)
            .GetAwaiter().GetResult();
    }

    public void Nonzero(IGpuBuffer input, IGpuBuffer strides, IGpuBuffer outputCapacity,
        IGpuBuffer outputCount, int length, int rank)
    {
        if (length <= 0) return;
        Dispatch4BufferAsync("Nonzero", WebGpuInstantNgpKernels.Nonzero, "main",
            input, strides, outputCapacity, outputCount,
            [BitConverter.Int32BitsToSingle(length), BitConverter.Int32BitsToSingle(rank), 0f, 0f], 1)
            .GetAwaiter().GetResult();
    }

    public void Mode(IGpuBuffer input, IGpuBuffer output, int length)
    {
        if (length <= 0) return;
        Dispatch2BufferAsync("ResidentMode", WebGpuInstantNgpKernels.Mode, "main",
            input, output,
            [BitConverter.Int32BitsToSingle(length), 0f, 0f, 0f], 1)
            .GetAwaiter().GetResult();
    }

    public void ConvertIndicesToInt32(
        IGpuBuffer numericIndices, IGpuBuffer int32Indices, int length)
    {
        if (length <= 0) return;
        Dispatch2BufferAsync("ResidentIndicesToInt32",
            WebGpuInstantNgpKernels.ConvertIndicesToInt32, "main",
            numericIndices, int32Indices,
            [BitConverter.Int32BitsToSingle(length), 0f, 0f, 0f], length)
            .GetAwaiter().GetResult();
    }

    public void IndexAdd(
        IGpuBuffer destination, IGpuBuffer indices, IGpuBuffer source, IGpuBuffer output,
        int outerSize, int sourceAxis, int destinationAxis, int innerSize)
    {
        int total = checked(outerSize * destinationAxis * innerSize);
        if (total <= 0) return;
        Dispatch4BufferAsync("ResidentIndexAdd", WebGpuInstantNgpKernels.IndexAdd, "main",
            destination, indices, source, output,
            [
                BitConverter.Int32BitsToSingle(outerSize),
                BitConverter.Int32BitsToSingle(sourceAxis),
                BitConverter.Int32BitsToSingle(destinationAxis),
                BitConverter.Int32BitsToSingle(innerSize),
            ], total).GetAwaiter().GetResult();
    }

    public void IndexSelect(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int outerSize, int sourceAxis, int indexAxis, int innerSize)
    {
        int total = checked(outerSize * indexAxis * innerSize);
        if (total <= 0) return;
        Dispatch3BufferAsync("ResidentIndexSelect", WebGpuInstantNgpKernels.IndexSelect, "main",
            source, indices, output,
            [
                BitConverter.Int32BitsToSingle(outerSize),
                BitConverter.Int32BitsToSingle(sourceAxis),
                BitConverter.Int32BitsToSingle(indexAxis),
                BitConverter.Int32BitsToSingle(innerSize),
            ], total).GetAwaiter().GetResult();
    }

    public void ScatterMaxWithArgmaxRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer argmax,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(outputRows * innerSize);
        if (total <= 0) return;
        Dispatch4BufferAsync("ResidentScatterMaxArgmaxRows",
            WebGpuInstantNgpKernels.ScatterMaxWithArgmaxRows, "main",
            source, indices, output, argmax,
            [
                BitConverter.Int32BitsToSingle(sourceRows),
                BitConverter.Int32BitsToSingle(innerSize),
                BitConverter.Int32BitsToSingle(outputRows),
                0f,
            ], total).GetAwaiter().GetResult();
    }

    public void UniformMeshLaplacian(
        IGpuBuffer faces, IGpuBuffer output, int numFaces, int numVertices)
    {
        int total = checked(numVertices * numVertices);
        if (total <= 0) return;
        Dispatch2BufferAsync("ResidentUniformMeshLaplacian",
            WebGpuInstantNgpKernels.UniformMeshLaplacian, "main", faces, output,
            [
                BitConverter.Int32BitsToSingle(numFaces),
                BitConverter.Int32BitsToSingle(numVertices),
                0f,
                0f,
            ], total).GetAwaiter().GetResult();
    }

    private static float[] ResidentRowsUniforms(int firstSize, int innerSize, int thirdSize) =>
        [
            BitConverter.Int32BitsToSingle(firstSize),
            BitConverter.Int32BitsToSingle(innerSize),
            BitConverter.Int32BitsToSingle(thirdSize),
            0f,
        ];

    public void ScatterAddRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(outputRows * innerSize); if (total <= 0) return;
        Dispatch3BufferAsync("ResidentScatterAddRows", WebGpuInstantNgpKernels.ScatterAddRows,
            "main", source, indices, output,
            ResidentRowsUniforms(sourceRows, innerSize, outputRows), total).GetAwaiter().GetResult();
    }

    public void ScatterMeanRowsWithCounts(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = Math.Max(checked(outputRows * innerSize), outputRows); if (total <= 0) return;
        Dispatch4BufferAsync("ResidentScatterMeanRows", WebGpuInstantNgpKernels.ScatterMeanRowsWithCounts,
            "main", source, indices, output, counts,
            ResidentRowsUniforms(sourceRows, innerSize, outputRows), total).GetAwaiter().GetResult();
    }

    public void ScatterSoftmaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int numGroups)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        Dispatch3BufferAsync("ResidentScatterSoftmaxRows", WebGpuInstantNgpKernels.ScatterSoftmaxRows,
            "main", source, indices, output,
            ResidentRowsUniforms(sourceRows, innerSize, numGroups), total).GetAwaiter().GetResult();
    }

    public void ScatterAddBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        Dispatch3BufferAsync("ResidentScatterAddBackwardRows",
            WebGpuInstantNgpKernels.ScatterAddBackwardRows, "main",
            gradOutput, indices, gradSource,
            ResidentRowsUniforms(sourceRows, innerSize, outputRows), total).GetAwaiter().GetResult();
    }

    public void ScatterMeanBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer counts, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        Dispatch4BufferAsync("ResidentScatterMeanBackwardRows",
            WebGpuInstantNgpKernels.ScatterMeanBackwardRows, "main",
            gradOutput, indices, counts, gradSource,
            ResidentRowsUniforms(sourceRows, innerSize, outputRows), total).GetAwaiter().GetResult();
    }

    public void ScatterMaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer argmax, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        Dispatch3BufferAsync("ResidentScatterMaxBackwardRows",
            WebGpuInstantNgpKernels.ScatterMaxBackwardRows, "main",
            gradOutput, argmax, gradSource,
            ResidentRowsUniforms(sourceRows, innerSize, outputRows), total).GetAwaiter().GetResult();
    }

    public void ScatterSoftmaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int numGroups)
    {
        int total = checked(sourceRows * innerSize); if (total <= 0) return;
        Dispatch4BufferAsync("ResidentScatterSoftmaxBackwardRows",
            WebGpuInstantNgpKernels.ScatterSoftmaxBackwardRows, "main",
            gradOutput, output, indices, gradSource,
            ResidentRowsUniforms(sourceRows, innerSize, numGroups), total).GetAwaiter().GetResult();
    }

    public void CtcLoss(
        IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer inputLengths,
        IGpuBuffer targetLengths, IGpuBuffer workspace, IGpuBuffer losses,
        int maxTime, int batchSize, int numClasses, int maxTargetLength, int blank)
    {
        if (batchSize <= 0) return;
        Dispatch6BufferAsync("CtcLoss", WebGpuInstantNgpKernels.CtcLoss, "main",
            logProbs, targets, inputLengths, targetLengths, workspace, losses,
            [
                BitConverter.Int32BitsToSingle(maxTime),
                BitConverter.Int32BitsToSingle(batchSize),
                BitConverter.Int32BitsToSingle(numClasses),
                BitConverter.Int32BitsToSingle(maxTargetLength),
                BitConverter.Int32BitsToSingle(blank),
                0f, 0f, 0f,
            ], batchSize).GetAwaiter().GetResult();
    }

    public void ImportanceSampling(
        IGpuBuffer tValuesCoarse, IGpuBuffer weightsCoarse, IGpuBuffer fineTValues,
        int numRays, int numCoarseSamples, int numFineSamples, uint seed)
    {
        int total = checked(numRays * numFineSamples);
        if (total <= 0) return;
        Dispatch3BufferAsync("ImportanceSampling", WebGpuInstantNgpKernels.ImportanceSampling,
            "main", tValuesCoarse, weightsCoarse, fineTValues,
            [
                BitConverter.Int32BitsToSingle(numRays),
                BitConverter.Int32BitsToSingle(numCoarseSamples),
                BitConverter.Int32BitsToSingle(numFineSamples),
                BitConverter.UInt32BitsToSingle(seed),
            ], total).GetAwaiter().GetResult();
    }

    public void Nms(
        IGpuBuffer boxes, IGpuBuffer scores, IGpuBuffer classIds, IGpuBuffer suppressed,
        IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length, float iouThreshold,
        int batched)
    {
        if (length <= 0) return;
        Dispatch6BufferAsync("ResidentNms", WebGpuInstantNgpKernels.Nms, "main",
            boxes, scores, classIds, suppressed, outputCapacity, outputCount,
            [
                BitConverter.Int32BitsToSingle(length),
                iouThreshold,
                BitConverter.Int32BitsToSingle(batched),
                0f,
            ], 1).GetAwaiter().GetResult();
    }

    public void GenerateSpiralIndices(
        IGpuBuffer vertices, IGpuBuffer faces, IGpuBuffer visited, IGpuBuffer currentRing,
        IGpuBuffer nextRing, IGpuBuffer output, int numVertices, int numFaces, int spiralLength)
    {
        if (numVertices <= 0 || spiralLength <= 0) return;
        Dispatch6BufferAsync("SpiralIndices", WebGpuInstantNgpKernels.SpiralIndices, "main",
            vertices, faces, visited, currentRing, nextRing, output,
            [
                BitConverter.Int32BitsToSingle(numVertices),
                BitConverter.Int32BitsToSingle(numFaces),
                BitConverter.Int32BitsToSingle(spiralLength),
                0f,
            ], 1).GetAwaiter().GetResult();
    }
}
#endif
