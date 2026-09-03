namespace AiDotNet.Tensors.Engines.DirectGpu.Metal;

public sealed partial class MetalBackend : IInstantNgpBackend, IFrexpBackend, IRenderingGeometryBackend, IUniqueConsecutiveBackend, INonzeroBackend, IModeBackend, IResidentIndexBackend, ICtcLossBackend, IImportanceSamplingBackend, INmsBackend, ISpiralIndicesBackend
{
    private void DispatchInstantNgp(
        string kernelName,
        IGpuBuffer first,
        IGpuBuffer second,
        IGpuBuffer output,
        int dispatchSize,
        int numPoints,
        int resolution,
        int tableSize,
        int featuresPerLevel,
        int levelOffset,
        int outputStride)
    {
        if (dispatchSize <= 0) return;
        ThrowIfDisposed();
        if (first is not MetalGpuBuffer firstBuffer ||
            second is not MetalGpuBuffer secondBuffer ||
            output is not MetalGpuBuffer outputBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, kernelName);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(dispatchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(firstBuffer, 0); encoder.SetBuffer(secondBuffer, 1);
        encoder.SetBuffer(outputBuffer, 2);
        encoder.SetBytes(numPoints, 3); encoder.SetBytes(resolution, 4);
        encoder.SetBytes(tableSize, 5); encoder.SetBytes(featuresPerLevel, 6);
        encoder.SetBytes(levelOffset, 7); encoder.SetBytes(outputStride, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void HashGridEncodeLevel(
        IGpuBuffer positions, IGpuBuffer hashTable, IGpuBuffer output,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride) =>
        DispatchInstantNgp("instant_ngp_hash_encode_level", positions, hashTable, output,
            checked(numPoints * featuresPerLevel), numPoints, resolution, tableSize,
            featuresPerLevel, levelOffset, outputStride);

    public void HashGridEncodeLevelBackward(
        IGpuBuffer positions, IGpuBuffer outputGradient, IGpuBuffer tableGradient,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride) =>
        DispatchInstantNgp("instant_ngp_hash_encode_level_backward", positions, outputGradient,
            tableGradient, checked(tableSize * featuresPerLevel), numPoints, resolution,
            tableSize, featuresPerLevel, levelOffset, outputStride);

    public void UniqueConsecutive(
        IGpuBuffer input, IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        if (input is not MetalGpuBuffer inputBuffer ||
            outputCapacity is not MetalGpuBuffer outputBuffer ||
            outputCount is not MetalGpuBuffer countBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "unique_consecutive_compact");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(inputBuffer, 0); encoder.SetBuffer(outputBuffer, 1);
        encoder.SetBuffer(countBuffer, 2); encoder.SetBytes(length, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Frexp(
        IGpuBuffer input, IGpuBuffer mantissa, IGpuBuffer exponent, int length)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "frexp_decompose");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(length);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)mantissa, 1);
        encoder.SetBuffer((MetalGpuBuffer)exponent, 2);
        encoder.SetBytes(length, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void UniqueConsecutiveWithInfo(
        IGpuBuffer input, IGpuBuffer outputValues, IGpuBuffer outputInverse,
        IGpuBuffer outputCounts, IGpuBuffer outputCount, int length,
        bool returnInverse, bool returnCounts)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "unique_consecutive_with_info");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)outputValues, 1);
        encoder.SetBuffer((MetalGpuBuffer)outputInverse, 2);
        encoder.SetBuffer((MetalGpuBuffer)outputCounts, 3);
        encoder.SetBuffer((MetalGpuBuffer)outputCount, 4);
        encoder.SetBytes(length, 5);
        encoder.SetBytes(returnInverse ? 1 : 0, 6);
        encoder.SetBytes(returnCounts ? 1 : 0, 7);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
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
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "project_gaussians_3d_to_2d");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(numGaussians);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)means3D, 0);
        encoder.SetBuffer((MetalGpuBuffer)covariances3D, 1);
        encoder.SetBuffer((MetalGpuBuffer)means2D, 2);
        encoder.SetBuffer((MetalGpuBuffer)covariances2D, 3);
        encoder.SetBuffer((MetalGpuBuffer)depths, 4); encoder.SetBuffer((MetalGpuBuffer)visible, 5);
        encoder.SetBytes(numGaussians, 6); encoder.SetBytes(covarianceStride, 7);
        encoder.SetBytes(imageWidth, 8); encoder.SetBytes(imageHeight, 9);
        encoder.SetBytes(v00, 10); encoder.SetBytes(v01, 11); encoder.SetBytes(v02, 12); encoder.SetBytes(v03, 13);
        encoder.SetBytes(v10, 14); encoder.SetBytes(v11, 15); encoder.SetBytes(v12, 16); encoder.SetBytes(v13, 17);
        encoder.SetBytes(v20, 18); encoder.SetBytes(v21, 19); encoder.SetBytes(v22, 20); encoder.SetBytes(v23, 21);
        encoder.SetBytes(p00, 22); encoder.SetBytes(p11, 23);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
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
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "sample_rays_with_occupancy");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)rayOrigins, 0); encoder.SetBuffer((MetalGpuBuffer)rayDirections, 1);
        encoder.SetBuffer((MetalGpuBuffer)occupancyBitfield, 2); encoder.SetBuffer((MetalGpuBuffer)positions, 3);
        encoder.SetBuffer((MetalGpuBuffer)directions, 4); encoder.SetBuffer((MetalGpuBuffer)validMask, 5);
        encoder.SetBuffer((MetalGpuBuffer)tValues, 6); encoder.SetBytes(numRays, 7);
        encoder.SetBytes(occupancyWordCount, 8); encoder.SetBytes(gridSize, 9); encoder.SetBytes(maxSamples, 10);
        encoder.SetBytes(minX, 11); encoder.SetBytes(minY, 12); encoder.SetBytes(minZ, 13);
        encoder.SetBytes(maxX, 14); encoder.SetBytes(maxY, 15); encoder.SetBytes(maxZ, 16);
        encoder.SetBytes(nearBound, 17); encoder.SetBytes(farBound, 18);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void UniqueSortedWithInfo(
        IGpuBuffer sortedInput, IGpuBuffer sortedOriginalIndices, IGpuBuffer outputValues,
        IGpuBuffer outputInverse, IGpuBuffer outputCounts, IGpuBuffer outputCount,
        int length, bool returnInverse, bool returnCounts)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "unique_sorted_with_info");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)sortedInput, 0);
        encoder.SetBuffer((MetalGpuBuffer)sortedOriginalIndices, 1);
        encoder.SetBuffer((MetalGpuBuffer)outputValues, 2);
        encoder.SetBuffer((MetalGpuBuffer)outputInverse, 3);
        encoder.SetBuffer((MetalGpuBuffer)outputCounts, 4);
        encoder.SetBuffer((MetalGpuBuffer)outputCount, 5);
        encoder.SetBytes(length, 6);
        encoder.SetBytes(returnInverse ? 1 : 0, 7);
        encoder.SetBytes(returnCounts ? 1 : 0, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Nonzero(IGpuBuffer input, IGpuBuffer strides, IGpuBuffer outputCapacity,
        IGpuBuffer outputCount, int length, int rank)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "nonzero_compact");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0); encoder.SetBuffer((MetalGpuBuffer)strides, 1);
        encoder.SetBuffer((MetalGpuBuffer)outputCapacity, 2); encoder.SetBuffer((MetalGpuBuffer)outputCount, 3);
        encoder.SetBytes(length, 4); encoder.SetBytes(rank, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Mode(IGpuBuffer input, IGpuBuffer output, int length)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_mode");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)input, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(length, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void ConvertIndicesToInt32(
        IGpuBuffer numericIndices, IGpuBuffer int32Indices, int length)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_indices_to_int32");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(length);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)numericIndices, 0);
        encoder.SetBuffer((MetalGpuBuffer)int32Indices, 1);
        encoder.SetBytes(length, 2);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void IndexAdd(
        IGpuBuffer destination, IGpuBuffer indices, IGpuBuffer source, IGpuBuffer output,
        int outerSize, int sourceAxis, int destinationAxis, int innerSize)
    {
        int total = checked(outerSize * destinationAxis * innerSize);
        if (total <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_index_add");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)destination, 0);
        encoder.SetBuffer((MetalGpuBuffer)indices, 1);
        encoder.SetBuffer((MetalGpuBuffer)source, 2);
        encoder.SetBuffer((MetalGpuBuffer)output, 3);
        encoder.SetBytes(outerSize, 4); encoder.SetBytes(sourceAxis, 5);
        encoder.SetBytes(destinationAxis, 6); encoder.SetBytes(innerSize, 7);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void IndexSelect(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int outerSize, int sourceAxis, int indexAxis, int innerSize)
    {
        int total = checked(outerSize * indexAxis * innerSize);
        if (total <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_index_select");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)source, 0);
        encoder.SetBuffer((MetalGpuBuffer)indices, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBytes(outerSize, 3); encoder.SetBytes(sourceAxis, 4);
        encoder.SetBytes(indexAxis, 5); encoder.SetBytes(innerSize, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void ScatterMaxWithArgmaxRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer argmax,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(outputRows * innerSize);
        if (total <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_scatter_max_argmax_rows");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)source, 0);
        encoder.SetBuffer((MetalGpuBuffer)indices, 1);
        encoder.SetBuffer((MetalGpuBuffer)output, 2);
        encoder.SetBuffer((MetalGpuBuffer)argmax, 3);
        encoder.SetBytes(sourceRows, 4); encoder.SetBytes(innerSize, 5); encoder.SetBytes(outputRows, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void UniformMeshLaplacian(
        IGpuBuffer faces, IGpuBuffer output, int numFaces, int numVertices)
    {
        int total = checked(numVertices * numVertices);
        if (total <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_uniform_mesh_laplacian");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)faces, 0);
        encoder.SetBuffer((MetalGpuBuffer)output, 1);
        encoder.SetBytes(numFaces, 2); encoder.SetBytes(numVertices, 3);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void DispatchResidentRows3(
        string name, IGpuBuffer first, IGpuBuffer second, IGpuBuffer third,
        int total, int firstSize, int innerSize, int thirdSize)
    {
        if (total <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, name);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)first, 0);
        encoder.SetBuffer((MetalGpuBuffer)second, 1);
        encoder.SetBuffer((MetalGpuBuffer)third, 2);
        encoder.SetBytes(firstSize, 3); encoder.SetBytes(innerSize, 4);
        encoder.SetBytes(thirdSize, 5);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    private void DispatchResidentRows4(
        string name, IGpuBuffer first, IGpuBuffer second, IGpuBuffer third, IGpuBuffer fourth,
        int total, int firstSize, int innerSize, int thirdSize)
    {
        if (total <= 0) return;
        ThrowIfDisposed();
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, name);
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer((MetalGpuBuffer)first, 0);
        encoder.SetBuffer((MetalGpuBuffer)second, 1);
        encoder.SetBuffer((MetalGpuBuffer)third, 2);
        encoder.SetBuffer((MetalGpuBuffer)fourth, 3);
        encoder.SetBytes(firstSize, 4); encoder.SetBytes(innerSize, 5);
        encoder.SetBytes(thirdSize, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void ScatterAddRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int outputRows) =>
        DispatchResidentRows3("resident_scatter_add_rows", source, indices, output,
            checked(outputRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterMeanRowsWithCounts(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts,
        int sourceRows, int innerSize, int outputRows) =>
        DispatchResidentRows4("resident_scatter_mean_rows_counts", source, indices, output, counts,
            Math.Max(checked(outputRows * innerSize), outputRows), sourceRows, innerSize, outputRows);

    public void ScatterSoftmaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int numGroups) =>
        DispatchResidentRows3("resident_scatter_softmax_rows", source, indices, output,
            checked(sourceRows * innerSize), sourceRows, innerSize, numGroups);

    public void ScatterAddBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows) =>
        DispatchResidentRows3("resident_scatter_add_backward_rows", gradOutput, indices, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterMeanBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer counts, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows) =>
        DispatchResidentRows4("resident_scatter_mean_backward_rows", gradOutput, indices, counts, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterMaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer argmax, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows) =>
        DispatchResidentRows3("resident_scatter_max_backward_rows", gradOutput, argmax, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterSoftmaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int numGroups) =>
        DispatchResidentRows4("resident_scatter_softmax_backward_rows", gradOutput, output, indices, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, numGroups);

    public void CtcLoss(
        IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer inputLengths,
        IGpuBuffer targetLengths, IGpuBuffer workspace, IGpuBuffer losses,
        int maxTime, int batchSize, int numClasses, int maxTargetLength, int blank)
    {
        if (batchSize <= 0) return;
        ThrowIfDisposed();
        if (logProbs is not MetalGpuBuffer logProbBuffer ||
            targets is not MetalGpuBuffer targetBuffer ||
            inputLengths is not MetalGpuBuffer inputLengthBuffer ||
            targetLengths is not MetalGpuBuffer targetLengthBuffer ||
            workspace is not MetalGpuBuffer workspaceBuffer ||
            losses is not MetalGpuBuffer lossBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "ctc_loss_forward");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(batchSize);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(logProbBuffer, 0); encoder.SetBuffer(targetBuffer, 1);
        encoder.SetBuffer(inputLengthBuffer, 2); encoder.SetBuffer(targetLengthBuffer, 3);
        encoder.SetBuffer(workspaceBuffer, 4); encoder.SetBuffer(lossBuffer, 5);
        encoder.SetBytes(maxTime, 6); encoder.SetBytes(batchSize, 7);
        encoder.SetBytes(numClasses, 8); encoder.SetBytes(maxTargetLength, 9);
        encoder.SetBytes(blank, 10);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void ImportanceSampling(
        IGpuBuffer tValuesCoarse, IGpuBuffer weightsCoarse, IGpuBuffer fineTValues,
        int numRays, int numCoarseSamples, int numFineSamples, uint seed)
    {
        int total = checked(numRays * numFineSamples);
        if (total <= 0) return;
        ThrowIfDisposed();
        if (tValuesCoarse is not MetalGpuBuffer tBuffer ||
            weightsCoarse is not MetalGpuBuffer weightBuffer ||
            fineTValues is not MetalGpuBuffer outputBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "importance_sampling");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(total);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(tBuffer, 0); encoder.SetBuffer(weightBuffer, 1);
        encoder.SetBuffer(outputBuffer, 2); encoder.SetBytes(numRays, 3);
        encoder.SetBytes(numCoarseSamples, 4); encoder.SetBytes(numFineSamples, 5);
        encoder.SetBytes(seed, 6);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void Nms(
        IGpuBuffer boxes, IGpuBuffer scores, IGpuBuffer classIds, IGpuBuffer suppressed,
        IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length, float iouThreshold,
        int batched)
    {
        if (length <= 0) return;
        ThrowIfDisposed();
        if (boxes is not MetalGpuBuffer boxBuffer || scores is not MetalGpuBuffer scoreBuffer ||
            classIds is not MetalGpuBuffer classBuffer || suppressed is not MetalGpuBuffer workspaceBuffer ||
            outputCapacity is not MetalGpuBuffer outputBuffer || outputCount is not MetalGpuBuffer countBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "resident_nms");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(boxBuffer, 0); encoder.SetBuffer(scoreBuffer, 1);
        encoder.SetBuffer(classBuffer, 2); encoder.SetBuffer(workspaceBuffer, 3);
        encoder.SetBuffer(outputBuffer, 4); encoder.SetBuffer(countBuffer, 5);
        encoder.SetBytes(length, 6); encoder.SetBytes(iouThreshold, 7); encoder.SetBytes(batched, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }

    public void GenerateSpiralIndices(
        IGpuBuffer vertices, IGpuBuffer faces, IGpuBuffer visited, IGpuBuffer currentRing,
        IGpuBuffer nextRing, IGpuBuffer output, int numVertices, int numFaces, int spiralLength)
    {
        if (numVertices <= 0 || spiralLength <= 0) return;
        ThrowIfDisposed();
        if (vertices is not MetalGpuBuffer vertexBuffer || faces is not MetalGpuBuffer faceBuffer ||
            visited is not MetalGpuBuffer visitedBuffer || currentRing is not MetalGpuBuffer currentBuffer ||
            nextRing is not MetalGpuBuffer nextBuffer || output is not MetalGpuBuffer outputBuffer)
            throw new ArgumentException("Buffers must be MetalGpuBuffer.");
        var pipeline = GetPipeline(AudioLibName, _audioLibrary, "generate_spiral_indices");
        var (threadgroups, threadsPerGroup) = pipeline.Calculate1DDispatch(1);
        using var encoder = _commandQueue.CreateScopedComputeEncoder();
        encoder.SetPipelineState(pipeline.Handle);
        encoder.SetBuffer(vertexBuffer, 0); encoder.SetBuffer(faceBuffer, 1);
        encoder.SetBuffer(visitedBuffer, 2); encoder.SetBuffer(currentBuffer, 3);
        encoder.SetBuffer(nextBuffer, 4); encoder.SetBuffer(outputBuffer, 5);
        encoder.SetBytes(numVertices, 6); encoder.SetBytes(numFaces, 7);
        encoder.SetBytes(spiralLength, 8);
        encoder.DispatchThreadgroups(threadgroups, threadsPerGroup);
    }
}
