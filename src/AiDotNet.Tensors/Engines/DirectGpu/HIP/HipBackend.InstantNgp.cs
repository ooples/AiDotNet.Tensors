namespace AiDotNet.Tensors.Engines.DirectGpu.HIP;

public sealed partial class HipBackend : IInstantNgpBackend, IUniqueConsecutiveBackend, INonzeroBackend, IModeBackend, IResidentIndexBackend, ICtcLossBackend, IImportanceSamplingBackend, INmsBackend, ISpiralIndicesBackend
{
    private IntPtr ResolveInstantNgpKernel(string name)
    {
        if (_audioModule == IntPtr.Zero)
            throw new InvalidOperationException("Instant-NGP HIP kernels were not compiled.");
        if (!_kernelCache.TryGetValue(name, out var kernel))
            throw new InvalidOperationException($"HIP kernel not found: {name}");
        return kernel;
    }

    public unsafe void HashGridEncodeLevel(
        IGpuBuffer positions, IGpuBuffer hashTable, IGpuBuffer output,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(numPoints * featuresPerLevel);
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel("instant_ngp_hash_encode_level");
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr p = positions.Handle, t = hashTable.Handle, o = output.Handle;
        int np = numPoints, r = resolution, ts = tableSize, f = featuresPerLevel;
        int offset = levelOffset, stride = outputStride;
        void** args = stackalloc void*[9];
        args[0] = &p; args[1] = &t; args[2] = &o; args[3] = &np;
        args[4] = &r; args[5] = &ts; args[6] = &f; args[7] = &offset; args[8] = &stride;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void HashGridEncodeLevelBackward(
        IGpuBuffer positions, IGpuBuffer outputGradient, IGpuBuffer tableGradient,
        int numPoints, int resolution, int tableSize, int featuresPerLevel,
        int levelOffset, int outputStride)
    {
        int total = checked(tableSize * featuresPerLevel);
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel("instant_ngp_hash_encode_level_backward");
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr p = positions.Handle, go = outputGradient.Handle, gt = tableGradient.Handle;
        int np = numPoints, r = resolution, ts = tableSize, f = featuresPerLevel;
        int offset = levelOffset, stride = outputStride;
        void** args = stackalloc void*[9];
        args[0] = &p; args[1] = &go; args[2] = &gt; args[3] = &np;
        args[4] = &r; args[5] = &ts; args[6] = &f; args[7] = &offset; args[8] = &stride;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void UniqueConsecutive(
        IGpuBuffer input, IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length)
    {
        if (length <= 0) return;
        var kernel = ResolveInstantNgpKernel("unique_consecutive_compact");
        IntPtr i = input.Handle, o = outputCapacity.Handle, c = outputCount.Handle;
        int n = length;
        void** args = stackalloc void*[4];
        args[0] = &i; args[1] = &o; args[2] = &c; args[3] = &n;
        LaunchKernel(kernel, 1, 1, args);
        Synchronize();
    }

    public unsafe void Nonzero(IGpuBuffer input, IGpuBuffer strides, IGpuBuffer outputCapacity,
        IGpuBuffer outputCount, int length, int rank)
    {
        if (length <= 0) return;
        var kernel = ResolveInstantNgpKernel("nonzero_compact");
        IntPtr i = input.Handle, s = strides.Handle, o = outputCapacity.Handle, c = outputCount.Handle;
        int n = length, r = rank;
        void** args = stackalloc void*[6];
        args[0] = &i; args[1] = &s; args[2] = &o; args[3] = &c; args[4] = &n; args[5] = &r;
        LaunchKernel(kernel, 1, 1, args);
        Synchronize();
    }

    public unsafe void Mode(IGpuBuffer input, IGpuBuffer output, int length)
    {
        if (length <= 0) return;
        var kernel = ResolveInstantNgpKernel("resident_mode");
        IntPtr i = input.Handle, o = output.Handle;
        int n = length;
        void** args = stackalloc void*[3];
        args[0] = &i; args[1] = &o; args[2] = &n;
        LaunchKernel(kernel, 1, 1, args);
        Synchronize();
    }

    public unsafe void ConvertIndicesToInt32(
        IGpuBuffer numericIndices, IGpuBuffer int32Indices, int length)
    {
        if (length <= 0) return;
        var kernel = ResolveInstantNgpKernel("resident_indices_to_int32");
        IntPtr i = numericIndices.Handle, o = int32Indices.Handle;
        int n = length;
        void** args = stackalloc void*[3];
        args[0] = &i; args[1] = &o; args[2] = &n;
        LaunchKernel(kernel, (uint)((length + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void IndexAdd(
        IGpuBuffer destination, IGpuBuffer indices, IGpuBuffer source, IGpuBuffer output,
        int outerSize, int sourceAxis, int destinationAxis, int innerSize)
    {
        int total = checked(outerSize * destinationAxis * innerSize);
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel("resident_index_add");
        IntPtr d = destination.Handle, i = indices.Handle, s = source.Handle, o = output.Handle;
        int outer = outerSize, srcAxis = sourceAxis, dstAxis = destinationAxis, inner = innerSize;
        void** args = stackalloc void*[8];
        args[0] = &d; args[1] = &i; args[2] = &s; args[3] = &o;
        args[4] = &outer; args[5] = &srcAxis; args[6] = &dstAxis; args[7] = &inner;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void IndexSelect(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int outerSize, int sourceAxis, int indexAxis, int innerSize)
    {
        int total = checked(outerSize * indexAxis * innerSize);
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel("resident_index_select");
        IntPtr s = source.Handle, i = indices.Handle, o = output.Handle;
        int outer = outerSize, srcAxis = sourceAxis, idxAxis = indexAxis, inner = innerSize;
        void** args = stackalloc void*[7];
        args[0] = &s; args[1] = &i; args[2] = &o; args[3] = &outer;
        args[4] = &srcAxis; args[5] = &idxAxis; args[6] = &inner;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void ScatterMaxWithArgmaxRows(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer argmax,
        int sourceRows, int innerSize, int outputRows)
    {
        int total = checked(outputRows * innerSize);
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel("resident_scatter_max_argmax_rows");
        IntPtr s = source.Handle, i = indices.Handle, o = output.Handle, a = argmax.Handle;
        int srcRows = sourceRows, inner = innerSize, outRows = outputRows;
        void** args = stackalloc void*[7];
        args[0] = &s; args[1] = &i; args[2] = &o; args[3] = &a;
        args[4] = &srcRows; args[5] = &inner; args[6] = &outRows;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void UniformMeshLaplacian(
        IGpuBuffer faces, IGpuBuffer output, int numFaces, int numVertices)
    {
        int total = checked(numVertices * numVertices);
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel("resident_uniform_mesh_laplacian");
        IntPtr f = faces.Handle, o = output.Handle;
        int faceCount = numFaces, vertexCount = numVertices;
        void** args = stackalloc void*[4];
        args[0] = &f; args[1] = &o; args[2] = &faceCount; args[3] = &vertexCount;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    private unsafe void LaunchResidentRows3(
        string name, IGpuBuffer first, IGpuBuffer second, IGpuBuffer third,
        int total, int firstSize, int innerSize, int thirdSize)
    {
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel(name);
        IntPtr a = first.Handle, b = second.Handle, c = third.Handle;
        void** args = stackalloc void*[6];
        args[0] = &a; args[1] = &b; args[2] = &c;
        args[3] = &firstSize; args[4] = &innerSize; args[5] = &thirdSize;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    private unsafe void LaunchResidentRows4(
        string name, IGpuBuffer first, IGpuBuffer second, IGpuBuffer third, IGpuBuffer fourth,
        int total, int firstSize, int innerSize, int thirdSize)
    {
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel(name);
        IntPtr a = first.Handle, b = second.Handle, c = third.Handle, d = fourth.Handle;
        void** args = stackalloc void*[7];
        args[0] = &a; args[1] = &b; args[2] = &c; args[3] = &d;
        args[4] = &firstSize; args[5] = &innerSize; args[6] = &thirdSize;
        LaunchKernel(kernel, (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize),
            DefaultBlockSize, args);
        Synchronize();
    }

    public void ScatterAddRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int outputRows) =>
        LaunchResidentRows3("resident_scatter_add_rows", source, indices, output,
            checked(outputRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterMeanRowsWithCounts(
        IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output, IGpuBuffer counts,
        int sourceRows, int innerSize, int outputRows) =>
        LaunchResidentRows4("resident_scatter_mean_rows_counts", source, indices, output, counts,
            Math.Max(checked(outputRows * innerSize), outputRows), sourceRows, innerSize, outputRows);

    public void ScatterSoftmaxRows(IGpuBuffer source, IGpuBuffer indices, IGpuBuffer output,
        int sourceRows, int innerSize, int numGroups) =>
        LaunchResidentRows3("resident_scatter_softmax_rows", source, indices, output,
            checked(sourceRows * innerSize), sourceRows, innerSize, numGroups);

    public void ScatterAddBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows) =>
        LaunchResidentRows3("resident_scatter_add_backward_rows", gradOutput, indices, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterMeanBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer indices, IGpuBuffer counts, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows) =>
        LaunchResidentRows4("resident_scatter_mean_backward_rows", gradOutput, indices, counts, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterMaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer argmax, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int outputRows) =>
        LaunchResidentRows3("resident_scatter_max_backward_rows", gradOutput, argmax, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, outputRows);

    public void ScatterSoftmaxBackwardRows(
        IGpuBuffer gradOutput, IGpuBuffer output, IGpuBuffer indices, IGpuBuffer gradSource,
        int sourceRows, int innerSize, int numGroups) =>
        LaunchResidentRows4("resident_scatter_softmax_backward_rows", gradOutput, output, indices, gradSource,
            checked(sourceRows * innerSize), sourceRows, innerSize, numGroups);

    public unsafe void CtcLoss(
        IGpuBuffer logProbs, IGpuBuffer targets, IGpuBuffer inputLengths,
        IGpuBuffer targetLengths, IGpuBuffer workspace, IGpuBuffer losses,
        int maxTime, int batchSize, int numClasses, int maxTargetLength, int blank)
    {
        if (batchSize <= 0) return;
        var kernel = ResolveInstantNgpKernel("ctc_loss_forward");
        uint grid = (uint)((batchSize + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr lp = logProbs.Handle, t = targets.Handle, il = inputLengths.Handle;
        IntPtr tl = targetLengths.Handle, w = workspace.Handle, o = losses.Handle;
        int mt = maxTime, bs = batchSize, nc = numClasses, mu = maxTargetLength, b = blank;
        void** args = stackalloc void*[11];
        args[0] = &lp; args[1] = &t; args[2] = &il; args[3] = &tl;
        args[4] = &w; args[5] = &o; args[6] = &mt; args[7] = &bs;
        args[8] = &nc; args[9] = &mu; args[10] = &b;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void ImportanceSampling(
        IGpuBuffer tValuesCoarse, IGpuBuffer weightsCoarse, IGpuBuffer fineTValues,
        int numRays, int numCoarseSamples, int numFineSamples, uint seed)
    {
        int total = checked(numRays * numFineSamples);
        if (total <= 0) return;
        var kernel = ResolveInstantNgpKernel("importance_sampling");
        uint grid = (uint)((total + DefaultBlockSize - 1) / DefaultBlockSize);
        IntPtr t = tValuesCoarse.Handle, w = weightsCoarse.Handle, o = fineTValues.Handle;
        int nr = numRays, nc = numCoarseSamples, nf = numFineSamples;
        uint s = seed;
        void** args = stackalloc void*[7];
        args[0] = &t; args[1] = &w; args[2] = &o; args[3] = &nr;
        args[4] = &nc; args[5] = &nf; args[6] = &s;
        LaunchKernel(kernel, grid, DefaultBlockSize, args);
        Synchronize();
    }

    public unsafe void Nms(
        IGpuBuffer boxes, IGpuBuffer scores, IGpuBuffer classIds, IGpuBuffer suppressed,
        IGpuBuffer outputCapacity, IGpuBuffer outputCount, int length, float iouThreshold,
        int batched)
    {
        if (length <= 0) return;
        var kernel = ResolveInstantNgpKernel("resident_nms");
        IntPtr b = boxes.Handle, s = scores.Handle, c = classIds.Handle;
        IntPtr w = suppressed.Handle, o = outputCapacity.Handle, count = outputCount.Handle;
        int n = length, batchMode = batched;
        float threshold = iouThreshold;
        void** args = stackalloc void*[9];
        args[0] = &b; args[1] = &s; args[2] = &c; args[3] = &w;
        args[4] = &o; args[5] = &count; args[6] = &n; args[7] = &threshold;
        args[8] = &batchMode;
        LaunchKernel(kernel, 1, 1, args);
        Synchronize();
    }

    public unsafe void GenerateSpiralIndices(
        IGpuBuffer vertices, IGpuBuffer faces, IGpuBuffer visited, IGpuBuffer currentRing,
        IGpuBuffer nextRing, IGpuBuffer output, int numVertices, int numFaces, int spiralLength)
    {
        if (numVertices <= 0 || spiralLength <= 0) return;
        var kernel = ResolveInstantNgpKernel("generate_spiral_indices");
        IntPtr v = vertices.Handle, f = faces.Handle, seen = visited.Handle;
        IntPtr current = currentRing.Handle, next = nextRing.Handle, o = output.Handle;
        int nv = numVertices, nf = numFaces, sl = spiralLength;
        void** args = stackalloc void*[9];
        args[0] = &v; args[1] = &f; args[2] = &seen; args[3] = &current;
        args[4] = &next; args[5] = &o; args[6] = &nv; args[7] = &nf; args[8] = &sl;
        LaunchKernel(kernel, 1, 1, args);
        Synchronize();
    }
}
